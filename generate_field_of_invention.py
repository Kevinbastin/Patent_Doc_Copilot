import re
from typing import Dict, List
from llm_runtime import llm_generate


# ============================================================
# DOMAIN & FEATURE EXTRACTION
# ============================================================
def extract_technical_domain(abstract: str) -> str:
    """Extract the primary technical domain from the abstract using LLM."""
    prompt = f"""Identify the primary technical domain for this patent abstract (e.g., "telecommunications", "biotechnology", "artificial intelligence").
Return ONLY the domain name in 1-3 words.

ABSTRACT:
{abstract[:500]}

DOMAIN:"""
    
    try:
        domain = llm_generate(
            prompt,
            max_new_tokens=10,
            temperature=0.1,
            system_prompt="You are a patent domain classifier. Output ONLY the domain name."
        ).strip().lower().rstrip('.')
        return domain or "technology"
    except Exception as e:
        print(f"LLM domain extraction failed: {e}")
        return "technology"


# ============================================================
# CLEANING & VALIDATION
# ============================================================
def clean_field_text(text: str) -> str:
    """Clean and format the generated field of invention text."""
    
    # Strip common model 'thinking' patterns
    thinking_patterns = [
        r"^Okay,?\s+(?:the user|I need|let me|I'll).*?(?:\.\s*|\n)",
        r"^Let me (?:start|analyze|think|draft|understand).*?(?:\.\s*|\n)",
        r"^I need to.*?(?:\.\s*|\n)",
        r"^First,? I(?:'ll| will| should).*?(?:\.\s*|\n)",
        r"^(?:Understood|Got it|Alright).*?(?:\.\s*|\n)",
        r"^The (?:user|abstract) (?:wants|provides|mentions).*?(?:\.\s*|\n)",
        r'Field of the Invention"\s*section.*?(?:\.\s*|\n)',
        r"^.*?requirements.*?(?:\.\s*|\n)",
    ]
    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Also strip mid-sentence chatter like ", okay, let's tackle this query. "
    mid_sentence_chatter = [
        r",?\s*okay,?\s*let'?s\s+(?:tackle|address|handle|work on)[^.]*\.\s*(?:the\s+)?",
        r",?\s*let me\s+(?:think|consider|analyze)[^.]*\.\s*",
        r",?\s*I'?ll\s+(?:draft|write|generate)[^.]*\.\s*",
        r"\s+okay,?\s*let'?s\s+(?:tackle|see)[^.]*\.\s*(?:the\s+)?",
        r"\s+okay,?\s+",
        r"\bokay,?\s*let'?s\s+(?:tackle|see|address)[^.]*\.\s*(?:the\s+user\s*,?\s*)?",
    ]
    for pattern in mid_sentence_chatter:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Clean up "the user," artifacts
    text = re.sub(r'\bthe\s+user\s*,?\s*', '', text, flags=re.IGNORECASE)
    
    # Ensure it starts with proper phrasing
    if not re.match(r'^(?:The present|This invention|The present disclosure)', text, re.IGNORECASE):
        # Try to find proper start
        match = re.search(r'(?:The present invention|This invention|The present disclosure)', text, re.IGNORECASE)
        if match:
            text = text[match.start():]
    
    # Remove common prefixes
    text = re.sub(
        r'(The present invention\s*){2,}',
        'The present invention ',
        text,
        flags=re.IGNORECASE
    )
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    text = text.strip()
    
    # Capitalize first letter
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    # Ensure it ends with a period
    if text and not text.endswith('.'):
        text += '.'
    
    return text


def validate_field_text(text: str) -> Dict[str, any]:
    """Validate the field of invention text against patent standards."""
    issues = []
    warnings = []
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?]+', text))
    
    # Check length (typically 2-3 sentences, 30-100 words)
    # Check length (2–3 sentences, 40–80 words preferred)
    if word_count < 40:
        issues.append("Too brief. Should be 40–80 words.")
    elif word_count > 100:
        issues.append("Too lengthy. Should be 40–80 words.")
    
    # Check for proper sentence structure
    if sentence_count < 1:
        issues.append("At least one complete sentence is required.")
    elif sentence_count > 4:
        warnings.append("Consider consolidating into 2–3 sentences.")
    
    # Check for required phrases
    standard_phrases = [
        "relates to", "pertains to", "concerns", "directed to",
        "field of", "area of", "relates generally to"
    ]
    has_standard_phrase = any(phrase in text.lower() for phrase in standard_phrases)
    
    if not has_standard_phrase:
        issues.append("Should use standard patent language (e.g., 'relates to', 'pertains to').")
    
    # Check for "present invention" or similar
    if "present invention" not in text.lower() and "this invention" not in text.lower() and "present disclosure" not in text.lower():
        warnings.append("Consider starting with 'The present invention' or 'This invention'.")
    
    # Check for marketing language (should be avoided)
    marketing_words = ["revolutionary", "groundbreaking", "innovative", "novel", "unique", "best", "advanced"]
    found_marketing = [word for word in marketing_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)]
    if found_marketing:
        issues.append(f"Avoid marketing language: {', '.join(found_marketing)}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "sentence_count": sentence_count
    }


# ============================================================
# FIELD OF INVENTION GENERATION
# ============================================================
def generate_field_of_invention(abstract: str, max_attempts: int = 3) -> Dict[str, any]:
    """
    Generates the 'Field of the Invention' section from an abstract.
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated field text and metadata
    """
    
    technical_domain = extract_technical_domain(abstract)
    
    # Enhanced prompt with patent-specific instructions and examples
    prompt = f"""You are an expert Indian Patent Attorney drafting a patent application.

TASK: Write the "Field of the Invention" section (40-80 words, 3 sentences).

MANDATORY STRUCTURE (follow exactly):

SENTENCE 1: "The present invention relates generally to [BROAD TECHNICAL FIELD]."
SENTENCE 2: "More particularly, the present invention relates to [SPECIFIC TECHNICAL AREA] for [APPLICATION/PURPOSE]."
SENTENCE 3: "The invention specifically pertains to [KEY TECHNICAL FEATURES mentioned in abstract]."

REQUIREMENTS:
- EXACTLY 3 sentences, 40-80 words total (CRITICAL - count your words)
- Start with "The present invention relates generally to..."
- Second sentence uses "More particularly"
- Third sentence uses "specifically pertains to"
- NO technical details, advantages, or how it works
- NO marketing language (novel, innovative, advanced, etc.)
- Use formal, technical language

GOOD EXAMPLE (65 words, 3 sentences):
"The present invention relates generally to the field of environmental monitoring systems. More particularly, the present invention relates to water quality assessment systems incorporating multiple sensor arrays for real-time parameter measurement. The invention specifically pertains to monitoring systems utilizing machine learning algorithms for data analysis and cloud-based connectivity for remote alerts and data visualization."

ABSTRACT TO ANALYZE:
{abstract.strip()}

Write the Field of Invention (3 sentences, 40-80 words):

The present invention relates generally to"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...", end=" ", flush=True)
            
            # CORRECTED: Removed max_input_tokens parameter
            raw_text = llm_generate(
                prompt,
                max_new_tokens=220,
                temperature=0.25 if attempt == 0 else 0.30 + (attempt * 0.08),
                top_p=0.88,
                repeat_penalty=1.18,
                stop_strings=[
                    "\n\n",
                    "\nBackground",
                    "\nBACKGROUND",
                    "Summary",
                    "SUMMARY",
                    "请",
                    "提供",
                    "剩余",
                    "部分",
                    "永续",
                    "递延"
                ]

            )
            
            if not raw_text or len(raw_text.strip()) < 20:
                print("Empty response")
                continue
            
            # Prepend "The present invention" if not already starting with it
            full_text = raw_text.strip()
            full_text_lower = full_text.lower()
            
            # Add proper start if missing
            if not full_text_lower.startswith("the present invention") and \
               not full_text_lower.startswith("this invention") and \
               not full_text_lower.startswith("the present disclosure"):
                # Check if it starts with "relates" - prepend properly
                if full_text_lower.startswith("relates"):
                    full_text = "The present invention " + full_text
                else:
                    full_text = "The present invention relates to " + full_text
            
            cleaned_text = clean_field_text(full_text)
            validation = validate_field_text(cleaned_text)
            
            # Calculate quality score
            score = len(validation["issues"]) * 10 + len(validation["warnings"]) * 2
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "word_count": validation["word_count"],
                "sentence_count": validation["sentence_count"],
                "attempt": attempt + 1,
                "technical_domain": technical_domain,
                "score": score
            }
            
            print(f"Score: {score}, Words: {validation['word_count']}")
            
            # Return immediately if perfect
            if validation["valid"] and len(validation["warnings"]) == 0:
                print("   ✅ Perfect!")
                return result
            
            # Keep track of best attempt
            if score < best_score:
                best_score = score
                best_result = result
        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Return best result if decent
    if best_result and best_result["score"] < 30:
        return best_result
    
    # Improved fallback with more specific text
    if "search" in abstract.lower() and "summar" in abstract.lower():
        fallback_text = "The present invention relates generally to information retrieval and natural language processing, particularly to systems and methods for generating natural language summaries of search results using generative models."
    elif "machine learning" in abstract.lower() or "ai" in abstract.lower():
        fallback_text = f"The present invention relates generally to {technical_domain}, and more particularly to systems and methods for data processing and analysis using machine learning techniques."
    else:
        fallback_text = f"The present invention relates generally to the field of {technical_domain}, and more particularly to systems and methods for implementing related technical solutions."
    
    return {
        "text": fallback_text,
        "valid": False,
        "issues": ["Generation failed after all attempts - using fallback"],
        "warnings": ["Manually review and edit to match your specific invention"],
        "word_count": len(fallback_text.split()),
        "sentence_count": len(re.findall(r'[.!?]+', fallback_text)),
        "attempt": max_attempts,
        "technical_domain": technical_domain,
        "score": 100
    }


# ============================================================
# ALTERNATIVE VERSIONS
# ============================================================
def generate_variations(abstract: str) -> List[Dict[str, str]]:
    """Generate multiple variations of the field of invention."""
    variations = []
    
    prompts = [
        {
            "label": "Broad-to-narrow focus",
            "prompt": f"""Write Field of Invention that starts broad then narrows down.

Abstract: {abstract[:600]}

Format: Start with "The present invention relates generally to [broad field], and more particularly to [specific application and technology]."

Write only the field text (2-3 sentences, 40-80 words):

The present invention"""
        },
        {
            "label": "Application-focused",
            "prompt": f"""Write Field of Invention focusing on the specific application and purpose.

Abstract: {abstract[:600]}

Format: Start with "This invention pertains to [application domain], specifically to [technical implementation and purpose]."

Write only the field text (2-3 sentences, 40-80 words):

This invention"""
        },
        {
            "label": "Technology-focused",
            "prompt": f"""Write Field of Invention emphasizing the underlying technology and methods.

Abstract: {abstract[:600]}

Format: Start with "The present disclosure relates to [technology area], particularly to [specific methods/systems and their application]."

Write only the field text (2-3 sentences, 40-80 words):

The present disclosure"""
        }
    ]
    
    for config in prompts:
        try:
            # CORRECTED: Removed max_input_tokens
            raw_text = llm_generate(
                config["prompt"],
                max_new_tokens=180,
                temperature=0.32,
                top_p=0.9,
                repeat_penalty=1.18,
                stop_strings=["\n\nBackground", "\n\n\n", "\n\n"]
            )
            
            if not raw_text or len(raw_text.strip()) < 20:
                continue
            
            # Prepend the starter phrase based on which prompt
            if "present invention" in config["prompt"]:
                full_text = "The present invention" + raw_text.strip()
            elif "This invention" in config["prompt"]:
                full_text = "This invention" + raw_text.strip()
            else:
                full_text = "The present disclosure" + raw_text.strip()
            
            cleaned = clean_field_text(full_text)
            validation = validate_field_text(cleaned)
            
            variations.append({
                "label": config["label"],
                "text": cleaned,
                "valid": validation["valid"],
                "word_count": validation["word_count"],
                "sentence_count": validation["sentence_count"],
                "issues": validation.get("issues", []),
                "warnings": validation.get("warnings", [])
            })
            
        except Exception as e:
            print(f"   Error generating {config['label']}: {e}")
            continue
    
    return variations


# ============================================================
# FORMATTING
# ============================================================
def format_for_patent(field_text: str, paragraph_number: str = "[0001]") -> str:
    """Format the field text with standard patent paragraph numbering."""
    return f"{paragraph_number} {field_text}"


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("         PATENT FIELD OF INVENTION GENERATOR")
    print("=" * 80)
    print("\n📥 Enter the invention abstract (press Enter twice to finish):")
    print("-" * 80)
    
    lines = []
    while True:
        line = input()
        if line.strip() == "" and lines:
            break
        if line.strip() != "":
            lines.append(line)
    
    abstract = " ".join(lines).strip()
    
    if not abstract:
        print("\n❌ No abstract provided. Exiting.")
        exit(1)
    
    print("\n🛠️  Generating 'Field of the Invention'...")
    
    result = generate_field_of_invention(abstract, max_attempts=3)
    
    print("\n" + "=" * 80)
    print("📘 FIELD OF THE INVENTION - VALIDATION REPORT")
    print("=" * 80)
    
    # Display validation status
    if result["valid"] and len(result.get("warnings", [])) == 0:
        print("\n✅ Status: EXCELLENT - Ready for patent application")
    elif result["valid"]:
        print("\n✅ Status: VALID - Minor improvements recommended")
    else:
        print("\n⚠️  Status: NEEDS REVIEW - Issues found")
    
    print("\n" + "-" * 80)
    print("📊 STATISTICS:")
    print(f"   • Word Count:        {result['word_count']} (optimal: 40-80)")
    print(f"   • Sentences:         {result['sentence_count']} (optimal: 2-3)")
    print(f"   • Technical Domain:  {result['technical_domain']}")
    print(f"   • Attempts Used:     {result['attempt']}/{3}")
    print(f"   • Quality Score:     {result['score']} (lower is better)")
    
    if result["issues"]:
        print("\n" + "-" * 80)
        print("🚨 CRITICAL ISSUES:")
        for i, issue in enumerate(result["issues"], 1):
            print(f"   {i}. {issue}")
    
    if result.get("warnings"):
        print("\n" + "-" * 80)
        print("⚠️  WARNINGS:")
        for i, warning in enumerate(result["warnings"], 1):
            print(f"   {i}. {warning}")
    
    print("\n" + "=" * 80)
    print("📝 GENERATED TEXT:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)
    
    print("\n" + "=" * 80)
    print("📄 FORMATTED FOR PATENT (with paragraph numbering):")
    print("-" * 80)
    print(format_for_patent(result["text"]))
    print("-" * 80)
    
    # Optional: Generate variations
    print("\n🔄 Generate alternative versions? (y/n): ", end="")
    if input().lower() == 'y':
        print("\n⏳ Generating variations...")
        variations = generate_variations(abstract)
        
        if variations:
            print("\n" + "=" * 80)
            print("📝 ALTERNATIVE VERSIONS:")
            print("=" * 80)
            
            for var in variations:
                print(f"\n--- {var['label']} ---")
                status = "✅" if var['valid'] else "⚠️"
                print(f"{status} | Words: {var['word_count']} | Sentences: {var['sentence_count']}")
                if var.get('issues'):
                    print(f"Issues: {'; '.join(var['issues'][:2])}")
                print("-" * 80)
                print(var['text'])
                print("-" * 80)
        else:
            print("\n⚠️  Could not generate variations.")
    
    print("\n" + "=" * 80)
    print("💡 TIPS FOR PERFECT FIELD OF INVENTION:")
    print("=" * 80)
    print("1. Start with 'The present invention' or 'This invention'")
    print("2. Use 'relates to', 'pertains to', or 'directed to'")
    print("3. Begin broad, then narrow to specific technical area")
    print("4. Keep it 40-80 words, 2-3 sentences")
    print("5. Avoid marketing language - stay technical and formal")
    print("6. State WHAT it is, not HOW it works or WHY it's better")
    print("=" * 80)
    print("\n✅ Generation complete!")
    print("=" * 80)
