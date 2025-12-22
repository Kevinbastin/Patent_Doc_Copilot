import re
from typing import Dict, List
from llm_runtime import llm_generate


# ============================================================
# DOMAIN & FEATURE EXTRACTION
# ============================================================
def extract_technical_domain(abstract: str) -> str:
    """Extract the primary technical domain from the abstract."""
    common_domains = {
        "artificial intelligence": ["ai", "machine learning", "neural network", "deep learning", "ml", "generative model"],
        "information retrieval": ["search", "query", "retrieval", "summarization", "information processing", "search results"],
        "natural language processing": ["natural language", "nlp", "text processing", "language model"],
        "computer science": ["computer", "software", "algorithm", "computation"],
        "telecommunications": ["wireless", "network", "communication", "5g", "antenna"],
        "renewable energy": [
            "wind power", "wind energy", "renewable energy", "turbine",
            "power generation", "generator", "energy conversion"
        ],
        "power engineering": [
            "power system", "power management", "electrical power",
            "energy storage", "inverter", "battery"
        ],
        "electrical engineering": ["circuit", "electrical", "electronic", "power"],
        "mechanical engineering": ["mechanical", "engine", "mechanism", "manufacturing"],
        "biotechnology": ["biotech", "genetic", "protein", "dna", "biological"],
        "pharmaceutical": ["drug", "pharmaceutical", "medicine", "therapeutic"],
        "medical devices": ["medical device", "diagnostic", "imaging", "surgical"],
        "semiconductor": ["semiconductor", "chip", "integrated circuit", "transistor"],
        "chemical engineering": ["chemical", "catalyst", "reaction", "synthesis"],
        "materials science": ["material", "composite", "polymer", "alloy"],
        "automotive": ["automotive", "vehicle", "automobile", "engine"],
        "aerospace": ["aerospace", "aircraft", "aviation", "flight"]
    }
    
    abstract_lower = abstract.lower()
    
    # Check for domain matches (prioritize more specific domains first)
    for domain, keywords in common_domains.items():
        if any(kw in abstract_lower for kw in keywords):
            return domain
    
    return "technology"


# ============================================================
# CLEANING & VALIDATION
# ============================================================
def clean_field_text(text: str) -> str:
    """Clean and format the generated field of invention text."""
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
    prompt = f"""You are an expert patent attorney drafting an Indian patent application.

TASK: Write ONLY the "Field of the Invention" section based on the abstract below.

IMPORTANT:
- Output MUST contain ONLY English ASCII characters.
- Do NOT generate Chinese, Japanese, Korean, or any non-English language.
- If unsure, STOP the sentence instead of switching language.

REQUIREMENTS:
1. Use formal, technical language (third person, present tense)
2. Start with standard phrases:
   - "The present invention relates generally to..."
   - "This invention pertains to..."
   - "The present disclosure relates to..."
3. Be concise: 2-3 sentences, 40-80 words total
4. State the technical field broadly, then narrow to specific area
5. Include the specific application or technology mentioned
6. Do NOT include technical details, advantages, or how it works
7. Do NOT use marketing language (novel, innovative, revolutionary, etc.)
8. Do NOT repeat the abstract verbatim

GOOD EXAMPLES:

Example 1:
"The present invention relates generally to wireless communication systems, and more particularly to methods and apparatus for improving signal transmission in 5G networks."

Example 2:
"This invention pertains to the field of medical imaging, specifically to enhanced MRI scanning techniques for early disease detection."

Example 3:
"The present disclosure relates to artificial intelligence systems, and more specifically to machine learning models for natural language processing and text generation."

Example 4:
"The present invention relates generally to information retrieval systems, particularly to methods and systems for generating natural language summaries of search results using generative language models with factual verification."

INVENTION ABSTRACT:
{abstract.strip()}

Now write ONLY the Field of the Invention text (no heading, no extra explanation, just the 2-3 sentences):

The present invention"""

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
            
            # Prepend "The present invention" since we prompted with it
            full_text = raw_text.strip()
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
