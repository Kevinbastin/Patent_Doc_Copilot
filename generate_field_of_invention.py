from llama_cpp import Llama
import re
from typing import Dict, List


# Path to your local GGUF model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model once with optimized settings
llm = Llama(
    model_path=LLM_PATH, device="cuda",
    n_ctx=4096,  # Increased for better context
    n_threads=4,
    n_batch=512
)


def extract_technical_components(abstract: str) -> Dict[str, any]:
    """
    Extract key technical components and domain from abstract.
    """
    components = {
        'broad_field': '',
        'specific_field': '',
        'technologies': [],
        'application': '',
        'key_features': []
    }
    
    # Detect broad technical domains
    broad_domains = {
        'IoT': ['internet of things', 'iot', 'sensor network', 'wireless sensor'],
        'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai'],
        'Telecommunications': ['communication system', 'wireless', 'network', 'transmission'],
        'Agricultural': ['agricultural', 'farming', 'crop', 'irrigation', 'soil'],
        'Medical': ['medical', 'healthcare', 'diagnosis', 'patient', 'clinical'],
        'Wildlife': ['wildlife', 'animal', 'conservation', 'conflict mitigation'],
        'Industrial': ['industrial', 'manufacturing', 'monitoring', 'automation']
    }
    
    abstract_lower = abstract.lower()
    
    for domain, keywords in broad_domains.items():
        if any(kw in abstract_lower for kw in keywords):
            components['broad_field'] = domain
            break
    
    # Extract specific technologies
    tech_patterns = [
        r'(LoRaWAN|LoRa|GSM|WiFi|Bluetooth|Zigbee|NFC)',
        r'(TinyML|YOLO|CNN|LSTM|transformer)',
        r'(sensor|detector|camera|microcontroller|processor)',
        r'(cloud server|edge computing|fog computing)',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        components['technologies'].extend(matches)
    
    # Extract application area
    app_patterns = [
        r'for\s+([^,\.]{10,50}?)(?:,|\.|comprising)',
        r'system\s+for\s+([^,\.]{10,50}?)(?:,|\.|comprising)',
    ]
    
    for pattern in app_patterns:
        match = re.search(pattern, abstract, re.IGNORECASE)
        if match:
            components['application'] = match.group(1).strip()
            break
    
    return components


def clean_field_text(text: str) -> str:
    """Clean and format the generated field of invention text."""
    # Remove common prefixes
    text = re.sub(r'^(Field of the Invention:|Field of Invention:|FIELD OF THE INVENTION:)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Ensure proper sentence structure
    text = text.strip()
    
    # Capitalize first letter if needed
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    # Ensure it ends with a period
    if text and not text.endswith('.'):
        text += '.'
    
    return text


def validate_field_text(text: str) -> Dict[str, any]:
    """
    Validate the field of invention text against Indian Patent Office standards.
    Based on real patent: should be 2-4 sentences, 40-120 words, hierarchical structure.
    """
    issues = []
    warnings = []
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?]+', text))
    
    # Check length (based on real patent example: 77 words, 3 sentences)
    if word_count < 30:
        issues.append("Too brief. Field should be 40-120 words based on patent standards.")
    elif word_count > 150:
        issues.append("Too lengthy. Keep it concise (40-120 words).")
    
    # Check sentence count (real patent has 3 sentences: general -> particular -> more particular)
    if sentence_count < 2:
        issues.append("Need at least 2 sentences (general field + specific focus).")
    elif sentence_count > 5:
        warnings.append("Consider consolidating into 2-4 sentences for clarity.")
    
    # Check for required phrases (from real patent)
    standard_phrases = [
        "present invention",
        "relates to",
        "pertains to", 
        "particularly to",
        "more particularly",
        "specifically to"
    ]
    
    has_present_invention = "present invention" in text.lower()
    has_relates = any(phrase in text.lower() for phrase in ["relates to", "pertains to"])
    has_hierarchy = any(phrase in text.lower() for phrase in ["particularly", "more particularly", "specifically"])
    
    if not has_present_invention:
        issues.append("Must include 'The present invention' or 'present invention'.")
    
    if not has_relates:
        issues.append("Must use 'relates to' or 'pertains to' (standard patent language).")
    
    if not has_hierarchy:
        warnings.append("Consider hierarchical structure: 'generally relates to... particularly to... more particularly...'")
    
    # Check for marketing language (should be avoided)
    marketing_words = ["revolutionary", "groundbreaking", "innovative", "novel", "unique", "best", "advanced"]
    found_marketing = [word for word in marketing_words if re.search(r'\b' + word + r'\b', text, re.IGNORECASE)]
    if found_marketing:
        issues.append(f"Avoid marketing language: {', '.join(found_marketing)}")
    
    # Check for technical specificity
    if not any(char.isdigit() or word in text.lower() for word in ['system', 'method', 'apparatus', 'device', 'module']):
        warnings.append("Include technical category (system, method, apparatus, etc.)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "sentence_count": sentence_count
    }


def generate_field_of_invention(abstract: str, max_attempts: int = 3) -> Dict[str, any]:
    """
    Generates the 'Field of the Invention' section matching Indian Patent Office format.
    
    Format (from real patent IN202541069047):
    Sentence 1: "The present invention generally relates to [broad field]..."
    Sentence 2: "...particularly to [more specific area]"
    Sentence 3: "More particularly, the present invention relates to [very specific implementation]"
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated field text and metadata
    """
    
    components = extract_technical_components(abstract)
    
    # Build enhanced prompt based on real patent structure
    prompt = f"""You are a patent attorney drafting the "Field of the Invention" section for an Indian patent application.

INVENTION ABSTRACT:
{abstract}

REAL PATENT EXAMPLE (Study this structure):
"The present invention generally relates to the field of wildlife conservation and conflict mitigation, particularly to Internet of Things (IoT) based remote monitoring and alerting system. More particularly, the present invention relates to an Internet of Things (IoT) based remote monitoring and multi-modal alerting system using an integrated dual-communication system for human-animal conflict mitigation."

REQUIREMENTS:
1. Write EXACTLY 2-3 sentences following this hierarchical structure:
   - Sentence 1: "The present invention generally relates to [broad field]..."
   - Sentence 2: "...particularly to [specific technology/system]"
   - Sentence 3: "More particularly, the present invention relates to [complete specific implementation]"

2. Use formal patent language:
   - Start with "The present invention"
   - Use "generally relates to", "particularly to", "more particularly"
   - Third person, present tense only
   - Technical but clear language

3. Target: 40-100 words total

4. DO NOT include:
   - Marketing terms (novel, innovative, revolutionary, etc.)
   - Technical details from claims
   - Advantages or benefits
   - How it works

5. DO include:
   - Broad technical field
   - Specific domain/application
   - Key technologies (IoT, AI, ML, sensors, etc.)
   - Precise system/method description

NOW WRITE THE FIELD OF THE INVENTION (only the text, no heading):

The present invention"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=300,
                temperature=0.25 if attempt == 0 else 0.35 + (attempt * 0.1),
                stop=["\n\nBACKGROUND", "BACKGROUND OF", "\n\n\n", "Summary:", "Claims:"],
                top_p=0.88,
                repeat_penalty=1.18
            )
            
            raw_text = "The present invention" + response["choices"][0]["text"].strip()
            cleaned_text = clean_field_text(raw_text)
            validation = validate_field_text(cleaned_text)
            
            # Calculate quality score (lower is better)
            score = len(validation["issues"]) * 10 + len(validation["warnings"]) * 2
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "word_count": validation["word_count"],
                "sentence_count": validation["sentence_count"],
                "attempt": attempt + 1,
                "components": components,
                "score": score
            }
            
            if validation["valid"] and len(validation["warnings"]) == 0:
                return result
            
            # Track best attempt
            if score < best_score:
                best_score = score
                best_result = result
                
        except Exception as e:
            continue
    
    return best_result if best_result else {
        "text": f"The present invention generally relates to {components.get('broad_field', 'technology')}, particularly to {components.get('application', 'systems and methods')}.",
        "valid": False,
        "issues": ["Generation failed, using fallback"],
        "warnings": [],
        "word_count": 0,
        "sentence_count": 0,
        "attempt": max_attempts,
        "components": components,
        "score": 999
    }


def format_for_patent_document(field_text: str, include_heading: bool = True) -> str:
    """
    Format the field text with Indian Patent Office standard heading and formatting.
    """
    output = ""
    
    if include_heading:
        output += "FIELD OF THE INVENTION\n\n"
    
    # Add the field text with proper line spacing
    output += field_text
    
    return output


def generate_alternative_versions(abstract: str) -> List[Dict]:
    """
    Generate multiple variations with different emphasis.
    """
    variations = []
    
    # Variation 1: Technology-focused
    prompt1 = f"""Write a Field of Invention emphasizing the technology stack.

Abstract: {abstract}

Structure:
The present invention generally relates to [technology field], particularly to [specific technologies]. More particularly, the present invention relates to [implementation].

Write only the field text:
The present invention"""

    # Variation 2: Application-focused
    prompt2 = f"""Write a Field of Invention emphasizing the application domain.

Abstract: {abstract}

Structure:
The present invention generally relates to [application domain], particularly to [specific application]. More particularly, the present invention relates to [detailed system].

Write only the field text:
The present invention"""

    # Variation 3: Problem-solution focused
    prompt3 = f"""Write a Field of Invention emphasizing the domain and solution.

Abstract: {abstract}

Structure:
The present invention generally relates to the field of [problem domain], particularly to [solution approach]. More particularly, the present invention relates to [specific implementation].

Write only the field text:
The present invention"""

    prompts = [prompt1, prompt2, prompt3]
    labels = ["Technology-focused", "Application-focused", "Problem-solution focused"]
    
    for i, prompt in enumerate(prompts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=280,
                temperature=0.3 + (i * 0.1),
                stop=["\n\nBACKGROUND", "\n\n\n"],
                top_p=0.9,
                repeat_penalty=1.2
            )
            
            raw_text = "The present invention" + response["choices"][0]["text"].strip()
            cleaned = clean_field_text(raw_text)
            validation = validate_field_text(cleaned)
            
            variations.append({
                "label": labels[i],
                "text": cleaned,
                "valid": validation["valid"],
                "word_count": validation["word_count"],
                "sentence_count": validation["sentence_count"]
            })
        except:
            continue
    
    return variations


def print_formatted_report(result: Dict):
    """Print a professional validation report."""
    print("\n" + "=" * 80)
    print("              FIELD OF THE INVENTION - VALIDATION REPORT")
    print("=" * 80)
    
    # Status
    if result["valid"] and len(result["warnings"]) == 0:
        print("\n✅ STATUS: EXCELLENT - Meets Indian Patent Office standards")
    elif result["valid"]:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: NEEDS REVISION - Critical issues found")
    
    # Metrics
    print("\n" + "-" * 80)
    print("📊 METRICS:")
    print(f"   Word Count:        {result['word_count']} words (optimal: 40-120)")
    print(f"   Sentence Count:    {result['sentence_count']} sentences (optimal: 2-4)")
    print(f"   Generation Attempt: {result['attempt']}")
    print(f"   Quality Score:     {result['score']} (lower is better)")
    
    # Technical components detected
    if result.get('components'):
        comp = result['components']
        print("\n" + "-" * 80)
        print("🔍 DETECTED COMPONENTS:")
        if comp.get('broad_field'):
            print(f"   Broad Field:    {comp['broad_field']}")
        if comp.get('application'):
            print(f"   Application:    {comp['application']}")
        if comp.get('technologies'):
            print(f"   Technologies:   {', '.join(comp['technologies'][:5])}")
    
    # Issues
    if result["issues"]:
        print("\n" + "-" * 80)
        print("🚨 CRITICAL ISSUES:")
        for i, issue in enumerate(result["issues"], 1):
            print(f"   {i}. {issue}")
    
    # Warnings
    if result["warnings"]:
        print("\n" + "-" * 80)
        print("⚠️  WARNINGS:")
        for i, warning in enumerate(result["warnings"], 1):
            print(f"   {i}. {warning}")
    
    # The field text
    print("\n" + "=" * 80)
    print("📝 GENERATED FIELD OF THE INVENTION:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)


# CLI for testing locally
if __name__ == "__main__":
    print("=" * 80)
    print("       INDIAN PATENT OFFICE COMPLIANT FIELD OF INVENTION GENERATOR")
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
    
    print("\n⏳ Generating 'Field of the Invention' (analyzing abstract and generating up to 3 versions)...")
    
    result = generate_field_of_invention(abstract, max_attempts=3)
    
    # Print detailed report
    print_formatted_report(result)
    
    # Show formatted version
    print("\n" + "=" * 80)
    print("📄 FORMATTED FOR PATENT DOCUMENT:")
    print("=" * 80)
    print(format_for_patent_document(result["text"]))
    
    # Optional: Generate variations
    print("\n🔄 Generate alternative versions with different emphasis? (y/n): ", end="")
    if input().lower() == 'y':
        print("\n⏳ Generating variations...")
        variations = generate_alternative_versions(abstract)
        
        if variations:
            print("\n" + "=" * 80)
            print("📝 ALTERNATIVE VERSIONS:")
            print("=" * 80)
            
            for i, var in enumerate(variations, 1):
                print(f"\n--- {var['label']} ---")
                print(f"Valid: {'✅' if var['valid'] else '❌'} | Words: {var['word_count']} | Sentences: {var['sentence_count']}")
                print("-" * 80)
                print(var['text'])
                print("-" * 80)
        else:
            print("\n⚠️  Could not generate variations.")
    
    print("\n" + "=" * 80)
    print("💡 TIPS FOR PERFECT FIELD OF INVENTION:")
    print("=" * 80)
    print("1. Follow the three-tier structure: general → particular → more particular")
    print("2. Always start with 'The present invention'")
    print("3. Use 'generally relates to', 'particularly to', 'more particularly'")
    print("4. Keep it 40-100 words, 2-4 sentences")
    print("5. State WHAT it is, not HOW it works or WHY it's better")
    print("6. Avoid marketing language - stay technical and formal")
    print("=" * 80)
