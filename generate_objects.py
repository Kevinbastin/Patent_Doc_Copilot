from llama_cpp import Llama
import re
from typing import Dict, List


# Path to your local Phi-3 model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model once
llm = Llama(
    model_path=LLM_PATH, device="cuda",
    n_ctx=4096,
    n_threads=4,
    verbose=False
)


def extract_invention_features(abstract: str) -> Dict[str, any]:
    """
    Extract key features from abstract to guide object generation.
    Real patent objects are based on technical features mentioned in abstract.
    """
    features = {
        'main_system': '',
        'key_technologies': [],
        'primary_benefit': '',
        'specific_features': []
    }
    
    abstract_lower = abstract.lower()
    
    # Extract main system/device name
    system_match = re.search(r'(?:A|An|The)\s+([^,]{15,80}?)\s+(?:comprising|for|system)', abstract, re.IGNORECASE)
    if system_match:
        features['main_system'] = system_match.group(1).strip()
    
    # Extract technologies
    tech_keywords = [
        'IoT', 'AI', 'machine learning', 'TinyML', 'LoRaWAN', 'GSM',
        'edge computing', 'cloud', 'sensor', 'wireless', 'dual communication',
        'neural network', 'deep learning', 'edge AI'
    ]
    
    for tech in tech_keywords:
        if tech.lower() in abstract_lower:
            features['key_technologies'].append(tech)
    
    # Extract specific features (configured to, comprising, includes)
    feature_patterns = [
        r'comprising[:\s]+([^\.]{20,100})',
        r'configured to\s+([^,\.]{15,60})',
        r'includes?\s+([^,\.]{15,60})'
    ]
    
    for pattern in feature_patterns:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        features['specific_features'].extend(matches[:5])
    
    return features


def clean_objects(text: str) -> str:
    """Clean and format the generated objects section."""
    # Remove header if added
    text = re.sub(r'^(OBJECTS OF THE INVENTION:?)\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists (we don't use numbers in patent objects)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points (patent format doesn't use bullets)
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove any === markers
    text = text.replace('===', '').strip()
    
    return text.strip()


def validate_objects(text: str) -> Dict[str, any]:
    """
    Validate objects section against Indian Patent Office standards.
    Real patent has: 1 primary object + 7 additional objects = 8 total objects.
    """
    issues = []
    warnings = []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    word_count = len(text.split())
    
    text_lower = text.lower()
    
    # Check for primary object (mandatory)
    has_primary = 'primary object' in text_lower or 'principal object' in text_lower
    if not has_primary:
        issues.append("Missing 'primary object' paragraph - must start with 'It is the primary object...'")
    
    # Count "It is another object" paragraphs
    another_object_count = len(re.findall(r'it is another object', text_lower))
    
    if another_object_count < 4:
        issues.append(f"Too few objects: found {another_object_count} 'another object' statements. Need at least 4-8.")
    elif another_object_count > 12:
        warnings.append(f"Many objects: {another_object_count}. Consider consolidating to 5-10.")
    
    # Check structure
    if not text.startswith('One or more'):
        warnings.append("Real patents often start with: 'One or more of the problems of the conventional prior arts...'")
    
    # Check for proper language
    required_phrases = ['primary object', 'another object', 'present invention']
    for phrase in required_phrases:
        if phrase not in text_lower:
            issues.append(f"Missing required phrase: '{phrase}'")
    
    # Check word count (real patent objects section: ~350 words)
    if word_count < 200:
        issues.append("Objects section too brief. Should be 250-500 words.")
    elif word_count > 600:
        warnings.append("Objects section lengthy. Consider conciseness.")
    
    # Check for technical specificity
    if not any(word in text_lower for word in ['system', 'method', 'apparatus', 'device', 'module']):
        warnings.append("Include technical category (system, method, apparatus, etc.)")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "paragraph_count": len(paragraphs),
        "has_primary": has_primary,
        "another_object_count": another_object_count
    }


def generate_objects_of_invention(abstract: str, max_attempts: int = 3) -> Dict[str, any]:
    """
    Generate 'Objects of the Invention' section matching Indian Patent Office format.
    
    Real patent structure (IN202541069047):
    Paragraph 1: "One or more of the problems of the conventional prior arts may be overcome..."
    Paragraph 2: "It is the primary object of the present invention to provide [main system]."
    Paragraphs 3-9: "It is another object of the present invention, wherein [specific feature]."
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated objects and metadata
    """
    
    features = extract_invention_features(abstract)
    
    # Build enhanced prompt based on real patent structure
    prompt = f"""You are a patent attorney drafting the "Objects of the Invention" section for an Indian Complete Specification patent application.

INVENTION ABSTRACT:
{abstract}

MAIN SYSTEM: {features.get('main_system', 'system')}
KEY TECHNOLOGIES: {', '.join(features.get('key_technologies', []))}

REAL PATENT EXAMPLE STRUCTURE (Study this carefully):

OBJECTS OF THE INVENTION

One or more of the problems of the conventional prior arts may be overcome by various embodiments of the system and method of the present invention.

It is the primary object of the present invention to provide an Internet of Things (IoT) based remote monitoring and multi-modal alerting system using an integrated dual wireless communication system for human-animal conflict mitigation.

It is another object of the present invention to provide a hybrid monitoring and dual-communication system to ensure continuous data transmission, situational awareness, and real-time alerts in critical field environments.

It is another object of the present invention, wherein the IoT based remote monitoring and multi-modal alerting system integrates edge-based Artificial intelligence (AI) processing, Long Range Wide Area Network (LoRaWAN), and Global System for Mobile Communication (GSM) technologies to deliver redundancy and high availability.

It is another object of the present invention, wherein the IoT based remote monitoring and multi-modal alerting system uses Raspberry Pi with TinyML for real-time accurate elephant detection at 97.1% accuracy together with multi-modal alerting through GSM, LoRaWAN modules and hybrid networking.

It is another object of the present invention, wherein the integrated dual-communication system for environmental monitoring provides dependable alerts across regions with weak network signals.

It is another object of the present invention, wherein the IoT based remote monitoring and multi-modal alerting system provides community-centric multi-modal alerting integrating both forest authorities and local villagers.

It is another object of the present invention, wherein the system as an all-in-one reliability system continues alerting functionality during single network breakdowns.

STRICT REQUIREMENTS:
1. Write EXACTLY in this format:

   Paragraph 1: "One or more of the problems of the conventional prior arts may be overcome by various embodiments of the system and method of the present invention."

   Paragraph 2: "It is the primary object of the present invention to provide [complete description of main system with key technologies]."

   Paragraphs 3-9: "It is another object of the present invention, wherein [specific feature or benefit]."
   OR
   "It is another object of the present invention to provide [specific capability]."

2. Write 6-10 objects total (1 primary + 5-9 additional)

3. Each "another object" should describe:
   - A specific technical feature
   - A unique capability or benefit
   - Integration of technologies
   - Reliability or performance aspect
   - User-facing feature or outcome

4. Use formal patent language:
   - "It is the primary object..."
   - "It is another object..."
   - "wherein the [system]..."
   - "to provide..."

5. Include specific technologies from the abstract (IoT, AI, ML, sensors, communication protocols, etc.)

6. Each object should be 1-3 sentences, focused on ONE aspect

7. DO NOT use bullet points or numbered lists

8. DO NOT use marketing language (revolutionary, best, amazing, etc.)

9. BE SPECIFIC - mention exact technologies, modules, percentages if relevant

NOW WRITE THE OBJECTS OF THE INVENTION (only the text, no heading):

One or more"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=1200,
                temperature=0.25 if attempt == 0 else 0.3 + (attempt * 0.1),
                stop=["SUMMARY OF THE INVENTION", "BRIEF DESCRIPTION", "\n\n\n\n\n"],
                top_p=0.85,
                repeat_penalty=1.18
            )
            
            raw_text = "One or more" + response["choices"][0]["text"].strip()
            cleaned_text = clean_objects(raw_text)
            validation = validate_objects(cleaned_text)
            
            # Calculate quality score (lower is better)
            score = len(validation["issues"]) * 20 + len(validation["warnings"]) * 5
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "word_count": validation["word_count"],
                "paragraph_count": validation["paragraph_count"],
                "has_primary": validation["has_primary"],
                "another_object_count": validation["another_object_count"],
                "attempt": attempt + 1,
                "features": features,
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
        "text": "",
        "valid": False,
        "issues": ["Generation failed"],
        "warnings": [],
        "word_count": 0,
        "paragraph_count": 0,
        "attempt": max_attempts,
        "score": 999
    }


def format_for_patent_document(objects_text: str, include_heading: bool = True) -> str:
    """
    Format the objects text with Indian Patent Office standard heading.
    """
    output = ""
    
    if include_heading:
        output += "OBJECTS OF THE INVENTION\n\n"
    
    output += objects_text
    
    return output


def print_formatted_report(result: Dict):
    """Print a professional validation report."""
    print("\n" + "=" * 80)
    print("          OBJECTS OF THE INVENTION - VALIDATION REPORT")
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
    print(f"   Word Count:           {result['word_count']} words (optimal: 250-500)")
    print(f"   Paragraph Count:      {result['paragraph_count']} paragraphs")
    print(f"   Primary Object:       {'✓' if result['has_primary'] else '✗'}")
    print(f"   Additional Objects:   {result['another_object_count']} (optimal: 5-10)")
    print(f"   Generation Attempt:   {result['attempt']}")
    print(f"   Quality Score:        {result['score']} (lower is better)")
    
    # Features detected
    if result.get('features'):
        feat = result['features']
        print("\n" + "-" * 80)
        print("🔍 DETECTED FEATURES:")
        if feat.get('main_system'):
            print(f"   Main System:       {feat['main_system'][:70]}...")
        if feat.get('key_technologies'):
            print(f"   Technologies:      {', '.join(feat['key_technologies'][:6])}")
    
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
    
    # The objects text
    print("\n" + "=" * 80)
    print("📝 GENERATED OBJECTS OF THE INVENTION:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)


# CLI for testing locally
if __name__ == "__main__":
    print("=" * 80)
    print("    INDIAN PATENT OFFICE COMPLIANT OBJECTS OF INVENTION GENERATOR")
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
    
    print("\n⏳ Generating 'Objects of the Invention' (analyzing abstract and generating up to 3 versions)...")
    
    result = generate_objects_of_invention(abstract, max_attempts=3)
    
    if not result["text"]:
        print("\n❌ ERROR:")
        for issue in result["issues"]:
            print(f"   {issue}")
        exit(1)
    
    # Print detailed report
    print_formatted_report(result)
    
    # Show formatted version
    print("\n" + "=" * 80)
    print("📄 FORMATTED FOR PATENT DOCUMENT (with heading):")
    print("=" * 80)
    print(format_for_patent_document(result["text"], include_heading=True))
    
    print("\n" + "=" * 80)
    print("💡 TIPS FOR PERFECT OBJECTS OF INVENTION:")
    print("=" * 80)
    print("1. Start with: 'One or more of the problems... may be overcome...'")
    print("2. Primary object: 'It is the primary object of the present invention to provide...'")
    print("3. Additional objects: 'It is another object of the present invention, wherein...'")
    print("4. Include 6-10 objects total (1 primary + 5-9 additional)")
    print("5. Each object describes ONE specific feature or benefit")
    print("6. Be technically specific - mention exact technologies")
    print("7. Use 'wherein' to connect objects to specific features")
    print("8. NO bullet points or numbered lists in final format")
    print("9. Each object = 1-3 sentences, focused and concise")
    print("10. Target: 250-500 words total")
    print("=" * 80)
