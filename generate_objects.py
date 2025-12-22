import re
from typing import Dict, List
from llm_runtime import llm_generate


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_key_features_from_abstract(abstract: str) -> Dict[str, any]:
    """Extract key features and advantages to guide objects generation."""
    features = {
        'main_problem': '',
        'key_technologies': [],
        'benefits': [],
        'applications': []
    }
    
    abstract_lower = abstract.lower()
    
    # Extract technologies
    tech_keywords = [
        'machine learning', 'ai', 'generative model', 'neural network',
        'iot', 'sensor', 'wireless', 'cloud', 'edge computing',
        'natural language', 'nlp', 'search', 'retrieval', 'summarization'
    ]
    
    for tech in tech_keywords:
        if tech in abstract_lower:
            features['key_technologies'].append(tech)
    
    # Extract potential benefits
    benefit_patterns = [
        r'(improv\w+|enhanc\w+|optim\w+|reduc\w+|increas\w+|minimi\w+|maximi\w+)',
        r'(accuracy|efficiency|speed|cost|time|performance|reliability)'
    ]
    
    for pattern in benefit_patterns:
        matches = re.findall(pattern, abstract_lower)
        features['benefits'].extend(matches[:3])
    
    return features


# ============================================================
# CLEANING & VALIDATION
# ============================================================
def clean_objects_text(text: str) -> str:
    """Clean and format the generated objects text."""
    # Remove heading if present
    text = re.sub(
        r'^(Objects of the Invention:|OBJECTS OF THE INVENTION:?)\s*',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Remove extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Process each object
    objects = [p.strip() for p in text.split('\n\n') if p.strip()]

    
    return '\n\n'.join(objects)


def validate_objects(text: str) -> Dict[str, any]:
    """Validate objects section against Indian Patent Office standards."""
    issues = []
    warnings = []
    
    # Count objects (should be separate lines/paragraphs)
    objects = [line.strip() for line in text.split('\n\n') if line.strip()]
    object_count = len(objects)
    word_count = len(text.split())
    text_lower = text.lower()
    
    # Check number of objects
    if object_count < 3:
        issues.append("Too few objects. Should have 3-8 distinct objects.")
    elif object_count > 12:
        warnings.append("Many objects (>12). Consider consolidating.")
    
    # Check word count
    if word_count < 60:
        issues.append("Objects section too brief. Should be at least 60–80 words.")
    elif word_count > 500:
        warnings.append("Objects section lengthy (>500 words). Consider condensing.")
    elif word_count > 350:
        warnings.append("Objects section moderately lengthy (>350 words).")
    
    # Check for required phrases
    required_starters = [
        'primary object', 'main object', 'principal object',
        'another object', 'further object', 'additional object',
        'object of', 'it is an object', 'it is therefore an object'
    ]
    
    has_proper_structure = any(phrase in text_lower for phrase in required_starters)
    if not has_proper_structure:
        issues.append("Objects should start with 'primary object', 'another object', etc.")
    
    # Check if first object mentions "primary" or "main"
    first_object_lower = objects[0].lower() if objects else ""
    has_primary = any(word in first_object_lower for word in ['primary', 'main', 'principal'])
    if not has_primary:
        warnings.append("First object should typically start with 'primary object' or 'main object'.")
    
    # Check for "to provide" structure
    has_to_provide = 'to provide' in text_lower
    if not has_to_provide:
        warnings.append("Objects typically use 'to provide' structure.")
    
    # Prohibited phrases
    prohibited = [
        'novel', 'innovative', 'revolutionary', 'groundbreaking',
        'best', 'superior', 'perfect'
    ]
    found_prohibited = [word for word in prohibited if re.search(r'\b' + word + r'\b', text_lower)]
    if found_prohibited:
        issues.append(f"Avoid marketing language: {', '.join(found_prohibited)}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "object_count": object_count,
        "word_count": word_count,
        "has_proper_structure": has_proper_structure,
        "has_primary": has_primary
    }


# ============================================================
# OBJECTS GENERATION
# ============================================================
def generate_objects_of_invention(abstract: str, max_attempts: int = 3) -> Dict[str, any]:
    """Generate 'Objects of the Invention' section for Indian Patent Office format."""
    features = extract_key_features_from_abstract(abstract)
    
    prompt = f"""You are a patent attorney drafting the "Objects of the Invention" section for an Indian Complete Specification patent application.

INVENTION ABSTRACT:
{abstract[:1000]}

KEY TECHNOLOGIES: {', '.join(features['key_technologies'][:5]) if features['key_technologies'] else 'N/A'}

REAL PATENT EXAMPLE STRUCTURE:

OBJECTS OF THE INVENTION

The primary object of the present invention is to provide a system that addresses the limitations of existing technologies.

Another object of the present invention is to provide a method that improves accuracy and efficiency.

A further object of the present invention is to provide a cost-effective solution for real-time processing.

Yet another object of the present invention is to provide a user-friendly interface for system configuration.

An additional object of the present invention is to provide enhanced reliability and fault tolerance.

Still another object of the present invention is to provide a scalable architecture suitable for various deployment scenarios.

STRICT REQUIREMENTS:
1. Write 4-8 distinct objects (each object = 1 paragraph/line)
2. Each object must start with one of these phrases:
   - "The primary object of the present invention is to provide..."
   - "Another object of the present invention is to provide..."
   - "A further object of the present invention is to provide..."
   - "Yet another object of the present invention is to provide..."
   - "An additional object of the present invention is to provide..."
   - "Still another object of the present invention is to provide..."

3. Structure of each object:
   - Start with required phrase
   - State WHAT is provided (system/method/apparatus)
   - Include specific benefit or capability
   - Keep each object to 1-2 sentences (20-40 words)

4. Content guidelines:
   - First object should be PRIMARY and most general
   - Subsequent objects should be more specific features/benefits
   - Focus on technical advantages, not marketing claims
   - Each object should be distinct (no repetition)

5. FORBIDDEN:
   - Marketing words: "revolutionary", "innovative", "best", "unique", "novel"
   - Vague benefits: "improved system", "better performance"
   - Comparisons to competitors

6. REQUIRED:
   - Use "to provide" structure
   - Mention specific technical capabilities
   - Focus on functional objectives, not implementation details

Write ONLY the objects (no heading). Each object on a new line. Start with:

The primary object"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...", end=" ", flush=True)
            
            generated = llm_generate(
                prompt,
                max_new_tokens=600,
                temperature=0.25 if attempt == 0 else 0.30 + (attempt * 0.08),
                top_p=0.88,
                repeat_penalty=1.18,
                stop_strings=["SUMMARY OF THE INVENTION", "BRIEF DESCRIPTION", "\n\n\n\n", "DETAILED DESCRIPTION"]
            )
            
            if not generated or len(generated.strip()) < 80:
                print("Too short")
                continue
            
            raw_text = generated.strip()
            cleaned_text = clean_objects_text(raw_text)
            validation = validate_objects(cleaned_text)
            
            score = len(validation["issues"]) * 15 + len(validation["warnings"]) * 3
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "object_count": validation["object_count"],
                "word_count": validation["word_count"],
                "has_proper_structure": validation["has_proper_structure"],
                "attempt": attempt + 1,
                "features": features,
                "score": score
            }
            
            print(f"Score: {score}, Objects: {validation['object_count']}, Words: {validation['word_count']}")
            
            if validation["valid"] and len(validation["warnings"]) <= 1:
                print("   ✅ Excellent!")
                return result
            
            if score < best_score:
                best_score = score
                best_result = result
        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return best_result if best_result else {
        "text": "",
        "valid": False,
        "issues": ["Generation failed after all attempts"],
        "warnings": [],
        "object_count": 0,
        "word_count": 0,
        "attempt": max_attempts,
        "features": features,
        "score": 999
    }


# ============================================================
# FORMATTING
# ============================================================
def format_for_patent_document(objects_text: str, include_heading: bool = True) -> str:
    """Format with Indian Patent Office standard formatting."""
    output = ""
    
    if include_heading:
        output += "OBJECTS OF THE INVENTION\n\n"
    
    output += objects_text
    return output


def print_formatted_report(result: Dict):
    """Print professional validation report."""
    print("\n" + "=" * 85)
    print("            OBJECTS OF THE INVENTION - VALIDATION REPORT")
    print("=" * 85)
    
    if result["valid"] and len(result["warnings"]) == 0:
        print("\n✅ STATUS: EXCELLENT - Meets Indian Patent Office standards")
    elif result["valid"]:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: NEEDS REVISION - Critical issues found")
    
    print("\n" + "-" * 85)
    print("📊 METRICS:")
    print(f"   Number of Objects:  {result['object_count']} (optimal: 4-8)")
    print(f"   Word Count:         {result['word_count']} words (optimal: 150-400)")
    print(f"   Attempt Used:       {result['attempt']}")
    print(f"   Quality Score:      {result['score']} (lower is better)")
    
    print("\n" + "-" * 85)
    print("📋 STRUCTURE VERIFICATION:")
    print(f"   Proper Structure:   {'✓' if result['has_proper_structure'] else '✗'}")
    
    if result.get('features'):
        feat = result['features']
        print("\n" + "-" * 85)
        print("🔍 DETECTED FEATURES:")
        if feat.get('key_technologies'):
            print(f"   Technologies:  {', '.join(feat['key_technologies'][:5])}")
    
    if result["issues"]:
        print("\n" + "-" * 85)
        print("🚨 CRITICAL ISSUES:")
        for i, issue in enumerate(result["issues"], 1):
            print(f"   {i}. {issue}")
    
    if result["warnings"]:
        print("\n" + "-" * 85)
        print("⚠️  WARNINGS:")
        for i, warning in enumerate(result["warnings"], 1):
            print(f"   {i}. {warning}")
    
    print("\n" + "=" * 85)
    print("📝 GENERATED OBJECTS OF THE INVENTION:")
    print("-" * 85)
    print(result["text"])
    print("-" * 85)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("=" * 85)
    print("      INDIAN PATENT OFFICE COMPLIANT OBJECTS OF INVENTION GENERATOR")
    print("=" * 85)
    print("\n📥 Enter the invention abstract (press Enter twice to finish):")
    print("-" * 85)
    
    lines = []
    while True:
        line = input()
        if line.strip() == "" and lines:
            break
        if line.strip():
            lines.append(line)
    
    abstract = " ".join(lines).strip()
    
    if not abstract:
        print("\n❌ No abstract provided. Exiting.")
        exit(1)
    
    print("\n⏳ Generating 'Objects of the Invention'...")
    
    result = generate_objects_of_invention(abstract, max_attempts=3)
    
    if not result["text"]:
        print("\n❌ ERROR: Generation failed")
        for issue in result["issues"]:
            print(f"   • {issue}")
        exit(1)
    
    print_formatted_report(result)
    
    print("\n" + "=" * 85)
    print("📄 FORMATTED FOR PATENT DOCUMENT:")
    print("=" * 85)
    print(format_for_patent_document(result["text"], include_heading=True))
    
    print("\n" + "=" * 85)
    print("💡 TIPS FOR PERFECT OBJECTS:")
    print("=" * 85)
    print("1. Start with 'primary object', then 'another object', 'further object', etc.")
    print("2. Use 'to provide' structure for each object")
    print("3. First object should be most general, subsequent ones more specific")
    print("4. Focus on technical capabilities, not marketing claims")
    print("5. Keep 4-8 distinct objects, each 20-40 words")
    print("6. Each object should describe a functional objective")
    print("=" * 85)
    print("\n✅ Generation complete!")
    print("=" * 85)
