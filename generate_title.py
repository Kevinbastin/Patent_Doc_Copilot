from llama_cpp import Llama
import re


# Path to your local Phi-3 model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model once
llm = Llama(model_path=LLM_PATH, device="cuda", n_ctx=2048, n_threads=4)


# USPTO MPEP 606 forbidden words that get automatically deleted
FORBIDDEN_STARTING_WORDS = [
    'a', 'an', 'the',
    'improved', 'improvement', 'improvements',
    'new', 'novel',
    'related to',
    'design', 'design for', 'design of',
    'ornamental', 'ornamental design'
]

# Words to avoid anywhere in title (subjective/non-technical)
WEAK_WORDS = [
    'innovative', 'advanced', 'efficient', 'effective', 'smart',
    'intelligent', 'modern', 'revolutionary', 'unique', 'special',
    'enhanced', 'optimized', 'superior', 'better', 'best'
]

# Technical connector words that ARE allowed
ALLOWED_CONNECTORS = [
    'and', 'or', 'for', 'with', 'using', 'via', 'in', 'of',
    'having', 'comprising', 'including'
]


def clean_title(title: str) -> str:
    """Clean and format the generated title according to USPTO/Indian Patent Office standards."""
    # Remove common prefixes that LLMs might add
    title = re.sub(r'^(Title:|Patent Title:|Generated Title:)\s*', '', title, flags=re.IGNORECASE)
    
    # Remove quotes if present
    title = title.strip('"\'`')
    
    # Remove any trailing periods (patent titles don't use periods)
    title = title.rstrip('.')
    
    # Remove forbidden starting words per MPEP 606
    for word in FORBIDDEN_STARTING_WORDS:
        pattern = r'^(' + re.escape(word) + r')\s+'
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    # Strip extra whitespace
    title = ' '.join(title.split())
    
    return title


def check_weak_words(title: str) -> list:
    """Identify weak/subjective words that shouldn't be in patent titles."""
    title_lower = title.lower()
    found_weak = []
    
    for word in WEAK_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', title_lower):
            found_weak.append(word)
    
    return found_weak


def check_specificity(title: str) -> tuple[bool, str]:
    """Check if title is specific enough (not too generic)."""
    # Generic patterns to avoid
    generic_patterns = [
        r'\bsystem\b.*\bsystem\b',  # "system...system" redundancy
        r'\bmethod\b.*\bmethod\b',  # "method...method" redundancy
        r'\bdevice\b.*\bdevice\b',  # device redundancy
        r'\bapparatus\b.*\bapparatus\b',  # apparatus redundancy
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return False, "Contains redundant category words (system, method, etc.)"
    
    # Check for overly generic single-word subjects
    generic_words = ['system', 'device', 'apparatus', 'method', 'process']
    words = title.split()
    
    if len(words) <= 3 and any(word.lower() in generic_words for word in words):
        return False, "Too generic - needs more technical specificity"
    
    return True, "Specific enough"


def validate_title(title: str) -> dict:
    """
    Comprehensive validation according to patent office standards.
    Returns dict with validation details.
    """
    issues = []
    warnings = []
    word_count = len(title.split())
    char_count = len(title)
    
    # CRITICAL ISSUES (must fix)
    
    # 1. Character limit (USPTO: 500 chars, but practical limit much lower)
    if char_count > 500:
        issues.append(f"Exceeds 500 character limit ({char_count} chars) - USPTO requirement")
    
    # 2. Word count validation
    if word_count < 3:
        issues.append(f"Too short ({word_count} words) - minimum 3 words recommended")
    elif word_count > 15:
        issues.append(f"Too long ({word_count} words) - Indian Patent Office recommends max 15 words")
    
    # 3. Ending punctuation
    if title.endswith('.'):
        issues.append("Remove period at end - patent titles don't use ending punctuation")
    
    # 4. Check for forbidden starting words
    first_word = title.split()[0].lower() if title.split() else ""
    if first_word in FORBIDDEN_STARTING_WORDS:
        issues.append(f"Starts with forbidden word '{first_word}' - will be removed by USPTO")
    
    # WARNINGS (should fix for quality)
    
    # 5. Check for weak/subjective words
    weak_found = check_weak_words(title)
    if weak_found:
        warnings.append(f"Contains subjective words: {', '.join(weak_found)} - use technical terms instead")
    
    # 6. Check specificity
    is_specific, spec_msg = check_specificity(title)
    if not is_specific:
        warnings.append(spec_msg)
    
    # 7. Check for proper technical category identifier
    category_words = ['system', 'method', 'apparatus', 'device', 'composition', 
                      'process', 'circuit', 'assembly', 'mechanism']
    has_category = any(re.search(r'\b' + word + r'\b', title, re.IGNORECASE) 
                       for word in category_words)
    if not has_category:
        warnings.append("Consider adding category identifier (system, method, apparatus, etc.)")
    
    # 8. Check capitalization style
    if title.isupper():
        cap_style = "ALL CAPS (USPTO standard)"
    elif title.istitle():
        cap_style = "Title Case (acceptable)"
    else:
        cap_style = "Mixed case"
        warnings.append("Consider using ALL CAPS or Title Case for consistency")
    
    # 9. Optimal word count
    if 5 <= word_count <= 12:
        word_quality = "Optimal"
    elif 3 <= word_count <= 15:
        word_quality = "Acceptable"
    else:
        word_quality = "Needs adjustment"
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "char_count": char_count,
        "cap_style": cap_style,
        "word_quality": word_quality
    }


def extract_key_features(abstract: str) -> dict:
    """Extract key technical features from abstract to guide title generation."""
    # Simple keyword extraction (could be enhanced with NLP)
    abstract_lower = abstract.lower()
    
    features = {
        "has_sensors": any(word in abstract_lower for word in ['sensor', 'detector', 'monitor']),
        "has_ml_ai": any(word in abstract_lower for word in ['machine learning', 'ml', 'ai', 'neural', 'prediction']),
        "has_iot": any(word in abstract_lower for word in ['iot', 'wireless', 'network', 'communication']),
        "has_control": any(word in abstract_lower for word in ['control', 'automation', 'automated']),
        "domain": None
    }
    
    # Detect domain
    domains = {
        'agricultural': ['crop', 'soil', 'irrigation', 'farm', 'agricultural'],
        'medical': ['medical', 'diagnosis', 'patient', 'health', 'clinical'],
        'industrial': ['manufacturing', 'industrial', 'production', 'factory'],
        'environmental': ['environmental', 'pollution', 'climate', 'emission']
    }
    
    for domain, keywords in domains.items():
        if any(kw in abstract_lower for kw in keywords):
            features['domain'] = domain
            break
    
    return features


def generate_title_from_abstract(abstract: str, max_attempts: int = 5) -> dict:
    """
    Generate a patent-quality title from an abstract.
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of regeneration attempts if validation fails
        
    Returns:
        dict with comprehensive results and validation
    """
    
    # Extract features to guide generation
    features = extract_key_features(abstract)
    
    # Build dynamic guidance based on abstract analysis
    feature_guidance = ""
    if features['has_ml_ai'] and features['has_sensors']:
        feature_guidance = "- Mention the sensor type AND the ML/AI capability\n"
    if features['domain']:
        feature_guidance += f"- Include the application domain ({features['domain']})\n"
    
    # Enhanced prompt with patent-specific instructions based on USPTO MPEP 606
    prompt = f"""You are an expert patent attorney drafting titles per USPTO MPEP 606 and WIPO ST.15 standards.

STRICT RULES (violating these makes title INVALID):
1. 5-12 words total (absolute maximum 15 words)
2. NO articles at start: "A", "An", "The"
3. FORBIDDEN WORDS anywhere: "improved", "novel", "new", "innovative", "advanced", "efficient", "smart"
4. NO ending punctuation (no periods, no commas at end)
5. NO subjective/marketing terms - only technical descriptors
6. Must indicate WHAT it is (system/method/apparatus) AND WHAT it does

REQUIRED ELEMENTS:
- Technical category (System, Method, Apparatus, Device, Circuit, Assembly, Composition, Process)
- Key innovation or distinguishing feature
- Application domain or function
{feature_guidance}

GOOD EXAMPLES (study the pattern):
✓ "MULTI-DEPTH SOIL SENSOR ARRAY WITH PREDICTIVE IRRIGATION CONTROL"
✓ "WIRELESS AGRICULTURAL MONITORING SYSTEM USING MACHINE LEARNING"
✓ "IOT-ENABLED CROP MANAGEMENT APPARATUS WITH MULTI-PARAMETER SENSING"
✓ "METHOD FOR AUTOMATED IRRIGATION SCHEDULING BASED ON SOIL MOISTURE PREDICTION"

BAD EXAMPLES (avoid these patterns):
✗ "An Improved Smart Agricultural System" (has article, "improved", "smart")
✗ "Novel IoT-Based Monitoring Device" (has "novel")
✗ "Advanced Precision Agriculture Technology." (has "advanced", has period)
✗ "Efficient Crop Monitoring System" (has "efficient", too generic)

Abstract to analyze:
{abstract}

Generate ONLY the patent title (no explanation, no prefix, no quotes):"""

    best_result = None
    best_score = -1
    
    for attempt in range(max_attempts):
        response = llm(
            prompt=prompt,
            max_tokens=60,
            temperature=0.2 if attempt == 0 else 0.3 + (attempt * 0.15),
            stop=["\n\n", "Abstract:", "Explanation:", "Note:", "Example:"],
            top_p=0.85,
            repeat_penalty=1.2
        )
        
        raw_title = response["choices"][0]["text"].strip()
        cleaned_title = clean_title(raw_title)
        
        validation = validate_title(cleaned_title)
        
        # Calculate score (higher is better)
        score = 0
        if validation['valid']:
            score += 100
        score -= len(validation['issues']) * 20
        score -= len(validation['warnings']) * 5
        if 5 <= validation['word_count'] <= 12:
            score += 20
        
        if score > best_score:
            best_score = score
            best_result = {
                "title": cleaned_title,
                "validation": validation,
                "attempt": attempt + 1,
                "score": score
            }
        
        # If perfect, stop early
        if validation['valid'] and len(validation['warnings']) == 0:
            break
    
    return best_result


def format_title_variants(title: str) -> dict:
    """Generate different formatting variants per patent office preferences."""
    return {
        "uspto_standard": title.upper(),  # USPTO prefers ALL CAPS
        "title_case": title.title(),       # Also acceptable
        "sentence_case": title.capitalize(),
        "original": title
    }


def print_validation_report(result: dict):
    """Print a detailed validation report."""
    val = result['validation']
    
    print("\n" + "=" * 80)
    print("                    PATENT TITLE VALIDATION REPORT")
    print("=" * 80)
    
    # Status
    if val['valid'] and len(val['warnings']) == 0:
        print("\n✅ STATUS: EXCELLENT - Ready for filing")
    elif val['valid']:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: INVALID - Critical issues must be fixed")
    
    # Metrics
    print("\n" + "-" * 80)
    print("📊 METRICS:")
    print(f"   Word Count:     {val['word_count']} words ({val['word_quality']})")
    print(f"   Character Count: {val['char_count']} chars (limit: 500)")
    print(f"   Capitalization: {val['cap_style']}")
    print(f"   Quality Score:  {result['score']}/100")
    print(f"   Attempts Used:  {result['attempt']}/{5}")
    
    # Critical Issues
    if val['issues']:
        print("\n" + "-" * 80)
        print("🚨 CRITICAL ISSUES (must fix):")
        for i, issue in enumerate(val['issues'], 1):
            print(f"   {i}. {issue}")
    
    # Warnings
    if val['warnings']:
        print("\n" + "-" * 80)
        print("⚠️  WARNINGS (recommended fixes):")
        for i, warning in enumerate(val['warnings'], 1):
            print(f"   {i}. {warning}")
    
    # Title Display
    print("\n" + "=" * 80)
    print("📋 GENERATED TITLE:")
    print("-" * 80)
    print(result['title'])
    print("-" * 80)


# Example CLI usage
if __name__ == "__main__":
    print("=" * 80)
    print("            USPTO/INDIAN PATENT OFFICE COMPLIANT TITLE GENERATOR")
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
    
    print("\n⏳ Generating patent-compliant title...")
    print("   (Analyzing abstract and generating up to 5 variations...)")
    
    result = generate_title_from_abstract(abstract, max_attempts=5)
    
    # Print validation report
    print_validation_report(result)
    
    # Show formatting variants
    variants = format_title_variants(result['title'])
    print("\n" + "=" * 80)
    print("📝 FORMATTING OPTIONS:")
    print("-" * 80)
    print(f"USPTO Standard (ALL CAPS):  {variants['uspto_standard']}")
    print(f"Title Case:                 {variants['title_case']}")
    print(f"Sentence Case:              {variants['sentence_case']}")
    
    # Best practice recommendation
    print("\n" + "-" * 80)
    print("💡 RECOMMENDATION:")
    print("   For USPTO filing:  Use ALL CAPS format")
    print("   For Indian Patent Office: Either ALL CAPS or Title Case acceptable")
    print("=" * 80)
