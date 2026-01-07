import re
import json
from typing import Dict, List
from llm_runtime import llm_generate


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

# Technical connector words that ARE allowed (kept for reference/possible future checks)
ALLOWED_CONNECTORS = [
    'and', 'or', 'for', 'with', 'using', 'via', 'in', 'of',
    'having', 'comprising', 'including'
]


def clean_title(title: str) -> str:
    """Clean and format the generated title according to USPTO/Indian Patent Office standards."""
    title = re.sub(r'^(Title:|Patent Title:|Generated Title:)\s*', '', title, flags=re.IGNORECASE)
    title = title.strip('"\'`')
    title = title.rstrip('.')

    for word in FORBIDDEN_STARTING_WORDS:
        pattern = r'^(' + re.escape(word) + r')\s+'
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)

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
    generic_patterns = [
        r'\bsystem\b.*\bsystem\b',
        r'\bmethod\b.*\bmethod\b',
        r'\bdevice\b.*\bdevice\b',
        r'\bapparatus\b.*\bapparatus\b',
    ]

    for pattern in generic_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return False, "Contains redundant category words (system, method, etc.)"

    generic_words = ['system', 'device', 'apparatus', 'method', 'process']
    words = title.split()

    if len(words) <= 3 and any(w.lower() in generic_words for w in words):
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

    if char_count > 500:
        issues.append(f"Exceeds 500 character limit ({char_count} chars) - USPTO requirement")

    if word_count < 3:
        issues.append(f"Too short ({word_count} words) - minimum 3 words recommended")
    elif word_count > 15:
        issues.append(f"Too long ({word_count} words) - Indian Patent Office recommends max 15 words")

    if title.endswith('.'):
        issues.append("Remove period at end - patent titles don't use ending punctuation")

    first_word = title.split()[0].lower() if title.split() else ""
    if first_word in FORBIDDEN_STARTING_WORDS:
        issues.append(f"Starts with forbidden word '{first_word}' - will be removed by USPTO")

    weak_found = check_weak_words(title)
    if weak_found:
        warnings.append(f"Contains subjective words: {', '.join(weak_found)} - use technical terms instead")

    is_specific, spec_msg = check_specificity(title)
    if not is_specific:
        warnings.append(spec_msg)

    category_words = ['system', 'method', 'apparatus', 'device', 'composition',
                      'process', 'circuit', 'assembly', 'mechanism']
    has_category = any(re.search(r'\b' + w + r'\b', title, re.IGNORECASE) for w in category_words)
    if not has_category:
        warnings.append("Consider adding category identifier (system, method, apparatus, etc.)")

    if title.isupper():
        cap_style = "ALL CAPS (USPTO standard)"
    elif title.istitle():
        cap_style = "Title Case (acceptable)"
    else:
        cap_style = "Mixed case"
        warnings.append("Consider using ALL CAPS or Title Case for consistency")

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
    """Extract key technical features using LLM to guide title generation."""
    prompt = f"""Identify the core technical subject and primary function of this invention.
Return exactly a JSON object with:
- category: (e.g., "system", "method", "composition", "process")
- subject: the main technical component or entity
- function: what it primary does or enables
- domain: technical field

ABSTRACT:
{abstract[:600]}

JSON OUTPUT:"""
    
    try:
        response = llm_generate(
            prompt,
            max_new_tokens=200,
            temperature=0.1,
            system_prompt="You are a patent analysis engine. Output ONLY valid JSON."
        )
        
        json_text = response.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].strip()
            
        return json.loads(json_text)
    except Exception as e:
        print(f"LLM feature extraction failed: {e}")
        return {"category": "system", "subject": "invention", "function": "technical operation", "domain": "technology"}


def generate_title_from_abstract(abstract: str, max_attempts: int = 5) -> dict:
    """Generate a patent-quality title from an abstract per Indian Patent Office requirements."""
    features = extract_key_features(abstract)
    
    # Determine if multiple categories apply
    abstract_lower = abstract.lower()
    categories = []
    if any(w in abstract_lower for w in ['system', 'apparatus', 'device', 'assembly']):
        categories.append('system/apparatus')
    if any(w in abstract_lower for w in ['method', 'process', 'steps', 'procedure']):
        categories.append('method/process')
    if any(w in abstract_lower for w in ['composition', 'compound', 'mixture', 'formulation']):
        categories.append('composition')
    
    scope_hint = " AND ".join(categories) if categories else "SYSTEM"

    prompt = f"""You are an Indian Patent Attorney drafting a title per Indian Patent Office (IPO) requirements.

INDIAN PATENT OFFICE TITLE REQUIREMENTS:
1. MAXIMUM 15 WORDS (preferably 8-12 words)
2. CLEAR AND SPECIFIC - must show purpose/functionality
3. SHOW SCOPE - indicate if product, process, apparatus, or use
4. TECHNICAL FOCUS - use technical terms only
5. NO VAGUE TERMS - avoid "smart", "novel", "improved", "advanced"
6. NO FANCY NAMES - no trade names, brand names, personal names
7. NO STARTING ARTICLES - never start with "A", "An", "The"
8. PRINTABLE CHARACTERS ONLY - standard ASCII

DETECTED SCOPE: {scope_hint}
TECHNICAL DOMAIN: {features.get('domain', 'technology')}
MAIN SUBJECT: {features.get('subject', 'invention')}
FUNCTION: {features.get('function', 'operation')}

TITLE STRUCTURE OPTIONS:
(a) For apparatus: "[Technical Name] APPARATUS/SYSTEM FOR [Function]"
(b) For method: "METHOD FOR [Technical Function] USING [Key Component]"
(c) For both: "[Name] SYSTEM AND METHOD FOR [Function]"

GOOD IPO TITLE EXAMPLES:
✓ "WATER QUALITY MONITORING SYSTEM WITH MULTI-SENSOR ARRAY" (8 words)
✓ "PHARMACEUTICAL COMPOSITION FOR TREATING RESPIRATORY DISORDERS" (7 words)
✓ "METHOD AND APPARATUS FOR REAL-TIME DATA PROCESSING" (8 words)
✓ "AUTOMATED IRRIGATION SYSTEM FOR PRECISION AGRICULTURE" (7 words)

BAD EXAMPLES (DO NOT USE):
✗ "A Novel Smart System..." (starts with article, uses "novel", "smart")
✗ "Advanced IoT-Based Method..." (uses "advanced")
✗ "The Improved Device..." (starts with "The", uses "improved")

ABSTRACT:
{abstract[:500]}

Generate ONLY the patent title (8-12 words, ALL CAPS, show scope):"""


    best_result = None
    best_score = -1

    for attempt in range(max_attempts):
        raw_title = llm_generate(
            prompt,
            max_new_tokens=60,
            temperature=0.2 if attempt == 0 else 0.3 + (attempt * 0.15),
            top_p=0.85,
            repeat_penalty=1.2,
            stop_strings=["\n\n", "Abstract:", "Explanation:", "Note:", "Example:"],
        )

        cleaned_title = clean_title(raw_title)
        validation = validate_title(cleaned_title)

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

        if validation['valid'] and len(validation['warnings']) == 0:
            break

    return best_result


def format_title_variants(title: str) -> dict:
    """Generate different formatting variants per patent office preferences."""
    return {
        "uspto_standard": title.upper(),
        "title_case": title.title(),
        "sentence_case": title.capitalize(),
        "original": title
    }


def print_validation_report(result: dict):
    """Print a detailed validation report."""
    val = result['validation']

    print("\n" + "=" * 80)
    print("                    PATENT TITLE VALIDATION REPORT")
    print("=" * 80)

    if val['valid'] and len(val['warnings']) == 0:
        print("\n✅ STATUS: EXCELLENT - Ready for filing")
    elif val['valid']:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: INVALID - Critical issues must be fixed")

    print("\n" + "-" * 80)
    print("📊 METRICS:")
    print(f"   Word Count:      {val['word_count']} words ({val['word_quality']})")
    print(f"   Character Count: {val['char_count']} chars (limit: 500)")
    print(f"   Capitalization:  {val['cap_style']}")
    print(f"   Quality Score:   {result['score']}/100")
    print(f"   Attempts Used:   {result['attempt']}/{5}")

    if val['issues']:
        print("\n" + "-" * 80)
        print("🚨 CRITICAL ISSUES (must fix):")
        for i, issue in enumerate(val['issues'], 1):
            print(f"   {i}. {issue}")

    if val['warnings']:
        print("\n" + "-" * 80)
        print("⚠️  WARNINGS (recommended fixes):")
        for i, warning in enumerate(val['warnings'], 1):
            print(f"   {i}. {warning}")

    print("\n" + "=" * 80)
    print("📋 GENERATED TITLE:")
    print("-" * 80)
    print(result['title'])
    print("-" * 80)


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
    print_validation_report(result)

    variants = format_title_variants(result['title'])
    print("\n" + "=" * 80)
    print("📝 FORMATTING OPTIONS:")
    print("-" * 80)
    print(f"USPTO Standard (ALL CAPS):  {variants['uspto_standard']}")
    print(f"Title Case:                 {variants['title_case']}")
    print(f"Sentence Case:              {variants['sentence_case']}")

    print("\n" + "-" * 80)
    print("💡 RECOMMENDATION:")
    print("   For USPTO filing:  Use ALL CAPS format")
    print("   For Indian Patent Office: Either ALL CAPS or Title Case acceptable")
    print("=" * 80)
