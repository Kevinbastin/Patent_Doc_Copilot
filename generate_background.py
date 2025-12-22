import re
from typing import Dict
from llm_runtime import llm_generate


# ============================================================
# DOMAIN EXTRACTION
# ============================================================
def extract_domain_statistics(abstract: str) -> Dict[str, any]:
    domain_info = {
        'domain': '',
        'problem_keywords': [],
        'technologies': [],
        'application': ''
    }

    abstract_lower = abstract.lower()

    domains = {
        'renewable energy': [
            'wind power', 'wind energy', 'renewable energy',
            'wind turbine', 'power generation', 'generator'
        ],
        'power engineering': [
            'power management', 'energy storage',
            'inverter', 'battery', 'electrical power'
        ],
        'mechanical energy systems': [
            'rotor', 'blades', 'shaft', 'mechanical',
            'aerodynamic', 'folding'
        ],
        'information retrieval': ['search', 'query', 'retrieval', 'search results', 'summarization', 'information processing'],
        'natural language processing': ['natural language', 'nlp', 'text processing', 'language model', 'generative model'],
        'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'neural network'],
        'wildlife conservation': ['wildlife', 'animal', 'conflict', 'elephant', 'conservation'],
        'agriculture': ['agricultural', 'farm', 'crop', 'soil', 'irrigation'],
        'healthcare': ['medical', 'patient', 'diagnosis', 'clinical', 'health'],
        'industrial': ['industrial', 'manufacturing', 'monitoring', 'safety'],
        'smart city': ['urban', 'city', 'infrastructure', 'traffic']
    }

    domain_scores = {}
    for domain, keywords in domains.items():
        domain_scores[domain] = sum(kw in abstract_lower for kw in keywords)

    best_domain = max(domain_scores, key=domain_scores.get)
    domain_info['domain'] = best_domain if domain_scores[best_domain] > 0 else "technology"

    tech_patterns = [
        'IoT', 'LoRaWAN', 'GSM', 'AI', 'machine learning', 'TinyML',
        'edge computing', 'cloud', 'sensor', 'wireless', 'generative model',
        'neural network', 'NLP', 'natural language processing'
    ]

    for tech in tech_patterns:
        if tech.lower() in abstract_lower:
            domain_info['technologies'].append(tech)

    return domain_info


def classify_invention_type(abstract: str) -> str:
    software_keywords = [
        "algorithm", "software", "model", "neural", "language",
        "prediction", "classification", "processing", "data"
    ]
    hardware_keywords = [
        "device", "apparatus", "generator", "turbine",
        "mechanical", "rotor", "battery", "inverter",
        "sensor", "controller", "circuit", "processor"
    ]

    text = abstract.lower()

    # Any physical component → HYBRID
    if any(k in text for k in hardware_keywords):
        return "hybrid"

    # Purely abstract/software
    if any(k in text for k in software_keywords):
        return "software"

    return "hybrid"


# ============================================================
# CLEANING & VALIDATION
# ============================================================
def clean_background_text(text: str) -> str:
    """Clean and format the generated background text."""
    text = re.sub(
        r'^(Background of the Invention:|BACKGROUND OF THE INVENTION:?)\s*',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Remove citation-style numbering like [1] at start of lines
    text = re.sub(r'^\[\d+\]\s*', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    cleaned_paragraphs = []

    for para in paragraphs:
        if para and not para[0].isupper():
            para = para[0].upper() + para[1:]

        if para and not para.endswith('.'):
            para += '.'

        if para:
            cleaned_paragraphs.append(para)

    return '\n\n'.join(cleaned_paragraphs)


def validate_background(text: str, domain_info: Dict[str, any], invention_type: str) -> Dict[str, any]:
    """Validate background section against Indian Patent Office standards."""
    issues = []
    warnings = []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    word_count = len(text.split())
    text_lower = text.lower()

    if word_count < 300:
        warnings.append("Background may be brief for a complex invention.")
    elif word_count > 1000:
        warnings.append("Background is lengthy (>1000 words). Consider condensing.")

    if len(paragraphs) < 4:
        issues.append("Background should have multiple paragraphs covering prior art and problems.")

    has_statistics = bool(re.search(r'\d+%|\d+\s*(million|billion)|\d+\s*(units|systems|devices)', text_lower))


    has_existing_tech = any(phrase in text_lower for phrase in [
        'existing', 'current', 'conventional', 'traditional', 'prior art',
        'known', 'typical', 'commonly', 'previously', 'presently'
    ])

    has_problems = any(phrase in text_lower for phrase in [
        'problem', 'limitation', 'drawback', 'disadvantage', 'difficulty',
        'challenge', 'suffer', 'inadequate', 'inefficient', 'lack', 'fail'
    ])

    has_prior_art_citations = any(
    phrase in text_lower for phrase in [
        "patent", "prior art", "known systems", "existing approaches",
        "earlier methods", "previously proposed", "non-patent literature"
    ]
)

    if not has_prior_art_citations:
        warnings.append(
            "Explicit patent citations are optional; generic references to existing systems are acceptable."
        )


    has_need = any(phrase in text_lower for phrase in [
        'accordingly, there exists a need',
        'there exists a need for improvement',
        'there exists a need for alternative approaches',
        'there exists a need for enhanced systems'
    ])


    if not has_existing_tech:
        issues.append("Missing discussion of existing technology/prior art.")

    if not has_problems:
        issues.append("Should identify problems/limitations with existing technology.")

    if not has_need:
        issues.append("Must end with statement of need (e.g., 'Accordingly, there exists a need...').")

    prohibited_phrases = [
        'the present invention',
        'our invention',
        'we developed',
        'we propose',
        'this invention',
        'addresses the problem',
        'overcomes the drawbacks',
        'solves the limitations',
        'improves upon existing'
    ]

    for phrase in prohibited_phrases:
        if phrase in text_lower:
            issues.append(f"Avoid describing your own invention in Background. Found: '{phrase}'")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "paragraph_count": len(paragraphs),
        "has_statistics": has_statistics,
        "has_existing_tech": has_existing_tech,
        "has_problems": has_problems,
        "has_prior_art_citations": has_prior_art_citations,
        "has_need": has_need
    }

def get_background_example_block(invention_type: str) -> str:
    if invention_type == "software":
        return """
[Paragraph 1: Problem statement with scale and data complexity]
Example: "Modern computing systems process large volumes of structured and unstructured data, creating challenges related to accuracy, scalability, latency, and reliability."

[Paragraph 2-3: Existing software-based approaches]
Example: "Conventional approaches rely on rule-based algorithms, statistical models, or machine learning techniques executed on centralized or distributed computing platforms..."
"""
    elif invention_type == "hardware":
        return """
[Paragraph 1: Problem statement with deployment and operational challenges]
Example: "Portable and stationary power generation devices are increasingly required in remote and off-grid environments, where reliability, efficiency, and ease of deployment are critical."

[Paragraph 2-3: Existing hardware-based technologies]
Example: "Conventional systems employ rigid mechanical assemblies, fixed structural components, and heavy materials that complicate transportation and installation..."
"""
    else:  # hybrid
        return """
[Paragraph 1: Problem statement involving integrated physical and computational systems]
Example: "Modern systems increasingly combine physical devices with embedded computation and control logic, introducing challenges in coordination, efficiency, and adaptability."

[Paragraph 2-3: Existing hybrid hardware-software approaches]
Example: "Existing solutions integrate mechanical components with embedded controllers, firmware, or software platforms to manage operation and performance..."
"""

# ============================================================
# BACKGROUND GENERATION
# ============================================================
def generate_background_locally(abstract: str, max_attempts: int = 3) -> Dict[str, any]:
    """Generate 'Background of the Invention' section for Indian Patent Office format."""
    domain_info = extract_domain_statistics(abstract)
    invention_type = classify_invention_type(abstract)
    example_block = get_background_example_block(invention_type)
    prompt = f"""You are a patent attorney drafting the "Background of the Invention" section for an Indian Complete Specification patent application.

INVENTION ABSTRACT:
{abstract[:1000]}

DOMAIN: {domain_info.get('domain', 'technology')}
TECHNOLOGIES: {', '.join(domain_info.get('technologies', [])) if domain_info.get('technologies') else 'N/A'}

REAL PATENT EXAMPLE STRUCTURE:

BACKGROUND OF THE INVENTION
{example_block}

[Paragraph 4-5: Limitations and drawbacks of current solutions]
Example: "However, existing solutions suffer from several limitations..."

[Paragraph 6-7: Additional existing approaches]
Example: "Other known methods include alternative architectures or configurations..."

[Paragraph 8-9: Further critique of existing solutions]
Example: "Despite these developments, current systems fail to adequately address..."

[Final Paragraph: Statement of need]
"Accordingly, there exists a need for..."

STRICT REQUIREMENTS:
1. Write 5-12 paragraphs (400-800 words total)
2. Structure must follow:
   - Para 1-2: Problem statement with scale or severity (quantitative data only if well-known) Include quantitative data only where appropriate
   - Para 3-4: Existing technologies and how they work
   - Para 5-7: Limitations, drawbacks, and challenges with current solutions
   - Para 8-9: Brief mention of other relevant technologies
   - Para 10-11: Additional critique
   - Final para: MUST end with "Accordingly, there exists a need for..."

3. Use formal, technical, third-person language
4. Include quantitative data where possible
5. CRITICAL: DO NOT describe YOUR invention or solution
6. FORBIDDEN phrases: "the present invention", "our system", "we propose", "this invention"
7. Final para MUST start with:
"Accordingly, there exists a need for improved or alternative systems or methods..."
8. Do NOT copy example sentences verbatim; use them only as guidance.

Write only the background text (no heading). Start with problem statement:

The"""

    best_result = None
    best_score = float('inf')

    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...", end=" ", flush=True)
            
            # CORRECTED: Removed max_input_tokens parameter
            generated = llm_generate(
                prompt,
                max_new_tokens=1800,
                temperature=0.28 if attempt == 0 else 0.32 + (attempt * 0.08),
                top_p=0.88,
                repeat_penalty=1.15,
                stop_strings=["OBJECTS OF THE INVENTION", "SUMMARY OF THE INVENTION", "\n\n\n\n\n", "FIELD OF"]
            )

            if not generated or len(generated.strip()) < 200:
                print("Too short")
                continue

            raw_text = generated.strip()
            cleaned_text = clean_background_text(raw_text)
            validation = validate_background(cleaned_text, domain_info, invention_type)

            score = len(validation["issues"]) * 15 + len(validation["warnings"]) * 3

            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "word_count": validation["word_count"],
                "paragraph_count": validation["paragraph_count"],
                "has_statistics": validation["has_statistics"],
                "has_existing_tech": validation["has_existing_tech"],
                "has_problems": validation["has_problems"],
                "has_prior_art_citations": validation["has_prior_art_citations"],
                "has_need": validation["has_need"],
                "attempt": attempt + 1,
                "domain_info": {
                    **domain_info,
                    "invention_type": invention_type
                },

                "score": score
            }

            print(f"Score: {score}, Words: {validation['word_count']}, Paras: {validation['paragraph_count']}")

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
        "word_count": 0,
        "paragraph_count": 0,
        "attempt": max_attempts,
        "domain_info": domain_info,
        "score": 999
    }


# ============================================================
# FORMATTING
# ============================================================
def format_for_patent_document(background_text: str, include_heading: bool = True, add_line_numbers: bool = False) -> str:
    """Format with Indian Patent Office standard formatting."""
    output = ""

    if include_heading:
        output += "BACKGROUND OF THE INVENTION\n\n"

    if not add_line_numbers:
        output += background_text
        return output

    # Lightweight line numbering
    lines = []
    line_counter = 1
    for para in background_text.split('\n\n'):
        # Crude wrap at 90 chars
        wrapped = [para[i:i+90] for i in range(0, len(para), 90)]
        for w in wrapped:
            if line_counter % 5 == 0:
                lines.append(f"{line_counter:3d} {w}")
            else:
                lines.append(f"    {w}")
            line_counter += 1
        lines.append("")

    output += "\n".join(lines)
    return output


def print_formatted_report(result: Dict):
    """Print professional validation report."""
    print("\n" + "=" * 85)
    print("           BACKGROUND OF THE INVENTION - VALIDATION REPORT")
    print("=" * 85)

    if result["valid"] and len(result["warnings"]) == 0:
        print("\n✅ STATUS: EXCELLENT - Meets Indian Patent Office standards")
    elif result["valid"]:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: NEEDS REVISION - Critical issues found")

    print("\n" + "-" * 85)
    print("📊 METRICS:")
    print(f"   Word Count:         {result['word_count']} words (optimal: 400-800)")
    print(f"   Paragraph Count:    {result['paragraph_count']} paragraphs (optimal: 5-12)")
    print(f"   Generation Attempt: {result['attempt']}")
    print(f"   Quality Score:      {result['score']} (lower is better)")

    print("\n" + "-" * 85)
    print("📋 CONTENT VERIFICATION:")
    print(f"   Statistics/Data:       {'✓' if result['has_statistics'] else '✗'}")
    print(f"   Existing Technology:   {'✓' if result['has_existing_tech'] else '✗'}")
    print(f"   Problems/Limitations:  {'✓' if result['has_problems'] else '✗'}")
    print(f"   Prior Art Citations:   {'✓' if result['has_prior_art_citations'] else '✗'}")
    print(f"   Statement of Need:     {'✓' if result['has_need'] else '✗'}")

    if result.get('domain_info'):
        info = result['domain_info']
        print("\n" + "-" * 85)
        print("🔍 DETECTED DOMAIN:")
        if info.get('domain'):
            print(f"   Domain:        {info['domain']}")
        if info.get('technologies'):
            print(f"   Technologies:  {', '.join(info['technologies'][:8])}")
        if info.get('invention_type'):
            print(f"   Invention Type:{info['invention_type'].upper():>15}")

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
    print("📝 GENERATED BACKGROUND OF THE INVENTION:")
    print("-" * 85)
    print(result["text"])
    print("-" * 85)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("=" * 85)
    print("     INDIAN PATENT OFFICE COMPLIANT BACKGROUND OF INVENTION GENERATOR")
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

    print("\n⏳ Generating 'Background of the Invention'...")
    print("   (This may take 30-60 seconds for longer text generation...)")
    
    result = generate_background_locally(abstract, max_attempts=3)

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

    print("\n🔄 Display with line numbers? (y/n): ", end="")
    if input().lower() == 'y':
        print("\n" + "=" * 85)
        print("📄 WITH LINE NUMBERS (every 5th line):")
        print("=" * 85)
        print(format_for_patent_document(result["text"], include_heading=True, add_line_numbers=True))

    print("\n" + "=" * 85)
    print("💡 TIPS FOR PERFECT BACKGROUND:")
    print("=" * 85)
    print("1. Should discuss existing technology and prior art (not your invention)")
    print("2. Identify problems and limitations with current solutions")
    print("3. Include statistics and quantitative data where possible")
    print("4. End with 'Accordingly, there exists a need for...'")
    print("5. Keep it 400-800 words, 5-12 paragraphs")
    print("6. Use formal, technical, third-person language")
    print("=" * 85)
    print("\n✅ Generation complete!")
    print("=" * 85)
