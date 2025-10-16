from llama_cpp import Llama
import re
from typing import Dict, List


# Path to your GGUF model
LLM_PATH = "./models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model ONCE with optimized settings
llm = Llama(
    model_path=LLM_PATH, device="cuda",
    n_ctx=6144,  # Increased for longer background sections
    n_threads=4,
    n_batch=512,
    verbose=False
)


def extract_domain_statistics(abstract: str) -> Dict[str, any]:
    """
    Extract domain-specific information to generate realistic statistics.
    Real patents include concrete data points.
    """
    domain_info = {
        'domain': '',
        'problem_keywords': [],
        'technologies': [],
        'application': ''
    }
    
    abstract_lower = abstract.lower()
    
    # Detect domain
    domains = {
        'wildlife conservation': ['wildlife', 'animal', 'conflict', 'elephant', 'conservation'],
        'agriculture': ['agricultural', 'farm', 'crop', 'soil', 'irrigation'],
        'healthcare': ['medical', 'patient', 'diagnosis', 'clinical', 'health'],
        'industrial': ['industrial', 'manufacturing', 'monitoring', 'safety'],
        'smart city': ['urban', 'city', 'infrastructure', 'traffic']
    }
    
    for domain, keywords in domains.items():
        if any(kw in abstract_lower for kw in keywords):
            domain_info['domain'] = domain
            break
    
    # Extract technologies
    tech_patterns = [
        'IoT', 'LoRaWAN', 'GSM', 'AI', 'machine learning', 'TinyML',
        'edge computing', 'cloud', 'sensor', 'wireless'
    ]
    
    for tech in tech_patterns:
        if tech.lower() in abstract_lower:
            domain_info['technologies'].append(tech)
    
    return domain_info


def clean_background_text(text: str) -> str:
    """Clean and format the generated background text."""
    # Remove header if LLM added it
    text = re.sub(r'^(Background of the Invention:|BACKGROUND OF THE INVENTION:?)\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove paragraph numbers if added
    text = re.sub(r'^\[\d+\]\s*', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Ensure proper paragraph structure
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    cleaned_paragraphs = []
    
    for para in paragraphs:
        # Capitalize first letter
        if para and not para[0].isupper():
            para = para[0].upper() + para[1:]
        
        # Ensure ends with period
        if para and not para.endswith('.'):
            para += '.'
        
        if para:
            cleaned_paragraphs.append(para)
    
    return '\n\n'.join(cleaned_paragraphs)


def validate_background(text: str) -> Dict[str, any]:
    """
    Validate background section against Indian Patent Office standards.
    Real patent has: 600+ words, 10+ paragraphs, statistics, prior art citations.
    """
    issues = []
    warnings = []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    word_count = len(text.split())
    
    # Check length (real patent background: ~650 words, 11 paragraphs)
    if word_count < 400:
        issues.append("Background too brief. Should be 400-800 words for adequate context.")
    elif word_count > 1000:
        warnings.append("Background is lengthy (>1000 words). Consider condensing.")
    
    # Check paragraph count (real patent has 10+ paragraphs)
    if len(paragraphs) < 5:
        issues.append("Background should have 5-12 paragraphs covering problem, existing solutions, prior art, limitations.")
    
    # Check for required elements
    text_lower = text.lower()
    
    # Problem statement with statistics (real patent has specific numbers)
    has_statistics = bool(re.search(r'\d+%|\d+ per year|\d+ deaths|\d+ increase', text))
    if not has_statistics:
        warnings.append("Consider adding statistics or quantitative data about the problem (e.g., '35% increase', '464 per year').")
    
    # Existing technology discussion
    has_existing_tech = any(phrase in text_lower for phrase in [
        'existing', 'current', 'conventional', 'traditional', 'prior art',
        'known', 'typical', 'commonly', 'previously', 'presently'
    ])
    
    # Problems/limitations
    has_problems = any(phrase in text_lower for phrase in [
        'problem', 'limitation', 'drawback', 'disadvantage', 'difficulty',
        'challenge', 'suffer', 'inadequate', 'inefficient', 'lack', 'fail'
    ])
    
    # Prior art citations (real patent cites specific patents and papers)
    has_prior_art_citations = bool(re.search(r'(CN|IN|US|KR|DE)\d{6,}|Non-patent literature', text))
    if not has_prior_art_citations:
        warnings.append("Consider citing specific prior art (e.g., CN109510971A, IN202041057018).")
    
    # Need statement
    has_need = any(phrase in text_lower for phrase in [
        'need', 'desire', 'would be', 'therefore', 'accordingly',
        'desirable', 'beneficial', 'accordingly, there exists'
    ])
    
    if not has_existing_tech:
        issues.append("Missing discussion of existing technology/prior art.")
    
    if not has_problems:
        issues.append("Should identify problems or limitations with existing technology.")
    
    if not has_need:
        issues.append("Must end with statement of need (e.g., 'Accordingly, there exists a need...').")
    
    # Check for prohibited content (describing your own invention)
    prohibited_phrases = [
        'the present invention solves', 'our invention', 'we developed',
        'we created', 'my invention', 'this invention addresses'
    ]
    
    for phrase in prohibited_phrases:
        if phrase in text_lower:
            issues.append(f"Avoid describing your own invention in Background. Found: '{phrase}'")
    
    # Check structure (real patent has: problem → existing tech → limitations → prior art → need)
    has_logical_flow = has_existing_tech and has_problems and has_need
    if not has_logical_flow:
        warnings.append("Ensure logical flow: (1) problem context, (2) existing solutions, (3) limitations, (4) prior art, (5) statement of need.")
    
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


def generate_background_locally(abstract: str, max_attempts: int = 3) -> Dict[str, any]:
    """
    Generate the 'Background of the Invention' section matching Indian Patent Office format.
    
    Real patent structure (IN202541069047):
    - Paragraph 1: Problem statement with statistics
    - Paragraph 2: More problem context with specific data
    - Paragraphs 3-4: Existing technologies and their limitations
    - Paragraph 5: General technical background (e.g., LPWAN definition)
    - Paragraph 6: Bridge to prior art
    - Paragraphs 7-11: Specific prior art citations with critique
    - Paragraph 12: Statement of need
    
    Args:
        abstract: The patent abstract text
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated background and metadata
    """
    
    domain_info = extract_domain_statistics(abstract)
    
    # Build enhanced prompt based on real patent structure
    prompt = f"""You are a patent attorney drafting the "Background of the Invention" section for an Indian Complete Specification patent application.

INVENTION ABSTRACT:
{abstract}

DOMAIN: {domain_info.get('domain', 'technology')}
TECHNOLOGIES: {', '.join(domain_info.get('technologies', []))}

REAL PATENT EXAMPLE STRUCTURE (Study this carefully):

BACKGROUND OF THE INVENTION

The growing tension between humans and elephants has developed into a crucial ecological conflict matter particularly throughout India because habitat destruction from urbanization and farming growth and deforestation pushes elephants into residential areas where they trigger severe losses for people alongside elephant decline. The number of people killed by elephants in encounters throughout India increased by 35% between the period of 2020–21 and 2023–24 with fatalities starting at 464 per year and reaching 629 per year. High numbers of elephants die from electric shocks as well as being killed by people and trains and through retaliatory action. The main causes behind this conflict stem from habitat destruction and the breakup of natural habitats; these causes resulted in the loss of more than 40% of elephant living spaces in the recent decade. The unpredictable patterns of rainfall due to climate change together with droughts make the situation worse by decreasing available natural vegetation for animals.

Presently, technologies addressing human-animal conflict primarily include physical barriers (trenches, fences), manual patrolling, and basic electronic systems such as camera traps and satellite collars. While these solutions aid in observation and post-incident analysis, they lack real-time detection and intelligent response mechanisms...

[Then discusses specific prior art with citations like CN109510971A, IN202041057018, etc.]

Accordingly, there exists a need for a holistic, cost-effective, and scalable system that combines edge-based AI processing with dual communication technologies for fault-tolerant, long-range, low-power alert transmission.

STRICT REQUIREMENTS:
1. Write 5-12 paragraphs (400-800 words total)
2. Structure:
   Para 1-2: Problem statement with specific statistics/numbers
   Para 3-4: Existing technologies and their operation
   Para 5-6: Limitations and challenges with current solutions
   Para 7-9: Brief mention of relevant technologies (optional: cite patent numbers like CN123456A, IN202012345)
   Para 10-11: Critique of existing solutions
   Para 12: "Accordingly, there exists a need for..." statement

3. Use formal, technical, third-person language
4. Include quantitative data where possible (percentages, time periods, numerical comparisons)
5. DO NOT describe YOUR invention - only existing technology and problems
6. DO NOT use phrases like "the present invention", "our system", "we propose"
7. End with "Accordingly, there exists a need for [what your invention provides]"

8. Prior art citation format (if included):
   - Patent: "CN109510971A disclosed..."
   - Non-patent: "Non-patent literature titled: [Title] disclosed..."

9. Use passive voice and present/past tense
10. Be objective, not promotional

NOW WRITE THE BACKGROUND OF THE INVENTION (only the text, no heading):

The"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.3 if attempt == 0 else 0.35 + (attempt * 0.1),
                stop=["OBJECTS OF THE INVENTION", "SUMMARY OF THE INVENTION", "\n\n\n\n\n"],
                top_p=0.88,
                repeat_penalty=1.15
            )
            
            raw_text = "The" + response["choices"][0]["text"].strip()
            cleaned_text = clean_background_text(raw_text)
            validation = validate_background(cleaned_text)
            
            # Calculate quality score (lower is better)
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
                "domain_info": domain_info,
                "score": score
            }
            
            if validation["valid"] and len(validation["warnings"]) <= 1:
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


def format_for_patent_document(background_text: str, include_heading: bool = True, 
                                add_line_numbers: bool = False) -> str:
    """
    Format the background text with Indian Patent Office standard formatting.
    Includes optional line numbering (every 5 lines) on right margin.
    """
    output = ""
    
    if include_heading:
        output += "BACKGROUND OF THE INVENTION\n\n"
    
    if add_line_numbers:
        # Add line numbers every 5 lines (like in real patent)
        lines = []
        line_counter = 1
        
        for para in background_text.split('\n\n'):
            para_lines = para.split('. ')
            for i, sent in enumerate(para_lines):
                if sent.strip():
                    if i < len(para_lines) - 1:
                        sent += '.'
                    
                    # Wrap at ~75 characters
                    wrapped = [sent[i:i+75] for i in range(0, len(sent), 75)]
                    
                    for wrap_line in wrapped:
                        line_text = wrap_line
                        
                        # Add line number every 5 lines
                        if line_counter % 5 == 0:
                            line_text += f"{line_counter:>80}"
                        
                        lines.append(line_text)
                        line_counter += 1
            
            lines.append("")  # Blank line between paragraphs
            line_counter += 1
        
        output += '\n'.join(lines)
    else:
        output += background_text
    
    return output


def print_formatted_report(result: Dict):
    """Print a professional validation report."""
    print("\n" + "=" * 85)
    print("           BACKGROUND OF THE INVENTION - VALIDATION REPORT")
    print("=" * 85)
    
    # Status
    if result["valid"] and len(result["warnings"]) == 0:
        print("\n✅ STATUS: EXCELLENT - Meets Indian Patent Office standards")
    elif result["valid"]:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: NEEDS REVISION - Critical issues found")
    
    # Metrics
    print("\n" + "-" * 85)
    print("📊 METRICS:")
    print(f"   Word Count:         {result['word_count']} words (optimal: 400-800)")
    print(f"   Paragraph Count:    {result['paragraph_count']} paragraphs (optimal: 5-12)")
    print(f"   Generation Attempt: {result['attempt']}")
    print(f"   Quality Score:      {result['score']} (lower is better)")
    
    # Content checks
    print("\n" + "-" * 85)
    print("📋 CONTENT VERIFICATION:")
    print(f"   Statistics/Data:       {'✓' if result['has_statistics'] else '✗'}")
    print(f"   Existing Technology:   {'✓' if result['has_existing_tech'] else '✗'}")
    print(f"   Problems/Limitations:  {'✓' if result['has_problems'] else '✗'}")
    print(f"   Prior Art Citations:   {'✓' if result['has_prior_art_citations'] else '✗'}")
    print(f"   Statement of Need:     {'✓' if result['has_need'] else '✗'}")
    
    # Domain info
    if result.get('domain_info'):
        info = result['domain_info']
        print("\n" + "-" * 85)
        print("🔍 DETECTED DOMAIN:")
        if info.get('domain'):
            print(f"   Domain:        {info['domain']}")
        if info.get('technologies'):
            print(f"   Technologies:  {', '.join(info['technologies'][:8])}")
    
    # Issues
    if result["issues"]:
        print("\n" + "-" * 85)
        print("🚨 CRITICAL ISSUES:")
        for i, issue in enumerate(result["issues"], 1):
            print(f"   {i}. {issue}")
    
    # Warnings
    if result["warnings"]:
        print("\n" + "-" * 85)
        print("⚠️  WARNINGS:")
        for i, warning in enumerate(result["warnings"], 1):
            print(f"   {i}. {warning}")
    
    # The background text
    print("\n" + "=" * 85)
    print("📝 GENERATED BACKGROUND OF THE INVENTION:")
    print("-" * 85)
    print(result["text"])
    print("-" * 85)


# CLI for testing locally
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
        if line.strip() != "":
            lines.append(line)
    
    abstract = " ".join(lines).strip()
    
    if not abstract:
        print("\n❌ No abstract provided. Exiting.")
        exit(1)
    
    print("\n⏳ Generating 'Background of the Invention' (analyzing abstract and generating up to 3 versions)...")
    
    result = generate_background_locally(abstract, max_attempts=3)
    
    if not result["text"]:
        print("\n❌ ERROR:")
        for issue in result["issues"]:
            print(f"   {issue}")
        exit(1)
    
    # Print detailed report
    print_formatted_report(result)
    
    # Show formatted version
    print("\n" + "=" * 85)
    print("📄 FORMATTED FOR PATENT DOCUMENT (with heading):")
    print("=" * 85)
    print(format_for_patent_document(result["text"], include_heading=True))
    
    # Optional: Show with line numbers
    print("\n🔄 Display with line numbers (like real patent)? (y/n): ", end="")
    if input().lower() == 'y':
        print("\n" + "=" * 85)
        print("📄 WITH LINE NUMBERS (Indian Patent Office style):")
        print("=" * 85)
        print(format_for_patent_document(result["text"], include_heading=True, add_line_numbers=True))
    
    print("\n" + "=" * 85)
    print("💡 TIPS FOR PERFECT BACKGROUND OF INVENTION:")
    print("=" * 85)
    print("1. Start with problem statement + specific statistics (e.g., '35% increase')")
    print("2. Describe existing technologies objectively (what they are, how they work)")
    print("3. Identify limitations and challenges with current solutions")
    print("4. Optional: Cite specific prior art (CN123456A, IN202012345, etc.)")
    print("5. Critique prior art briefly but objectively")
    print("6. End with: 'Accordingly, there exists a need for...'")
    print("7. NEVER describe your own invention in Background")
    print("8. Use third person, present/past tense, passive voice")
    print("9. Include quantitative data where possible (numbers, percentages)")
    print("10. Aim for 400-800 words, 5-12 paragraphs")
    print("=" * 85)
