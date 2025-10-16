from llama_cpp import Llama
import re
from typing import Dict, List


# Path to your locally downloaded Phi-3 model (.gguf file)
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model ONCE at module level
llm = Llama(
    model_path=LLM_PATH, device="cuda",
    n_ctx=4096,
    n_threads=4,
    n_batch=512,
    verbose=False
)


def extract_figure_info_from_abstract(abstract: str) -> Dict[str, any]:
    """
    Extract information from abstract to suggest figures.
    Real patents typically have 5-10 figures.
    """
    info = {
        'system_components': [],
        'subsystems': [],
        'has_method': False,
        'has_data': False,
        'suggested_count': 5
    }
    
    abstract_lower = abstract.lower()
    
    # Check for method/process
    info['has_method'] = any(word in abstract_lower for word in ['method', 'process', 'steps', 'algorithm'])
    
    # Check for data/results
    info['has_data'] = any(word in abstract_lower for word in ['comparative', 'results', 'latency', 'accuracy', 'performance'])
    
    # Extract main system components
    component_patterns = [
        r'comprising[:\s]+([^\.]{20,150})',
        r'includes?\s+([^\.]{20,100})',
        r'consists of\s+([^\.]{20,100})'
    ]
    
    for pattern in component_patterns:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        if matches:
            # Split by commas and semicolons
            parts = re.split(r'[,;]\s*', matches[0])
            info['system_components'].extend([p.strip() for p in parts[:5]])
    
    # Estimate figure count
    base_count = 3  # Minimum: overview + main system + one detail
    if info['system_components']:
        base_count += min(len(info['system_components']), 3)
    if info['has_method']:
        base_count += 1
    if info['has_data']:
        base_count += 2
    
    info['suggested_count'] = min(base_count, 10)
    
    return info


def clean_brief_description(text: str) -> str:
    """Clean and format the brief description text."""
    # Remove header if added
    text = re.sub(r'^(BRIEF DESCRIPTION OF THE DRAWINGS:?)\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove markdown/formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Standardize figure format
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Ensure "Figure X:" format (capital F, colon)
        line = re.sub(r'^[Ff]igure\s*(\d+[A-Z]?)[\s:]*', r'Figure \1: ', line)
        line = re.sub(r'^FIG\.?\s*(\d+[A-Z]?)[\s:]*', r'Figure \1: ', line)
        
        # Ensure ends with period
        if line and not line.endswith('.'):
            line += '.'
        
        lines.append(line)
    
    return '\n'.join(lines)


def validate_brief_description(text: str, expected_count: int = None) -> Dict[str, any]:
    """
    Validate brief description against Indian Patent Office standards.
    Real patent format: "Figure X: illustrates [description]."
    """
    issues = []
    warnings = []
    
    text_lower = text.lower()
    
    # Extract figure numbers
    figure_matches = re.findall(r'Figure\s+(\d+[A-Z]?)', text)
    figure_numbers = [int(re.match(r'(\d+)', f).group(1)) for f in figure_matches]
    
    if not figure_numbers:
        issues.append("No figures found. Must have at least 3-5 figures.")
        return {
            "valid": False,
            "issues": issues,
            "warnings": warnings,
            "figure_count": 0
        }
    
    # Check sequential numbering
    expected_sequence = list(range(1, max(figure_numbers) + 1))
    if sorted(set(figure_numbers)) != expected_sequence:
        issues.append(f"Figures must be numbered sequentially: {expected_sequence}")
    
    # Check minimum figures
    if len(set(figure_numbers)) < 3:
        issues.append("Need at least 3 figures (minimum for patents).")
    
    # Validate each line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    for i, line in enumerate(lines):
        fig_num = i + 1
        
        # Must start with "Figure X:"
        if not re.match(r'^Figure\s+\d+[A-Z]?:\s+', line):
            issues.append(f"Line {i+1}: Must start with 'Figure {fig_num}: '")
        
        # Must contain "illustrates" (Indian Patent Office standard)
        if 'illustrates' not in line.lower():
            warnings.append(f"Figure {fig_num}: Should use 'illustrates' (standard patent language)")
        
        # Should end with period
        if not line.endswith('.'):
            issues.append(f"Figure {fig_num}: Must end with period")
        
        # Check for "according to the present invention" for system figures
        if any(word in line.lower() for word in ['system', 'block diagram', 'setup', 'apparatus', 'device']):
            if 'according to the present invention' not in line.lower():
                warnings.append(f"Figure {fig_num}: System figures should end with 'according to the present invention'")
    
    # Check expected count
    if expected_count and len(set(figure_numbers)) != expected_count:
        warnings.append(f"Generated {len(set(figure_numbers))} figures, expected {expected_count}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "figure_count": len(set(figure_numbers))
    }


def generate_brief_description(abstract: str, num_figures: int = None, 
                               figure_descriptions: str = "", max_attempts: int = 3) -> Dict[str, any]:
    """
    Generate 'Brief Description of the Drawings' section matching Indian Patent Office format.
    
    Real patent format (IN202541069047):
    Figure 1: illustrates a block diagram of IoT based remote monitoring and multi-modal 
              alerting system for human-animal conflict mitigation according to the present invention.
    Figure 2: illustrates setup of the IoT based remote monitoring and multi-modal alerting 
              system according to the present invention.
    ...
    Figure 6: illustrates a comparative network reliability across locations.
    Figure 7: illustrates a latency of edge-based AI decision-making (time to deliver alert).
    
    Args:
        abstract: The patent abstract text
        num_figures: Number of figures (auto-estimated if None)
        figure_descriptions: Optional user-provided figure descriptions
        max_attempts: Number of generation attempts if validation fails
        
    Returns:
        Dictionary containing the generated brief description and metadata
    """
    
    # Extract information from abstract
    fig_info = extract_figure_info_from_abstract(abstract)
    
    if num_figures is None:
        num_figures = fig_info['suggested_count']
    
    # Build prompt based on real patent format
    prompt = f"""You are a patent attorney drafting the "Brief Description of the Drawings" section for an Indian Complete Specification patent application.

INVENTION ABSTRACT:
{abstract}

{f"USER-PROVIDED FIGURE INFO: {figure_descriptions}" if figure_descriptions else ""}

NUMBER OF FIGURES TO GENERATE: {num_figures}

REAL PATENT EXAMPLE (Study this exact format):

BRIEF DESCRIPTION OF THE DRAWINGS

Figure 1: illustrates a block diagram of IoT based remote monitoring and multi-modal alerting system for human-animal conflict mitigation according to the present invention.
Figure 2: illustrates setup of the IoT based remote monitoring and multi-modal alerting system according to the present invention.
Figure 3: illustrates a block diagram of an integrated dual-communication system according to the present invention.
Figure 4: illustrates a block diagram of LoRaWAN module according to the present invention.
Figure 5: illustrates a block diagram of GSM module according to the present invention.
Figure 6: illustrates a comparative network reliability across locations.
Figure 7: illustrates a latency of edge-based AI decision-making (time to deliver alert).

STRICT REQUIREMENTS:
1. Write EXACTLY {num_figures} figure descriptions in this format:
   "Figure X: illustrates [drawing type] of [what it shows] according to the present invention."
   OR (for data/comparison figures):
   "Figure X: illustrates [description of data/comparison]."

2. Always use lowercase "illustrates" after the colon

3. Drawing types to use:
   - "a block diagram of..." (for system architecture, subsystems, modules)
   - "setup of..." (for physical deployment/installation)
   - "a flowchart depicting..." (for method/process)
   - "a detailed view of..." (for component details)
   - "a comparative..." (for comparison charts)
   - "a [metric] of..." (for performance data: latency, accuracy, reliability, etc.)

4. For system/apparatus figures: END with "according to the present invention"
5. For data/comparison figures: NO "according to present invention" - just describe the data

6. Each description is ONE line (can wrap if long, but one sentence)

7. Number figures sequentially: Figure 1, Figure 2, Figure 3, etc.

8. Each line ends with a period

9. Be specific about what each figure shows based on the abstract content

TYPICAL FIGURE STRUCTURE for this type of invention:
- Figure 1: Overall system block diagram
- Figure 2-4: Subsystem block diagrams or setup views
- Figure 5: Method flowchart (if applicable)
- Figure 6-7: Performance data/comparisons (if applicable)

NOW WRITE THE BRIEF DESCRIPTION OF THE DRAWINGS (only the text, no heading):

Figure 1:"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=600,
                temperature=0.2 if attempt == 0 else 0.25 + (attempt * 0.1),
                stop=["DETAILED DESCRIPTION", "SUMMARY OF", "\n\n\n\n"],
                top_p=0.85,
                repeat_penalty=1.2
            )
            
            raw_text = "Figure 1:" + response["choices"][0]["text"].strip()
            cleaned_text = clean_brief_description(raw_text)
            validation = validate_brief_description(cleaned_text, num_figures)
            
            # Calculate quality score (lower is better)
            score = len(validation["issues"]) * 20 + len(validation["warnings"]) * 5
            
            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "figure_count": validation["figure_count"],
                "expected_count": num_figures,
                "attempt": attempt + 1,
                "fig_info": fig_info,
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
        "figure_count": 0,
        "expected_count": num_figures,
        "attempt": max_attempts
    }

# Add this function for backward compatibility with app.py
def generate_drawing_descriptions(abstract: str, num_figures: int = None, max_attempts: int = 2) -> Dict[str, any]:
    """
    Backward compatibility wrapper for existing app.py.
    This function name matches what app.py expects to import.
    
    Generates 'Brief Description of Drawings' section.
    
    Args:
        abstract: The patent abstract text
        num_figures: Number of figures (auto-estimated if None)
        max_attempts: Number of generation attempts
        
    Returns:
        Dictionary with generated text and validation results
    """
    return generate_brief_description(abstract, num_figures, "", max_attempts)

def format_for_patent_document(brief_desc_text: str, include_heading: bool = True) -> str:
    """
    Format the brief description with Indian Patent Office standard heading.
    """
    output = ""
    
    if include_heading:
        output += "BRIEF DESCRIPTION OF THE DRAWINGS\n"
    
    output += brief_desc_text
    
    return output


def print_formatted_report(result: Dict):
    """Print a professional validation report."""
    print("\n" + "=" * 80)
    print("       BRIEF DESCRIPTION OF DRAWINGS - VALIDATION REPORT")
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
    print(f"   Expected Figures:     {result['expected_count']}")
    print(f"   Generated Figures:    {result['figure_count']}")
    print(f"   Generation Attempt:   {result['attempt']}")
    print(f"   Quality Score:        {result['score']} (lower is better)")
    
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
    
    # The brief description text
    print("\n" + "=" * 80)
    print("📝 GENERATED BRIEF DESCRIPTION OF DRAWINGS:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)


# CLI for testing locally
if __name__ == "__main__":
    print("=" * 80)
    print("   INDIAN PATENT OFFICE COMPLIANT BRIEF DESCRIPTION OF DRAWINGS GENERATOR")
    print("=" * 80)
    
    print("\n📥 Enter the invention abstract (press Enter twice to finish):")
    print("-" * 80)
    abstract_lines = []
    while True:
        line = input()
        if line.strip() == "" and abstract_lines:
            break
        if line.strip() != "":
            abstract_lines.append(line)
    abstract = " ".join(abstract_lines).strip()
    
    if not abstract:
        print("\n❌ No abstract provided. Exiting.")
        exit(1)
    
    print("\n🔢 How many figures do you need? (press Enter for auto-suggestion): ", end="")
    num_input = input().strip()
    num_figures = int(num_input) if num_input.isdigit() else None
    
    print("\n📝 Do you have specific figure descriptions? (optional, press Enter to skip):")
    print("    Example: 'Figure 1 system overview, Figure 2 communication module...'")
    print("-" * 80)
    fig_desc_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        fig_desc_lines.append(line)
    figure_descriptions = " ".join(fig_desc_lines).strip()
    
    print("\n⏳ Generating 'Brief Description of Drawings' (analyzing and generating up to 3 versions)...")
    
    result = generate_brief_description(abstract, num_figures, figure_descriptions, max_attempts=3)
    
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
    print("💡 TIPS FOR PERFECT BRIEF DESCRIPTION OF DRAWINGS:")
    print("=" * 80)
    print("1. Format: 'Figure X: illustrates [description].' (lowercase 'illustrates')")
    print("2. System figures: End with 'according to the present invention'")
    print("3. Data figures: Just describe the data (no 'according to...')")
    print("4. Drawing types: block diagram, setup, flowchart, detailed view, comparative")
    print("5. Sequential numbering: Figure 1, Figure 2, Figure 3...")
    print("6. One line per figure (can wrap if long)")
    print("7. Each line ends with period")
    print("8. Be specific - mention key components/subsystems")
    print("9. Typical count: 5-10 figures for complete patents")
    print("10. Match the order figures appear in Detailed Description")
    print("=" * 80)
