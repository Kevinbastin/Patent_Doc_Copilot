import re
from typing import Dict
from llm_runtime import llm_generate


# ============================================================
# FIGURE INFORMATION EXTRACTION
# ============================================================
def extract_figure_info_from_abstract(abstract: str) -> Dict[str, any]:
    """Extract information from abstract to suggest figures."""
    info = {
        'system_components': [],
        'subsystems': [],
        'has_method': False,
        'has_data': False,
        'suggested_count': 5
    }

    abstract_lower = abstract.lower()

    component_patterns = [
        r'comprising[:\s]+([^\.]{20,150})',
        r'includes?\s+([^\.]{20,100})',
        r'consists of\s+([^\.]{20,100})'
    ]

    for pattern in component_patterns:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        if matches:
            parts = re.findall(
                r'\b(unit|module|assembly|system|device|controller|housing|shaft|rotor|stator)\b',
                matches[0],
                re.IGNORECASE
            )
            info['system_components'].extend(parts[:5])


    base_count = 3

    if info['system_components']:
        base_count += min(len(info['system_components']), 3)


    # Final safe cap for Indian patents
    info['suggested_count'] = min(max(base_count, 3), 5)

    return info


# ============================================================
# CLEANING & VALIDATION
# ============================================================
def clean_brief_description(text: str) -> str:
    # Remove heading
    text = re.sub(
        r'^(BRIEF DESCRIPTION OF THE DRAWINGS:?)\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Force each Figure onto a new line
    text = re.sub(r'\s+(Figure\s+\d+)', r'\n\1', text)

    # Clean formatting
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Normalize to "Figure X illustrates" format (no colon)
        line = re.sub(
            r'^Figure\s*(\d+)\s*:?\s*illustrates',
            r'Figure \1 illustrates',
            line,
            flags=re.IGNORECASE
        )
        
        # If still has colon format, convert it
        if re.match(r'^Figure\s+\d+\s*:', line, re.IGNORECASE):
            line = re.sub(r'^Figure\s*(\d+)\s*:\s*', r'Figure \1 illustrates ', line, flags=re.IGNORECASE)

        if not line.endswith('.'):
            line += '.'

        lines.append(line)

    return "\n".join(lines)




def validate_brief_description(text: str, expected_count: int = None) -> Dict[str, any]:
    """Validate brief description against Indian Patent Office standards."""
    issues = []
    warnings = []

    figure_matches = re.findall(r'Figure\s+(\d+[A-Z]?)', text)
    figure_numbers = [int(re.match(r'(\d+)', f).group(1)) for f in figure_matches] if figure_matches else []

    if not figure_numbers:
        issues.append("No figures found. Must have at least 3-5 figures.")
        return {"valid": False, "issues": issues, "warnings": warnings, "figure_count": 0}

    expected_sequence = list(range(1, max(figure_numbers) + 1))
    if sorted(set(figure_numbers)) != expected_sequence:
        issues.append("Figures must be numbered sequentially")

    if len(set(figure_numbers)) < 3:
        issues.append("Need at least 3 figures (minimum for patents).")

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    for i, line in enumerate(lines):
        fig_num = i + 1
        functional_verbs = [
            "adjust", "control", "monitor", "regulate",
            "process", "convert", "generate", "manage",
            "detect", "transmit", "stabilize", "compute"
        ]

        if any(v in line.lower() for v in functional_verbs):
            issues.append(
                f"Figure {fig_num}: Functional or operational language is not permitted in drawings"
            )

        # Accept both "Figure X:" and "Figure X illustrates" formats
        if not re.match(r'^Figure\s+\d+[A-Z]?\s+(illustrates|:)', line, re.IGNORECASE):
            issues.append(f"Line {i+1}: Must start with 'Figure X illustrates'")

        if "illustrates" not in line.lower():
           issues.append(f"Figure {fig_num}: Must use the word 'illustrates'")


        if not line.endswith('.'):
            issues.append(f"Figure {fig_num}: Must end with period")

        # Check for "in accordance with" or "according to" for structural figures
        if any(word in line.lower() for word in ['system', 'block diagram', 'setup', 'apparatus', 'device', 'view']):
            if 'in accordance with' not in line.lower() and 'according to the present invention' not in line.lower():
                warnings.append(
                    f"Figure {fig_num}: Consider adding 'in accordance with an embodiment of the present invention'"
                )


    if expected_count and len(set(figure_numbers)) != expected_count:
        warnings.append(f"Generated {len(set(figure_numbers))} figures, expected {expected_count}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "figure_count": len(set(figure_numbers))
    }


# ============================================================
# BRIEF DESCRIPTION GENERATION
# ============================================================
def generate_brief_description(
    abstract: str,
    num_figures: int = None,
    figure_descriptions: str = "",
    max_attempts: int = 3
) -> Dict[str, any]:
    """Generate 'Brief Description of the Drawings' section matching Indian Patent Office format."""
    fig_info = extract_figure_info_from_abstract(abstract)

    if num_figures is None:
        num_figures = fig_info['suggested_count']

    # Extract invention name from abstract (just the main noun phrase)
    invention_name = "the present invention"
    # Try to extract: "A [device/system/apparatus]" up to "comprising" or first comma
    match = re.search(r'^A\s+([a-zA-Z\s]+?)(?:\s+comprising|\s+for|\s+with|,|\.)', abstract, re.IGNORECASE)
    if match:
        invention_name = "the " + match.group(1).strip().lower()
    else:
        # Fallback: get first 3-5 words after "A"
        match = re.search(r'^A\s+((?:\w+\s*){1,4})', abstract, re.IGNORECASE)
        if match:
            invention_name = "the " + match.group(1).strip().lower()
    
    prompt = f"""You are a senior patent attorney drafting the "Brief Description of the Drawings" for an Indian Complete Specification patent.

INVENTION NAME: {invention_name}
INVENTION ABSTRACT: {abstract[:800]}

{f"USER DRAWING NOTES: {figure_descriptions}" if figure_descriptions else ""}

NUMBER OF FIGURES TO GENERATE: {num_figures}

=== TRANSFORMATION LOGIC ===
Transform simple drawing notes into formal patent language:

INPUT: Fig 1: Front view of bottle
OUTPUT: Figure 1 illustrates a front view of {invention_name}, in accordance with an embodiment of the present invention.

INPUT: Fig 2: Top view showing cap
OUTPUT: Figure 2 illustrates a top planar view of {invention_name} showing the cap assembly, in accordance with an embodiment of the present invention.

INPUT: Fig 3: Exploded view
OUTPUT: Figure 3 illustrates an exploded perspective view of {invention_name}, depicting the internal component arrangement, in accordance with an embodiment of the present invention.

INPUT: Fig 4: Flowchart of app
OUTPUT: Figure 4 illustrates a flowchart depicting the method of operation of {invention_name}.

INPUT: Fig 5: Block diagram
OUTPUT: Figure 5 illustrates a block diagram of the system architecture of {invention_name}, in accordance with an embodiment of the present invention.

=== MANDATORY FORMAT ===
- Line format: "Figure X illustrates [a/an] [view type] [of invention name], [context phrase]."
- Use "illustrates" (not "is" or "shows")
- Add view qualifiers: "front", "side", "top planar", "perspective", "cross-sectional", "exploded"
- Include invention name: "{invention_name}"
- End structural figures with: "in accordance with an embodiment of the present invention."
- End flowcharts/method figures with: "depicting the method of [action] of {invention_name}."
- Each figure on separate line
- Write exactly {num_figures} figures (Figure 1 to Figure {num_figures})

NOW GENERATE {num_figures} FIGURE DESCRIPTIONS (one per line):

Figure 1 illustrates"""


    best_result = None
    best_score = float('inf')

    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...", end=" ", flush=True)
            
            # CORRECTED: Removed max_input_tokens parameter
            generated = llm_generate(
                prompt,
                max_new_tokens=650,
                temperature=0.18 if attempt == 0 else 0.22 + (attempt * 0.08),
                top_p=0.85,
                repeat_penalty=1.22,
                stop_strings=["\n\n\n", "\n\nDETAILED", "\n\nCLAIMS"]
            )

            if not generated or len(generated.strip()) < 50:
                print("Too short")
                continue

            raw_text = generated.strip()
            cleaned_text = clean_brief_description(raw_text)
            validation = validate_brief_description(cleaned_text, num_figures)

            score = len(validation["issues"]) * 20 + len(validation["warnings"]) * 5

            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "figure_count": validation["figure_count"],
                "expected_count": num_figures,
                "attempt": attempt + 1,
                "score": score
            }

            print(f"Score: {score}, Figures: {validation['figure_count']}/{num_figures}")
            print("----- GENERATED TEXT -----")
            print(cleaned_text)
            print("--------------------------")


            if validation["valid"] and len(validation["warnings"]) <= 1:
                print("   ✅ Good!")
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
        "issues": ["Generation failed"],
        "warnings": [],
        "figure_count": 0,
        "expected_count": num_figures,
        "attempt": max_attempts,
        "score": 999
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================
def generate_drawing_descriptions(abstract: str, num_figures: int = None, max_attempts: int = 2) -> Dict[str, any]:
    """Backward compatibility wrapper for existing app.py."""
    return generate_brief_description(abstract, num_figures, "", max_attempts)


# ============================================================
# FORMATTING
# ============================================================
def format_for_patent_document(brief_desc_text: str, include_heading: bool = True) -> str:
    """Format the brief description with Indian Patent Office standard heading."""
    if include_heading:
        return f"BRIEF DESCRIPTION OF THE DRAWINGS\n\n{brief_desc_text}"
    return brief_desc_text


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("   BRIEF DESCRIPTION OF DRAWINGS GENERATOR")
    print("=" * 80)

    print("\n📥 Abstract (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "" and lines:
            break
        if line.strip():
            lines.append(line)
    
    abstract = " ".join(lines).strip()

    if not abstract:
        print("❌ Abstract required")
        exit(1)

    print("\n🔢 Number of figures? (Enter for auto): ", end="")
    num_input = input().strip()
    num_figures = int(num_input) if num_input.isdigit() else None

    print("\n⏳ Generating...")
    result = generate_brief_description(abstract, num_figures)

    if not result["text"]:
        print("\n❌ ERROR:")
        for issue in result["issues"]:
            print(f"   • {issue}")
        exit(1)

    print("\n" + "=" * 80)
    print(f"✅ Generated {result['figure_count']}/{result['expected_count']} figures | Attempt: {result['attempt']}")

    if result["issues"]:
        print("\n🚨 ISSUES:")
        for issue in result["issues"]:
            print(f"   • {issue}")

    if result["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in result["warnings"]:
            print(f"   • {warning}")

    print("\n" + "=" * 80)
    print("📝 GENERATED TEXT:")
    print("-" * 80)
    print(result["text"])
    print("-" * 80)
    
    print("\n📄 FORMATTED FOR PATENT:")
    print("=" * 80)
    print(format_for_patent_document(result["text"]))
    print("=" * 80)
