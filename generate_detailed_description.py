import re
from typing import Dict, List
from llm_runtime import llm_generate


# ============================================================
# COMPONENT EXTRACTION WITH REFERENCE NUMERALS
# ============================================================
def assign_reference_numerals(components: List[str], start: int = 101) -> Dict[str, str]:
        """
        Assign one UNIQUE numeral per component.
        Guaranteed one-to-one mapping.
        """
        registry = {}
        current = start

        for comp in components:
            if comp not in registry:
                registry[comp] = f"({current})"
                current += 1

        return registry
def extract_components_with_numerals(abstract: str, claims: str) -> Dict[str, str]:
    """
    Extract components and assign reference numerals like real patents.
    """
    components = []

    component_patterns = [
        r'(\w+(?:\s+\w+){0,3}\s+(?:module|unit|controller|generator|engine|processor|interface|system|device|database|network|memory))',
        r'plurality of\s+([\w\s]{5,50})',
        r'at least one\s+([\w\s]{5,50})',
    ]




    blacklist = [
        "present invention",
        "example structure",
        "real patent",
        "system name",
        "application domain",
        "invention relates",
        "according to",
    ]

    text = abstract + " " + claims

    for pattern in component_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            m_str = m if isinstance(m, str) else " ".join([x for x in m if x])
            m_str = m_str.strip()
            m_str = re.sub(r'\s+', ' ', m_str).strip()
            m_str = m_str.rstrip('.,;')
            comp_lower = m_str.lower()


            # 🚫 FILTER NON-COMPONENT TEXT
            if any(bad in comp_lower for bad in blacklist):
                continue

            if len(comp_lower) > 10:
                components.append(m_str)

    # Remove duplicates while preserving order
    seen = set()
    unique_components = []
    for comp in components:
        comp_lower = comp.lower()
        if comp_lower not in seen:
            seen.add(comp_lower)
            unique_components.append(comp)

    # Assign reference numerals
    numbered_components = {}
    components = unique_components[:20]
    numbered_components = assign_reference_numerals(components)
    return numbered_components



# ============================================================
# CLEANING & VALIDATION
# ============================================================
def clean_detailed_description(text: str) -> str:
    """Clean and format the detailed description."""
    text = re.sub(
        r'^(DETAILED DESCRIPTION.*?\n){1,}',
        '',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def validate_detailed_description(text: str, components: Dict) -> Dict[str, any]:
    """Validate against Indian Patent Office standards for detailed description."""
    issues = []
    warnings = []

    word_count = len(text.split())
    text_lower = text.lower()
    unused = [
        num for comp, num in components.items()
        if num not in text
    ]

    if unused:
        warnings.append(
            f"Some assigned reference numerals are not explicitly referenced in the description: {', '.join(unused[:5])}."
        )

    if word_count < 800:
        issues.append("Detailed description should be at least 800 words.")
    elif word_count < 1200:
        warnings.append("Description is acceptable but could be more detailed.")




    has_numerals = bool(re.search(r'\(\d+[a-z]?\)', text))
    if not has_numerals:
        issues.append("Missing reference numerals (e.g., (101), (102), (103a)). Must reference components.")

    has_working = 'working:' in text_lower or 'operation:' in text_lower

    has_embodiments = 'embodiment' in text_lower

    has_referring = 'referring to figure' in text_lower or 'referring to figures' in text_lower

    if not has_referring:
        warnings.append(
            "Consider including 'Referring to the accompanying figures...' for completeness."
        )
    if not has_working:
        warnings.append(
            "An operational description may be included under 'Working' or 'Operation', depending on the nature of the invention."
        )


    if not has_embodiments:
        warnings.append("Should include 'In an embodiment,' and 'In another embodiment,' clauses")

    if 'comprises' not in text_lower and 'comprising' not in text_lower:
        warnings.append("Use 'comprises' or 'comprising' to describe components")

    if 'configured to' not in text_lower:
        warnings.append("Use 'configured to' to describe component functions")
    def detect_reference_conflicts(text: str) -> Dict[str, List[str]]:
        """
        Detect same reference numeral used for multiple components.
        """
        pattern = re.compile(r'([A-Za-z][A-Za-z\s\-]{3,50})\s*\((\d+[a-z]?)\)')
        matches = pattern.findall(text)

        usage = {}
        for comp, num in matches:
            comp = comp.strip().lower()
            usage.setdefault(num, set()).add(comp)

        conflicts = {
            num: list(comps)
            for num, comps in usage.items()
            if len(comps) > 1
        }

        return conflicts
    conflicts = detect_reference_conflicts(text)

    if conflicts:
        for num, comps in list(conflicts.items())[:5]:
            issues.append(
                f"Reference numeral ({num}) is used for multiple components: {', '.join(comps)}. "
                "Each component must have a unique numeral."
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "has_reference_numerals": has_numerals,
        "has_working_section": has_working,
        "has_embodiments": has_embodiments
    }


# ============================================================
# DETAILED DESCRIPTION GENERATION
# ============================================================
def generate_detailed_description(
    abstract: str,
    claims: str,
    drawing_summary: str,
    field_of_invention: str = "",
    background: str = "",
    objects: str = "",
    max_attempts: int = 2
) -> Dict[str, any]:
    """
    Generate 'Detailed Description of the Invention' matching Indian Patent Office format.
    """
    components = extract_components_with_numerals(abstract, claims)
    component_list = "\n".join([f"   • {comp} {num}" for comp, num in list(components.items())[:12]])

    prompt = f"""You are a senior Indian patent attorney drafting the
"DETAILED DESCRIPTION OF THE INVENTION" for an Indian Complete Specification.

INVENTION CONTEXT (for reference only):
{f"FIELD OF INVENTION: {field_of_invention[:200]}" if field_of_invention else ""}
{f"BACKGROUND OF THE INVENTION: {background[:400]}..." if background else ""}
{f"OBJECTS OF THE INVENTION: {objects[:250]}..." if objects else ""}

ABSTRACT:
{abstract[:1000]}

CLAIMS (FIRST CLAIM):
{claims[:800]}

DRAWINGS SUMMARY:
{drawing_summary[:400]}

COMPONENT REFERENCE NUMERALS
(Use these reference numerals consistently throughout the description):
{component_list}

DRAFTING STYLE AND STRUCTURE GUIDANCE:

DETAILED DESCRIPTION OF THE INVENTION WITH REFERENCE TO THE ACCOMPANYING FIGURES

The present invention as herein described relates to an invention as defined in the abstract and claims and is configured to achieve the intended technical objectives.

Referring to the accompanying figures, the invention comprises one or more components, modules, units, or functional elements, each identified by a corresponding reference numeral, and arranged to cooperatively perform the disclosed functions.

Each component comprises one or more sub-components and is configured to perform a defined operational function. The components interact with one another through appropriate physical, electrical, logical, or data communication interfaces to enable coordinated operation of the invention.

In an embodiment, a component employs a specific structural arrangement, algorithm, control logic, or functional configuration to achieve an intended operation in accordance with the invention.

In another embodiment, the invention includes alternative configurations, additional features, or modified control strategies to enhance performance, reliability, scalability, adaptability, or operational efficiency under varying conditions.

OPERATION / WORKING:
The invention operates as follows. The operational flow may include one or more of the following stages, depending on the nature of the invention:
• initialization or activation of components;
• execution of processing, control, or transformation functions;
• interaction or exchange of data, signals, energy, or materials between components;
• monitoring, analysis, or decision-making based on predefined parameters; and
• generation of an output, response, or completed result.

The invention may be implemented using an appropriate combination of hardware, software, firmware, computational logic, or electromechanical elements, depending on the application domain.

Further embodiments describe variations in implementation, deployment environments, operating conditions, or integration with external systems, without departing from the scope of the invention.

STRICT REQUIREMENTS (MANDATORY):
1. Use reference numerals in round brackets (e.g., (101), (102), (103a)) consistently.
2. Start with: "The present invention as herein described relates to..."
3. Use phrasing such as: "Referring to the accompanying figures..."
4. Include an operational description section titled "Working:" or "Operation:" as appropriate.
5. Use "In an embodiment," and "In another embodiment," multiple times.
6. Length: 1000–2000 words.
7. Use formal, technical, third-person patent language.
8. Describe how components interact, connect, or cooperate.
9. Do NOT include claims, advantages lists, summaries, or marketing language.

Write ONLY the detailed description text (do NOT include headings):

The present invention as herein described relates to
"""



    best_result = None
    best_score = float('inf')

    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...", end=" ", flush=True)
            print("(Generating long text, please wait 30-60 seconds...)", flush=True)
            
            # CORRECTED: Removed max_input_tokens parameter
            generated = llm_generate(
                prompt,
                max_new_tokens=3000,
                temperature=0.28 if attempt == 0 else 0.32 + (attempt * 0.05),
                top_p=0.85,
                repeat_penalty=1.15,
                stop_strings=[
                    "WE CLAIM",
                    "CLAIMS:",
                    "CLAIMS",
                    "I claim",
                    "SUMMARY OF THE INVENTION",
                    "BRIEF DESCRIPTION OF THE DRAWINGS",
                    "\n\n\n\n\n\n"
                ]
            )

            if not generated or len(generated.strip()) < 300:
                print("Too short, retrying...")
                continue

            if not generated.strip().lower().startswith("the present invention as herein described relates to"):
                raw_text = "The present invention as herein described relates to " + generated.strip()
            else:
                raw_text = generated.strip()

            cleaned_text = clean_detailed_description(raw_text)
            validation = validate_detailed_description(cleaned_text, components)

            score = len(validation["issues"]) * 20 + len(validation["warnings"]) * 5

            result = {
                "text": cleaned_text,
                "valid": validation["valid"],
                "issues": validation["issues"],
                "warnings": validation["warnings"],
                "word_count": validation["word_count"],
                "has_reference_numerals": validation["has_reference_numerals"],
                "has_working_section": validation["has_working_section"],
                "has_embodiments": validation["has_embodiments"],
                "components": components,
                "attempt": attempt + 1,
                "score": score
            }

            print(f"   Score: {score}, Words: {validation['word_count']}")

            if validation["valid"] and len(validation["warnings"]) <= 2:
                print("   ✅ Good quality!")
                return result

            if score < best_score:
                best_score = score
                best_result = result

        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return best_result if best_result else {
        "text": "",
        "valid": False,
        "issues": ["Generation failed - section requires extended context"],
        "warnings": [],
        "word_count": 0,
        "attempt": max_attempts,
        "score": 999,
        "components": components
    }


# ============================================================
# FORMATTING
# ============================================================
def format_for_patent_document(detailed_desc_text: str, include_heading: bool = True) -> str:
    """Format with standard heading."""
    if include_heading:
        return (
            "DETAILED DESCRIPTION OF THE INVENTION WITH REFERENCE TO THE ACCOMPANYING FIGURES\n\n"
            + detailed_desc_text
        )
    return detailed_desc_text


def print_formatted_report(result: Dict):
    """Print professional validation report."""
    print("\n" + "=" * 85)
    print("        DETAILED DESCRIPTION OF THE INVENTION - VALIDATION REPORT")
    print("=" * 85)

    if result["valid"] and len(result["warnings"]) <= 1:
        print("\n✅ STATUS: EXCELLENT - Meets Indian Patent Office standards")
    elif result["valid"]:
        print("\n✅ STATUS: VALID - Minor improvements recommended")
    else:
        print("\n❌ STATUS: NEEDS REVISION - Critical issues found")

    print("\n" + "-" * 85)
    print("📊 METRICS:")
    print(f"   Word Count:         {result['word_count']} words (optimal: 1500-3000)")
    print(f"   Generation Attempt: {result['attempt']}")
    print(f"   Quality Score:      {result['score']} (lower is better)")

    print("\n" + "-" * 85)
    print("📋 CONTENT VERIFICATION:")
    print(f"   Reference Numerals:  {'✓' if result['has_reference_numerals'] else '✗'}")
    print(f"   Working Section:     {'✓' if result['has_working_section'] else '✗'}")
    print(f"   Embodiments:         {'✓' if result['has_embodiments'] else '✗'}")

    if result.get('components'):
        print("\n" + "-" * 85)
        print(f"🔍 COMPONENT REFERENCE NUMERALS ASSIGNED: {len(result['components'])}")
        for comp, num in list(result['components'].items())[:8]:
            print(f"   {num} {comp[:60]}")

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
    print("📝 GENERATED TEXT (showing first 2000 characters):")
    print("-" * 85)
    preview_text = result["text"][:2000]
    print(preview_text)
    if len(result["text"]) > 2000:
        print("\n... (text continues, total " + str(len(result["text"])) + " characters)")
    print("-" * 85)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("=" * 85)
    print("    DETAILED DESCRIPTION OF THE INVENTION GENERATOR")
    print("    (Indian Patent Office Format - Most Complex Section)")
    print("=" * 85)

    print("\n⚠️  NOTE: Detailed Description is the longest patent section (typically 10-25 pages).")
    print("    For best results, provide complete inputs from earlier sections.\n")

    print("📥 Enter invention abstract (press Enter twice to finish):")
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
        print("Using demo abstract...")
        abstract = "A system comprising sensors and communication modules for monitoring."

    print("\n📥 Enter first claim (or press Enter to skip):")
    claims = input("> ").strip()
    if not claims:
        claims = "A system comprising: sensors; a processor; and communication modules."

    print("\n📥 Enter drawing summary (or press Enter to skip):")
    drawing_summary = input("> ").strip()
    if not drawing_summary:
        drawing_summary = "Figure 1 shows system overview. Figure 2 shows components. Figure 3 shows data flow."

    print("\n⏳ Generating detailed description...")
    print("   (This will take 30-60 seconds for long text generation...)\n")
    
    result = generate_detailed_description(
        abstract=abstract,
        claims=claims,
        drawing_summary=drawing_summary,
        max_attempts=2
    )

    if not result["text"]:
        print("\n❌ ERROR: Could not generate detailed description")
        for issue in result.get("issues", []):
            print(f"   • {issue}")
        exit(1)

    print_formatted_report(result)

    print("\n" + "=" * 85)
    print("📄 FORMATTED FOR PATENT DOCUMENT:")
    print("=" * 85)
    print(format_for_patent_document(result["text"], include_heading=True)[:3000])
    if len(result["text"]) > 3000:
        print("\n... (showing first 3000 chars, full text is " + str(len(result["text"])) + " chars)")

    print("\n" + "=" * 85)
    print("💡 TIPS FOR PERFECT DETAILED DESCRIPTION:")
    print("=" * 85)
    print("1. Use reference numerals (101), (102), (103a) consistently")
    print("2. Start with 'Referring to Figures X to Y...'")
    print("3. Include 'Working:' section with step-by-step operation")
    print("4. Add 3-5 use case scenarios")
    print("5. Use 'In an embodiment' and 'In another embodiment' clauses")
    print("6. Describe component interactions and connections")
    print("7. Aim for 1500-3000 words for comprehensive coverage")
    print("=" * 85)
    print("\n✅ Generation complete!")
    print("=" * 85)
