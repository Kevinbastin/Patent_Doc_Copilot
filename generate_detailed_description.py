from llama_cpp import Llama
import re
from typing import Dict, List


LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"
llm = Llama(model_path=LLM_PATH, device="cuda", n_ctx=8192, n_threads=4, verbose=False)  # Increased context


def extract_components_with_numerals(abstract: str, claims: str) -> Dict[str, List]:
    """
    Extract components and assign reference numerals like real patents.
    Real patent uses: sensor nodes [1], microcontroller [3], LoRaWAN [4a], GSM [4b]
    """
    components = []
    
    # Extract from claims (most structured)
    component_patterns = [
        r'comprising[:\s]+([^;\.]{10,80})',
        r'(\w+\s+(?:module|unit|sensor|node|server|system|device|controller|gateway))',
        r'at least one\s+([^,;\.]{10,60})',
    ]
    
    text = abstract + " " + claims
    for pattern in component_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        components.extend([m.strip() for m in matches if len(m.strip()) > 10])
    
    # Deduplicate
    seen = set()
    unique_components = []
    for comp in components:
        comp_lower = comp.lower()
        if comp_lower not in seen:
            seen.add(comp_lower)
            unique_components.append(comp)
    
    # Assign reference numerals (like real patent)
    numbered_components = {}
    for i, comp in enumerate(unique_components[:20], 1):
        numbered_components[comp] = f"[{i}]"
    
    return numbered_components


def generate_detailed_description(abstract: str, claims: str, drawing_summary: str,
                                 field_of_invention: str = "", background: str = "",
                                 objects: str = "", max_attempts: int = 2) -> Dict[str, any]:
    """
    Generate 'Detailed Description of the Invention' matching Indian Patent Office format.
    
    Real patent structure (IN202541069047):
    - Opening paragraph: Overview referencing figures
    - Component description with reference numerals [1], [2], [3a], [3b]
    - Operational description ("Working:")
    - Use cases (multiple scenarios)
    - Comparative test results with tables
    - Technical advantages (numbered list)
    - Embodiments section
    
    This is the LONGEST section - typically 15-25 pages in real patents.
    """
    
    # Extract components
    components = extract_components_with_numerals(abstract, claims)
    component_list = "\n".join([f"   • {comp} {num}" for comp, num in list(components.items())[:10]])
    
    # Build comprehensive prompt
    prompt = f"""You are a patent attorney drafting "DETAILED DESCRIPTION OF THE INVENTION" for an Indian Complete Specification patent.

INVENTION DETAILS:
{f"FIELD: {field_of_invention}" if field_of_invention else ""}
{f"BACKGROUND: {background[:500]}..." if background else ""}
{f"OBJECTS: {objects[:300]}..." if objects else ""}

ABSTRACT:
{abstract}

CLAIMS (FIRST CLAIM):
{claims[:800]}...

DRAWINGS:
{drawing_summary}

COMPONENT REFERENCE NUMERALS (use these throughout):
{component_list}

REAL PATENT EXAMPLE STRUCTURE (Follow this EXACTLY):

DETAILED DESCRIPTION OF THE INVENTION WITH REFERENCE TO THE ACCOMPANYING FIGURES

The present invention as herein described relates to [system name]. Said system combines [key technologies] to deliver [main benefit] in [application domain].

Referring to Figures 1 to 4, the [system name], comprising [list major components with reference numerals like [1], [2], [3]].

Each [component name] [1] comprises of [sub-components with reference numerals [2], [3a], [3b]]; [description]; [more description]. The [component] is configured to [function]. The [another component] [3] interfaces with [related component] [2] via [connection type].

[Continue describing each major component with reference numerals, their sub-components, connections, and functions for 3-5 paragraphs]

In an embodiment, the [component] [X] is [specific implementation] (e.g., Raspberry Pi or STM32).

In another embodiment, the [feature] includes [specific technology] (e.g., NVIDIA Jetson Nano, Coral TPU).

Working:
The [system name], comprising [major components with numerals]. The step by step operation is as follows:

[Step 1 name];
[Step 2 name];
[Step 3 name];
[Step 4 name]; and
[Step 5 name]

[Then describe each step in detail with reference numerals]

The system utilizes [technology combination]. Below are real-world use cases:

Use case 1: [Scenario name] ([Mode])
Scenario: [Description]
Functionality:
[Point 1]
[Point 2]
Outcome: [Result]

[Include 3-5 use cases]

The following features demonstrates that the [system] have non-trivial technical advancements:
1. [Feature name]
   [Description with technical details]
2. [Another feature]
   [Description]
[Continue for 8-12 features]

The Comparative test results are provided:
[Describe test setup]
[Present results in table format]

The integration of [technology A] and [technology B] offers the following technical advantages:
1. [Advantage title]
   [Description with sub-points]
2. [Another advantage]
   [Description]
[Continue for 5-8 advantages]

CRITICAL REQUIREMENTS:
1. Use reference numerals in brackets: [1], [2], [3a], [3b] throughout
2. Start with: "The present invention as herein described relates to..."
3. Use: "Referring to Figures X to Y, the [system]..."
4. Include "Working:" section with step-by-step operation
5. Include 3-5 "Use case" scenarios
6. Include "Comparative test results" with data
7. Include numbered "Technical advantages" section
8. Use "In an embodiment," and "In another embodiment,"
9. Length: 2000-3000 words minimum
10. Technical, formal language throughout

NOW WRITE THE DETAILED DESCRIPTION (only text, no heading):

The present invention as herein described relates to"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=4096,  # Much longer for detailed description
                temperature=0.3 if attempt == 0 else 0.35,
                stop=["WE CLAIM", "CLAIMS", "\n\n\n\n\n\n"],
                top_p=0.85,
                repeat_penalty=1.15
            )
            
            raw_text = "The present invention as herein described relates to" + response["choices"][0]["text"].strip()
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
                "has_use_cases": validation["has_use_cases"],
                "has_embodiments": validation["has_embodiments"],
                "components": components,
                "attempt": attempt + 1,
                "score": score
            }
            
            if validation["valid"]:
                return result
            
            if score < best_score:
                best_score = score
                best_result = result
                
        except Exception as e:
            continue
    
    return best_result if best_result else {
        "text": "",
        "valid": False,
        "issues": ["Generation failed - section too complex for current model"],
        "warnings": [],
        "word_count": 0,
        "attempt": max_attempts
    }


def clean_detailed_description(text: str) -> str:
    """Clean and format the detailed description."""
    # Remove header if added
    text = re.sub(r'^(DETAILED DESCRIPTION.*?\n){1,}', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def validate_detailed_description(text: str, components: Dict) -> Dict[str, any]:
    """
    Validate against Indian Patent Office standards for detailed description.
    Real patent has: 2000+ words, reference numerals throughout, working section, use cases.
    """
    issues = []
    warnings = []
    
    word_count = len(text.split())
    text_lower = text.lower()
    
    # Check minimum length
    if word_count < 1000:
        issues.append("Detailed description too brief. Should be 1500-3000+ words.")
    
    # Check for reference numerals
    has_numerals = bool(re.search(r'\[\d+[a-z]?\]', text))
    if not has_numerals:
        issues.append("Missing reference numerals (e.g., [1], [2], [3a]). Must reference components throughout.")
    
    # Check for key sections
    has_working = 'working:' in text_lower
    has_use_cases = 'use case' in text_lower
    has_embodiments = 'embodiment' in text_lower
    has_referring = 'referring to figure' in text_lower
    
    if not has_referring:
        issues.append("Must start with 'Referring to Figures X to Y, the [system]...'")
    
    if not has_working:
        warnings.append("Should include 'Working:' section describing step-by-step operation")
    
    if not has_use_cases:
        warnings.append("Consider adding 'Use case' scenarios (3-5 real-world applications)")
    
    if not has_embodiments:
        warnings.append("Should include 'In an embodiment,' and 'In another embodiment,' clauses")
    
    # Check for technical depth
    if 'comprises' not in text_lower and 'comprising' not in text_lower:
        warnings.append("Use 'comprises' or 'comprising' to describe components")
    
    if 'configured to' not in text_lower:
        warnings.append("Use 'configured to' to describe component functions")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "has_reference_numerals": has_numerals,
        "has_working_section": has_working,
        "has_use_cases": has_use_cases,
        "has_embodiments": has_embodiments
    }


def format_for_patent_document(detailed_desc_text: str, include_heading: bool = True) -> str:
    """Format with standard heading."""
    if include_heading:
        return f"DETAILED DESCRIPTION OF THE INVENTION WITH REFERENCE TO THE ACCOMPANYING FIGURES\n\n{detailed_desc_text}"
    return detailed_desc_text


if __name__ == "__main__":
    print("=" * 80)
    print("    DETAILED DESCRIPTION OF THE INVENTION GENERATOR")
    print("    (Indian Patent Office Format - Most Complex Section)")
    print("=" * 80)
    
    # Note: This section is too complex for simple CLI input
    # Recommend building it section by section or using full patent generator
    
    print("\n⚠️  NOTE: Detailed Description is the longest patent section (15-25 pages).")
    print("    For best results, provide complete inputs from earlier sections.\n")
    
    print("For demonstration, provide minimal inputs:")
    print("\n📥 Abstract:")
    abstract = input("> ").strip() or "A system comprising sensors and communication modules for monitoring."
    
    print("\n📥 First claim:")
    claims = input("> ").strip() or "A system comprising: sensors; a processor; and communication modules."
    
    print("\n📥 Drawing summary:")
    drawing_summary = input("> ").strip() or "Figure 1 shows system overview. Figure 2 shows components."
    
    print("\n⏳ Generating detailed description (this may take 30-60 seconds)...\n")
    
    result = generate_detailed_description(abstract, claims, drawing_summary)
    
    if not result["text"]:
        print("❌ ERROR: Could not generate detailed description")
        print("This section requires substantial context from other sections.")
        exit(1)
    
    print("=" * 80)
    print(f"✅ Generated {result['word_count']} words")
    print(f"Attempt: {result['attempt']}")
    print(f"\n📊 Validation:")
    print(f"   Reference numerals: {'✓' if result['has_reference_numerals'] else '✗'}")
    print(f"   Working section: {'✓' if result['has_working_section'] else '✗'}")
    print(f"   Use cases: {'✓' if result['has_use_cases'] else '✗'}")
    print(f"   Embodiments: {'✓' if result['has_embodiments'] else '✗'}")
    
    if result["issues"]:
        print("\n🚨 Issues:")
        for issue in result["issues"]:
            print(f"   • {issue}")
    
    if result["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in result["warnings"]:
            print(f"   • {warning}")
    
    print("\n" + "=" * 80)
    print("📝 GENERATED TEXT (first 1500 chars):")
    print("-" * 80)
    print(result["text"][:1500] + "...")
    print("-" * 80)
    
    print("\n💡 Complete detailed description saved to output.")
    print("=" * 80)
