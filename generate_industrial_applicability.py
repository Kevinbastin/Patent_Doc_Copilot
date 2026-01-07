import re
from typing import Dict
from llm_runtime import llm_generate  # shared cached model runtime


# ============================================================
# INDUSTRIAL APPLICABILITY GENERATION
# ============================================================
def generate_industrial_applicability(
    abstract: str,
    field_of_invention: str = "",
    max_attempts: int = 2
) -> Dict[str, any]:
    """
    Draft 'INDUSTRIAL APPLICABILITY' for an Indian Complete Specification.
    Complies with Section 2(1)(ac) of the Indian Patent Act.
    """

    prompt = f"""You are a senior Indian patent attorney drafting the
"INDUSTRIAL APPLICABILITY" section for an Indian Complete Specification.

INVENTION ABSTRACT:
{abstract[:600]}

{f"FIELD OF INVENTION: {field_of_invention}" if field_of_invention else ""}

STRICT REQUIREMENTS (MANDATORY):
1. Write ONLY one paragraph.
2. Start EXACTLY with the sentence:
   "The present invention is capable of industrial application and can be made or used in industry in accordance with the provisions of the Indian Patent Act."
3. Clearly indicate that the invention can be manufactured or used in industry.
4. DOMAIN CONSTRAINT: Industrial applicability MUST be limited to the SAME field as the invention.
   - If the abstract describes a "glucose monitoring ring", mention healthcare/medical devices.
   - If the abstract describes a "hydraulic valve", mention industrial machinery.
   - Do NOT mention unrelated industries (e.g., water treatment for a glucose ring).
5. If a FIELD OF INVENTION is provided, use ONLY that field.
6. Do NOT mention:
   - advantages
   - performance improvements
   - results or benefits
   - comparison with prior art
   - algorithms, software, AI, or machine learning
   - unrelated industries or applications
7. Use formal patent drafting language only.
8. Length: 80–150 words.

NOW WRITE (only text, no heading):

The present invention is capable of industrial application"""

    best_result = None
    best_score = float("inf")

    for attempt in range(max_attempts):
        generated = llm_generate(
            prompt=prompt,
            max_new_tokens=320,
            temperature=0.2,
            top_p=0.85,
            repeat_penalty=1.15,
            stop_strings=[
                "CLAIMS",
                "SUMMARY",
                "DETAILED DESCRIPTION",
                "WE CLAIM"
            ],
        )

        raw_text = generated.strip()
        cleaned_text = clean_industrial_applicability(raw_text)
        validation = validate_industrial_applicability(cleaned_text)

        score = len(validation["issues"]) * 20 + len(validation["warnings"]) * 5

        result = {
            "text": cleaned_text,
            "valid": validation["valid"],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
            "word_count": validation["word_count"],
            "attempt": attempt + 1,
            "score": score
        }

        if validation["valid"]:
            return result

        if score < best_score:
            best_score = score
            best_result = result

    # Pass field to fallback for domain-specific text
    fallback_text = fallback_industrial_applicability(field_of_invention)
    return best_result if best_result else {
        "text": fallback_text,
        "valid": True,
        "issues": [],
        "warnings": ["Fallback text used"],
        "word_count": len(fallback_text.split()),
        "attempt": max_attempts,
        "score": 0
    }


# ============================================================
# CLEANING
# ============================================================
def clean_industrial_applicability(text: str) -> str:
    """Remove artifacts and unsafe language."""
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove prohibited advantage language
    banned_phrases = [
        "improves", "reduces", "enhances", "optimizes",
        "efficient", "cost-effective", "high performance",
        "better", "faster", "superior"
    ]
    for b in banned_phrases:
        text = re.sub(rf'\b{b}\b', '', text, flags=re.IGNORECASE)

    return text


# ============================================================
# VALIDATION
# ============================================================
def validate_industrial_applicability(text: str) -> Dict[str, any]:
    issues = []
    warnings = []

    word_count = len(text.split())
    text_lower = text.lower()

    mandatory_start = (
        "the present invention is capable of industrial application "
        "and can be made or used in industry in accordance with the provisions "
        "of the indian patent act"
    )

    if not text_lower.startswith(mandatory_start):
        issues.append("Must start with mandatory opening sentence")

    if "industry" not in text_lower:
        issues.append("Must explicitly mention industry")

    if word_count < 80:
        issues.append("Industrial applicability is too short")
    elif word_count > 150:
        warnings.append("Longer than recommended (80–150 words ideal)")

    banned_terms = [
        "algorithm", "machine learning", "artificial intelligence",
        "advantage", "benefit", "comparison", "prior art"
    ]
    found = [b for b in banned_terms if b in text_lower]
    if found:
        issues.append(f"Contains disallowed terms: {found}")
    if text.count('.') > 3:
        warnings.append("Paragraph may be fragmented; ensure single-paragraph structure")


    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count
    }


# ============================================================
# FALLBACK (EXAMINER-SAFE)
# ============================================================
def fallback_industrial_applicability(field: str = "") -> str:
    """Generate domain-specific fallback instead of generic text."""
    if field:
        return (
            f"The present invention is capable of industrial application and can be "
            f"made or used in industry in accordance with the provisions of the Indian "
            f"Patent Act. The invention is particularly applicable in the field of {field}, "
            f"where it may be manufactured, implemented, or utilized by those skilled in the art. "
            f"The invention is suitable for adoption in industries related to {field} and allied sectors."
        )
    else:
        return (
            "The present invention is capable of industrial application and can be "
            "made or used in industry in accordance with the provisions of the Indian "
            "Patent Act. The invention may be manufactured, implemented, or utilized "
            "in industrial environments relevant to the disclosed technical field, "
            "and is suitable for adoption by those skilled in the art."
        )



# ============================================================
# SIMPLE INTERFACE
# ============================================================
def industrial_applicability_from_abstract(
    abstract: str,
    field_of_invention: str = ""
) -> str:
    result = generate_industrial_applicability(
        abstract,
        field_of_invention=field_of_invention
    )

    if result["valid"]:
        return result["text"]

    # Pass field to fallback for domain-specific text
    return fallback_industrial_applicability(field_of_invention)



# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    sample_abstract = (
        "A system for monitoring agricultural parameters comprising sensor units, "
        "a processing unit, and a control module for managing irrigation operations."
    )

    print("=" * 80)
    print("TESTING INDUSTRIAL APPLICABILITY (IPO FORMAT)")
    print("=" * 80)
    print(industrial_applicability_from_abstract(sample_abstract))
    print("=" * 80)
