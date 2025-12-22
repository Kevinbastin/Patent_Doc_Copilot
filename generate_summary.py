import re
from typing import Dict

from llm_runtime import llm_generate  # shared cached model runtime


# ============================================================
# SUMMARY GENERATION
# ============================================================
def generate_summary_of_invention(
    abstract: str,
    claims: str = "",
    max_attempts: int = 3
) -> Dict[str, any]:
    """
    Draft "SUMMARY OF THE INVENTION" for an Indian Complete Specification.
    Uses shared llm_generate() from llm_runtime so the model is NOT reloaded here.
    """

    prompt = f"""You are a senior Indian patent attorney drafting "SUMMARY OF THE INVENTION" for an Indian Complete Specification patent.

INVENTION ABSTRACT:
{abstract}

{f"FIRST CLAIM: {claims[:500]}" if claims else ""}

STRICT REQUIREMENTS:
1. Start with: "Thus according to the basic aspect of the present invention, there is provided..."
2. Use "comprising:" to list ONLY high-level structural components
3. "wherein" clauses MUST describe functional relationships, NOT implementation details
4. DO NOT mention:
   - communication protocols (Bluetooth, Wi-Fi, GSM, LoRa)
   - sensor types (infrared, ultrasonic, temperature sensors)
   - algorithms, software, firmware, or control logic
5. Technical effects should be stated generically (e.g., "adaptive control", "dynamic regulation")
6. Length: 300–500 words


NOW WRITE (only text, no heading):

Thus according to the basic aspect of the present invention, there is provided"""

    best_result = None
    best_score = float("inf")

    for attempt in range(max_attempts):
        generated = llm_generate(
            prompt,
            max_new_tokens=1200,
            temperature=0.25 if attempt == 0 else 0.3 + attempt * 0.1,
            top_p=0.85,
            repeat_penalty=1.18,
            stop_strings=["BRIEF DESCRIPTION", "BRIEF DESCRIPTION OF THE DRAWINGS"],
        )

        raw_text = generated.strip()

        cleaned_text = clean_summary(raw_text)
        validation = validate_summary(cleaned_text)

        score = len(validation["issues"]) * 20 + len(validation["warnings"]) * 5

        result = {
            "text": cleaned_text,
            "valid": validation["valid"],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
            "word_count": validation["word_count"],
            "has_comprising": validation["has_comprising"],
            "has_wherein": validation["has_sufficient_wherein"],
            "aspect_count": validation["aspect_count"],
            "attempt": attempt + 1,
            "score": score
        }


        if validation["valid"] and len(validation["warnings"]) <= 1:
            return result

        if score < best_score:
            best_score = score
            best_result = result

    return best_result if best_result else {
        "text": "",
        "valid": False,
        "issues": ["Generation failed"],
        "warnings": [],
        "word_count": 0,
        "has_comprising": False,
        "has_wherein": False,
        "aspect_count": 0,
        "attempt": max_attempts,
        "score": 999
    }


# ============================================================
# CLEAN SUMMARY
# ============================================================
def clean_summary(text: str) -> str:
    text = re.sub(r'^(SUMMARY OF THE INVENTION:?)\s*', '', text, flags=re.I)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove implementation-specific parentheticals
    text = re.sub(
        r'\((?:bluetooth|wi[- ]?fi|gsm|lora|infrared|algorithm)[^)]+\)',
        '',
        text,
        flags=re.I
    )

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()




# ============================================================
# VALIDATION
# ============================================================
def validate_summary(text: str) -> Dict[str, any]:
    issues = []
    warnings = []

    word_count = len(text.split())
    text_lower = text.lower()
    # ❌ Over-limiting implementation details (NOT allowed in Summary)
    overlimiting_terms = [
        "bluetooth", "wi-fi", "wifi", "lora", "lorawan", "gsm",
        "infrared", "ir sensor", "thermometer", "ultrasonic",
        "microcontroller", "arduino", "raspberry pi",
        "algorithm", "machine learning", "neural network",
        "firmware", "software code"
    ]

    found_terms = [t for t in overlimiting_terms if t in text_lower]
    if found_terms:
        issues.append(
            f"Summary includes implementation-specific details {found_terms}. "
            "Move these to Detailed Description."
        )

    if not text.startswith("Thus according to the basic aspect"):
        issues.append("Must start with required opening phrase")

    has_comprising = "comprising" in text_lower
    if not has_comprising:
        issues.append("Missing 'comprising' clause")

    wherein_count = text_lower.count("wherein")
    if wherein_count < 5:
        warnings.append(f"Only {wherein_count} 'wherein' clauses found (5–8 recommended)")
    elif wherein_count > 8:
        warnings.append(f"Too many 'wherein' clauses ({wherein_count}); consider reducing")

    aspect_count = len(re.findall(
        r"It is (?:another|a further|yet another) aspect of the present invention",
        text,
        flags=re.IGNORECASE
    ))
    if aspect_count < 3:
        warnings.append(f"Only {aspect_count} aspect statements found (3–5 recommended)")

    # Enforce length strictly
    if word_count < 300:
        issues.append("Summary too short (minimum 300 words required)")
    elif word_count > 520:
        warnings.append("Summary exceeds recommended maximum length (500 words)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "has_comprising": has_comprising,
        "wherein_count": wherein_count,
        "has_sufficient_wherein": wherein_count >= 5,
        "aspect_count": aspect_count
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================
def summarize_abstract(abstract: str) -> str:
    result = generate_summary_of_invention(abstract)
    if result and result.get("text"):
        return result["text"]
    return (
        "Thus according to the basic aspect of the present invention, "
        f"there is provided {abstract}"
    )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    sample_abstract = (
        "An Internet of Things (IoT) based remote monitoring and alerting system "
        "for human-animal conflict mitigation, comprising a plurality of "
        "field-deployed sensor nodes, a central master node, a cloud server, "
        "and a power management unit."
    )

    print("=" * 80)
    print("TESTING SUMMARY OF THE INVENTION")
    print("=" * 80)
    print(summarize_abstract(sample_abstract))
    print("=" * 80)
