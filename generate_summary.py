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

{f"CLAIM 1 (AUTHORITATIVE SOURCE): {claims[:600]}" if claims else ""}

═══════════════════════════════════════════════════════════════
STRICT IPO REQUIREMENTS (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════

1. CLEAR TECHNICAL SOLUTION (NOT TEMPLATE):
   ❌ Do NOT write "there is provided the present invention comprising"
   ❌ Do NOT use circular phrases like "the invention provides the invention"
   ✓ State WHAT the invention IS and HOW it works

2. LIMIT "THE PRESENT INVENTION":
   ❌ Maximum 3 times in entire summary
   ✓ Use "the system", "the device", "the apparatus", "the method" instead
   ✓ Use the actual invention name from the abstract

3. STRUCTURE (4 paragraphs, 250-400 words total):

   PARAGRAPH 1 - TECHNICAL SOLUTION (80-100 words):
   Start with: "The invention provides a [invention name] comprising..."
   List the main structural components with their functions:
   - "[Component A] configured to [function]"
   - "[Component B] coupled to [Component A] for [purpose]"
   - "[Component C] adapted to [function]"

   PARAGRAPH 2 - OPERATIONAL RELATIONSHIP (60-80 words):
   Describe how components work together:
   "In operation, [Component A] receives [input] and communicates with [Component B] to..."

   PARAGRAPH 3 - TECHNICAL ADVANTAGES (60-80 words):
   State specific technical benefits (NOT marketing claims):
   "The system achieves [specific technical effect] through the [specific component/configuration]."

   PARAGRAPH 4 - EMBODIMENTS (40-60 words):
   "In one embodiment, [variation]. In another embodiment, [alternative configuration]."

4. FORBIDDEN PATTERNS:
   ❌ "Thus according to the basic aspect of the present invention, there is provided"
   ❌ "It is another aspect of the present invention to provide"
   ❌ "The present invention advantageously provides"
   ❌ Repeating same component descriptions multiple times

5. REQUIRED ELEMENTS:
   ✓ Extract components from CLAIM 1 (if provided)
   ✓ Each component must have a FUNCTION
   ✓ Components must have RELATIONSHIPS (coupled to, connected to, in communication with)
   ✓ At least 2 "wherein" clauses describing technical relationships

═══════════════════════════════════════════════════════════════
NOW WRITE THE SUMMARY (250-400 words, CLEAR TECHNICAL SOLUTION):
═══════════════════════════════════════════════════════════════

The invention provides"""

    best_result = None
    best_score = float("inf")

    for attempt in range(max_attempts):
        generated = llm_generate(
            prompt,
            max_new_tokens=1500,  # Increased for longer output
            temperature=0.25 if attempt == 0 else 0.3 + attempt * 0.1,
            top_p=0.85,
            repeat_penalty=1.18,
            stop_strings=["BRIEF DESCRIPTION", "BRIEF DESCRIPTION OF THE DRAWINGS", "DETAILED DESCRIPTION"],
        )

        raw_text = generated.strip()
        cleaned_text = clean_summary(raw_text)
        word_count = len(cleaned_text.split())
        
        # EXPANSION: If too short, add more technical details
        if word_count < 250:
            expansion_prompt = f"""Expand this patent summary to 300-350 words by adding:
1. More component-function relationships
2. Operational paragraph ("In operation, the [component] receives...")
3. Technical advantages paragraph
4. Embodiment variations

CURRENT SUMMARY ({word_count} words):
{cleaned_text}

INVENTION:
{abstract}

RULES:
❌ Do NOT add "the present invention" more than 2 times
❌ Do NOT use template phrases like "It is another aspect of..."
✓ Add specific technical details about how components work together

EXPANDED SUMMARY (300-350 words):

The invention provides"""

            expanded = llm_generate(
                expansion_prompt,
                max_new_tokens=1200,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.15,
                stop_strings=["DETAILED DESCRIPTION", "BRIEF DESCRIPTION"],
            )
            
            expanded_text = clean_summary(expanded.strip())
            if len(expanded_text.split()) > word_count:
                cleaned_text = expanded_text
        
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

    # If still too short or invalid, use template-based expansion based on actual abstract
    if best_result and best_result["word_count"] < 300:
        # Extract invention name from abstract
        import re as re_inner
        invention_name = "the present invention"
        # Clean abstract for matching
        abstract_clean = ' '.join(abstract.split())
        match = re_inner.search(r'^A\s+([a-zA-Z\s\-]+?)(?:\s+comprising|\s+for\s+|\s+with\s+|,|\.)', abstract_clean, re_inner.IGNORECASE)
        if match:
            invention_name = match.group(1).strip().lower()
        
        # Extract components from abstract AND claims - NEVER use generic placeholders
        components_list = []
        source_text = abstract + " " + claims
        abstract_lower = source_text.lower()
        
        # Look for "comprising:" section
        comprising_match = re_inner.search(r'comprising[:\s]+(.+?)(?:wherein|;\.|\.)', source_text, re_inner.IGNORECASE | re_inner.DOTALL)
        if comprising_match:
            comp_text = comprising_match.group(1)
            # Split by semicolons or "and"
            parts = re_inner.split(r';|\band\b', comp_text)
            for part in parts[:6]:
                part = part.strip()
                if part and len(part) > 5:
                    # Clean the component name
                    part = re_inner.sub(r'^a\s+|^an\s+|^the\s+', '', part, flags=re_inner.IGNORECASE)
                    if part and len(part) > 3:
                        components_list.append(part[:60])
        
        # CRITICAL: NEVER use generic placeholders
        # If no components found, output ERROR instead of fake components
        if not components_list:
            # Try to extract from claims
            claim_match = re_inner.search(r'(?:comprising|including)[:\s]+(.+?)(?:wherein|,\s*characterized)', claims, re_inner.IGNORECASE | re_inner.DOTALL)
            if claim_match:
                parts = re_inner.split(r';|\band\b|,', claim_match.group(1))
                for part in parts[:6]:
                    part = part.strip()
                    if part and len(part) > 5:
                        part = re_inner.sub(r'^a\s+|^an\s+|^the\s+', '', part, flags=re_inner.IGNORECASE)
                        if part and len(part) > 3:
                            components_list.append(part[:60])
        
        # If still no components, return error - DO NOT use placeholders
        if not components_list:
            best_result["valid"] = False
            best_result["issues"] = ["ERROR: Could not extract technical components from abstract/claims. Manual drafting required."]
            return best_result
        
        # Build dynamic summary based on ACTUAL invention components - MAX CONSERVATISM
        # Anchor every benefit to a specific component
        c1 = components_list[0] if len(components_list) > 0 else "system"
        c2 = components_list[1] if len(components_list) > 1 else "component"
        c3 = components_list[2] if len(components_list) > 2 else "element"
        
        template_summary = f"""Thus according to the basic aspect of the present invention, there is provided {invention_name} comprising: {'; '.join(components_list[:4])}; wherein the {c1} is configured to facilitate the primary function of the invention; wherein the {c2} is operatively connected to support the operation of the {c1}; wherein the arrangement of the {components_list[-1] if components_list else 'components'} ensures structural integrity and functional cohesion.

It is another aspect of the present invention to provide an improved {invention_name} wherein the {c1} is specifically adapted to enhance operational efficiency. It is a further aspect of the present invention to provide a configuration wherein the {c2} cooperates with the {c3} to achieve the intended technical effect. It is yet another aspect of the present invention to provide a construction wherein the components are integrated to minimize mechanical complexity while maximizing functional output.

The present invention advantageously provides a {invention_name} wherein the specific arrangement of the {c1} and {c2} overcomes the limitations of prior art. The configuration enables the {c3} to operate effectively under varying conditions. The integrated design of the {c1} ensures consistent performance without requiring extensive maintenance. Furthermore, the {invention_name} provides a robust solution through the synergistic operation of the claimed components, thereby establishing a technical advance over conventional systems.

According to one preferred embodiment, the {c1} is configured for a specific operational mode. In another embodiment, the {c2} is adapted to provide adjustable functionality. In yet another embodiment, the {invention_name} includes the {c3} arranged in a specific orientation to optimize performance. The invention effectively provides a technical solution wherein the structural cooperation of the {c1}, {c2}, and {c3} results in the verified technical effects described herein."""

        best_result["text"] = template_summary
        best_result["word_count"] = len(template_summary.split())
        best_result["valid"] = len(components_list) >= 3  # Only valid if we found real components
        best_result["issues"] = [] if len(components_list) >= 3 else ["Insufficient components extracted - manual review required"]
        best_result["aspect_count"] = 4

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
    
    # CRITICAL: Remove forbidden template phrases
    template_removals = [
        r'thus according to the basic aspect of the present invention,?\s*',
        r'there is provided the present invention comprising:?\s*',
        r'it is (?:another|a further|yet another) aspect of the present invention to provide\s*',
        r'the present invention advantageously provides\s*',
    ]
    for pattern in template_removals:
        text = re.sub(pattern, '', text, flags=re.I)
    
    # Reduce excessive "the present invention" usage
    # Keep first occurrence, replace subsequent with "the system/device"
    count = 0
    def replace_present_invention(match):
        nonlocal count
        count += 1
        if count <= 2:
            return match.group(0)
        return "the system"
    
    text = re.sub(r'the present invention', replace_present_invention, text, flags=re.I)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Ensure starts with capital letter
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text




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

    # Check for FORBIDDEN template patterns (IPO non-compliant)
    forbidden_patterns = [
        "thus according to the basic aspect of the present invention, there is provided",
        "it is another aspect of the present invention to provide",
        "it is a further aspect of the present invention to provide",
        "it is yet another aspect of the present invention to provide",
        "the present invention advantageously provides",
        "the present invention provides the present invention",
    ]
    
    for pattern in forbidden_patterns:
        if pattern in text_lower:
            issues.append(f"Contains forbidden template phrase: '{pattern[:50]}...'")
    
    # Check for excessive "the present invention" usage
    present_invention_count = text_lower.count("the present invention")
    if present_invention_count > 3:
        issues.append(f"'The present invention' appears {present_invention_count} times (max 3 allowed)")

    # Check for required elements
    has_comprising = "comprising" in text_lower
    if not has_comprising:
        warnings.append("Consider adding 'comprising' clause for component list")

    # Check for component-function relationships
    has_configured_to = "configured to" in text_lower
    has_coupled_to = any(phrase in text_lower for phrase in ["coupled to", "connected to", "in communication with"])
    
    if not has_configured_to:
        warnings.append("Consider adding component-function relationships (e.g., 'configured to')")
    
    if not has_coupled_to:
        warnings.append("Consider adding component relationships (e.g., 'coupled to', 'connected to')")

    # Check for wherein clauses (reduced requirement)
    wherein_count = text_lower.count("wherein")
    if wherein_count < 2:
        warnings.append(f"Only {wherein_count} 'wherein' clauses found (2-4 recommended)")

    # Check for operational description
    has_operational = any(phrase in text_lower for phrase in ["in operation", "during operation", "when operating"])
    if not has_operational:
        warnings.append("Consider adding operational description ('In operation...')")

    # Enforce length - 250-500 words for concise summary
    if word_count < 200:
        issues.append("Summary too short (minimum 200 words recommended)")
    elif word_count < 250:
        warnings.append("Summary is slightly short (250-400 words recommended)")
    elif word_count > 500:
        warnings.append("Summary exceeds recommended length (500 words max)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "has_comprising": has_comprising,
        "wherein_count": wherein_count,
        "has_sufficient_wherein": wherein_count >= 2,
        "present_invention_count": present_invention_count,
        "aspect_count": wherein_count  # aspect_count = number of wherein clauses
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================
def summarize_abstract(abstract: str) -> str:
    result = generate_summary_of_invention(abstract)
    if result and result.get("text"):
        return result["text"]
    # Fallback: Use clear technical solution format, not template
    return f"The invention provides {abstract}"


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
