import re
import json
from typing import Dict, List, Tuple
from enum import Enum
from llm_runtime import llm_generate


# ============================================================
# ENTERPRISE ABSTRACT ANALYSIS FUNCTIONS (Priority 1-3 Fixes)
# ============================================================

class AbstractType(Enum):
    """Classification of patent abstract types for branched generation."""
    APPARATUS = "apparatus"      # Device, system, machine
    METHOD = "method"           # Process, method, steps
    COMPOSITION = "composition"  # Chemical, pharmaceutical, formulation
    PROCESS = "process"         # Manufacturing, fabrication, semiconductor


def classify_abstract_type(abstract: str) -> AbstractType:
    """
    Classify abstract to branch generation logic.
    Determines if invention is Apparatus, Method, Composition, or Process.
    Uses LLM for universal input support - works for ANY invention type.
    """
    prompt = f"""Classify this patent abstract into exactly ONE category:
- APPARATUS: Physical device, system, machine, or hardware
- METHOD: A way of doing something, steps, algorithm
- COMPOSITION: Chemical, pharmaceutical, material composition
- PROCESS: Manufacturing, fabrication, or production process

ABSTRACT:
{abstract[:500]}

Reply with ONLY one word: APPARATUS, METHOD, COMPOSITION, or PROCESS
"""
    
    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.1,
            system_prompt="Classify the patent type. Reply with exactly one word."
        )
        
        if response:
            response_upper = response.strip().upper()
            if "METHOD" in response_upper:
                return AbstractType.METHOD
            elif "COMPOSITION" in response_upper:
                return AbstractType.COMPOSITION
            elif "PROCESS" in response_upper:
                return AbstractType.PROCESS
            else:
                return AbstractType.APPARATUS
    except Exception as e:
        print(f"Abstract classification error: {e}")
    
    # Default to APPARATUS (most common)
    return AbstractType.APPARATUS


def detect_abstract_domain(abstract: str) -> str:
    """
    Priority 1 Fix: Derive application domain from abstract.
    NEVER hardcode domains - always extract from actual abstract.
    Returns a concise domain description for constraining generated content.
    """
    prompt = f"""Identify the technical domain of this patent abstract in 3-5 words.

ABSTRACT:
{abstract[:500]}

Examples of domains:
- "non-invasive glucose monitoring wearables"
- "industrial wastewater treatment systems"
- "semiconductor chip fabrication"
- "pharmaceutical drug delivery"
- "autonomous vehicle navigation"

Return ONLY the domain phrase, nothing else:"""
    
    try:
        domain = llm_generate(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.1,
            system_prompt="Extract technical domain from patent abstract. Be very concise."
        )
        if domain and len(domain.strip()) > 5:
            return domain.strip().lower().replace('"', '')
    except Exception as e:
        print(f"Domain detection failed: {e}")
    
    # Fallback: extract key nouns from abstract
    words = abstract.lower().split()[:20]
    key_words = [w for w in words if len(w) > 5 and w not in ['comprising', 'including', 'wherein', 'according']]
    return " ".join(key_words[:5]) if key_words else "the present invention"


def calculate_adaptive_length(components: Dict[str, str], abstract_type: AbstractType) -> Tuple[int, int]:
    """
    Priority 2 Fix: Calculate adaptive word count based on complexity.
    NOT a fixed 2500 words - scale based on invention complexity.
    
    Returns:
        Tuple of (min_words, max_words)
    """
    num_components = len(components)
    
    # Base length by abstract type
    type_multipliers = {
        AbstractType.APPARATUS: 1.0,      # Standard device
        AbstractType.METHOD: 0.9,         # Methods often shorter
        AbstractType.COMPOSITION: 1.2,    # Chemical needs more detail
        AbstractType.PROCESS: 1.3         # Fabrication needs steps
    }
    
    multiplier = type_multipliers.get(abstract_type, 1.0)
    
    # Calculate based on component count
    if num_components <= 3:
        base_min, base_max = 1200, 1500
    elif num_components <= 6:
        base_min, base_max = 1600, 2000
    elif num_components <= 10:
        base_min, base_max = 2000, 2500
    else:
        base_min, base_max = 2500, 3500
    
    return int(base_min * multiplier), int(base_max * multiplier)


# Flag for fallback mode (Priority 4: Fail loudly)
FALLBACK_WARNING = """
⚠️ WARNING: NON-FILING DRAFT ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This detailed description was generated using fallback templates
because the primary generation failed or produced insufficient content.

DO NOT FILE THIS DOCUMENT as-is. It requires:
1. Manual review by a patent attorney
2. Domain-specific technical verification
3. Claim-description consistency check

This is marked as a DRAFT for reference only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ============================================================
# COMPONENT EXTRACTION WITH REFERENCE NUMERALS
# ============================================================
def assign_reference_numerals(components: List[str], start: int = 10, increment: int = 10) -> Dict[str, str]:
        """
        Assign one UNIQUE numeral per component.
        Guaranteed one-to-one mapping.
        Uses 10-series increments (10, 20, 30...) for IPO format.
        """
        registry = {}
        current = start

        for comp in components:
            if comp not in registry:
                registry[comp] = f"({current})"
                current += increment

        return registry
def extract_components_with_numerals(abstract: str, claims: str) -> Dict[str, str]:
    """
    Extract components using LLM and assign reference numerals like real patents.
    Universal support for all invention types.
    """
    prompt = f"""Extract the core technical components (modules, sensors, chemicals, layers, units) from this abstract and claim.
Return exactly a JSON list of strings (component names only). Limit to top 15 essential components.

ABSTRACT:
{abstract[:500]}

CLAIM:
{claims[:500]}

JSON LIST:"""

    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.1,
            system_prompt="You are a patent component extractor. Output ONLY a valid JSON list of strings."
        )
        
        json_text = response.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].strip()
            
        components = json.loads(json_text)
        if not isinstance(components, list):
             components = []
    except Exception as e:
        print(f"LLM component extraction failed: {e}")
        components = ["system", "processor", "memory", "input interface", "output interface"]

    # Assign reference numerals
    numbered_components = assign_reference_numerals(components)
    return numbered_components


def extract_component_descriptions(abstract: str, components: Dict[str, str]) -> Dict[str, str]:
    """
    Use LLM to extract actual descriptions for each component from the abstract.
    This generates invention-specific content, NOT template filler.
    Returns descriptions as sentence FRAGMENTS starting with verbs (e.g., "houses the primary battery unit").
    """
    comp_list = list(components.keys())[:10]
    
    prompt = f"""Based on this patent abstract, write a SHORT functional description for each component.

ABSTRACT:
{abstract}

COMPONENTS:
{', '.join(comp_list)}

CRITICAL RULES:
1. Each description must START WITH A VERB (e.g., "contains", "monitors", "controls")
2. Each description should be 10-20 words maximum
3. Describe the SPECIFIC function from the abstract, not generic text
4. Do NOT start with "A" or "The" or the component name

GOOD examples:
- "houses the primary battery unit and thermal management components"
- "monitors real-time biometric data including heart rate and skin conductance"
- "controls power distribution based on connected module type"

BAD examples (do NOT do this):
- "A protective enclosure that houses..." (starts with "A")
- "The main housing contains..." (starts with "The")
- "Housing - this is the main component..." (starts with component name)

Return ONLY a JSON object:
{{"component_name": "verb phrase describing function"}}

JSON:"""

    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=800,
            temperature=0.2,
            system_prompt="You are a patent attorney. Generate concise verb phrases describing component functions. Each phrase MUST start with a verb."
        )
        
        json_text = response.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].strip()
        
        # Find JSON object in response
        import re
        json_match = re.search(r'\{[^{}]+\}', json_text, re.DOTALL)
        if json_match:
            descriptions = json.loads(json_match.group())
            # Validate and fix any descriptions that don't start with a verb
            fixed = {}
            for comp, desc in descriptions.items():
                desc = desc.strip()
                # If it starts with "A " or "The ", remove it
                if desc.lower().startswith("a "):
                    desc = desc[2:]
                if desc.lower().startswith("the "):
                    desc = desc[4:]
                # If it still doesn't start with a verb, add a default
                if not desc or desc[0].isupper():
                    desc = f"comprises the {comp.lower()} of the invention"
                fixed[comp] = desc
            return fixed
    except Exception as e:
        print(f"Component description extraction failed: {e}")
    
    # Fallback: return empty dict (will use abstract-derived descriptions)
    return {}


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



def generate_dynamic_template(abstract: str, components: Dict[str, str]) -> str:
    """
    Generate a context-aware detailed description template based on the actual invention.
    This ensures consistency between the abstract and the detailed description.
    """
    # Extract invention name from abstract
    invention_name = "the present invention"
    invention_field = "mechanical engineering"
    
    import re
    # Clean abstract for matching
    abstract_clean = ' '.join(abstract.split())
    match = re.search(r'^A\s+([a-zA-Z\s\-]+?)(?:\s+comprising|\s+for\s+|\s+with\s+|,|\.)', abstract_clean, re.IGNORECASE)
    if match:
        invention_name = match.group(1).strip().lower()
    
    # Use LLM to determine invention field - works for ANY invention type
    field_prompt = f"""What technical field does this invention belong to?

ABSTRACT:
{abstract[:500]}

Return ONLY a short phrase (3-6 words) describing the field, such as:
- "portable power systems"
- "wearable biometric devices"
- "agricultural automation systems"
- "medical diagnostic apparatus"

FIELD:"""

    try:
        field_response = llm_generate(
            prompt=field_prompt,
            max_new_tokens=50,
            temperature=0.1,
            system_prompt="You are a patent classifier. Return ONLY the technical field name, nothing else."
        )
        if field_response:
            invention_field = field_response.strip().strip('"').strip("'")
            # Clean up common prefixes
            if invention_field.lower().startswith("the field of "):
                invention_field = invention_field[13:]
            if invention_field.lower().startswith("field of "):
                invention_field = invention_field[9:]
    except Exception as e:
        print(f"Field detection failed, using fallback: {e}")
        invention_field = f"{invention_name} technology"
    
    # Get component names and reference numerals - FAIL LOUDLY if none
    comp_items = list(components.items())
    if not comp_items:
        # CRITICAL: Do NOT use generic placeholders - return error template
        return f"""[0001] ERROR: No invention-specific components provided.
        
[0002] This is a NON-FILING DRAFT. Components could not be extracted.

[0003] Please provide a more detailed abstract with specific structural components.

Manual drafting is required for the detailed description of {invention_name}."""
    
    # *** CRITICAL: Extract ACTUAL descriptions from abstract using LLM ***
    # This replaces template filler with invention-specific content
    comp_descriptions = extract_component_descriptions(abstract, components)
    
    # Build component descriptions
    main_name, main_ref = comp_items[0]
    
    # Connection phrases to vary descriptions
    connection_phrases = [
        "is operatively connected to",
        "is in communication with", 
        "is coupled to",
        "interfaces with",
        "is positioned relative to",
        "cooperates with",
        "is integrated within",
        "is mounted on"
    ]
    
    # Generate structural paragraphs using ACTUAL extracted descriptions
    structural_paragraphs = []
    for i, (comp_name, comp_ref) in enumerate(comp_items[:8]):
        para_num = 6 + i
        conn_phrase = connection_phrases[i % len(connection_phrases)]
        
        # *** USE LLM-EXTRACTED DESCRIPTION instead of filler ***
        if comp_name in comp_descriptions:
            func_desc = comp_descriptions[comp_name]
        else:
            # Fallback: derive from component name
            func_desc = f"performs the function of {comp_name} within the {invention_name}"
        
        # Get adjacent component for connection description
        if i > 0:
            prev_name, prev_ref = comp_items[i-1]
            connection_desc = f"The {comp_name} {comp_ref} {conn_phrase} the {prev_name} {prev_ref}."
        else:
            connection_desc = f"The {comp_name} {comp_ref} forms part of the {invention_name}."
        
        structural_paragraphs.append(
            f"[{para_num:04d}] Referring to Figure 1, the {comp_name} {comp_ref} {func_desc}. "
            f"{connection_desc} "
            f"In the preferred embodiment, the {comp_name} {comp_ref} is configured to operate as described herein."
        )
    
    # Generate working paragraphs with invention-specific operational details
    working_paragraphs = []
    step_num = 21
    for i, (comp_name, comp_ref) in enumerate(comp_items[:5]):
        # Use extracted description for working section too
        if comp_name in comp_descriptions:
            func_desc = comp_descriptions[comp_name]
        else:
            func_desc = f"performs its designated function within the {invention_name}"
        
        working_paragraphs.append(
            f"[{step_num + i:04d}] In operation, the {comp_name} {comp_ref} {func_desc}. "
            f"The {comp_name} {comp_ref} operates in coordination with the other components of the {invention_name} to achieve the desired functionality."
        )
    
    # Build the full template
    template = f"""[0001] The following detailed description, taken in conjunction with the accompanying drawings, discloses the present invention in sufficient detail to enable a person skilled in the art to practice the invention without undue experimentation. Reference will now be made to the preferred embodiments of a {invention_name} as illustrated in the figures.

[0002] The present invention relates generally to the field of {invention_field}. More specifically, the invention pertains to {invention_name} and methods for its construction and operation.

[0003] In accordance with Section 10(4)(b) of the Patents Act, 1970, the following describes the BEST METHOD known to the applicant for performing the invention. The preferred embodiment represents the optimal configuration for the {invention_name} achieving maximum efficiency and reliability.

[0004] In the preferred embodiment of the present invention, the {main_name} {main_ref} comprises: {'; '.join([f'{name} {ref}' for name, ref in comp_items[:4]])}. These components are assembled and connected to form a functional {invention_name}.

[0005] The preferred materials for the {invention_name} include materials appropriate to the field of {invention_field}. The specifications enable complete reproduction of the invention by a person skilled in the art without undue experimentation.

{chr(10).join(structural_paragraphs)}

[0020] Working: In operation, the {invention_name} functions according to the following sequence. The operation begins when the user initiates the {main_name} {main_ref}.

{chr(10).join(working_paragraphs)}

[0030] The {invention_name} includes features that optimize performance during operation. These features ensure reliable and consistent functionality.

[0031] Error handling mechanisms detect and respond to abnormal conditions. The system is designed to maintain safe operation under various conditions.

[0035] In a preferred embodiment, the {main_name} {main_ref} is manufactured from suitable materials selected for durability and performance. Alternative materials may be used depending on specific application requirements.

[0036] While the preferred embodiment has been described, alternative embodiments may employ different configurations to achieve similar functionality.

[0037] In another embodiment of the present invention, the {invention_name} may be configured with additional features for enhanced performance.

[0038] In a further embodiment, the {invention_name} includes modified components for specialized applications.

[0040] The foregoing description of the specific embodiments will so fully reveal the general nature of the embodiments herein that others skilled in the art can readily modify and adapt the invention for various applications without departing from the generic concept.

[0041] Industrial Applicability: The present invention finds industrial applicability in the field of {invention_field} and related applications.

[0042] Advantages: The present invention provides significant advantages including improved functionality, enhanced reliability, and practical utility in the intended applications."""

    return template




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

    # Adaptive word count validation (NOT fixed 2500)
    # Word count checked against min_words/max_words at generation time
    if word_count < 1200:
        issues.append(f"Detailed description is too short ({word_count} words). Minimum is 1200 words for simple inventions.")
    elif word_count < 1500:
        warnings.append(f"Description is {word_count} words. May be sufficient for simple inventions but could be expanded.")




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
    max_attempts: int = 2,
    component_registry: dict = None  # Unified registry for consistent numerals
) -> Dict[str, any]:
    """
    Generate 'Detailed Description of the Invention' matching Indian Patent Office format.
    Enterprise mode: Adaptive length, domain-constrained, abstract-type aware.
    """
    # PRIORITY 3: Classify abstract type for branched generation
    abstract_type = classify_abstract_type(abstract)
    print(f"   Abstract type: {abstract_type.value}")
    
    # PRIORITY 1: Detect domain from abstract (NEVER hardcode)
    detected_domain = detect_abstract_domain(abstract)
    print(f"   Detected domain: {detected_domain}")
    
    # USE UNIFIED REGISTRY if provided, otherwise extract locally
    if component_registry and component_registry.get("components"):
        # Use consistent numerals from unified registry
        components = {name: f"({num})" for name, num in component_registry["components"].items()}
        print(f"   Using unified registry with {len(components)} components")
    else:
        # Fallback: extract components locally (may cause inconsistency)
        components = extract_components_with_numerals(abstract, claims)
    
    # PRIORITY 2: Calculate adaptive word count (NOT fixed 2500)
    min_words, max_words = calculate_adaptive_length(components, abstract_type)
    print(f"   Target word count: {min_words}-{max_words} words")
    
    component_list = "\n".join([f"   • {comp} {num}" for comp, num in list(components.items())[:15]])
    
    # Create component descriptions for the prompt
    comp_items = list(components.items())
    system_num = comp_items[0][1] if comp_items else "(100)"
    
    drawing_clause = "WITH REFERENCE TO THE ACCOMPANYING FIGURES" if drawing_summary.strip() else ""
    
    # Extract claim_1 for strict enforcement
    claim_1 = claims.split('\n\n')[0] if '\n\n' in claims else claims[:800]

    # ENTERPRISE IPO PROMPT: Strict claim-1 enforcement, domain constraints, self-check
    prompt = f"""You are a senior Indian patent attorney drafting the 
"DETAILED DESCRIPTION OF THE INVENTION" for an Indian Complete Specification 
(Form 2) under the Patents Act, 1970.

STRICT IPO MODE — FAILURE IS PREFERRED OVER GENERIC OUTPUT.

────────────────────────────────────────────
INPUTS
────────────────────────────────────────────
ABSTRACT:
{abstract[:600]}

CLAIM 1 (MANDATORY):
{claim_1}

DRAWING SUMMARY:
{drawing_summary[:400] if drawing_summary.strip() else "Figure 1: System overview according to the present invention."}

INVENTION TYPE:
{abstract_type.value.upper()}

TECHNICAL DOMAIN:
{detected_domain}

COMPONENTS WITH REFERENCE NUMERALS:
{component_list}

TARGET WORD RANGE:
{min_words} to {max_words} words (ADAPTIVE — NOT FIXED)

────────────────────────────────────────────
NON-NEGOTIABLE RULES (READ CAREFULLY)
────────────────────────────────────────────

1. EVERY technical component you describe MUST appear explicitly in CLAIM 1.
2. DO NOT invent components.
3. DO NOT use placeholders such as:
   "main assembly", "first component", "second component", "unit", "module"
   unless they appear verbatim in CLAIM 1.

4. If CLAIM 1 does not contain at least THREE concrete technical components,
   OUTPUT EXACTLY THIS AND STOP:
   "ERROR: Claim 1 lacks sufficient technical disclosure for IPO-compliant drafting."

5. ALL content MUST belong ONLY to this domain:
   "{detected_domain}"
   ❌ Do NOT mention unrelated industries, applications, or technologies.

6. Word count MUST fall WITHIN the target range.
   ❌ Do NOT pad text to reach a fixed length.

7. If sufficient invention-specific detail cannot be generated,
   OUTPUT EXACTLY THIS AND STOP:
   "ERROR: Insufficient technical disclosure — manual drafting required."

8. STRICT TERMINOLOGY CONSISTENCY:
   ✓ Use ONLY technical terms that appear in the ABSTRACT and CLAIM 1.
   ❌ Do NOT substitute alternative technical terms.
   ❌ Do NOT invent equivalent terms (e.g., if abstract says "optical sensor", 
      do NOT write "electrochemical sensor" or "photoelectric sensor").
   Example: If abstract says "multi-spectral optical sensor", use EXACTLY that term throughout.

────────────────────────────────────────────
REQUIRED STRUCTURE (FOLLOW EXACTLY)
────────────────────────────────────────────

[0001] PREAMBLE  
Standard Form-2 preamble enabling a person skilled in the art.

[0002] TECHNICAL FIELD  
Broad field → specific field (domain-consistent).

[0003-0005] BEST METHOD (Section 10(4)(b))  
Describe the preferred embodiment using ONLY Claim-1 components.

CRITICAL: ABSTRACT EXPANSION REQUIREMENT
Every technical concept mentioned in the ABSTRACT must be:
✓ Fully described in the detailed description
✓ Explained with sufficient detail for reproduction
✓ Supported by the structural description
If the abstract mentions "multi-spectral optical sensor", the description MUST explain how it works.

[0006-0020] STRUCTURAL DESCRIPTION  
For EACH component in Claim 1:
- Name the component
- Use its reference numeral
- Describe its structure
- Describe how it cooperates with other claimed components

[0021-0035] WORKING / OPERATION  
Step-by-step operation derived strictly from Claim 1 relationships.

[0036-0040] ALTERNATIVE EMBODIMENTS  
ONLY variations of the SAME invention.
NO new domains. NO new components.

[0041] INDUSTRIAL APPLICABILITY  
Limited strictly to the detected domain: {detected_domain}

[0042] CONCLUSION  
Standard scope-preserving conclusion.

────────────────────────────────────────────
LANGUAGE CONSTRAINTS
────────────────────────────────────────────
✓ Use "comprising", "configured to", "wherein"
✓ Functional language anchored to structure
✓ Examiner-neutral tone
❌ No marketing language
❌ No implementation brands or algorithms

BANNED BUZZWORDS (Unless in Abstract/Claim 1):
❌ AES-256, RSA, SHA-256 (specific cryptography)
❌ REST API, MQTT, HTTP, WebSocket (specific protocols)
❌ Cloud, AWS, Azure, Docker (specific platforms)
❌ Multi-factor authentication, OAuth (specific auth)
❌ Machine learning, AI, neural network (software methods)
Only mention these if EXPLICITLY stated in Abstract or Claim 1.
If not in Abstract, describe functionally instead:
  - Instead of "AES-256 encryption" → "encryption module"
  - Instead of "REST API" → "communication interface"

STYLE RULES:
❌ Do NOT repeat "the present invention" more than 3 times total
✓ Use "the invention", "the system", "the device" instead
❌ Do NOT use excessive verbosity in background
✓ Keep paragraphs focused and technical

GRAMMAR RULES (CRITICAL):
✓ Use complete, grammatically correct sentences
✓ Maintain subject-verb agreement
✓ Use proper article usage (a/an/the)
✓ Avoid run-on sentences - use periods, semicolons, or commas appropriately
✓ Proofread each paragraph before moving to the next
❌ Do NOT use sentence fragments
❌ Do NOT mix tenses within a paragraph

BACKGROUND SECTION RULES:
✓ Keep background to maximum 2-3 paragraphs
✓ Focus on technical problem being solved
❌ Do NOT include lengthy prior art discussions
❌ Do NOT use marketing language in background

────────────────────────────────────────────
FINAL SELF-CHECK (MANDATORY)
────────────────────────────────────────────
Before outputting, verify internally:
✓ No generic placeholders
✓ Every component maps to Claim 1
✓ Domain consistency maintained
✓ Word count within target range
✓ TERMINOLOGY: Using EXACT terms from abstract (no substitutions)
✓ PARAGRAPH NUMBERING: Strictly sequential [0001], [0002], [0003]... with NO GAPS
  ❌ Do NOT jump from [0005] to [0030]
  ❌ Do NOT repeat paragraph numbers
  ❌ Do NOT skip numbers

────────────────────────────────────────────
NOW WRITE THE COMPLETE
"DETAILED DESCRIPTION OF THE INVENTION"
Start at paragraph [0001].

DETAILED DESCRIPTION OF THE INVENTION {drawing_clause}

[0001] The following detailed description,"""



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

            if not generated.strip().lower().startswith("the present invention as herein described relates to") and not generated.strip().startswith("[0001]"):
                raw_text = "[0001] The following detailed description relates to " + generated.strip()
            else:
                raw_text = generated.strip()

            cleaned_text = clean_detailed_description(raw_text)
            word_count = len(cleaned_text.split())
            
            # EXPANSION: If too short, add more content (use adaptive min_words)
            if word_count < min_words:
                print(f"   Expanding from {word_count} words to reach {min_words}...")
                
                # Add Working section if missing or short
                if "working:" not in cleaned_text.lower() and "[0021]" not in cleaned_text:
                    working_section = f"""

[0021] Working: In operation, the system {comp_items[0][1] if comp_items else '(100)'} functions as follows. When power is applied, the system initializes all components and performs self-diagnostic checks to ensure proper functionality.

[0022] Upon successful initialization, the sensor array {comp_items[0][1] if comp_items else '(101)'} begins collecting environmental data at predetermined intervals. The collected data is transmitted to the processing unit {comp_items[1][1] if len(comp_items) > 1 else '(102)'} through a dedicated communication interface.

[0023] The processing unit {comp_items[1][1] if len(comp_items) > 1 else '(102)'} receives the raw data and applies signal conditioning algorithms to filter noise and enhance signal quality. If the processed data indicates abnormal conditions, the processing unit triggers an alert sequence.

[0024] The alert sequence involves transmitting the processed data to the communication module {comp_items[2][1] if len(comp_items) > 2 else '(103)'} which establishes a secure connection with the remote server. The data is encrypted and transmitted using industry-standard protocols.

[0025] Upon receiving the data, the remote server stores the information in a database and generates appropriate notifications. If threshold values are exceeded, the server pushes real-time alerts to registered mobile devices.

[0026] The mobile application {comp_items[3][1] if len(comp_items) > 3 else '(104)'} receives the alerts and displays them to the user in an intuitive format. The user can acknowledge the alert, view historical data, or configure system parameters remotely.

[0027] In continuous monitoring mode, the system repeats the data collection and transmission cycle at regular intervals. The interval duration is configurable based on application requirements and power constraints.

[0028] The system includes power management features that optimize energy consumption during idle periods. When no data transmission is required, the communication module enters a low-power standby mode."""
                    cleaned_text += working_section
                
                # Add Best Method section if missing
                if "[0036]" not in cleaned_text:
                    # DYNAMIC: Generate invention-specific best method section using LLM
                    # Do NOT use hardcoded templates (they cause domain leakage)
                    best_method_prompt = f"""Generate 5 alternative embodiment paragraphs for this patent:

INVENTION FROM ABSTRACT: {abstract[:400]}

Write paragraphs [0036]-[0040] for Alternative Embodiments section.
Each paragraph must:
1. Start with "[00XX] In"
2. Describe an alternative for THIS SPECIFIC invention only
3. Use the SAME domain/field as the abstract
4. Be 50-80 words each

IMPORTANT:
- Do NOT mention water quality, aquaculture, SCADA, or wastewater
- Only describe alternatives relevant to: {abstract[:100]}

WRITE:"""
                    
                    try:
                        alt_response = llm_generate(
                            prompt=best_method_prompt,
                            max_new_tokens=500,
                            temperature=0.3,
                            system_prompt=f"You are writing patent alternatives ONLY for: {abstract[:100]}. Never mention unrelated domains."
                        )
                        if alt_response and len(alt_response.strip()) > 100:
                            best_method_section = "\n\n" + alt_response.strip()
                        else:
                            # Minimal fallback - invention-aware
                            invention_type = abstract.split('.')[0] if abstract else "the system"
                            best_method_section = f"""

[0036] In a preferred embodiment of the present invention, the primary components are manufactured from high-grade materials selected for optimal performance and durability in the intended application.

[0037] While specific sensor technologies have been described, alternative sensing mechanisms may be employed in different embodiments to achieve specific measurement requirements.

[0038] In another embodiment, the system may be configured with alternative power sources or connectivity options to suit different deployment environments.

[0039] In a further embodiment, additional features may be incorporated to enhance functionality while maintaining the core operational principles.

[0040] In yet another embodiment, the form factor and physical arrangement may be modified to accommodate different use cases while preserving the essential inventive concept."""
                    except Exception as e:
                        print(f"Alternative embodiments generation failed: {e}")
                        best_method_section = ""
                    
                    cleaned_text += best_method_section
                
                # Add Conclusion if missing
                if "[0041]" not in cleaned_text:
                    conclusion_section = """

[0041] The foregoing description of the specific embodiments will so fully reveal the general nature of the embodiments herein that others can, by applying current knowledge, readily modify and/or adapt for various applications such specific embodiments without departing from the generic concept, and, therefore, such adaptations and modifications should and are intended to be comprehended within the meaning and range of equivalents of the disclosed embodiments. It is to be understood that the phraseology or terminology employed herein is for the purpose of description and not of limitation. The scope of the invention is defined by the appended claims."""
                    cleaned_text += conclusion_section

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

    # Final expansion if still too short (use adaptive min_words)
    if best_result and best_result["word_count"] < min_words:
        print(f"   Adding expansion to reach {min_words} words (domain: {detected_domain})...")
        
        # DYNAMIC: Generate invention-specific expansion using LLM
        # Do NOT use hardcoded templates (they cause domain leakage)
        expansion_prompt = f"""Generate additional paragraphs [0029]-[0035] and [0042]-[0043] for this patent:

INVENTION: {abstract[:400]}

Write paragraphs covering:
- [0029] Operational details specific to THIS invention
- [0030] Configuration options for THIS invention
- [0031] Error handling for THIS invention
- [0032] Safety/security features for THIS invention
- [0033] Update/maintenance for THIS invention
- [0034] Diagnostic features for THIS invention
- [0035] Integration capabilities for THIS invention
- [0042] Industrial applicability for THIS SPECIFIC invention (not generic industries)
- [0043] Advantages of THIS SPECIFIC invention

CRITICAL DOMAIN CONSTRAINT:
- This invention is in the domain of: {detected_domain}
- ONLY describe features relevant to: {detected_domain}
- Do NOT mention any unrelated domains or industries
- Each paragraph 40-60 words
- Must be specific to the invention in the abstract

WRITE:"""
        
        try:
            expansion_response = llm_generate(
                prompt=expansion_prompt,
                max_new_tokens=800,
                temperature=0.3,
                system_prompt=f"Write patent paragraphs ONLY for: {abstract[:100]}. Never add unrelated content."
            )
            if expansion_response and len(expansion_response.strip()) > 200:
                additional_content = "\n\n" + expansion_response.strip()
            else:
                # Minimal generic fallback - NO domain-specific content
                additional_content = f"""

[0042] Industrial Applicability: The present invention as described herein finds practical applicability in its intended field as defined by the abstract and claims.

[0043] Advantages: The present invention provides advantages over prior art systems as described in the detailed embodiments above."""
        except Exception as e:
            print(f"Expansion generation failed: {e}")
            additional_content = ""
        
        best_result["text"] += additional_content
        best_result["word_count"] = len(best_result["text"].split())
        best_result["valid"] = best_result["word_count"] >= min_words

    # FALLBACK: If still no result, generate complete template using abstract context
    # CRITICAL: Fallback must NEVER be marked as valid (enterprise safety)
    if not best_result or best_result["word_count"] < 500:
        print("   ⚠️ USING FALLBACK TEMPLATE - NON-FILING DRAFT ⚠️")
        
        # Generate context-aware template based on actual abstract and components
        template = generate_dynamic_template(abstract, components)

        # ISSUE 3 FIX: Fallback must be marked as INVALID
        best_result = {
            "text": FALLBACK_WARNING + "\n\n" + template,
            "valid": False,  # NEVER valid - enterprise safety
            "issues": ["Fallback template used — this is a NON-FILING draft that requires manual drafting"],
            "warnings": ["Template-based generation - not invention-specific"],
            "word_count": len(template.split()),
            "has_reference_numerals": True,
            "has_working_section": True,
            "has_embodiments": True,
            "components": components,
            "attempt": max_attempts,
            "score": 100,  # High score = low quality
            "is_fallback": True
        }

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
