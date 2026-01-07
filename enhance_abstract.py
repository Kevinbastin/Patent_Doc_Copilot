"""
WIPO ST.12/A Compliant Abstract Enhancer
=========================================
Generates patent abstracts following WIPO Standard ST.12/A guidelines:
- 50-150 words (preferably), max 250
- Concise technical disclosure
- NO phrases like "The present invention relates to"
- NO legal phraseology ("said", "means", "wherein")
- Focus on WHAT IS NEW
"""

import re
from typing import Dict
from llm_runtime import llm_generate


def enhance_abstract(raw_input: str) -> Dict:
    """
    Enhance a rough invention description into a WIPO ST.12/A compliant abstract.
    
    Args:
        raw_input: Any description of the invention
        
    Returns:
        Dict with enhanced abstract and metadata
    """
    
    if not raw_input or len(raw_input.strip()) < 20:
        return {
            "enhanced_abstract": "",
            "status": "error",
            "message": "Input too short (minimum 20 characters)"
        }
    
    # WIPO ST.12/A compliant prompt
    abstract_prompt = f"""Generate a patent abstract following WIPO Standard ST.12/A.

INVENTION DESCRIPTION:
{raw_input[:1200]}

WIPO ST.12/A REQUIREMENTS:

1. LENGTH: 50-150 words (CRITICAL - target 100-120 words)

2. STRUCTURE FOR A SYSTEM/APPARATUS:
   - Start directly with what the invention IS (no preamble)
   - Describe organization and operation
   - Mention key components and their functions
   - State the main technical advantage

3. FORBIDDEN PHRASES (DO NOT USE):
   - "The present invention relates to..."
   - "This disclosure concerns..."
   - "The invention defined by this disclosure..."
   - "said" (use "the" instead)
   - "means" (use specific component names)
   - "wherein"
   - "comprises"/"comprising" at sentence start

4. FOCUS ON WHAT IS NEW:
   - Lead with the novel technical feature
   - Be specific about the technical solution
   - Mention the technical effect/advantage

5. STYLE:
   - Concise, clear sentences
   - Technical but readable
   - No marketing claims ("better", "improved", "novel")
   - No speculative applications

EXAMPLE GOOD ABSTRACT:
"A heart valve with an annular valve body defining an orifice and having a plurality of struts forming a pair of cages on opposite sides of the orifice. A spherical closure member is captively held within the cages and is moved by blood flow between the open and closed positions in check valve fashion."

NOW WRITE THE ABSTRACT (50-150 words, start directly with the invention):"""

    enhanced = llm_generate(
        abstract_prompt,
        max_new_tokens=500,
        temperature=0.2,
        stop_strings=["Claims:", "CLAIMS", "Background:", "---", "\n\n\n"]
    )
    
    # Clean the output
    enhanced_text = clean_abstract(enhanced)
    word_count = len(enhanced_text.split())
    
    # If too short, try expansion
    if word_count < 50:
        expansion_prompt = f"""Expand this abstract to 80-120 words while maintaining WIPO ST.12/A format:

CURRENT ({word_count} words): {enhanced_text}

ORIGINAL INVENTION: {raw_input[:500]}

Add more technical details about:
- Component organization
- How it operates
- Technical advantage

Write expanded abstract (80-120 words, NO "The present invention"):"""

        expanded = llm_generate(
            expansion_prompt,
            max_new_tokens=400,
            temperature=0.25,
            stop_strings=["---", "\n\n"]
        )
        expanded_text = clean_abstract(expanded)
        if len(expanded_text.split()) > word_count:
            enhanced_text = expanded_text
            word_count = len(enhanced_text.split())
    
    # Validate
    validation = validate_abstract(enhanced_text)
    
    return {
        "enhanced_abstract": enhanced_text,
        "original_input": raw_input[:500],
        "word_count": word_count,
        "validation": validation,
        "status": "success" if validation["is_valid"] else "needs_review"
    }


def clean_abstract(text: str) -> str:
    """Clean and format the abstract per WIPO ST.12/A."""
    
    # Remove common headers
    text = re.sub(r'^(ABSTRACT:?|Abstract:?)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove forbidden opening phrases
    forbidden_starts = [
        r'^The present invention relates to\s*',
        r'^This disclosure concerns\s*',
        r'^The invention defined by this disclosure\s*',
        r'^This invention relates to\s*',
        r'^According to the invention\s*',
    ]
    for pattern in forbidden_starts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # CRITICAL: Remove redundant title at the start of abstract
    # Pattern: "Title Title A system..." → "A system..."
    # Detects when the same phrase appears twice at the start
    words = text.split()
    if len(words) >= 6:
        # Check for repeated title pattern (e.g., "Solar Vehicle Solar Vehicle A...")
        for title_len in range(2, min(8, len(words) // 2)):
            first_segment = ' '.join(words[:title_len]).lower()
            second_segment = ' '.join(words[title_len:title_len*2]).lower()
            if first_segment == second_segment:
                # Remove the first occurrence of the title
                text = ' '.join(words[title_len:])
                break
        
        # Also check for "Title A title-based..." pattern
        # e.g., "Modular Solar Vehicle A modular solar vehicle system..."
        first_words = ' '.join(words[:4]).lower()
        for i in range(4, min(12, len(words))):
            if words[i].lower() in ['a', 'an', 'the']:
                following_words = ' '.join(words[i+1:i+4]).lower()
                if first_words.startswith(following_words[:15]) or following_words.startswith(first_words[:15]):
                    # Remove title from beginning
                    text = ' '.join(words[i:])
                    break
    
    # Remove markdown
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    # Capitalize first letter if needed
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text


def validate_abstract(text: str) -> Dict:
    """Validate abstract against WIPO ST.12/A requirements."""
    
    issues = []
    warnings = []
    
    word_count = len(text.split())
    text_lower = text.lower()
    
    # Length check (WIPO: 50-150 preferred, max 250)
    if word_count < 50:
        issues.append(f"Too short: {word_count} words (minimum 50)")
    elif word_count > 250:
        issues.append(f"Too long: {word_count} words (maximum 250)")
    elif word_count > 150:
        warnings.append(f"Above preferred range: {word_count} words (50-150 preferred)")
    
    # Forbidden phrases check
    forbidden_phrases = [
        "the present invention",
        "this disclosure concerns",
        "the invention relates to",
        "according to the present invention",
    ]
    for phrase in forbidden_phrases:
        if phrase in text_lower:
            issues.append(f"Contains forbidden phrase: '{phrase}'")
    
    # Legal phraseology check 
    legal_words = ["said ", " means ", "wherein "]
    found_legal = [w.strip() for w in legal_words if w in text_lower]
    if found_legal:
        warnings.append(f"Contains legal phraseology: {found_legal}")
    
    # Marketing terms check
    marketing = ["novel", "innovative", "revolutionary", "improved", "better", "unique"]
    found_marketing = [t for t in marketing if t in text_lower]
    if found_marketing:
        warnings.append(f"Contains marketing terms: {found_marketing}")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "in_preferred_range": 50 <= word_count <= 150
    }


def format_for_filing(abstract: str) -> str:
    """Format abstract with standard heading."""
    return f"""ABSTRACT

{abstract}"""


# Test
if __name__ == "__main__":
    rough_input = """
    I made a water monitoring device that uses sensors to check water quality.
    It has pH sensor, turbidity sensor, and dissolved oxygen sensor.
    It connects to cloud and sends alerts to phone app when water is bad.
    Uses machine learning to predict water quality trends.
    Very useful for environmental monitoring and fish farming.
    """
    
    print("=" * 70)
    print("🔧 WIPO ST.12/A ABSTRACT ENHANCER")
    print("=" * 70)
    
    print("\n📝 ORIGINAL INPUT:")
    print(rough_input[:200])
    
    result = enhance_abstract(rough_input)
    
    print("\n✨ ENHANCED ABSTRACT (WIPO ST.12/A):")
    print("-" * 70)
    print(result["enhanced_abstract"])
    print("-" * 70)
    
    print(f"\n📊 Word count: {result['word_count']}")
    print(f"✅ Valid: {result['validation']['is_valid']}")
    print(f"📏 In preferred range (50-150): {result['validation']['in_preferred_range']}")
    
    if result['validation']['issues']:
        print(f"❌ Issues: {result['validation']['issues']}")
    if result['validation']['warnings']:
        print(f"⚠️ Warnings: {result['validation']['warnings']}")
    
    print("=" * 70)
