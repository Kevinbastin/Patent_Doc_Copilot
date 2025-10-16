from llama_cpp import Llama
import re
from typing import Dict


# Path to your local Phi-3 GGUF model
LLM_PATH = "/app/models/models/phi-3-mini-4k-instruct-q4.gguf"


# Load the model
llm = Llama(model_path=LLM_PATH, device="cuda", n_ctx=4096, n_threads=4, verbose=False)


def generate_summary_of_invention(abstract: str, claims: str = "", max_attempts: int = 3) -> Dict[str, any]:
    """
    Generate 'SUMMARY OF THE INVENTION' section matching Indian Patent Office format.
    
    Real patent structure (IN202541069047):
    - Opens with: "Thus according to the basic aspect of the present invention, there is provided..."
    - Describes the invention in structured format (like Claim 1 expanded)
    - Lists components with indentation
    - Multiple "wherein" clauses
    - Followed by "It is another aspect..." statements (3-5 additional aspects)
    
    This is NOT a summary of the abstract - it's a technical restatement!
    """
    
    # Build prompt based on real patent format
    prompt = f"""You are a patent attorney drafting "SUMMARY OF THE INVENTION" for an Indian Complete Specification patent.

INVENTION ABSTRACT:
{abstract}

{f"FIRST CLAIM: {claims[:500]}" if claims else ""}

REAL PATENT EXAMPLE STRUCTURE:

SUMMARY OF THE INVENTION

Thus according to the basic aspect of the present invention, there is provided an Internet of Things (IoT) based remote monitoring and alerting system for mitigating human-animal conflict, comprising:
a plurality of field-deployed sensor nodes, each sensor node comprising:
   at least one animal detection module;
   a microcontroller unit configured to process input from said animal detection module;
   an integrated wireless dual communication system comprising Long Range Wide Area Network (LoRaWAN) module and Global System for Mobile Communication (GSM) module; and
   a deterrent unit configured to activate upon detection of an animal presence;
a central master node configured to receive detection data from said sensor nodes;
a cloud server; and
a power management unit,
   wherein the microcontroller unit is configured to execute machine learning or TinyML models,
   wherein the LoRaWAN module is prioritized for low-power transmission,
   wherein the GSM module acts as a fall back channel,
   and wherein a web server operates with a user-friendly interface.

It is another aspect of the present invention, wherein the sensor nodes are arranged in a mesh or star topology.

It is another aspect of the present invention, wherein the animal detection module is selected from PIR sensors, cameras, and acoustic sensors.

STRICT REQUIREMENTS:
1. Start with: "Thus according to the basic aspect of the present invention, there is provided..."
2. List main system with "comprising:" followed by components
3. Multiple "wherein" clauses (5-8)
4. Follow with 3-5 "It is another aspect..." statements
5. Length: 300-500 words

NOW WRITE (only text, no heading):

Thus according to the basic aspect of the present invention, there is provided"""

    best_result = None
    best_score = float('inf')
    
    for attempt in range(max_attempts):
        try:
            response = llm(
                prompt=prompt,
                max_tokens=1200,
                temperature=0.25 if attempt == 0 else 0.3 + (attempt * 0.1),
                stop=["BRIEF DESCRIPTION", "\n\n\n\n\n"],
                top_p=0.85,
                repeat_penalty=1.18
            )
            
            raw_text = "Thus according to the basic aspect of the present invention, there is provided" + response["choices"][0]["text"].strip()
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
                "has_wherein": validation["has_wherein"],
                "aspect_count": validation["aspect_count"],
                "attempt": attempt + 1,
                "score": score
            }
            
            if validation["valid"] and len(validation["warnings"]) <= 1:
                return result
            
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
        "attempt": max_attempts
    }


def clean_summary(text: str) -> str:
    """Clean and format the summary text."""
    text = re.sub(r'^(SUMMARY OF THE INVENTION:?)\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\n{3,}', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def validate_summary(text: str) -> Dict[str, any]:
    """Validate against Indian Patent Office standards for Summary."""
    issues = []
    warnings = []
    
    word_count = len(text.split())
    text_lower = text.lower()
    
    if not text.startswith('Thus according to the basic aspect'):
        issues.append("Must start with 'Thus according to the basic aspect...'")
    
    has_comprising = 'comprising' in text_lower
    if not has_comprising:
        issues.append("Must include 'comprising:' to list components")
    
    wherein_count = text_lower.count('wherein')
    has_wherein = wherein_count > 0
    if wherein_count < 3:
        warnings.append(f"Should have multiple 'wherein' clauses (found {wherein_count})")
    
    aspect_matches = re.findall(r'It is another aspect of the present invention', text)
    aspect_count = len(aspect_matches)
    if aspect_count < 2:
        warnings.append(f"Should have 3-5 'It is another aspect...' statements (found {aspect_count})")
    
    if word_count < 200:
        issues.append("Summary too brief. Should be 300-500 words.")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count,
        "has_comprising": has_comprising,
        "has_wherein": has_wherein,
        "aspect_count": aspect_count
    }


# BACKWARD COMPATIBILITY FUNCTION - For existing app.py
def summarize_abstract(abstract: str) -> str:
    """
    Backward compatibility wrapper for existing app.py.
    Generates SUMMARY OF THE INVENTION section.
    
    Note: This is NOT a condensed summary - it's a structured technical restatement!
    """
    result = generate_summary_of_invention(abstract)
    
    if result and result.get("text"):
        return result["text"]
    else:
        # Fallback if generation fails
        return f"Thus according to the basic aspect of the present invention, there is provided {abstract}"


# Example usage
if __name__ == "__main__":
    sample_abstract = """An Internet of Things (IoT) based remote monitoring and alerting system for human-animal conflict mitigation, comprising a plurality of field-deployed sensor nodes, a central master node, a cloud server, and a power management unit. Each sensor node comprises at least one animal detection module, a microcontroller unit, an integrated dual wireless communication system comprising LoRaWAN module and GSM module, and a deterrent unit configured to activate upon detection of an animal presence."""
    
    print("Testing summarize_abstract (backward compatibility):")
    print("=" * 80)
    result = summarize_abstract(sample_abstract)
    print(result)
    print("=" * 80)
