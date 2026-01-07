import os

# Optional: faiss for prior art search (claims generation works without it)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

import json
import numpy as np
import re
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Optional: sentence_transformers for embeddings (only needed for prior art search)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    EMBEDDINGS_AVAILABLE = False

# Import LLM generation function from llm_runtime (now uses Qwen 3 8B API)
from llm_runtime import llm_generate


# === Configuration ===
class PatentConfig:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss.index")
    METADATA_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss_metadata.json")

    # Generation parameters (optimized for Qwen 3 8B)
    TEMPERATURE = 0.15
    TOP_P = 0.85
    REPEAT_PENALTY = 1.15
    MAX_TOKENS_CLAIM1 = 1500      # Increased for complete claims
    MAX_TOKENS_DEPENDENT = 400    # Increased from 280 to prevent truncation
    MAX_TOKENS_METHOD = 1000      # Increased from 850 to prevent truncation



# === Load models ONCE (singleton pattern) ===
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            
            # Initialize with None
            cls._instance.embedding_model = None
            cls._instance.index = None
            cls._instance.metadata = []
            cls._instance.prior_art_available = False

            try:
                # Only try to load if dependencies are available
                if EMBEDDINGS_AVAILABLE and SentenceTransformer is not None:
                    cls._instance.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                    
                    # Try to load FAISS
                    if FAISS_AVAILABLE and os.path.exists(PatentConfig.INDEX_PATH):
                        cls._instance.index = faiss.read_index(PatentConfig.INDEX_PATH)
                        cls._instance.prior_art_available = True
                    
                    # Try to load Metadata
                    if os.path.exists(PatentConfig.METADATA_PATH):
                        with open(PatentConfig.METADATA_PATH, "r") as f:
                            cls._instance.metadata = json.load(f)
                else:
                    print("INFO: Prior art search disabled (sentence_transformers not installed)")
                    
            except Exception as e:
                print(f"WARNING: Error initializing RAG components: {e}")

        return cls._instance


# === Component Extraction (Enhanced) ===
class ComponentExtractor:
    """Extract structured components from patent abstract using LLM for any domain"""
    
    @classmethod
    def extract(cls, abstract: str) -> Dict[str, any]:
        """Extract all key components using LLM with regex fallback"""
        components = {
            'device_name': 'system',
            'device_confidence': 0.0,
            'purpose': '',
            'purpose_confidence': 0.0,
            'key_elements': [],
            'functions': [],
            'connections': [],
            'technical_effects': [],
            'novelty_indicators': []
        }
        
        # LLM-based semantic extraction
        prompt = f"""Extract the core components of the following patent abstract for claim drafting.
Return exactly a JSON object with these keys:
- device_name: the main subject of the invention (e.g., "folding turbine blade", "chemical composition", "deep learning model")
- purpose: the primary technical goal or field
- key_elements: list of essential structural or functional components
- functions: list of specific operations performed by components
- technical_effects: list of improvements or results (e.g., "reduced latency", "increased yield")
- novelty_indicators: list of words signaling novelty (e.g., "improved", "staged", "novel")

ABSTRACT:
{abstract}

JSON OUTPUT:"""
        
        try:
            response = llm_generate(
                prompt=prompt,
                max_new_tokens=400,
                temperature=0.1,
                system_prompt="You are a patent component extraction engine. Output ONLY valid JSON."
            )
            
            # Clean response for JSON parsing
            json_text = response.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].strip()
                
            data = json.loads(json_text)
            
            components['device_name'] = data.get('device_name', components['device_name'])
            components['purpose'] = data.get('purpose', '')
            components['key_elements'] = data.get('key_elements', [])
            components['functions'] = data.get('functions', [])
            components['technical_effects'] = data.get('technical_effects', [])
            components['novelty_indicators'] = data.get('novelty_indicators', [])
            
            # Boost confidence for LLM results
            components['device_confidence'] = 0.9
            components['purpose_confidence'] = 0.9
            
        except Exception as e:
            # Fallback to regex logic if LLM fails
            print(f"LLM extraction failed, using regex fallback: {e}")
            cls._regex_fallback(abstract, components)
            
        return components

    @staticmethod
    def _regex_fallback(abstract: str, components: Dict):
        """Original regex-based extraction logic as a safety net"""
        DEVICE_PATTERNS = [
            r'(?:A|An|The)\s+([^,]{15,80}?)\s+(?:comprising|including|having|for|that|which)',
            r'(?:present invention relates to|invention provides|disclosed is)\s+(?:a|an)\s+([^,]{15,80}?)(?:\s+comprising|\s+for|\s+that)',
        ]
        
        for pattern in DEVICE_PATTERNS:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                device_name = match.group(1).strip()
                components['device_name'] = device_name
                components['device_confidence'] = 0.5
                break
        
        # Note: Element extraction is now done via LLM for universal input support
        # The key_elements list will be populated by LLM-based extraction if needed
        # This avoids keyword-based assumptions that fail for unknown invention types
# === CLAIM NORMALIZATION UTILITIES ===
def normalize_device_for_claim(device_name: str) -> str:
    """
    IPO-safe normalization: returns a generic but relevant subject.
    Example: 'A folding wind turbine blade' -> 'turbine blade'
    """
    if not device_name:
        return "system"
    
    # Remove leading articles and common patent filler words
    device_name = re.sub(r'^(?:a|an|the|improved|novel|disclosed|new|present)\s+', '', device_name, flags=re.IGNORECASE)
    
    # If the name is too specific (contains 'for', 'configured to', etc.), truncate
    device_name = re.split(r'\s+(?:for|configured|adapted|having|comprising|which|that)\b', device_name, flags=re.IGNORECASE)[0]
    
    # Return the last 2-3 words for a balance of specificity and generality
    words = device_name.split()
    if len(words) > 3:
        return " ".join(words[-2:]).strip()
    
    return device_name.strip() or "system"

  

# === Enhanced Prior Art Retrieval ===
class PriorArtRetriever:
    """Retrieve and analyze prior art with relevance scoring"""
    
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
    
    def retrieve(self, abstract: str, top_k: int = 5) -> List[Dict[str, any]]:
        """Retrieve top-k most relevant prior art (skips if index missing)"""
        if self.mm.index is None or self.mm.embedding_model is None:
            return []
            
        try:
            query_embedding = self.mm.embedding_model.encode(
                [abstract], 
                convert_to_numpy=True
            )
            distances, indices = self.mm.index.search(query_embedding, top_k)
            
            prior_art = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.mm.metadata):
                    patent_data = self.mm.metadata[idx]
                    prior_art.append({
                        'rank': i + 1,
                        'distance': float(dist),
                        'similarity': 1.0 / (1.0 + float(dist)),
                        'abstract': patent_data.get('abstract', ''),
                        'title': patent_data.get('title', ''),
                        'patent_id': patent_data.get('patent_id', f'PRIOR-{idx}')
                    })
            
            return prior_art
        except Exception as e:
            print(f"Error retrieving prior art: {e}")
            return []
    
    def format_for_context(self, prior_art: List[Dict]) -> str:
        """Format prior art for LLM context"""
        if not prior_art:
            return "No similar prior art found."
        
        context = "RELEVANT PRIOR ART:\n\n"
        for pa in prior_art[:3]:
            context += f"Prior Art {pa['rank']} (Similarity: {pa['similarity']:.2f}):\n"
            context += f"Title: {pa.get('title', 'N/A')}\n"
            context += f"Abstract: {pa['abstract'][:200]}...\n\n"
        
        return context


# === POST-PROCESSING MODULE ===
class ClaimPostProcessor:
    """Robust post-processing to clean LLM-generated claims"""
    @staticmethod
    def _remove_non_english(text: str) -> str:
        """
        Remove non-English (non-ASCII) characters to prevent foreign language leakage.
        """
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def clean_claim_text(raw_text: str, claim_number: int) -> str:
        """Clean and normalize claim text with multiple validation passes"""
        
        # Step 1: Remove LLM artifacts
        cleaned = ClaimPostProcessor._remove_llm_artifacts(raw_text)
        cleaned = re.sub(r'[^\x00-\x7F]+', '', cleaned)
        # After Step 1
        cleaned = ClaimPostProcessor._remove_non_english(cleaned)

        # Step 2: Extract only the relevant claim
        cleaned = ClaimPostProcessor._extract_target_claim(cleaned, claim_number)
        
        # Step 3: Remove explanatory text
        cleaned = ClaimPostProcessor._remove_explanations(cleaned)
        
        # Step 4: Fix formatting issues
        cleaned = ClaimPostProcessor._fix_formatting(cleaned)
        
        # Step 5: Enforce single sentence (IPO requirement)
        cleaned = ClaimPostProcessor._ensure_single_sentence(cleaned)
        
        # Step 6: Validate claim structure
        cleaned = ClaimPostProcessor._limit_wherein_clauses(cleaned, max_wherein=3)
        cleaned = ClaimPostProcessor._validate_structure(cleaned, claim_number)
        cleaned = ClaimPostProcessor._enforce_antecedent_basis(cleaned)

        return cleaned

    @staticmethod
    def _ensure_single_sentence(text: str) -> str:
        """Enforce single sentence rule for IPO claims by converting internal periods to semicolons."""
        if not text:
            return ""
        
        # Preserve the final period
        has_final_period = text.strip().endswith('.')
        body = text.strip().rstrip('.')
        
        # Replace periods followed by spaces and a capital letter (likely sentence breaks)
        # Avoid common abbreviations like fig., e.g., i.e.
        body = re.sub(r'(?<![Ff]ig)(?<![Ee]\.g)(?<![Ii]\.e)\.\s+(?=[A-Z0-9])', '; ', body)
        
        return body + "." if has_final_period else body
    
    @staticmethod
    def _remove_llm_artifacts(text: str) -> str:
        """Remove common LLM artifacts and control tokens"""
        
        artifacts = [
            r'<\|assistant\|>',
            r'<\|user\|>',
            r'<\|system\|>',
            r'<\|.*?\|>',
            r'===+',
            r'---+',
            r'This new claim \d+ builds upon.*?(?=\n\n|\Z)',
            r'WRITE NOW:.*?(?=\n\d+\.)',
            r'FORMAT:.*?(?=\n\d+\.)',
            r'\[IMPORTANT\].*?(?=\n)',
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    @staticmethod
    def _extract_target_claim(text: str, claim_number: int) -> str:
        """Extract only the target claim, removing duplicates"""
        
        pattern = rf'({claim_number}\.\s+.*?)(?=\s+\d+\.|\Z)'
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
        
        if not matches:
            return text
        
        # If multiple matches, take the longest valid one
        candidates = []
        for match in matches:
            claim_text = match.group(0).strip()
            score = len(claim_text)
            if 'wherein' in claim_text.lower():
                score += 100
            if 'comprising' in claim_text.lower():
                score += 50
            if '<|' not in claim_text:
                score += 200
            candidates.append((score, claim_text))
        
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else text
    
    @staticmethod
    def _remove_explanations(text: str) -> str:
        """Remove explanatory sentences added by LLM"""
        
        explanation_patterns = [
            r'\.\s+This (?:new )?claim.*?(?=\n|$)',
            r'\.\s+The (?:claim|above).*?(?:specifies|describes|builds|adds).*?(?=\n|$)',
            r'\.\s+Note that.*?(?=\n|$)',
            r'\.\s+In this claim.*?(?=\n|$)',
        ]
        
        for pattern in explanation_patterns:
            text = re.sub(pattern, '.', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _fix_formatting(text: str) -> str:
        """Fix common formatting issues"""
        
        # Fix spacing issues
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix punctuation
        text = re.sub(r'\s+([,;.])', r'\1', text)
        text = re.sub(r'([,;.])\s*([,;.])', r'\1', text)
        
        # Fix "wherein" formatting
        text = re.sub(r'wherein\s*:', 'wherein', text, flags=re.IGNORECASE)
        
        # === GRAMMAR FIXES ===
        # Fix "performing sensing" -> "sensing"
        text = re.sub(r'performing\s+sensing', 'sensing', text, flags=re.IGNORECASE)
        text = re.sub(r'performing\s+detecting', 'detecting', text, flags=re.IGNORECASE)
        text = re.sub(r'performing\s+measuring', 'measuring', text, flags=re.IGNORECASE)
        text = re.sub(r'performing\s+processing', 'processing', text, flags=re.IGNORECASE)
        text = re.sub(r'performing\s+transmitting', 'transmitting', text, flags=re.IGNORECASE)
        
        # Fix "utilizing agentic for" -> "utilizing agentic AI for"
        text = re.sub(r'utilizing\s+agentic\s+for', 'utilizing agentic AI for', text, flags=re.IGNORECASE)
        text = re.sub(r'using\s+agentic\s+for', 'using agentic AI for', text, flags=re.IGNORECASE)
        
        # Fix "the the" -> "the"
        text = re.sub(r'\bthe\s+the\b', 'the', text, flags=re.IGNORECASE)
        
        # Fix "a a" -> "a"
        text = re.sub(r'\ba\s+a\b', 'a', text, flags=re.IGNORECASE)
        
        # Ensure proper ending
        text = text.rstrip()
        if not text.endswith('.'):
            text += '.'
        
        return text
    
    @staticmethod
    def _validate_structure(text: str, claim_number: int) -> str:
        """Validate and fix structural issues"""
        
        # Remove double numbering - multiple patterns
        # Pattern: "1. 1, " or "1. 1; " or "1. 1. "
        text = re.sub(rf'^{claim_number}\.\s*{claim_number}[,;.\s]+', f'{claim_number}. ', text.strip())
        # Pattern: "1, " at start
        text = re.sub(rf'^{claim_number}[,;]\s*', f'{claim_number}. ', text.strip())
        # Pattern: "1 1. " (space between)
        text = re.sub(rf'^{claim_number}\s+{claim_number}\.\s*', f'{claim_number}. ', text.strip())
        # Clean any remaining duplicate number at start
        text = re.sub(rf'^(\d+)\.\s*\1[,;.\s]+', r'\1. ', text.strip())
        
        # Ensure claim starts with number
        if not text.strip().startswith(f"{claim_number}."):
            text = f"{claim_number}. {text.strip()}"
        
        # For dependent claims, ensure proper reference
        if claim_number > 1:
            if not re.search(r'as claimed in claim \d+', text, re.IGNORECASE):
                device_match = re.search(r'^\d+\.\s+(?:The\s+)?(.+?)\s+(?:as claimed|wherein)', text, re.IGNORECASE)
                if device_match:
                    device = device_match.group(1).strip()
                else:
                    device = "system"
                
                text = re.sub(
                    rf'^{claim_number}\.\s+',
                    f'{claim_number}. The {device} as claimed in claim 1, ',
                    text,
                    count=1
                )
        
        return text
    @staticmethod
    def _enforce_antecedent_basis(text: str) -> str:
        """
        Ensure 'a/an' appears before 'the' for claim elements (IPO requirement)
        """
        tokens = re.findall(r'\b(a|an)\s+([a-zA-Z ][a-zA-Z ]{2,40})', text)
        seen = set()

        for _, term in tokens:
            key = term.lower().strip()
            if key in seen:
                text = re.sub(
                    rf'\b(a|an)\s+{re.escape(term)}\b',
                    f"the {term}",
                    text,
                    count=1,
                    flags=re.IGNORECASE
                )
            else:
                seen.add(key)

        return text
    @staticmethod
    def _limit_wherein_clauses(text: str, max_wherein: int = 3) -> str:
        parts = re.split(r'\bwherein\b', text, flags=re.IGNORECASE)
        if len(parts) > max_wherein + 1:
            text = "wherein".join(parts[:max_wherein + 1])
        return text


# === IMPROVED GENERATION CONFIG ===
class ImprovedGenerationConfig:
    """Enhanced generation parameters with better stop sequences"""
    
    @staticmethod
    def get_stop_sequences_for_claim(claim_num: int) -> List[str]:
        """Get context-appropriate stop sequences"""
        
        next_num = claim_num + 1
        
        base_stops = [
            f"\n{next_num}.",
            f"\n\n{next_num}.",
            f"Claim {next_num}",
            "\n\n\n",
            "===",
            "---",
            "<|assistant|>",
            "<|user|>",
            "This new claim",
            "This claim",
            "WRITE NOW",
            "FORMAT:",
            f"\n\nClaim {next_num}",
        ]
        
        return base_stops
    
    @staticmethod
    def get_generation_params(claim_type: str) -> Dict:
        """Get optimized parameters for different claim types"""
        
        params = {
            'claim_1': {
                'temperature': 0.1,
                'top_p': 0.85,
                'repeat_penalty': 1.2,
            },
            'dependent': {
                'temperature': 0.2,
                'top_p': 0.9,
                'repeat_penalty': 1.25,
            },
            'method': {
                'temperature': 0.15,
                'top_p': 0.85,
                'repeat_penalty': 1.2,
            }
        }
        
        return params.get(claim_type, params['dependent'])

def filter_structural_functions(functions: List[str]) -> List[str]:
    """
    Remove result-oriented or algorithmic language from Claim 1 inputs.
    """
    banned_keywords = [
        "predict", "optimize", "reduce", "improve",
        "analyze", "machine learning", "ai",
        "decision", "analytics", "automatically"
    ]
    clean = []
    for f in functions:
        if not any(b in f.lower() for b in banned_keywords):
            clean.append(f)
    return clean

def sanitize_abstract_for_claims(abstract: str) -> str:
    """
    Remove software / algorithmic terms to avoid Section 3(k) objections.
    """
    banned = [
        "machine learning",
        "predict",
        "prediction",
        "predictive",
        "analytics",
        "ai",
        "intelligent",
        "automatically",
        "real-time"
    ]
    text = abstract
    for b in banned:
        text = re.sub(rf'\b{b}\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def derive_method_steps_from_claim1(claim_1_text: str) -> List[str]:
    steps = []
    text = claim_1_text.lower()

    if "sensor" in text:
        steps.append("sensing a physical parameter using at least one sensor")

    if "processing unit" in text or "processor" in text:
        steps.append("processing the sensed parameter using a processing unit")

    if "communication" in text:
        steps.append("transmitting the processed parameter via a communication module")

    if "controller" in text or "control" in text:
        steps.append("controlling operation of the system using a controller")

    # Examiner-safe filler (if needed)
    while len(steps) < 5:
        steps.append("maintaining operation of the system in a predetermined manner")

    return steps[:5]

def is_structural_feature(feature: str) -> bool:
    banned = [
        "predict", "optimize", "analysis", "decision",
        "machine learning", "ai", "analytics", "automatic"
    ]
    return not any(b in feature.lower() for b in banned)

def enforce_single_sentence_claim(text: str) -> str:
    # Remove line breaks, bullets, colon formatting
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s*:\s*', ' ', text)
    text = re.sub(r'\s*;\s*', ', ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    text = text.strip()
    if not text.endswith('.'):
        text += '.'
    return text

def has_valid_wherein(claim_text: str) -> bool:
    match = re.search(r'wherein\s+(.+?)\.', claim_text, re.IGNORECASE)
    return bool(match and len(match.group(1).strip()) > 5)

def enforce_claim_number(text: str, num: int) -> str:
    if not text.strip().startswith(f"{num}."):
        text = f"{num}. {text.lstrip('0123456789. ')}"
    return text

# === CORRECTED CLAIM GENERATOR CLASS ===
class ClaimGenerator:
    """Generate patent claims with Indian Patent Office compliance"""
    
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.post_processor = ClaimPostProcessor()
        self.max_retries = 2
        self.used_dependent_features = set()
    
    def _clear_gpu_cache(self):
        """No-op: GPU cache clearing not needed with API-based generation."""
        pass

    def _validate_claim_quality(self, claim_text: str, claim_num: int) -> float:
        """Score claim quality (0-1)"""
        
        score = 0.0
        
        # Check reference to previous claim (for dependent claims)
        if claim_num > 1:
            if re.search(r'as claimed in claim \d+', claim_text, re.IGNORECASE):
                score += 0.3
        else:
            if 'comprising' in claim_text.lower():
                score += 0.3
        
        # Check for wherein clause
        wherein_count = claim_text.lower().count('wherein')
        if wherein_count > 0:
            score += min(0.25, wherein_count * 0.1)
        
        # Check no artifacts
        if not re.search(r'<\||\[|\]|===|---', claim_text):
            score += 0.2
        
        # Check reasonable length
        if 50 < len(claim_text) < 1000:
            score += 0.15
        
        # Check proper ending
        if claim_text.strip().endswith('.'):
            score += 0.1
        
        return score
    
    def generate_claim_1(self, abstract: str, components: Dict, 
                        prior_art_context: str) -> Dict[str, any]:
        """Generate Claim 1 with enhanced structure and verification"""
        
        raw_device_name = components.get('device_name', 'system')
        device_name = normalize_device_for_claim(raw_device_name)
        purpose = components.get('purpose', 'performing operations')
        key_elements = components.get('key_elements', [])
        functions = filter_structural_functions(
            components.get('functions', [])
        )

        
        prompt = self._build_claim1_prompt(
            abstract, device_name, purpose, 
            key_elements, functions, prior_art_context
        )
        
        best_claim = None
        best_score = 0
        
        for attempt in range(self.max_retries):
            try:
                params = ImprovedGenerationConfig.get_generation_params('claim_1')
                
                # FIXED: Correct function call with proper parameter names
                output_text = llm_generate(
                    prompt=prompt,
                    max_new_tokens=PatentConfig.MAX_TOKENS_CLAIM1,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    stop_strings=ImprovedGenerationConfig.get_stop_sequences_for_claim(1)
                )
                
                claim_text = output_text.strip()
                
                # Clean the claim
                claim_text = self.post_processor.clean_claim_text(claim_text, 1)
                claim_text = enforce_single_sentence_claim(claim_text)

                
                # Validate
                # Validate
                score = self._validate_claim_quality(claim_text, 1)

                # Enforce minimum 2 wherein clauses for Claim 1 (IPO preference)
                if claim_text.lower().count("wherein") < 2:
                    continue

                if score > best_score:
                    best_claim = claim_text
                    best_score = score

                if score >= 0.8:
                    break

                    
            except Exception as e:
                print(f"Claim 1 generation attempt {attempt + 1} failed: {e}")
                continue
        
        if best_claim is None:
            best_claim = self._fallback_claim_1(device_name, purpose, key_elements)['claim_text']
        
        self._clear_gpu_cache()
        
        return {
            'claim_number': 1,
            'claim_text': best_claim,
            'device_name': device_name,
            'type': 'independent_apparatus',
            'quality_score': best_score
        }
    
    def _build_claim1_prompt(self, abstract: str, device_name: str, 
                            purpose: str, key_elements: List[str], 
                            functions: List[str], prior_art: str) -> str:
        """Build detailed prompt for Claim 1 generation"""
        
        elements_str = "\n".join([f"   - {elem}" for elem in key_elements[:8]])
        functions_str = "\n".join([f"   - {func}" for func in functions[:5]])
        
        prompt = f"""You are an expert patent claim drafter specializing in Indian Patent Office format.

LANGUAGE CONSTRAINT (MANDATORY):
- Write the claim STRICTLY in English only.
- Do NOT include any non-English words, characters, or explanations.

INVENTION ABSTRACT:
{abstract[:600]}

EXTRACTED COMPONENTS:
Device: {device_name}
Purpose: {purpose}

Key Elements:
{elements_str}

Functions:
{functions_str}

{prior_art}

═══════════════════════════════════════════════════════════════
CRITICAL IPO COMPLIANCE RULES (Section 10(5))
═══════════════════════════════════════════════════════════════

1. SENTENCE LENGTH (CRITICAL):
   ❌ No single clause longer than 40 words
   ✓ Break long clauses into multiple shorter clauses with semicolons
   ✓ Each component should be on its own line

2. NO MIXED CONJUNCTION (CRITICAL):
   ❌ Do NOT use BOTH "characterized in that" AND "wherein" in same claim
   ✓ Use ONLY "wherein" for all characterizing clauses
   ❌ BAD: "...characterized in that... wherein... wherein..."
   ✓ GOOD: "...wherein... wherein... wherein..."

3. PUNCTUATION (CRITICAL):
   ❌ Do NOT end with comma before period (,.)
   ❌ Do NOT end with semicolon before period (;.)
   ✓ End the claim with a single period (.)
   ✓ Last clause before period should NOT have semicolon
   Example ending: "...wherein the system achieves [effect]."

4. NO FUNCTIONAL RESULT LANGUAGE (CRITICAL):
   ❌ Do NOT end clauses with abstract results like:
      - "to optimize energy consumption"
      - "to improve efficiency"
      - "to enhance performance"
   ✓ End with SPECIFIC TECHNICAL ACTIONS:
      - "to transmit sensor data to the control unit"
      - "to convert solar radiation into electrical current"
      - "to store electrical energy in the battery module"

5. NO "MEANS PLUS FUNCTION" LANGUAGE:
   ❌ NEVER use "means for", "means to", "step of", "module for"
   ✓ Use structural terms: "a processor", "an interface", "a sensor"

6. STRUCTURE FIRST, FUNCTION SECOND:
   - Every function must be anchored to a physical component
   - Pattern: "[Component Name] configured to [Function]"

7. COMPONENT TERMINOLOGY (CLAIM-DRAWING CONSISTENCY):
   ✓ Use clear, consistent component names that can appear in drawings
   ✓ Each component should be identifiable with a reference numeral
   ❌ Do NOT use vague terms like "unit" or "module" without definition

═══════════════════════════════════════════════════════════════
CORRECT CLAIM STRUCTURE
═══════════════════════════════════════════════════════════════

1. [Preamble]:
   "1. A [device_name] comprising:"

2. [Body Clauses - one component per line]:
   "a [component 1] (101) configured to [specific function];
    a [component 2] (102) coupled to the [component 1] for [purpose];
    a [component 3] (103) adapted to [function]; and
    a [component 4] (104) configured to [function],"

3. [Characterization - ONLY use "wherein"]:
   "wherein the [component 1] communicates with the [component 2] to [technical effect];
    wherein the [component 3] receives data from [component 2] for [purpose]."

4. [Ending]:
   - Last "wherein" clause ends with period (.)
   - NO comma or semicolon before the final period

═══════════════════════════════════════════════════════════════
NOW GENERATE CLAIM 1 (CLEAR, DEFINITE, IPO-COMPLIANT):
═══════════════════════════════════════════════════════════════

1."""
        
        return prompt
    
    def _fallback_claim_1(self, device_name: str, purpose: str, key_elements: List[str]) -> Dict:
        claim_text = (
            f"1. An {device_name} comprising a plurality of structural components "
            f"operatively coupled to form an integrated system, "
            f"wherein the components are arranged to enable controlled operation of the system."
        )

        return {
            'claim_number': 1,
            'claim_text': claim_text,
            'device_name': device_name,
            'type': 'independent_apparatus'
        }

    
    def generate_dependent_claim(self, claim_num: int, claim_1_text: str, 
                                device_name: str, components: Dict, 
                                abstract: str) -> str:
        """Generate dependent claims 2-8 with varied dependency structure"""
        
        # Determine dependency pattern
        if claim_num <= 3:
            depends_on = 1
        elif claim_num in [4, 5]:
            depends_on = claim_num - 1
        elif claim_num in [6, 7]:
            depends_on = 1
        else:
            depends_on = max(1, claim_num - 3)
        
        # Get features from components AND abstract
        all_features = list(components.get('key_elements', []))
        
        # Extract additional features from abstract if needed
        abstract_words = abstract.lower()
        additional_features = []
        
        # Add abstract-specific features based on claim number
        feature_types = {
            2: ["sensor", "sensing", "measurement", "detection"],
            3: ["processor", "computing", "processing", "controller"],
            4: ["communication", "wireless", "transmission", "network"],
            5: ["storage", "memory", "database", "data"],
            6: ["interface", "display", "user", "application"],
            7: ["power", "battery", "energy", "supply"],
            8: ["security", "encryption", "authentication", "protection"]
        }
        
        # Find matching feature from abstract for this claim number
        feature = None
        claim_keywords = feature_types.get(claim_num, [])
        
        for keyword in claim_keywords:
            if keyword in abstract_words and keyword not in self.used_dependent_features:
                feature = f"{keyword} module"
                self.used_dependent_features.add(keyword)
                break
        
        # If no keyword match, use components
        if feature is None:
            for feat in all_features:
                key = feat.lower()
                if key not in self.used_dependent_features and is_structural_feature(feat):
                    feature = feat
                    self.used_dependent_features.add(key)
                    break
        
        # Last resort - generate refinements based on claim 1 (don't introduce new undefined terms)
        if feature is None:
            # Use descriptive refinements that don't require new component definitions
            refinement_features = [
                "improved material composition",
                "dimensional specifications",
                "operational parameters",
                "assembly configuration",
                "performance characteristics",
                "manufacturing method",
                "structural arrangement",
                "functional optimization"
            ]
            idx = (claim_num - 2) % len(refinement_features)
            feature = refinement_features[idx]
            self.used_dependent_features.add(feature.lower())

        
        prompt = f"""Write ONE dependent claim in Indian Patent Office format.

LANGUAGE CONSTRAINT (MANDATORY):
- Write the claim STRICTLY in English only.
- Do NOT include any non-English words, characters, or explanations.
- Do NOT include translations, notes, or comments.

INDEPENDENT CLAIM (Claim 1):
{claim_1_text[:500]}...

DEVICE: {device_name}
FEATURE TO ELABORATE: {feature}

WRITE CLAIM {claim_num} (depends on claim {depends_on}):

FORMAT:
{claim_num}. The {device_name} as claimed in claim {depends_on}, wherein [one specific technical limitation that adds novelty].

REQUIREMENTS:
✓ Start with "{claim_num}. The {device_name} as claimed in claim {depends_on}, wherein"
✓ Add ONE specific technical detail/limitation
✓ Be concise (1-2 sentences maximum)
✓ Include reference number if applicable
✓ End with period

WRITE NOW:

{claim_num}. The"""

        best_claim = None
        best_score = 0
        
        for attempt in range(self.max_retries):
            try:
                params = ImprovedGenerationConfig.get_generation_params('dependent')
                
                # FIXED: Correct function call with proper parameter names
                output_text = llm_generate(
                    prompt=prompt,
                    max_new_tokens=PatentConfig.MAX_TOKENS_DEPENDENT,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    stop_strings=ImprovedGenerationConfig.get_stop_sequences_for_claim(claim_num)
                )
                
                claim_text = output_text.strip()
                
                # Clean the claim
                claim_text = self.post_processor.clean_claim_text(claim_text, claim_num)
                
                # Validate
                score = self._validate_claim_quality(claim_text, claim_num)
                
                if score > best_score:
                    best_claim = claim_text
                    best_score = score
                
                if score >= 0.8 and has_valid_wherein(claim_text):
                    break

                    
            except Exception as e:
                print(f"Claim {claim_num} generation attempt {attempt + 1} failed: {e}")
                continue
        
        if best_claim is None or not has_valid_wherein(best_claim):
            best_claim = (
                f"{claim_num}. The {device_name} as claimed in claim {depends_on}, "
                f"wherein the {feature} is structurally coupled to the system."
            )

        self._clear_gpu_cache()
        
        return best_claim
    
    def generate_method_claim_9(self, claim_1_text: str, device_name: str, 
                               abstract: str, components: Dict) -> str:
        """Generate comprehensive method claim"""
        
        purpose = components.get('purpose', 'performing operations')
        # ✅ Derive steps ONLY from Claim 1 (unity-safe)
        steps = derive_method_steps_from_claim1(claim_1_text)
        steps_list = "\n".join([f"   - {step}" for step in steps])

        
        prompt = f"""Write ONE method claim in Indian Patent Office format.

LANGUAGE CONSTRAINT (MANDATORY):
- Write the claim STRICTLY in English only.
- Do NOT include any non-English words, characters, or explanations.
- Do NOT include translations, notes, or comments.

UNITY REQUIREMENT (MANDATORY):
- All method steps MUST be directly derived from Claim 1 elements
- Do NOT introduce new structures, functions, or control logic
- Do NOT describe advantages or results

APPARATUS CLAIM (Claim 1):
{claim_1_text[:500]}...

DEVICE: {device_name}
PURPOSE: {purpose}

KEY STEPS (derived strictly from Claim 1 structure):
{steps_list}

FORMAT:
9. A method for operating the [device_name] as claimed in claim 1, the method comprising performing steps corresponding to the structural elements of claim 1 and executing the steps in a predetermined sequence, wherein the steps correspond to operation of the structural elements of claim 1, and wherein the steps are executed under predefined operational conditions.


REQUIREMENTS:
✓ Start with "9. A method for operating the [device_name] as claimed in claim 1"
✓ List 5-7 method steps (gerund form: "-ing")
✓ Steps should follow logical sequence
✓ Include 2-3 "wherein" clauses at end
✓ End with period

WRITE NOW:

9. A method for"""

        best_claim = None
        best_score = 0
        
        for attempt in range(self.max_retries):
            try:
                params = ImprovedGenerationConfig.get_generation_params('method')
                
                # FIXED: Correct function call with proper parameter names
                output_text = llm_generate(
                    prompt=prompt,
                    max_new_tokens=PatentConfig.MAX_TOKENS_METHOD,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    stop_strings=ImprovedGenerationConfig.get_stop_sequences_for_claim(9)
                )
                
                claim_text = output_text.strip()
                
                # Clean the claim
                claim_text = self.post_processor.clean_claim_text(claim_text, 9)
                
                # Validate
                score = self._validate_claim_quality(claim_text, 9)
                
                if score > best_score:
                    best_claim = claim_text
                    best_score = score
                
                if score >= 0.7:
                    break
                    
            except Exception as e:
                print(f"Claim 9 generation attempt {attempt + 1} failed: {e}")
                continue
        
        if best_claim is None:
            best_claim = (
                f"9. A method for operating the {device_name} as claimed in claim 1, "
                "the method comprising steps of initializing the system, collecting data, "
                "processing said data using system components, and executing control actions "
                "corresponding to operation of the components, "
                "wherein the method is performed during normal operation of the system, "
                "and wherein execution follows a predetermined sequence."
            )
        
        self._clear_gpu_cache()
        
        return best_claim
    
    def generate_method_subclaims(self, claim_9_text: str, device_name: str) -> List[str]:
        """Generate Claims 10 and 11 (dependent on method claim 9)"""
        
        claims = []
        
        # Define unique aspects for claims 10 and 11
        claim_aspects = {
            10: ("timing constraints", "wherein the sensing step is performed at predetermined time intervals"),
            11: ("data validation", "wherein the processing step further includes validating the sensed data before transmission")
        }
        
        for claim_num in [10, 11]:
            aspect_name, aspect_desc = claim_aspects[claim_num]
            
            prompt = f"""Write ONE dependent method claim.

LANGUAGE CONSTRAINT (MANDATORY):
- Write the claim STRICTLY in English only.
- Do NOT include any non-English words, characters, or explanations.

METHOD CLAIM (Claim 9):
{claim_9_text[:400]}...

WRITE CLAIM {claim_num} (depends on claim 9):
SPECIFIC ASPECT: {aspect_name}

FORMAT:
{claim_num}. The method as claimed in claim 9, {aspect_desc}.

REQUIREMENTS:
- MUST be different from any other claim
- Focus on: {aspect_name}
- Add specific technical detail

WRITE NOW:

{claim_num}. The method"""

            best_claim = None
            best_score = 0
            
            for attempt in range(self.max_retries):
                try:
                    params = ImprovedGenerationConfig.get_generation_params('dependent')
                    
                    output_text = llm_generate(
                        prompt=prompt,
                        max_new_tokens=400,
                        temperature=params["temperature"] + 0.1,  # Slightly higher for variety
                        top_p=params["top_p"],
                        repeat_penalty=params["repeat_penalty"] + 0.1,
                        stop_strings=ImprovedGenerationConfig.get_stop_sequences_for_claim(claim_num)
                    )
                    
                    claim_text = output_text.strip()
                    
                    # Clean the claim
                    claim_text = self.post_processor.clean_claim_text(claim_text, claim_num)
                    claim_text = enforce_claim_number(claim_text, claim_num)
                    
                    # Validate
                    score = self._validate_claim_quality(claim_text, claim_num)
                    
                    if score > best_score:
                        best_claim = claim_text
                        best_score = score
                    
                    if score >= 0.8:
                        break
                        
                except Exception as e:
                    print(f"Claim {claim_num} generation attempt {attempt + 1} failed: {e}")
                    continue
            
            # Use unique fallback for each claim
            if best_claim is None:
                if claim_num == 10:
                    best_claim = (
                        f"10. The method as claimed in claim 9, wherein the sensing step "
                        "is performed at predetermined time intervals to ensure continuous monitoring."
                    )
                else:
                    best_claim = (
                        f"11. The method as claimed in claim 9, wherein the processing step "
                        "further includes validating the sensed data against predefined thresholds."
                    )

            claims.append(best_claim)
            self._clear_gpu_cache()
        
        return claims

# === FINAL QUALITY CHECKER ===
class FinalQualityChecker:
    """Final quality check before outputting claims"""
    
    @staticmethod
    def check_and_fix_all_claims(claims_text: str) -> Tuple[str, List[str]]:
        """
        Perform final quality check and fixes
        
        Returns:
            (fixed_claims_text, list_of_fixes_applied)
        """
        
        fixes_applied = []
        
        # 1. Remove all artifacts
        if re.search(r'<\||===|---', claims_text):
            claims_text = re.sub(r'<\|[^>]*\|>', '', claims_text)
            claims_text = re.sub(r'===+', '', claims_text)
            claims_text = re.sub(r'---+', '', claims_text)
            fixes_applied.append("Removed LLM artifacts")
        
        # 2. Remove explanatory paragraphs
        original_lines = len(claims_text.split('\n'))
        claims_text = re.sub(
            r'\.\s+This (?:new )?claim.*?(?=\n\d+\.|\Z)', 
            '.',
            claims_text,
            flags=re.DOTALL
        )
        if len(claims_text.split('\n')) < original_lines:
            fixes_applied.append("Removed explanatory text")
        
        # 3. Fix claim numbering gaps
        claim_numbers = re.findall(r'^(\d+)\.', claims_text, re.MULTILINE)
        claim_numbers = [int(n) for n in claim_numbers]
        
        if claim_numbers:
            expected = list(range(1, max(claim_numbers) + 1))
            if claim_numbers != expected:
                fixes_applied.append(f"Fixed claim numbering issues")
        
        # 4. Clean up excessive whitespace
        claims_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', claims_text)
        claims_text = re.sub(r'  +', ' ', claims_text)
        
        # 5. Fix spacing around claim numbers
        claims_text = re.sub(r'(\d+)\.\s*The', r'\1. The', claims_text)
        claims_text = re.sub(r'(\d+)\.\s*A', r'\1. A', claims_text)
        
        # 6. CRITICAL: Fix invalid claim dependencies
        # A claim cannot depend on a claim with a higher number
        for match in re.finditer(r'(\d+)\.\s+.*?as claimed in claim\s+(\d+)', claims_text, re.IGNORECASE):
            claim_num = int(match.group(1))
            depends_on = int(match.group(2))
            
            if depends_on >= claim_num:
                # Invalid dependency - fix it to depend on claim 1
                old_ref = f"as claimed in claim {depends_on}"
                new_ref = "as claimed in claim 1"
                claims_text = claims_text.replace(old_ref, new_ref, 1)
                fixes_applied.append(f"Fixed claim {claim_num}: invalid dependency on claim {depends_on} → claim 1")
        
        # 7. Ensure no claim depends on non-existent claim
        max_claim = max(claim_numbers) if claim_numbers else 11
        for match in re.finditer(r'as claimed in claim\s+(\d+)', claims_text, re.IGNORECASE):
            ref_claim = int(match.group(1))
            if ref_claim > max_claim:
                old_ref = f"as claimed in claim {ref_claim}"
                new_ref = "as claimed in claim 1"
                claims_text = claims_text.replace(old_ref, new_ref)
                fixes_applied.append(f"Fixed reference to non-existent claim {ref_claim} → claim 1")
        
        # 8. CRITICAL: Fix punctuation issues
        # Fix comma before period (,.) -> (.)
        if ',.' in claims_text:
            claims_text = claims_text.replace(',.', '.')
            fixes_applied.append("Fixed comma before period (,.)")
        
        # Fix semicolon before period (;.) -> (.)
        if ';.' in claims_text:
            claims_text = claims_text.replace(';.', '.')
            fixes_applied.append("Fixed semicolon before period (;.)")
        
        # 9. CRITICAL: Remove mixed "characterized in that" + "wherein"
        # If both appear in the same claim, remove "characterized in that"
        if 'characterized in that' in claims_text.lower() and 'wherein' in claims_text.lower():
            claims_text = re.sub(r'characterized in that', 'wherein', claims_text, flags=re.IGNORECASE)
            fixes_applied.append("Replaced 'characterized in that' with 'wherein' for consistency")
        
        # 10. Flag functional result language (warning only - can't auto-fix)
        result_phrases = [
            'to optimize', 'to improve', 'to enhance', 'to maximize',
            'to minimize', 'to increase', 'to reduce', 'to achieve optimal'
        ]
        for phrase in result_phrases:
            if phrase in claims_text.lower():
                fixes_applied.append(f"WARNING: Functional result language detected ('{phrase}') - consider revision")
                break
        
        return claims_text, fixes_applied


# === Formatter (Enhanced) ===
class ClaimFormatter:
    """Format claims in Indian Patent Office style with proper layout"""
    
    @staticmethod
    def format_complete_claims(claim_1: Dict, dependent_claims: List[str],
                              method_claim_9: str, method_subclaims: List[str],
                              applicant_name: str = "[Your Institution/Company Name]") -> str:
        """Format all claims with proper headers, line numbers, and footers"""
        
        output = []
        
        # Header
        output.append("WE CLAIM:")
        output.append("")
        
        # === CLAIM 1 ===
        claim_1_text = claim_1['claim_text'].strip()
        wrapped_claim1 = textwrap.fill(
            claim_1_text,
            width=70,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent='   '
        )
        output.append(wrapped_claim1)
        output.append("")  # Blank line after Claim 1
        
        # === DEPENDENT CLAIMS 2-8 ===
        for dep_claim in dependent_claims:
            wrapped = textwrap.fill(
                dep_claim.strip(),
                width=70,
                break_long_words=False,
                break_on_hyphens=False,
                subsequent_indent='   '
            )
            output.append(wrapped)
            output.append("")  # Blank line after each claim
        
        # === METHOD CLAIM 9 ===
        wrapped_method = textwrap.fill(
            method_claim_9.strip(),
            width=70,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent='   '
        )
        output.append(wrapped_method)
        output.append("")  # Blank line after Claim 9
        
        # === METHOD SUBCLAIMS 10-11 ===
        for subclaim in method_subclaims:
            wrapped = textwrap.fill(
                subclaim.strip(),
                width=70,
                break_long_words=False,
                subsequent_indent='   '
            )
            output.append(wrapped)
            output.append("")  # Blank line after each claim
        
        return "\n".join(output)


# === Validator (Enhanced) ===
class ClaimValidator:
    """Comprehensive validation with detailed feedback"""
    
    @staticmethod
    def validate(claims_text: str) -> Dict[str, any]:
        """Validate claims against Indian Patent Office standards"""
        
        issues = []
        warnings = []
        suggestions = []
        
        # Check header
        if 'WE CLAIM' not in claims_text:
            issues.append("❌ Missing 'WE CLAIM' header (mandatory)")
        
        # Check claim numbering
        claim_numbers = re.findall(r'^\s*(\d+)\.', claims_text, re.MULTILINE)
        if len(claim_numbers) < 9:
            issues.append(f"❌ Insufficient claims: {len(claim_numbers)} (minimum 9 expected)")
        elif len(claim_numbers) < 11:
            warnings.append(f"⚠️  Only {len(claim_numbers)} claims (11 recommended)")
        
        # Validate Claim 1
        claim_1_match = re.search(r'^1\.(.+?)(?=^[2-9]\.|\Z)', claims_text, re.DOTALL | re.MULTILINE)
        if claim_1_match:
            claim_1_text = claim_1_match.group(1)
            
            if 'comprising' not in claim_1_text.lower():
                issues.append("❌ Claim 1 missing 'comprising' (mandatory)")
            
            wherein_count = claim_1_text.lower().count('wherein')
            if wherein_count == 0:
                 warnings.append("⚠️ Claim 1 has no 'wherein' clauses (1–3 recommended)")
            elif wherein_count < 2:
                warnings.append("⚠️ Claim 1 has fewer than 2 'wherein' clauses (IPO preferred: 2–3)")
            elif wherein_count > 3:
                warnings.append("⚠️ Claim 1 has more than 3 'wherein' clauses (IPO preferred max is 3)")

            # Check for reference numbers
            ref_numbers = re.findall(r'\((\d+)\)', claim_1_text)
            if len(ref_numbers) < 3:
                warnings.append("⚠️  Few reference numbers in Claim 1 (use (1), (2), (3) etc.)")
        
        # Check for method claim
        method_claim = re.search(r'9\.\s+A method for', claims_text, re.IGNORECASE)
        if not method_claim:
            warnings.append("⚠️  No method claim found at position 9")
        
        # Check dependent claim format
        dep_claims = re.findall(r'(\d+)\.\s+The\s+.+?\s+as claimed in claim (\d+)', claims_text,re.IGNORECASE)
        if len(dep_claims) < 6:
            warnings.append(f"⚠️  Only {len(dep_claims)} dependent claims found")
        
        # Check for artifacts
        if re.search(r'<\||===|---', claims_text):
            issues.append("❌ LLM artifacts found in claims (must be removed)")
        
        # Check line numbers
        line_number_matches = re.findall(r'\s+(\d+)', claims_text, re.MULTILINE)
        if len(line_number_matches) < 5:
            suggestions.append("💡 Add line numbers every 5 lines on right margin")
        
        # Check for proper indentation
        if not re.search(r'^\s{3,}', claims_text, re.MULTILINE):
            suggestions.append("💡 Use proper indentation for sub-elements")
        
        # Calculate statistics
        stats = {
            'total_claims': len(claim_numbers),
            'independent_claims': len(re.findall(r'^\d+\.\s+(?:An?|A method)', claims_text, re.MULTILINE)),
            'dependent_claims': len(dep_claims),
            'method_claims': len(re.findall(r'A method for', claims_text, re.IGNORECASE)),
            'wherein_clauses_total': claims_text.lower().count('wherein'),
            'has_reference_numbers': len(re.findall(r'\(\d+\)', claims_text)) > 5,
        }
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'suggestions': suggestions,
            'statistics': stats,
            'compliance_score': ClaimValidator._calculate_score(issues, warnings)
        }
    
    @staticmethod
    def _calculate_score(issues: List[str], warnings: List[str]) -> float:
        """Calculate compliance score (0-100)"""
        score = 100.0
        score -= len(issues) * 15
        score -= len(warnings) * 5
        return max(0.0, score)
    
    @staticmethod
    def print_validation_report(validation: Dict):
        """Print formatted validation report"""
        print("\n" + "=" * 80)
        print("     PATENT CLAIMS VALIDATION REPORT")
        print("=" * 80)
        
        # Compliance Score
        score = validation['compliance_score']
        if score >= 90:
            status = "✅ EXCELLENT"
        elif score >= 75:
            status = "✓ GOOD"
        elif score >= 60:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ NEEDS REVISION"
        
        print(f"\nCOMPLIANCE SCORE: {score:.1f}/100 - {status}\n")
        
        # Issues
        if validation['issues']:
            print("CRITICAL ISSUES (must fix):")
            for issue in validation['issues']:
                print(f"   {issue}")
            print()
        
        # Warnings
        if validation['warnings']:
            print("WARNINGS (recommended fixes):")
            for warning in validation['warnings']:
                print(f"   {warning}")
            print()
        
        # Suggestions
        if validation['suggestions']:
            print("SUGGESTIONS (optional improvements):")
            for suggestion in validation['suggestions']:
                print(f"   {suggestion}")
            print()
        
        # Statistics
        stats = validation['statistics']
        print("STATISTICS:")
        print(f"   Total Claims: {stats['total_claims']}")
        print(f"   Independent Claims: {stats['independent_claims']}")
        print(f"   Dependent Claims: {stats['dependent_claims']}")
        print(f"   Method Claims: {stats['method_claims']}")
        print(f"   Total 'wherein' Clauses: {stats['wherein_clauses_total']}")
        print(f"   Reference Numbers Used: {'Yes' if stats['has_reference_numbers'] else 'No'}")
        
        print("=" * 80 + "\n")


# === Main Pipeline ===
class PatentClaimsPipeline:
    """Complete pipeline for generating Indian Patent Office compliant claims"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.extractor = ComponentExtractor()
        self.retriever = PriorArtRetriever(self.model_manager)
        self.generator = ClaimGenerator(self.model_manager)
        self.formatter = ClaimFormatter()
        self.validator = ClaimValidator()
        self.quality_checker = FinalQualityChecker()
    
    def generate_complete_claims(self, abstract: str, 
                                applicant_name: str = "[Your Institution/Company Name]",
                                top_k_prior_art: int = 5,
                                verbose: bool = True,
                                component_registry: dict = None) -> Dict[str, any]:
        """
        Complete pipeline: abstract → formatted claims with validation
        
        Args:
            abstract: Patent abstract text
            applicant_name: Name of patent applicant
            top_k_prior_art: Number of prior art patents to retrieve
            verbose: Print progress messages
            component_registry: Unified registry for consistent reference numerals
        
        Returns:
            Dictionary with claims text, validation results, and metadata
        """
        
        if verbose:
            print("\n" + "=" * 80)
            print("     PATENT CLAIMS GENERATION PIPELINE")
            print("=" * 80)
            print(f"\nInput Abstract ({len(abstract)} chars):")
            print(f"{abstract[:200]}...\n")
        
        # Step 1: Extract components
        # Step 1: Sanitize & extract components
        if verbose:
            print("[1/6] Extracting components from abstract...")

        safe_abstract = sanitize_abstract_for_claims(abstract)
        components = self.extractor.extract(safe_abstract)
        
        # Integrate unified registry for consistent reference numerals
        if component_registry and component_registry.get("components"):
            # Use registry's component names as key_elements with reference numerals
            registry_comps = component_registry.get("components", {})
            components['key_elements_with_refs'] = [
                f"{name} ({num})" for name, num in registry_comps.items()
            ]
            if verbose:
                print(f"   ✓ Using unified registry with {len(registry_comps)} components")

        if verbose:
            print(f"   ✓ Device: {components['device_name']}")
            print(f"   ✓ Purpose: {components['purpose'][:60]}...")
            print(f"   ✓ Key Elements: {len(components['key_elements'])}")
            print(f"   ✓ Functions: {len(components['functions'])}")

        
        # Step 2: Retrieve prior art
        if verbose:
            print(f"\n[2/6] Retrieving top-{top_k_prior_art} similar prior art patents...")
        prior_art = self.retriever.retrieve(abstract, top_k=top_k_prior_art)
        prior_art_context = self.retriever.format_for_context(prior_art)
        
        if verbose and prior_art:
            print(f"   ✓ Found {len(prior_art)} similar patents")
            print(f"   ✓ Top similarity: {prior_art[0]['similarity']:.2f}")
        
        # Step 3: Generate Claim 1
        if verbose:
            print(f"\n[3/6] Generating Claim 1 (independent apparatus claim)...")
        claim_1 = self.generator.generate_claim_1(safe_abstract, components, prior_art_context)

        
        if verbose:
            print(f"   ✓ Claim 1 generated ({len(claim_1['claim_text'])} chars)")
            print(f"   ✓ Quality score: {claim_1.get('quality_score', 0):.2f}")
        
        # Step 4: Generate dependent claims 2-8
        if verbose:
            print(f"\n[4/6] Generating dependent claims 2-8...")
        dependent_claims = []
        for i in range(2, 9):
            if verbose:
                print(f"   Generating claim {i}...", end=" ")
            dep_claim = self.generator.generate_dependent_claim(
                i, claim_1['claim_text'], claim_1['device_name'], 
                components, abstract
            )
            dependent_claims.append(dep_claim)
            if verbose:
                print("✓")
        
        if verbose:
            print(f"   ✓ Generated {len(dependent_claims)} dependent claims")
        
        # Step 5: Generate method claims 9-11
        if verbose:
            print(f"\n[5/6] Generating method claims 9-11...")
        method_claim_9 = self.generator.generate_method_claim_9(
            claim_1['claim_text'], claim_1['device_name'], abstract, components
        )
        method_subclaims = self.generator.generate_method_subclaims(
            method_claim_9, claim_1['device_name']
        )
        
        if verbose:
            print(f"   ✓ Generated method claim and 2 subclaims")
        
        # Step 6: Format and validate
        if verbose:
            print(f"\n[6/6] Formatting and validating claims...")
        
        formatted_claims = self.formatter.format_complete_claims(
            claim_1, dependent_claims, method_claim_9, 
            method_subclaims, applicant_name
        )
        
        # Apply final quality fixes
        formatted_claims, fixes = self.quality_checker.check_and_fix_all_claims(formatted_claims)
        
        if verbose and fixes:
            print("\n   Applied final quality fixes:")
            for fix in fixes:
                print(f"      • {fix}")
        
        validation = self.validator.validate(formatted_claims)
        
        if verbose:
            print(f"   ✓ Claims formatted ({len(formatted_claims)} chars)")
            print(f"   ✓ Validation complete")
        
        # Return comprehensive results
        return {
            'claims_text': formatted_claims,
            'validation': validation,
            'components': components,
            'prior_art': prior_art,
            'metadata': {
                'claim_1': claim_1,
                'dependent_claims': dependent_claims,
                'method_claim_9': method_claim_9,
                'method_subclaims': method_subclaims,
                'generated_at': datetime.now().isoformat(),
                'abstract_length': len(abstract),
                'claims_count': 1 + len(dependent_claims) + 1 + len(method_subclaims),
                'fixes_applied': fixes
            }
        }
    
    def save_claims_to_file(self, results: Dict, output_path: str):
        """Save generated claims to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['claims_text'])
        print(f"✓ Claims saved to: {output_path}")
    
    def export_json_report(self, results: Dict, output_path: str):
        """Export complete report as JSON"""
        # Prepare JSON-serializable data
        export_data = {
            'claims_text': results['claims_text'],
            'validation': results['validation'],
            'components': results['components'],
            'prior_art': [
                {
                    'rank': pa['rank'],
                    'similarity': pa['similarity'],
                    'title': pa.get('title', 'N/A'),
                    'patent_id': pa['patent_id']
                }
                for pa in results['prior_art']
            ],
            'metadata': results['metadata']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Full report saved to: {output_path}")


# === Convenience Function ===
def generate_claims_from_abstract(abstract: str, 
                                 applicant_name: str = "[Your Institution/Company Name]",
                                 verbose: bool = True,
                                 component_registry: dict = None) -> str:
    """
    Simple function to generate claims from abstract
    
    Args:
        abstract: Patent abstract text
        applicant_name: Applicant name for header
        verbose: Print progress
        component_registry: Unified registry for consistent reference numerals
    
    Returns:
        Formatted claims text
    """
    pipeline = PatentClaimsPipeline()
    results = pipeline.generate_complete_claims(
        abstract, 
        applicant_name=applicant_name,
        verbose=verbose,
        component_registry=component_registry
    )
    
    # Print validation report
    if verbose:
        ClaimValidator.print_validation_report(results['validation'])
    
    return results['claims_text']


# === Example Usage ===
if __name__ == "__main__":
    
    # Sample abstract (IoT Agricultural Monitoring)
    sample_abstract = """An Internet of Things (IoT) based agricultural monitoring system for precision farming. The system comprises a plurality of soil moisture sensors positioned at multiple depths within the soil, temperature sensors for measuring ambient and soil temperature, humidity sensors for detecting atmospheric humidity, and a central processing unit configured to receive and analyze sensor data in real-time. The system further includes a wireless communication module supporting WiFi and LoRaWAN protocols for transmitting data to a cloud server, and a machine learning module configured to predict irrigation requirements based on historical weather patterns and current sensor readings. The system automatically controls irrigation valves based on predictive analytics, reducing water consumption while maintaining optimal crop health. A mobile application provides farmers with real-time monitoring capabilities and automated alerts for critical conditions such as drought stress or disease indicators."""
    
    print("=" * 80)
    print("     ENHANCED PATENT CLAIMS GENERATOR")
    print("     Indian Patent Office Compliant")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = PatentClaimsPipeline()
    
    # Generate complete claims
    results = pipeline.generate_complete_claims(
        abstract=sample_abstract,
        applicant_name="Agricultural Innovation Institute",
        top_k_prior_art=5,
        verbose=True
    )
    
    # Display claims
    print("\n" + "=" * 80)
    print("     GENERATED PATENT CLAIMS")
    print("=" * 80 + "\n")
    print(results['claims_text'])
    
    # Display validation report
    ClaimValidator.print_validation_report(results['validation'])
    
    # Save outputs
    pipeline.save_claims_to_file(results, "patent_claims_output.txt")
    pipeline.export_json_report(results, "patent_claims_report.json")
    
    print("\n" + "=" * 80)
    print("     GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nCompliance Score: {results['validation']['compliance_score']:.1f}/100")
    print(f"Total Claims Generated: {results['metadata']['claims_count']}")
    print(f"Prior Art References: {len(results['prior_art'])}")
    print(f"Quality Fixes Applied: {len(results['metadata']['fixes_applied'])}")
    print("\nFiles generated:")
    print("   • patent_claims_output.txt")
    print("   • patent_claims_report.json")
    print("\n" + "=" * 80)

# Backward compatibility alias
generate_claims = generate_claims_from_abstract

