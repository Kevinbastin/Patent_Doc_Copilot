import os
import faiss
import json
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import re
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# === Configuration ===
class PatentConfig:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LLM_PATH = os.path.join(BASE_DIR, "models", "models", "phi-3-mini-4k-instruct-q4.gguf")
    INDEX_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss.index")
    METADATA_PATH = os.path.join(BASE_DIR, "data", "bigpatent_tiny", "faiss_metadata.json")
    
    # Generation parameters
    TEMPERATURE = 0.15
    TOP_P = 0.85
    REPEAT_PENALTY = 1.15
    MAX_TOKENS_CLAIM1 = 2048
    MAX_TOKENS_DEPENDENT = 256
    MAX_TOKENS_METHOD = 768


# === Load models ONCE (singleton pattern) ===
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.llm = Llama(
                model_path=PatentConfig.LLM_PATH, 
                n_gpu_layers=-1,  # Use GPU if available
                n_ctx=8192, 
                n_threads=4,
                verbose=False
            )
            cls._instance.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            cls._instance.index = faiss.read_index(PatentConfig.INDEX_PATH)
            with open(PatentConfig.METADATA_PATH, "r") as f:
                cls._instance.metadata = json.load(f)
        return cls._instance


# === Component Extraction (Enhanced) ===
class ComponentExtractor:
    """Extract structured components from patent abstract using multiple strategies"""
    
    COMPONENT_PATTERNS = [
        r'\b(\w+\s+(?:module|unit|sensor|system|node|server|interface|device|structure|'
        r'controller|processor|engine|detector|emitter|absorber|condenser|line|tube|'
        r'pipe|valve|circuit|mechanism|assembly|apparatus|means|element|component))\b'
    ]
    
    DEVICE_PATTERNS = [
        r'(?:A|An|The)\s+([^,]{15,80}?)\s+(?:comprising|including|having|for|that|which)',
        r'(?:present invention relates to|invention provides|disclosed is)\s+(?:a|an)\s+([^,]{15,80}?)(?:\s+comprising|\s+for|\s+that)',
        r'(?:system|apparatus|device|method)\s+for\s+([^,]{15,80}?)(?:\s+comprising|\s+including)',
    ]
    
    PURPOSE_PATTERNS = [
        r'(?:system|device|apparatus|method)\s+for\s+([^,\.]{15,100})',
        r'configured to\s+([^,\.]{15,80})',
        r'adapted to\s+([^,\.]{15,80})',
        r'operable to\s+([^,\.]{15,80})',
    ]
    
    @classmethod
    def extract(cls, abstract: str) -> Dict[str, any]:
        """Extract all key components with confidence scoring"""
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
        
        # Extract device name with confidence
        for pattern in cls.DEVICE_PATTERNS:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                device_name = match.group(1).strip()
                confidence = len(device_name) / 80.0
                if confidence > components['device_confidence']:
                    components['device_name'] = device_name
                    components['device_confidence'] = confidence
        
        # Extract purpose
        for pattern in cls.PURPOSE_PATTERNS:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                purpose = match.group(1).strip()
                if len(purpose) > len(components['purpose']):
                    components['purpose'] = purpose
                    components['purpose_confidence'] = min(1.0, len(purpose) / 80.0)
        
        # Extract components from "comprising" clauses
        comprising_matches = re.finditer(
            r'(?:comprising|including|having)\s+([^\.]+?)(?:\.|;|and wherein)',
            abstract,
            re.IGNORECASE
        )
        
        seen_elements = set()
        for match in comprising_matches:
            comprising_text = match.group(1)
            parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', comprising_text)
            for part in parts:
                part = part.strip()
                if len(part) > 10 and part.lower() not in seen_elements:
                    components['key_elements'].append(part)
                    seen_elements.add(part.lower())
        
        # Extract component keywords
        for pattern in cls.COMPONENT_PATTERNS:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            for comp in matches:
                comp_clean = comp.strip().lower()
                if comp_clean not in seen_elements and len(comp_clean) > 5:
                    components['key_elements'].append(comp.strip())
                    seen_elements.add(comp_clean)
        
        # Extract functions
        function_matches = re.findall(
            r'(?:configured|operable|adapted|designed|arranged)\s+to\s+([^,\.]{10,80})',
            abstract,
            re.IGNORECASE
        )
        components['functions'] = [f.strip() for f in function_matches]
        
        # Extract technical effects
        effect_keywords = ['reducing', 'increasing', 'improving', 'enhancing', 
                          'minimizing', 'maximizing', 'optimizing', 'enabling']
        for keyword in effect_keywords:
            pattern = rf'{keyword}\s+([^,\.]+)'
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            components['technical_effects'].extend([m.strip() for m in matches])
        
        # Extract novelty indicators
        novelty_keywords = ['novel', 'new', 'improved', 'innovative', 'unique', 
                           'first', 'unlike', 'superior', 'advantageous']
        for keyword in novelty_keywords:
            if re.search(rf'\b{keyword}\b', abstract, re.IGNORECASE):
                components['novelty_indicators'].append(keyword)
        
        return components


# === Enhanced Prior Art Retrieval ===
class PriorArtRetriever:
    """Retrieve and analyze prior art with relevance scoring"""
    
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
    
    def retrieve(self, abstract: str, top_k: int = 5) -> List[Dict[str, any]]:
        """Retrieve top-k most relevant prior art with metadata"""
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
    def clean_claim_text(raw_text: str, claim_number: int) -> str:
        """Clean and normalize claim text with multiple validation passes"""
        
        # Step 1: Remove LLM artifacts
        cleaned = ClaimPostProcessor._remove_llm_artifacts(raw_text)
        
        # Step 2: Extract only the relevant claim
        cleaned = ClaimPostProcessor._extract_target_claim(cleaned, claim_number)
        
        # Step 3: Remove explanatory text
        cleaned = ClaimPostProcessor._remove_explanations(cleaned)
        
        # Step 4: Fix formatting issues
        cleaned = ClaimPostProcessor._fix_formatting(cleaned)
        
        # Step 5: Validate claim structure
        cleaned = ClaimPostProcessor._validate_structure(cleaned, claim_number)
        
        return cleaned
    
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
        
        pattern = rf'^{claim_number}\.\s+(.+?)(?=^\d+\.|===|\Z)'
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
        text = re.sub(r'  +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        
        # Fix punctuation
        text = re.sub(r'\s+([,;.])', r'\1', text)
        text = re.sub(r'([,;.])\s*([,;.])', r'\1', text)
        
        # Fix "wherein" formatting
        text = re.sub(r'wherein\s*:', 'wherein', text, flags=re.IGNORECASE)
        
        # Ensure proper ending
        text = text.rstrip()
        if not text.endswith('.'):
            text += '.'
        
        return text
    
    @staticmethod
    def _validate_structure(text: str, claim_number: int) -> str:
        """Validate and fix structural issues"""
        
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
            f"{next_num}. The",
            f"{next_num}. A",
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


# === ENHANCED CLAIM GENERATOR ===
class ClaimGenerator:
    """Generate patent claims with Indian Patent Office compliance"""
    
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.llm = model_manager.llm
        self.post_processor = ClaimPostProcessor()
        self.max_retries = 2
    
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
        
        device_name = components.get('device_name', 'system')
        purpose = components.get('purpose', 'performing operations')
        key_elements = components.get('key_elements', [])
        functions = components.get('functions', [])
        
        prompt = self._build_claim1_prompt(
            abstract, device_name, purpose, 
            key_elements, functions, prior_art_context
        )
        
        best_claim = None
        best_score = 0
        
        for attempt in range(self.max_retries):
            try:
                params = ImprovedGenerationConfig.get_generation_params('claim_1')
                
                output = self.llm(
                    prompt=prompt,
                    max_tokens=PatentConfig.MAX_TOKENS_CLAIM1,
                    **params,
                    stop=ImprovedGenerationConfig.get_stop_sequences_for_claim(1)
                )
                
                claim_text = "1." + output["choices"][0]["text"].strip()
                
                # Clean the claim
                claim_text = self.post_processor.clean_claim_text(claim_text, 1)
                
                # Validate
                score = self._validate_claim_quality(claim_text, 1)
                
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

TASK: Write Claim 1 in EXACT Indian Patent Office format.

MANDATORY STRUCTURE:
1. [Preamble]: "An [device_name] for [purpose], comprising:"
2. [Body]: List of elements with reference numbers (1), (2), etc.
   - Use hierarchical indentation for sub-components
   - Each element should have functional description
   - Use "configured to", "operable to", "adapted to"
3. [Wherein Clauses]: At least 3 "wherein" clauses describing:
   - Technical relationships between components
   - Operational characteristics
   - Functional advantages
4. [Ending]: Must end with period

CRITICAL REQUIREMENTS:
✓ Use reference numbers: (1), (2), (3), etc.
✓ Proper indentation for sub-components
✓ "comprising:" after preamble
✓ "wherein" clauses at end (minimum 3)
✓ "and" before last wherein clause
✓ End with period "."
✓ Be technically specific and detailed
✓ Include functional relationships

NOW GENERATE CLAIM 1 FOR THE GIVEN INVENTION:

1."""
        
        return prompt
    
    def _fallback_claim_1(self, device_name: str, purpose: str, 
                         key_elements: List[str]) -> Dict:
        """Generate fallback Claim 1 if generation fails"""
        
        elements_text = ""
        for i, elem in enumerate(key_elements[:5], 1):
            elements_text += f"   {elem} ({i}),\n"
        
        claim_text = f"""1. {device_name.capitalize()} for {purpose}, comprising:
{elements_text}   a processing unit (6) configured to control said components,
   a communication module (7) configured to transmit data,
   and
   a power supply unit (8),
   wherein the processing unit (6) is configured to execute control algorithms,
   wherein the communication module (7) enables data exchange between components,
   and
   wherein the system operates autonomously to achieve the intended purpose."""
        
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
        
        # Select feature to elaborate
        all_features = (
            components.get('key_elements', []) + 
            components.get('functions', []) + 
            components.get('technical_effects', [])
        )
        
        if all_features:
            feature = all_features[(claim_num - 2) % len(all_features)]
        else:
            feature = "component"
        
        prompt = f"""Write ONE dependent claim in Indian Patent Office format.

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
                
                output = self.llm(
                    prompt=prompt,
                    max_tokens=PatentConfig.MAX_TOKENS_DEPENDENT,
                    **params,
                    stop=ImprovedGenerationConfig.get_stop_sequences_for_claim(claim_num)
                )
                
                claim_text = f"{claim_num}. The" + output["choices"][0]["text"].strip()
                
                # Clean the claim
                claim_text = self.post_processor.clean_claim_text(claim_text, claim_num)
                
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
        
        if best_claim is None:
            best_claim = f"{claim_num}. The {device_name} as claimed in claim {depends_on}, wherein the {feature} provides enhanced functionality."
        
        return best_claim
    
    def generate_method_claim_9(self, claim_1_text: str, device_name: str, 
                               abstract: str, components: Dict) -> str:
        """Generate comprehensive method claim"""
        
        purpose = components.get('purpose', 'performing operations')
        functions = components.get('functions', [])
        
        functions_text = "\n".join([f"   - {func}" for func in functions[:5]])
        
        prompt = f"""Write ONE method claim in Indian Patent Office format.

APPARATUS CLAIM (Claim 1):
{claim_1_text[:500]}...

DEVICE: {device_name}
PURPOSE: {purpose}

KEY FUNCTIONS:
{functions_text}

WRITE CLAIM 9 (independent method claim):

FORMAT:
9. A method for [purpose] using the [device_name] as claimed in claim 1, the method comprising:
   [step 1: data collection/initialization]
   [step 2: processing/analyzing]
   [step 3: decision making]
   [step 4: action/control]
   [step 5: output/communication]
   wherein [condition about data flow or timing],
   and
   wherein [condition about user interaction or final outcome].

REQUIREMENTS:
✓ Start with "9. A method for [purpose] using the [device_name] as claimed in claim 1"
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
                
                output = self.llm(
                    prompt=prompt,
                    max_tokens=PatentConfig.MAX_TOKENS_METHOD,
                    **params,
                    stop=ImprovedGenerationConfig.get_stop_sequences_for_claim(9)
                )
                
                claim_text = "9. A method for" + output["choices"][0]["text"].strip()
                
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
            best_claim = f"9. A method for using the {device_name} as claimed in claim 1, the method comprising steps of initializing the system, collecting data, processing said data, making decisions based on analysis, and executing control actions, wherein the method operates in real-time, and wherein user interaction is provided through an interface."
        
        return best_claim
    
    def generate_method_subclaims(self, claim_9_text: str, device_name: str) -> List[str]:
        """Generate Claims 10 and 11 (dependent on method claim 9)"""
        
        claims = []
        
        for claim_num in [10, 11]:
            prompt = f"""Write ONE dependent method claim.

METHOD CLAIM (Claim 9):
{claim_9_text[:400]}...

WRITE CLAIM {claim_num} (depends on claim 9):

FORMAT:
{claim_num}. The method as claimed in claim 9, wherein [specific procedural detail, condition, or algorithm].

WRITE NOW:

{claim_num}. The method"""

            best_claim = None
            best_score = 0
            
            for attempt in range(self.max_retries):
                try:
                    params = ImprovedGenerationConfig.get_generation_params('dependent')
                    
                    output = self.llm(
                        prompt=prompt,
                        max_tokens=300,
                        **params,
                        stop=ImprovedGenerationConfig.get_stop_sequences_for_claim(claim_num)
                    )
                    
                    claim_text = f"{claim_num}. The method" + output["choices"][0]["text"].strip()
                    
                    # Clean the claim
                    claim_text = self.post_processor.clean_claim_text(claim_text, claim_num)
                    
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
            
            if best_claim is None:
                best_claim = f"{claim_num}. The method as claimed in claim 9, wherein the processing includes optimizing parameters based on real-time conditions."
            
            claims.append(best_claim)
        
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
        output.append("WE CLAIM")
        output.append("")
        
        line_counter = 1
        
        # === CLAIM 1 ===
        claim_1_lines = claim_1['claim_text'].split('\n')
        for line in claim_1_lines:
            if line_counter % 5 == 0:
                output.append(f"{line:<70}{line_counter:>5}")
            else:
                output.append(line)
            line_counter += 1
        output.append("")
        
        # === DEPENDENT CLAIMS 2-8 ===
        for dep_claim in dependent_claims:
            wrapped = textwrap.fill(
                dep_claim,
                width=70,
                break_long_words=False,
                break_on_hyphens=False,
                subsequent_indent='   '
            )
            
            for line in wrapped.split('\n'):
                if line_counter % 5 == 0:
                    output.append(f"{line:<70}{line_counter:>5}")
                else:
                    output.append(line)
                line_counter += 1
            output.append("")
        
        # === METHOD CLAIM 9 ===
        for line in method_claim_9.split('\n'):
            if line_counter % 5 == 0:
                output.append(f"{line:<70}{line_counter:>5}")
            else:
                output.append(line)
            line_counter += 1
        output.append("")
        
        # === METHOD SUBCLAIMS 10-11 ===
        for subclaim in method_subclaims:
            wrapped = textwrap.fill(
                subclaim,
                width=70,
                break_long_words=False,
                subsequent_indent='   '
            )
            
            for line in wrapped.split('\n'):
                if line_counter % 5 == 0:
                    output.append(f"{line:<70}{line_counter:>5}")
                else:
                    output.append(line)
                line_counter += 1
            output.append("")
        
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
                issues.append("❌ Claim 1 has no 'wherein' clauses (minimum 2 required)")
            elif wherein_count < 2:
                warnings.append(f"⚠️  Claim 1 has only {wherein_count} 'wherein' clause (3-5 recommended)")
            
            # Check for reference numbers
            ref_numbers = re.findall(r'\((\d+)\)', claim_1_text)
            if len(ref_numbers) < 3:
                warnings.append("⚠️  Few reference numbers in Claim 1 (use (1), (2), (3) etc.)")
        
        # Check for method claim
        method_claim = re.search(r'9\.\s+A method for', claims_text, re.IGNORECASE)
        if not method_claim:
            warnings.append("⚠️  No method claim found at position 9")
        
        # Check dependent claim format
        dep_claims = re.findall(r'(\d+)\.\s+The\s+\w+\s+as claimed in claim (\d+)', claims_text)
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
                                verbose: bool = True) -> Dict[str, any]:
        """
        Complete pipeline: abstract → formatted claims with validation
        
        Args:
            abstract: Patent abstract text
            applicant_name: Name of patent applicant
            top_k_prior_art: Number of prior art patents to retrieve
            verbose: Print progress messages
        
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
        if verbose:
            print("[1/6] Extracting components from abstract...")
        components = self.extractor.extract(abstract)
        
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
        claim_1 = self.generator.generate_claim_1(abstract, components, prior_art_context)
        
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
                'claims_count': len(dependent_claims) + len(method_subclaims) + 1,
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
                                 verbose: bool = True) -> str:
    """
    Simple function to generate claims from abstract
    
    Args:
        abstract: Patent abstract text
        applicant_name: Applicant name for header
        verbose: Print progress
    
    Returns:
        Formatted claims text
    """
    pipeline = PatentClaimsPipeline()
    results = pipeline.generate_complete_claims(
        abstract, 
        applicant_name=applicant_name,
        verbose=verbose
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
    print(f"Total Claims Generated: {results['metadata']['claims_count'] + 1}")
    print(f"Prior Art References: {len(results['prior_art'])}")
    print(f"Quality Fixes Applied: {len(results['metadata']['fixes_applied'])}")
    print("\nFiles generated:")
    print("   • patent_claims_output.txt")
    print("   • patent_claims_report.json")
    print("\n" + "=" * 80)