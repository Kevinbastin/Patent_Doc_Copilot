"""
Qwen 3 API Client for Patent_Doc_Copilot
=========================================

Professional-grade API client supporting OpenRouter and HuggingFace Inference API.
Designed for patent drafting with proper error handling, retry logic, and rate limiting.

Usage:
    from qwen3_api import Qwen3Client
    
    client = Qwen3Client()  # Uses OPENROUTER_API_KEY from environment
    response = client.generate("Draft a patent claim for...")
"""

import os
import re
import json
import time
import logging
from typing import Optional, List
from dataclasses import dataclass

# Robust .env loader
def load_env_file():
    """Load .env file manually if python-dotenv is missing."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Strip quotes if present
                    value = value.strip().strip("'").strip('"')
                    os.environ[key.strip()] = value

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_env_file()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """API configuration with sensible defaults for patent drafting."""
    
    # Provider settings
    provider: str = "openrouter"
    
    # Model defaults - Using Qwen 3 8B
    openrouter_model: str = "qwen/qwen3-8b"
    huggingface_model: str = "Qwen/Qwen3-8B"
    
    # API endpoints
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    huggingface_base_url: str = "https://router.huggingface.co/hf-inference/models"
    
    # Generation defaults (optimized for patent drafting)
    default_max_tokens: int = 1024
    default_temperature: float = 0.3
    default_top_p: float = 0.85
    default_repetition_penalty: float = 1.15
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    request_timeout: int = 120  # Patents can require longer generation
    
    # Rate limiting
    min_request_interval: float = 0.5  # seconds between requests


# =============================================================================
# API CLIENT
# =============================================================================

class Qwen3Client:
    """
    Professional Qwen 3 8B API client with support for multiple providers.
    
    Features:
        - Automatic retry with exponential backoff
        - Rate limiting to avoid API throttling
        - Proper error handling and logging
        - Support for thinking mode (Qwen 3 feature)
        - Environment-based configuration
    
    Example:
        >>> client = Qwen3Client()
        >>> response = client.generate(
        ...     prompt="Draft an independent patent claim for a smart sensor system",
        ...     max_tokens=500,
        ...     temperature=0.2
        ... )
        >>> print(response)
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[APIConfig] = None
    ):
        """
        Initialize the Qwen 3 API client.
        
        Args:
            provider: API provider ('openrouter' or 'huggingface'). 
                      Defaults to QWEN3_PROVIDER env var or 'openrouter'.
            model: Model identifier. Defaults to 'qwen/qwen3-8b'.
            api_key: API key. Defaults to provider-specific env var.
            config: Custom APIConfig object.
        """
        self.config = config or APIConfig()
        
        # Determine provider
        self.provider = provider or os.getenv("QWEN3_PROVIDER", self.config.provider)
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        
        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set {'OPENROUTER_API_KEY' if self.provider == 'openrouter' else 'HUGGINGFACE_API_KEY'} "
                f"environment variable or pass api_key parameter."
            )
        
        # Determine model - Default to Qwen 3 8B
        if model:
            self.model = model
        elif os.getenv("QWEN3_MODEL"):
            self.model = os.getenv("QWEN3_MODEL")
        elif self.provider == "openrouter":
            self.model = self.config.openrouter_model
        else:
            self.model = self.config.huggingface_model
        
        # Setup session with retry logic
        self.session = self._create_session()
        
        # Rate limiting
        self._last_request_time = 0
        
        logger.info(f"✅ Qwen3Client initialized: provider={self.provider}, model={self.model}")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.min_request_interval:
            time.sleep(self.config.min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False
    ) -> str:
        """
        Generate text using Qwen 3 8B API.
        
        Args:
            prompt: The user prompt/instruction.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling parameter.
            repetition_penalty: Penalty for repeated tokens.
            stop_sequences: List of strings that stop generation.
            system_prompt: Optional system message for context.
            enable_thinking: Enable Qwen 3's thinking mode for complex reasoning.
        
        Returns:
            Generated text string.
        
        Raises:
            APIError: If the API request fails after retries.
        """
        self._rate_limit()
        
        # Use defaults if not specified
        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature
        top_p = top_p if top_p is not None else self.config.default_top_p
        repetition_penalty = repetition_penalty or self.config.default_repetition_penalty
        
        # Build system prompt
        if system_prompt is None:
            system_prompt = (
                "You are an expert patent attorney and technical writer. "
                "Generate precise, legally-compliant patent documentation. "
                "Follow patent office formatting requirements strictly."
            )
        
        if enable_thinking:
            system_prompt += " /think"  # Qwen 3 thinking mode trigger
        
        try:
            if self.provider == "openrouter":
                return self._generate_openrouter(
                    prompt, system_prompt, max_tokens, temperature, 
                    top_p, repetition_penalty, stop_sequences
                )
            else:
                return self._generate_huggingface(
                    prompt, system_prompt, max_tokens, temperature,
                    top_p, stop_sequences
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            raise APIError(f"Network error during API call: {e}") from e
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise APIError(f"Failed to generate text: {e}") from e
    
    def _generate_openrouter(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using OpenRouter API."""
        url = f"{self.config.openrouter_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/patent-doc-copilot",
            "X-Title": "PatentDoc Copilot"
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        logger.debug(f"Sending request to OpenRouter: model={self.model}")
        
        response = self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout
        )
        
        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                error_detail = response.text
            raise APIError(f"OpenRouter API error ({response.status_code}): {error_detail}")
        
        result = response.json()
        # logger.info(f"DEBUG: API result: {result}") # Temporarily disabled
        
        if "choices" not in result or not result["choices"]:
            raise APIError(f"No choices in API response. Full result: {result}")
        
        choice = result["choices"][0]
        message = choice.get("message", {})
        
        # 1. Try standard content
        content = message.get("content")
        
        # 2. Try reasoning/thought fields (Qwen 3 often uses these)
        if not content:
            content = message.get("reasoning") or message.get("thought") or choice.get("text")
            
        # 3. Last resort - look for any non-empty string in the message
        if not content:
            for val in message.values():
                if isinstance(val, str) and len(val.strip()) > 10:
                    content = val
                    break
                    
        content = content or ""
        original_unfiltered = content
        
        # --- INTELLIGENT PATENT CONTENT EXTRACTION ---
        # For Qwen 3 reasoning-only outputs, find the actual patent content
        if content:
            # Look for patent-specific content markers
            patent_content_markers = [
                r'(The present invention relates.*?)(?:\n\n|\Z)',
                r'(This invention pertains.*?)(?:\n\n|\Z)',
                r'(The present disclosure relates.*?)(?:\n\n|\Z)',
                r'(Thus according to.*?)(?:\n\n|\Z)',
                r'(\d+\.\s+A\s+(?:method|system|apparatus|device|composition).*?)(?:\n\n|\Z)',
            ]
            for marker in patent_content_markers:
                match = re.search(marker, content, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > 50:  # Ensure we got substantial content
                        content = extracted
                        break
        
        # --- DEEP PERFECTION FILTER ---
        # A) JSON Extractor: If we expect data, prioritize the block in braces FIRST
        # This handles cases where JSON is inside reasoning tags that would be stripped.
        if "{" in content and "}" in content:
            # Try to find all JSON-like blocks and pick the one that actually parses
            import json
            candidates = re.findall(r'(\{.+\})', content, re.DOTALL)
            # Try from largest to smallest
            candidates.sort(key=len, reverse=True)
            for cand in candidates:
                try:
                    # Basic sanity check (ensure it's not just a single word in braces)
                    if ":" in cand and len(cand) > 10:
                        json.loads(cand) # Verify it actually parses
                        return cand.strip()
                except:
                    continue

        # B) Strip all varieties of thinking/reasoning tags (JUST THE TAGS, not content)
        # This prevents stripping the actual answer if it's wrapped in tags.
        content = re.sub(r'</?(?:think|thought|reasoning)>', '', content, flags=re.IGNORECASE)

        # C) AGGRESSIVE CHATTER FILTER for reasoning-only outputs
        # Remove common model "thinking out loud" patterns
        thinking_patterns = [
            r"^Okay,?\s+(?:the user|I need|let me|I'll).*?(?:\.\s*|\n)",
            r"^Let me (?:start|analyze|think|draft|understand).*?(?:\.\s*|\n)",
            r"^I need to.*?(?:\.\s*|\n)",
            r"^First,? I(?:'ll| will| should).*?(?:\.\s*|\n)",
            r"^(?:Understood|Got it|Alright).*?(?:\.\s*|\n)",
            r"^The (?:user|abstract) (?:wants|provides|mentions).*?(?:\.\s*|\n)",
            r"^\*\*.*?\*\*\s*\n?",  # Bold headers like **Title:**
        ]
        for pattern in thinking_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        # D) Strip common assistant chatter efficiently
        # Look for the start of actual patent content
        patent_triggers = [
            r'The present invention',
            r'\d+\.\s+[A-Z]',
            r'TITLE:',
            r'FIELD OF',
            r'SUMMARY OF',
            r'DETAILED DESCRIPTION',
            r'OBJECTS OF'
        ]
        for trigger in patent_triggers:
            match = re.search(trigger, content, re.IGNORECASE)
            if match and 5 < match.start() < 500:
                # If we find a patent trigger after some chatter, strip the chatter
                preamble = content[:match.start()].strip()
                if any(kw in preamble.lower() for kw in ['certainly', 'here is', 'okay', 'draft', 'assistant', 'understood', 'let me', 'i need']):
                    content = content[match.start():]
                    break

        # E) Clean up residual markdown formatting
        content = re.sub(r'^```[a-z]*\n', '', content, flags=re.I)
        content = re.sub(r'\n```$', '', content)
        content = content.replace("```json", "").replace("```", "")

        # F) SAFETY FALLBACK: If we accidentally stripped EVERYTHING, revert to original
        if not content.strip() and original_unfiltered.strip():
            return original_unfiltered.strip()

        return content.strip()
    
    def _generate_huggingface(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using HuggingFace Inference API."""
        url = f"{self.config.huggingface_base_url}/{self.model}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format as chat-style prompt for Qwen 3
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False
            }
        }
        
        if stop_sequences:
            payload["parameters"]["stop_sequences"] = stop_sequences
        
        logger.debug(f"Sending request to HuggingFace: model={self.model}")
        
        response = self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout
        )
        
        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", response.text)
            except Exception:
                error_detail = response.text
            raise APIError(f"HuggingFace API error ({response.status_code}): {error_detail}")
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            content = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            content = result.get("generated_text", "")
        else:
            raise APIError(f"Unexpected response format: {result}")
        
        # Clean up any chat markers
        content = content.replace("<|im_end|>", "").strip()
        
        return content


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton client instance
_default_client: Optional[Qwen3Client] = None


def get_client() -> Qwen3Client:
    """Get or create the default Qwen3Client instance."""
    global _default_client
    if _default_client is None:
        _default_client = Qwen3Client()
    return _default_client


def generate(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.85,
    repetition_penalty: float = 1.15,
    stop_sequences: Optional[List[str]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Convenience function for quick text generation using Qwen 3 8B.
    
    Args:
        prompt: The user prompt/instruction.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        repetition_penalty: Penalty for repeated tokens.
        stop_sequences: List of strings that stop generation.
        system_prompt: Optional system message.
    
    Returns:
        Generated text string.
    """
    return get_client().generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_sequences=stop_sequences,
        system_prompt=system_prompt
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qwen 3 8B API Client")
    print("=" * 60)
    
    try:
        # Initialize client
        client = Qwen3Client()
        print(f"\n✅ Client initialized successfully")
        print(f"   Provider: {client.provider}")
        print(f"   Model: {client.model}")
        
        # Test generation
        print("\n🔄 Testing text generation...")
        response = client.generate(
            prompt="Write a one-sentence description of what a patent is.",
            max_tokens=100,
            temperature=0.3
        )
        print(f"\n📝 Response:\n{response}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n⚠️ Configuration error: {e}")
        print("\nTo fix this:")
        print("1. Get an API key from https://openrouter.ai/keys")
        print("2. Set the environment variable:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        
    except APIError as e:
        print(f"\n❌ API error: {e}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
