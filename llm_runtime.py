"""
LLM Runtime Module for Patent_Doc_Copilot
==========================================

Provides a unified interface for LLM text generation using Qwen 3 8B via API.
All patent generation modules import from this file.

This replaces the previous local model implementation with API-based generation,
enabling CPU-only deployment while accessing state-of-the-art language models.

Usage:
    from llm_runtime import llm_generate
    
    response = llm_generate(
        prompt="Draft a patent claim for...",
        max_new_tokens=500,
        temperature=0.3
    )
"""

import os
import logging
import json
import re
from typing import Optional, List

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

from qwen3_api import Qwen3Client, APIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_MSG = (
    "You are an expert Indian patent attorney and technical writer. "
    "Your goal is to generate perfect, legally-compliant patent documentation following "
    "Indian Patent Office (IPO) requirements. "
    "STRICT RULE: Do NOT use conversational filler (e.g., 'Certainly', 'Okay', 'I will help'). "
    "Output ONLY the requested technical content or data format. "
    "CRITICAL: All claims MUST be a single sentence starting with a numeral (e.g., '1.'). "
    "/no_think"
)

# Singleton client instance
_client: Optional[Qwen3Client] = None


def get_llm() -> Qwen3Client:
    """
    Get or create the Qwen 3 8B API client.
    
    This function maintains a singleton instance for efficiency.
    The client is configured via environment variables:
        - OPENROUTER_API_KEY: API key for OpenRouter
        - QWEN3_PROVIDER: Provider choice ('openrouter' or 'huggingface')
        - QWEN3_MODEL: Model identifier (default: 'qwen/qwen3-8b')
    
    Returns:
        Qwen3Client instance
    
    Raises:
        ValueError: If API key is not configured
    """
    global _client
    
    if _client is None:
        logger.info("🚀 Initializing Qwen 3 8B API client...")
        _client = Qwen3Client()
        logger.info(f"✅ Qwen 3 8B ready (model: {_client.model})")
    
    return _client


def llm_generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.85,
    repeat_penalty: float = 1.15,
    stop_strings: Optional[List[str]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate text using Qwen 3 8B API.
    
    This function provides a drop-in replacement for the previous local model
    implementation, maintaining the same interface for compatibility with all
    patent generation modules.
    
    Args:
        prompt: The input prompt/instruction for generation.
        max_new_tokens: Maximum number of tokens to generate (default: 512).
        temperature: Sampling temperature, lower = more focused (default: 0.3).
        top_p: Nucleus sampling parameter (default: 0.85).
        repeat_penalty: Penalty for repeated tokens (default: 1.15).
        stop_strings: Optional list of strings that stop generation.
        system_prompt: Optional custom system message (uses default if None).
    
    Returns:
        Generated text string.
    
    Raises:
        APIError: If the API request fails after retries.
    
    Example:
        >>> from llm_runtime import llm_generate
        >>> claim = llm_generate(
        ...     prompt="Write Claim 1 for a smart sensor invention",
        ...     max_new_tokens=300,
        ...     temperature=0.2
        ... )
    """
    client = get_llm()
    
    # Use default patent-focused system prompt if not specified
    if system_prompt is None:
        system_prompt = SYSTEM_MSG
    
    try:
        response = client.generate(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            stop_sequences=stop_strings,
            system_prompt=system_prompt
        )
        return response
        
    except APIError as e:
        logger.error(f"❌ LLM generation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in llm_generate: {e}")
        raise APIError(f"Generation failed: {e}") from e


def reset_client():
    """
    Reset the singleton client instance.
    
    Useful for testing or when API configuration changes.
    """
    global _client
    _client = None
    logger.info("🔄 LLM client reset")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Runtime with Qwen 3 8B API")
    print("=" * 60)
    
    try:
        # Test basic generation
        print("\n🔄 Testing llm_generate()...")
        response = llm_generate(
            prompt="Write a brief one-sentence title for a patent about a smart home energy management system.",
            max_new_tokens=50,
            temperature=0.3
        )
        print(f"\n📝 Generated title:\n{response}")
        
        # Test with custom parameters
        print("\n🔄 Testing with custom parameters...")
        response = llm_generate(
            prompt="Write one claim element for a sensor system.",
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.2,
            stop_strings=["\n\n", "Claim 2"]
        )
        print(f"\n📝 Generated claim element:\n{response}")
        
        print("\n" + "=" * 60)
        print("✅ LLM Runtime tests passed!")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n⚠️ Configuration error: {e}")
        print("\nPlease set your API key:")
        print("  export OPENROUTER_API_KEY='your-api-key-here'")
        
    except APIError as e:
        print(f"\n❌ API error: {e}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
