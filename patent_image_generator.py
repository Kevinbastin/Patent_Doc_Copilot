"""
Patent Image Generator
======================
Generates patent drawing images using Hugging Face SDXL (FREE).

Model: stabilityai/stable-diffusion-xl-base-1.0
Cost: FREE (uses HUGGINGFACE_API_KEY)
"""

import os
import requests
import logging
from typing import Dict, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration - Hugging Face SDXL (FREE)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Recommended models for local deployment on RTX 5090 32GB VRAM
# Based on 2024 benchmarks and user's existing Qwen setup
LOCAL_MODEL_RECOMMENDATIONS = """
## Recommended Image Generation Models for RTX 5090 (32GB VRAM)

### 🏆 BEST OPTIONS (Based on 2024 Benchmarks):

1. **Qwen-Image** ⭐ RECOMMENDED (Already using Qwen API)
   - From Alibaba Cloud - same provider as your text model
   - Complex text rendering capability
   - Precise image editing and generation
   - Available via DashScope API / Alibaba Cloud
   - Best for: Technical drawings with text labels
   
   API setup (uses your existing Alibaba/Qwen credentials):
   ```python
   # Via DashScope (Alibaba Cloud)
   from dashscope import ImageSynthesis
   
   response = ImageSynthesis.call(
       model="qwen-image",
       prompt="technical patent drawing, black and white...",
       n=1,
       size="1024*1024"
   )
   ```

2. **FLUX.1-dev** - 12B parameters
   - BEST for: Text rendering, precision, technical details
   - Superior prompt adherence for complex diagrams
   - VRAM: ~24GB (fits with quantization on 32GB)
   
   ```python
   from diffusers import FluxPipeline
   import torch
   
   pipe = FluxPipeline.from_pretrained(
       "black-forest-labs/FLUX.1-dev",
       torch_dtype=torch.bfloat16
   ).to("cuda")
   
   image = pipe("technical patent drawing...").images[0]
   ```

3. **FLUX.1-schnell** - 12B parameters (FAST)
   - 10x faster than FLUX.1-dev
   - Good quality, quick iterations
   - VRAM: ~24GB

4. **Stable Diffusion XL (SDXL)** - 6.6B parameters
   - Falls well under 10B parameter limit
   - Excellent detail and clarity
   - VRAM: ~12GB
   
   ```python
   from diffusers import StableDiffusionXLPipeline
   import torch
   
   pipe = StableDiffusionXLPipeline.from_pretrained(
       'stabilityai/stable-diffusion-xl-base-1.0',
       torch_dtype=torch.float16
   ).to('cuda')
   
   image = pipe('technical patent drawing...').images[0]
   ```

### Benchmark Summary (Technical Drawings):
| Model | Text Rendering | Precision | Speed | VRAM |
|-------|---------------|-----------|-------|------|
| Qwen-Image | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | API |
| FLUX.1-dev | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 24GB |
| FLUX.1-schnell | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 24GB |
| SDXL | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 12GB |

### Quick Setup for Local:
```bash
pip install diffusers transformers accelerate torch bitsandbytes
```
"""





class PatentImageGenerator:
    """
    Generate patent-style technical drawings using AI image generation.
    Uses Stability AI API or generates prompts for local/external use.
    """
    
    def __init__(self):
        """Initialize the patent image generator."""
        self.output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_key = HUGGINGFACE_API_KEY
    
    def _create_system_diagram_prompt(self, components: Dict) -> str:
        """Create prompt for system block diagram (Figure 1)."""
        main_system = components.get("main_system", "System")
        comps = components.get("components", ["Component A", "Component B"])
        inputs = components.get("inputs", ["Input"])
        outputs = components.get("outputs", ["Output"])
        
        # Assign reference numerals
        ref_nums = {main_system: 100}
        for i, comp in enumerate(comps[:6]):
            ref_nums[comp] = 110 + i * 10
        
        # IMPORTANT: SDXL cannot generate readable text - focus on clean geometric shapes
        num_boxes = min(len(comps), 4)
        
        prompt = f"""A clean professional technical patent line drawing, black and white only:

EXACT SPECIFICATION:
- Pure white background
- One large outer rectangle (main system boundary)
- {num_boxes} smaller rectangles inside arranged in a grid
- Simple arrows connecting the boxes showing data flow
- Lines are clean, solid black, professional weight
- NO TEXT, NO LABELS, NO NUMBERS (these will be added manually)
- Style: minimalist engineering blueprint
- Very clean and simple geometric shapes only

This is Figure 1 for a patent filing - must be pristine black lines on white.
Art style: technical patent illustration, vector-like clean lines, simple geometric shapes."""

        return prompt
    
    def _create_flowchart_prompt(self, components: Dict) -> str:
        """Create prompt for method flowchart (Figure 2)."""
        steps = components.get("process_steps", ["Step 1", "Step 2", "Step 3"])
        num_steps = min(len(steps), 5)
        
        # IMPORTANT: SDXL cannot generate readable text
        prompt = f"""A clean professional technical flowchart, black and white only:

EXACT SPECIFICATION:
- Pure white background
- One oval at the top (START)
- {num_steps} rectangles stacked vertically below (process steps)
- One diamond shape (decision point)
- One oval at the bottom (END)
- Simple arrows connecting all shapes top to bottom
- Lines are clean, solid black, professional weight
- NO TEXT, NO LABELS (labels will be added manually)
- Style: minimalist flowchart
- Very clean geometric shapes only

This is Figure 2 for a patent filing - must be pristine black lines on white.
Art style: technical flowchart, simple geometric shapes, clean lines."""

        return prompt
    
    def _create_sequence_diagram_prompt(self, components: Dict) -> str:
        """Create prompt for sequence/interaction diagram (Figure 3)."""
        comps = components.get("components", ["Module A", "Module B"])[:3]
        num_participants = min(len(comps) + 1, 4)
        
        # IMPORTANT: SDXL cannot generate readable text
        prompt = f"""A clean professional UML sequence diagram, black and white only:

EXACT SPECIFICATION:
- Pure white background
- {num_participants} vertical dashed lines (lifelines) evenly spaced
- Small rectangles at top of each lifeline (participants)
- Horizontal arrows between lifelines showing messages
- Thin vertical rectangles on lifelines (activation boxes)
- Lines are clean, solid black, professional weight
- NO TEXT, NO LABELS (labels will be added manually)
- Style: minimalist UML sequence diagram
- Very clean geometric shapes and lines only

This is Figure 3 for a patent filing - must be pristine black lines on white.
Art style: technical UML diagram, simple lines and rectangles."""

        return prompt

    def _create_detail_view_prompt(self, component_name: str, ref_num: int) -> str:
        """Create prompt for component detail view (Figure 4)."""
        # IMPORTANT: SDXL cannot generate readable text
        prompt = f"""A clean professional technical exploded view diagram, black and white only:

EXACT SPECIFICATION:
- Pure white background
- One main rectangular component outline
- 4-5 smaller rectangles inside showing sub-components
- Dashed lines connecting sub-parts to the main body
- Simple geometric shapes representing internal parts
- Lines are clean, solid black, professional weight
- NO TEXT, NO LABELS, NO NUMBERS (will be added manually)
- Style: minimalist engineering exploded view
- Very clean geometric shapes only

This is Figure 4 for a patent filing - must be pristine black lines on white.
Art style: technical exploded view, simple geometric shapes."""

        return prompt

    def get_image_prompts_for_api(self, abstract: str, components: Dict) -> List[Dict]:
        """
        Get structured prompts for image generation.
        
        Returns list of dicts with figure_number, title, and prompt.
        """
        prompts = []
        
        # Figure 1: Block Diagram
        prompts.append({
            "figure_number": 1,
            "title": "System Block Diagram",
            "prompt": self._create_system_diagram_prompt(components),
            "filename": "patent_fig1_block_diagram.png"
        })
        
        # Figure 2: Flowchart
        prompts.append({
            "figure_number": 2,
            "title": "Method Flowchart",
            "prompt": self._create_flowchart_prompt(components),
            "filename": "patent_fig2_flowchart.png"
        })
        
        # Figure 3: Sequence Diagram
        prompts.append({
            "figure_number": 3,
            "title": "Sequence Diagram",
            "prompt": self._create_sequence_diagram_prompt(components),
            "filename": "patent_fig3_sequence.png"
        })
        
        # Figure 4: Detail View (if components exist)
        if components.get("components"):
            first_comp = components["components"][0]
            prompts.append({
                "figure_number": 4,
                "title": f"Detailed View of {first_comp}",
                "prompt": self._create_detail_view_prompt(first_comp, 110),
                "filename": "patent_fig4_detail.png"
        })
        
        return prompts
    
    def generate_image(self, prompt: str, filename: str) -> Optional[str]:
        """
        Generate patent drawing image using Hugging Face SDXL (FREE).
        
        Model: stabilityai/stable-diffusion-xl-base-1.0
        Cost: FREE (uses HUGGINGFACE_API_KEY from .env)
        """
        if not HUGGINGFACE_API_KEY:
            logger.warning("HUGGINGFACE_API_KEY not configured in .env")
            return None
        
        try:
            response = requests.post(
                "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json={"inputs": prompt},
                timeout=120
            )
            
            if response.status_code == 200:
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                # Post-process for IPO compliance (binary black/white)
                self._make_ipo_compliant(filepath)
                
                logger.info(f"✅ Generated: {filepath}")
                return filepath
            else:
                logger.error(f"❌ SDXL generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Image generation error: {e}")
            return None
    
    def _make_ipo_compliant(self, image_path: str):
        """
        Post-process image for IPO compliance.
        Converts to binary black/white for crisp printing.
        """
        try:
            from PIL import Image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            # Threshold: Lighter than 150 becomes WHITE, darker becomes BLACK
            img = img.point(lambda x: 0 if x < 150 else 255, '1')
            img.save(image_path)
            logger.info(f"✅ IPO post-processing applied: {image_path}")
        except ImportError:
            logger.warning("PIL not available for post-processing")
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
    
    def get_reference_numerals_table(self, components: Dict) -> str:
        """Generate reference numerals table for patent specification."""
        table = "REFERENCE NUMERALS\n\n"
        table += "=" * 50 + "\n"
        table += f"{'Numeral':<12} {'Description':<38}\n"
        table += "=" * 50 + "\n"
        
        # Main system
        main = components.get("main_system", "System")
        table += f"{'100':<12} {main:<38}\n"
        
        # Components
        for i, comp in enumerate(components.get("components", [])[:6]):
            ref = 110 + i * 10
            table += f"{ref:<12} {comp:<38}\n"
        
        table += "=" * 50 + "\n"
        
        # Step numbers
        steps = components.get("process_steps", [])
        if steps:
            table += "\nSTEP NUMBERS (Flowchart)\n"
            table += "-" * 50 + "\n"
            for i, step in enumerate(steps[:8]):
                table += f"S{101+i:<11} {step[:40]}\n"
            table += "-" * 50 + "\n"
        
        return table
    
    @staticmethod
    def get_local_model_guide() -> str:
        """Return guide for setting up local image generation."""
        return LOCAL_MODEL_RECOMMENDATIONS


def extract_components_for_images(abstract: str) -> Dict:
    """
    Extract components from abstract for image generation.
    Uses LLM for intelligent extraction.
    """
    from llm_runtime import llm_generate
    
    prompt = f"""Analyze this patent abstract and extract components for technical drawing generation.

ABSTRACT:
{abstract}

Extract these elements (be specific and technical):
1. MAIN_SYSTEM: The main invention name/title
2. COMPONENTS: List of 4-6 key hardware/software components
3. INPUTS: What data/signals go into the system
4. OUTPUTS: What the system produces
5. PROCESS_STEPS: List of 4-6 method steps (use action verbs)
6. ACTORS: Who/what interacts with the system

Return EXACTLY in this format:
MAIN_SYSTEM: [name]
COMPONENTS: [comp1], [comp2], [comp3], [comp4]
INPUTS: [input1], [input2]
OUTPUTS: [output1], [output2]
PROCESS_STEPS: [step1], [step2], [step3], [step4]
ACTORS: [actor1], [actor2]
"""
    
    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=400,
            temperature=0.2,
            system_prompt="Extract technical components precisely for patent technical drawings."
        )
        
        if response:
            return _parse_components(response)
    except Exception as e:
        logger.error(f"Component extraction error: {e}")
    
    # Fallback
    return {
        "main_system": "Invention System",
        "components": ["Processing Unit", "Input Module", "Output Module", "Controller"],
        "inputs": ["User Input", "Data"],
        "outputs": ["Processed Output", "Result"],
        "process_steps": ["Receive Input", "Process Data", "Generate Output", "Return Result"],
        "actors": ["User", "System"]
    }


def _parse_components(response: str) -> Dict:
    """Parse LLM response into component dictionary."""
    result = {
        "main_system": "System",
        "components": [],
        "inputs": [],
        "outputs": [],
        "process_steps": [],
        "actors": []
    }
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.startswith("MAIN_SYSTEM:"):
            result["main_system"] = line.split(":", 1)[1].strip()
        elif line.startswith("COMPONENTS:"):
            result["components"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif line.startswith("INPUTS:"):
            result["inputs"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif line.startswith("OUTPUTS:"):
            result["outputs"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif line.startswith("PROCESS_STEPS:"):
            result["process_steps"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif line.startswith("ACTORS:"):
            result["actors"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
    
    return result


def generate_patent_images(abstract: str, use_api: bool = False) -> Dict:
    """
    Main function to generate patent images or prompts.
    
    Args:
        abstract: Patent abstract text
        use_api: If True, attempt to generate images via API
    
    Returns:
        Dict with prompts, components, and optional image paths
    """
    # Extract components
    components = extract_components_for_images(abstract)
    
    # Create generator and get prompts
    generator = PatentImageGenerator()
    prompts = generator.get_image_prompts_for_api(abstract, components)
    ref_table = generator.get_reference_numerals_table(components)
    
    result = {
        "components": components,
        "prompts": prompts,
        "reference_numerals": ref_table,
        "images": []
    }
    
    # Generate images if API is available
    if use_api and (STABILITY_API_KEY or OPENROUTER_API_KEY):
        for prompt_info in prompts:
            image_path = generator.generate_with_stability_api(
                prompt_info["prompt"],
                prompt_info["filename"]
            )
            if image_path:
                result["images"].append({
                    "figure": prompt_info["figure_number"],
                    "path": image_path
                })
    
    return result


# CLI testing
if __name__ == "__main__":
    print("=" * 70)
    print("PATENT IMAGE GENERATOR - TEST")
    print("=" * 70)
    
    test_abstract = """
    A smart monitoring system for industrial environments, comprising: 
    sensor units, a processing hub with machine learning, and an alert module.
    The method includes data collection, ML analysis, and alert generation.
    """
    
    print("\n📥 Test Abstract:")
    print(test_abstract.strip())
    
    print("\n⏳ Generating image prompts...")
    result = generate_patent_images(test_abstract)
    
    print("\n📋 Components Extracted:")
    print(f"   Main System: {result['components'].get('main_system')}")
    print(f"   Components: {result['components'].get('components')}")
    
    print("\n📊 Generated Prompts:")
    for p in result["prompts"]:
        print(f"\n   Figure {p['figure_number']}: {p['title']}")
        print(f"   {p['prompt'][:150]}...")
    
    print("\n📋 Reference Numerals:")
    print(result["reference_numerals"])
    
    print("\n" + "=" * 70)
    print("LOCAL MODEL RECOMMENDATIONS:")
    print("=" * 70)
    print(PatentImageGenerator.get_local_model_guide())
