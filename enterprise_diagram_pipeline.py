"""
Enterprise Patent Diagram Pipeline
====================================
4-Step Professional Patent Diagram Generation System

Step 1: Deconstructor - Extract JSON components (handled by ComponentRegistry.to_json())
Step 2: Visualizer - Dual-path generation (AI for mechanical, Mermaid for logic)
Step 3: Vector Polish - Raster to SVG conversion
Step 4: Assembler - Label overlay on images
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum

# Check for PIL availability
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed. Image processing features disabled.")

# Import component registry
from component_registry import ComponentRegistry, create_unified_registry


# ============================================================
# STEP 1: DECONSTRUCTOR (JSON EXTRACTION)
# ============================================================
# This is handled by ComponentRegistry.to_json() method
# Returns: {"100": "Component Name", "110": "Other Component"}


# ============================================================
# STEP 2: VISUALIZER (DUAL-PATH GENERATION)
# ============================================================

class DiagramType(Enum):
    """Classification of patent diagram types for routing."""
    # Path A: Mechanical views -> AI Image Generation
    PERSPECTIVE = "perspective"
    FRONT_VIEW = "front_view"
    SIDE_VIEW = "side_view"
    TOP_VIEW = "top_view"
    INTERNAL = "internal"
    CROSS_SECTION = "cross_section"
    EXPLODED = "exploded"
    ASSEMBLY = "assembly"
    
    # Path B: Logic diagrams -> Mermaid Code Generation
    BLOCK_DIAGRAM = "block_diagram"
    FLOWCHART = "flowchart"
    SEQUENCE_DIAGRAM = "sequence"
    CIRCUIT = "circuit"
    DATA_FLOW = "data_flow"


def get_diagram_path(diagram_type: DiagramType) -> str:
    """Determine which generation path to use."""
    mechanical_types = [
        DiagramType.PERSPECTIVE, DiagramType.FRONT_VIEW, DiagramType.SIDE_VIEW,
        DiagramType.TOP_VIEW, DiagramType.INTERNAL, DiagramType.CROSS_SECTION,
        DiagramType.EXPLODED, DiagramType.ASSEMBLY
    ]
    return "AI_IMAGE" if diagram_type in mechanical_types else "MERMAID"


def classify_drawing_type(description: str) -> DiagramType:
    """Classify drawing type from description text."""
    desc_lower = description.lower()
    
    # Check for mechanical views
    if 'perspective' in desc_lower:
        return DiagramType.PERSPECTIVE
    elif 'front' in desc_lower and 'view' in desc_lower:
        return DiagramType.FRONT_VIEW
    elif 'side' in desc_lower and 'view' in desc_lower:
        return DiagramType.SIDE_VIEW
    elif 'top' in desc_lower and 'view' in desc_lower:
        return DiagramType.TOP_VIEW
    elif 'internal' in desc_lower or 'inside' in desc_lower:
        return DiagramType.INTERNAL
    elif 'cross-section' in desc_lower or 'sectional' in desc_lower:
        return DiagramType.CROSS_SECTION
    elif 'exploded' in desc_lower:
        return DiagramType.EXPLODED
    elif 'assembly' in desc_lower:
        return DiagramType.ASSEMBLY
    
    # Check for logic diagrams
    elif 'block diagram' in desc_lower or 'schematic' in desc_lower:
        return DiagramType.BLOCK_DIAGRAM
    elif 'flowchart' in desc_lower or 'flow chart' in desc_lower or 'process' in desc_lower:
        return DiagramType.FLOWCHART
    elif 'sequence' in desc_lower:
        return DiagramType.SEQUENCE_DIAGRAM
    elif 'circuit' in desc_lower:
        return DiagramType.CIRCUIT
    elif 'data flow' in desc_lower:
        return DiagramType.DATA_FLOW
    
    # Default to block diagram
    return DiagramType.BLOCK_DIAGRAM


def generate_ai_image_prompt(
    diagram_type: DiagramType,
    invention_name: str,
    components_json: Dict[str, str]
) -> str:
    """
    Generate optimized AI prompt for mechanical views.
    Path A: For perspective, front, internal, cross-section views.
    """
    # Build component list with numerals
    comp_text = ", ".join([
        f"[{num}] {name}" for num, name in list(components_json.items())[:6]
    ])
    
    # Type-specific view descriptions
    view_prompts = {
        DiagramType.PERSPECTIVE: f"3/4 perspective view showing the overall form and exterior",
        DiagramType.FRONT_VIEW: f"Front elevation view showing the primary face",
        DiagramType.SIDE_VIEW: f"Side elevation view showing the profile",
        DiagramType.TOP_VIEW: f"Top plan view looking down on the device",
        DiagramType.INTERNAL: f"Cutaway view revealing internal components and mechanisms",
        DiagramType.CROSS_SECTION: f"Cross-sectional view with hatching showing cut surfaces",
        DiagramType.EXPLODED: f"Exploded view with parts separated along center axis, assembly lines shown",
        DiagramType.ASSEMBLY: f"Assembly diagram showing how parts connect together"
    }
    
    view_desc = view_prompts.get(diagram_type, "technical illustration")
    
    # Build the optimized prompt
    prompt = f"""Technical patent line drawing of {invention_name}.
{view_desc}.
Components shown with reference numerals: {comp_text}.

STYLE REQUIREMENTS (CRITICAL):
- Black and white outlines ONLY
- NO shading, NO gradients, NO colors
- White background
- High contrast, crisp clean lines
- Engineering sketch style
- Reference numerals must be clearly readable
- Professional patent illustration quality
- Single consistent line weight
"""
    return prompt


def generate_mermaid_code(
    diagram_type: DiagramType,
    invention_name: str,
    components_json: Dict[str, str]
) -> str:
    """
    Generate Mermaid code for logic diagrams.
    Path B: For block diagrams, flowcharts, sequence diagrams.
    """
    if diagram_type == DiagramType.BLOCK_DIAGRAM:
        return _generate_block_diagram_mermaid(invention_name, components_json)
    elif diagram_type == DiagramType.FLOWCHART:
        return _generate_flowchart_mermaid(invention_name, components_json)
    elif diagram_type == DiagramType.SEQUENCE_DIAGRAM:
        return _generate_sequence_mermaid(invention_name, components_json)
    else:
        return _generate_block_diagram_mermaid(invention_name, components_json)


def _generate_block_diagram_mermaid(name: str, components: Dict[str, str]) -> str:
    """Generate Mermaid block diagram code."""
    comp_items = list(components.items())[:6]
    
    mermaid = f"""```mermaid
%%{{init: {{
  'theme': 'base',
  'themeVariables': {{
    'primaryColor': '#ffffff',
    'primaryBorderColor': '#000000',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'fontFamily': 'Times New Roman, serif'
  }}
}}}}%%
flowchart TB
    subgraph SYSTEM["<b>{name}</b><br/>({comp_items[0][0] if comp_items else '100'})"]
        direction LR
"""
    
    for i, (num, comp_name) in enumerate(comp_items[1:], 1):
        mermaid += f'        C{i}["<b>{comp_name}</b><br/>({num})"]\n'
    
    mermaid += "    end\n\n"
    
    # Add connections
    num_comps = len(comp_items) - 1
    if num_comps >= 2:
        mermaid += "    C1 --> C2\n"
    if num_comps >= 3:
        mermaid += "    C2 --> C3\n"
    if num_comps >= 4:
        mermaid += "    C1 -.-> C4\n"
    if num_comps >= 5:
        mermaid += "    C4 --> C5\n"
    
    mermaid += """
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px
    classDef systemBox fill:#fafafa,stroke:#000000,stroke-width:3px
    class SYSTEM systemBox
```"""
    
    return mermaid


def _generate_flowchart_mermaid(name: str, components: Dict[str, str]) -> str:
    """Generate Mermaid flowchart code."""
    mermaid = f"""```mermaid
%%{{init: {{'theme': 'base', 'themeVariables': {{'lineColor': '#000000'}}}}}}%%
flowchart TD
    START(["<b>START</b>"])
    S101["S101: Initialize System"]
    S102["S102: Process Input"]
    S103["S103: Validate Data"]
    D1{{"Valid?"}}
    S104["S104: Execute Operation"]
    S105["S105: Generate Output"]
    END(["<b>END</b>"])
    
    START --> S101
    S101 --> S102
    S102 --> S103
    S103 --> D1
    D1 -->|Yes| S104
    D1 -->|No| S102
    S104 --> S105
    S105 --> END
    
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px
    classDef decision fill:#ffffff,stroke:#000000,stroke-width:2px
```"""
    return mermaid


def _generate_sequence_mermaid(name: str, components: Dict[str, str]) -> str:
    """Generate Mermaid sequence diagram code."""
    comp_items = list(components.items())[:4]
    
    actors = []
    for num, comp_name in comp_items:
        safe_name = comp_name.replace(" ", "_")[:15]
        actors.append((safe_name, num, comp_name))
    
    mermaid = f"""```mermaid
sequenceDiagram
    participant U as User
"""
    for safe_name, num, full_name in actors[:3]:
        mermaid += f"    participant {safe_name} as {full_name} ({num})\n"
    
    if len(actors) >= 2:
        mermaid += f"\n    U->>+{actors[0][0]}: Request\n"
        mermaid += f"    {actors[0][0]}->>+{actors[1][0]}: Process\n"
        mermaid += f"    {actors[1][0]}-->>-{actors[0][0]}: Response\n"
        mermaid += f"    {actors[0][0]}-->>-U: Result\n"
    
    mermaid += "```"
    return mermaid


# ============================================================
# STEP 3: VECTOR POLISH (RASTER TO SVG)
# ============================================================

def convert_to_vector(input_path: str, output_path: str = None) -> str:
    """
    Convert raster image (PNG) to vector (SVG) using edge detection.
    Uses potrace-style conversion for crisp patent lines.
    
    Args:
        input_path: Path to input PNG/JPG image
        output_path: Path for output SVG (auto-generated if None)
        
    Returns:
        Path to output SVG file
    """
    if not HAS_PIL:
        return input_path  # Return original if PIL not available
    
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_vector.svg"
    
    try:
        # Open and convert to grayscale
        img = Image.open(input_path).convert('L')
        width, height = img.size
        
        # Threshold to pure black/white
        threshold = 128
        img = img.point(lambda p: 255 if p > threshold else 0)
        
        # Convert to bitmap data
        pixels = list(img.getdata())
        
        # Generate SVG with paths
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <g fill="none" stroke="black" stroke-width="1">
'''
        
        # Simple edge detection - find black pixels and create paths
        for y in range(height):
            x = 0
            while x < width:
                idx = y * width + x
                if pixels[idx] == 0:  # Black pixel
                    # Start of a black segment
                    start_x = x
                    while x < width and pixels[y * width + x] == 0:
                        x += 1
                    # Draw line for this segment
                    svg_content += f'    <line x1="{start_x}" y1="{y}" x2="{x}" y2="{y}"/>\n'
                else:
                    x += 1
        
        svg_content += '''  </g>
</svg>'''
        
        # Write SVG file
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        return output_path
        
    except Exception as e:
        print(f"Vector conversion error: {e}")
        return input_path


# ============================================================
# STEP 4: ASSEMBLER (LABEL OVERLAY)
# ============================================================

def overlay_labels(
    image_path: str,
    components_json: Dict[str, str],
    output_path: str = None
) -> str:
    """
    Overlay reference numeral labels on diagram image.
    Places clean, readable text labels (100, 110) at calculated positions.
    
    Args:
        image_path: Path to input image
        components_json: Dict mapping numeral to component name
        output_path: Path for output image (auto-generated if None)
        
    Returns:
        Path to labeled image
    """
    if not HAS_PIL:
        return image_path
    
    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = f"{base}_labeled.png"
    
    try:
        # Open image
        img = Image.open(image_path).convert('RGBA')
        draw = ImageDraw.Draw(img)
        
        # Try to load a nice font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Calculate label positions (distributed around edges)
        width, height = img.size
        margin = 50
        
        positions = [
            (margin, margin),                    # Top-left
            (width - margin, margin),            # Top-right
            (margin, height - margin),           # Bottom-left
            (width - margin, height - margin),   # Bottom-right
            (width // 2, margin),                # Top-center
            (width // 2, height - margin),       # Bottom-center
        ]
        
        # Draw labels with leader lines
        for i, (num, name) in enumerate(list(components_json.items())[:6]):
            if i < len(positions):
                x, y = positions[i]
                
                # Draw white background for readability
                bbox = draw.textbbox((x, y), num, font=font)
                padding = 5
                draw.rectangle([
                    bbox[0] - padding, bbox[1] - padding,
                    bbox[2] + padding, bbox[3] + padding
                ], fill='white', outline='black')
                
                # Draw text
                draw.text((x, y), num, fill='black', font=font)
        
        # Save
        img.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Label overlay error: {e}")
        return image_path


# ============================================================
# MAIN PIPELINE
# ============================================================

class EnterpriseDiagramPipeline:
    """
    Complete 4-step enterprise patent diagram generation pipeline.
    """
    
    def __init__(self, output_dir: str = "patent_diagrams"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        self.components_json = {}
    
    def step1_deconstruct(self, abstract: str, drawing_summary: str = "") -> Dict[str, str]:
        """
        Step 1: Deconstructor - Extract structured JSON from abstract.
        """
        registry = create_unified_registry(abstract, drawing_summary)
        self.components_json = registry.to_json()
        print(f"Step 1 Complete: Extracted {len(self.components_json)} components")
        return self.components_json
    
    def step2_visualize(self, fig_num: int, description: str) -> Tuple[str, str]:
        """
        Step 2: Visualizer - Generate diagram content via dual-path routing.
        
        Returns:
            Tuple of (path_type, content) where:
            - path_type is "AI_IMAGE" or "MERMAID"
            - content is prompt string or Mermaid code
        """
        diagram_type = classify_drawing_type(description)
        path = get_diagram_path(diagram_type)
        
        # Get invention name from first component
        invention_name = list(self.components_json.values())[0] if self.components_json else "System"
        
        if path == "AI_IMAGE":
            content = generate_ai_image_prompt(diagram_type, invention_name, self.components_json)
        else:
            content = generate_mermaid_code(diagram_type, invention_name, self.components_json)
        
        print(f"Step 2 Complete: Fig {fig_num} -> {path} ({diagram_type.value})")
        return path, content
    
    def step3_vectorize(self, image_path: str) -> str:
        """
        Step 3: Vector Polish - Convert raster to SVG.
        """
        svg_path = convert_to_vector(image_path)
        print(f"Step 3 Complete: Converted to SVG -> {svg_path}")
        return svg_path
    
    def step4_assemble(self, image_path: str) -> str:
        """
        Step 4: Assembler - Overlay reference numeral labels.
        """
        labeled_path = overlay_labels(image_path, self.components_json)
        print(f"Step 4 Complete: Labels overlaid -> {labeled_path}")
        return labeled_path
    
    def process_drawing_summary(self, abstract: str, drawing_summary: str) -> List[Dict]:
        """
        Process complete drawing summary through the pipeline.
        
        Returns:
            List of figure generation info
        """
        # Step 1: Extract components
        self.step1_deconstruct(abstract, drawing_summary)
        
        # Parse figures from drawing summary
        figures = []
        fig_pattern = re.compile(
            r'(?:Figure|FIG\.?)\s*(\d+)[:\s]+(.+?)(?=(?:Figure|FIG\.?)\s*\d+|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        for match in fig_pattern.finditer(drawing_summary):
            fig_num = int(match.group(1))
            description = match.group(2).strip()
            
            # Step 2: Generate content
            path_type, content = self.step2_visualize(fig_num, description)
            
            figures.append({
                "fig_num": fig_num,
                "description": description,
                "path_type": path_type,
                "content": content,
                "components_json": self.components_json.copy()
            })
        
        return figures


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTERPRISE PATENT DIAGRAM PIPELINE TEST")
    print("=" * 60)
    
    abstract = """
    A smart waste sorting system comprising a main housing unit, 
    a solar panel array for power generation, a robotic arm for 
    material sorting, and an ultrasonic sensor array for detection.
    """
    
    drawing_summary = """
    Figure 1 is a perspective view of the waste sorting system showing the housing (100) and solar panel (110).
    Figure 2 is an exploded view showing the robotic arm (120) and sensor array (130).
    Figure 3 is a block diagram of the control system showing the processor (140).
    Figure 4 is a flowchart of the sorting process.
    """
    
    pipeline = EnterpriseDiagramPipeline()
    figures = pipeline.process_drawing_summary(abstract, drawing_summary)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for fig in figures:
        print(f"\nFigure {fig['fig_num']}:")
        print(f"  Path: {fig['path_type']}")
        print(f"  Description: {fig['description'][:50]}...")
        if fig['path_type'] == "MERMAID":
            print(f"  Content preview: {fig['content'][:100]}...")
        else:
            print(f"  Prompt preview: {fig['content'][:100]}...")
    
    print("\n" + "=" * 60)
    print("JSON Components:")
    print(json.dumps(pipeline.components_json, indent=2))
