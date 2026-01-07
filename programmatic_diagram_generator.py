"""
Programmatic Patent Diagram Generator
=====================================
Generates clean, IPO-compliant patent diagrams using matplotlib/PIL.
Produces pure black/white diagrams with proper reference numerals.
"""

import os
from typing import Dict, List, Optional
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Arrow
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from llm_runtime import llm_generate


class ProgrammaticDiagramGenerator:
    """Generate clean patent diagrams programmatically."""
    
    def __init__(self, output_dir: str = "generated_diagrams"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        self.components: Dict[str, int] = {}
        self.ref_num = 10  # Start with 10-series for IPO consistency
    
    def extract_components(self, abstract: str) -> Dict[str, int]:
        """Extract components from abstract and assign reference numerals."""
        # CRITICAL: Extract invention-specific components, NEVER use generic placeholders
        prompt = f"""Extract the 4-6 MAIN STRUCTURAL COMPONENTS from this patent abstract.
Return ONLY the actual component names as they appear in the abstract.
Do NOT use generic names like "Component1", "Module A", "Unit", etc.

ABSTRACT:
{abstract[:800]}

Return ONLY the actual component names from the abstract, one per line.
Example output for a glucose monitoring ring:
Ring Housing
Optical Sensor
AI Inference Engine
Battery
Wireless Transmitter
"""
        response = llm_generate(prompt, max_new_tokens=200, temperature=0.1)
        
        components = {}
        ref = 100  # Use 100-series for IPO consistency
        for line in response.strip().split('\n'):
            comp = line.strip()
            # CRITICAL: Skip generic placeholder names
            generic_names = ["component", "module", "unit", "system", "device", "element", "part"]
            if (comp and len(comp) > 2 and len(comp) < 50 and 
                not any(g in comp.lower() for g in generic_names)):
                components[comp] = ref
                ref += 10  # Increment by 10 for IPO format
        
        # FAIL LOUDLY if no invention-specific components found
        if not components:
            # Try to extract from abstract directly using regex
            import re
            # Look for "comprising:" section
            match = re.search(r'comprising[:\s]+(.+?)(?:wherein|;\.)', abstract, re.IGNORECASE | re.DOTALL)
            if match:
                parts = re.split(r';|\band\b|,', match.group(1))
                ref = 100
                for part in parts[:6]:
                    part = part.strip()
                    part = re.sub(r'^a\s+|^an\s+|^the\s+', '', part, flags=re.IGNORECASE)
                    if part and len(part) > 3 and len(part) < 50:
                        components[part[:40]] = ref
                        ref += 10
        
        # If STILL no components, return error marker
        if not components:
            components = {"ERROR: No components extracted": 0}
        
        self.components = components
        return components
    
    def generate_block_diagram_pil(self, title: str = "System") -> str:
        """Generate block diagram using PIL (fallback if no matplotlib)."""
        if not HAS_PIL:
            return ""
        
        # Create white image
        width, height = 800, 600
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        # Draw title
        main_ref = list(self.components.values())[0] if self.components else 10
        title_text = f"{title} ({main_ref})"
        draw.text((width//2 - 80, 20), title_text, fill='black', font=title_font)
        
        # Draw outer box
        draw.rectangle([40, 50, width-40, height-50], outline='black', width=2)
        
        # Draw component boxes
        comps = list(self.components.items())[:6]
        box_width = 150
        box_height = 60
        
        if len(comps) <= 3:
            # Horizontal layout
            start_x = (width - len(comps) * (box_width + 30)) // 2
            y = height // 2 - box_height // 2
            
            for i, (name, ref) in enumerate(comps):
                x = start_x + i * (box_width + 30)
                draw.rectangle([x, y, x + box_width, y + box_height], outline='black', width=2, fill='white')
                
                # Component name and ref
                text = f"{name[:15]}"
                draw.text((x + 10, y + 15), text, fill='black', font=font)
                draw.text((x + 10, y + 35), f"({ref})", fill='black', font=font)
                
                # Draw arrows
                if i > 0:
                    arrow_y = y + box_height // 2
                    draw.line([x - 25, arrow_y, x - 5, arrow_y], fill='black', width=2)
                    draw.polygon([(x-5, arrow_y-5), (x-5, arrow_y+5), (x, arrow_y)], fill='black')
        else:
            # Grid layout
            cols = 2
            rows = (len(comps) + 1) // 2
            cell_w = (width - 100) // cols
            cell_h = (height - 150) // rows
            
            for i, (name, ref) in enumerate(comps):
                col = i % cols
                row = i // cols
                
                x = 60 + col * cell_w + (cell_w - box_width) // 2
                y = 80 + row * cell_h + (cell_h - box_height) // 2
                
                draw.rectangle([x, y, x + box_width, y + box_height], outline='black', width=2, fill='white')
                text = f"{name[:12]}"
                draw.text((x + 10, y + 15), text, fill='black', font=font)
                draw.text((x + 10, y + 35), f"({ref})", fill='black', font=font)
        
        # Save
        filepath = os.path.join(self.output_dir, "fig1_block_diagram.png")
        img.save(filepath)
        return filepath
    
    def generate_block_diagram_mpl(self, title: str = "System") -> str:
        """Generate block diagram using matplotlib."""
        if not HAS_MATPLOTLIB:
            return self.generate_block_diagram_pil(title)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')
        
        # Main system box
        main_ref = list(self.components.values())[0] if self.components else 10
        main_box = FancyBboxPatch((0.5, 0.5), 9, 6, boxstyle="round,pad=0.05",
                                   facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(main_box)
        ax.text(5, 6.2, f"{title} ({main_ref})", ha='center', fontsize=12, fontweight='bold')
        
        # Component boxes
        comps = list(self.components.items())[1:5]  # Skip main system
        n = len(comps)
        
        for i, (name, ref) in enumerate(comps):
            x = 1.5 + (i % 2) * 4
            y = 4.5 - (i // 2) * 2.5
            
            box = FancyBboxPatch((x, y), 3, 1.5, boxstyle="round,pad=0.02",
                                  facecolor='white', edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + 1.5, y + 0.9, name[:15], ha='center', fontsize=10)
            ax.text(x + 1.5, y + 0.4, f"({ref})", ha='center', fontsize=9)
        
        # Draw connections
        if n >= 2:
            ax.annotate('', xy=(4.5, 5.25), xytext=(4.5, 5.25),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        filepath = os.path.join(self.output_dir, "fig1_block_diagram.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return filepath
    
    def generate_flowchart(self, steps: List[str] = None) -> str:
        """Generate process flowchart."""
        if not HAS_PIL:
            return ""
        
        if not steps:
            steps = ["Receive Input", "Process Data", "Analyze Results", "Generate Output"]
        
        width, height = 600, 800
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw title
        draw.text((width//2 - 80, 20), "Process Flowchart", fill='black', font=font)
        
        # Start/End ovals
        box_width = 180
        box_height = 40
        center_x = width // 2
        
        # START
        y = 60
        draw.ellipse([center_x - 50, y, center_x + 50, y + 35], outline='black', width=2, fill='white')
        draw.text((center_x - 25, y + 10), "START", fill='black', font=font)
        
        # Steps
        prev_y = y + 35
        for i, step in enumerate(steps[:6]):
            y = 120 + i * 100
            step_num = f"S{101 + i}"
            
            # Arrow from previous
            draw.line([center_x, prev_y, center_x, y], fill='black', width=2)
            draw.polygon([(center_x-5, y-10), (center_x+5, y-10), (center_x, y)], fill='black')
            
            # Step box
            draw.rectangle([center_x - box_width//2, y, center_x + box_width//2, y + box_height],
                          outline='black', width=2, fill='white')
            draw.text((center_x - box_width//2 + 10, y + 12), f"{step_num}: {step[:20]}", fill='black', font=font)
            
            prev_y = y + box_height
        
        # END
        y = prev_y + 30
        draw.line([center_x, prev_y, center_x, y], fill='black', width=2)
        draw.polygon([(center_x-5, y-10), (center_x+5, y-10), (center_x, y)], fill='black')
        draw.ellipse([center_x - 50, y, center_x + 50, y + 35], outline='black', width=2, fill='white')
        draw.text((center_x - 20, y + 10), "END", fill='black', font=font)
        
        filepath = os.path.join(self.output_dir, "fig2_flowchart.png")
        img.save(filepath)
        return filepath


def generate_patent_diagrams(abstract: str) -> Dict[str, str]:
    """
    Main function to generate all patent diagrams.
    
    Args:
        abstract: Patent abstract text
        
    Returns:
        Dictionary mapping diagram names to file paths
    """
    generator = ProgrammaticDiagramGenerator()
    
    # Extract components
    components = generator.extract_components(abstract)
    
    # Get main system name
    main_system = "System"
    if "comprising" in abstract.lower():
        import re
        match = re.search(r'^A\s+([a-zA-Z\s]+?)(?:\s+comprising)', abstract, re.IGNORECASE)
        if match:
            main_system = match.group(1).strip()
    
    diagrams = {}
    
    # Generate block diagram
    block_path = generator.generate_block_diagram_mpl(main_system)
    if block_path:
        diagrams["fig1_block_diagram"] = block_path
    
    # Generate flowchart
    flow_path = generator.generate_flowchart()
    if flow_path:
        diagrams["fig2_flowchart"] = flow_path
    
    return diagrams


if __name__ == "__main__":
    print("=" * 60)
    print("PROGRAMMATIC PATENT DIAGRAM GENERATOR")
    print("=" * 60)
    
    test_abstract = """
    A smart water monitoring system comprising:
    a sensor array for measuring water quality parameters,
    a processing unit with machine learning capabilities,
    a wireless communication module for data transmission,
    and a mobile application for user alerts.
    """
    
    diagrams = generate_patent_diagrams(test_abstract)
    
    print(f"\n✅ Generated {len(diagrams)} diagrams:")
    for name, path in diagrams.items():
        print(f"   - {name}: {path}")
