"""
Patent Diagram Generator
=========================
Generates IPO-compliant patent diagrams using Mermaid syntax.
- Block diagrams (system architecture)
- Flowcharts (method/process)
- Sequence diagrams (component interactions)
"""

from llm_runtime import llm_generate
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Import IPO-compliant diagram generator for formal patent drawings
from ipo_patent_diagrams import (
    IPODiagramGenerator,
    generate_ipo_diagrams as generate_ipo_formal_diagrams
)

# Import physical drawing generator for structural views
try:
    from patent_drawing_generator import generate_patent_drawings as generate_physical_drawings
    PHYSICAL_DRAWING_AVAILABLE = True
except ImportError:
    PHYSICAL_DRAWING_AVAILABLE = False


# ============================================================
# DRAWING TYPE CLASSIFICATION
# ============================================================
class DrawingType(Enum):
    """Classification of patent drawing types."""
    # Physical/mechanical views - CANNOT auto-generate
    PERSPECTIVE = "perspective"
    CROSS_SECTION = "cross-section"
    EXPLODED = "exploded"
    FRONT_VIEW = "front view"
    SIDE_VIEW = "side view"
    TOP_VIEW = "top view"
    ISOMETRIC = "isometric"
    ELEVATION = "elevation"
    DETAIL_VIEW = "detail view"
    SECTIONAL = "sectional"
    
    # Schematic diagrams - CAN auto-generate
    BLOCK_DIAGRAM = "block diagram"
    FLOWCHART = "flowchart"
    SEQUENCE_DIAGRAM = "sequence diagram"
    CIRCUIT_DIAGRAM = "circuit diagram"
    SYSTEM_ARCHITECTURE = "system architecture"


# View types that require CAD/manual illustration
PHYSICAL_VIEW_TYPES = [
    'perspective', 'cross-section', 'sectional', 'exploded', 'isometric', 
    'elevation', 'front view', 'side view', 'top view', 'bottom view', 
    'rear view', 'detail view', 'assembly view', 'cutaway'
]

# View types we can auto-generate
SCHEMATIC_VIEW_TYPES = [
    'block diagram', 'flowchart', 'sequence diagram', 'circuit diagram',
    'system diagram', 'architecture', 'data flow', 'process flow'
]

# Note: Removed keyword-based hardware detection for universal input support.
# The system now provides a universal warning for ALL inventions about schematic limitations.

# Critical notice about schematic vs physical drawings
SCHEMATIC_LIMITATION_WARNING = """
⚠️ IMPORTANT: SCHEMATIC DIAGRAMS ARE SUPPLEMENTARY ONLY

The auto-generated diagrams (block diagrams, flowcharts, sequence diagrams) show 
LOGICAL FLOW and SYSTEM ARCHITECTURE only. They DO NOT show:
- Physical structure or dimensions
- Mechanical arrangement
- Optical paths
- Electro-chemical component layout
- Assembly relationships

FOR IPO COMPLIANCE: If your invention involves physical, mechanical, optical, or 
electro-chemical components, you MUST create professional CAD/technical drawings 
showing actual physical structure.

REQUIRED DRAWING TYPES FOR PHYSICAL INVENTIONS:
- Cross-sectional views (showing internal arrangement)
- Exploded views (showing assembly)
- Isometric/perspective views (showing 3D structure)
- Detail views (showing critical features)

TOOLS: AutoCAD, SolidWorks, Fusion 360, CATIA, Inventor

The schematic diagrams below should be filed as SUPPLEMENTARY figures only.
"""


def get_diagram_limitations_notice() -> str:
    """
    Returns an explicit notice that schematic diagrams are supplementary only.
    Physical drawings are REQUIRED for inventions with physical structure.
    """
    return (
        "⚠️ SUPPLEMENTARY ONLY: These schematic diagrams show logical flow, NOT physical structure. "
        "For inventions with mechanical, optical, or electro-chemical components, "
        "you MUST create CAD drawings (cross-sections, exploded views, isometric views) for IPO compliance. "
        "These schematics alone are NOT sufficient for patent filing."
    )


def parse_drawing_summary(drawing_summary: str) -> List[Dict]:
    """
    Parse user's drawing summary to extract figure types and descriptions.
    
    Args:
        drawing_summary: User-provided description of drawings
        
    Returns:
        List of dicts with figure info:
        [
            {"fig_num": 1, "type": "perspective", "is_physical": True, 
             "description": "exterior housing", "can_generate": False},
            {"fig_num": 2, "type": "cross-section", "is_physical": True,
             "description": "internal arrangement", "can_generate": False},
            ...
        ]
    """
    if not drawing_summary or not drawing_summary.strip():
        return []
    
    figures = []
    
    # Pattern to match figures: "FIG. 1", "Figure 1", "Fig 1:"
    fig_pattern = re.compile(
        r'(?:FIG\.?\s*|Figure\s*|Fig\.?\s*)(\d+)[:\s]+(.+?)(?=(?:FIG\.?\s*|Figure\s*|Fig\.?\s*)\d+|$)',
        re.IGNORECASE | re.DOTALL
    )
    
    matches = fig_pattern.findall(drawing_summary)
    
    if not matches:
        # Try simpler pattern for numbered lists
        simple_pattern = re.compile(r'(\d+)[.)\s]+(.+?)(?=\d+[.)\s]|$)', re.DOTALL)
        matches = simple_pattern.findall(drawing_summary)
    
    for fig_num_str, description in matches:
        fig_num = int(fig_num_str)
        description = description.strip()
        description_lower = description.lower()
        
        # Detect view type
        view_type = "unknown"
        is_physical = False
        can_generate = False
        
        # Check for physical view types
        for pv in PHYSICAL_VIEW_TYPES:
            if pv in description_lower:
                view_type = pv
                is_physical = True
                can_generate = False
                break
        
        # Check for schematic types (if not already identified as physical)
        if view_type == "unknown":
            for sv in SCHEMATIC_VIEW_TYPES:
                if sv in description_lower:
                    view_type = sv
                    is_physical = False
                    can_generate = True
                    break
        
        # Default to schematic if no type detected
        if view_type == "unknown":
            # Look for keywords to determine default
            if any(kw in description_lower for kw in ['view', 'showing', 'illustrating']):
                view_type = "schematic"
                can_generate = True
            else:
                view_type = "block diagram"
                can_generate = True
        
        figures.append({
            "fig_num": fig_num,
            "type": view_type,
            "is_physical": is_physical,
            "can_generate": can_generate,
            "description": description[:200]  # Truncate long descriptions
        })
    
    return figures


def get_drawing_warnings(parsed_figures: List[Dict]) -> List[str]:
    """
    Generate warnings for figures that cannot be auto-generated.
    Detects specification-drawing inconsistencies when physical views are declared.
    
    Args:
        parsed_figures: Output from parse_drawing_summary()
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    physical_figs = [f for f in parsed_figures if f.get("is_physical", False)]
    
    if physical_figs:
        fig_nums = [str(f["fig_num"]) for f in physical_figs]
        view_types = list(set(f["type"] for f in physical_figs))
        
        # CRITICAL CONSISTENCY WARNING
        warnings.append(
            f"⚠️ SPECIFICATION-DRAWING INCONSISTENCY DETECTED:\n"
            f"Your Brief Description declares Figures {', '.join(fig_nums)} as physical views ({', '.join(view_types)}).\n"
            f"However, the system can ONLY generate schematic diagrams (block/flowchart/sequence).\n"
            f"You MUST create these physical drawings manually using CAD software before filing."
        )
        
        # Detailed guidance for each physical view type
        for fig in physical_figs:
            view_type = fig.get("type", "").lower()
            fig_num = fig.get("fig_num")
            desc = fig.get("description", "")[:50]
            
            if "cross-section" in view_type or "sectional" in view_type:
                warnings.append(
                    f"Figure {fig_num} ({view_type}): Create using CAD. Show internal arrangement, layer order, "
                    f"component spacing. Include dimension lines and hatching for cut surfaces."
                )
            elif "exploded" in view_type:
                warnings.append(
                    f"Figure {fig_num} ({view_type}): Create using CAD. Show disassembled components with "
                    f"alignment lines indicating assembly relationships. Number each part."
                )
            elif "isometric" in view_type or "perspective" in view_type:
                warnings.append(
                    f"Figure {fig_num} ({view_type}): Create using CAD. Show 3D view with major features visible. "
                    f"Include reference numerals pointing to key components."
                )
            elif "elevation" in view_type or "side" in view_type or "front" in view_type:
                warnings.append(
                    f"Figure {fig_num} ({view_type}): Create using CAD. Show orthographic projection with "
                    f"dimensions and component labels."
                )
        
        warnings.append(
            "📋 AUTO-GENERATED DIAGRAMS BELOW ARE SUPPLEMENTARY ONLY.\n"
            "They show logical relationships, NOT physical structure.\n"
            "Do NOT use them as replacements for the physical views declared in your specification."
        )
    
    return warnings


def generate_placeholder_for_physical_view(fig_num: int, view_type: str, description: str) -> str:
    """
    Generate a placeholder message for physical views that can't be auto-generated.
    """
    return f"""```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    FIGURE {fig_num}                              │
│                                                             │
│             {view_type.upper()} VIEW                              │
│                                                             │
│   [INSERT {view_type.upper()} DRAWING HERE]                         │
│                                                             │
│   Description: {description[:50]}...                          │
│                                                             │
│   This view requires CAD/illustration software.            │
│   Suggested tools: AutoCAD, SolidWorks, Adobe Illustrator  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```"""


def generate_elaborate_block_diagram_with_llm(abstract: str, components: Dict) -> str:
    """
    Use LLM to generate an elaborate, invention-specific Mermaid block diagram.
    This creates tailored diagrams instead of generic templates.
    
    Args:
        abstract: Patent abstract text
        components: Dict with main_system, components, inputs, outputs
        
    Returns:
        Complete Mermaid code for an elaborate block diagram
    """
    main_system = components.get("main_system", "System")
    comp_list = components.get("components", [])[:6]
    inputs = components.get("inputs", [])[:2]
    outputs = components.get("outputs", [])[:2]
    
    # Build component reference table
    ref_table = ""
    for i, comp in enumerate(comp_list):
        ref_table += f"  - ({20 + i*10}) {comp}\n"
    
    prompt = f"""Create an elaborate Mermaid flowchart diagram for this patent invention.

INVENTION: {main_system}

COMPONENTS (with reference numerals):
{ref_table}

INPUTS: {', '.join(inputs) if inputs else 'User Input, Data Input'}
OUTPUTS: {', '.join(outputs) if outputs else 'Processed Output'}

ABSTRACT CONTEXT:
{abstract[:500]}

CREATE a professional Mermaid flowchart with:
1. Main system container with the invention name and (10) reference
2. All components as labeled boxes with their reference numerals
3. Logical connections showing data/signal flow between components
4. Input nodes (rounded) on the left
5. Output nodes (rounded) on the right  
6. Decision diamonds if there are conditional paths
7. Subgroups for related components

FORMAT RULES:
- Use "flowchart TB" or "flowchart LR" for layout
- Put reference numerals in parentheses like (10), (20), (30)
- Use --> for solid arrows, -.-> for dashed arrows
- Use subgraph for grouping related components
- Add descriptive edge labels like |data flow| or |signal|

Return ONLY valid Mermaid code starting with ```mermaid and ending with ```.
NO explanations, just the Mermaid diagram code.

```mermaid"""

    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=1200,
            temperature=0.3,
            system_prompt="You are a patent diagram expert. Generate clean, professional Mermaid diagrams. Return ONLY Mermaid code."
        )
        
        if response:
            # Clean and validate the response
            mermaid_code = response.strip()
            
            # Ensure it starts with ```mermaid
            if not mermaid_code.startswith("```mermaid"):
                mermaid_code = "```mermaid\n" + mermaid_code
            
            # Ensure it ends with ```
            if not mermaid_code.rstrip().endswith("```"):
                mermaid_code = mermaid_code.rstrip() + "\n```"
            
            # Add figure caption
            mermaid_code += f"\n\n**FIG. 1** — Detailed block diagram illustrating the system architecture and component interconnections of {main_system}, in accordance with a preferred embodiment of the present invention."
            
            return mermaid_code
            
    except Exception as e:
        print(f"LLM diagram generation error: {e}")
    
    # Fallback to template-based diagram
    return None


def generate_elaborate_flowchart_with_llm(abstract: str, components: Dict) -> str:
    """
    Use LLM to generate an elaborate, invention-specific Mermaid flowchart.
    Creates detailed process flow diagrams tailored to the invention.
    """
    main_system = components.get("main_system", "System")
    steps = components.get("process_steps", [])[:8]
    
    steps_text = "\n".join(f"  - S{101+i}: {step}" for i, step in enumerate(steps))
    
    # CRITICAL: If no steps extracted, fail loudly
    if not steps:
        return None  # Force fallback to error diagram
    
    prompt = f"""Create an elaborate Mermaid flowchart for this patent method/process.

INVENTION: {main_system}

PROCESS STEPS (USE EXACTLY THESE - DO NOT INVENT NEW ONES):
{steps_text}

ABSTRACT:
{abstract[:400]}

CREATE a professional Mermaid flowchart with:
1. START node (rounded rectangle)
2. Each step as a labeled box with step number (S101, S102, etc.)
3. Decision diamonds for validation/conditional steps with Yes/No paths
4. Parallel processing paths if applicable
5. Error handling path that loops back
6. END node (rounded rectangle)

FORMAT RULES:
- Use "flowchart TD" for top-down layout
- Step boxes: S101["S101: Step Description"]
- Decision diamonds: D1{{Decision?}}
- Arrows with labels: -->|Yes| or -->|No|
- START/END: START(["START"]) and END(["END"])

Return ONLY valid Mermaid code starting with ```mermaid and ending with ```.

```mermaid"""

    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=1000,
            temperature=0.3,
            system_prompt="You are a patent diagram expert. Generate clean Mermaid flowcharts. Return ONLY Mermaid code."
        )
        
        if response:
            mermaid_code = response.strip()
            
            if not mermaid_code.startswith("```mermaid"):
                mermaid_code = "```mermaid\n" + mermaid_code
            
            if not mermaid_code.rstrip().endswith("```"):
                mermaid_code = mermaid_code.rstrip() + "\n```"
            
            mermaid_code += f"\n\n**FIG. 2** — Flowchart depicting the method of operation of {main_system}, illustrating the process steps in accordance with a preferred embodiment of the present invention."
            
            return mermaid_code
            
    except Exception as e:
        print(f"LLM flowchart generation error: {e}")
    
    return None







def extract_invention_components(abstract: str, claim_1_text: str = None) -> Dict:
    """
    Extract key components from the abstract OR Claim 1 for diagram generation.
    Works for ANY invention type - hardware, software, chemical, biological, etc.
    If claim_1_text is provided, use Claim 1 as authoritative source (IPO requirement).
    """
    # CRITICAL: If Claim 1 is provided, use it as authoritative source
    if claim_1_text:
        return _extract_from_claim_1(claim_1_text)
    
    prompt = f"""You are analyzing a patent abstract to extract components for diagram generation.
Your task is to identify the ACTUAL components mentioned in the abstract - DO NOT invent generic components.

ABSTRACT:
{abstract}

CRITICAL RULES:
1. Extract ONLY components that are EXPLICITLY mentioned in the abstract
2. Use the EXACT terminology from the abstract (do not paraphrase)
3. If the abstract mentions "nanofiber mesh" - write "nanofiber mesh", not "filter"
4. If the abstract mentions "UV-C chamber" - write "UV-C chamber", not "light source"
5. DO NOT add generic components like "controller", "processor", "module" unless explicitly mentioned

Extract and return in this EXACT format:

MAIN_SYSTEM: [The main invention name as stated in the abstract]
COMPONENTS: [exact component 1 from abstract], [exact component 2], [exact component 3], [exact component 4]
INPUTS: [what enters the system as per abstract]
OUTPUTS: [what the system produces as per abstract]
PROCESS_STEPS: [step 1 described in abstract], [step 2], [step 3], [step 4]
ACTORS: [who/what interacts with the system as per abstract]

IMPORTANT: If a field cannot be extracted from the abstract, write "Not specified" - do NOT invent generic values.
"""
    
    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.1,  # Lower temperature for more precise extraction
            system_prompt="Extract EXACT components from the abstract. Use the same terminology as the abstract."
        )
        
        if response:
            result = _parse_components(response)
            # Validate - if empty components, try one more time with different approach
            if not result.get("components") or len(result.get("components", [])) < 2:
                return _fallback_extract_components(abstract)
            return result
    except Exception as e:
        print(f"Component extraction error: {e}")
    
    # FAIL LOUDLY - do NOT use generic placeholders
    return _get_error_result("Component extraction failed")


def _extract_from_claim_1(claim_1_text: str) -> Dict:
    """Extract components from Claim 1 for 1:1 diagram consistency."""
    prompt = f"""Extract the EXACT structural components from this patent Claim 1.
Use the EXACT terminology from the claim - do not paraphrase or substitute.

CLAIM 1:
{claim_1_text[:1000]}

Extract and return in this EXACT format:
MAIN_SYSTEM: [the subject from the preamble, e.g., "water purification apparatus"]
COMPONENTS: [exact component 1], [exact component 2], [exact component 3], [exact component 4]
INPUTS: [what enters the system]
OUTPUTS: [what the system produces]
PROCESS_STEPS: [step 1], [step 2], [step 3]

Use ONLY terms that appear in the claim. Do NOT add generic components.
"""
    
    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=400,
            temperature=0.1,
            system_prompt="Extract EXACT components from patent claim. Use identical terminology."
        )
        
        if response:
            result = _parse_components(response)
            if result.get("components"):
                return result
    except Exception as e:
        print(f"Claim 1 extraction error: {e}")
    
    return _get_error_result("Could not extract components from Claim 1")


def _fallback_extract_components(abstract: str) -> Dict:
    """
    Fallback extraction using a simpler, more direct approach.
    Works for ANY input type by extracting noun phrases.
    """
    prompt = f"""List the main physical or functional parts mentioned in this text.

TEXT:
{abstract}

List the parts in this format:
MAIN_SYSTEM: [main invention name]
COMPONENTS: [part1], [part2], [part3], [part4], [part5]

Use the exact words from the text. List at least 3 parts.
"""
    
    try:
        response = llm_generate(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.1,
            system_prompt="List exact parts mentioned in the text."
        )
        
        if response:
            result = _parse_components(response)
            if result.get("components") and len(result.get("components", [])) >= 2:
                return result
    except Exception as e:
        print(f"Fallback extraction error: {e}")
    
    return _get_error_result("Could not extract components - please provide more detailed abstract")


def _get_error_result(message: str) -> Dict:
    """Return error result instead of generic placeholders."""
    return {
        "main_system": f"ERROR: {message}",
        "components": [],
        "inputs": [],
        "outputs": [],
        "process_steps": [],
        "actors": [],
        "error": True
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
    
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("MAIN_SYSTEM:"):
            result["main_system"] = line.split(":", 1)[1].strip()
        elif line.startswith("COMPONENTS:"):
            parts = line.split(":", 1)[1].strip()
            result["components"] = [c.strip() for c in parts.split(",") if c.strip()]
        elif line.startswith("INPUTS:"):
            parts = line.split(":", 1)[1].strip()
            result["inputs"] = [c.strip() for c in parts.split(",") if c.strip()]
        elif line.startswith("OUTPUTS:"):
            parts = line.split(":", 1)[1].strip()
            result["outputs"] = [c.strip() for c in parts.split(",") if c.strip()]
        elif line.startswith("PROCESS_STEPS:"):
            parts = line.split(":", 1)[1].strip()
            result["process_steps"] = [c.strip() for c in parts.split(",") if c.strip()]
        elif line.startswith("ACTORS:"):
            parts = line.split(":", 1)[1].strip()
            result["actors"] = [c.strip() for c in parts.split(",") if c.strip()]
    
    return result


def generate_block_diagram(abstract: str, components: Optional[Dict] = None) -> str:
    """
    Generate a system block diagram (Fig. 1 style).
    Shows main components with structural relationships and flow paths.
    Improved to show spatial arrangement hints, not just logic flow.
    """
    if not components:
        components = extract_invention_components(abstract)
    
    main = components.get("main_system", "System")
    # CRITICAL: NEVER use generic placeholders - use extracted components only
    comps = components.get("components", [])
    if not comps:
        # Return error diagram instead of generic placeholders
        return """```mermaid
graph TB
    ERROR["❌ ERROR: Could not extract components from abstract"]
    NOTE["Please provide more detailed abstract with component names"]
    ERROR --> NOTE
```"""
    inputs = components.get("inputs", ["Input"])
    outputs = components.get("outputs", ["Output"])
    steps = components.get("process_steps", [])
    
    # Determine flow type from abstract/steps for better annotations
    abstract_lower = abstract.lower() if abstract else ""
    flow_type = "signal"  # default
    if any(w in abstract_lower for w in ["air", "filter", "purif", "ventil", "gas"]):
        flow_type = "airflow"
    elif any(w in abstract_lower for w in ["water", "liquid", "fluid", "pump"]):
        flow_type = "fluid flow"
    elif any(w in abstract_lower for w in ["electric", "power", "battery", "solar"]):
        flow_type = "power"
    elif any(w in abstract_lower for w in ["data", "sensor", "signal", "wireless"]):
        flow_type = "data"
    
    def display_name(s, ref_num):
        """Create full display name with reference numeral - NO TRUNCATION"""
        clean_name = s.replace('"', "'").strip()
        return f"{clean_name} ({ref_num})"
    
    # Build Mermaid diagram with STRUCTURAL RELATIONSHIPS
    mermaid = f"""```mermaid
graph TB
    %% NOTE: SCHEMATIC DIAGRAM - Shows logical arrangement only
    %% For physical structure, create CAD drawings separately
    
    subgraph MAIN["{main} (100)"]
        direction TB
"""
    
    # Add components with FULL NAMES and reference numerals
    for i, comp in enumerate(comps[:6], 1):
        comp_id = f"C{i}"
        ref_num = 100 + i * 10  # 110, 120, 130, etc.
        full_name = display_name(comp, ref_num)
        mermaid += f'        {comp_id}["{full_name}"]\n'
    
    mermaid += "    end\n\n"
    
    # Add inputs with flow type annotation
    for i, inp in enumerate(inputs[:2], 1):
        inp_clean = inp.replace('"', "'").strip()
        mermaid += f'    IN{i}(("{inp_clean}")) -->|"{flow_type} in"| C1\n'
    
    # Connect components with DESCRIPTIVE flow annotations
    num_comps = min(len(comps), 6)
    for i in range(1, num_comps):
        # Use step descriptions if available for annotations
        if steps and i-1 < len(steps):
            step_desc = steps[i-1][:30].replace('"', "'")  # Truncate for label only
            mermaid += f'    C{i} -->|"{step_desc}"| C{i+1}\n'
        else:
            mermaid += f'    C{i} -->|"{flow_type}"| C{i+1}\n'
    
    # Add inter-component relationships (bidirectional for control signals)
    if num_comps >= 4:
        # Add control/feedback path from later component to earlier
        mermaid += f'    C{num_comps} -.->|"control signal"| C1\n'
    
    # Add outputs with flow type annotation
    for i, out in enumerate(outputs[:2], 1):
        out_clean = out.replace('"', "'").strip()
        mermaid += f'    C{num_comps} -->|"{flow_type} out"| OUT{i}(("{out_clean}"))\n'
    
    # Add note about schematic nature
    mermaid += """
    %% Styling for visual clarity
    style C1 fill:#e1f5fe,stroke:#01579b
    style C2 fill:#b3e5fc,stroke:#0277bd
    style C3 fill:#81d4fa,stroke:#0288d1
    style C4 fill:#4fc3f7,stroke:#039be5
    style MAIN fill:#fafafa,stroke:#333
```

**⚠️ NOTE: This is a SCHEMATIC diagram showing logical relationships only.**
**For inventions with physical structure, create CAD drawings (cross-sections, exploded views) separately.**"""
    
    return mermaid


def generate_flowchart(abstract: str, components: Optional[Dict] = None) -> str:
    """
    Generate a process flowchart (Fig. 2 style).
    Shows method steps with decision points.
    """
    if not components:
        components = extract_invention_components(abstract)
    
    # CRITICAL: NEVER use generic placeholders - use extracted steps only
    steps = components.get("process_steps", [])
    if not steps:
        # Return error diagram instead of generic placeholders
        return """```mermaid
flowchart TD
    ERROR["❌ ERROR: Could not extract process steps from abstract"]
    NOTE["Please provide more detailed abstract with method/process steps"]
    ERROR --> NOTE
```"""
    
    mermaid = """```mermaid
flowchart TD
    START(["START"]) --> S1
"""
    
    # Add steps with FULL names and step reference numerals
    for i, step in enumerate(steps[:6], 1):
        step_num = f"S{200 + i}"  # S201, S202, etc.
        # Clean step text but do NOT truncate
        step_clean = step.replace('"', "'").strip()
        mermaid += f'    S{i}["{step_num}: {step_clean}"]\n'
    
    # Connect steps with arrows
    for i in range(1, min(len(steps), 6)):
        mermaid += f"    S{i} --> S{i+1}\n"
    
    # Add decision point after step 2 if we have enough steps
    if len(steps) >= 4:
        mermaid += """
    S2 --> D1{"Validation Check?"}
    D1 -->|Pass| S3
    D1 -->|Fail| S1
"""
    
    # End
    last_step = min(len(steps), 6)
    mermaid += f"""
    S{last_step} --> END(["END"])
    
    style START fill:#c8e6c9
    style END fill:#ffcdd2
    style D1 fill:#fff9c4
```"""
    
    return mermaid


def generate_sequence_diagram(abstract: str, components: Optional[Dict] = None) -> str:
    """
    Generate a sequence diagram (Fig. 3 style).
    Shows interactions between components over time.
    """
    if not components:
        components = extract_invention_components(abstract)
    
    # CRITICAL: NEVER use generic placeholders
    comps = components.get("components", [])
    if not comps or len(comps) < 2:
        return """```mermaid
sequenceDiagram
    participant ERROR as ❌ ERROR
    participant NOTICE as Could not extract components
    ERROR->>NOTICE: Please provide detailed abstract
    NOTICE-->>ERROR: With at least 3 component names
```"""
    
    actors = components.get("actors", [])
    steps = components.get("process_steps", [])
    
    # Use actual components from abstract/claim - NO generic names
    participants = comps[:3]  # Use first 3 components as participants
    
    mermaid = """```mermaid
sequenceDiagram
    autonumber
"""
    
    # Add participants with FULL names - no truncation
    for i, p in enumerate(participants[:3]):
        p_id = f"P{i+1}"
        p_clean = p.replace('"', "'").strip()
        mermaid += f'    participant {p_id} as "{p_clean} ({300 + i*10})"\n'
    
    mermaid += "\n"
    
    # Generate invention-specific interactions
    if len(participants) >= 3:
        step1 = steps[0] if steps else "Transmit data"
        step2 = steps[1] if len(steps) > 1 else "Process received data"
        step3 = steps[2] if len(steps) > 2 else "Return processed result"
        
        mermaid += f"""    P1->>+P2: {step1}
    Note over P2: Processing operation
    P2->>+P3: {step2}
    P3-->>-P2: {step3}
    P2-->>-P1: Transmit response
    Note over P1,P3: Operation sequence complete
```"""
    else:
        mermaid += """    P1->>P2: Send request
    P2-->>P1: Return response
```"""
    
    return mermaid


def generate_all_diagrams(abstract: str, drawing_summary: str = "", claim_1_text: str = None) -> Dict[str, str]:
    """
    Generate all diagram types for a patent application.
    
    Args:
        abstract: Patent abstract text
        drawing_summary: Optional user-provided drawing summary describing figures
        claim_1_text: CRITICAL - Claim 1 text for authoritative component extraction.
                     If provided, diagrams will use EXACTLY the components from Claim 1
                     to ensure Section 10(4)(c) compliance (Claims ↔ Drawings consistency).
    
    Returns dict with:
    - Preview diagrams (colored, for quick visualization)
    - IPO Formal diagrams (black/white with reference numerals, for filing)
    - Warnings if physical views requested
    - Placeholders for views that can't be auto-generated
    """
    # CRITICAL: Extract components from Claim 1 (if provided) for Section 10(4)(c) compliance
    # This ensures every claimed element appears in drawings with reference numerals
    if claim_1_text:
        components = extract_invention_components(abstract, claim_1_text)
        if components.get("error"):
            print(f"WARNING: Claim 1 component extraction failed, falling back to abstract")
            components = extract_invention_components(abstract)
    else:
        components = extract_invention_components(abstract)
    
    # Parse drawing summary to detect user-specified view types
    parsed_figures = parse_drawing_summary(drawing_summary) if drawing_summary else []
    warnings = get_drawing_warnings(parsed_figures) if parsed_figures else []
    
    # UNIVERSAL: Add diagram limitations notice for ALL inventions
    # No keyword-based detection - works for any input type
    warnings.insert(0, get_diagram_limitations_notice())
    
    # Check if user requested physical views
    has_physical_views = any(f.get("is_physical", False) for f in parsed_figures)
    
    # Generate IPO-compliant formal diagrams
    ipo_generator = IPODiagramGenerator(components)
    ipo_diagrams = ipo_generator.generate_all_diagrams()
    
    result = {
        # Preview diagrams (colored - for app visualization)
        "fig1_block": generate_block_diagram(abstract, components),
        "fig2_flowchart": generate_flowchart(abstract, components),
        "fig3_sequence": generate_sequence_diagram(abstract, components),
        
        # IPO Formal diagrams (black/white - for patent filing)
        "fig1_block_ipo": ipo_diagrams.get("fig1_block_ipo", ""),
        "fig2_flowchart_ipo": ipo_diagrams.get("fig2_flowchart_ipo", ""),
        "fig3_sequence_ipo": ipo_diagrams.get("fig3_sequence_ipo", ""),
        
        # Reference numerals and descriptions
        "reference_numerals": ipo_diagrams.get("reference_numerals", ""),
        "brief_description_ipo": ipo_diagrams.get("brief_description", ""),
        
        # Components for other uses (from Claim 1 if provided)
        "components": components,
        "claim_1_synced": claim_1_text is not None,  # Flag indicating Claim-Drawing sync
        
        # Metadata about drawing types
        "parsed_figures": parsed_figures,
        "has_physical_views": has_physical_views,
        "drawing_warnings": warnings,
    }
    
    # DYNAMIC DETECTION: Use LLM to determine if invention needs physical drawings
    # This is NOT keyword-based - it analyzes the abstract semantically
    invention_type = _detect_invention_type_with_llm(abstract)
    result["invention_type"] = invention_type
    
    # If invention is PHYSICAL, generate actual structural drawings
    if invention_type == "physical" and PHYSICAL_DRAWING_AVAILABLE:
        try:
            physical_result = generate_physical_drawings(abstract, claim_1_text, components)
            if physical_result.get("success"):
                result["physical_drawings"] = physical_result["drawings"]
                result["physical_brief_description"] = physical_result.get("brief_description", "")
                result["has_physical_drawings"] = True
                # Update warnings to note physical drawings were generated
                result["drawing_warnings"].append(
                    "✅ Physical structural drawings generated (Pre-CAD drafts - finalize in CAD for filing)"
                )
            else:
                result["has_physical_drawings"] = False
                result["drawing_warnings"].append(
                    f"⚠️ Physical drawing generation failed: {physical_result.get('error', 'Unknown error')}"
                )
        except Exception as e:
            result["has_physical_drawings"] = False
            result["drawing_warnings"].append(f"⚠️ Physical drawing error: {str(e)}")
    elif invention_type == "physical":
        # Physical invention but generator not available
        result["has_physical_drawings"] = False
        result["drawing_warnings"].append(
            "⚠️ This is a PHYSICAL invention. IPO requires structural drawings. "
            "Use the 'AI Physical Drawings' section to generate them."
        )
    else:
        # Software/method invention - schematic diagrams are sufficient
        result["has_physical_drawings"] = False
    
    # Add placeholders for user-requested physical views if not already generated
    if has_physical_views and not result.get("has_physical_drawings"):
        result["physical_view_placeholders"] = {}
        for fig in parsed_figures:
            if fig.get("is_physical", False):
                result["physical_view_placeholders"][f"fig{fig['fig_num']}"] = \
                    generate_placeholder_for_physical_view(
                        fig["fig_num"], 
                        fig["type"], 
                        fig["description"]
                    )
    
    return result


def _detect_invention_type_with_llm(abstract: str) -> str:
    """
    Use LLM to detect if invention is physical or software/method.
    NO KEYWORDS - purely semantic analysis.
    
    Returns: 'physical' or 'software_method'
    """
    if not abstract or len(abstract.strip()) < 20:
        return "software_method"  # Default to schematic-only
    
    try:
        prompt = f"""Analyze this invention and classify it.

INVENTION ABSTRACT:
{abstract[:500]}

QUESTION: Does this invention have PHYSICAL STRUCTURE (hardware, apparatus, device, mechanical parts)?
Answer ONLY with one word: PHYSICAL or SOFTWARE

Answer:"""
        
        response = llm_generate(prompt, max_new_tokens=10, temperature=0.1)
        
        if response:
            response_lower = response.strip().lower()
            if "physical" in response_lower:
                return "physical"
            elif "software" in response_lower:
                return "software_method"
        
        # If unclear, default to physical (safer - generates more drawings)
        return "physical"
        
    except Exception as e:
        print(f"Invention type detection failed: {e}")
        return "physical"  # Default to physical (safer)


def get_figure_descriptions(diagrams: Dict) -> str:
    """
    Generate Brief Description of Drawings text for the figures.
    IPO format compliant with proper reference numerals.
    """
    # If IPO brief description is available, use it
    if diagrams.get("brief_description_ipo"):
        return diagrams.get("brief_description_ipo")
    
    # Fallback to generated description
    components = diagrams.get("components", {})
    main_system = components.get("main_system", "the invention")
    comps = components.get("components", [])
    steps = components.get("process_steps", [])
    
    # Use reference numerals if available
    ref_table = diagrams.get("reference_numerals", "")
    
    description = """BRIEF DESCRIPTION OF THE DRAWINGS

The accompanying drawings illustrate the preferred embodiment of the present invention:

Figure 1: illustrates a block diagram of {main_system} showing the arrangement and interconnection of the main components according to the present invention.

Figure 2: illustrates a flowchart depicting the method of operation of {main_system} according to the present invention.

Figure 3: illustrates a sequence diagram showing the interaction between components during operation, demonstrating the data flow and communication protocol according to the present invention.

""".format(main_system=main_system)
    
    # Add reference numerals
    if ref_table:
        description += ref_table
    else:
        description += "REFERENCE NUMERALS\n\n"
        for i, comp in enumerate(comps[:6]):
            ref_num = 10 + i * 10  # Use 10-series for IPO consistency
            description += f"{ref_num} - {comp}\n"
    
    return description


if __name__ == "__main__":
    # Test with sample abstract
    test_abstract = """
    A smart monitoring system for industrial environments, comprising: 
    a plurality of sensor units distributed across a facility; 
    a central processing hub configured to analyze data using machine learning; 
    and an alert module for real-time anomaly detection.
    """
    
    print("Generating Patent Diagrams...")
    print("=" * 60)
    
    diagrams = generate_all_diagrams(test_abstract)
    
    print("\n📊 FIGURE 1 - Block Diagram:")
    print(diagrams["fig1_block"])
    
    print("\n📈 FIGURE 2 - Flowchart:")
    print(diagrams["fig2_flowchart"])
    
    print("\n🔄 FIGURE 3 - Sequence Diagram:")
    print(diagrams["fig3_sequence"])
    
    print("\n📝 Figure Descriptions:")
    print(get_figure_descriptions(diagrams))
