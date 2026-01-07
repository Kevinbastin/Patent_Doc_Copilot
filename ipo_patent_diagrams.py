"""
IPO-Compliant Patent Diagram Generator
=======================================
Generates patent drawings that comply with Indian Patent Office (IPO) specifications
per Rule 15 of the Patents Rules, 2003.

Features:
- Black/white line drawings only (no colors)
- Proper reference numeral system (100, 110, 120...)
- Standard patent flowchart notation (S101, S102...)
- A4 page sizing with proper margins
- PNG/SVG export capability
"""

from typing import Dict, List, Tuple, Optional
from llm_runtime import llm_generate
import re


class IPODiagramGenerator:
    """
    Generate IPO-compliant patent diagrams with proper reference numerals
    and formatting according to Indian Patent Office standards.
    """
    
    # Reference numeral configuration - 10-series for consistency
    REF_NUM_START = 10
    REF_NUM_INCREMENT = 10
    
    # Step numbering for flowcharts
    STEP_PREFIX = "S"
    STEP_NUM_START = 101  # Keep step numbers in 100-series (S101, S102...)
    
    def __init__(self, components: Dict = None):
        """
        Initialize with invention components.
        
        Args:
            components: Dictionary with keys:
                - main_system: str
                - components: List[str]
                - inputs: List[str]
                - outputs: List[str]
                - process_steps: List[str]
                - actors: List[str]
        """
        self.components = components or {}
        self.reference_numerals: Dict[str, int] = {}
        self.step_numbers: Dict[str, str] = {}
        
        if components:
            self._assign_reference_numerals()
            self._assign_step_numbers()
    
    def _assign_reference_numerals(self):
        """Assign standard patent reference numerals to components."""
        ref_num = self.REF_NUM_START
        
        # Main system gets first number (100)
        main_system = self.components.get("main_system", "System")
        self.reference_numerals[main_system] = ref_num
        ref_num += self.REF_NUM_INCREMENT
        
        # Components get sequential numbers (110, 120, 130...)
        for comp in self.components.get("components", [])[:10]:
            if comp not in self.reference_numerals:
                self.reference_numerals[comp] = ref_num
                ref_num += self.REF_NUM_INCREMENT
        
        # Inputs get numbers (continuing sequence)
        for inp in self.components.get("inputs", [])[:3]:
            if inp not in self.reference_numerals:
                self.reference_numerals[inp] = ref_num
                ref_num += self.REF_NUM_INCREMENT
        
        # Outputs get numbers
        for out in self.components.get("outputs", [])[:3]:
            if out not in self.reference_numerals:
                self.reference_numerals[out] = ref_num
                ref_num += self.REF_NUM_INCREMENT
    
    def _assign_step_numbers(self):
        """Assign step numbers for flowchart (S101, S102...)."""
        step_num = self.STEP_NUM_START
        
        for step in self.components.get("process_steps", [])[:10]:
            self.step_numbers[step] = f"{self.STEP_PREFIX}{step_num}"
            step_num += 1
    
    def generate_block_diagram(self) -> str:
        """
        Generate IPO-compliant block diagram (Figure 1).
        Premium professional layout with proper component hierarchy.
        
        Returns:
            Mermaid code for black/white block diagram with reference numerals.
        """
        main = self.components.get("main_system", "System")
        comps = self.components.get("components", ["Module A", "Module B", "Module C"])
        inputs = self.components.get("inputs", ["Input"])
        outputs = self.components.get("outputs", ["Output"])
        
        # Get reference numerals
        main_ref = self.reference_numerals.get(main, 10)
        
        # Clean and format names for professional display
        def format_name(name, max_len=22):
            name = name.strip()
            if len(name) > max_len:
                return name[:max_len-2] + "..."
            return name
        
        # Build premium Mermaid diagram
        mermaid = f"""```mermaid
%%{{init: {{
  'theme': 'base',
  'themeVariables': {{
    'primaryColor': '#ffffff',
    'primaryBorderColor': '#000000',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'fontFamily': 'Times New Roman, serif',
    'fontSize': '12px',
    'edgeLabelBackground': '#ffffff'
  }},
  'flowchart': {{
    'nodeSpacing': 50,
    'rankSpacing': 60,
    'curve': 'basis',
    'padding': 20
  }}
}}}}%%
flowchart TB
    %% ═══════════════════════════════════════════════════════════
    %% FIGURE 1: SYSTEM BLOCK DIAGRAM
    %% ═══════════════════════════════════════════════════════════
    
    subgraph SYSTEM["<b>{format_name(main, 35)}</b><br/>Reference: ({main_ref})"]
        direction LR
"""
        
        # Create organized component layout
        num_comps = min(len(comps), 6)
        
        # Row 1: First 3 components
        if num_comps >= 1:
            mermaid += "\n        %% Row 1: Primary Components\n"
            for i in range(min(3, num_comps)):
                comp = comps[i]
                comp_ref = self.reference_numerals.get(comp, 20 + i * 10)
                mermaid += f'        C{i+1}["<b>{format_name(comp)}</b><br/>({comp_ref})"]\n'
        
        # Row 2: Next 3 components
        if num_comps > 3:
            mermaid += "\n        %% Row 2: Secondary Components\n"
            for i in range(3, min(6, num_comps)):
                comp = comps[i]
                comp_ref = self.reference_numerals.get(comp, 20 + i * 10)
                mermaid += f'        C{i+1}["<b>{format_name(comp)}</b><br/>({comp_ref})"]\n'
        
        mermaid += "    end\n\n"
        
        # Input signals
        mermaid += "    %% Input Signals\n"
        for i, inp in enumerate(inputs[:2]):
            inp_ref = self.reference_numerals.get(inp, 100 + i * 10)
            mermaid += f'    INPUT{i+1}(("{format_name(inp, 12)}"))\n'
        
        # Output signals  
        mermaid += "\n    %% Output Signals\n"
        for i, out in enumerate(outputs[:2]):
            out_ref = self.reference_numerals.get(out, 110 + i * 10)
            mermaid += f'    OUTPUT{i+1}(("{format_name(out, 12)}"))\n'
        
        # Connection logic
        mermaid += "\n    %% Signal Flow Connections\n"
        
        # Inputs connect to first component
        for i in range(min(2, len(inputs))):
            mermaid += f"    INPUT{i+1} ==> C1\n"
        
        # Internal component connections (grid pattern)
        if num_comps >= 2:
            mermaid += "    C1 --> C2\n"
        if num_comps >= 3:
            mermaid += "    C2 --> C3\n"
        if num_comps >= 4:
            mermaid += "    C1 -.-> C4\n"
            mermaid += "    C4 --> C5\n" if num_comps >= 5 else ""
        if num_comps >= 5:
            mermaid += "    C3 <--> C5\n"
        if num_comps >= 6:
            mermaid += "    C5 --> C6\n"
            mermaid += "    C4 -.-> C6\n"
        
        # Outputs from last components
        last = min(num_comps, 3)
        for i in range(min(2, len(outputs))):
            mermaid += f"    C{last} ==> OUTPUT{i+1}\n"
        
        # Premium IPO-compliant styling
        mermaid += """
    %% ═══════════════════════════════════════════════════════════
    %% IPO-COMPLIANT STYLING (Rule 15, Patents Rules 2003)
    %% ═══════════════════════════════════════════════════════════
    
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000,font-weight:normal
    classDef systemBox fill:#fafafa,stroke:#000000,stroke-width:3px,color:#000000
    classDef inputOutput fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    
    class SYSTEM systemBox
    class INPUT1,INPUT2,OUTPUT1,OUTPUT2 inputOutput
    
    linkStyle default stroke:#000000,stroke-width:2px
```

**FIG. 1** — Block diagram illustrating the overall system architecture and component interconnections of the present invention, in accordance with a preferred embodiment."""
        
        return mermaid
    
    def generate_flowchart(self) -> str:
        """
        Generate IPO-compliant process flowchart (Figure 2).
        Professional layout with clear step numbering and decision points.
        
        Returns:
            Mermaid code for flowchart with step numbers (S101, S102...).
        """
        steps = self.components.get("process_steps", ["Step 1", "Step 2", "Step 3", "Step 4"])
        
        # Truncate long step names
        def truncate(text, max_len=35):
            return text[:max_len] + "..." if len(text) > max_len else text
        
        # Build professional Mermaid flowchart
        mermaid = f"""```mermaid
%%{{init: {{
  'theme': 'base',
  'themeVariables': {{
    'primaryColor': '#ffffff',
    'primaryBorderColor': '#000000',
    'primaryTextColor': '#000000',
    'lineColor': '#000000',
    'fontFamily': 'Arial, sans-serif'
  }},
  'flowchart': {{
    'nodeSpacing': 30,
    'rankSpacing': 40,
    'curve': 'basis'
  }}
}}}}%%
flowchart TD
    START(["<b>START</b>"])
"""
        
        # Add steps with proper formatting
        num_steps = min(len(steps), 6)
        for i, step in enumerate(steps[:num_steps]):
            step_num = f"S{self.STEP_NUM_START + i}"
            step_clean = truncate(step)
            mermaid += f'    {step_num}["{step_num}<br/>{step_clean}"]\n'
        
        # Create sequential connections
        mermaid += "\n    %% Sequential Flow\n"
        mermaid += f"    START --> S{self.STEP_NUM_START}\n"
        
        for i in range(num_steps - 1):
            step_num = f"S{self.STEP_NUM_START + i}"
            next_step = f"S{self.STEP_NUM_START + i + 1}"
            mermaid += f"    {step_num} --> {next_step}\n"
        
        # Add decision point if we have enough steps
        if num_steps >= 3:
            mermaid += f"""
    %% Decision Point
    S{self.STEP_NUM_START + 1} --> D1{{{"Condition<br/>Valid?"}}}
    D1 -->|Yes| S{self.STEP_NUM_START + 2}
    D1 -->|No| S{self.STEP_NUM_START}
"""
        
        # End connection
        last_step = f"S{self.STEP_NUM_START + num_steps - 1}"
        mermaid += f"""
    %% Termination
    {last_step} --> END(["<b>END</b>"])
    
    %% IPO-Compliant Styling
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    classDef startend fill:#e8e8e8,stroke:#000000,stroke-width:2px,color:#000000
    classDef decision fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
    
    class START,END startend
    class D1 decision
```

**FIG. 2** - Flowchart depicting the method of operation according to the present invention"""
        
        return mermaid
    
    def generate_sequence_diagram(self) -> str:
        """
        Generate IPO-compliant sequence diagram (Figure 3).
        
        Returns:
            Mermaid code for sequence diagram showing interactions.
        """
        actors = self.components.get("actors", ["User", "System"])
        comps = self.components.get("components", ["Module A", "Module B"])[:3]
        
        # Ensure we have at least 2 participants
        participants = actors[:1] + comps[:2]
        if len(participants) < 3:
            participants = ["User", "Controller", "Processor"]
        
        # Build Mermaid sequence diagram
        mermaid = """```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryBorderColor': '#000000', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'actorBorder': '#000000', 'actorBkg': '#ffffff', 'actorTextColor': '#000000' }}}%%
sequenceDiagram
    autonumber
"""
        
        # Add participants with reference numerals
        for i, p in enumerate(participants[:3]):
            p_clean = p[:15].replace(' ', '_')
            ref_num = self.reference_numerals.get(p, 10 + i * 10)
            mermaid += f"    participant {p_clean} as {p} ({ref_num})\n"
        
        mermaid += "\n"
        
        # Generate interactions
        p1, p2, p3 = [p[:15].replace(' ', '_') for p in participants[:3]]
        
        mermaid += f"""    {p1}->>+{p2}: Request data
    Note over {p2}: Process input
    {p2}->>+{p3}: Forward request
    {p3}-->>-{p2}: Return result
    {p2}-->>-{p1}: Send response
    Note over {p1},{p3}: Transaction complete
```

**Fig. 3** - Sequence diagram illustrating interaction between components according to the present invention"""
        
        return mermaid
    
    def generate_detailed_block_diagram(self, subsystem_name: str = None) -> str:
        """
        Generate detailed view of a subsystem (Figure 4+).
        
        Args:
            subsystem_name: Name of subsystem to detail (uses first component if None)
        
        Returns:
            Mermaid code for detailed subsystem diagram.
        """
        comps = self.components.get("components", ["Module A"])
        
        if not subsystem_name and comps:
            subsystem_name = comps[0]
        
        ref_num = self.reference_numerals.get(subsystem_name, 20)
        
        # Generate sub-components
        sub_comps = [f"{subsystem_name} Unit {i+1}" for i in range(4)]
        
        mermaid = f"""```mermaid
%%{{init: {{'theme': 'base', 'themeVariables': {{ 'primaryColor': '#ffffff', 'primaryBorderColor': '#000000', 'primaryTextColor': '#000000', 'lineColor': '#000000' }}}}}}%%
graph LR
    subgraph SUB["{subsystem_name} ({ref_num})"]
        SUB1["{sub_comps[0]} ({ref_num}a)"]
        SUB2["{sub_comps[1]} ({ref_num}b)"]
        SUB3["{sub_comps[2]} ({ref_num}c)"]
        SUB4["{sub_comps[3]} ({ref_num}d)"]
        
        SUB1 --> SUB2
        SUB2 --> SUB3
        SUB3 --> SUB4
    end
    
    IN([Input]) --> SUB1
    SUB4 --> OUT([Output])
    
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000
```

**Fig. 4** - Detailed block diagram of {subsystem_name} according to the present invention"""
        
        return mermaid
    
    def get_reference_table(self) -> str:
        """
        Generate reference numeral table for the patent specification.
        
        Returns:
            Formatted reference numeral table text.
        """
        table = "REFERENCE NUMERALS\n\n"
        table += "=" * 50 + "\n"
        table += f"{'Numeral':<12} {'Description':<38}\n"
        table += "=" * 50 + "\n"
        
        # Sort by reference number
        sorted_refs = sorted(self.reference_numerals.items(), key=lambda x: x[1])
        
        for name, ref_num in sorted_refs:
            table += f"{ref_num:<12} {name:<38}\n"
        
        table += "=" * 50 + "\n"
        
        # Add step numbers if available
        if self.step_numbers:
            table += "\nSTEP NUMBERS (Flowchart)\n"
            table += "-" * 50 + "\n"
            for step, step_num in self.step_numbers.items():
                step_short = step[:40] if len(step) > 40 else step
                table += f"{step_num:<12} {step_short}\n"
            table += "-" * 50 + "\n"
        
        return table
    
    def get_brief_description_of_drawings(self) -> str:
        """
        Generate IPO-compliant Brief Description of Drawings section.
        
        Returns:
            Formatted brief description text following IPO format.
        """
        main_system = self.components.get("main_system", "the present invention")
        
        description = """BRIEF DESCRIPTION OF THE DRAWINGS

The accompanying drawings illustrate the preferred embodiment of the present invention:

"""
        # Figure 1 - Block Diagram
        components_list = self.components.get("components", [])
        comp_refs = [f"({self.reference_numerals.get(c, 10)}) {c}" for c in components_list[:4]]
        comp_text = ", ".join(comp_refs) if comp_refs else "the main components"
        
        description += f"""Figure 1: illustrates a block diagram of {main_system} showing the arrangement and interconnection of {comp_text} according to the present invention.

"""
        
        # Figure 2 - Flowchart
        steps = self.components.get("process_steps", [])
        steps_text = ", ".join(steps[:3]) if steps else "the main process steps"
        
        description += f"""Figure 2: illustrates a flowchart depicting the method of operation of {main_system}, including the steps of {steps_text} according to the present invention.

"""
        
        # Figure 3 - Sequence Diagram
        description += f"""Figure 3: illustrates a sequence diagram showing the interaction between components during operation of {main_system}, demonstrating the data flow and communication protocol according to the present invention.

"""
        
        # Reference numerals section
        description += self.get_reference_table()
        
        return description
    
    def generate_all_diagrams(self) -> Dict[str, str]:
        """
        Generate all IPO-compliant diagrams.
        
        Returns:
            Dictionary containing all diagram types and reference table.
        """
        return {
            "fig1_block_ipo": self.generate_block_diagram(),
            "fig2_flowchart_ipo": self.generate_flowchart(),
            "fig3_sequence_ipo": self.generate_sequence_diagram(),
            "reference_numerals": self.get_reference_table(),
            "brief_description": self.get_brief_description_of_drawings()
        }


def extract_invention_components_for_ipo(abstract: str) -> Dict:
    """
    Extract key components from abstract for IPO diagram generation.
    Uses LLM for intelligent extraction of COMPLETE component names.
    
    Args:
        abstract: Patent abstract text
    
    Returns:
        Dictionary with extracted components
    """
    prompt = f"""Analyze this patent abstract and extract components for IPO-compliant diagram generation.

ABSTRACT:
{abstract}

CRITICAL RULES:
1. Use COMPLETE names from the abstract (e.g., "lightweight frame" not just "lightweight")
2. Include 2-4 word descriptive names (e.g., "transparent display system" not just "display")
3. Extract the EXACT terminology used in the abstract
4. For PROCESS_STEPS, use verb phrases (e.g., "performing object recognition")

Extract:
1. MAIN_SYSTEM: The full invention name (e.g., "AI-enhanced augmented reality glasses")
2. COMPONENTS: 4-6 key physical/software components with complete names
3. INPUTS: Data/signals entering the system
4. OUTPUTS: Results/signals produced
5. PROCESS_STEPS: 4-6 main method steps as verb phrases
6. ACTORS: Who/what interacts (user, device, server)

Return in this EXACT format:
MAIN_SYSTEM: [complete invention name]
COMPONENTS: [full name 1], [full name 2], [full name 3], [full name 4]
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
            system_prompt="Extract invention components precisely for patent drawings. Use technical terminology."
        )
        
        if response:
            return _parse_components_response(response)
    except Exception as e:
        print(f"Component extraction error: {e}")
    
    # Fallback with generic components
    return {
        "main_system": "Invention System",
        "components": ["Processing Unit", "Input Module", "Output Module", "Controller"],
        "inputs": ["User Input", "Data"],
        "outputs": ["Processed Output", "Result"],
        "process_steps": ["Receive Input", "Process Data", "Generate Output", "Return Result"],
        "actors": ["User", "System"]
    }


def _parse_components_response(response: str) -> Dict:
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


def generate_ipo_diagrams(abstract: str, drawing_summary: str = "") -> Dict[str, str]:
    """
    Main function to generate all IPO-compliant patent diagrams.
    
    Args:
        abstract: Patent abstract text
        drawing_summary: User's drawing description (contains their reference numerals)
    
    Returns:
        Dictionary containing:
        - fig1_block_ipo: Block diagram (Mermaid)
        - fig2_flowchart_ipo: Flowchart (Mermaid)
        - fig3_sequence_ipo: Sequence diagram (Mermaid)
        - reference_numerals: Reference numeral table
        - brief_description: Brief Description of Drawings text
        - components: Extracted components dict
    """
    # Extract components from abstract
    components = extract_invention_components_for_ipo(abstract)
    
    # If user provided drawing_summary, extract THEIR reference numerals and use them
    if drawing_summary:
        import re
        # Pattern: "component name (number)" 
        pattern = r'([a-zA-Z][a-zA-Z\s\-]+?)\s*\((\d+)\)'
        matches = re.findall(pattern, drawing_summary, re.IGNORECASE)
        
        user_refs = {}
        for name, num_str in matches:
            # Clean the component name - remove leading verbs and articles
            name_clean = name.strip()
            
            # Remove common leading phrases that aren't part of component name
            remove_phrases = [
                'showing the ', 'shows the ', 'showing ', 'shows ',
                'illustrating the ', 'illustrates the ', 'illustrating ', 'illustrates ',
                'detailing the ', 'details the ', 'detailing ', 'details ',
                'depicting the ', 'depicts the ', 'depicting ', 'depicts ',
                'including the ', 'includes the ', 'including ', 'includes ',
                'and the ', 'and ', 'the ', 'a ', 'an '
            ]
            for phrase in remove_phrases:
                if name_clean.lower().startswith(phrase):
                    name_clean = name_clean[len(phrase):]
                    break
            
            name_clean = name_clean.strip()
            
            # Capitalize properly (Title Case)
            name_clean = name_clean.title()
            
            if len(name_clean) > 2:
                user_refs[name_clean] = int(num_str)
        
        # Override auto-generated reference numerals with user's
        if user_refs:
            # Use user's clean component names
            components["components"] = list(user_refs.keys())[:6]
    
    # Create IPO diagram generator
    generator = IPODiagramGenerator(components)
    
    # If user provided numerals, override the generator's reference numerals
    if drawing_summary:
        import re
        pattern = r'([a-zA-Z][a-zA-Z\s\-]+?)\s*\((\d+)\)'
        matches = re.findall(pattern, drawing_summary, re.IGNORECASE)
        for name, num_str in matches:
            name_clean = name.strip()
            # Apply same cleaning
            remove_phrases = [
                'showing the ', 'shows the ', 'showing ', 'shows ',
                'illustrating the ', 'illustrates the ', 'illustrating ', 'illustrates ',
                'detailing the ', 'details the ', 'detailing ', 'details ',
                'depicting the ', 'depicts the ', 'depicting ', 'depicts ',
                'and the ', 'and ', 'the ', 'a ', 'an '
            ]
            for phrase in remove_phrases:
                if name_clean.lower().startswith(phrase):
                    name_clean = name_clean[len(phrase):]
                    break
            name_clean = name_clean.strip().title()
            if len(name_clean) > 2:
                generator.reference_numerals[name_clean] = int(num_str)
    
    # Generate all diagrams
    diagrams = generator.generate_all_diagrams()
    diagrams["components"] = components
    
    return diagrams


# CLI testing
if __name__ == "__main__":
    print("=" * 70)
    print("    IPO-COMPLIANT PATENT DIAGRAM GENERATOR")
    print("    Indian Patent Office Drawing Standards")
    print("=" * 70)
    
    # Test with sample abstract
    test_abstract = """
    A smart monitoring system for industrial environments, comprising: 
    a plurality of sensor units distributed across a facility; 
    a central processing hub configured to analyze data using machine learning; 
    and an alert module for real-time anomaly detection.
    The method includes receiving sensor data, processing the data through 
    neural network algorithms, detecting anomalies, and generating alerts.
    """
    
    print("\n📥 Test Abstract:")
    print("-" * 70)
    print(test_abstract.strip())
    print("-" * 70)
    
    print("\n⏳ Generating IPO-compliant diagrams...")
    
    diagrams = generate_ipo_diagrams(test_abstract)
    
    print("\n📊 FIGURE 1 - Block Diagram (IPO Format):")
    print("=" * 70)
    print(diagrams.get("fig1_block_ipo", ""))
    
    print("\n📈 FIGURE 2 - Flowchart (IPO Format):")
    print("=" * 70)
    print(diagrams.get("fig2_flowchart_ipo", ""))
    
    print("\n🔄 FIGURE 3 - Sequence Diagram (IPO Format):")
    print("=" * 70)
    print(diagrams.get("fig3_sequence_ipo", ""))
    
    print("\n📋 Reference Numerals:")
    print("=" * 70)
    print(diagrams.get("reference_numerals", ""))
    
    print("\n📝 Brief Description of Drawings:")
    print("=" * 70)
    print(diagrams.get("brief_description", ""))
