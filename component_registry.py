"""
Unified Component Registry
===========================
Single source of truth for component names and reference numerals.
All patent section generators MUST use this registry to ensure consistency.
"""

import re
from typing import Dict, List, Tuple
from llm_runtime import llm_generate


class ComponentRegistry:
    """
    Unified registry for patent components and reference numerals.
    Ensures consistent numbering across ALL patent sections.
    """
    
    def __init__(self, start_num: int = 100, increment: int = 10):
        self.components: Dict[str, int] = {}
        self.next_num = start_num
        self.increment = increment  # Increment between reference numerals (100, 110, 120...)
        self.main_invention: str = ""
        self.invention_field: str = ""
    
    def extract_from_abstract(self, abstract: str) -> Dict[str, int]:
        """
        Extract components from abstract using LLM and assign reference numerals.
        This is the SINGLE source of truth for all patent sections.
        """
        # Clean abstract
        abstract_clean = ' '.join(abstract.split())
        
        # Extract invention name
        match = re.search(r'^A\s+([a-zA-Z\s\-]+?)(?:\s+comprising|\s+for\s+|\s+with\s+|,|\.)', 
                          abstract_clean, re.IGNORECASE)
        if match:
            self.main_invention = match.group(1).strip().lower()
        else:
            self.main_invention = "invention"
        
        # Use LLM to determine field - works for ANY invention type
        field_prompt = f"""What technical field does this invention belong to? Return ONLY a short phrase (2-4 words).

ABSTRACT: {abstract[:300]}

FIELD:"""
        
        try:
            field_response = llm_generate(
                prompt=field_prompt,
                max_new_tokens=30,
                temperature=0.1,
                system_prompt="Return ONLY the field name, nothing else."
            )
            if field_response:
                self.invention_field = field_response.strip().strip('"').strip("'")[:50]
            else:
                self.invention_field = f"{self.main_invention} technology"
        except Exception:
            self.invention_field = f"{self.main_invention} technology"
        
        # Use LLM to extract components with COMPLETE names
        prompt = f"""Extract the main technical components from this patent abstract.
Return ONLY a JSON list of COMPLETE component names (2-4 words each).

ABSTRACT:
{abstract[:800]}

RULES:
1. Use FULL names from the abstract (e.g., "lightweight frame" not just "frame")
2. Include descriptive modifiers (e.g., "transparent display system" not just "display")
3. Extract the EXACT terminology used in the abstract
4. Include the main invention as the first component

Return ONLY a JSON list like: ["Complete Component Name 1", "Complete Component Name 2"]
Maximum 10 components.

JSON LIST:"""

        try:
            response = llm_generate(
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.1,
                system_prompt="You are a patent component extractor. Output ONLY valid JSON."
            )
            
            # Parse JSON
            import json
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                components_list = json.loads(json_match.group())
                
                # Build registry with consistent numbering
                for comp in components_list[:10]:
                    comp_clean = str(comp).strip()
                    if comp_clean and comp_clean not in self.components:
                        self.components[comp_clean] = self.next_num
                        self.next_num += self.increment  # Use consistent increment for IPO format
        except Exception as e:
            print(f"Component extraction error: {e}")
        
        # Fallback if no components extracted
        if not self.components:
            self._create_fallback_components(abstract)
        
        return self.components
    
    def extract_from_claim_1(self, claim_1_text: str) -> Dict[str, int]:
        """
        CRITICAL: Extract components from GENERATED Claim 1 text.
        This is the AUTHORITATIVE source for diagram components.
        Ensures 1:1 mapping between Claim 1 and Diagrams (IPO requirement).
        
        Args:
            claim_1_text: The generated Claim 1 text
            
        Returns:
            Dict mapping component names to reference numerals
        """
        # Clear existing components - Claim 1 is authoritative
        self.components = {}
        self.next_num = self.start_num
        
        # Extract preamble (invention name)
        preamble_match = re.search(r'^1\.\s*(?:An?|The)\s+([^,]+?)(?:\s+comprising|,)', 
                                    claim_1_text, re.IGNORECASE)
        if preamble_match:
            self.main_invention = preamble_match.group(1).strip().lower()
        
        # Use LLM to extract EXACT components from Claim 1
        prompt = f"""Extract the EXACT structural components from this patent Claim 1.
Return ONLY the component NAMES as they appear in the claim (no reference numerals).

CLAIM 1:
{claim_1_text[:1000]}

RULES:
1. Extract component names EXACTLY as written in the claim
2. Include main invention in the list
3. Do NOT include "means for" or functional descriptions
4. Do NOT add components not in the claim
5. Return as JSON list

Example output: ["ring housing", "optical sensor", "processor", "wireless transmitter"]

JSON LIST:"""

        try:
            response = llm_generate(
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.1,
                system_prompt="Extract exact component names from patent claim. Output ONLY valid JSON list."
            )
            
            # Parse JSON
            import json
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                components_list = json.loads(json_match.group())
                
                # Build registry with consistent numbering
                for comp in components_list[:10]:
                    comp_clean = str(comp).strip().lower()
                    if comp_clean and comp_clean not in self.components:
                        self.components[comp_clean] = self.next_num
                        self.next_num += self.increment
        except Exception as e:
            print(f"Claim 1 component extraction error: {e}")
        
        # Fallback: regex extraction from claim structure
        if not self.components:
            # Look for "a/an [component]" patterns
            matches = re.findall(r'(?:a|an)\s+([a-z][a-z\s]{2,30})(?:\s+configured|\s+coupled|\s+comprising|;|,)', 
                                 claim_1_text.lower())
            for match in matches[:10]:
                comp = match.strip()
                if comp and comp not in self.components and len(comp) > 3:
                    self.components[comp] = self.next_num
                    self.next_num += self.increment
        
        # FAIL LOUDLY if no components found - do NOT use generic placeholders
        if not self.components:
            self.components = {"ERROR: No components in Claim 1": 0}
            print("⚠️ WARNING: Could not extract components from Claim 1")
        
        return self.components
    
    def _create_fallback_components(self, abstract: str):
        """Create fallback components from abstract keywords."""
        # Try to extract nouns from "comprising" section
        match = re.search(r'comprising[:\s]+(.+?)(?:wherein|\.)', abstract, re.IGNORECASE | re.DOTALL)
        if match:
            comp_text = match.group(1)
            parts = re.split(r'[;,]|\band\b', comp_text)
            for part in parts[:8]:
                part = part.strip()
                if len(part) > 3:
                    # Clean up
                    part = re.sub(r'^a\s+|^an\s+|^the\s+', '', part, flags=re.IGNORECASE)
                    part = part.split('(')[0].strip()
                    if part and part not in self.components:
                        self.components[part[:40]] = self.next_num
                        self.next_num += self.increment
        
        # FAIL LOUDLY - do NOT use generic placeholders
        if not self.components:
            self.components = {"ERROR: No components extracted": 0}
            print("⚠️ WARNING: Could not extract components from abstract")
    
    def extract_from_drawing_summary(self, drawing_summary: str) -> Dict[str, int]:
        """
        Extract component names and reference numerals from user's drawing summary.
        Uses LLM for intelligent extraction of clean component names.
        
        Returns:
            Dict mapping clean component names to reference numerals
        """
        if not drawing_summary:
            return {}
        
        user_refs = {}
        
        # Use LLM to extract clean component names with their numerals
        prompt = f"""Extract component names and reference numerals from this patent drawing summary.

DRAWING SUMMARY:
{drawing_summary}

RULES:
1. Extract ONLY the component name, not verbs like "showing" or "illustrating"
2. Component names should be 1-4 words maximum
3. Include the reference numeral in parentheses for each component

EXAMPLES:
"showing the housing (100)" → housing: 100
"illustrates the solar panel array (110)" → solar panel array: 110
"detailing the robotic arm (120)" → robotic arm: 120

Return ONLY a list in this exact format, one per line:
component_name: number

START:
"""
        try:
            response = llm_generate(
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.1,
                system_prompt="Extract only component names and numbers. Be very concise."
            )
            
            if response:
                # Parse response
                for line in response.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            name = parts[0].strip().lower()
                            try:
                                num = int(parts[1].strip().replace('(', '').replace(')', ''))
                                if len(name) > 1 and num > 0:
                                    user_refs[name] = num
                            except ValueError:
                                continue
        except Exception as e:
            print(f"LLM extraction error: {e}")
            # Fallback to simple pattern
            import re
            pattern = r'the\s+([a-z]+(?:\s+[a-z]+)?)\s*\((\d+)\)'
            for match in re.finditer(pattern, drawing_summary.lower()):
                name = match.group(1).strip()
                num = int(match.group(2))
                if name not in ['the', 'a', 'an']:
                    user_refs[name] = num
        
        # Store in components
        for name, num in user_refs.items():
            self.components[name] = num
            if num >= self.next_num:
                self.next_num = num + self.increment
        
        return user_refs
    
    def merge_with_user_references(self, abstract: str, drawing_summary: str) -> Dict[str, int]:
        """
        Extract components from both abstract AND drawing_summary.
        User's drawing_summary references take PRIORITY and REPLACE auto-extracted.
        
        This is the main method to call for unified component extraction.
        """
        # First, extract from abstract (auto-generated numerals)
        self.extract_from_abstract(abstract)
        
        # Then, extract user's exact numerals from drawing_summary
        user_refs = self.extract_from_drawing_summary(drawing_summary)
        
        # If user provided references, REPLACE auto-generated with user's
        # (Clear duplicates by only keeping user's version)
        if user_refs:
            # Keep auto-extracted components that don't conflict with user's
            final_components = {}
            used_nums = set(user_refs.values())
            
            # Add user's references first (PRIORITY)
            for name, num in user_refs.items():
                final_components[name] = num
            
            # Add auto-extracted that don't have same numeral (avoid duplicates)
            for name, num in self.components.items():
                if num not in used_nums and name not in user_refs:
                    # Only add if different enough from user's
                    name_lower = name.lower()
                    if not any(uname.lower() in name_lower or name_lower in uname.lower() 
                               for uname in user_refs.keys()):
                        final_components[name] = num
            
            self.components = final_components
        
        return self.components
    
    def get_ref(self, component: str) -> str:
        """Get reference numeral for a component."""
        if component in self.components:
            return f"({self.components[component]})"
        # Default to first component number
        return f"({min(self.components.values()) if self.components else 10})"
    
    def get_ref_num(self, component: str) -> int:
        """Get numeric reference for a component."""
        return self.components.get(component, min(self.components.values()) if self.components else 10)
    
    def format_component(self, component: str) -> str:
        """Format component with its reference numeral: 'Component (ref)'"""
        return f"{component} {self.get_ref(component)}"
    
    def get_main_invention(self) -> str:
        """Get the main invention name."""
        return self.main_invention or "the present invention"
    
    def get_all_components(self) -> Dict[str, int]:
        """Get all registered components."""
        return self.components.copy()
    
    def get_components_list(self) -> List[Tuple[str, str]]:
        """Get list of (component_name, reference_string) tuples."""
        return [(name, f"({num})") for name, num in self.components.items()]
    
    def to_json(self) -> Dict[str, str]:
        """
        Export components as enterprise JSON for diagram generation.
        Format: {"100": "Component Name", "110": "Other Component"}
        This is the Deconstructor output for the diagram pipeline.
        """
        import json
        return {str(num): name.title() for name, num in self.components.items()}
    
    def to_json_string(self) -> str:
        """Return JSON string representation."""
        import json
        return json.dumps(self.to_json(), indent=2)
    
    def to_dict(self) -> Dict:
        """Export registry as dictionary for session state storage."""
        return {
            "components": self.components,
            "main_invention": self.main_invention,
            "invention_field": self.invention_field,
            "next_num": self.next_num,
            "increment": self.increment
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComponentRegistry':
        """Create registry from dictionary (session state recovery)."""
        registry = cls()
        registry.components = data.get("components", {})
        registry.main_invention = data.get("main_invention", "")
        registry.invention_field = data.get("invention_field", "")
        registry.next_num = data.get("next_num", 10)
        registry.increment = data.get("increment", 10)
        return registry


# Global function for easy access
def create_unified_registry(abstract: str, drawing_summary: str = "") -> ComponentRegistry:
    """
    Create a unified component registry from abstract AND drawing_summary.
    User's reference numerals from drawing_summary take PRIORITY.
    
    Call this ONCE at the start of patent generation.
    Pass the returned registry to ALL section generators.
    
    Args:
        abstract: Patent abstract text
        drawing_summary: User's drawing description (contains their reference numerals)
    """
    registry = ComponentRegistry()
    
    if drawing_summary:
        # Use merge method that prioritizes user's numerals from drawing_summary
        registry.merge_with_user_references(abstract, drawing_summary)
    else:
        # Just extract from abstract
        registry.extract_from_abstract(abstract)
    
    return registry


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED COMPONENT REGISTRY TEST")
    print("=" * 60)
    
    test_abstract = """
    A hydroponic tower system comprising:
    a vertical support column for structural support,
    stackable planting modules arranged vertically,
    a nutrient delivery conduit for water circulation,
    an LED grow light array for plant illumination,
    and a control unit for automated operation.
    """
    
    registry = create_unified_registry(test_abstract)
    
    print(f"\nMain Invention: {registry.get_main_invention()}")
    print(f"Field: {registry.invention_field}")
    print(f"\nComponents:")
    for name, ref_num in registry.components.items():
        print(f"  - {name}: ({ref_num})")
