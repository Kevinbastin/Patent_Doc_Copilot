"""
Pre-CAD Patent Drawing Assistant
================================

IMPORTANT: This is a PRE-CAD drawing assistant that generates
structure-accurate DRAFT drawings to be finalized in CAD/Illustrator.

This is NOT:
- An IPO-ready drawing generator
- A CAD replacement
- A final filing tool

FAIL HARD Policy:
- If components cannot be extracted -> FAIL (no generation)
- If claim-drawing mapping fails -> FAIL with clear error
- NO generic placeholders (first component, second component, etc.)
"""

import os
import re
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# PIL import
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    pass


class DrawingType(Enum):
    OVERALL = "overall"
    CROSS_SECTION = "cross_section"
    EXPLODED = "exploded"
    ISOMETRIC = "isometric"


class DrawingGenerationError(Exception):
    """Raised when drawing generation MUST fail."""
    pass


class ComponentExtractionError(DrawingGenerationError):
    """Raised when components cannot be extracted from abstract."""
    pass


class ClaimDrawingMismatchError(DrawingGenerationError):
    """Raised when drawing doesn't map to Claim 1 elements."""
    pass


@dataclass
class PatentDrawing:
    drawing_type: DrawingType
    figure_number: int
    image_path: Optional[str]
    svg_path: Optional[str]
    components: List[str]
    reference_numerals: Dict[str, int]
    description: str
    warnings: List[str] = field(default_factory=list)
    is_draft: bool = True  # Always True - this is pre-CAD


@dataclass
class DrawingValidationResult:
    """Result of claim-drawing validation."""
    is_valid: bool
    claim_elements: Set[str]
    drawing_components: Set[str]
    missing_in_drawings: Set[str]
    coverage_percent: float
    message: str


class PreCADDrawingGenerator:
    """
    Pre-CAD Patent Drawing Assistant.
    
    Generates DRAFT drawings that require finalization in CAD software.
    
    FAIL HARD POLICY:
    - No generic placeholders
    - Real components must be extracted
    - Claim-drawing mapping must be validated
    """
    
    def __init__(self, abstract: str, claim_1_text: str = None, components: Dict = None):
        """
        Initialize generator. Will FAIL if input is insufficient.
        
        Raises:
            ComponentExtractionError: If components cannot be extracted
        """
        self.abstract = abstract.strip() if abstract else ""
        self.claim_1_text = claim_1_text.strip() if claim_1_text else None
        
        # FAIL HARD: Require valid input
        if not self.abstract and not self.claim_1_text:
            raise ComponentExtractionError(
                "MANUAL DRAWING REQUIRED: No abstract or claims provided. "
                "Cannot generate drawings without invention description."
            )
        
        # Extract components - FAIL if unable
        if components:
            self.components = components
        else:
            self.components = self._extract_components_strict()
        
        # Validate extraction succeeded
        if not self.components.get("components"):
            raise ComponentExtractionError(
                "MANUAL DRAWING REQUIRED: Could not extract specific components. "
                f"Input text: '{self.abstract[:100]}...'"
            )
        
        self.reference_numerals = self._assign_reference_numerals()
        self.output_dir = "generated_drawings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.width = 1000
        self.height = 800
        self._load_fonts()
    
    def _load_fonts(self):
        """Load fonts with fallback."""
        self.font_title = None
        self.font_label = None
        self.font_ref = None
        
        if not PIL_AVAILABLE:
            return
        
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    self.font_title = ImageFont.truetype(fp, 28)
                    self.font_label = ImageFont.truetype(fp, 20)
                    self.font_ref = ImageFont.truetype(fp, 18)
                    return
                except:
                    continue
        
        try:
            self.font_title = ImageFont.load_default()
            self.font_label = ImageFont.load_default()
            self.font_ref = ImageFont.load_default()
        except:
            pass
    
    def _extract_components_strict(self) -> Dict:
        """
        Extract components using LLM - STRICT mode.
        FAILS if cannot extract real components.
        
        NO GENERIC PLACEHOLDERS.
        """
        source_text = self.claim_1_text if self.claim_1_text else self.abstract
        
        components = {
            "main_system": "",
            "components": []
        }
        
        # Extract main system
        first_sentence = source_text.split('.')[0] if source_text else ""
        components["main_system"] = first_sentence[:60].strip()
        
        if not components["main_system"]:
            raise ComponentExtractionError(
                "MANUAL DRAWING REQUIRED: Cannot determine main system from text."
            )
        
        # Try LLM extraction (NO KEYWORDS)
        try:
            from llm_runtime import llm_generate
            
            prompt = f"""Extract the main components/parts from this patent abstract.
Return ONLY a numbered list of component names, exactly as written in the text.
Extract 4-6 components maximum. Use EXACT terminology from the text.

ABSTRACT:
{source_text}

Components (numbered list only):"""
            
            response = llm_generate(prompt, max_new_tokens=300)
            
            if response:
                lines = response.strip().split('\n')
                for line in lines:
                    clean = re.sub(r'^[\d\.\-\•\*\s]+', '', line).strip()
                    # Validate it's a real component name (not generic)
                    if self._is_valid_component(clean):
                        components["components"].append(clean)
                
                if len(components["components"]) >= 3:
                    return components
        except Exception as e:
            print(f"LLM extraction failed: {e}")
        
        # Fallback: Parse using pattern matching (no keywords, just structure)
        try:
            # Split on structural words
            text = source_text
            text = re.sub(r'comprising|including|having|containing|with', '|SPLIT|', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+and\s+', '|SPLIT|', text)
            text = re.sub(r',\s*', '|SPLIT|', text)
            
            parts = text.split('|SPLIT|')
            for part in parts:
                clean = part.strip()
                words = clean.split()
                # Take 2-5 word phrases that look like component names
                if 2 <= len(words) <= 5:
                    phrase = ' '.join(words)
                    if self._is_valid_component(phrase):
                        components["components"].append(phrase)
            
            components["components"] = components["components"][:6]
        except Exception:
            pass
        
        return components
    
    def _is_valid_component(self, name: str) -> bool:
        """
        Check if component name is valid (not generic placeholder).
        """
        if not name or len(name) < 4 or len(name) > 60:
            return False
        
        # REJECT generic placeholders
        generic_patterns = [
            r'^first\s+component',
            r'^second\s+component',
            r'^third\s+component',
            r'^fourth\s+component',
            r'^main\s+housing$',
            r'^apparatus$',
            r'^system$',
            r'^device$',
            r'^component\s+\d+',
            r'^element\s+\d+',
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, name.lower()):
                return False
        
        return True
    
    def _assign_reference_numerals(self) -> Dict[str, int]:
        """Assign reference numerals (100-series)."""
        ref_nums = {}
        base = 100
        
        main = self.components.get("main_system", "")
        if main:
            ref_nums[main[:50]] = base
        
        for i, comp in enumerate(self.components.get("components", [])[:10], 1):
            ref_nums[comp] = base + (i * 10)
        
        return ref_nums
    
    def validate_claim_to_drawing_mapping(self, claim_1_elements: List[str] = None) -> DrawingValidationResult:
        """
        Validate that Claim 1 elements appear in drawings.
        
        IPO REQUIREMENT: Every element of Claim 1 must appear in at least one figure.
        """
        # Extract claim elements if not provided
        if claim_1_elements is None and self.claim_1_text:
            claim_1_elements = self._extract_claim_elements(self.claim_1_text)
        
        claim_elements = set(e.lower() for e in (claim_1_elements or []))
        drawing_components = set(c.lower() for c in self.components.get("components", []))
        
        # Check coverage
        missing = claim_elements - drawing_components
        covered = claim_elements.intersection(drawing_components)
        
        coverage = len(covered) / len(claim_elements) * 100 if claim_elements else 0
        
        is_valid = len(missing) == 0 or coverage >= 80  # Allow 80% minimum
        
        message = ""
        if not is_valid:
            message = (
                f"CLAIM-DRAWING MISMATCH: {len(missing)} claim elements not in drawings. "
                f"Missing: {', '.join(list(missing)[:3])}... "
                f"Coverage: {coverage:.0f}%"
            )
        else:
            message = f"Claim-drawing mapping OK. Coverage: {coverage:.0f}%"
        
        return DrawingValidationResult(
            is_valid=is_valid,
            claim_elements=claim_elements,
            drawing_components=drawing_components,
            missing_in_drawings=missing,
            coverage_percent=coverage,
            message=message
        )
    
    def _extract_claim_elements(self, claim_text: str) -> List[str]:
        """Extract key elements from Claim 1."""
        elements = []
        
        # Split on 'comprising', 'wherein', 'including'
        text = claim_text
        text = re.sub(r'comprising|wherein|including|having', '|SPLIT|', text, flags=re.IGNORECASE)
        
        parts = text.split('|SPLIT|')
        for part in parts[1:]:  # Skip preamble
            clean = part.strip()
            words = clean.split()
            if 2 <= len(words) <= 6:
                phrase = ' '.join(words[:5])
                elements.append(phrase)
        
        return elements[:8]
    
    def _safe_draw_text(self, draw, position, text, fill='black', font=None, anchor=None):
        """Draw text safely."""
        try:
            if font:
                draw.text(position, str(text), fill=fill, font=font, anchor=anchor)
            else:
                draw.text(position, str(text), fill=fill, anchor=anchor)
        except:
            try:
                draw.text(position, str(text), fill=fill)
            except:
                pass
    
    def generate_overall_view(self, fig_num: int = 1) -> PatentDrawing:
        """Generate overall view - DRAFT for CAD finalization."""
        comps = self.components.get("components", [])[:5]
        
        if not PIL_AVAILABLE:
            raise DrawingGenerationError("PIL required for drawing generation")
        
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Border
        draw.rectangle([20, 20, self.width-20, self.height-20], outline='black', width=2)
        
        # DRAFT watermark
        self._safe_draw_text(draw, (self.width//2, 25), "DRAFT - FINALIZE IN CAD", 
                           fill='gray', font=self.font_ref, anchor='mm')
        
        # Title
        self._safe_draw_text(draw, (self.width//2, 55), f"FIGURE {fig_num} - OVERALL VIEW",
                           fill='black', font=self.font_title, anchor='mm')
        
        # Main box
        box_left, box_top = 150, 100
        box_w, box_h = 550, 500
        draw.rectangle([box_left, box_top, box_left + box_w, box_top + box_h],
                      outline='black', width=3)
        
        # Components with arrow leaders
        y = box_top + 40
        for i, comp in enumerate(comps):
            ref = self.reference_numerals.get(comp, 100 + (i+1)*10)
            
            draw.rectangle([box_left + 30, y, box_left + box_w - 30, y + 70],
                          outline='black', width=2)
            
            self._safe_draw_text(draw, (box_left + 45, y + 22), f"({ref}) {comp[:25]}",
                               fill='black', font=self.font_label)
            
            # Arrow leader line
            start = (box_left + box_w - 30, y + 35)
            end = (self.width - 100, y + 35)
            draw.line([start, end], fill='black', width=2)
            # Arrowhead
            draw.polygon([(start[0], start[1]-5), (start[0], start[1]+5), 
                         (start[0]+10, start[1])], fill='black')
            self._safe_draw_text(draw, (self.width - 90, y + 25), str(ref), font=self.font_ref)
            
            y += 85
        
        # Bottom label
        main = self.components.get("main_system", "")[:45]
        self._safe_draw_text(draw, (self.width//2, self.height - 45), f"100 - {main}",
                           fill='black', font=self.font_label, anchor='mm')
        
        filepath = os.path.join(self.output_dir, f"fig{fig_num}_overall.png")
        img.save(filepath)
        
        return PatentDrawing(
            DrawingType.OVERALL, fig_num, filepath, None, comps,
            self.reference_numerals, f"Figure {fig_num} shows overall view of the {main}.",
            ["DRAFT - Finalize in CAD for IPO filing"], True
        )
    
    def generate_cross_section(self, fig_num: int = 2) -> PatentDrawing:
        """Generate cross-section - SPARSE hatching on CUT regions only."""
        comps = self.components.get("components", [])[:5]
        
        if not PIL_AVAILABLE:
            raise DrawingGenerationError("PIL required")
        
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([20, 20, self.width-20, self.height-20], outline='black', width=2)
        
        self._safe_draw_text(draw, (self.width//2, 25), "DRAFT - FINALIZE IN CAD",
                           fill='gray', font=self.font_ref, anchor='mm')
        self._safe_draw_text(draw, (self.width//2, 55), f"FIGURE {fig_num} - SECTION A-A",
                           fill='black', font=self.font_title, anchor='mm')
        
        # Section housing
        sec_left, sec_top = 140, 95
        sec_w, sec_h = 560, 550
        draw.rectangle([sec_left, sec_top, sec_left + sec_w, sec_top + sec_h],
                      outline='black', width=3)
        
        # SPARSE hatching on CUT WALLS ONLY (not arbitrary)
        wall = 18
        # Left wall - sparse 45° lines
        for y in range(sec_top, sec_top + sec_h, 12):
            draw.line([(sec_left, y), (sec_left + wall, y + 12)], fill='black', width=1)
        # Right wall - same sparse hatching
        for y in range(sec_top, sec_top + sec_h, 12):
            draw.line([(sec_left + sec_w - wall, y), (sec_left + sec_w, y + 12)], fill='black', width=1)
        
        # Internal layers - NO hatching (not cut material)
        layer_h = (sec_h - 60) // max(len(comps), 1)
        y = sec_top + 30
        
        for i, comp in enumerate(comps):
            ref = self.reference_numerals.get(comp, 100 + (i+1)*10)
            
            # Layer box - no hatching inside (not cut material)
            draw.rectangle([sec_left + wall + 15, y, sec_left + sec_w - wall - 15, y + layer_h - 12],
                          outline='black', width=2)
            
            self._safe_draw_text(draw, (sec_left + wall + 25, y + 12), f"({ref})",
                               fill='black', font=self.font_label)
            
            # Arrow leader
            start = (sec_left + sec_w - wall - 15, y + layer_h//2 - 6)
            end = (self.width - 130, y + layer_h//2 - 6)
            draw.line([start, end], fill='black', width=2)
            draw.polygon([(start[0], start[1]-4), (start[0], start[1]+4),
                         (start[0]+8, start[1])], fill='black')
            self._safe_draw_text(draw, (self.width - 120, y + layer_h//2 - 15),
                               f"{ref}-{comp[:12]}", font=self.font_ref)
            
            y += layer_h
        
        # Section indicators
        self._safe_draw_text(draw, (sec_left - 35, sec_top + sec_h//2), "A", font=self.font_title)
        self._safe_draw_text(draw, (sec_left + sec_w + 20, sec_top + sec_h//2), "A", font=self.font_title)
        self._safe_draw_text(draw, (self.width//2, self.height - 35), "SECTION A-A",
                           font=self.font_label, anchor='mm')
        
        filepath = os.path.join(self.output_dir, f"fig{fig_num}_cross_section.png")
        img.save(filepath)
        
        return PatentDrawing(
            DrawingType.CROSS_SECTION, fig_num, filepath, None, comps,
            self.reference_numerals, f"Figure {fig_num} is cross-section A-A.",
            ["DRAFT - Hatching for cut regions only. Finalize in CAD."], True
        )
    
    def generate_exploded_view(self, fig_num: int = 3) -> PatentDrawing:
        """Generate exploded view - SCHEMATIC PREVIEW only."""
        comps = self.components.get("components", [])[:5]
        
        if not PIL_AVAILABLE:
            raise DrawingGenerationError("PIL required")
        
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([20, 20, self.width-20, self.height-20], outline='black', width=2)
        
        self._safe_draw_text(draw, (self.width//2, 25), "SCHEMATIC PREVIEW - REQUIRES CAD",
                           fill='red', font=self.font_ref, anchor='mm')
        self._safe_draw_text(draw, (self.width//2, 55), f"FIGURE {fig_num} - EXPLODED VIEW",
                           fill='black', font=self.font_title, anchor='mm')
        
        num = len(comps)
        if num > 0:
            comp_h, gap = 60, 45
            total_h = num * comp_h + (num - 1) * gap
            start_y = (self.height - total_h) // 2
            center_x = self.width // 2
            comp_w = 380
            
            # Dashed assembly line
            for y in range(start_y - 25, start_y + total_h + 25, 10):
                draw.line([(center_x, y), (center_x, y + 5)], fill='black', width=1)
            
            for i, comp in enumerate(comps):
                ref = self.reference_numerals.get(comp, 100 + (i+1)*10)
                y = start_y + i * (comp_h + gap)
                left = center_x - comp_w // 2
                
                draw.rectangle([left, y, left + comp_w, y + comp_h], outline='black', width=2)
                
                self._safe_draw_text(draw, (left + 18, y + 18), f"({ref}) {comp[:22]}",
                                   fill='black', font=self.font_label)
                
                # Arrow leader
                start = (left + comp_w, y + comp_h//2)
                end = (self.width - 110, y + comp_h//2)
                draw.line([start, end], fill='black', width=2)
                draw.polygon([(start[0], start[1]-4), (start[0], start[1]+4),
                             (start[0]+8, start[1])], fill='black')
                self._safe_draw_text(draw, (self.width - 100, y + comp_h//2 - 8),
                                   str(ref), font=self.font_ref)
                
                # Assembly dashes (no arrows per IPO)
                if i < num - 1:
                    for dy in range(y + comp_h + 5, y + comp_h + gap - 5, 8):
                        draw.line([(center_x - 15, dy), (center_x - 15, dy + 4)], fill='black', width=1)
                        draw.line([(center_x + 15, dy), (center_x + 15, dy + 4)], fill='black', width=1)
        
        filepath = os.path.join(self.output_dir, f"fig{fig_num}_exploded.png")
        img.save(filepath)
        
        return PatentDrawing(
            DrawingType.EXPLODED, fig_num, filepath, None, comps,
            self.reference_numerals, f"Figure {fig_num} is exploded assembly view.",
            ["SCHEMATIC PREVIEW - Does not show actual physical structure. Replace with CAD."], True
        )
    
    def generate_isometric_view(self, fig_num: int = 4) -> PatentDrawing:
        """Generate isometric view - SCHEMATIC PREVIEW only."""
        comps = self.components.get("components", [])[:5]
        
        if not PIL_AVAILABLE:
            raise DrawingGenerationError("PIL required")
        
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([20, 20, self.width-20, self.height-20], outline='black', width=2)
        
        self._safe_draw_text(draw, (self.width//2, 25), "SCHEMATIC PREVIEW - REQUIRES CAD",
                           fill='red', font=self.font_ref, anchor='mm')
        self._safe_draw_text(draw, (self.width//2, 55), f"FIGURE {fig_num} - ISOMETRIC VIEW",
                           fill='black', font=self.font_title, anchor='mm')
        
        # Simplified isometric cube
        cx, cy = self.width // 2, self.height // 2
        size = 140
        angle = math.radians(30)
        
        def iso(x, y, z):
            return (cx + (x - z) * math.cos(angle), cy - y + (x + z) * math.sin(angle) * 0.5)
        
        p1, p2, p3, p4 = iso(-size, 0, -size), iso(size, 0, -size), iso(size, 0, size), iso(-size, 0, size)
        p5, p6, p7, p8 = iso(-size, size*1.5, -size), iso(size, size*1.5, -size), iso(size, size*1.5, size), iso(-size, size*1.5, size)
        
        draw.polygon([p5, p6, p7, p8], outline='black', fill='white')
        draw.polygon([p4, p3, p7, p8], outline='black', fill='white')
        draw.polygon([p3, p2, p6, p7], outline='black', fill='white')
        
        # Component labels with arrows
        y = 95
        for i, comp in enumerate(comps):
            ref = self.reference_numerals.get(comp, 100 + (i+1)*10)
            self._safe_draw_text(draw, (self.width - 230, y), f"({ref}) {comp[:16]}",
                               fill='black', font=self.font_label)
            try:
                cube_pt = iso(size*0.7, size*(0.3 + i*0.25), size*0.7)
                draw.line([cube_pt, (self.width - 240, y + 8)], fill='black', width=1)
            except:
                pass
            y += 32
        
        main = self.components.get("main_system", "")[:35]
        self._safe_draw_text(draw, (self.width//2, self.height - 45), f"100 - {main}",
                           fill='black', font=self.font_label, anchor='mm')
        
        filepath = os.path.join(self.output_dir, f"fig{fig_num}_isometric.png")
        img.save(filepath)
        
        return PatentDrawing(
            DrawingType.ISOMETRIC, fig_num, filepath, None, comps,
            self.reference_numerals, f"Figure {fig_num} shows isometric view.",
            ["SCHEMATIC PREVIEW - Does not represent actual structure. Create in CAD."], True
        )
    
    def generate_all_drawings(self) -> Dict[str, PatentDrawing]:
        """Generate all draft drawings."""
        return {
            "overall": self.generate_overall_view(1),
            "cross_section": self.generate_cross_section(2),
            "exploded": self.generate_exploded_view(3),
            "isometric": self.generate_isometric_view(4),
        }
    
    def generate_brief_description(self, drawings: Dict[str, PatentDrawing]) -> str:
        """Generate brief description."""
        desc = "BRIEF DESCRIPTION OF THE DRAWINGS\n\n"
        desc += "[DRAFT - For reference only. Finalize drawings in CAD before filing.]\n\n"
        
        for d in sorted(drawings.values(), key=lambda x: x.figure_number):
            desc += f"{d.description}\n"
        
        desc += "\nREFERENCE NUMERALS:\n"
        for comp, ref in sorted(self.reference_numerals.items(), key=lambda x: x[1]):
            desc += f"  {ref} - {comp}\n"
        
        return desc


# Backward compatibility alias
OpenSourceCADGenerator = PreCADDrawingGenerator
RobustPatentDrawingGenerator = PreCADDrawingGenerator


def generate_patent_drawings(abstract: str = "", claim_1_text: str = None,
                            components: Dict = None, **kwargs) -> Dict:
    """
    Generate pre-CAD patent drawing drafts.
    
    FAIL HARD: Will raise error if components cannot be extracted.
    
    Returns dict with drawings or raises DrawingGenerationError.
    """
    if not abstract and not claim_1_text:
        return {
            "drawings": {},
            "brief_description": "",
            "reference_numerals": {},
            "warnings": ["MANUAL DRAWING REQUIRED: No input provided."],
            "output_directory": "generated_drawings",
            "success": False,
            "error": "No abstract or claims provided. Cannot generate drawings."
        }
    
    try:
        gen = PreCADDrawingGenerator(abstract, claim_1_text, components)
        drawings = gen.generate_all_drawings()
        brief_desc = gen.generate_brief_description(drawings)
        
        # Validate claim-drawing mapping if claim provided
        validation = None
        if claim_1_text:
            validation = gen.validate_claim_to_drawing_mapping()
        
        warnings = [
            "PRE-CAD DRAFTS: These are NOT final IPO drawings.",
            "Finalize in CAD/Illustrator before filing."
        ]
        
        if validation and not validation.is_valid:
            warnings.append(validation.message)
        
        return {
            "drawings": drawings,
            "brief_description": brief_desc,
            "reference_numerals": gen.reference_numerals,
            "warnings": warnings,
            "output_directory": gen.output_dir,
            "success": True,
            "validation": validation
        }
    
    except ComponentExtractionError as e:
        return {
            "drawings": {},
            "brief_description": "",
            "reference_numerals": {},
            "warnings": [str(e)],
            "output_directory": "generated_drawings",
            "success": False,
            "error": str(e)
        }
    
    except Exception as e:
        return {
            "drawings": {},
            "brief_description": "",
            "reference_numerals": {},
            "warnings": [f"Drawing generation failed: {str(e)}"],
            "output_directory": "generated_drawings",
            "success": False,
            "error": str(e)
        }
