"""
Enterprise-Level Patent Output Utilities
=========================================
Professional post-processing for all patent generation modules.
"""

import re
from typing import Dict, List, Optional


class ProfessionalPatentCleaner:
    """Enterprise-grade output cleaning for patent documents."""
    
    # LLM artifacts to remove
    LLM_ARTIFACTS = [
        r'^(Okay|Sure|Certainly|Of course|Here is|Here\'s|I\'ll|Let me)[^.]*\.\s*',
        r'^(The user|Based on|According to the abstract)[^.]*\.\s*',
        r'^\*\*[^*]+\*\*\s*\n?',  # Bold headers
        r'\*\*([^*]+)\*\*',  # Inline bold
        r'__([^_]+)__',  # Underscores
        r'```[a-z]*\n?',  # Code blocks
        r'\n```',
    ]
    
    # IPO-compliant starting phrases
    IPO_SUMMARY_START = "Thus according to the basic aspect of the present invention, there is provided"
    IPO_FIELD_STARTS = [
        "The present invention relates",
        "This invention pertains",
        "The present disclosure relates"
    ]
    
    @classmethod
    def clean_llm_artifacts(cls, text: str) -> str:
        """Remove all LLM conversational artifacts."""
        for pattern in cls.LLM_ARTIFACTS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        return text.strip()
    
    @classmethod
    def clean_title(cls, title: str) -> str:
        """Professional title cleaning for IPO compliance."""
        # Remove any prefixes
        title = re.sub(r'^(Title:|Patent Title:)\s*', '', title, flags=re.IGNORECASE)
        title = title.strip('"\'`')
        title = title.rstrip('.')
        
        # Remove forbidden starting articles
        title = re.sub(r'^(A|An|The)\s+', '', title, flags=re.IGNORECASE)
        
        # Remove numbering artifacts
        title = re.sub(r'^\d+\.\s*', '', title)
        
        # Clean up whitespace
        title = ' '.join(title.split())
        
        # Uppercase for patent format
        return title.upper()
    
    @classmethod
    def clean_summary(cls, summary: str) -> str:
        """Professional summary cleaning for IPO compliance."""
        summary = cls.clean_llm_artifacts(summary)
        
        # Ensure starts with proper phrase
        if not summary.lower().startswith("thus according"):
            # Try to find proper start
            match = re.search(r'(Thus according to)', summary, re.IGNORECASE)
            if match:
                summary = summary[match.start():]
            else:
                summary = cls.IPO_SUMMARY_START + " " + summary
        
        # Clean up "wherein" clauses
        summary = re.sub(r'\bwhere\b', 'wherein', summary, flags=re.IGNORECASE)
        
        return summary.strip()
    
    @classmethod
    def clean_claims(cls, claims: str) -> str:
        """Professional claims cleaning - ensure single sentences."""
        claims = cls.clean_llm_artifacts(claims)
        
        # Split into individual claims
        claim_lines = []
        for line in claims.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Ensure proper numbering
            if not re.match(r'^\d+\.', line):
                continue
            
            # Convert multiple sentences to single sentence using semicolons
            # Replace ". " with "; " within a claim (except at end)
            line = re.sub(r'\.\s+(?=[A-Z])', '; ', line.rstrip('.')) + '.'
            
            claim_lines.append(line)
        
        return '\n\n'.join(claim_lines)
    
    @classmethod
    def clean_field_of_invention(cls, field: str) -> str:
        """Professional field of invention cleaning."""
        field = cls.clean_llm_artifacts(field)
        
        # Ensure starts with proper phrase
        valid_start = any(field.lower().startswith(s.lower()) for s in cls.IPO_FIELD_STARTS)
        if not valid_start:
            match = re.search(r'(The present invention|This invention)', field, re.IGNORECASE)
            if match:
                field = field[match.start():]
            else:
                field = "The present invention relates generally to " + field
        
        # Ensure ends with period
        if not field.endswith('.'):
            field += '.'
        
        return field.strip()
    
    @classmethod
    def clean_background(cls, background: str) -> str:
        """Professional background cleaning."""
        background = cls.clean_llm_artifacts(background)
        
        # Remove any section headers
        background = re.sub(r'^(BACKGROUND|Background of the Invention:?)\s*\n*', '', 
                           background, flags=re.IGNORECASE)
        
        # Ensure proper paragraph structure
        paragraphs = [p.strip() for p in background.split('\n\n') if p.strip()]
        
        # Each paragraph should end with period
        cleaned_paragraphs = []
        for p in paragraphs:
            p = ' '.join(p.split())  # Clean whitespace
            if p and not p.endswith('.'):
                p += '.'
            cleaned_paragraphs.append(p)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    @classmethod
    def clean_drawings_description(cls, drawings: str) -> str:
        """Professional drawings description cleaning."""
        drawings = cls.clean_llm_artifacts(drawings)
        
        # Ensure each figure is on its own line
        lines = []
        for line in drawings.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Standardize figure format
            line = re.sub(r'^(FIG\.|Figure|FIGURE)\s*(\d+)', r'Figure \2:', line)
            
            # Ensure "illustrates" after Figure X:
            if re.match(r'^Figure \d+:', line) and 'illustrates' not in line.lower():
                line = re.sub(r'^(Figure \d+:)\s*', r'\1 illustrates ', line)
            
            # Ensure ends with period
            if line and not line.endswith('.'):
                line += '.'
            
            lines.append(line)
        
        return '\n'.join(lines)


def validate_ipo_compliance(section_name: str, content: str) -> Dict[str, any]:
    """
    Validate any patent section for IPO compliance.
    Returns detailed validation report.
    """
    issues = []
    warnings = []
    word_count = len(content.split())
    
    if section_name == "title":
        if word_count < 8:
            issues.append("Title too short (minimum 8 words)")
        elif word_count > 20:
            warnings.append("Title may be too long (recommended 10-15 words)")
        
        if content[0].islower():
            issues.append("Title should be capitalized")
    
    elif section_name == "summary":
        if not content.lower().startswith("thus according"):
            issues.append("Summary must start with 'Thus according to...'")
        
        if word_count < 200:
            warnings.append("Summary may be too short")
        elif word_count > 600:
            warnings.append("Summary may be too long")
        
        if "comprising" not in content.lower():
            warnings.append("Consider using 'comprising:' to list components")
    
    elif section_name == "claims":
        claim_lines = [l for l in content.split('\n') if l.strip() and re.match(r'^\d+\.', l.strip())]
        
        for i, claim in enumerate(claim_lines, 1):
            # Check for multiple sentences (excluding semicolons)
            sentences = re.split(r'\.\s+(?=[A-Z])', claim)
            if len(sentences) > 1:
                issues.append(f"Claim {i}: Must be single sentence")
    
    elif section_name == "field":
        valid_start = any(content.lower().startswith(s.lower()) for s in 
                         ["the present invention", "this invention", "the present disclosure"])
        if not valid_start:
            issues.append("Field must start with 'The present invention relates...'")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "word_count": word_count
    }


# Singleton cleaner instance
cleaner = ProfessionalPatentCleaner()
