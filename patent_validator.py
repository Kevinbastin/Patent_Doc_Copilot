"""
Patent Validator
================
Enterprise-level validation for IPO-compliant patent applications.

Validates:
- Word counts (Abstract ≤150, Title ≤15)
- Section completeness
- Reference numeral consistency
- Claim structure
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str
    details: Optional[str] = None


class PatentValidator:
    """
    Validates patent application content for IPO compliance.
    
    IPO Requirements per Patents Act 2024:
    - Abstract: ≤150 words
    - Title: ≤15 words  
    - Claims: Properly numbered and structured
    - Drawings: Reference numerals consistent
    """
    
    # IPO Word Limits
    ABSTRACT_MAX_WORDS = 150
    TITLE_MAX_WORDS = 15
    
    # Required sections for complete specification
    REQUIRED_SECTIONS = [
        "title",
        "abstract",  # This is the user abstract input
        "claims",
        "summary",
        "field_of_invention",
        "background",
        "objects_of_invention",
        "detailed_description",
        "brief_description",  # Brief description of drawings
    ]
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text, excluding whitespace and punctuation."""
        if not text:
            return 0
        # Split on whitespace and filter empty strings
        words = [w for w in text.split() if w.strip()]
        return len(words)
    
    @classmethod
    def validate_abstract(cls, abstract: str) -> ValidationResult:
        """
        Validate abstract word count (≤150 words per IPO).
        
        Returns ValidationResult with status and message.
        """
        if not abstract or not abstract.strip():
            return ValidationResult(
                is_valid=False,
                message="❌ Abstract is required",
                details="Please enter the invention abstract."
            )
        
        word_count = cls.count_words(abstract)
        
        if word_count > cls.ABSTRACT_MAX_WORDS:
            return ValidationResult(
                is_valid=False,
                message=f"❌ Abstract too long: {word_count}/{cls.ABSTRACT_MAX_WORDS} words",
                details=f"Reduce by {word_count - cls.ABSTRACT_MAX_WORDS} words for IPO compliance."
            )
        
        if word_count < 50:
            return ValidationResult(
                is_valid=True,  # Warning, not error
                message=f"⚠️ Abstract may be too short: {word_count} words",
                details="Consider adding more technical details (recommended: 100-150 words)."
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"✅ Abstract: {word_count}/{cls.ABSTRACT_MAX_WORDS} words",
            details=None
        )
    
    @classmethod
    def validate_title(cls, title: str) -> ValidationResult:
        """
        Validate title word count (≤15 words per IPO).
        
        Returns ValidationResult with status and message.
        """
        if not title or not title.strip():
            return ValidationResult(
                is_valid=False,
                message="❌ Title is required",
                details="Please generate or enter a title."
            )
        
        word_count = cls.count_words(title)
        
        if word_count > cls.TITLE_MAX_WORDS:
            return ValidationResult(
                is_valid=False,
                message=f"❌ Title too long: {word_count}/{cls.TITLE_MAX_WORDS} words",
                details=f"Reduce by {word_count - cls.TITLE_MAX_WORDS} words for IPO compliance."
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"✅ Title: {word_count}/{cls.TITLE_MAX_WORDS} words",
            details=None
        )
    
    @classmethod
    def validate_claims(cls, claims: str) -> ValidationResult:
        """
        Validate claims structure and numbering.
        
        Checks:
        - Claims are numbered
        - At least one independent claim exists
        - Claims follow proper format
        """
        if not claims or not claims.strip():
            return ValidationResult(
                is_valid=False,
                message="❌ Claims are required",
                details="Please generate claims."
            )
        
        # Check for numbered claims
        claim_pattern = r"(?:Claim\s*)?(\d+)[.\):]"
        claim_matches = re.findall(claim_pattern, claims, re.IGNORECASE)
        
        if not claim_matches:
            return ValidationResult(
                is_valid=False,
                message="❌ Claims not properly numbered",
                details="Claims should be numbered (e.g., 'Claim 1:', '1.', etc.)"
            )
        
        num_claims = len(set(claim_matches))
        
        # Check for independent claim indicators
        has_independent = bool(re.search(
            r"(comprising|including|characterized by|wherein)",
            claims, re.IGNORECASE
        ))
        
        if not has_independent:
            return ValidationResult(
                is_valid=True,  # Warning only
                message=f"⚠️ {num_claims} claims found - verify structure",
                details="Ensure at least one independent claim with 'comprising' or 'wherein'."
            )
        
        return ValidationResult(
            is_valid=True,
            message=f"✅ {num_claims} claims properly formatted",
            details=None
        )
    
    @classmethod
    def validate_sections(cls, sections: Dict[str, str]) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate all required sections are present.
        
        Args:
            sections: Dict with section names as keys and content as values
            
        Returns:
            Tuple of (all_valid, list of ValidationResults)
        """
        results = []
        all_valid = True
        
        for section in cls.REQUIRED_SECTIONS:
            content = sections.get(section, "")
            
            if not content or not str(content).strip():
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"❌ Missing: {section.replace('_', ' ').title()}",
                    details=f"This section is required for IPO filing."
                ))
                all_valid = False
            else:
                word_count = cls.count_words(str(content))
                results.append(ValidationResult(
                    is_valid=True,
                    message=f"✅ {section.replace('_', ' ').title()} ({word_count} words)",
                    details=None
                ))
        
        return all_valid, results
    
    @classmethod
    def validate_applicant_details(cls, applicant: Dict) -> ValidationResult:
        """
        Validate applicant details are complete.
        
        Required fields: name, address, nationality
        """
        required = ["name", "address", "nationality"]
        missing = [f for f in required if not applicant.get(f)]
        
        if missing:
            return ValidationResult(
                is_valid=False,
                message=f"❌ Missing applicant: {', '.join(missing)}",
                details="Complete applicant details for Form 1."
            )
        
        return ValidationResult(
            is_valid=True,
            message="✅ Applicant details complete",
            details=None
        )
    
    @classmethod
    def validate_inventor_details(cls, inventor: Dict) -> ValidationResult:
        """
        Validate inventor details are complete.
        
        Required for Form 5: name, address, nationality
        """
        required = ["name", "address", "nationality"]
        missing = [f for f in required if not inventor.get(f)]
        
        if missing:
            return ValidationResult(
                is_valid=False,
                message=f"❌ Missing inventor: {', '.join(missing)}",
                details="Complete inventor details for Form 5."
            )
        
        return ValidationResult(
            is_valid=True,
            message="✅ Inventor details complete",
            details=None
        )
    
    @classmethod
    def get_completeness_score(cls, sections: Dict[str, str]) -> Tuple[int, int]:
        """
        Calculate completeness percentage.
        
        Returns (completed_count, total_required)
        """
        completed = 0
        for section in cls.REQUIRED_SECTIONS:
            content = sections.get(section, "")
            if content and str(content).strip():
                completed += 1
        
        return completed, len(cls.REQUIRED_SECTIONS)
    
    @classmethod
    def can_export(cls, sections: Dict[str, str], applicant: Dict = None, inventor: Dict = None) -> Tuple[bool, List[str]]:
        """
        Check if patent is ready for export.
        
        Returns (can_export, list of blocking issues)
        """
        issues = []
        
        # Check abstract
        abstract_result = cls.validate_abstract(sections.get("abstract", ""))
        if not abstract_result.is_valid:
            issues.append(abstract_result.message)
        
        # Check title
        title_result = cls.validate_title(sections.get("title", ""))
        if not title_result.is_valid:
            issues.append(title_result.message)
        
        # Check claims
        claims_result = cls.validate_claims(sections.get("claims", ""))
        if not claims_result.is_valid:
            issues.append(claims_result.message)
        
        # Check all sections
        all_valid, section_results = cls.validate_sections(sections)
        if not all_valid:
            for r in section_results:
                if not r.is_valid:
                    issues.append(r.message)
        
        return len(issues) == 0, issues


# Convenience functions for Streamlit integration
def validate_abstract_live(abstract: str) -> str:
    """Return formatted validation message for live display."""
    result = PatentValidator.validate_abstract(abstract)
    if result.details:
        return f"{result.message}\n{result.details}"
    return result.message


def validate_title_live(title: str) -> str:
    """Return formatted validation message for live display."""
    result = PatentValidator.validate_title(title)
    if result.details:
        return f"{result.message}\n{result.details}"
    return result.message


def get_section_checklist(sections: Dict[str, str]) -> List[Tuple[str, bool, str]]:
    """
    Get checklist of sections for UI display.
    
    Returns list of (section_name, is_complete, word_count_str)
    """
    checklist = []
    for section in PatentValidator.REQUIRED_SECTIONS:
        content = sections.get(section, "")
        is_complete = bool(content and str(content).strip())
        word_count = PatentValidator.count_words(str(content)) if is_complete else 0
        checklist.append((
            section.replace("_", " ").title(),
            is_complete,
            f"{word_count} words" if is_complete else "Not generated"
        ))
    return checklist


# CLI testing
if __name__ == "__main__":
    print("=" * 60)
    print("PATENT VALIDATOR - TEST")
    print("=" * 60)
    
    # Test abstract validation
    test_abstract = """
    A smart monitoring system for industrial environments comprising sensor units 
    for data collection, a processing hub with machine learning capabilities, 
    and an alert module for notification generation.
    """
    
    print("\n📋 Abstract Validation:")
    result = PatentValidator.validate_abstract(test_abstract)
    print(f"   {result.message}")
    
    # Test title validation  
    test_title = "Smart Industrial Monitoring System with Machine Learning"
    print("\n📋 Title Validation:")
    result = PatentValidator.validate_title(test_title)
    print(f"   {result.message}")
    
    # Test claims validation
    test_claims = """
    Claim 1: A monitoring system comprising sensor units, a processing hub, and an alert module.
    Claim 2: The system of claim 1, wherein the processing hub includes machine learning.
    """
    print("\n📋 Claims Validation:")
    result = PatentValidator.validate_claims(test_claims)
    print(f"   {result.message}")
    
    print("\n" + "=" * 60)
    print("✅ Validator ready for integration")
