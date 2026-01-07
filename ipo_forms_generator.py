"""
IPO Patent Forms Generator
==========================
Generates IPO Form 1 (Application) and Form 5 (Declaration of Inventorship)
for Indian Patent Office filings.

These forms are required for complete patent application submission.
"""

from typing import Dict, Optional
from datetime import datetime


class IPOFormsGenerator:
    """Generate IPO-compliant patent forms."""
    
    @staticmethod
    def generate_form_1(
        title: str,
        applicant_name: str,
        applicant_address: str,
        applicant_nationality: str,
        inventor_name: str,
        inventor_address: str,
        inventor_nationality: str,
        applicant_category: str = "NATURAL PERSON",
        application_type: str = "ORDINARY",
        is_provisional: bool = False,
        priority_claim: Optional[Dict] = None
    ) -> str:
        """
        Generate IPO Form 1 (Application for Grant of Patent).
        
        Args:
            title: Invention title
            applicant_name: Name of applicant
            applicant_address: Full address
            applicant_nationality: Nationality
            inventor_name: Name of inventor
            inventor_address: Inventor's address
            inventor_nationality: Inventor's nationality
            applicant_category: NATURAL PERSON, STARTUP, SMALL ENTITY, OTHER
            application_type: ORDINARY, CONVENTION, PCT NATIONAL PHASE
            is_provisional: True for provisional, False for complete
            priority_claim: Dict with country, application_no, date if claiming priority
        
        Returns:
            Formatted Form 1 text
        """
        current_date = datetime.now().strftime("%d/%m/%Y")
        spec_type = "PROVISIONAL" if is_provisional else "COMPLETE"
        
        form = f"""
================================================================================
                    THE PATENTS ACT, 1970
                    (39 OF 1970)
                    
                    THE PATENTS RULES, 2003
                    
                              FORM 1
                    
              APPLICATION FOR GRANT OF PATENT
                    [See section 7, 54 & 135 and rule 20(1)]
================================================================================

1. APPLICANT(S):

   (a) Name          : {applicant_name.upper()}
   (b) Nationality   : {applicant_nationality.upper()}
   (c) Address       : {applicant_address}
   
   Category of Applicant: [{applicant_category.upper()}]
   
   [ ] Natural Person
   [ ] Startup (as defined in notification of DPIIT)
   [ ] Small Entity (as defined in MSMED Act, 2006)
   [ ] Educational Institution
   [ ] Other

--------------------------------------------------------------------------------

2. INVENTOR(S):

   (a) Name          : {inventor_name.upper()}
   (b) Nationality   : {inventor_nationality.upper()}
   (c) Address       : {inventor_address}

--------------------------------------------------------------------------------

3. TITLE OF THE INVENTION:

   {title}

--------------------------------------------------------------------------------

4. ADDRESS FOR SERVICE IN INDIA:

   {applicant_address}

--------------------------------------------------------------------------------

5. APPLICATION TYPE: [{application_type.upper()}]

   [{'X' if application_type == 'ORDINARY' else ' '}] Ordinary Application
   [{'X' if application_type == 'CONVENTION' else ' '}] Convention Application
   [{'X' if application_type == 'PCT NATIONAL PHASE' else ' '}] PCT National Phase Application
   [ ] Divisional Application
   [ ] Patent of Addition

--------------------------------------------------------------------------------

6. SPECIFICATION TYPE: [{spec_type}]

   [{'X' if is_provisional else ' '}] Provisional Specification
   [{'X' if not is_provisional else ' '}] Complete Specification

--------------------------------------------------------------------------------

7. PRIORITY CLAIM (if any):
"""
        
        if priority_claim:
            form += f"""
   Country           : {priority_claim.get('country', 'N/A')}
   Application No.   : {priority_claim.get('application_no', 'N/A')}
   Filing Date       : {priority_claim.get('date', 'N/A')}
"""
        else:
            form += """
   [ ] No priority claimed
"""
        
        form += f"""
--------------------------------------------------------------------------------

8. DECLARATIONS:

   I/We, the applicant(s), hereby declare that:
   
   (a) I am/We are the true and first inventor(s) of the invention described
       in the specification.
   
   (b) The complete specification accompanying this application fully
       describes the invention and the manner in which it is to be performed.
   
   (c) The claims define the scope of the invention for which protection
       is sought.
   
   (d) I/We have not filed any other application for patent on the same
       invention in India.

--------------------------------------------------------------------------------

9. DOCUMENTS FILED WITH THIS APPLICATION:

   [X] Form 1 - Application for Grant of Patent
   [X] Form 2 - Complete Specification (Provisional/Complete)
   [X] Form 3 - Statement and Undertaking (if applicable)
   [X] Form 5 - Declaration of Inventorship
   [ ] Form 26 - Authorization of Agent (if applicable)
   [ ] Priority Documents (if claiming priority)
   [ ] Sequence Listing (if applicable)

--------------------------------------------------------------------------------

Date: {current_date}

Signature: _________________________

Name: {applicant_name.upper()}

================================================================================
                    FOR OFFICE USE ONLY
================================================================================

Application Number: ___________________
Filing Date: _________________________
Fee Received: ________________________
Receipt Number: ______________________

================================================================================
"""
        return form
    
    @staticmethod
    def generate_form_5(
        inventor_name: str,
        inventor_address: str,
        inventor_nationality: str,
        title: str,
        applicant_name: str,
        contribution: str = "sole inventor"
    ) -> str:
        """
        Generate IPO Form 5 (Declaration of Inventorship).
        
        Args:
            inventor_name: Name of inventor
            inventor_address: Inventor's address
            inventor_nationality: Inventor's nationality
            title: Invention title
            applicant_name: Name of applicant
            contribution: Description of inventor's contribution
        
        Returns:
            Formatted Form 5 text
        """
        current_date = datetime.now().strftime("%d/%m/%Y")
        
        form = f"""
================================================================================
                    THE PATENTS ACT, 1970
                    (39 OF 1970)
                    
                    THE PATENTS RULES, 2003
                    
                              FORM 5
                    
              DECLARATION AS TO INVENTORSHIP
                    [See section 10(6) and rule 13(6)]
================================================================================

APPLICATION DETAILS:

Title of Invention: {title}

Application Number: _________________ (To be filled after filing)

--------------------------------------------------------------------------------

DECLARATION:

I, {inventor_name.upper()}, of {inventor_address}, nationality 
{inventor_nationality.upper()}, do hereby solemnly declare as follows:

1. I am the {contribution} of the invention described in the complete 
   specification filed with the above application.

2. I have made a substantial contribution to the conception and/or 
   development of the invention.

3. My contribution to the invention includes:
   
   - Conception of the inventive concept
   - Development of the technical solution
   - Reduction of the invention to practice
   - Contributions to the claims and specification

4. I understand that making a false declaration is punishable under 
   Section 25(1)(h) and Section 64(1)(j) of the Patents Act, 1970.

5. I have not derived the subject matter of the invention from any 
   other person or source without proper authorization.

--------------------------------------------------------------------------------

APPLICANT RELATIONSHIP:

The applicant, {applicant_name.upper()}, is filing this application as:

[X] The inventor (same person)
[ ] Assignee of the inventor
[ ] Legal representative of the inventor
[ ] Employer of the inventor (in case of employee invention)

--------------------------------------------------------------------------------

VERIFICATION:

I, {inventor_name.upper()}, do hereby verify that the contents of this 
declaration are true to my knowledge and belief, and nothing material 
has been concealed therefrom.

Verified at _________________ on this {current_date}

Signature: _________________________

Name: {inventor_name.upper()}

Address: {inventor_address}

--------------------------------------------------------------------------------

WITNESS (if required):

1. Name: _________________________
   Address: _______________________
   Signature: _____________________

2. Name: _________________________
   Address: _______________________
   Signature: _____________________

================================================================================
                    FOR OFFICE USE ONLY
================================================================================

Received on: _____________________
Verified by: _____________________

================================================================================
"""
        return form
    
    @staticmethod
    def generate_form_3(
        applicant_name: str,
        title: str,
        other_applications: list = None
    ) -> str:
        """
        Generate IPO Form 3 (Statement and Undertaking under Section 8).
        
        Args:
            applicant_name: Name of applicant
            title: Invention title
            other_applications: List of dicts with country, app_no, status
        
        Returns:
            Formatted Form 3 text
        """
        current_date = datetime.now().strftime("%d/%m/%Y")
        
        form = f"""
================================================================================
                    THE PATENTS ACT, 1970
                    (39 OF 1970)
                    
                    THE PATENTS RULES, 2003
                    
                              FORM 3
                    
        STATEMENT AND UNDERTAKING UNDER SECTION 8
                    [See section 8 and rule 12]
================================================================================

APPLICATION DETAILS:

Title of Invention: {title}

Application Number: _________________ (To be filled after filing)

--------------------------------------------------------------------------------

STATEMENT:

I, {applicant_name.upper()}, the applicant, do hereby state and undertake 
as follows:

1. STATEMENT OF APPLICATIONS FILED OUTSIDE INDIA:
"""
        
        if other_applications:
            form += "\n   The following applications have been filed:\n\n"
            for i, app in enumerate(other_applications, 1):
                form += f"""   {i}. Country: {app.get('country', 'N/A')}
      Application No.: {app.get('app_no', 'N/A')}
      Filing Date: {app.get('date', 'N/A')}
      Status: {app.get('status', 'Pending')}

"""
        else:
            form += """
   [ X ] No application for patent has been filed outside India for the 
         same or substantially the same invention.
"""
        
        form += f"""
--------------------------------------------------------------------------------

2. UNDERTAKING:

   I/We undertake that up to the date of grant of patent in India, I/We 
   shall keep the Controller informed in writing of the following:
   
   (a) Details of all applications filed outside India for the same or 
       substantially the same invention;
   
   (b) The status of such applications including but not limited to 
       information relating to grant, refusal, or abandonment.

--------------------------------------------------------------------------------

3. DECLARATION:

   I/We declare that the particulars given above are true to the best of 
   my/our knowledge and belief.

--------------------------------------------------------------------------------

Date: {current_date}

Signature: _________________________

Name: {applicant_name.upper()}

================================================================================
"""
        return form


def generate_ipo_forms(
    title: str,
    applicant_name: str,
    applicant_address: str,
    applicant_nationality: str = "INDIAN",
    inventor_name: str = None,
    inventor_address: str = None,
    inventor_nationality: str = None,
    applicant_category: str = "NATURAL PERSON",
    is_provisional: bool = False
) -> Dict[str, str]:
    """
    Convenience function to generate all required IPO forms.
    
    Returns dict with form_1, form_3, form_5 keys.
    """
    # Default inventor to applicant if not provided
    inventor_name = inventor_name or applicant_name
    inventor_address = inventor_address or applicant_address
    inventor_nationality = inventor_nationality or applicant_nationality
    
    generator = IPOFormsGenerator()
    
    return {
        "form_1": generator.generate_form_1(
            title=title,
            applicant_name=applicant_name,
            applicant_address=applicant_address,
            applicant_nationality=applicant_nationality,
            inventor_name=inventor_name,
            inventor_address=inventor_address,
            inventor_nationality=inventor_nationality,
            applicant_category=applicant_category,
            is_provisional=is_provisional
        ),
        "form_3": generator.generate_form_3(
            applicant_name=applicant_name,
            title=title
        ),
        "form_5": generator.generate_form_5(
            inventor_name=inventor_name,
            inventor_address=inventor_address,
            inventor_nationality=inventor_nationality,
            title=title,
            applicant_name=applicant_name
        )
    }


# Test
if __name__ == "__main__":
    forms = generate_ipo_forms(
        title="A Smart Industrial Monitoring System",
        applicant_name="John Doe",
        applicant_address="123 Tech Street, Bangalore 560001, Karnataka, India",
        applicant_nationality="INDIAN",
        applicant_category="STARTUP"
    )
    
    print("=" * 70)
    print("IPO FORMS GENERATOR TEST")
    print("=" * 70)
    print(forms["form_1"][:1500] + "...")
    print("\n" + "=" * 70)
    print("Form 5 preview:")
    print(forms["form_5"][:1000] + "...")
