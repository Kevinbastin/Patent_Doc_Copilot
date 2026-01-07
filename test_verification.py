"""
Final Verification Script for Patent_Doc_Copilot
Tests the generalized system across multiple technical domains.
"""

from generate_title import generate_title_from_abstract
from generate_summary import summarize_abstract
from generate_field_of_invention import generate_field_of_invention

TEST_ABSTRACTS = {
    "AI/Software": "A method for federated learning in a distributed network of edge devices, comprising: distributing a global model to the devices; performing local training on each device using local data; transmitting local model updates to a central server; and aggregating the updates to refine the global model while preserving data privacy.",
    "Chemical/Pharma": "A pharmaceutical composition for treating inflammatory diseases, comprising: a therapeutically effective amount of Compound X; a stabilizing polymer matrix; and a pH-buffered aqueous carrier, wherein the composition exhibits sustained release over 24 hours.",
    "Mechanical": "A multi-stage centrifugal pump for high-pressure fluid transport, comprising: a plurality of impellers arranged in series; a common drive shaft; and a pressure-balancing valve assembly configured to reduce axial thrust during operation."
}

def run_quick_verification():
    print("=" * 80)
    print("PATENT_DOC_COPILOT - GENERALIZATION VERIFICATION")
    print("=" * 80)
    
    for domain, abstract in TEST_ABSTRACTS.items():
        print(f"\n{'='*40}")
        print(f"DOMAIN: {domain}")
        print(f"{'='*40}")
        print(f"Abstract: {abstract[:80]}...")
        
        # Test Title
        try:
            title_res = generate_title_from_abstract(abstract)
            print(f"\n✅ TITLE: {title_res['title']}")
        except Exception as e:
            print(f"\n❌ Title failed: {e}")
        
        # Test Field
        try:
            field_res = generate_field_of_invention(abstract)
            field_text = field_res['text'][:200].replace('\n', ' ')
            print(f"\n✅ FIELD: {field_text}...")
        except Exception as e:
            print(f"\n❌ Field failed: {e}")

        # Test Summary
        try:
            summary_text = summarize_abstract(abstract)
            summary_preview = summary_text[:200].replace('\n', ' ')
            print(f"\n✅ SUMMARY: {summary_preview}...")
        except Exception as e:
            print(f"\n❌ Summary failed: {e}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_quick_verification()
