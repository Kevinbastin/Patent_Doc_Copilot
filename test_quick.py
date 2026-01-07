"""
Quick Test - Patent_Doc_Copilot
Tests with a single generic abstract.
"""

from generate_summary import summarize_abstract
from generate_field_of_invention import generate_field_of_invention

# Generic test abstract
TEST_ABSTRACT = """
A smart monitoring system for industrial environments, comprising: 
a plurality of sensor units distributed across a facility; 
a central processing hub receiving data from the sensor units; 
an alert module configured to detect anomalies; and 
a user interface for real-time visualization.
"""

def quick_test():
    print("=" * 60)
    print("QUICK PATENT GENERATION TEST")
    print("=" * 60)
    print(f"Abstract: {TEST_ABSTRACT.strip()[:80]}...")
    
    # Test Summary (this works well)
    print("\n[Summary Generation]")
    try:
        summary = summarize_abstract(TEST_ABSTRACT)
        print(f"✅ Result:\n{summary[:300]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test Field
    print("\n[Field of Invention]")
    try:
        field_res = generate_field_of_invention(TEST_ABSTRACT)
        print(f"✅ Result: {field_res['text']}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    quick_test()
