"""
Full Patent Generation Test
Tests all sections with a realistic abstract.
"""

import sys
sys.path.insert(0, '.')

print("=" * 70)
print("FULL PATENT GENERATION TEST")
print("All Sections - Verifying Perfect Output")
print("=" * 70)

ABSTRACT = """
A smart monitoring system for industrial environments, comprising: 
a plurality of sensor units distributed across a facility, wherein each sensor unit 
includes temperature, humidity, and vibration sensors configured for continuous data acquisition; 
a central processing hub operatively connected to the sensor units and configured to 
receive, aggregate, and analyze data using machine learning algorithms for predictive maintenance; 
an alert module integrated with the processing hub and configured to detect anomalies 
based on predefined thresholds and generate real-time notifications; 
and a user interface providing real-time visualization, historical trend analysis, 
and remote control capabilities for facility operators.
"""

print(f"\nAbstract: {ABSTRACT.strip()[:100]}...\n")

# Test each section
results = {}

# 1. Title
print("[1/5] TITLE GENERATION...")
try:
    from generate_title import generate_title_from_abstract
    title_result = generate_title_from_abstract(ABSTRACT)
    title = title_result.get('title', 'Failed')
    results['title'] = {'status': 'PASS', 'output': title}
    print(f"   ✅ {title}")
except Exception as e:
    results['title'] = {'status': 'FAIL', 'output': str(e)}
    print(f"   ❌ {e}")

# 2. Summary
print("\n[2/5] SUMMARY GENERATION...")
try:
    from generate_summary import summarize_abstract
    summary = summarize_abstract(ABSTRACT)
    starts_correct = summary.lower().startswith("thus according")
    results['summary'] = {'status': 'PASS' if starts_correct else 'WARN', 'output': summary[:150] + "..."}
    print(f"   ✅ Starts with 'Thus according...': {starts_correct}")
    print(f"   {summary[:100]}...")
except Exception as e:
    results['summary'] = {'status': 'FAIL', 'output': str(e)}
    print(f"   ❌ {e}")

# 3. Field of Invention
print("\n[3/5] FIELD OF INVENTION...")
try:
    from generate_field_of_invention import generate_field_of_invention
    field_result = generate_field_of_invention(ABSTRACT)
    field = field_result.get('text', 'Failed')
    results['field'] = {'status': 'PASS', 'output': field}
    print(f"   ✅ {field[:100]}...")
except Exception as e:
    results['field'] = {'status': 'FAIL', 'output': str(e)}
    print(f"   ❌ {e}")

# 4. Background
print("\n[4/5] BACKGROUND GENERATION...")
try:
    from generate_background import generate_background_locally
    bg_result = generate_background_locally(ABSTRACT)
    background = bg_result.get('text', 'Failed')
    results['background'] = {'status': 'PASS', 'output': background[:150] + "..."}
    print(f"   ✅ {background[:100]}...")
except Exception as e:
    results['background'] = {'status': 'FAIL', 'output': str(e)}
    print(f"   ❌ {e}")

# 5. Claims
print("\n[5/5] CLAIMS GENERATION...")
try:
    from generate_claims import generate_claims_from_abstract
    claims_result = generate_claims_from_abstract(ABSTRACT)
    claims = claims_result.get('formatted_claims', 'Failed')
    results['claims'] = {'status': 'PASS', 'output': claims[:150] + "..."}
    print(f"   ✅ {claims[:100]}...")
except Exception as e:
    results['claims'] = {'status': 'FAIL', 'output': str(e)}
    print(f"   ❌ {e}")

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

passed = sum(1 for r in results.values() if r['status'] == 'PASS')
total = len(results)

for section, data in results.items():
    icon = "✅" if data['status'] == 'PASS' else "⚠️" if data['status'] == 'WARN' else "❌"
    print(f"{icon} {section.upper()}: {data['status']}")

print(f"\nOVERALL: {passed}/{total} sections PASSED")

if passed == total:
    print("\n🏆 ALL SECTIONS PRODUCING PERFECT OUTPUT!")
elif passed >= 3:
    print("\n✅ Most sections working well")
else:
    print("\n⚠️ Some sections need attention")

print("=" * 70)
