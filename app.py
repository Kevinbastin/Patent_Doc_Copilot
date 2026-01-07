import streamlit as st
import os
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# ------------ PAGE CONFIG (Must be first) ---------------
st.set_page_config(
    page_title="PatentDoc Co-Pilot | IPO Patent Drafting",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------ PROFESSIONAL CSS STYLING ---------------
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Section headers */
    .section-header {
        background: #F0F4F8;
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #1E88E5;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem;
        font-weight: 600;
        color: #1A1A2E;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Button improvements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------ LOCAL MODULE IMPORTS ---------------
from generate_title import generate_title_from_abstract
from generate_claims import generate_claims_from_abstract
from generate_summary import summarize_abstract
from generate_field_of_invention import generate_field_of_invention
from generate_background import generate_background_locally
from generate_detailed_description import generate_detailed_description
from generate_brief_description import generate_brief_description
from component_registry import create_unified_registry, ComponentRegistry
from generate_summary_of_drawings import generate_drawing_descriptions as generate_drawing
from generate_industrial_applicability import generate_industrial_applicability
from generate_objects import generate_objects_of_invention
from ipc_classifier import classify_cpc as classify_ipc
from export_to_pdf import create_patent_pdf
from prior_art_checker import check_prior_art, format_prior_art_report
from generate_diagrams import generate_all_diagrams, get_figure_descriptions
from patent_drawing_generator import generate_patent_drawings, DrawingType
from patent_image_generator import PatentImageGenerator, extract_components_for_images
from patent_validator import PatentValidator, validate_abstract_live, validate_title_live, get_section_checklist

# -------------- PROFESSIONAL HEADER ------------------
st.markdown("""
<div class="main-header">
    <h1>📝 PatentDoc Co-Pilot</h1>
    <p>Enterprise-Level IPO-Compliant Patent Drafting System</p>
</div>
""", unsafe_allow_html=True)

with st.expander("⚠️ Legal Disclaimer & AI Usage Notice", expanded=False):
    st.markdown("""
    This tool uses local AI language models (Phi-3, Llama.cpp, etc.) to help draft patent documents.

    ⚠️ **Warning:** The AI-generated content may contain factual errors, legal inaccuracies, or formatting that does not comply with Indian Patent Office (IPO) guidelines.
    Always consult a qualified **patent attorney** before relying on the generated material for filing or legal use.

    📜 **Model License Notice:** The underlying LLMs (e.g., Phi-3) are subject to their respective open-source licenses.

    🔍 **Responsibility Disclaimer:** This app is a research prototype. It is **not** a substitute for professional legal services or patent counsel.
    """)

# ---------------- MAIN INPUT FIELDS ---------------------
abstract = st.text_area("📄 Enter Invention Abstract", height=200, 
                        help="Enter your invention description - it will be enhanced to IPO format")
st.session_state["abstract_input"] = abstract

# Enhance Abstract Button
col_enhance, col_info = st.columns([1, 3])
with col_enhance:
    enhance_clicked = st.button("✨ Enhance to IPO Format", help="Convert rough description to proper IPO abstract")

if enhance_clicked and abstract:
    with st.spinner("🔄 Enhancing abstract to IPO format..."):
        try:
            from enhance_abstract import enhance_abstract
            result = enhance_abstract(abstract)
            
            if result["status"] == "success" or result["status"] == "needs_review":
                st.session_state["enhanced_abstract"] = result["enhanced_abstract"]
                st.session_state["enhance_result"] = result
                st.rerun()  # Rerun to show the result
            else:
                st.error(f"❌ Enhancement failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"❌ Error enhancing abstract: {e}")

# Display enhanced abstract if available (persists after rerun)
if st.session_state.get("enhance_result") and st.session_state.get("enhanced_abstract"):
    result = st.session_state["enhance_result"]
    enhanced = st.session_state["enhanced_abstract"]
    
    st.divider()
    st.success(f"✅ Abstract enhanced! ({result.get('word_count', len(enhanced.split()))} words)")
    st.markdown("### ✨ Enhanced IPO Abstract")
    st.code(enhanced, language=None)
    
    # Show validation
    validation = result.get("validation", {})
    if validation.get("in_preferred_range"):
        st.success("✅ Word count in preferred range (50-150)")
    if validation.get("warnings"):
        for w in validation["warnings"]:
            st.warning(f"⚠️ {w}")
    
    # Clear button
    if st.button("🔄 Use Original Abstract Instead", key="clear_enhanced"):
        st.session_state["enhanced_abstract"] = None
        st.session_state["enhance_result"] = None
        st.rerun()
    st.divider()

# Use enhanced abstract if available
if st.session_state.get("enhanced_abstract"):
    abstract = st.session_state["enhanced_abstract"]

# Live Abstract Word Count Validation
if abstract:
    word_count = PatentValidator.count_words(abstract)
    if word_count > 250:
        st.error(f"❌ Abstract: {word_count}/250 words - **Too long!** Reduce by {word_count - 250} words.")
    elif word_count < 50:
        st.warning(f"⚠️ Abstract: {word_count}/250 words - Consider adding more detail (100-250 recommended)")
    else:
        st.success(f"✅ Abstract: {word_count}/250 words")

drawing_summary = st.text_area("🎨 Enter Drawing Summary (optional)", height=150)

# ---------------- APPLICANT/INVENTOR DETAILS (for IPO Forms) --------------------
with st.expander("📋 Applicant & Inventor Details (for IPO Forms 1, 3, 5)", expanded=False):
    st.info("💡 Fill these details to generate complete IPO filing forms")
    
    col_app, col_inv = st.columns(2)
    
    with col_app:
        st.markdown("**APPLICANT DETAILS**")
        applicant_name = st.text_input("Applicant Name", key="applicant_name", 
                                       placeholder="e.g., John Doe / ABC Technologies Pvt Ltd")
        applicant_address = st.text_area("Applicant Address", key="applicant_address", height=80,
                                         placeholder="Full address including PIN code")
        applicant_nationality = st.selectbox("Nationality", 
                                             ["INDIAN", "FOREIGN"], key="applicant_nationality")
        applicant_category = st.selectbox("Category (for fee calculation)", 
                                          ["NATURAL PERSON", "STARTUP", "SMALL ENTITY", "EDUCATIONAL INSTITUTION", "OTHER"],
                                          key="applicant_category")
    
    with col_inv:
        st.markdown("**INVENTOR DETAILS**")
        same_as_applicant = st.checkbox("Same as Applicant", value=True, key="same_inventor")
        if same_as_applicant:
            inventor_name = applicant_name
            inventor_address = applicant_address
            inventor_nationality = applicant_nationality
            st.text_input("Inventor Name", value=applicant_name or "", disabled=True)
        else:
            inventor_name = st.text_input("Inventor Name", key="inventor_name")
            inventor_address = st.text_area("Inventor Address", key="inventor_address", height=80)
            inventor_nationality = st.selectbox("Inventor Nationality", 
                                                ["INDIAN", "FOREIGN"], key="inventor_nationality")
    
    # Store in session state
    st.session_state["applicant_details"] = {
        "name": applicant_name if 'applicant_name' in dir() else "",
        "address": applicant_address if 'applicant_address' in dir() else "",
        "nationality": applicant_nationality if 'applicant_nationality' in dir() else "INDIAN",
        "category": applicant_category if 'applicant_category' in dir() else "NATURAL PERSON"
    }
    st.session_state["inventor_details"] = {
        "name": inventor_name if 'inventor_name' in dir() else "",
        "address": inventor_address if 'inventor_address' in dir() else "",
        "nationality": inventor_nationality if 'inventor_nationality' in dir() else "INDIAN"
    }

# ----------------- SESSION STATE INIT -------------------
for key in [
    "title", "claims", "summary", "field_of_invention",
    "background","objects_of_invention", "detailed_description", "brief_description", 
    "summary_drawings", "diagrams", "ipc_result", "prior_art_result",
    "image_components", "image_prompts", "generated_images"
]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ------------------- PRIOR ART CHECK (INDIAN FOCUS) -------------------
st.markdown("---")
st.subheader("🔍 Step 1: Prior Art Check (Before Drafting)")
st.info("💡 **Recommended:** Check for similar Indian patents before drafting to ensure novelty.")

col1, col2 = st.columns([2, 1])
with col1:
    # Regular API-based search
    if st.button("🔍 Quick Search (API)", type="secondary"):
        if abstract:
            with st.spinner("🔎 Searching patent databases via APIs..."):
                try:
                    from prior_art_checker import check_prior_art, format_prior_art_report
                    results = check_prior_art(abstract)
                    st.session_state.prior_art_result = results
                    st.success("✅ Quick Search Complete!")
                except Exception as e:
                    st.error(f"❌ Search failed: {e}")
        else:
            st.warning("⚠️ Please enter an abstract first.")
    
    # Multi-Agent Deep Analysis
    if st.button("🤖 Deep Analysis (5 Agents)", type="primary"):
        if abstract:
            with st.spinner("🤖 Running 5-Agent CrewAI Analysis (2-3 minutes)..."):
                try:
                    # First get API results
                    from prior_art_checker import check_prior_art
                    api_results = check_prior_art(abstract)
                    
                    # Then run multi-agent analysis
                    from prior_art_multi_agent import analyze_prior_art_with_agents, format_agent_report
                    agent_results = analyze_prior_art_with_agents(abstract, api_results)
                    
                    # Combine results
                    combined = {
                        **api_results,
                        "multi_agent_analysis": agent_results.get("full_analysis", ""),
                        "agent_recommendation": agent_results.get("recommendation", ""),
                        "analysis_type": "multi_agent"
                    }
                    st.session_state.prior_art_result = combined
                    st.success("✅ 5-Agent Deep Analysis Complete!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Multi-agent analysis failed: {e}")
        else:
            st.warning("⚠️ Please enter an abstract first.")

with col2:
    st.markdown("""
    **This checks:**
    - 🇮🇳 Indian patents (via InPASS)
    - 🌍 Global patents
    - 📊 Novelty assessment
    """)

if st.session_state.get('prior_art_result'):
    results = st.session_state.prior_art_result
    
    # Check if multi-agent analysis was used
    is_multi_agent = results.get('analysis_type') == 'multi_agent'
    
    if is_multi_agent:
        st.markdown("### 🤖 Multi-Agent Analysis Result")
        recommendation = results.get('agent_recommendation', 'N/A')
        if "PROCEED" in recommendation.upper():
            st.success(recommendation)
        elif "DO NOT" in recommendation.upper():
            st.error(recommendation)
        else:
            st.warning(recommendation)
        
        with st.expander("📊 Full 5-Agent Analysis", expanded=True):
            st.markdown(results.get('multi_agent_analysis', 'No analysis available'))
    
    # Display novelty assessment prominently
    st.markdown("### 📊 Novelty Assessment")
    novelty = results.get('novelty_assessment', '')
    novelty_score = results.get('novelty_score', 50)
    risk_level = results.get('risk_level', 'MEDIUM')
    
    # Display score with color coding
    if novelty_score >= 70 or risk_level == "LOW":
        st.success(f"🟢 **Novelty Score: {novelty_score}/100** | Risk: {risk_level}")
    elif novelty_score >= 40 or risk_level == "MEDIUM":
        st.warning(f"🟡 **Novelty Score: {novelty_score}/100** | Risk: {risk_level}")
    else:
        st.error(f"🔴 **Novelty Score: {novelty_score}/100** | Risk: {risk_level}")
    
    if novelty:
        st.info(novelty)
    
    # Analysis
    analysis = results.get('analysis', '')
    if analysis:
        with st.expander("📝 Detailed Analysis", expanded=True):
            st.markdown(analysis)
    
    # Similar Patents Found
    patents = results.get('patents_found', [])
    if patents:
        st.markdown("### 📑 Similar Patents Found")
        for i, p in enumerate(patents[:6], 1):
            sim_score = p.get('similarity_score', 0)
            emoji = "🔴" if sim_score > 70 else "🟡" if sim_score > 40 else "🟢"
            with st.expander(f"{emoji} {p.get('patent_number', 'N/A')} ({sim_score}% similar)", expanded=False):
                st.markdown(f"**Title:** {p.get('title', 'N/A')}")
                st.markdown(f"**Source:** {p.get('source', 'N/A')}")
                link = p.get('link', '')
                if link:
                    st.markdown(f"[🔗 View Patent]({link})")
    
    # IPC Codes
    ipc_codes = results.get('ipc_codes', results.get('ipc_codes_suggested', []))
    if ipc_codes:
        st.markdown("### 📋 IPC Codes for IPO Filing")
        st.code(" | ".join(ipc_codes))
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Recommendations")
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # ==================== MANUAL SEARCH BUTTONS ====================
    st.markdown("---")
    st.markdown("### 🔍 Manual Patent Search (Click to Open)")
    st.info("📌 **Tip:** Use the IPC codes above and keywords from your abstract for best results")
    
    # Generate pre-filled search URLs
    search_terms = results.get('search_terms_used', [])
    query = "+".join(search_terms[:6]) if search_terms else "patent+search"
    
    # Row 1: Official Indian Patent Office
    st.markdown("#### 🇮🇳 Indian Patent Office (IPO)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <a href="https://iprsearch.ipindia.gov.in/publicsearch" target="_blank" style="
            display: block;
            background: linear-gradient(135deg, #FF9933, #138808);
            color: white;
            padding: 15px 20px;
            text-align: center;
            text-decoration: none;
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            margin: 5px 0;
        ">🇮🇳 InPASS - Indian Patent Search</a>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <a href="https://ipindia.gov.in/journal-patent.htm" target="_blank" style="
            display: block;
            background: linear-gradient(135deg, #138808, #FF9933);
            color: white;
            padding: 15px 20px;
            text-align: center;
            text-decoration: none;
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            margin: 5px 0;
        ">📰 IPO Patent Journal</a>
        """, unsafe_allow_html=True)
    
    # Row 2: Global Patent Databases
    st.markdown("#### 🌍 Global Patent Databases")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <a href="https://patents.google.com/?q={query}" target="_blank" style="
            display: block;
            background: #4285F4;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">🔍 Google Patents</a>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <a href="https://worldwide.espacenet.com/patent/search?q={query}" target="_blank" style="
            display: block;
            background: #003399;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">🇪🇺 Espacenet</a>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <a href="https://patentscope.wipo.int/search/en/search.jsf?query={query}" target="_blank" style="
            display: block;
            background: #00A0D2;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">🌐 WIPO</a>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <a href="https://ppubs.uspto.gov/pubwebapp/static/pages/searchable/home.html" target="_blank" style="
            display: block;
            background: #002868;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">🇺🇸 USPTO</a>
        """, unsafe_allow_html=True)
    
    # Row 3: Research Databases
    st.markdown("#### 📚 Research & Academic")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <a href="https://scholar.google.com/scholar?q={query}" target="_blank" style="
            display: block;
            background: #4285F4;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">📖 Google Scholar</a>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <a href="https://www.lens.org/lens/search/patent/structured" target="_blank" style="
            display: block;
            background: #6B4C9A;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">🔬 Lens.org</a>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <a href="https://www.freepatentsonline.com/" target="_blank" style="
            display: block;
            background: #2E7D32;
            color: white;
            padding: 12px 10px;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
        ">📋 Free Patents</a>
        """, unsafe_allow_html=True)
    
    # Search Tips
    with st.expander("💡 How to Search Effectively", expanded=False):
        st.markdown("""
        **For IPO (Indian Patent Office):**
        1. Go to InPASS and select "Patent" search
        2. Enter IPC codes from above (e.g., G06F, H04L)
        3. Add keywords from your invention
        4. Check both granted patents AND applications
        
        **For Global Databases:**
        1. Use the same IPC codes for consistency
        2. Try different keyword combinations
        3. Check patents from the last 20 years
        4. Look for both utility patents and applications
        
        **Keywords to try:**
        """ + ", ".join(search_terms[:8] if search_terms else ["your", "keywords", "here"]))

st.markdown("---")

# ===================== ENTERPRISE: GENERATE ALL SECTIONS =====================
st.markdown("## 🚀 Enterprise Mode: One-Click Generation")

col_gen1, col_gen2 = st.columns(2)

with col_gen1:
    if st.button("⚡ Generate ALL Patent Sections", type="primary", use_container_width=True):
        if not abstract:
            st.warning("Please enter an abstract first.")
        else:
            progress_bar = st.progress(0, text="Creating unified component registry...")
            
            # CREATE UNIFIED REGISTRY ONCE - This ensures consistent reference numerals across ALL sections
            # User's reference numerals from drawing_summary take PRIORITY
            try:
                registry = create_unified_registry(abstract, drawing_summary)
                st.session_state.component_registry = registry.to_dict()
                user_refs_count = len([k for k,v in registry.components.items() if v in [10,15,20,25,30,35,40,45,50]])
                st.success(f"✅ Component registry created: {len(registry.components)} components (including {user_refs_count} from your drawing summary)")
            except Exception as e:
                st.warning(f"⚠️ Registry creation warning: {e}. Using defaults.")
                registry = ComponentRegistry()
                registry._create_fallback_components(abstract)
                st.session_state.component_registry = registry.to_dict()
            
            progress_bar.progress(0.05, text="Starting generation...")
            
            sections = [
                ("Title", generate_title_from_abstract),
                ("Claims", generate_claims_from_abstract),
                ("Summary", summarize_abstract),
                ("Field of Invention", generate_field_of_invention),
                ("Background", generate_background_locally),
                ("Objects", generate_objects_of_invention),
                ("Brief Description", generate_brief_description),
                ("Industrial Applicability", generate_industrial_applicability),
            ]
            
            total = len(sections) + 1  # +1 for detailed description
            
            for i, (name, func) in enumerate(sections):
                progress_bar.progress((i / total), text=f"Generating {name}...")
                try:
                    result = func(abstract)
                    key = name.lower().replace(" ", "_").replace("of_", "of_")
                    
                    # Handle different return types
                    if isinstance(result, dict):
                        value = result.get("text", result.get("title", result.get(key, str(result))))
                    else:
                        value = str(result) if result else ""
                    
                    # Store in correct session state key
                    if name == "Title":
                        st.session_state.title = value
                    elif name == "Claims":
                        # Claims need registry for consistent numerals - generate separately
                        claims_result = generate_claims_from_abstract(
                            abstract,
                            component_registry=st.session_state.get("component_registry")
                        )
                        st.session_state.claims = claims_result if isinstance(claims_result, str) else claims_result.get("text", str(claims_result))
                    elif name == "Summary":
                        st.session_state.summary = value
                    elif name == "Field of Invention":
                        st.session_state.field_of_invention = value
                    elif name == "Background":
                        st.session_state.background = value
                    elif name == "Objects":
                        st.session_state.objects_of_invention = value
                    elif name == "Brief Description":
                        st.session_state.brief_description = value
                    elif name == "Industrial Applicability":
                        st.session_state.industrial_applicability = value
                        
                except Exception as e:
                    st.warning(f"⚠️ {name} generation failed: {e}")
            
            # Generate Detailed Description (needs claims and registry for consistent numerals)
            progress_bar.progress(0.9, text="Generating Detailed Description (this takes longer)...")
            try:
                claims = st.session_state.get("claims", "")
                result = generate_detailed_description(
                    abstract=abstract,
                    claims=claims,
                    drawing_summary=drawing_summary,
                    field_of_invention=st.session_state.get("field_of_invention", ""),
                    background=st.session_state.get("background", ""),
                    objects=st.session_state.get("objects_of_invention", ""),
                    component_registry=st.session_state.get("component_registry")  # Pass unified registry
                )
                if isinstance(result, dict):
                    st.session_state.detailed_description = result.get("text", "")
                else:
                    st.session_state.detailed_description = str(result)
            except Exception as e:
                st.warning(f"⚠️ Detailed Description failed: {e}")
            
            progress_bar.progress(1.0, text="✅ All sections generated!")
            st.success("✅ All patent sections generated! Scroll down to review.")
            st.balloons()

with col_gen2:
    st.info("""
    **Enterprise One-Click generates:**
    - Title (≤15 words)
    - Claims (IPO format)
    - Summary
    - Field of Invention
    - Background
    - Objects
    - Brief Description of Drawings
    - Detailed Description (with embodiments)
    - Industrial Applicability
    """)

st.markdown("---")

# ------------------- GENERATION BUTTONS -----------------
# ------------------- GENERATION BUTTONS (FIXED) -----------------
# ------------------- GENERATION BUTTONS (FIXED) -----------------
if st.button("📌 Generate Title"):
    with st.spinner("Generating title..."):
        try:
            result = generate_title_from_abstract(abstract)
            if isinstance(result, dict):
                st.session_state.title = result.get("title", "")
            else:
                st.session_state.title = result
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Title generation failed: {e}")

if st.session_state.title:
    title_text = st.session_state.title
    title_word_count = PatentValidator.count_words(title_text)
    
    with st.expander("📘 Title"):
        st.write(title_text)
        
        # Title Word Count Validation
        if title_word_count > 15:
            st.error(f"❌ Title: {title_word_count}/15 words - **Too long!** Reduce by {title_word_count - 15} words.")
        elif title_word_count < 5:
            st.warning(f"⚠️ Title: {title_word_count}/15 words - Consider making it more descriptive")
        else:
            st.success(f"✅ Title: {title_word_count}/15 words - IPO compliant")


if st.button("🔖 Generate Claims"):
    with st.spinner("Generating claims..."):
        try:
            # Use unified registry for consistent reference numerals
            result = generate_claims_from_abstract(
                abstract,
                component_registry=st.session_state.get("component_registry")
            )
            if isinstance(result, dict):
                st.session_state.claims = result.get("text", result.get("claims", ""))
            else:
                st.session_state.claims = result
            st.success("Done!")
        except Exception as e:
            st.error(f"Claim generation failed: {e}")

if st.session_state.claims:
    with st.expander("🧾 Claims"):
        st.write(st.session_state.claims)


if st.button("🧷 Generate Summary"):
    with st.spinner("Generating summary..."):
        try:
            result = summarize_abstract(abstract)
            if isinstance(result, dict):
                st.session_state.summary = result.get("text", result.get("summary", ""))
            else:
                st.session_state.summary = result
            st.success("✅ Summary generated!")
        except Exception as e:
            st.error(f"❌ Summary generation failed: {e}")

if st.session_state.summary:
    with st.expander("📄 Summary"):
        st.write(st.session_state.summary)


if st.button("📚 Field of the Invention"):
    with st.spinner("Generating field of the invention..."):
        try:
            result = generate_field_of_invention(abstract)
            if isinstance(result, dict):
                st.session_state.field_of_invention = result.get("text", result.get("field", ""))
            else:
                st.session_state.field_of_invention = result
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Field generation failed: {e}")

if st.session_state.field_of_invention:
    with st.expander("📘 Field of the Invention"):
        st.write(st.session_state.field_of_invention)


if st.button("🧠 Background"):
    with st.spinner("Generating background..."):
        try:
            result = generate_background_locally(abstract)
            if isinstance(result, dict):
                st.session_state.background = result.get("text", result.get("background", ""))
            else:
                st.session_state.background = result
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Background generation failed: {e}")

if st.session_state.background:
    with st.expander("🔍 Background"):
        st.write(st.session_state.background)


if st.button("🎯 Objects of the Invention"):
    with st.spinner("Generating objects of the invention..."):
        try:
            result = generate_objects_of_invention(abstract)
            if isinstance(result, dict):
                text = result.get("text", result.get("objects", ""))
            else:
                text = result
            
            # Clean markdown headers and formatting
            import re
            text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # Remove markdown headers
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
            text = re.sub(r'__([^_]+)__', r'\1', text)  # Remove underline
            
            st.session_state.objects_of_invention = text
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Objects generation failed: {e}")

if st.session_state.objects_of_invention:
    with st.expander("🎯 Objects of the Invention"):
        st.text(st.session_state.objects_of_invention)  # Use st.text instead of st.write


if st.button("📝 Detailed Description"):
    # Check if claims exist first
    if not st.session_state.get("claims"):
        st.warning("⚠️ Please generate Claims first!")
    else:
        with st.spinner("Generating detailed description (this may take 30-60 seconds)..."):
            try:
                # Get claims and drawing summary
                claims_text = st.session_state.get("claims", "Claims not generated yet.")
                drawings_text = drawing_summary if drawing_summary and drawing_summary.strip() else "No drawings provided."
                
                # Debug info
                st.write(f"**Debug Info:**")
                st.write(f"- Abstract length: {len(abstract)} chars")
                st.write(f"- Claims length: {len(claims_text)} chars")
                st.write(f"- Drawings length: {len(drawings_text)} chars")
                
                # Generate
                result = generate_detailed_description(
                    abstract,
                    claims_text,
                    drawings_text
                )
                
                # Handle result
                if isinstance(result, dict):
                    text = result.get("text", result.get("description", ""))
                else:
                    text = result
                
                # Debug output
                st.write(f"**Generated length:** {len(text)} characters")
                
                # Store
                if text and len(text) > 50:
                    st.session_state.detailed_description = text
                    st.success("✅ Done! Scroll down to see the detailed description.")
                else:
                    st.session_state.detailed_description = "⚠️ Generated description is too short or empty."
                    st.error("⚠️ Generated description is too short. Check model output.")
                    
            except Exception as e:
                error_msg = f"❌ Exception: {str(e)}"
                st.session_state.detailed_description = error_msg
                st.error(f"❌ Detailed description generation failed: {e}")
                st.exception(e)  # Show full traceback

if st.session_state.get("detailed_description") and len(st.session_state.get("detailed_description", "")) > 50:
    with st.expander("📘 Detailed Description", expanded=True):
        st.markdown(st.session_state.detailed_description)

if st.button("📊 Brief Description of Drawings"):
    if not abstract:
        st.warning("Please enter the abstract.")
    else:
        with st.spinner("Generating brief description of drawings..."):
            try:
                # CRITICAL: Use get_figure_descriptions if diagrams were already generated
                # This ensures descriptions match actual diagram types (block diagram, flowchart, sequence)
                if st.session_state.get("diagrams"):
                    # Use descriptions that match actual generated diagrams
                    brief_desc = get_figure_descriptions(st.session_state.diagrams)
                    st.session_state.brief_description = brief_desc
                    st.info("ℹ️ Brief Description synced with generated diagrams (block diagram, flowchart, sequence diagram)")
                else:
                    # Fallback to LLM-based generation if no diagrams yet
                    result = generate_brief_description(abstract, drawing_summary)
                    if isinstance(result, dict):
                        st.session_state.brief_description = result.get("text", result.get("description", ""))
                    else:
                        st.session_state.brief_description = result or "⚠️ No output generated."
                    st.warning("⚠️ Generate Diagrams first for accurate Brief Description matching actual drawings.")
                st.success("Done!")
            except Exception as e:
                st.error(f"❌ Brief description generation failed: {e}")

if st.session_state.get("brief_description"):
    with st.expander("🖼️ Brief Description of the Drawings"):
        st.write(st.session_state.brief_description)


if st.button("🏭 Industrial Applicability"):
    if not abstract:
        st.warning("Please enter the abstract.")
    else:
        with st.spinner("Generating industrial applicability..."):
            try:
                result = generate_industrial_applicability(
                    abstract=abstract,
                    field_of_invention=field_of_invention if 'field_of_invention' in locals() else ""
                )

                if isinstance(result, dict):
                    st.session_state.industrial_applicability = result.get("text", "")
                else:
                    st.session_state.industrial_applicability = result or "⚠️ No output generated."

                st.success("Done!")
            except Exception as e:
                st.error(f"❌ Industrial applicability generation failed: {e}")

if st.session_state.get("industrial_applicability"):
    with st.expander("🏭 Industrial Applicability"):
        st.write(st.session_state.industrial_applicability)

if st.button("🖼️ Summary of Drawings"):
    if not abstract:
        st.warning("Please enter the invention abstract.")
    else:
        with st.spinner("Generating summary of drawings..."):
            try:
                result = generate_drawing(abstract)
                if isinstance(result, dict):
                    st.session_state.summary_drawings = result.get("text", "")
                else:
                    st.session_state.summary_drawings = result or "⚠️ No output generated."
                st.success("Done!")
            except Exception as e:
                st.error(f"❌ Drawing summary failed: {e}")

if st.session_state.get("summary_drawings"):
    with st.expander("📷 Summary of Drawings"):
        st.write(st.session_state.summary_drawings)


# ------------------ PATENT DIAGRAMS (Auto-Generate) ----------------------
st.markdown("## 📊 Patent Diagrams (IPO Figures)")

# Detect if user provided physical view descriptions
physical_view_keywords = ['perspective', 'exploded', 'cross-section', 'isometric', 'elevation', 
                          'front view', 'side view', 'top view', 'bottom view', 'rear view',
                          'sectional', 'assembly', 'detail view']
has_physical_views = any(kw in drawing_summary.lower() for kw in physical_view_keywords) if drawing_summary else False

if has_physical_views:
    st.warning("""
    ⚠️ **Physical Drawing Views Detected**
    
    Your drawing summary describes physical views (perspective, exploded, cross-section, etc.) which require **manual CAD/illustration tools** to create accurately.
    
    The auto-generated diagrams below are **schematic diagrams** (block diagram, flowchart, sequence) suitable for:
    - System architecture documentation
    - Method/process visualization
    - Component interaction flows
    
    For physical engineering views, please use:
    - **CAD software** (AutoCAD, SolidWorks, Fusion 360)
    - **Illustration tools** (Adobe Illustrator, Inkscape)
    - **AI image generators** (see Image Generation section below)
    
    The **Brief Description of Drawings** section will correctly use YOUR descriptions for physical views.
    """)
else:
    st.info("🎨 Auto-generate patent diagrams: Block Diagram (Fig 1), Flowchart (Fig 2), Sequence Diagram (Fig 3)")

# Store physical views detection in session state for PDF export
st.session_state.has_physical_views = has_physical_views

if has_physical_views:
    st.info("""
    📝 **Your Brief Description of Drawings is ready** (see above section).
    
    Since you've described physical views (perspective, exploded, cross-section), 
    please create actual drawings using CAD software and include them in your filing.
    
    The PDF export will include placeholders: `[Insert Figure X Drawing Here]`
    """)
    
    # Don't show diagram generation button for physical views
    if st.button("📊 Generate Schematic Diagrams Anyway (for reference only)"):
        if abstract:
            with st.spinner("Generating reference diagrams..."):
                try:
                    # CRITICAL: Pass Claim 1 to ensure Claims ↔ Drawings consistency (Section 10(4)(c))
                    claim_1_text = None
                    if st.session_state.get("claims"):
                        claims_text = st.session_state.claims
                        # Extract Claim 1 from claims text
                        if "1." in claims_text:
                            claim_1_text = claims_text.split("2.")[0] if "2." in claims_text else claims_text
                    
                    diagrams = generate_all_diagrams(abstract, drawing_summary, claim_1_text)
                    st.session_state.diagrams = diagrams
                    st.warning("⚠️ These are SCHEMATIC diagrams, not the physical views you described. Use for reference only.")
                    # Show any additional warnings from parsing
                    for warn in diagrams.get("drawing_warnings", []):
                        st.warning(warn)
                except Exception as e:
                    st.error(f"❌ Diagram generation failed: {e}")
else:
    if st.button("📊 Generate Patent Diagrams"):
        if not abstract:
            st.warning("Please enter the invention abstract.")
        else:
            with st.spinner("Generating IPO-compliant diagrams using AI..."):
                try:
                    # CRITICAL: Pass Claim 1 to ensure Claims ↔ Drawings consistency (Section 10(4)(c))
                    claim_1_text = None
                    if st.session_state.get("claims"):
                        claims_text = st.session_state.claims
                        # Extract Claim 1 from claims text
                        if "1." in claims_text:
                            claim_1_text = claims_text.split("2.")[0] if "2." in claims_text else claims_text
                    
                    diagrams = generate_all_diagrams(abstract, drawing_summary, claim_1_text)
                    st.session_state.diagrams = diagrams
                    
                    # SYNC: Copy physical drawings to display section if generated
                    if diagrams.get("has_physical_drawings") and diagrams.get("physical_drawings"):
                        st.session_state.physical_drawings = {
                            "drawings": diagrams["physical_drawings"],
                            "brief_description": diagrams.get("physical_brief_description", ""),
                            "reference_numerals": diagrams.get("components", {}).get("reference_numerals", {}),
                            "warnings": ["Auto-generated from diagram pipeline"],
                            "success": True
                        }
                        st.success(f"✅ Diagrams + Physical drawings generated! (Type: {diagrams.get('invention_type', 'unknown')})")
                    elif diagrams.get("claim_1_synced"):
                        st.success("✅ Diagrams generated from Claim 1 - Section 10(4)(c) compliant!")
                    else:
                        st.success("✅ Diagrams generated. Generate Claims first for best consistency.")
                    
                    for warn in diagrams.get("drawing_warnings", []):
                        st.info(warn)
                except Exception as e:
                    st.error(f"❌ Diagram generation failed: {e}")

# ============== AI PHYSICAL DRAWING GENERATION ================
st.markdown('<div class="section-header">🏗️ AI Physical Drawings (CAD-Style)</div>', unsafe_allow_html=True)
st.info("""
**NEW! Generate AI-powered physical drawings:**
- **Isometric View** (FIG. 1) - 3D external view
- **Exploded View** (FIG. 2) - Assembly components  
- **Cross-Section View** (FIG. 3) - Internal structure
- **Detail View** (FIG. 4) - Component close-up

⚠️ These are AI-generated conceptual drawings. May need refinement for final IPO submission.
""")

if st.button("🏗️ Generate AI Physical Drawings"):
    if not abstract:
        st.warning("Please enter the invention abstract.")
    else:
        with st.spinner("Generating AI-powered physical drawings..."):
            try:
                # Get claim 1 for component extraction
                claim_1_text = None
                if st.session_state.get("claims"):
                    claims_text = st.session_state.claims
                    if "1." in claims_text:
                        claim_1_text = claims_text.split("2.")[0] if "2." in claims_text else claims_text
                
                # Generate drawings
                result = generate_patent_drawings(abstract, claim_1_text)
                st.session_state.physical_drawings = result
                
                st.success("✅ Physical drawing placeholders generated!")
                
                # Show warnings
                for warn in result.get("warnings", []):
                    st.warning(f"⚠️ {warn}")
                
            except Exception as e:
                st.error(f"❌ Physical drawing generation failed: {e}")

# Display generated physical drawings
if st.session_state.get("physical_drawings"):
    result = st.session_state.physical_drawings
    
    st.markdown("### 📐 Generated Drawing Placeholders")
    
    # Show reference numerals
    st.markdown("**Reference Numerals:**")
    ref_nums = result.get("reference_numerals", {})
    ref_text = " | ".join([f"({v}) {k[:30]}" for k, v in sorted(ref_nums.items(), key=lambda x: x[1])[:8]])
    st.code(ref_text)
    
    # Show each drawing
    drawings = result.get("drawings", {})
    for dtype, drawing in drawings.items():
        with st.expander(f"📄 Figure {drawing.figure_number}: {dtype.upper().replace('_', ' ')} VIEW"):
            st.markdown(f"**Description:** {drawing.description}")
            
            if drawing.image_path and os.path.exists(drawing.image_path):
                if drawing.image_path.endswith('.png'):
                    st.image(drawing.image_path, caption=f"Figure {drawing.figure_number}")
                else:
                    with open(drawing.image_path, 'r') as f:
                        st.text(f.read())
            
            st.markdown("**Components to show:**")
            for comp in drawing.components[:6]:
                ref = drawing.reference_numerals.get(comp, "?")
                st.markdown(f"- ({ref}) {comp}")
            
            if drawing.warnings:
                for w in drawing.warnings:
                    st.warning(w)
    
    # Show brief description
    st.markdown("### 📝 Brief Description of Drawings")
    st.text_area("Copy this for your patent document:", result.get("brief_description", ""), height=200)


if st.session_state.get("diagrams"):
    diagrams = st.session_state.diagrams
    
    # View mode toggle
    view_mode = st.radio(
        "📺 View Mode:",
        ["Preview (Colored)", "📋 IPO Formal (for Patent Filing)"],
        horizontal=True,
        help="Preview mode shows colored diagrams. IPO Formal mode shows black/white diagrams with reference numerals suitable for Indian Patent Office filing."
    )
    
    if view_mode == "Preview (Colored)":
        # Preview diagrams (colored Mermaid - for quick visualization)
        st.markdown("### 🎨 Preview Diagrams (Colored)")
        
        # Figure 1 - Block Diagram
        with st.expander("📊 Figure 1: Block Diagram (System Architecture)", expanded=True):
            st.markdown("**Block diagram showing main components and connections**")
            st.markdown(diagrams.get("fig1_block", ""))
        
        # Figure 2 - Flowchart
        with st.expander("📈 Figure 2: Flowchart (Method/Process)", expanded=True):
            st.markdown("**Flowchart depicting the method steps**")
            st.markdown(diagrams.get("fig2_flowchart", ""))
        
        # Figure 3 - Sequence Diagram  
        with st.expander("🔄 Figure 3: Sequence Diagram (Interactions)", expanded=True):
            st.markdown("**Sequence diagram showing component interactions**")
            st.markdown(diagrams.get("fig3_sequence", ""))
    
    else:
        # IPO Formal diagrams (black/white with reference numerals)
        st.markdown("### 📋 IPO-Compliant Patent Figures")
        st.success("✅ These diagrams follow Indian Patent Office drawing specifications:")
        st.markdown("""
        - ✓ **Black & White only** (no colors)
        - ✓ **Reference numerals** (100, 110, 120...)
        - ✓ **Step numbering** (S101, S102...) for flowcharts
        - ✓ **Standard patent format** with figure captions
        """)
        
        # Figure 1 - Block Diagram (IPO)
        with st.expander("📊 Figure 1: Block Diagram (IPO Format)", expanded=True):
            st.markdown(diagrams.get("fig1_block_ipo", ""))
        
        # Figure 2 - Flowchart (IPO)
        with st.expander("📈 Figure 2: Flowchart (IPO Format)", expanded=True):
            st.markdown(diagrams.get("fig2_flowchart_ipo", ""))
        
        # Figure 3 - Sequence Diagram (IPO)
        with st.expander("🔄 Figure 3: Sequence Diagram (IPO Format)", expanded=True):
            st.markdown(diagrams.get("fig3_sequence_ipo", ""))
        
        # Reference Numerals Table
        with st.expander("📋 Reference Numerals Table", expanded=True):
            st.code(diagrams.get("reference_numerals", "No reference numerals generated."))
    
    # Brief Description of Drawings (for patent specification)
    st.markdown("---")
    with st.expander("📝 Brief Description of Drawings (IPO Format)", expanded=False):
        fig_desc = get_figure_descriptions(diagrams)
        st.text(fig_desc)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Brief Description",
                data=fig_desc,
                file_name="brief_description_of_drawings.txt",
                mime="text/plain"
            )
        with col2:
            # Download all diagrams as a combined text file
            all_diagrams_text = f"""PATENT DRAWINGS - IPO COMPLIANT
================================

{diagrams.get("fig1_block_ipo", "")}

---

{diagrams.get("fig2_flowchart_ipo", "")}

---

{diagrams.get("fig3_sequence_ipo", "")}

---

{fig_desc}
"""
            st.download_button(
                label="📥 Download All Diagrams",
                data=all_diagrams_text,
                file_name="patent_diagrams_ipo.txt",
                mime="text/plain"
            )

# ------------------ PATENT IMAGE GENERATION ----------------------
st.markdown("## 🖼️ Patent Drawing Images (AI Generated)")
st.info("🎨 Generate actual patent drawing images using the same Qwen/OpenRouter API as other sections.")

col_img1, col_img2 = st.columns(2)

with col_img1:
    if st.button("🖼️ Generate Image Prompts"):
        if not abstract:
            st.warning("Please enter the invention abstract first.")
        else:
            with st.spinner("Analyzing invention components..."):
                try:
                    components = extract_components_for_images(abstract)
                    st.session_state.image_components = components
                    
                    img_generator = PatentImageGenerator()
                    prompts = img_generator.get_image_prompts_for_api(abstract, components)
                    st.session_state.image_prompts = prompts
                    
                    st.success("✅ Image prompts ready! Copy them to use with image generators.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

with col_img2:
    st.success("✅ **Perfect Diagrams** - Clean black/white with reference numerals")
    if st.button("📐 Generate Perfect Diagrams (Programmatic)"):
        if not abstract:
            st.warning("Please enter the invention abstract first.")
        else:
            with st.spinner("Generating perfect patent diagrams..."):
                try:
                    from programmatic_diagram_generator import generate_patent_diagrams
                    
                    diagram_paths = generate_patent_diagrams(abstract)
                    st.session_state.programmatic_diagrams = diagram_paths
                    
                    st.success("✅ Perfect diagrams generated!")
                    
                    # Display generated diagrams
                    for fig_name, fig_path in diagram_paths.items():
                        with st.expander(f"📊 {fig_name.replace('_', ' ').title()}", expanded=True):
                            st.image(fig_path, caption=fig_name.replace('_', ' ').title())
                            with open(fig_path, "rb") as f:
                                st.download_button(
                                    f"📥 Download {fig_name}.png",
                                    f.read(),
                                    file_name=f"{fig_name}.png",
                                    mime="image/png"
                                )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Display programmatic diagrams if available
if st.session_state.get("programmatic_diagrams"):
    st.markdown("### 📊 Generated Patent Diagrams")
    st.info("✅ These diagrams are IPO-compliant with proper reference numerals")
    
    for fig_name, fig_path in st.session_state.programmatic_diagrams.items():
        with st.expander(f"📐 {fig_name.replace('_', ' ').title()}", expanded=True):
            try:
                st.image(fig_path, caption=fig_name.replace('_', ' ').title())
                with open(fig_path, "rb") as f:
                    st.download_button(
                        f"📥 Download {fig_name}.png",
                        f.read(),
                        file_name=f"{fig_name}.png",
                        mime="image/png",
                        key=f"dl_{fig_name}"
                    )
            except Exception as e:
                st.error(f"Could not display: {e}")


st.markdown("## 📚 IPC Classifier (Indian Patents)")
if st.button("🏷️ Classify IPC"):
    with st.spinner("Classifying IPC..."):
        try:
            result = classify_ipc(abstract)
            st.session_state.ipc_result = result or "⚠️ No result."
            st.success("Done!")
        except Exception as e:
            st.session_state.ipc_result = f"❌ Exception: {e}"
            st.error(f"❌ IPC classification failed: {e}")

if st.session_state.get("ipc_result"):
    with st.expander("🔍 IPC Classification"):
        st.code(st.session_state.ipc_result)

st.markdown("---")
st.markdown("## 🔍 Patent Quality Verification")
st.info("🤖 6 AI Agents will analyze your patent for quality and compliance")

if st.button("✅ Run 5-Agent Verification"):
    # Check if required sections exist
    required = ['title', 'claims', 'abstract_input', 'background', 'summary']
    missing = [s for s in required if not st.session_state.get(s)]
    
    if missing:
        st.warning(f"⚠️ Please generate these sections first: {', '.join(missing)}")
    else:
        with st.spinner("🤖 5 AI Agents verifying your patent... This may take 1-2 minutes"):
            try:
                from patent_verifier import verify_patent_5_sections
                
                # Prepare 5 critical sections for verification
                sections_to_verify = {
                    'title': st.session_state.get('title', ''),
                    'abstract': st.session_state.get('abstract_input', ''),
                    'claims': st.session_state.get('claims', ''),
                    'background': st.session_state.get('background', ''),
                    'summary': st.session_state.get('summary', '')
                }
                
                # Run verification (this is where the real work happens)
                report = verify_patent_5_sections(sections_to_verify)
                
                # Display results
                st.success("✅ Verification Complete!")
                with st.expander("📊 Verification Report", expanded=True):
                    st.text(report)  # Display formatted report
                
                # Store report in session state
                st.session_state.verification_report = report
                
            except ImportError:
                st.error("❌ Patent verifier module not found")
                st.info("💡 Make sure patent_verifier.py is in the same directory")
            except Exception as e:
                st.error(f"❌ Verification failed: {str(e)}")
                st.info("💡 Check that Ollama is running: ollama serve")
                st.info("💡 Ensure Llama 3.1 8B is installed: ollama pull llama3.1:8b")

# Show previous report if exists
if st.session_state.get('verification_report'):
    with st.expander("📋 View Previous Verification Report"):
        st.text(st.session_state.verification_report)

st.markdown("---")
# ----------------------- EXPORT -------------------------
# ----------------------- EXPORT -------------------------
st.markdown("---")
st.markdown("## 📄 Export Patent Document")

# ===================== SECTION COMPLETENESS CHECKLIST =====================
st.markdown("### ✅ Section Completeness Checklist")
st.info("All sections must be generated before export for IPO-compliant filing.")

# Gather all sections for validation
validation_sections = {
    "title": st.session_state.get("title", ""),
    "abstract": st.session_state.get("abstract_input", ""),
    "claims": st.session_state.get("claims", ""),
    "summary": st.session_state.get("summary", ""),
    "field_of_invention": st.session_state.get("field_of_invention", ""),
    "background": st.session_state.get("background", ""),
    "objects_of_invention": st.session_state.get("objects_of_invention", ""),
    "detailed_description": st.session_state.get("detailed_description", ""),
    "brief_description": st.session_state.get("brief_description", ""),
}

# Get checklist
checklist = get_section_checklist(validation_sections)
completed_count, total_count = PatentValidator.get_completeness_score(validation_sections)
completion_percent = int((completed_count / total_count) * 100) if total_count > 0 else 0

# Display progress bar
st.progress(completion_percent / 100, text=f"Completion: {completed_count}/{total_count} sections ({completion_percent}%)")

# Display checklist
with st.expander("📋 View Section Status", expanded=False):
    for section_name, is_complete, word_info in checklist:
        if is_complete:
            st.markdown(f"✅ **{section_name}** - {word_info}")
        else:
            st.markdown(f"❌ **{section_name}** - {word_info}")

# Check if ready for export
can_export, issues = PatentValidator.can_export(validation_sections)

if not can_export:
    st.warning(f"⚠️ **{len(issues)} issues** must be resolved before export:")
    for issue in issues[:5]:  # Show first 5 issues
        st.markdown(f"- {issue}")
else:
    st.success("✅ **All sections complete!** Ready for IPO-compliant export.")
# ===================== END CHECKLIST =====================

export_abstract = st.session_state.get("abstract_input", "")

# ✅ CORRECT ORDER: Indian Patent Office Standard Structure
pdf_sections = {
    "Abstract": export_abstract or "[Not Provided]",  # 1. Abstract FIRST
    "Title": st.session_state.get("title", "[Not Generated]"),  # 2. Title
    "Field of the Invention": st.session_state.get("field_of_invention", "[Not Generated]"),  # 3. Field
    "Background of the Invention": st.session_state.get("background", "[Not Generated]"),  # 4. Background
    "Objects of the Invention": st.session_state.get("objects_of_invention", "[Not Generated]"),  # 5. Objects
    "Summary of the Invention": st.session_state.get("summary", "[Not Generated]"),  # 6. Summary ✅ ADDED
    "Brief Description of the Drawings": st.session_state.get("brief_description", "[Not Generated]"),  # 7. Brief Desc
    "Detailed Description of the Invention": st.session_state.get("detailed_description", "[Not Generated]"),  # 8. Detailed
    "Industrial Applicability": st.session_state.get("industrial_applicability"),
    "Claims": st.session_state.get("claims", "[Not Generated]"),  # 9. Claims LAST
}

st.session_state.generated_sections = pdf_sections

# Show preview with section order
with st.expander("📋 Preview Export Sections (Indian Patent Office Order)"):
    section_list = [
        ("1️⃣", "Abstract"),
        ("2️⃣", "Title"),
        ("3️⃣", "Field of the Invention"),
        ("4️⃣", "Background of the Invention"),
        ("5️⃣", "Objects of the Invention"),
        ("6️⃣", "Summary of the Invention"),
        ("7️⃣", "Brief Description of the Drawings"),
        ("8️⃣", "Detailed Description of the Invention"),
        ("9️⃣", "Industrial Applicability"),
        ("🔟", "Claims")
    ]
    
    for emoji, section_name in section_list:
        content = pdf_sections.get(section_name, "[Not Generated]")
        status = "✅" if content and content != "[Not Generated]" and content != "[Not Provided]" else "❌"
        word_count = len(content.split()) if content and content not in ["[Not Generated]", "[Not Provided]"] else 0
        st.write(f"{emoji} {status} **{section_name}**: {word_count} words")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧾 Generate PDF (with Diagrams)")
    if export_abstract.strip():
        if st.button("📄 Generate PDF"):
            with st.spinner("Creating IPO-compliant PDF with diagrams..."):
                try:
                    from export_to_pdf import create_patent_pdf, get_diagram_paths
                    
                    # Get diagram paths
                    diagram_paths = get_diagram_paths()
                    
                    # Create PDF with diagrams
                    pdf_path = create_patent_pdf(pdf_sections, diagram_paths=diagram_paths)
                    st.success(f"✅ PDF Generated with {len(diagram_paths)} diagrams!")
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Patent PDF",
                            data=f,
                            file_name="patent_document_ipo.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter an abstract before generating the PDF.")

with col2:
    st.markdown("### 📝 Export as DOCX (with Diagrams)")
    if export_abstract.strip():
        if st.button("📝 Generate Indian Patent Office DOCX"):
            with st.spinner("Creating IPO-compliant DOCX with diagrams..."):
                try:
                    from export_to_pdf import create_patent_docx, get_diagram_paths
                    from io import BytesIO
                    
                    # Get diagram paths
                    diagram_paths = get_diagram_paths()
                    
                    # Get applicant info
                    applicant_details = st.session_state.get("applicant_details", {})
                    inventor_details = st.session_state.get("inventor_details", {})
                    
                    # Create DOCX with diagrams
                    docx_path = "patent_application_ipo.docx"
                    create_patent_docx(
                        pdf_sections,
                        docx_path,
                        applicant_name=applicant_details.get("name", ""),
                        inventor_name=inventor_details.get("name", ""),
                        diagram_paths=diagram_paths
                    )
                    
                    with open(docx_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Indian Patent Office DOCX",
                            data=f.read(),
                            file_name="patent_application_ipo.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    
                    st.success(f"✅ IPO DOCX generated with {len(diagram_paths)} diagrams!")
                    st.info("📋 Includes: Title Page → Abstract → 7 Sections → Claims → Drawing Sheets")
                    
                except Exception as e:
                    st.error(f"❌ DOCX generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter an abstract before generating the DOCX.")

# ===================== IPO FORMS GENERATION =====================
st.markdown("---")
st.markdown("### 📋 IPO Filing Forms (Form 1, 3, 5)")
st.info("💡 These forms are required for complete IPO patent filing. Fill applicant details above first.")

# Check if applicant details are available
applicant_details = st.session_state.get("applicant_details", {})
inventor_details = st.session_state.get("inventor_details", {})
title = st.session_state.get("title", "")

if applicant_details.get("name") and title:
    if st.button("📋 Generate IPO Filing Forms (Form 1, 3, 5)"):
        with st.spinner("Generating IPO forms..."):
            try:
                from ipo_forms_generator import generate_ipo_forms
                
                forms = generate_ipo_forms(
                    title=title,
                    applicant_name=applicant_details.get("name", ""),
                    applicant_address=applicant_details.get("address", ""),
                    applicant_nationality=applicant_details.get("nationality", "INDIAN"),
                    inventor_name=inventor_details.get("name", ""),
                    inventor_address=inventor_details.get("address", ""),
                    inventor_nationality=inventor_details.get("nationality", "INDIAN"),
                    applicant_category=applicant_details.get("category", "NATURAL PERSON")
                )
                
                st.success("✅ IPO Forms Generated!")
                
                # Form 1
                with st.expander("📄 Form 1 - Application for Grant of Patent", expanded=True):
                    st.text_area("Form 1 Content", forms["form_1"], height=400, key="form1_display")
                    st.download_button(
                        "📥 Download Form 1",
                        forms["form_1"],
                        file_name="IPO_Form_1_Application.txt",
                        mime="text/plain"
                    )
                
                # Form 3
                with st.expander("📄 Form 3 - Statement and Undertaking"):
                    st.text_area("Form 3 Content", forms["form_3"], height=300, key="form3_display")
                    st.download_button(
                        "📥 Download Form 3",
                        forms["form_3"],
                        file_name="IPO_Form_3_Statement.txt",
                        mime="text/plain"
                    )
                
                # Form 5
                with st.expander("📄 Form 5 - Declaration of Inventorship"):
                    st.text_area("Form 5 Content", forms["form_5"], height=300, key="form5_display")
                    st.download_button(
                        "📥 Download Form 5",
                        forms["form_5"],
                        file_name="IPO_Form_5_Declaration.txt",
                        mime="text/plain"
                    )
                    
            except Exception as e:
                st.error(f"❌ Form generation failed: {e}")
else:
    missing = []
    if not applicant_details.get("name"):
        missing.append("Applicant Name")
    if not title:
        missing.append("Title (generate patent sections first)")
    st.warning(f"⚠️ Missing required info: {', '.join(missing)}")

st.markdown("---")

if st.button("🔄 Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

