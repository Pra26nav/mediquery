"""
app.py — MediQuery Medical Document Intelligence
=================================================
Multi-agent RAG system for medical document analysis.
Agents: Extractor -> Analyzer -> Summarizer
Stack: CrewAI + LangChain + FAISS + Groq + Streamlit
"""
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="MediQuery",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {}

# Hero
st.markdown("""
<div class="hero">
    <h1>🏥 MediQuery</h1>
    <p>Medical Document Intelligence — powered by AI agents</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Upload Document")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}")
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("MediQuery is for informational purposes only. Always consult a qualified healthcare professional.")

# Main
if not st.session_state.vector_store:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem;">🏥</div>
        <div style="font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; margin-bottom:0.5rem;">Upload a medical document</div>
        <div style="color:#888; font-size:0.9rem;">Research papers, drug leaflets, patient reports — ask anything about them.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="entity-card"><h4>🔬 Entity Extraction</h4><p>Automatically identifies symptoms, drugs, dosages, and diagnoses.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="entity-card"><h4>🧠 Clinical Analysis</h4><p>3-agent pipeline cross-references findings with document context.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="entity-card"><h4>💊 Plain English</h4><p>Complex medical language converted to patient-friendly summaries.</p></div>', unsafe_allow_html=True)
else:
    st.info("Document loaded. Ask a question below.")