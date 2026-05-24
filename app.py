from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai import Crew, Process
from tasks import create_medical_tasks
import hashlib, pickle

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

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
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
else:
    meta = st.session_state.doc_meta
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-num">1</div><div class="stat-label">Document</div></div>
        <div class="stat-box"><div class="stat-num">{meta['pages']}</div><div class="stat-label">Pages</div></div>
        <div class="stat-box"><div class="stat-num">{meta['chunks']}</div><div class="stat-label">Chunks</div></div>
        <div class="stat-box"><div class="stat-num">{len(st.session_state.chat_history)//2}</div><div class="stat-label">Questions</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}<div class="chat-source">{msg.get("sources","")}</div></div>', unsafe_allow_html=True)

    # Quick questions
    st.markdown("**Quick questions:**")
    quick = ["What are the main findings?", "List all medications mentioned", "What are the side effects?", "Summarize for a patient"]
    cols = st.columns(4)
    for i, q in enumerate(quick):
        with cols[i]:
            if st.button(q, key=f"q_{i}"):
                st.session_state["pending_q"] = q
                st.rerun()

    question = st.text_input("Ask about this document:", placeholder="e.g. What dosage is recommended?")
    if st.session_state.get("pending_q"):
        question = st.session_state.pop("pending_q")

    if st.button("🔬 Analyze", disabled=not question):
        with st.spinner("3 agents analyzing..."):
            docs = st.session_state.vector_store.similarity_search(question, k=4)
            context = "\n\n".join([d.page_content for d in docs])
            sources = list(set([
                line.strip("[]") for d in docs
                for line in d.page_content.split("\n")
                if line.startswith("[PAGE")
            ]))
            sources_str = " | ".join([f"📍 {s}" for s in sources[:3]])

            tasks = create_medical_tasks(context, question)
            crew = Crew(
                agents=[tasks[0].agent, tasks[1].agent, tasks[2].agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=False
            )
            result = crew.kickoff()
            answer = str(result)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources_str})
            st.rerun()
            
            