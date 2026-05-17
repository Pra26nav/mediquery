# 🏥 MediQuery — Medical Document Intelligence

AI-powered medical document analysis using multi-agent CrewAI pipeline + RAG.

## What it does
Upload any medical PDF (research paper, drug leaflet, patient report) and:
- Extract medical entities (symptoms, drugs, dosages, diagnoses)
- Get clinical analysis grounded in document
- Receive patient-friendly summaries
- Ask follow-up questions in natural language

## Architecture
PDF Upload
|
PyPDF Text Extraction
|
FAISS Vector Index
|
User Question -> FAISS similarity_search (k=4)
|
CrewAI 3-Agent Pipeline:
Agent 1 (Extractor) -> Medical entity extraction
Agent 2 (Analyzer)  -> Clinical cross-referencing
Agent 3 (Summarizer)-> Patient-friendly output
|
Answer + Source Citations

## Tech Stack
- CrewAI — multi-agent orchestration
- LangChain + FAISS — RAG pipeline
- Groq (Llama 3.3 70B) — LLM inference
- HuggingFace Embeddings — local, no API cost
- Streamlit — UI

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add `.env`:
GROQ_API_KEY=your_key_here

Run:
```bash
streamlit run app.py
```

## Use Cases
- Understand medical research papers
- Decode drug information leaflets
- Get plain-English explanations of medical reports
- Quick reference for clinical documents