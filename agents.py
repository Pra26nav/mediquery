"""
agents.py — MediQuery AI Agents
================================
Three specialized medical AI agents:
- extractor: Pulls medical entities (symptoms, drugs, dosages)
- analyzer: Cross-references entities with document context
- summarizer: Generates patient-friendly summary
"""
from crewai import Agent
from dotenv import load_dotenv

load_dotenv()

MODEL = "groq/llama-3.3-70b-versatile"

extractor = Agent(
    role="Medical Entity Extractor",
    goal="Extract all medical entities from document context including symptoms, diagnoses, medications, dosages, and procedures",
    backstory="Expert medical data analyst trained to identify and categorize clinical information from medical documents with high precision.",
    llm=MODEL,
    verbose=True
)

analyzer = Agent(
    role="Medical Context Analyzer",
    goal="Analyze extracted medical entities and cross-reference them with document context to provide accurate clinical insights",
    backstory="Experienced clinical analyst who interprets medical data, identifies relationships between conditions and treatments, and flags important warnings.",
    llm=MODEL,
    verbose=True
)

summarizer = Agent(
    role="Patient-Friendly Summarizer",
    goal="Convert complex medical analysis into clear, simple language that patients and non-medical users can understand",
    backstory="Medical communication specialist who translates clinical jargon into plain English while maintaining accuracy and completeness.",
    llm=MODEL,
    verbose=True
)