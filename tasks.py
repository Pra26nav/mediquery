"""
tasks.py — MediQuery CrewAI Tasks
===================================
Defines sequential tasks for medical document analysis:
1. extract_task: Entity extraction from document
2. analyze_task: Clinical analysis and cross-referencing
3. summarize_task: Patient-friendly summary generation
"""
from crewai import Task
from agents import extractor, analyzer, summarizer


def create_medical_tasks(document_context: str, user_question: str):
    extract_task = Task(
        description=f"""Analyze this medical document context and extract ALL medical entities.

DOCUMENT CONTEXT:
{document_context}

USER QUESTION: {user_question}

Extract and list:
1. Symptoms/Conditions mentioned
2. Medications and dosages
3. Medical procedures
4. Lab values or test results
5. Diagnoses
6. Warnings or contraindications

Format as structured bullet points.""",
        expected_output="Structured list of all medical entities found in the document.",
        agent=extractor
    )

    analyze_task = Task(
        description=f"""Using the extracted medical entities, analyze the clinical context.

USER QUESTION: {user_question}

Provide:
1. Direct answer to the user question based on document
2. Clinical significance of key findings
3. Relationships between conditions and treatments
4. Any important warnings or drug interactions
5. What is NOT mentioned (gaps in information)

Be precise and cite specific parts of the document.""",
        expected_output="Detailed clinical analysis answering the user question.",
        agent=analyzer,
        context=[extract_task]
    )

    summarize_task = Task(
        description=f"""Convert the clinical analysis into a clear, patient-friendly response.

USER QUESTION: {user_question}

Requirements:
- Use simple, everyday language (no jargon)
- Keep it concise but complete
- Highlight the most important points
- Add a disclaimer: "This is based on the uploaded document only. Always consult a qualified healthcare professional."
- Format with clear sections""",
        expected_output="Patient-friendly summary answering the user question clearly.",
        agent=summarizer,
        context=[analyze_task]
    )

    return [extract_task, analyze_task, summarize_task]