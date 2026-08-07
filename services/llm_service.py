import os
import requests
from typing import List, Dict, Any
from ingest.embedder import get_client

def generate_answer(query: str, context_chunks: List[Dict[str, Any]], model_provider: str = "gemini") -> str:
    """
    Use the selected LLM to answer the query based ONLY on the provided context chunks.
    """
    # Extract just the text from the context chunks
    context_texts = [chunk.get("text_content", "") for chunk in context_chunks]
    context_str = "\n\n".join(context_texts)
    
    prompt = f"""You are RAMS AI, a friendly and knowledgeable assistant for IIT Madras BS Degree students.

Your job is to answer the student's question using the context passages provided below.

## Instructions:

1. **Be thorough**: Extract ALL relevant information from the context — do not leave out details. Students rely on your answer being complete.
2. **Explain acronyms**: When you first mention an acronym (OPPE, GAA, SCT, etc.), write the full form followed by the acronym in parentheses.
3. **Cover all aspects**: Where applicable, explain:
   - **What** it is (definition/description)
   - **How** it works (process/steps)
   - **When** it happens (dates, deadlines, timelines)
   - **Eligibility/conditions** (who is eligible, requirements)
   - **Important rules** (restrictions, limits, penalties)
4. **Structure your answer** using Markdown:
   - Use **headings** (##, ###) to organize multi-part answers
   - Use **bullet points** for lists of rules or steps
   - Use **bold** for key terms and important points
5. **Stay grounded**: only use facts present in the context. Do not invent information.
6. If the context contains conflicting or conditional information, present both sides and explain when each applies.
7. If the context truly contains **nothing relevant** to the question, respond with:
   "I'm sorry, I couldn't find information about that in the available documents. Could you try rephrasing your question?"
8. Keep your tone helpful and student-friendly.

Context:
{context_str}

Student's Question:
{query}"""

    try:
        if model_provider == "aipipe":
            return _generate_with_aipipe(prompt)
        else:
            return _generate_with_gemini(prompt)
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        return "I encountered an error while trying to generate an answer."

def _generate_with_gemini(prompt: str) -> str:
    from google.genai import types
    client = get_client()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            seed=42,
        ),
    )
    return response.text

def _generate_with_aipipe(prompt: str) -> str:
    api_key = os.getenv("AI_PIPE_API_KEY")
    if not api_key:
        raise ValueError("AI_PIPE_API_KEY is not set in the environment.")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 42
    }
    
    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    response.raise_for_status()
    data = response.json()
    
    return data["choices"][0]["message"]["content"]

def generate_faqs(questions: List[str], count: int = 5, model_provider: str = "gemini") -> str:
    """
    Given a list of user questions, use the LLM to generate a JSON array of FAQs.
    """
    questions_str = "\n".join([f"- {q}" for q in questions])
    prompt = f"""You are an expert at analyzing user queries and creating helpful FAQs.
Please analyze the following recent user questions:

{questions_str}

Identify the {count} most common or important themes.
For each theme, formulate a clear, generic Question and a concise Answer.
Return the result strictly as a valid JSON array of objects, where each object has "question" and "answer" string properties.
Example format:
[
  {{"question": "How do I reset my password?", "answer": "Go to settings..."}}
]
Do not include markdown blocks like ```json or any other text outside the JSON array.
"""
    try:
        if model_provider == "aipipe":
            return _generate_with_aipipe(prompt)
        else:
            return _generate_with_gemini(prompt)
    except Exception as e:
        print(f"Error generating FAQs with LLM: {e}")
        return "[]"
