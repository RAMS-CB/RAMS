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
    
    prompt = f"""You are an assistant.

Use ONLY the information provided below.

If the answer is not present, say
"I couldn't find that information."

Context:
{context_str}

Question:
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
    client = get_client()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
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
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    response.raise_for_status()
    data = response.json()
    
    return data["choices"][0]["message"]["content"]
