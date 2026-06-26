from typing import List, Dict, Any
from ingest.embedder import get_client

def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Use Gemini Flash to answer the query based ONLY on the provided context chunks.
    """
    client = get_client()
    
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
        # We use an available flash model.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        return "I encountered an error while trying to generate an answer."
