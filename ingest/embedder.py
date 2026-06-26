import os
import math
import time
from typing import List, Optional

from google import genai
from google.genai import types

from database.mongo_connection import get_db_connection


DEFAULT_MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_OUTPUT_DIMENSIONALITY = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "384"))
DEFAULT_BATCH_SIZE = int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", "32"))
DEFAULT_REQUEST_DELAY_SECONDS = float(os.getenv("GEMINI_EMBEDDING_REQUEST_DELAY_SECONDS", "0"))
NORMALIZE_TRUNCATED_GEMINI_001 = os.getenv("GEMINI_NORMALIZE_TRUNCATED_001", "true").lower() != "false"
DB_NAME = "rams_db"
COLLECTION_NAME = "chunks"

_client_cache: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """Return a cached Gemini client configured from GEMINI_API_KEY."""
    global _client_cache

    if _client_cache is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        _client_cache = genai.Client(api_key=api_key)

    return _client_cache


def _embedding_values(embedding) -> List[float]:
    values = getattr(embedding, "values", None)
    if values is None and isinstance(embedding, dict):
        values = embedding.get("values")
    if values is None:
        raise ValueError("Gemini embedding response did not include values.")
    return [float(value) for value in values]


def _normalize(vector: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
) -> List[List[float]]:
    """
    Generate Gemini embedding vectors for a list of text strings.

    The default output dimensionality is 384 to match the previous
    all-MiniLM-L6-v2 vectors and avoid requiring a MongoDB vector index rebuild.
    Set GEMINI_EMBEDDING_DIMENSIONS if your Atlas vector index uses another size.
    """
    if not texts:
        return []

    client = get_client()
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        
        max_retries = 6
        retry_delay = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=output_dimensionality,
                    ),
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    print(f"Gemini Embedding API rate limited (429). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
        
        if response is None:
            raise RuntimeError("Failed to generate embeddings after maximum retries due to rate limits.")
            
        batch_vectors = [_embedding_values(embedding) for embedding in response.embeddings]
        if (
            NORMALIZE_TRUNCATED_GEMINI_001
            and model_name == "gemini-embedding-001"
            and output_dimensionality != 3072
        ):
            batch_vectors = [_normalize(vector) for vector in batch_vectors]
        vectors.extend(batch_vectors)

        if show_progress:
            print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} text(s).")

        if DEFAULT_REQUEST_DELAY_SECONDS and start + batch_size < len(texts):
            time.sleep(DEFAULT_REQUEST_DELAY_SECONDS)

    return vectors


def generate_query_embedding(
    query: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
) -> List[float]:
    """Generate a Gemini embedding optimized for retrieval queries."""
    return generate_embeddings(
        [query],
        model_name=model_name,
        batch_size=1,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=output_dimensionality,
    )[0]


def _get_collection():
    """Return the MongoDB chunks collection."""
    client = get_db_connection()
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def embed_and_update_chunks(
    doc_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """
    Fetch all chunks for doc_id from MongoDB, generate embeddings, and update them.

    Returns the number of chunks updated.
    """
    collection = _get_collection()
    chunks = list(collection.find({"doc_id": doc_id}).sort("chunk_index", 1))

    if not chunks:
        print(f"No chunks found for doc '{doc_id}'.")
        return 0

    texts = [chunk["text_content"] for chunk in chunks]
    print(f"Generating Gemini embeddings for {len(texts)} chunk(s) of doc '{doc_id}'...")
    embeddings = generate_embeddings(texts, model_name=model_name, batch_size=batch_size)

    updated = 0
    for chunk_doc, embedding in zip(chunks, embeddings):
        collection.update_one(
            {"_id": chunk_doc["_id"]},
            {"$set": {"embedding": embedding}},
        )
        updated += 1

    print(f"Updated {updated} chunk(s) with Gemini embeddings for doc '{doc_id}'.")
    return updated


def embed_and_update_all(
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> int:
    """
    Generate + store embeddings for chunks.

    By default this only targets chunks missing an embedding. Set force=True to
    overwrite every stored embedding, which is useful when changing providers.

    Returns the total number of chunks updated.
    """
    collection = _get_collection()
    query = {} if force else {"$or": [{"embedding": None}, {"embedding": {"$exists": False}}]}
    chunks = list(
        collection.find(query).sort([("doc_id", 1), ("chunk_index", 1)])
    )

    if not chunks:
        print("No chunks found to embed." if force else "All chunks already have embeddings.")
        return 0

    texts = [chunk["text_content"] for chunk in chunks]
    action = "Re-embedding" if force else "Back-filling"
    print(f"{action} Gemini embeddings for {len(texts)} chunk(s)...")
    embeddings = generate_embeddings(texts, model_name=model_name, batch_size=batch_size)

    updated = 0
    for chunk_doc, embedding in zip(chunks, embeddings):
        collection.update_one(
            {"_id": chunk_doc["_id"]},
            {"$set": {"embedding": embedding}},
        )
        updated += 1

    print(f"{action} complete for {updated} chunk(s).")
    return updated


if __name__ == "__main__":
    sample_texts = [
        "RAMS is a Retrieval-Augmented Management System.",
        "It fetches documents and generates embeddings.",
        "Relevant context is used to answer user questions.",
    ]
    vecs = generate_embeddings(sample_texts, show_progress=True)
    for i, vector in enumerate(vecs):
        print(f"Text {i}: dim={len(vector)}, first 5 values={vector[:5]}")
