import os
import importlib.util
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from database.mongo_connection import get_db_connection

# ── Load ChunkModel by file path (filename has no dots now but kept
#    consistent with chunker.py's import style) ──────────────────────
_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "chunk_model.py")
_spec = importlib.util.spec_from_file_location("chunk_model", _model_path)
_chunk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunk_mod)
ChunkModel = _chunk_mod.ChunkModel


# ── Configuration ───────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"   # lightweight, 384-dim embeddings
DB_NAME = "rams_db"
COLLECTION_NAME = "chunks"

# Module-level cache so the model is loaded only once per process
_model_cache: dict = {}


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Return a cached SentenceTransformer model.
    Loads the model on first call; subsequent calls reuse the instance.
    """
    if model_name not in _model_cache:
        print(f"Loading embedding model '{model_name}'...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print("Model loaded.")
    return _model_cache[model_name]


# ── Reusable core function ──────────────────────────────────────────

def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[List[float]]:
    """
    Generate vector embeddings for a list of text strings.

    This is the **reusable** function you can import anywhere:
        from ingest.embedder import generate_embeddings
        vectors = generate_embeddings(["hello world", "another sentence"])

    Args:
        texts:          List of strings to embed.
        model_name:     HuggingFace model identifier (default: all-MiniLM-L6-v2).
        batch_size:     Batch size for encoding.
        show_progress:  Show a progress bar during encoding.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    # Convert numpy arrays → plain Python lists for JSON / MongoDB compatibility
    return [vec.tolist() for vec in embeddings]


# ── MongoDB helpers ─────────────────────────────────────────────────

def _get_collection():
    """Return the MongoDB chunks collection."""
    client = get_db_connection()
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def embed_and_update_chunks(
    doc_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> int:
    """
    Fetch all chunks for *doc_id* from MongoDB, generate embeddings,
    and update each chunk document with its embedding vector.

    Returns the number of chunks updated.
    """
    collection = _get_collection()

    # 1. Fetch chunks ordered by chunk_index
    cursor = collection.find({"doc_id": doc_id}).sort("chunk_index", 1)
    chunks = list(cursor)

    if not chunks:
        print(f"No chunks found for doc '{doc_id}'.")
        return 0

    texts = [c["text_content"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunk(s) of doc '{doc_id}'...")

    # 2. Generate embeddings in one batch
    embeddings = generate_embeddings(texts, model_name=model_name, batch_size=batch_size)

    # 3. Update each chunk document with its embedding
    updated = 0
    for chunk_doc, embedding in zip(chunks, embeddings):
        collection.update_one(
            {"_id": chunk_doc["_id"]},
            {"$set": {"embedding": embedding}},
        )
        updated += 1

    print(f"Updated {updated} chunk(s) with embeddings for doc '{doc_id}'.")
    return updated


def embed_and_update_all(
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
) -> int:
    """
    Find every chunk that has no embedding yet and generate + store one.
    Useful as a bulk back-fill operation.

    Returns the total number of chunks updated.
    """
    collection = _get_collection()

    # Only target chunks missing an embedding
    cursor = collection.find(
        {"$or": [{"embedding": None}, {"embedding": {"$exists": False}}]}
    ).sort("doc_id", 1)

    chunks = list(cursor)
    if not chunks:
        print("All chunks already have embeddings.")
        return 0

    texts = [c["text_content"] for c in chunks]
    print(f"Back-filling embeddings for {len(texts)} chunk(s)...")

    embeddings = generate_embeddings(texts, model_name=model_name, batch_size=batch_size)

    updated = 0
    for chunk_doc, embedding in zip(chunks, embeddings):
        collection.update_one(
            {"_id": chunk_doc["_id"]},
            {"$set": {"embedding": embedding}},
        )
        updated += 1

    print(f"Back-filled {updated} chunk(s) with embeddings.")
    return updated


# ── Quick local test ─────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Test the reusable function standalone ──
    sample_texts = [
        "RAMS is a Retrieval-Augmented Management System.",
        "It fetches documents and generates embeddings.",
        "Relevant context is used to answer user questions.",
    ]
    vecs = generate_embeddings(sample_texts)
    for i, v in enumerate(vecs):
        print(f"  Text {i}: dim={len(v)}, first 5 values={v[:5]}")

    # ── Test the MongoDB flow (requires chunks in DB) ──
    # Uncomment the line below if chunks for 'test_doc' already exist:
    # count = embed_and_update_chunks("test_doc")
    # print(f"Updated {count} chunks in MongoDB.")
