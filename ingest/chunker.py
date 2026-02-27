import re
import os
import hashlib
import importlib.util
from typing import List, Optional
from datetime import datetime

from database.mongo_connection import get_db_connection

# chunk_model.py has a dot in the filename → load by file path
_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "chunk_model.py")
_spec = importlib.util.spec_from_file_location("chunk_model", _model_path)
_chunk_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunk_mod)
ChunkModel = _chunk_mod.ChunkModel


# ── Chunking Configuration ──────────────────────────────────────────
CHUNK_SIZE = 500        # target characters per chunk
CHUNK_OVERLAP = 50      # overlap between consecutive chunks
DB_NAME = "rams_db"
COLLECTION_NAME = "chunks"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into chunks of roughly *chunk_size* characters.

    Strategy:
      1. Split on sentence boundaries so chunks don't cut mid‑sentence.
      2. Accumulate sentences until the next one would exceed *chunk_size*.
      3. Consecutive chunks share *overlap* characters for context continuity.

    Returns a list of chunk strings (empty list if input is blank).
    """
    if not text or not text.strip():
        return []

    # Split on sentence‑ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence still fits, accumulate
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            # Save current chunk if non‑empty
            if current_chunk:
                chunks.append(current_chunk)

            # Start next chunk with overlap from the tail of the previous one
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = f"{overlap_text} {sentence}".strip()
            else:
                current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ── MongoDB persistence ─────────────────────────────────────────────

def _get_collection():
    """Return the MongoDB chunks collection."""
    client = get_db_connection()
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def save_chunks_to_mongo(
    doc_id: str,
    chunks: List[str],
    metadata: Optional[dict] = None,
) -> List[str]:
    """
    Persist a list of text chunks to MongoDB using the ChunkModel schema.

    Steps:
      1. Delete any existing chunks for *doc_id* (full replacement strategy).
      2. Build ChunkModel documents for each chunk.
      3. Bulk‑insert into the ``chunks`` collection.

    Returns the list of inserted MongoDB ``_id`` strings.
    """
    collection = _get_collection()

    # Remove old chunks for this document so we always have a fresh set
    delete_result = collection.delete_many({"doc_id": doc_id})
    print(f"Deleted {delete_result.deleted_count} old chunk(s) for doc '{doc_id}'.")

    if not chunks:
        print("No chunks to save.")
        return []

    # Build documents via pydantic model
    documents = []
    for idx, text_content in enumerate(chunks):
        chunk = ChunkModel(
            doc_id=doc_id,
            chunk_index=idx,
            text_content=text_content,
            metadata={
                **(metadata or {}),
                "content_hash": hashlib.md5(text_content.encode()).hexdigest(),
            },
            created_at=datetime.utcnow(),
        )
        documents.append(chunk.model_dump(by_alias=True, exclude_none=True))

    result = collection.insert_many(documents)
    inserted_ids = [str(oid) for oid in result.inserted_ids]

    print(f"Saved {len(inserted_ids)} chunk(s) for doc '{doc_id}' to MongoDB.")
    return inserted_ids


def get_chunks_from_mongo(doc_id: str) -> List[ChunkModel]:
    """Retrieve all chunks for a given *doc_id*, ordered by chunk_index."""
    collection = _get_collection()
    cursor = collection.find({"doc_id": doc_id}).sort("chunk_index", 1)
    return [
        ChunkModel(**{**doc, "_id": str(doc["_id"])})
        for doc in cursor
    ]


# ── Convenience entry‑point ─────────────────────────────────────────

def chunk_and_store(
    doc_id: str,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    metadata: Optional[dict] = None,
) -> List[str]:
    """
    End‑to‑end helper: chunk the text **and** persist to MongoDB.

    Returns the list of inserted ``_id`` strings.
    """
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Text chunked into {len(chunks)} piece(s) (size={chunk_size}, overlap={overlap}).")
    return save_chunks_to_mongo(doc_id, chunks, metadata)


# ── Quick local test ─────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "RAMS is a Retrieval-Augmented Management System. "
        "It fetches documents from external sources like Google Docs. "
        "The text is then extracted and cleaned. "
        "Next the text is split into chunks for embedding. "
        "Each chunk is stored in MongoDB with metadata. "
        "Embeddings are generated and stored in a FAISS index. "
        "When a user asks a question, relevant chunks are retrieved. "
        "The retrieved context is sent to an LLM for a grounded answer."
    )

    ids = chunk_and_store(doc_id="test_doc", text=sample, chunk_size=200, overlap=30)
    print(f"\nInserted IDs: {ids}")

    # Verify round‑trip
    stored = get_chunks_from_mongo("test_doc")
    for c in stored:
        print(f"  [{c.chunk_index}] {c.text_content[:80]}...")
