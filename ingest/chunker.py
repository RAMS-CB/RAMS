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


from database.sanity_client import query_sanity, mutate_sanity

# ── Sanity persistence ─────────────────────────────────────────────

def save_chunks_to_mongo(
    doc_id: str,
    chunks: List[str],
    metadata: Optional[dict] = None,
) -> List[str]:
    """
    Persist a list of text chunks to Sanity using the ChunkModel schema.

    Steps:
      1. Delete any existing chunks for *doc_id* (full replacement strategy).
      2. Build chunk documents for each chunk.
      3. Bulk-insert into Sanity (_type: "chunk").

    Returns the list of inserted Sanity ``_id`` strings.
    """
    # 1. Delete old chunks for this doc_id
    existing_ids = query_sanity('*[_type == "chunk" && doc_id == $doc_id]._id', {"doc_id": doc_id}) or []
    mutations = [{"delete": {"id": cid}} for cid in existing_ids]
    
    if not chunks:
        if mutations:
            mutate_sanity(mutations)
            print(f"Deleted {len(mutations)} old chunk(s) for doc '{doc_id}'.")
        print("No new chunks to save.")
        return []

    inserted_ids = []
    # 2. Build new chunk documents
    for idx, text_content in enumerate(chunks):
        chunk_id = f"chunk_{hashlib.md5(f'{doc_id}_{idx}'.encode()).hexdigest()}"
        chunk_doc = {
            "_id": chunk_id,
            "_type": "chunk",
            "doc_id": doc_id,
            "chunk_index": idx,
            "text_content": text_content,
            "metadata": {
                **(metadata or {}),
                "content_hash": hashlib.md5(text_content.encode()).hexdigest(),
            },
            "created_at": datetime.utcnow().isoformat(),
        }
        mutations.append({"createOrReplace": chunk_doc})
        inserted_ids.append(chunk_id)

    mutate_sanity(mutations)
    print(f"Saved {len(inserted_ids)} chunk(s) for doc '{doc_id}' to Sanity.")
    return inserted_ids


def get_chunks_from_mongo(doc_id: str) -> List[ChunkModel]:
    """Retrieve all chunks for a given *doc_id*, ordered by chunk_index."""
    query = '*[_type == "chunk" && doc_id == $doc_id] | order(chunk_index asc)'
    raw_chunks = query_sanity(query, {"doc_id": doc_id}) or []
    return [
        ChunkModel(**{**doc, "_id": str(doc["_id"])})
        for doc in raw_chunks
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
    End‑to‑end helper: chunk the text **and** persist to Sanity.

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
        "Each chunk is stored in Sanity with metadata. "
        "Embeddings are generated and stored with chunks. "
        "When a user asks a question, relevant chunks are retrieved. "
        "The retrieved context is sent to an LLM for a grounded answer."
    )

    ids = chunk_and_store(doc_id="test_doc", text=sample, chunk_size=200, overlap=30)
    print(f"\nInserted IDs: {ids}")

    # Verify round‑trip
    stored = get_chunks_from_mongo("test_doc")
    for c in stored:
        print(f"  [{c.chunk_index}] {c.text_content[:80]}...")

