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
CHUNK_SIZE = 1000       # target characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
DB_NAME = "rams_db"
COLLECTION_NAME = "chunks"


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    doc_id: str = "",
    metadata: Optional[dict] = None,
) -> List[str]:
    """
    Split *text* into chunks of roughly *chunk_size* characters with header-aware prefixing.

    Strategy:
      1. Parse Markdown headings (#, ##, ###, etc.) line by line to maintain a hierarchical header context.
      2. Group lines into sections under their current header path (e.g., [Doc > Header > Subheader]).
      3. For each section, split text on sentence boundaries so chunks don't cut mid‑sentence.
      4. Accumulate sentences until the next one would exceed *chunk_size* (accounting for prefix length).
      5. Prefix every generated chunk with `[Doc > Header] ` so standalone vector search retains exact context hierarchy.
    """
    if not text or not text.strip():
        return []

    # Determine base document title
    doc_title = (metadata or {}).get("title") or (metadata or {}).get("source_title") or doc_id or "Document"

    # 1. Parse lines and segment by active headings
    lines = text.strip().split("\n")
    sections: List[tuple] = []
    current_headers: List[str] = []
    current_section_lines: List[str] = []

    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if header_match:
            if current_section_lines and any(l.strip() for l in current_section_lines):
                sections.append((list(current_headers), current_section_lines))
                current_section_lines = []

            level = len(header_match.group(1))
            heading_text = header_match.group(2).strip()

            if level == 1 and doc_title in ("Document", "test_doc", "") and not current_headers:
                doc_title = heading_text
                current_headers = [heading_text]
            else:
                target_depth = max(0, level - 1)
                current_headers = current_headers[:target_depth]
                current_headers.append(heading_text)
        else:
            current_section_lines.append(line)

    if current_section_lines and any(l.strip() for l in current_section_lines):
        sections.append((list(current_headers), current_section_lines))

    if not sections and text.strip():
        sections = [([], [text.strip()])]

    chunks: List[str] = []

    # 2. Chunk each section separately with its header prefix
    for active_headers, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue

        path_parts = [doc_title] + [h for h in active_headers if h != doc_title]
        header_path = " > ".join(path_parts)
        prefix = f"[{header_path}] " if header_path else ""

        effective_chunk_size = max(200, chunk_size - len(prefix))

        sentences = re.split(r'(?<=[.!?])\s+', section_text)

        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) + 1 <= effective_chunk_size:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(f"{prefix}{current_chunk}")

                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = f"{overlap_text} {sentence}".strip()
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(f"{prefix}{current_chunk}")

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
    chunks = chunk_text(text, chunk_size, overlap, doc_id=doc_id, metadata=metadata)
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

