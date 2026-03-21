import os
import faiss
import json
import random
import numpy as np
from typing import List, Dict, Tuple

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

INDEX_FILE = os.path.join(DB_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(DB_DIR, "faiss_metadata.json")

# Default dimension for "all-MiniLM-L6-v2"
DIMENSION = 384 

def _load_or_create_index() -> Tuple[faiss.Index, dict]:
    """Loads the FAISS index and metadata, or creates them if missing."""
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        # We wrap IndexFlatL2 with IndexIDMap so we can remove vectors later by ID.
        base_index = faiss.IndexFlatL2(DIMENSION)
        index = faiss.IndexIDMap(base_index)
        
    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
            
    return index, metadata

def _save_index(index: faiss.Index, metadata: dict):
    """Persists the FAISS index and the metadata mapping to disk."""
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

def replace_vectors_in_faiss(doc_id: str, embeddings: list):
    """
    Replaces existing FAISS vectors for a document with the new embeddings.
    Since FAISS uses integer IDs, we maintain a registry in a JSON file
    mapping `doc_id` to its corresponding `vector_ids`.
    """
    if not embeddings:
        print(f"No embeddings to insert for doc_id {doc_id}.")
        return

    index, metadata = _load_or_create_index()
    
    # 1. Remove old vectors if they exist
    if doc_id in metadata and len(metadata[doc_id]) > 0:
        old_ids = np.array(metadata[doc_id], dtype=np.int64)
        try:
            index.remove_ids(old_ids)
            print(f"Removed {len(old_ids)} old vectors from FAISS for document '{doc_id}'.")
        except Exception as e:
            print(f"Could not remove old vectors from FAISS ({e}). They might not exist.", flush=True)
            
    # 2. Add new vectors
    emb_array = np.array(embeddings, dtype=np.float32)
    
    if emb_array.shape[1] != DIMENSION:
        print(f"Warning: Expected embedding dimension {DIMENSION}, got {emb_array.shape[1]}. Make sure the index dimension matches the model.")
        
    new_ids = []
    for _ in range(len(embeddings)):
        # Generate a random 63-bit signed integer ID for FAISS tracking
        new_ids.append(random.randint(0, 2**63 - 1))
        
    ids_array = np.array(new_ids, dtype=np.int64)
    
    index.add_with_ids(emb_array, ids_array)
    metadata[doc_id] = new_ids
    
    # 3. Save changes
    _save_index(index, metadata)
    print(f"Inserted {len(new_ids)} new vectors into FAISS for document '{doc_id}'.")


def sync_all_from_mongo():
    """
    Reads all chunks with embeddings from MongoDB, groups them by doc_id, 
    and backfills them into FAISS.
    """
    from database.mongo_connection import get_db_connection
    client = get_db_connection()
    db = client["rams_db"]
    collection = db["chunks"]
    
    # Fetch all chunks that have an embedding, sorted by doc_id and chunk_index
    cursor = collection.find({"embedding": {"$ne": None}, "embedding": {"$exists": True}}).sort([("doc_id", 1), ("chunk_index", 1)])
    
    current_doc_id = None
    current_embeddings = []
    
    docs_processed = 0
    total_chunks = 0
    
    for chunk in cursor:
        doc_id = chunk.get("doc_id")
        embedding = chunk.get("embedding")
        
        if current_doc_id is None:
            current_doc_id = doc_id
            
        if doc_id != current_doc_id:
            if current_embeddings:
                replace_vectors_in_faiss(current_doc_id, current_embeddings)
                docs_processed += 1
                total_chunks += len(current_embeddings)
            current_doc_id = doc_id
            current_embeddings = []
            
        if embedding:
            current_embeddings.append(embedding)
        
    if current_doc_id is not None and current_embeddings:
        replace_vectors_in_faiss(current_doc_id, current_embeddings)
        docs_processed += 1
        total_chunks += len(current_embeddings)
        
    print(f"Sync complete: Processed {docs_processed} documents with {total_chunks} total chunks.")
    return docs_processed, total_chunks


def delete_vectors_from_faiss(doc_id: str):
    """
    Removes all FAISS vectors and metadata registry entries for a given document.
    """
    index, metadata = _load_or_create_index()
    
    if doc_id in metadata and len(metadata[doc_id]) > 0:
        old_ids = np.array(metadata[doc_id], dtype=np.int64)
        try:
            index.remove_ids(old_ids)
            print(f"Deleted {len(old_ids)} vectors from FAISS for document '{doc_id}'.")
        except Exception as e:
            print(f"Could not delete vectors from FAISS ({e}).")
            
        del metadata[doc_id]
        _save_index(index, metadata)
    else:
        print(f"No FAISS vectors found to delete for document '{doc_id}'.")
