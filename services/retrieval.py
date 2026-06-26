import os
from typing import List, Dict, Any

from database.mongo_connection import get_db_connection

def search_similar_chunks(query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Query the MongoDB Atlas Vector Search index using the provided query embedding.
    Returns the top matching chunks including their text content and metadata.
    """
    client = get_db_connection()
    db = client["rams_db"]
    collection = db["chunks"]
    
    # Vector Search Pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 0,
                "doc_id": 1,
                "chunk_index": 1,
                "text_content": 1,
                "metadata": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        if results:
            return results
    except Exception as e:
        print(f"Error executing vector search on MongoDB Atlas: {e}")
        print("Note: Ensure you have manually created the 'vector_index' Vector Search Index in the MongoDB Atlas UI.")
        print("Falling back to local cosine similarity computation...")

    # Fallback to local cosine similarity if Vector Search index is not present or returns empty
    try:
        import numpy as np
        query_vec = np.array(query_embedding)
        chunks = list(collection.find({"embedding": {"$exists": True}}))
        
        scored_chunks = []
        for chunk in chunks:
            emb = np.array(chunk["embedding"])
            # Compute cosine similarity
            norm_q = np.linalg.norm(query_vec)
            norm_c = np.linalg.norm(emb)
            if norm_q > 0 and norm_c > 0:
                sim = np.dot(query_vec, emb) / (norm_q * norm_c)
            else:
                sim = 0
            
            scored_chunks.append({
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index"),
                "text_content": chunk.get("text_content"),
                "metadata": chunk.get("metadata"),
                "score": float(sim)
            })
            
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:limit]
    except ImportError:
        print("numpy is not installed. Cannot fallback to local cosine similarity.")
        return []
    except Exception as e:
        print(f"Local fallback search failed: {e}")
        return []

def expand_chunk_context(base_chunk: Dict[str, Any], query_embedding: List[float]) -> List[Dict[str, Any]]:
    """
    Expands the context around a base chunk dynamically. 
    It fetches up to 2 chunks ahead and behind. If they are relevant (score >= threshold),
    it continues fetching more chunks.
    """
    try:
        import numpy as np
    except ImportError:
        print("numpy is required for context expansion.")
        return [base_chunk]

    client = get_db_connection()
    collection = client["rams_db"]["chunks"]

    doc_id = base_chunk.get("doc_id")
    base_idx = base_chunk.get("chunk_index")
    if doc_id is None or base_idx is None:
        return [base_chunk]
        
    query_vec = np.array(query_embedding)
    expanded_chunks = {base_idx: base_chunk}
    
    # Calculate dynamic threshold based on the base chunk's score
    base_score = base_chunk.get("score", 0.65)
    threshold = max(0.62, base_score - 0.05)
    print(f"Base chunk score: {base_score:.3f}. Using dynamic threshold: {threshold:.3f}")
    
    def fetch_and_evaluate(idx: int) -> bool:
        if idx in expanded_chunks:
            return True
        doc = collection.find_one({"doc_id": doc_id, "chunk_index": idx})
        if not doc:
            return False
        
        emb = doc.get("embedding")
        if not emb:
            return False
            
        emb_vec = np.array(emb)
        norm_q = np.linalg.norm(query_vec)
        norm_c = np.linalg.norm(emb_vec)
        if norm_q > 0 and norm_c > 0:
            sim = np.dot(query_vec, emb_vec) / (norm_q * norm_c)
        else:
            sim = 0
            
        if sim >= threshold:
            print(f"Expanding chunk -> Found relevant chunk {idx} for doc {doc_id} (Score: {sim:.3f})")
            expanded_chunks[idx] = {
                "doc_id": doc.get("doc_id"),
                "chunk_index": doc.get("chunk_index"),
                "text_content": doc.get("text_content"),
                "metadata": doc.get("metadata"),
                "score": float(sim)
            }
            return True
        else:
            print(f"Stopping expansion -> Chunk {idx} for doc {doc_id} is irrelevant (Score: {sim:.3f})")
            return False

    # Expand ahead (forward)
    current_ahead = base_idx
    while True:
        relevant_found = False
        for i in range(1, 3):
            if fetch_and_evaluate(current_ahead + i):
                relevant_found = True
        if relevant_found:
            current_ahead += 2
        else:
            break
            
    # Expand behind (backward)
    current_behind = base_idx
    while True:
        relevant_found = False
        for i in range(1, 3):
            if fetch_and_evaluate(current_behind - i):
                relevant_found = True
        if relevant_found:
            current_behind -= 2
        else:
            break
            
    return list(expanded_chunks.values())

def retrieve_documents(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Generate an embedding for the text query and perform a vector search.
    Then expands the context adaptively.
    This is the main entry point for the frontend / API.
    """
    from ingest.embedder import generate_query_embedding
    
    query_embedding = generate_query_embedding(query)
    base_chunks = search_similar_chunks(query_embedding, limit=limit)
    
    all_expanded_chunks = {}
    for chunk in base_chunks:
        expanded = expand_chunk_context(chunk, query_embedding)
        for ec in expanded:
            # Use a unique identifier to deduplicate (doc_id + chunk_index)
            key = f"{ec['doc_id']}_{ec['chunk_index']}"
            if key not in all_expanded_chunks:
                all_expanded_chunks[key] = ec
                
    # Convert back to list and sort sequentially by chunk_index to ensure chronological reading
    final_chunks = list(all_expanded_chunks.values())
    final_chunks.sort(key=lambda x: (x["doc_id"], x["chunk_index"]))
    
    return final_chunks
