import os
from typing import List, Dict, Any
import numpy as np
from database.sanity_client import query_sanity

def search_similar_chunks(query_embedding: List[float], limit: int = 6) -> List[Dict[str, Any]]:
    """
    Search Pinecone for the most similar chunks.
    Returns top matching chunks sorted by relevance score.
    """
    from database.pinecone_client import search_vectors
    try:
        results = search_vectors(query_embedding, limit=limit)
        
        scored_chunks = []
        if hasattr(results, 'matches'):
            for match in results.matches:
                scored_chunks.append({
                    "doc_id": match.metadata.get("doc_id"),
                    "chunk_index": int(match.metadata.get("chunk_index", 0)),
                    "text_content": match.metadata.get("text_content", ""),
                    "metadata": match.metadata,
                    "score": float(match.score)
                })
            
        return scored_chunks
    except Exception as e:
        print(f"Error executing similarity search on Pinecone: {e}")
        return []

def expand_chunk_context(base_chunk: Dict[str, Any], query_embedding: List[float]) -> List[Dict[str, Any]]:
    """
    Expands context around base_chunk dynamically by fetching neighboring chunks from Sanity.
    +-1 neighbors are included unconditionally, while +-2 neighbors are checked against the adaptive threshold.
    """
    doc_id = base_chunk.get("doc_id")
    base_idx = base_chunk.get("chunk_index")
    if doc_id is None or base_idx is None:
        return [base_chunk]
        
    query_vec = np.array(query_embedding)
    expanded_chunks = {base_idx: base_chunk}
    
    base_score = base_chunk.get("score", 0.65)
    threshold = max(0.68, base_score - 0.04)
    print(f"Base chunk score: {base_score:.3f}. Using dynamic threshold: {threshold:.3f} for +-2 expansion")
    
    def fetch_and_evaluate(idx: int, check_threshold: bool = True) -> bool:
        if idx in expanded_chunks:
            return True
            
        groq_query = '*[_type == "chunk" && doc_id == $doc_id && chunk_index == $idx][0]'
        doc = query_sanity(groq_query, {"doc_id": doc_id, "idx": idx})
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
            sim = 0.0
            
        if not check_threshold or sim >= threshold:
            reason = "Unconditional (+-1 neighbor)" if not check_threshold else f"Score: {sim:.3f} >= {threshold:.3f}"
            print(f"Expanding chunk -> Found relevant chunk {idx} for doc {doc_id} ({reason})")
            expanded_chunks[idx] = {
                "doc_id": doc.get("doc_id"),
                "chunk_index": doc.get("chunk_index"),
                "text_content": doc.get("text_content"),
                "metadata": doc.get("metadata"),
                "score": float(sim)
            }
            return True
        else:
            print(f"Stopping expansion -> Chunk {idx} for doc {doc_id} is irrelevant (Score: {sim:.3f} < {threshold:.3f})")
            return False

    # Expand ahead (+1 unconditional, +2 adaptive)
    if fetch_and_evaluate(base_idx + 1, check_threshold=False):
        fetch_and_evaluate(base_idx + 2, check_threshold=True)
            
    # Expand behind (-1 unconditional, -2 adaptive)
    if fetch_and_evaluate(base_idx - 1, check_threshold=False):
        fetch_and_evaluate(base_idx - 2, check_threshold=True)
            
    return list(expanded_chunks.values())


def retrieve_documents(query: str, limit: int = 6) -> List[Dict[str, Any]]:
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
