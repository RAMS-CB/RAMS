import os
from typing import List, Dict, Any

from database.mongo_connection import get_db_connection

def search_similar_chunks(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
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

def retrieve_documents(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Generate an embedding for the text query and perform a vector search.
    This is the main entry point for the frontend / API.
    """
    from ingest.embedder import generate_query_embedding
    
    query_embedding = generate_query_embedding(query)
    return search_similar_chunks(query_embedding, limit=limit)
