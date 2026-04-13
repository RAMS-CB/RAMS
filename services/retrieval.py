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
        return results
    except Exception as e:
        print(f"Error executing vector search on MongoDB Atlas: {e}")
        print("Note: Ensure you have manually created the 'vector_index' Vector Search Index in the MongoDB Atlas UI.")
        return []
