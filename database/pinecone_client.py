import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rams")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in the environment.")
        
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, create if not
    # Note: Creating an index can take a minute on Pinecone Serverless
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{index_name}' with 384 dimensions...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)

def upsert_vectors(vectors: List[Dict[str, Any]]):
    """
    Upsert a batch of vectors to Pinecone.
    Format expected: [{"id": "chunk_xyz", "values": [0.1, 0.2, ...], "metadata": {"doc_id": "doc_xyz", "chunk_index": 0, "text_content": "..."}}]
    """
    if not vectors:
        return
        
    index = get_pinecone_index()
    
    # Pinecone recommends batches of around 100 for optimal performance
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    print(f"Upserted {len(vectors)} vectors to Pinecone index.")

def search_vectors(query_embedding: List[float], limit: int = 6):
    """
    Search Pinecone for the most similar vectors.
    """
    index = get_pinecone_index()
    results = index.query(
        vector=query_embedding,
        top_k=limit,
        include_metadata=True
    )
    return results
