import os
from database.mongo_connection import get_db_connection

def test_retrieval():
    client = get_db_connection()
    db = client["rams_db"]
    collection = db["chunks"]

    from ingest.embedder import generate_query_embedding
    
    query = input("\nEnter your question: ")
    print(f"Generating embedding for: '{query}'...")
    query_embedding = generate_query_embedding(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 5
            }
        },
        {
            "$project": {
                "text_content": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    print("Running vector search pipeline...")
    try:
        results = list(collection.aggregate(pipeline))
        print(f"Found {len(results)} results:")
        for res in results:
            print(res)
    except Exception as e:
        print(f"Error executing vector search: {e}")

if __name__ == "__main__":
    test_retrieval()
