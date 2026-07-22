import os
from services.retrieval import search_similar_chunks, retrieve_documents

def test_retrieval():
    query = input("\nEnter your question: ")
    print(f"Retrieving documents for: '{query}'...")
    results = retrieve_documents(query, limit=5)
    print(f"Found {len(results)} results:")
    for res in results:
        print(f"  [Doc {res.get('doc_id')} | Chunk {res.get('chunk_index')} | Score {res.get('score', 0):.3f}] {res.get('text_content', '')[:80]}...")

if __name__ == "__main__":
    test_retrieval()
