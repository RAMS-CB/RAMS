import os
import sys

# Add parent directory to sys.path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.sanity_client import query_sanity
from database.pinecone_client import upsert_vectors
from dotenv import load_dotenv

def backfill():
    load_dotenv()
    
    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY is missing from your environment variables (.env).")
        print("Please add it and run this script again.")
        sys.exit(1)
        
    print("Fetching chunks from Sanity...")
    # Get all chunks that already have embeddings
    query = '*[_type == "chunk" && defined(embedding)]'
    chunks = query_sanity(query) or []
    
    if not chunks:
        print("No chunks with embeddings found in Sanity.")
        return
        
    print(f"Found {len(chunks)} chunks with embeddings. Preparing for Pinecone upsert...")
    
    pinecone_vectors = []
    for chunk in chunks:
        pinecone_vectors.append({
            "id": chunk["_id"],
            "values": chunk["embedding"],
            "metadata": {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "text_content": chunk.get("text_content", "")
            }
        })
        
    print("Upserting to Pinecone...")
    upsert_vectors(pinecone_vectors)
    print("Backfill complete.")

if __name__ == "__main__":
    backfill()
