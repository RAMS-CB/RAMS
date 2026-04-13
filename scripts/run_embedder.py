import os
import sys
import requests

# Add the project root to sys.path so we can import ingest and database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.embedder import embed_and_update_all

def main():
    print("Starting background embedding process...")
    # This function automatically targets chunks missing an embedding, 
    # generates embeddings, and updates the documents in MongoDB.
    updated_count = embed_and_update_all()
    print(f"Finished embedding process. Processed {updated_count} chunks.")

if __name__ == "__main__":
    main()
