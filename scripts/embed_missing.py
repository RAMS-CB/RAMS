import os
import sys

# Add the project root to sys.path so we can import ingest and database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.embedder import embed_and_update_all

def main():
    print("Starting background embedding process (Missing Chunks Only)...")
    # Calling with force=False will only embed chunks that don't have embeddings yet
    updated_count = embed_and_update_all(force=False)
    print(f"Finished embedding process. Backfilled {updated_count} missing chunks.")

if __name__ == "__main__":
    main()
