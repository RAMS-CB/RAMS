import os
import sys

# Add parent directory to sys.path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.embedder import embed_and_update_all

def main():
    print("Starting embedding generation for all missing chunks...")
    updated_count = embed_and_update_all(force=False)
    print(f"Embedding complete! Updated {updated_count} chunks.")

if __name__ == "__main__":
    main()
