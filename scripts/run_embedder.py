import os
import sys
import argparse

# Add the project root to sys.path so we can import ingest and database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.embedder import embed_and_update_all, embed_and_update_chunks

def main():
    parser = argparse.ArgumentParser(description="Generate Gemini embeddings for stored RAMS chunks.")
    parser.add_argument("--doc-id", help="Only embed chunks for one document.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings instead of only filling missing ones.",
    )
    args = parser.parse_args()

    print("Starting background embedding process...")
    if args.doc_id:
        updated_count = embed_and_update_chunks(args.doc_id)
    else:
        updated_count = embed_and_update_all(force=args.force)
    print(f"Finished embedding process. Processed {updated_count} chunks.")

if __name__ == "__main__":
    main()
