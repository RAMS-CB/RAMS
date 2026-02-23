import time
import hashlib
import schedule
import requests
import sys
import os
from typing import Dict, Any

# Add parent directory to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming the following folder structure is used to import modules for the pipeline:
from ingest.loader import extract_text
# from ingest.chunker import chunk_text
# from ingest.embedder import generate_embeddings
# from services.retrieval import replace_vectors_in_faiss

def fetch_latest_document() -> Dict[str, Any]:
    """Fetch the latest document metadata (like its hash) to know if it changed."""
    # Replace with your actual published Google Doc URL
    google_doc_url = "https://docs.google.com/document/d/e/2PACX-1vQ.../pub"
    
    try:
        # We need to fetch it to know if it changed, unfortunately published
        # Google docs don't always give a reliable ETag without downloading.
        response = requests.get(google_doc_url)
        response.raise_for_status()
        
        # Simple hash of the raw HTML content to detect changes
        content_hash = hashlib.md5(response.content).hexdigest()
        
        return {
            "id": "google_doc_main",
            "content_hash": content_hash,
            "url": google_doc_url,
            # We can optionally pass the raw HTML forward so we don't have to download again
            "raw_html": response.text 
        }
    except Exception as e:
        print(f"Error fetching Google Doc: {e}")
        return {
            "id": "google_doc_main",
            "content_hash": "error",
            "url": google_doc_url,
            "raw_html": ""
        }

def get_stored_document_hash(doc_id: str) -> str:
    """Check Mongo for the currently stored version/hash of the document."""
    # Mock implementation
    return "old_hash_000"

def update_mongo_metadata(doc_id: str, new_hash: str):
    """Update MongoDB metadata with the new document hash and status."""
    print(f"Updated Mongo metadata for document {doc_id} with new hash {new_hash}")

def chunk_text(text: str) -> list:
    # This would typically be in ingest.chunker
    return [text]

def generate_embeddings(chunks: list) -> list:
    # This would typically be in ingest.embedder
    return [[0.1, 0.2, 0.3]]

def replace_vectors_in_faiss(doc_id: str, embeddings: list):
    # This would typically be in services or a vector DB module
    print(f"Replaced vectors in FAISS for document {doc_id}")

def process_pipeline():
    """Main pipeline to process document and update DBs if changed."""
    print("Scheduler Triggered: Checking for document updates...")
    
    # Fetch latest document
    latest_doc = fetch_latest_document()
    doc_id = latest_doc.get("id")
    new_hash = latest_doc.get("content_hash")
    
    # Check if changed (hash/version compare)
    stored_hash = get_stored_document_hash(doc_id)
    
    if new_hash != stored_hash:
        print(f"Document {doc_id} has changed. Processing pipeline...")
        
        # Extract text (loader)
        text = extract_text(latest_doc)
        
        # Chunk text (chunker)
        chunks = chunk_text(text)
        
        # Generate embeddings (embedder)
        embeddings = generate_embeddings(chunks)
        
        # Replace vectors in FAISS
        replace_vectors_in_faiss(doc_id, embeddings)
        
        # Update Mongo metadata
        update_mongo_metadata(doc_id, new_hash)
        print("Pipeline execution completed successfully.")
    else:
        # Do nothing
        print(f"Document {doc_id} has not changed. Skipping processing.")

def start_scheduler():
    """Start the scheduler to run the pipeline periodically."""
    # For example, run every hour
    schedule.every(1).hours.do(process_pipeline)
    print("Scheduler started...")
    
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

if __name__ == "__main__":
    # Test the pipeline execution once directly
    process_pipeline()
