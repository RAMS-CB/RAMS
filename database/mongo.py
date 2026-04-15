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
from database.mongo_connection import get_db_connection
from api import trigger_github_action




import re

def fetch_document_content(doc_id: str, url: str) -> Dict[str, Any]:
    """Fetch the document content from its URL to check if it changed.
    Uses native Google Docs export, or Jina Reader API to properly execute JS 
    and extract clean markdown text for generic sites."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Check if it's a Google Doc. If so, intercept it!
        # Google docs render text on a <canvas>, which Jina and scrapers can't read.
        # But we can forcefully export the raw text:
        gdoc_match = re.search(r'(docs\.google\.com/document/d/[a-zA-Z0-9_-]+)', url)
        if gdoc_match:
            base_url = gdoc_match.group(1)
            target_url = f"https://{base_url}/export?format=txt"
            response = requests.get(target_url, headers=headers)
        else:
            # For generic urls and SPAs, use Jina Reader for markdown clean-up
            jina_url = f"https://r.jina.ai/{url}"
            response = requests.get(jina_url, headers=headers)
            
        response.raise_for_status()
        
        # Simple hash of the extracted Markdown/text to detect changes
        content_hash = hashlib.md5(response.content).hexdigest()
        
        return {
            "id": doc_id,
            "content_hash": content_hash,
            "url": url,
            "raw_html": response.text  # This is actually clean Text / Markdown now
        }
    except Exception as e:
        print(f"Error fetching document '{doc_id}' at '{url}': {e}")
        return {
            "id": doc_id,
            "content_hash": "error",
            "url": url,
            "raw_html": ""
        }





def get_stored_document_hash(doc_id: str) -> str:
    """Check Mongo for the currently stored version/hash of the document."""
    client = get_db_connection()
    db = client["rams_db"]
    
    # Check the "metadata" collection for this document
    doc = db["metadata"].find_one({"doc_id": doc_id})
    if doc and "content_hash" in doc:
        return doc["content_hash"]
        
    return ""

def update_mongo_metadata(doc_id: str, new_hash: str):
    """Update MongoDB metadata with the new document hash and status."""
    client = get_db_connection()
    db = client["rams_db"]
    
    # Upsert the new hash into the "metadata" collection
    db["metadata"].update_one(
        {"doc_id": doc_id},
        {"$set": {
            "content_hash": new_hash,
            "last_updated_timestamp": time.time()
        }},
        upsert=True
    )
    print(f"Updated Mongo metadata for document '{doc_id}' with new hash '{new_hash}'")

def chunk_text(text: str) -> list:
    # This would typically be in ingest.chunker
    return [text]



def process_pipeline():
    """Main pipeline to process all documents and update DBs if changed."""
    print("Scheduler Triggered: Checking for document updates...")
    
    client = get_db_connection()
    db = client["rams_db"]
    sources = list(db["document_sources"].find({}))
    
    if not sources:
        print("No document sources registered. Pipeline execution skipped.")
        return
        
    for source in sources:
        doc_id = source.get("doc_id")
        url = source.get("url")
        
        if not doc_id or not url:
            continue
            
        print(f"Checking document source: {doc_id}...")
        
        # Fetch latest document
        latest_doc = fetch_document_content(doc_id, url)
        new_hash = latest_doc.get("content_hash")
        
        if new_hash == "error":
            continue
        
        # Check if changed (hash/version compare)
        stored_hash = get_stored_document_hash(doc_id)
        
        if new_hash != stored_hash:
            print(f"Document {doc_id} has changed. Processing pipeline...")
            
            # Extract text (loader)
            text = extract_text(latest_doc)
            
            # Create chunks AND store them in Mongo (chunker)
            from ingest.chunker import chunk_and_store
            chunk_and_store(doc_id=doc_id, text=text)
            
            # Trigger GitHub Action to generate embeddings in the background
            trigger_github_action("embed_chunks")
            
            # Update Mongo metadata
            update_mongo_metadata(doc_id, new_hash)
            print(f"Pipeline execution for {doc_id} initiated and sent to GitHub Actions.")
        else:
            # Do nothing
            print(f"Document {doc_id} has not changed. Skipping processing.")
            
    print("Global Pipeline checking cycle finished.")

def start_scheduler():
    """Start the scheduler to run the pipeline periodically."""
    # Run every Sunday at 2:00 AM
    schedule.every().sunday.at("02:00").do(process_pipeline)
    print("Scheduler started. Pipeline will run every Sunday at 02:00 AM.")
    
    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute

if __name__ == "__main__":
    # Test the pipeline execution once directly
    # process_pipeline()
    start_scheduler()
