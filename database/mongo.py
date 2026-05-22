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




import re

def fetch_document_content(doc_id: str, url: str) -> Dict[str, Any]:
    """Fetch the document content from its URL to check if it changed.
    Uses native Google Docs export, or Jina Reader API to properly execute JS 
    and extract clean markdown text for generic sites. Recursively fetches linked documents."""
    import urllib.parse
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "X-With-Links-Summary": "true"
    }

    def clean_google_link(link: str) -> str:
        if "google.com/url" in link:
            m = re.search(r'q=([^&]+)', link)
            if m: return urllib.parse.unquote(m.group(1))
        return link

    url_to_serial = {}
    serial_counter = 1
    seen_urls = set([url])
    
    queue = [(url, "Primary Document", 0)]
    final_content = ""
    embedded_blocks = []
    original_gdoc_id = None

    while queue:
        current_url, linked_text, depth = queue.pop(0)
        is_primary = (depth == 0)
        
        target_jina_url = f"https://r.jina.ai/{current_url}"
        
        gdoc_match = re.search(r'(docs\.google\.com/document/d/[a-zA-Z0-9_-]+)', current_url)
        if gdoc_match:
            base_url = gdoc_match.group(1)
            target_jina_url = f"https://r.jina.ai/https://{base_url}/export?format=html"
            if is_primary:
                doc_id_match = re.search(r'/document/d/([a-zA-Z0-9_-]+)', current_url)
                if doc_id_match:
                    original_gdoc_id = doc_id_match.group(1)
        
        if is_primary:
            print(f"Fetching primary document via Jina: {current_url}")
        else:
            serial_num = url_to_serial.get(current_url)
            print(f"Fetching linked document {serial_num} (depth {depth}): {current_url}")
            
        try:
            response = requests.get(target_jina_url, headers=headers, timeout=20)
            if response.status_code != 200:
                print(f"Failed to fetch {current_url} (HTTP {response.status_code})")
                continue
            
            content = response.text
            links = re.findall(r'\[(.*?)\]\((https?://.*?)\)', content)
            
            # Map valid links and safely replace inline markdown
            for text, raw_link in links:
                clean_link = clean_google_link(raw_link)
                
                # Exclude self-references and loops to the original google doc anchor links
                if clean_link == current_url:
                    continue
                if original_gdoc_id and original_gdoc_id in clean_link:
                    continue
                if clean_link.startswith("https://r.jina.ai") or not clean_link.startswith("http"):
                    continue
                    
                if clean_link not in url_to_serial:
                    url_to_serial[clean_link] = serial_counter
                    serial_counter += 1
                    
                    if clean_link not in seen_urls:
                        seen_urls.add(clean_link)
                        queue.append((clean_link, text, depth + 1))
                        
                # Update markdown correctly in text
                serial = url_to_serial[clean_link]
                original_markdown = f"[{text}]({raw_link})"
                new_markdown = f"[[{serial}] {text}]({raw_link})"
                content = content.replace(original_markdown, new_markdown)
                
            if is_primary:
                final_content += content
            else:
                serial_num = url_to_serial.get(current_url, 999)
                embedded_blocks.append((serial_num, current_url, content))
                
        except Exception as e:
            print(f"Error fetching {current_url}: {e}")
            continue

    if embedded_blocks:
        final_content += "\n\n# Embedded Documents\n\n"
        embedded_blocks.sort(key=lambda x: x[0])
        for serial_num, block_url, block_content in embedded_blocks:
            final_content += f"## Serial Number {serial_num}: {block_url}\n\n"
            final_content += block_content + "\n\n"

    try:
        content_hash = hashlib.md5(final_content.encode('utf-8', errors='ignore')).hexdigest()
        return {
            "id": doc_id,
            "content_hash": content_hash,
            "url": url,
            "raw_html": final_content
        }
    except Exception as hash_e:
        print(f"Error finalizing document '{doc_id}': {hash_e}")
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
            
            # Generate embeddings directly with Gemini.
            from ingest.embedder import embed_and_update_chunks
            embedded_count = embed_and_update_chunks(doc_id)
            
            # Update Mongo metadata
            update_mongo_metadata(doc_id, new_hash)
            print(f"Pipeline execution for {doc_id} completed with {embedded_count} embedded chunk(s).")
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
