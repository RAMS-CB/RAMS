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
    and extract clean markdown text for generic sites. Recursively fetches linked documents up to depth 1, max 10 documents."""
    import urllib.parse
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "X-With-Links-Summary": "true"
    }
    
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    def clean_google_link(link: str) -> str:
        if "google.com/url" in link:
            m = re.search(r'q=([^&]+)', link)
            if m: link = urllib.parse.unquote(m.group(1))
        
        # Strip URL fragments to prevent crawling the same page for different sections
        link = link.split('#')[0]
        
        # Normalize Google Docs URLs by stripping query parameters and formatting them consistently
        gdoc_m = re.search(r'(docs\.google\.com/document/(?:u/\d+/)?d/(?:e/)?([a-zA-Z0-9_-]+))', link)
        if gdoc_m:
            gdoc_id = gdoc_m.group(2)
            if "/pub" in link:
                link = f"https://docs.google.com/document/d/e/{gdoc_id}/pub"
            else:
                link = f"https://docs.google.com/document/d/{gdoc_id}"
                
        # Strip trailing slashes to normalize URLs
        if link.endswith('/'):
            link = link[:-1]
            
        return link

    def should_crawl_link(link: str) -> bool:
        # Exclude images, videos, and non-content formats
        exclude_patterns = [
            r'\.(png|jpg|jpeg|gif|svg|webp|ico|mp4|avi|mov|mp3|wav|zip|tar|gz|rar|exe|dmg|pkg)$',
            r'docs-images-rt',
            r'youtube\.com',
            r'youtu\.be',
            r'drive\.google\.com/file',
            r'google\.com/maps',
            r'/abuse',
            r'zohocommerce\.com'
        ]
        for pattern in exclude_patterns:
            if re.search(pattern, link, re.IGNORECASE):
                return False
        return True

    url_to_serial = {}
    serial_counter = 1
    
    cleaned_start_url = clean_google_link(url)
    seen_urls = set([cleaned_start_url])
    
    queue = [(cleaned_start_url, "Primary Document", 0)]
    final_content = ""
    embedded_blocks = []
    original_gdoc_id = None

    # Parse primary Google Doc ID to prevent self-looping
    primary_gdoc_match = re.search(r'docs\.google\.com/document/(?:u/\d+/)?d/(?:e/)?([a-zA-Z0-9_-]+)', cleaned_start_url)
    if primary_gdoc_match:
        original_gdoc_id = primary_gdoc_match.group(1)
        print(f"Detected primary Google Doc ID: {original_gdoc_id}")

    while queue:
        current_url, linked_text, depth = queue.pop(0)
        is_primary = (depth == 0)
        
        target_jina_url = f"https://r.jina.ai/{current_url}"
        
        gdoc_match = re.search(r'docs\.google\.com/document/(?:u/\d+/)?d/(?:e/)?([a-zA-Z0-9_-]+)', current_url)
        if gdoc_match:
            gdoc_id = gdoc_match.group(1)
            if "/pub" in current_url:
                clean_pub_url = f"https://docs.google.com/document/d/e/{gdoc_id}/pub"
                target_jina_url = f"https://r.jina.ai/{clean_pub_url}"
            else:
                target_jina_url = f"https://r.jina.ai/https://docs.google.com/document/d/{gdoc_id}/export?format=html"
        
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
            
            for text, raw_link in links:
                clean_link = clean_google_link(raw_link)
                
                # Exclude self-references and loops to the original google doc anchor links
                if clean_link == current_url:
                    continue
                if original_gdoc_id and original_gdoc_id in clean_link:
                    continue
                if clean_link.startswith("https://r.jina.ai") or not clean_link.startswith("http"):
                    continue
                
                # Only queue linked documents if we are at depth 0 (meaning we only crawl depth 1)
                # and the link passes our validation filters, and we haven't reached the limit of 10.
                if depth < 1 and clean_link not in seen_urls:
                    if should_crawl_link(clean_link):
                        if len(url_to_serial) < 10:
                            url_to_serial[clean_link] = serial_counter
                            serial_counter += 1
                            seen_urls.add(clean_link)
                            queue.append((clean_link, text, depth + 1))
                        
                # Update markdown correctly in text with serial references if it's in our serial map
                if clean_link in url_to_serial:
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
