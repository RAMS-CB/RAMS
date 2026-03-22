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
    
    if updated_count > 0:
        # If we embedded new chunks, tell the Render API to sync FAISS!
        render_url = os.getenv("RENDER_API_URL")
        if render_url:
            # Ensure no trailing slash
            render_url = render_url.rstrip('/')
            sync_endpoint = f"{render_url}/sync-faiss"
            print(f"Notifying Render API to sync FAISS at: {sync_endpoint}")
            try:
                # Add a reasonable timeout in case Render is slow
                response = requests.post(sync_endpoint, timeout=30)
                if response.status_code == 200:
                    print("Successfully triggered FAISS sync on the Render API.")
                    print(response.json())
                else:
                    print(f"Failed to trigger FAISS sync. Status code: {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"Error calling Render API sync endpoint: {e}")
        else:
            print("RENDER_API_URL environment variable is not set. Skipping FAISS sync trigger.")

if __name__ == "__main__":
    main()
