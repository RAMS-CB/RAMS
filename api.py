import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main database connection strictly for startup lifecycle
from database.mongo_connection import get_db_connection

app = FastAPI(
    title="RAMS API",
    description="API for the RAMS Document Processing Pipeline",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

# Enable CORS for the frontend origin and others
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading
import logging

def start_embedding_job(doc_id: Optional[str] = None, force: bool = False):
    """Generate missing Gemini embeddings in a background thread."""
    def run_job():
        try:
            from ingest.embedder import embed_and_update_all, embed_and_update_chunks

            if doc_id:
                updated = embed_and_update_chunks(doc_id)
                logging.info("Gemini embedding job completed for %s: %s chunk(s)", doc_id, updated)
            else:
                updated = embed_and_update_all(force=force)
                logging.info("Gemini embedding job completed: %s chunk(s)", updated)
        except Exception as e:
            logging.exception("Gemini embedding job failed: %s", e)

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()
    return thread

# Startup event to ensure database connection is ready
@app.on_event("startup")
async def startup_event():
    try:
        print("Initializing FastAPI Server...")
        get_db_connection()
        
        # Start the Sunday night scheduler in a separate daemon thread 
        # so it doesn't block the API requests!
        from database.mongo import start_scheduler
        scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
        scheduler_thread.start()
        print("Background pipeline scheduler successfully booted alongside API.")
        
    except Exception as e:
        print(f"Failed to connect to database during startup: {e}")
        # Not exiting here so the API still loads, but it logs the error.

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the RAMS API!"}

# Example Request Model
class QueryRequest(BaseModel):
    query: str

class ChunkRequest(BaseModel):
    text: str
    doc_id: str = "test_doc"
    chunk_size: int = 500
    overlap: int = 50
    store: bool = False

class EmbedRequest(BaseModel):
    doc_id: str

class EmbedAllRequest(BaseModel):
    force: bool = False

class MetadataUpdateRequest(BaseModel):
    doc_id: str
    new_hash: str

class DocSourceRequest(BaseModel):
    title: str
    doc_id: str
    url: str

class BulkDocSourceRequest(BaseModel):
    sources: List[DocSourceRequest]

# API Route for Retrieval
@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    try:
        from services.retrieval import retrieve_documents
        from services.llm_service import generate_answer
        
        # 1. Retrieve relevant chunks from the database
        chunks = retrieve_documents(request.query)
        
        # 2. Extract out the actual text and scores to send to the frontend
        context_snippets = [
            {
                "text": chunk.get("text_content", ""), 
                "score": chunk.get("score", 0), 
                "doc_id": chunk.get("doc_id", "unknown")
            } 
            for chunk in chunks
        ]
        
        # 3. Generate answer using Gemini Flash based on the retrieved context
        final_answer = generate_answer(request.query, chunks)
        
        return {
            "question": request.query,
            "answer": final_answer,
            "context": context_snippets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk")
async def create_chunks(request: ChunkRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        if request.store:
            from ingest.chunker import chunk_and_store
            
            inserted_ids = chunk_and_store(
                doc_id=request.doc_id, 
                text=request.text, 
                chunk_size=request.chunk_size, 
                overlap=request.overlap
            )
            
            start_embedding_job(request.doc_id)
            
            return {
                "message": "Chunks created, stored, and queued for Gemini embedding",
                "doc_id": request.doc_id,
                "chunk_count": len(inserted_ids),
                "inserted_ids": inserted_ids
            }
        else:
            from ingest.chunker import chunk_text
            chunks = chunk_text(
                text=request.text, 
                chunk_size=request.chunk_size, 
                overlap=request.overlap
            )
            return {
                "message": "Chunks created successfully (not stored)",
                "doc_id": request.doc_id,
                "chunk_count": len(chunks),
                "chunks": chunks
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def generate_embeddings_endpoint(request: EmbedRequest):
    if not request.doc_id:
        raise HTTPException(status_code=400, detail="doc_id cannot be empty")
        
    try:
        start_embedding_job(request.doc_id)
        
        return {
            "message": "Gemini embedding job started successfully",
            "doc_id": request.doc_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/all")
async def generate_all_embeddings_endpoint(request: EmbedAllRequest):
    try:
        start_embedding_job(force=request.force)
        action = "full Gemini re-embedding" if request.force else "Gemini embedding backfill"

        return {
            "message": f"{action} job started successfully",
            "force": request.force,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/missing")
async def generate_missing_embeddings_endpoint():
    """Trigger background job to embed only chunks that don't have embeddings yet."""
    try:
        start_embedding_job(force=False)
        return {
            "message": "Gemini embedding backfill job (missing only) started successfully",
            "force": False,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-metadata")
async def update_metadata_endpoint(request: MetadataUpdateRequest):
    """Manually update the metadata (e.g. hash) of a document in MongoDB"""
    if not request.doc_id or not request.new_hash:
        raise HTTPException(status_code=400, detail="doc_id and new_hash cannot be empty")
        
    try:
        from database.mongo import update_mongo_metadata
        update_mongo_metadata(doc_id=request.doc_id, new_hash=request.new_hash)
        
        return {
            "message": "Document metadata updated successfully in MongoDB",
            "doc_id": request.doc_id,
            "new_hash": request.new_hash
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/document-sources", methods=["POST", "PUT"])
async def add_document_source(request: DocSourceRequest):
    try:
        from database.mongo_connection import get_db_connection
        client = get_db_connection()
        db = client["rams_db"]
        db["document_sources"].update_one(
            {"doc_id": request.doc_id},
            {"$set": {
                "title": request.title,
                "url": request.url
            }},
            upsert=True
        )
        
        # Run the extraction, chunking, and action trigger in the background
        import threading
        import traceback
        import sys
        def process_new_url_bg():
            try:
                print(f"[BG-THREAD] Step 1: Starting background processing for {request.doc_id} ({request.url})", flush=True)
                
                from database.mongo import fetch_document_content, update_mongo_metadata
                from ingest.loader import extract_text
                from ingest.chunker import chunk_and_store
                print(f"[BG-THREAD] Step 2: Imports successful", flush=True)
                
                latest_doc = fetch_document_content(request.doc_id, request.url)
                new_hash = latest_doc.get("content_hash")
                raw_html = latest_doc.get("raw_html", "")
                print(f"[BG-THREAD] Step 3: Fetched document. Hash={new_hash}, Content length={len(raw_html)}", flush=True)
                
                if new_hash == "error":
                    print(f"[BG-THREAD] ABORT: fetch_document_content returned error hash for {request.doc_id}", flush=True)
                    return
                
                text = extract_text(latest_doc)
                print(f"[BG-THREAD] Step 4: Extracted text. Length={len(text)}", flush=True)
                
                if not text or not text.strip():
                    print(f"[BG-THREAD] ABORT: Extracted text is empty for {request.doc_id}", flush=True)
                    return
                
                inserted_ids = chunk_and_store(doc_id=request.doc_id, text=text)
                print(f"[BG-THREAD] Step 5: Chunking done. {len(inserted_ids)} chunks stored", flush=True)
                
                from ingest.embedder import embed_and_update_chunks
                embedded_count = embed_and_update_chunks(request.doc_id)
                print(f"[BG-THREAD] Step 6: Gemini embeddings updated for {embedded_count} chunks", flush=True)
                
                update_mongo_metadata(request.doc_id, new_hash)
                print(f"[BG-THREAD] Step 7: COMPLETE - Successfully processed {request.doc_id}!", flush=True)
                
            except Exception as bg_e:
                print(f"[BG-THREAD] EXCEPTION during processing of {request.doc_id}: {bg_e}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
        
        thread = threading.Thread(target=process_new_url_bg, daemon=True)
        thread.start()

        return {"message": "Document source added and immediate processing started in background", "doc_id": request.doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document-sources/bulk")
async def add_multiple_document_sources(request: BulkDocSourceRequest):
    try:
        from database.mongo_connection import get_db_connection
        client = get_db_connection()
        db = client["rams_db"]
        
        for source in request.sources:
            db["document_sources"].update_one(
                {"doc_id": source.doc_id},
                {"$set": {
                    "title": source.title,
                    "url": source.url
                }},
                upsert=True
            )
            
        import threading
        def process_multiple_urls_bg(sources):
            try:
                from database.mongo import fetch_document_content, update_mongo_metadata
                from ingest.loader import extract_text
                from ingest.chunker import chunk_and_store
                
                for source in sources:
                    print(f"Background processing immediately started for new URL: {source.url}")
                    latest_doc = fetch_document_content(source.doc_id, source.url)
                    new_hash = latest_doc.get("content_hash")
                    
                    if new_hash != "error":
                        text = extract_text(latest_doc)
                        chunk_and_store(doc_id=source.doc_id, text=text)
                        update_mongo_metadata(source.doc_id, new_hash)
                        print(f"Successfully processed {source.doc_id}")
                
                from ingest.embedder import embed_and_update_all
                embedded_count = embed_and_update_all()
                print(f"Successfully updated Gemini embeddings for {embedded_count} chunk(s)!")
                
            except Exception as bg_e:
                print(f"Error during bulk background processing: {bg_e}")
                
        thread = threading.Thread(target=process_multiple_urls_bg, args=(request.sources,), daemon=True)
        thread.start()
        
        return {
            "message": f"{len(request.sources)} document sources added and bulk processing started", 
            "doc_ids": [s.doc_id for s in request.sources]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-sources")
async def list_document_sources():
    try:
        from database.mongo_connection import get_db_connection
        client = get_db_connection()
        db = client["rams_db"]
        sources = list(db["document_sources"].find({}, {"_id": 0}))
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document-sources/{doc_id}")
async def delete_document_source(doc_id: str):
    """Permanently deletes a document link, its metadata, and its chunks/embeddings."""
    try:
        from database.mongo_connection import get_db_connection
        client = get_db_connection()
        db = client["rams_db"]
        
        # 1. Remove from document sources list
        result = db["document_sources"].delete_one({"doc_id": doc_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # 2. Remove from metadata tracker
        db["metadata"].delete_one({"doc_id": doc_id})
        
        # 3. Remove all MongoDB chunks
        db["chunks"].delete_many({"doc_id": doc_id})
        
        return {"message": f"Document '{doc_id}' and all its embeddings/chunks were permanently deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Make sure this runs on a different port or the same port depending on your needs.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
