import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main database connection strictly for startup lifecycle
from database.mongo_connection import get_db_connection

app = FastAPI(
    title="RAMS API",
    description="API for the RAMS Document Processing Pipeline",
    version="1.0.0"
)

import threading

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

class MetadataUpdateRequest(BaseModel):
    doc_id: str
    new_hash: str

class DocSourceRequest(BaseModel):
    title: str
    doc_id: str
    url: str

# Example API Route for Retrieval (can connect to services later)
@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Here you would typically call services.retrieval and services.llm_service
    # e.g. answer = process_query(request.query)
    
    # Returning a mock response for now
    if not request.query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    return {
        "question": request.query,
        "answer": "This is a mock answer. Services not fully connected yet."
    }

@app.post("/chunk")
async def create_chunks(request: ChunkRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        if request.store:
            from ingest.chunker import chunk_and_store
            from ingest.embedder import embed_and_update_chunks
            
            inserted_ids = chunk_and_store(
                doc_id=request.doc_id, 
                text=request.text, 
                chunk_size=request.chunk_size, 
                overlap=request.overlap
            )
            
            # Automatically embed if store=True
            embeddings_count = embed_and_update_chunks(doc_id=request.doc_id)
            
            return {
                "message": "Chunks created, stored, and embedded successfully",
                "doc_id": request.doc_id,
                "chunk_count": len(inserted_ids),
                "embeddings_generated": embeddings_count,
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
        from ingest.embedder import embed_and_update_chunks
        
        updated_count = embed_and_update_chunks(doc_id=request.doc_id)
        
        return {
            "message": "Embeddings generated and updated successfully",
            "doc_id": request.doc_id,
            "embeddings_generated": updated_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-faiss")
async def sync_faiss_endpoint():
    """Bulk backfills FAISS vector DB with all chunks currently stored in MongoDB"""
    try:
        from services.retrieval import sync_all_from_mongo
        docs, chunks = sync_all_from_mongo()
        return {
            "message": "Successfully synced MongoDB embeddings to FAISS",
            "documents_processed": docs,
            "total_chunks_synced": chunks
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

@app.post("/document-sources")
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
        return {"message": "Document source added/updated successfully", "doc_id": request.doc_id}
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
    """Permanently deletes a document link, its metadata, its chunks, and its FAISS embeddings."""
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
        
        # 4. Remove all FAISS vectors
        from services.retrieval import delete_vectors_from_faiss
        delete_vectors_from_faiss(doc_id)
        
        return {"message": f"Document '{doc_id}' and all its embeddings/chunks were permanently deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Render and other hosting providers set the PORT environment variable.
    # Default to 8000 for local development if PORT is not set.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
