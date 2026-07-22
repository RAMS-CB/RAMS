import sys
import os
from fastapi import FastAPI, HTTPException, Depends, status, Header
from pydantic import BaseModel
from typing import List, Optional


# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Sanity client helpers
from database.sanity_client import query_sanity, mutate_sanity, ping_sanity

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
        print("Initializing FastAPI Server with Sanity Content Lake...")
        ping_sanity()
        
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

# Import authentication modules
from datetime import datetime, timedelta
from models.user import (
    RegisterRequest, GoogleAuthRequest, LoginRequest, TokenResponse,
    RefreshTokenRequest, UserResponse, UserRole
)
from database.users import (
    get_user_by_email, get_user_by_username, create_user,
    update_user_refresh_token, get_user_by_id, delete_user_by_id,
    get_all_users
)
from services.auth import (
    hash_password, verify_password, hash_refresh_token,
    create_access_token, create_refresh_token, decode_token,
    verify_google_token, get_current_user, get_current_admin_user,
    get_current_super_admin_user, create_pin_token, verify_pin_token
)



@app.post("/auth/google")
async def google_auth(req: GoogleAuthRequest):
    # Verify the Google ID token
    google_user = verify_google_token(req.id_token)
    email = google_user.get("email")
    full_name = google_user.get("name", "Google User")
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token does not contain email address"
        )
        
    # Check if user already exists
    user = get_user_by_email(email)
    if not user:
        return {
            "registered": False,
            "email": email,
            "name": full_name,
            "id_token": req.id_token
        }
        
    # Promote 'rams.cb.0429@gmail.com' to super_admin automatically if not already set
    if email.strip().lower() == "rams.cb.0429@gmail.com":
        if user.get("role") != UserRole.SUPER_ADMIN.value:
            from database.users import update_user_role
            update_user_role(user["_id"], UserRole.SUPER_ADMIN.value)
            user["role"] = UserRole.SUPER_ADMIN.value
        
    # Generate tokens
    access_token = create_access_token(data={"sub": user["_id"], "role": user["role"]})
    refresh_token_jwt, refresh_token_val = create_refresh_token(data={"sub": user["_id"]})
    
    # Store hashed refresh token in MongoDB
    hashed_rt = hash_refresh_token(refresh_token_val)
    update_user_refresh_token(user["_id"], hashed_rt)
    
    # Track last_login for admin stats
    try:
        mutate_sanity([{
            "patch": {
                "id": user["_id"],
                "set": {"last_login": datetime.utcnow().isoformat()}
            }
        }])
    except Exception:
        pass
        
    return {
        "registered": True,
        "access_token": access_token,
        "refresh_token": refresh_token_jwt
    }

@app.post("/auth/register")
async def register(req: RegisterRequest):
    # Verify the Google ID token
    google_user = verify_google_token(req.id_token)
    email = google_user.get("email")
    full_name = google_user.get("name", "Google User")
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token does not contain email address"
        )
        
    # Check if user already exists
    user = get_user_by_email(email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already registered. Please login."
        )

    # Derive username from email
    base_username = email.split("@")[0]
    import re as re_username
    base_username = re_username.sub(r'[^a-zA-Z0-9]', '', base_username)
    username = base_username
    counter = 1
    while get_user_by_username(username):
        username = f"{base_username}{counter}"
        counter += 1

    # Determine degree
    domain = email.split("@")[-1].lower() if "@" in email else ""
    degree = None
    if domain.endswith("iitm.ac.in"):
        if domain.startswith("ds."):
            degree = "data science and applications"
        elif domain.startswith("es."):
            degree = "electronic systems"
        elif domain.startswith("mg."):
            degree = "management and data science"
        elif domain.startswith("ae."):
            degree = "aeronautics and space technology"
        else:
            degree = "faculty"

    role = UserRole.USER.value
    if email.strip().lower() == "rams.cb.0429@gmail.com":
        role = UserRole.SUPER_ADMIN.value

    user_dict = {
        "username": username,
        "email": email,
        "full_name": full_name,
        "hashed_password": None,
        "role": role,
        "hashed_refresh_token": None,
        "created_at": datetime.utcnow(),
        "profession": req.profession,
        "level": req.level,
        "faculty_type": req.faculty_type,
        "age": req.age,
        "degree": degree or req.degree,
        "source": req.source,
        "interested_programme": req.interested_programme
    }
    user = create_user(user_dict)
    
    # Generate tokens
    access_token = create_access_token(data={"sub": user["_id"], "role": user["role"]})
    refresh_token_jwt, refresh_token_val = create_refresh_token(data={"sub": user["_id"]})
    
    # Store hashed refresh token in MongoDB
    hashed_rt = hash_refresh_token(refresh_token_val)
    update_user_refresh_token(user["_id"], hashed_rt)
    
    return {
        "registered": True,
        "access_token": access_token,
        "refresh_token": refresh_token_jwt
    }

@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(req: RefreshTokenRequest):
    # Decode the refresh token (validates signature & expiration)
    payload = decode_token(req.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type (refresh token required)"
        )
        
    user_id = payload.get("sub")
    token_val = payload.get("jti")
    
    if not user_id or not token_val:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload"
        )
        
    user = get_user_by_id(user_id)
    if not user or not user.get("hashed_refresh_token"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or logged out"
        )
        
    # Check if the refresh token matches the one in DB
    hashed_rt = hash_refresh_token(token_val)
    if user["hashed_refresh_token"] != hashed_rt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token"
        )
        
    # Generate new tokens
    access_token = create_access_token(data={"sub": user["_id"], "role": user["role"]})
    refresh_token_jwt, new_refresh_token_val = create_refresh_token(data={"sub": user["_id"]})
    
    # Store new hashed refresh token
    new_hashed_rt = hash_refresh_token(new_refresh_token_val)
    update_user_refresh_token(user["_id"], new_hashed_rt)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token_jwt
    }

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    update_user_refresh_token(current_user["_id"], None)
    return {"message": "Successfully logged out"}

@app.delete("/auth/account")
async def delete_account(current_user: dict = Depends(get_current_user)):
    success = delete_user_by_id(current_user["_id"])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete account")
    return {"message": "Account successfully deleted"}

@app.post("/auth/heartbeat")
async def heartbeat(current_user: dict = Depends(get_current_user)):
    """Lightweight heartbeat — get_current_user already updates last_active."""
    return {"status": "ok"}

class PinRequest(BaseModel):
    pin: str

@app.post("/admin/verify-pin")
async def verify_admin_pin(req: PinRequest, current_super_admin: dict = Depends(get_current_super_admin_user)):
    expected_pin = os.getenv("SUPER_ADMIN_PIN")
    if not expected_pin:
        raise HTTPException(status_code=500, detail="Super admin PIN not configured")
    if req.pin != expected_pin:
        raise HTTPException(status_code=403, detail="Invalid PIN")
    
    pin_token = create_pin_token()
    return {"pin_token": pin_token}

@app.get("/admin/users", response_model=List[UserResponse])
async def get_admin_users(
    x_pin_token: Optional[str] = Header(None),
    current_super_admin: dict = Depends(get_current_super_admin_user)
):
    users_data = get_all_users()
    is_unlocked = verify_pin_token(x_pin_token)
    
    result = []
    for u in users_data:
        if not is_unlocked:
            # Mask data
            if u.get("email"):
                parts = u["email"].split("@")
                if len(parts) == 2:
                    u["email"] = f"{parts[0][0]}***@{parts[1]}"
                else:
                    u["email"] = "***"
            
            # Keep name visible as requested, but hash/mask everything else
            u["profession"] = "***" if u.get("profession") else None
            u["level"] = "***" if u.get("level") else None
            u["faculty_type"] = "***" if u.get("faculty_type") else None
            u["age"] = None
            u["degree"] = "***" if u.get("degree") else None
            u["source"] = "***" if u.get("source") else None
            u["interested_programme"] = "***" if u.get("interested_programme") else None
            
        result.append(UserResponse(**u))
        
    return result

class UpdateUserRoleRequest(BaseModel):
    role: str

@app.put("/admin/users/{user_id}/role")
async def update_user_role_route(
    user_id: str,
    req: UpdateUserRoleRequest,
    current_super_admin: dict = Depends(get_current_super_admin_user)
):
    if req.role not in [UserRole.ADMIN.value, UserRole.USER.value]:
        raise HTTPException(status_code=400, detail="Invalid role. Must be 'admin' or 'user'")
        
    target_user = get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Cannot change the super admin's role
    if target_user.get("email", "").strip().lower() == "rams.cb.0429@gmail.com":
        raise HTTPException(status_code=403, detail="Cannot modify Super Admin role")
        
    from database.users import update_user_role
    success = update_user_role(user_id, req.role)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update user role")
        
    return {"message": f"Successfully updated user role to {req.role}", "user_id": user_id, "role": req.role}


# Example Request Model
class QueryRequest(BaseModel):
    query: str
    model_provider: Optional[str] = "gemini"

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
async def ask_question(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    try:
        from services.retrieval import retrieve_documents
        from services.llm_service import generate_answer
        
        # 1. Retrieve relevant chunks from the database
        chunks = retrieve_documents(request.query)
        
        # 2. Extract out the actual text and scores to send to the frontend in a readable format
        formatted_context = []
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            idx = chunk.get("chunk_index", "N/A")
            score = chunk.get("score", 0.0)
            text = chunk.get("text_content", "").strip()
            formatted_context.append(f"--- Doc: {doc_id} | Chunk: {idx} | Score: {score:.3f} ---\n{text}")
            
        context_string = "\n\n".join(formatted_context)
        
        # 3. Generate answer using chosen model based on the retrieved context
        final_answer = generate_answer(request.query, chunks, request.model_provider)
        
        # Append the formatted context directly to the answer message so it displays in the frontend chat
        final_answer += "\n\n### Retrieved Context Sources\n" + context_string
        
        # Log the query for admin stats
        try:
            mutate_sanity([{
                "create": {
                    "_type": "query_log",
                    "user_id": current_user.get("_id"),
                    "query": request.query,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }])
        except Exception:
            pass
        
        return {
            "question": request.query,
            "answer": final_answer,
            "context": context_string
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk")
async def create_chunks(request: ChunkRequest, current_user: dict = Depends(get_current_user)):
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
async def generate_embeddings_endpoint(request: EmbedRequest, current_super_admin: dict = Depends(get_current_super_admin_user)):
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
async def generate_all_embeddings_endpoint(request: EmbedAllRequest, current_super_admin: dict = Depends(get_current_super_admin_user)):
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
async def generate_missing_embeddings_endpoint(current_super_admin: dict = Depends(get_current_super_admin_user)):
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
async def update_metadata_endpoint(request: MetadataUpdateRequest, current_super_admin: dict = Depends(get_current_super_admin_user)):
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
async def add_document_source(request: DocSourceRequest, current_super_admin: dict = Depends(get_current_super_admin_user)):
    try:
        mutate_sanity([{
            "createOrReplace": {
                "_id": f"src_{request.doc_id}",
                "_type": "document_source",
                "doc_id": request.doc_id,
                "title": request.title,
                "url": request.url,
                "created_at": datetime.utcnow().isoformat()
            }
        }])
        
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
async def add_multiple_document_sources(request: BulkDocSourceRequest, current_super_admin: dict = Depends(get_current_super_admin_user)):
    try:
        mutations = [{
            "createOrReplace": {
                "_id": f"src_{source.doc_id}",
                "_type": "document_source",
                "doc_id": source.doc_id,
                "title": source.title,
                "url": source.url,
                "created_at": datetime.utcnow().isoformat()
            }
        } for source in request.sources]
        mutate_sanity(mutations)
            
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
async def list_document_sources(current_user: dict = Depends(get_current_user)):
    try:
        sources = query_sanity('*[_type == "document_source"]{doc_id, title, url}') or []
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document-sources/{doc_id}")
async def delete_document_source(doc_id: str, current_super_admin: dict = Depends(get_current_super_admin_user)):
    """Permanently deletes a document link, its metadata, and its chunks/embeddings."""
    try:
        src_id = query_sanity('*[_type == "document_source" && doc_id == $doc_id][0]._id', {"doc_id": doc_id})
        if not src_id:
            raise HTTPException(status_code=404, detail="Document not found")
            
        mutations = [{"delete": {"id": src_id}}, {"delete": {"id": f"meta_{doc_id}"}}]
        
        chunk_ids = query_sanity('*[_type == "chunk" && doc_id == $doc_id]._id', {"doc_id": doc_id}) or []
        for cid in chunk_ids:
            mutations.append({"delete": {"id": cid}})
            
        mutate_sanity(mutations)
        return {"message": f"Document '{doc_id}' and all its embeddings/chunks were permanently deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/stats")
async def get_admin_stats(current_super_admin: dict = Depends(get_current_super_admin_user)):
    """Return platform-wide statistics and chart data for the admin stats page."""
    try:
        now = datetime.utcnow()
        today_start_iso = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        week_ago_iso = (now - timedelta(days=7)).isoformat()
        five_min_ago_iso = (now - timedelta(minutes=5)).isoformat()
        
        # Counts using GROQ
        total_users = query_sanity('count(*[_type == "user"])') or 0
        active_now = query_sanity('count(*[_type == "user" && last_active >= $five_min_ago])', {"five_min_ago": five_min_ago_iso}) or 0
        logins_today = query_sanity('count(*[_type == "user" && last_login >= $today_start])', {"today_start": today_start_iso}) or 0
        new_users_week = query_sanity('count(*[_type == "user" && created_at >= $week_ago])', {"week_ago": week_ago_iso}) or 0
        
        questions_today = query_sanity('count(*[_type == "query_log" && timestamp >= $today_start])', {"today_start": today_start_iso}) or 0
        total_questions = query_sanity('count(*[_type == "query_log"])') or 0
        
        total_documents = query_sanity('count(*[_type == "document_source"])') or 0
        total_chunks = query_sanity('count(*[_type == "chunk"])') or 0
        
        # ── Chart data: Questions per day (last 7 days) ──
        questions_per_day = []
        try:
            recent_logs = query_sanity('*[_type == "query_log" && timestamp >= $week_ago]{timestamp}', {"week_ago": week_ago_iso}) or []
            log_days = {}
            for l in recent_logs:
                ts = l.get("timestamp", "")[:10]
                if ts:
                    log_days[ts] = log_days.get(ts, 0) + 1
            for i in range(7):
                d = (now - timedelta(days=6 - i)).strftime("%Y-%m-%d")
                questions_per_day.append({"date": d, "count": log_days.get(d, 0)})
        except Exception:
            questions_per_day = [{"date": (now - timedelta(days=6 - i)).strftime("%Y-%m-%d"), "count": 0} for i in range(7)]
            
        # ── Chart data: New users per day (last 7 days) ──
        users_per_day = []
        try:
            recent_users = query_sanity('*[_type == "user" && created_at >= $week_ago]{created_at}', {"week_ago": week_ago_iso}) or []
            user_days = {}
            for u in recent_users:
                ca = u.get("created_at", "")[:10]
                if ca:
                    user_days[ca] = user_days.get(ca, 0) + 1
            for i in range(7):
                d = (now - timedelta(days=6 - i)).strftime("%Y-%m-%d")
                users_per_day.append({"date": d, "count": user_days.get(d, 0)})
        except Exception:
            users_per_day = [{"date": (now - timedelta(days=6 - i)).strftime("%Y-%m-%d"), "count": 0} for i in range(7)]
            
        # ── Chart data: Profession breakdown ──
        profession_breakdown = []
        try:
            professions = query_sanity('*[_type == "user" && defined(profession)].profession') or []
            counts = {}
            for p in professions:
                if p:
                    counts[p] = counts.get(p, 0) + 1
            sorted_p = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            profession_breakdown = [{"profession": p, "count": c} for p, c in sorted_p]
        except Exception:
            pass
            
        return {
            "total_users": total_users,
            "active_now": active_now,
            "logins_today": logins_today,
            "questions_today": questions_today,
            "total_questions": total_questions,
            "new_users_week": new_users_week,
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "server_time": now.isoformat(),
            "questions_per_day": questions_per_day,
            "users_per_day": users_per_day,
            "profession_breakdown": profession_breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from models.faq import FAQCreate, FAQResponse
import json
import uuid

@app.get("/faq", response_model=List[FAQResponse])
async def get_faqs():
    try:
        faqs = query_sanity('*[_type == "faq"] | order(created_at desc)') or []
        for faq in faqs:
            faq["_id"] = str(faq["_id"])
        return faqs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/faq", response_model=FAQResponse)
async def create_faq(req: FAQCreate, current_super_admin: dict = Depends(get_current_super_admin_user)):
    try:
        faq_id = f"faq_{uuid.uuid4().hex}"
        faq_doc = {
            "_id": faq_id,
            "_type": "faq",
            "question": req.question,
            "answer": req.answer,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": current_super_admin.get("_id")
        }
        mutate_sanity([{"createOrReplace": faq_doc}])
        return faq_doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/faq/{faq_id}")
async def delete_faq(faq_id: str, current_super_admin: dict = Depends(get_current_super_admin_user)):
    try:
        mutate_sanity([{"delete": {"id": faq_id}}])
        return {"message": "FAQ deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/faq/generate")
async def generate_faqs_endpoint(current_super_admin: dict = Depends(get_current_super_admin_user)):
    try:
        from services.llm_service import generate_faqs
        
        recent_queries = query_sanity('*[_type == "query_log"] | order(timestamp desc)[0...50].query') or []
        questions = [q for q in recent_queries if q]
        
        if not questions:
            raise HTTPException(status_code=400, detail="No query history found to generate FAQs")
            
        generated_json_str = generate_faqs(questions, count=5)
        
        clean_json_str = generated_json_str.strip()
        if clean_json_str.startswith("```json"):
            clean_json_str = clean_json_str[7:]
        if clean_json_str.startswith("```"):
            clean_json_str = clean_json_str[3:]
        if clean_json_str.endswith("```"):
            clean_json_str = clean_json_str[:-3]
            
        try:
            faq_list = json.loads(clean_json_str.strip())
        except json.JSONDecodeError:
            print("Failed to decode LLM response:", generated_json_str)
            raise HTTPException(status_code=500, detail="LLM generated invalid JSON")
            
        mutations = []
        for faq_data in faq_list:
            if "question" in faq_data and "answer" in faq_data:
                faq_doc = {
                    "_id": f"faq_{uuid.uuid4().hex}",
                    "_type": "faq",
                    "question": faq_data["question"],
                    "answer": faq_data["answer"],
                    "created_at": datetime.utcnow().isoformat(),
                    "created_by": "system_llm"
                }
                mutations.append({"createOrReplace": faq_doc})
                
        if mutations:
            mutate_sanity(mutations)
            
        return {"message": f"Successfully generated and added {len(mutations)} FAQs."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Make sure this runs on a different port or the same port depending on your needs.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
