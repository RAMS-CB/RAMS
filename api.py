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

# Startup event to ensure database connection is ready
@app.on_event("startup")
async def startup_event():
    try:
        print("Initializing FastAPI Server...")
        get_db_connection()
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

if __name__ == "__main__":
    import uvicorn
    # Make sure this runs on a different port or the same port depending on your needs.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
