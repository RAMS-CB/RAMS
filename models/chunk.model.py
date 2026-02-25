from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChunkModel(BaseModel):
    """
    Pydantic schema for storing document chunks in MongoDB.
    
    Fields:
    - id: Optional string representation of the MongoDB _id.
    - doc_id: The ID of the parent document this chunk belongs to (e.g., 'google_doc_main').
    - chunk_index: The sequential index of this chunk within the document to maintain ordering.
    - text_content: The actual text string of the chunk.
    - mementos: Optional dictionary to store any additional metadata (e.g., hash, source url).
    - created_at: Timestamp when this chunk was created/stored.
    """
    id: Optional[str] = Field(None, alias="_id")
    doc_id: str
    chunk_index: int
    text_content: str
    metadata: Optional[dict] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
