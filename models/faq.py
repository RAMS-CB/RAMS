from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FAQBase(BaseModel):
    question: str
    answer: str

class FAQCreate(FAQBase):
    pass

class FAQResponse(FAQBase):
    id: str = Field(..., alias="_id")
    created_at: datetime
    created_by: Optional[str] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
