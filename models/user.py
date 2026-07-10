from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from enum import Enum
from datetime import datetime

class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    USER = "user"

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=3, max_length=100) 
    full_name: str = Field(..., min_length=1, max_length=100)
    profession: Optional[str] = None
    level: Optional[str] = None
    faculty_type: Optional[str] = None
    age: Optional[int] = None
    degree: Optional[str] = None
    source: Optional[str] = None
    interested_programme: Optional[str] = None

class RegisterRequest(BaseModel):
    id_token: str
    profession: Optional[str] = None
    level: Optional[str] = None
    faculty_type: Optional[str] = None
    age: Optional[int] = None
    source: Optional[str] = None
    interested_programme: Optional[str] = None

class GoogleAuthRequest(BaseModel):
    id_token: str

class LoginRequest(BaseModel):
    username_or_email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UserResponse(UserBase):
    id: Optional[str] = Field(None, alias="_id")
    role: UserRole
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
