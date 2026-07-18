from typing import Optional, Dict, Any
from bson import ObjectId
from database.mongo_connection import get_db_connection

def get_users_collection():
    client = get_db_connection()
    db = client["rams_db"]
    return db["users"]

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        col = get_users_collection()
        user = col.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])
        return user
    except Exception:
        return None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    col = get_users_collection()
    user = col.find_one({"email": email.strip().lower()})
    if user:
        user["_id"] = str(user["_id"])
    return user

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    col = get_users_collection()
    user = col.find_one({"username": username.strip().lower()})
    if user:
        user["_id"] = str(user["_id"])
    return user

def create_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    col = get_users_collection()
    
    # Ensure lowercase email & username for case-insensitivity
    user_data["email"] = user_data["email"].strip().lower()
    user_data["username"] = user_data["username"].strip().lower()
    
    # Insert
    result = col.insert_one(user_data)
    user_data["_id"] = str(result.inserted_id)
    return user_data

def update_user_refresh_token(user_id: str, hashed_refresh_token: Optional[str]) -> bool:
    try:
        col = get_users_collection()
        result = col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"hashed_refresh_token": hashed_refresh_token}}
        )
        return result.modified_count > 0
    except Exception:
        return False

def update_user_role(user_id: str, role: str) -> bool:
    try:
        col = get_users_collection()
        result = col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": role}}
        )
        return result.modified_count > 0
    except Exception:
        return False

def delete_user_by_id(user_id: str) -> bool:
    try:
        col = get_users_collection()
        result = col.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0
    except Exception:
        return False

def get_all_users() -> list:
    try:
        col = get_users_collection()
        users_cursor = col.find()
        users = []
        for user in users_cursor:
            user["_id"] = str(user["_id"])
            users.append(user)
        return users
    except Exception:
        return []
