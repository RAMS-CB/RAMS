import uuid
from typing import Optional, Dict, Any
from database.sanity_client import query_sanity, mutate_sanity

def _format_user(user: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not user:
        return None
    # Ensure _id is a string
    if "_id" in user:
        user["_id"] = str(user["_id"])
    return user

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        query = '*[_type == "user" && _id == $user_id][0]'
        user = query_sanity(query, {"user_id": user_id})
        return _format_user(user)
    except Exception as e:
        print(f"Error in get_user_by_id: {e}")
        return None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    try:
        query = '*[_type == "user" && email == $email][0]'
        user = query_sanity(query, {"email": email.strip().lower()})
        return _format_user(user)
    except Exception as e:
        print(f"Error in get_user_by_email: {e}")
        return None

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    try:
        query = '*[_type == "user" && username == $username][0]'
        user = query_sanity(query, {"username": username.strip().lower()})
        return _format_user(user)
    except Exception as e:
        print(f"Error in get_user_by_username: {e}")
        return None

def create_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure lowercase email & username for case-insensitivity
    user_data["email"] = user_data["email"].strip().lower()
    user_data["username"] = user_data["username"].strip().lower()
    
    # Generate unique ID for user document if not provided
    doc_id = user_data.get("_id") or f"user_{uuid.uuid4().hex}"
    
    doc = {
        "_id": doc_id,
        "_type": "user",
        **{k: v for k, v in user_data.items() if k != "_id"}
    }
    
    if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
        doc["created_at"] = doc["created_at"].isoformat()
        
    result = mutate_sanity([{"createOrReplace": doc}])
    user_data["_id"] = doc_id
    return user_data

def update_user_refresh_token(user_id: str, hashed_refresh_token: Optional[str]) -> bool:
    try:
        mutation = {
            "patch": {
                "id": user_id,
                "set": {"hashed_refresh_token": hashed_refresh_token} if hashed_refresh_token else {},
                "unset": [] if hashed_refresh_token else ["hashed_refresh_token"]
            }
        }
        mutate_sanity([mutation])
        return True
    except Exception as e:
        print(f"Error updating refresh token: {e}")
        return False

def update_user_role(user_id: str, role: str) -> bool:
    try:
        mutation = {
            "patch": {
                "id": user_id,
                "set": {"role": role}
            }
        }
        mutate_sanity([mutation])
        return True
    except Exception as e:
        print(f"Error updating user role: {e}")
        return False

def delete_user_by_id(user_id: str) -> bool:
    try:
        mutation = {"delete": {"id": user_id}}
        mutate_sanity([mutation])
        return True
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False

def get_all_users() -> list:
    try:
        query = '*[_type == "user"]'
        users = query_sanity(query) or []
        return [_format_user(u) for u in users if u]
    except Exception as e:
        print(f"Error fetching all users: {e}")
        return []

