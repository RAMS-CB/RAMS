import os
import sys
import uuid

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set ALLOW_MOCK_GOOGLE to true for test
os.environ["ALLOW_MOCK_GOOGLE"] = "true"

from models.user import UserCreate, UserRole, LoginRequest, GoogleAuthRequest, RefreshTokenRequest
from services.auth import (
    hash_password, verify_password, create_access_token, create_refresh_token,
    decode_token, hash_refresh_token, verify_google_token
)
from database.users import (
    create_user, get_user_by_email, get_user_by_username,
    update_user_refresh_token, get_user_by_id
)

def run_tests():
    print("==================================================")
    print("Starting Auth System Verification Tests...")
    print("==================================================")
    
    # 1. Test Password Hashing
    print("\n--- 1. Testing Password Hashing & Verification ---")
    password = "SuperSecretPassword123"
    hashed = hash_password(password)
    assert hashed != password, "Password hashing failed to change plain text"
    assert verify_password(password, hashed), "Password verification failed"
    assert not verify_password("wrong_password", hashed), "Password verification accepted wrong password"
    print("SUCCESS: Hashing and validation behave correctly.")
    
    # 2. Test User Creation in MongoDB
    print("\n--- 2. Testing MongoDB User Storage ---")
    test_id = str(uuid.uuid4())[:8]
    username = f"testuser_{test_id}"
    email = f"test_{test_id}@example.com"
    full_name = f"Test User {test_id}"
    
    user_in = UserCreate(
        username=username,
        email=email,
        full_name=full_name,
        password=password,
        role=UserRole.USER
    )
    
    user_dict = {
        "username": user_in.username,
        "email": user_in.email,
        "full_name": user_in.full_name,
        "hashed_password": hash_password(user_in.password),
        "role": user_in.role.value,
        "hashed_refresh_token": None
    }
    
    user_record = create_user(user_dict)
    assert user_record["_id"] is not None, "Failed to retrieve inserted user ID"
    user_id = user_record["_id"]
    print(f"Created user with ID: {user_id}")
    
    # Verify lookup
    user_by_email = get_user_by_email(email)
    assert user_by_email is not None, "Could not lookup user by email"
    assert user_by_email["username"] == username, "User details mismatch"
    
    user_by_uname = get_user_by_username(username)
    assert user_by_uname is not None, "Could not lookup user by username"
    assert user_by_uname["email"] == email, "User details mismatch"
    print("SUCCESS: MongoDB CRUD & indexing lookup work correctly.")
    
    # 3. Test JWT Generation & Decode
    print("\n--- 3. Testing JWT Access & Refresh Tokens ---")
    access_token = create_access_token(data={"sub": user_id, "role": user_record["role"]})
    print(f"Generated Access Token: {access_token[:20]}...")
    
    payload = decode_token(access_token)
    assert payload["sub"] == user_id, "Decoded user ID mismatch"
    assert payload["role"] == "user", "Decoded role mismatch"
    assert payload["type"] == "access", "Token type mismatch"
    
    refresh_token_jwt, refresh_token_val = create_refresh_token(data={"sub": user_id})
    print(f"Generated Refresh Token: {refresh_token_jwt[:20]}...")
    print(f"Token random JTI: {refresh_token_val}")
    
    ref_payload = decode_token(refresh_token_jwt)
    assert ref_payload["sub"] == user_id, "Refresh token user ID mismatch"
    assert ref_payload["type"] == "refresh", "Token type mismatch"
    print("SUCCESS: JWT Generation, Signing, and Decoding work correctly.")
    
    # 4. Test Refresh Token Storage and Verification in MongoDB
    print("\n--- 4. Testing Refresh Token Hashing & Database Match ---")
    hashed_rt = hash_refresh_token(refresh_token_val)
    update_success = update_user_refresh_token(user_id, hashed_rt)
    assert update_success, "Failed to update user refresh token in DB"
    
    updated_user = get_user_by_id(user_id)
    assert updated_user["hashed_refresh_token"] == hashed_rt, "Hashed refresh token stored does not match"
    print("SUCCESS: Hashed refresh token saved and fetched correctly.")
    
    # 5. Test Mock Google Token Verification
    print("\n--- 5. Testing Mock Google Auth Token ---")
    mock_info = verify_google_token("mock_google_id_token")
    assert mock_info["email"] == "mockuser@gmail.com", "Mock Google info mismatch"
    print("SUCCESS: Mock Google authentication successfully processed.")
    
    print("\n==================================================")
    print("All Auth tests passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
