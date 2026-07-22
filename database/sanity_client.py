import os
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

SANITY_PROJECT_ID = os.getenv("SANITY_PROJECT_ID")
SANITY_DATASET = os.getenv("SANITY_DATASET", "production")
SANITY_API_TOKEN = os.getenv("SANITY_API_TOKEN")
SANITY_API_VERSION = os.getenv("SANITY_API_VERSION", "2024-01-01")

def _get_base_url() -> str:
    if not SANITY_PROJECT_ID:
        raise ValueError("SANITY_PROJECT_ID not found in environment variables. Please check your .env file.")
    return f"https://{SANITY_PROJECT_ID}.api.sanity.io/v{SANITY_API_VERSION}/data"

def _get_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if SANITY_API_TOKEN:
        headers["Authorization"] = f"Bearer {SANITY_API_TOKEN}"
    return headers

def query_sanity(query: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Executes a GROQ query against Sanity API.
    Returns the 'result' field of the Sanity response.
    """
    base_url = _get_base_url()
    url = f"{base_url}/query/{SANITY_DATASET}"
    
    payload = {"query": query}
    if params:
        payload["params"] = params
        
    response = requests.post(url, json=payload, headers=_get_headers(), timeout=20)
    
    if response.status_code != 200:
        raise RuntimeError(f"Sanity Query Error (HTTP {response.status_code}): {response.text}")
        
    data = response.json()
    return data.get("result")

def mutate_sanity(mutations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes document mutations (create, createOrReplace, patch, delete) against Sanity API.
    """
    if not SANITY_API_TOKEN:
        raise ValueError("SANITY_API_TOKEN not found in environment variables. Mutations require a write token.")
        
    base_url = _get_base_url()
    url = f"{base_url}/mutate/{SANITY_DATASET}"
    
    payload = {"mutations": mutations}
    response = requests.post(url, json=payload, headers=_get_headers(), timeout=20)
    
    if response.status_code != 200:
        raise RuntimeError(f"Sanity Mutation Error (HTTP {response.status_code}): {response.text}")
        
    return response.json()

def ping_sanity() -> bool:
    """
    Tests connectivity to Sanity API.
    """
    try:
        res = query_sanity("*[_type == 'system.schema'][0]")
        print("Successfully connected to Sanity Content Lake!")
        return True
    except Exception as e:
        print(f"Failed to connect to Sanity: {e}")
        return False

if __name__ == "__main__":
    ping_sanity()
