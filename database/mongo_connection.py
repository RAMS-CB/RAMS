import os
from dotenv import load_dotenv
from database.sanity_client import ping_sanity, query_sanity, mutate_sanity

load_dotenv()

def get_db_connection():
    """
    Backwards-compatible helper: ping Sanity API.
    """
    ping_sanity()
    return True

if __name__ == "__main__":
    get_db_connection()

