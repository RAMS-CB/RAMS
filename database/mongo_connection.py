import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Load environment variables
load_dotenv()

def get_db_connection():
    """
    Connects to MongoDB using the MONGO_ID connection string.
    Returns the MongoClient instance.
    """
    mongo_uri = os.getenv("MONGO_ID")
    
    if not mongo_uri:
        raise ValueError("MONGO_ID not found in environment variables. Please check your .env file.")
        
    try:
        # Use certifi to provide the required root certificates for MongoDB Atlas
        client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
        # Test the connection
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

if __name__ == "__main__":
    # Test the connection when the script is run directly
    client = get_db_connection()
    if client:
        print("Connection ready to use!")
