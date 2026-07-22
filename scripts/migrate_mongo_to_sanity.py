import os
import sys
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.sanity_client import mutate_sanity

load_dotenv()

MONGO_URI = os.getenv("MONGO_ID")
if not MONGO_URI:
    print("Error: MONGO_ID not found in .env")
    sys.exit(1)

import re

def clean_id(val: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(val))

def migrate():
    print("Connecting to MongoDB Atlas...")
    try:
        mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = mongo_client["rams_db"]
        mongo_client.admin.command('ping')
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

    BATCH_SIZE = 50

    def batch_mutate(mutations):
        for i in range(0, len(mutations), BATCH_SIZE):
            chunk = mutations[i:i + BATCH_SIZE]
            mutate_sanity(chunk)

    # 1. Migrate Users
    users = list(db["users"].find({}))
    print(f"Migrating {len(users)} users...")
    user_mutations = []
    for u in users:
        doc_id = clean_id(u["_id"])
        doc = {
            "_id": f"user_{doc_id}",
            "_type": "user",
            "username": u.get("username"),
            "email": u.get("email"),
            "full_name": u.get("full_name"),
            "hashed_password": u.get("hashed_password"),
            "role": u.get("role", "user"),
            "hashed_refresh_token": u.get("hashed_refresh_token"),
            "profession": u.get("profession"),
            "level": u.get("level"),
            "faculty_type": u.get("faculty_type"),
            "age": u.get("age"),
            "degree": u.get("degree"),
            "source": u.get("source"),
            "interested_programme": u.get("interested_programme"),
        }
        if u.get("created_at") and hasattr(u["created_at"], "isoformat"):
            doc["created_at"] = u["created_at"].isoformat()
        if u.get("last_active") and hasattr(u["last_active"], "isoformat"):
            doc["last_active"] = u["last_active"].isoformat()
        if u.get("last_login") and hasattr(u["last_login"], "isoformat"):
            doc["last_login"] = u["last_login"].isoformat()

        user_mutations.append({"createOrReplace": doc})
    if user_mutations:
        batch_mutate(user_mutations)
    print(f"Migrated {len(user_mutations)} users to Sanity.")

    # 2. Migrate Document Sources
    sources = list(db["document_sources"].find({}))
    print(f"Migrating {len(sources)} document sources...")
    source_mutations = []
    for s in sources:
        doc_id = s.get("doc_id")
        if not doc_id:
            continue
        safe_id = clean_id(doc_id)
        doc = {
            "_id": f"src_{safe_id}",
            "_type": "document_source",
            "doc_id": doc_id,
            "title": s.get("title", ""),
            "url": s.get("url", "")
        }
        source_mutations.append({"createOrReplace": doc})
    if source_mutations:
        batch_mutate(source_mutations)
    print(f"Migrated {len(source_mutations)} document sources to Sanity.")

    # 3. Migrate Metadata
    meta_docs = list(db["metadata"].find({}))
    print(f"Migrating {len(meta_docs)} metadata documents...")
    meta_mutations = []
    for m in meta_docs:
        doc_id = m.get("doc_id")
        if not doc_id:
            continue
        safe_id = clean_id(doc_id)
        doc = {
            "_id": f"meta_{safe_id}",
            "_type": "doc_metadata",
            "doc_id": doc_id,
            "content_hash": m.get("content_hash", ""),
            "last_updated_timestamp": m.get("last_updated_timestamp")
        }
        meta_mutations.append({"createOrReplace": doc})
    if meta_mutations:
        batch_mutate(meta_mutations)
    print(f"Migrated {len(meta_mutations)} metadata documents to Sanity.")

    # 4. Migrate Chunks
    chunks = list(db["chunks"].find({}))
    print(f"Migrating {len(chunks)} chunks...")
    chunk_mutations = []
    for c in chunks:
        cid = clean_id(c["_id"])
        doc = {
            "_id": f"chunk_{cid}",
            "_type": "chunk",
            "doc_id": c.get("doc_id"),
            "chunk_index": c.get("chunk_index"),
            "text_content": c.get("text_content"),
            "metadata": c.get("metadata", {}),
        }
        if "embedding" in c and c["embedding"]:
            doc["embedding"] = c["embedding"]
        if c.get("created_at") and hasattr(c["created_at"], "isoformat"):
            doc["created_at"] = c["created_at"].isoformat()

        chunk_mutations.append({"createOrReplace": doc})
    if chunk_mutations:
        batch_mutate(chunk_mutations)
    print(f"Migrated {len(chunk_mutations)} chunks to Sanity.")

    # 5. Migrate Query Logs
    logs = list(db["query_logs"].find({}))
    print(f"Migrating {len(logs)} query logs...")
    log_mutations = []
    for l in logs:
        lid = clean_id(l["_id"])
        doc = {
            "_id": f"log_{lid}",
            "_type": "query_log",
            "user_id": clean_id(l.get("user_id")) if l.get("user_id") else None,
            "query": l.get("query"),
        }
        if l.get("timestamp") and hasattr(l["timestamp"], "isoformat"):
            doc["timestamp"] = l["timestamp"].isoformat()
        log_mutations.append({"createOrReplace": doc})
    if log_mutations:
        batch_mutate(log_mutations)
    print(f"Migrated {len(log_mutations)} query logs to Sanity.")

    # 6. Migrate FAQs
    faqs = list(db["faqs"].find({}))
    print(f"Migrating {len(faqs)} FAQs...")
    faq_mutations = []
    for f in faqs:
        fid = clean_id(f["_id"])
        doc = {
            "_id": f"faq_{fid}",
            "_type": "faq",
            "question": f.get("question"),
            "answer": f.get("answer"),
            "created_by": clean_id(f.get("created_by")) if f.get("created_by") else None
        }
        if f.get("created_at") and hasattr(f["created_at"], "isoformat"):
            doc["created_at"] = f["created_at"].isoformat()
        faq_mutations.append({"createOrReplace": doc})
    if faq_mutations:
        batch_mutate(faq_mutations)
    print(f"Migrated {len(faq_mutations)} FAQs to Sanity.")

    print("\nData migration from MongoDB to Sanity completed successfully!")

if __name__ == "__main__":
    migrate()
