import numpy as np
from database.mongo_connection import get_db_connection
from ingest.embedder import generate_query_embedding

def test():
    db = get_db_connection()['rams_db']
    query = "When is the assignment due?"
    q_emb = generate_query_embedding(query)
    
    chunks = list(db['chunks'].find({"embedding": {"\$exists": True}}))
    for c in chunks:
        c['score'] = np.dot(q_emb, c['embedding']) / (np.linalg.norm(q_emb) * np.linalg.norm(c['embedding']))
        
    chunks.sort(key=lambda x: x['score'], reverse=True)
    top_1 = chunks[0]
    
    print(f"Top Score: {top_1['score']}")
    
    c_list = [c['score'] for c in chunks]
    print(f"Scores > 0.60: {sum(1 for s in c_list if s > 0.60)}")
    print(f"Scores > 0.65: {sum(1 for s in c_list if s > 0.65)}")
    print(f"Scores > 0.68: {sum(1 for s in c_list if s > 0.68)}")
    print(f"Scores > 0.70: {sum(1 for s in c_list if s > 0.70)}")

test()
