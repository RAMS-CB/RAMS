import numpy as np
from database.mongo_connection import get_db_connection
from ingest.embedder import generate_query_embedding

def test():
    db = get_db_connection()['rams_db']
    query = "When is the assignment due?"
    q_emb = generate_query_embedding(query)
    
    # Get top 3 chunks
    chunks = list(db['chunks'].find({"embedding": {"\$exists": True}}))
    for c in chunks:
        c['score'] = np.dot(q_emb, c['embedding']) / (np.linalg.norm(q_emb) * np.linalg.norm(c['embedding']))
        
    chunks.sort(key=lambda x: x['score'], reverse=True)
    top_3 = chunks[:3]
    
    for tc in top_3:
        print(f"\nTop Chunk (index {tc['chunk_index']}, score {tc['score']}): {tc['text_content'][:50]}")
        # fetch 2 ahead
        ahead = [c for c in chunks if c['doc_id'] == tc['doc_id'] and c['chunk_index'] in [tc['chunk_index']+1, tc['chunk_index']+2]]
        for a in ahead:
            print(f"  Ahead Chunk (index {a['chunk_index']}, score {a['score']}): {a['text_content'][:50]}")
            
test()
