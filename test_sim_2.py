import numpy as np
from database.sanity_client import query_sanity
from ingest.embedder import generate_query_embedding

def test():
    query = "When is the assignment due?"
    q_emb = generate_query_embedding(query)
    
    chunks = query_sanity('*[_type == "chunk" && defined(embedding)]') or []
    if not chunks:
        print("No chunks found with embeddings.")
        return
        
    for c in chunks:
        emb = c['embedding']
        c['score'] = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        
    chunks.sort(key=lambda x: x['score'], reverse=True)
    top_1 = chunks[0]
    
    print(f"Top Score: {top_1['score']}")
    
    c_list = [c['score'] for c in chunks]
    print(f"Scores > 0.60: {sum(1 for s in c_list if s > 0.60)}")
    print(f"Scores > 0.65: {sum(1 for s in c_list if s > 0.65)}")
    print(f"Scores > 0.68: {sum(1 for s in c_list if s > 0.68)}")
    print(f"Scores > 0.70: {sum(1 for s in c_list if s > 0.70)}")

test()
