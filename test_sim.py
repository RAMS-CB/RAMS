import numpy as np
from database.sanity_client import query_sanity
from ingest.embedder import generate_query_embedding

def test():
    query = "When is the assignment due?"
    q_emb = generate_query_embedding(query)
    
    chunks = query_sanity('*[_type == "chunk" && defined(embedding)]') or []
    for c in chunks:
        emb = c['embedding']
        c['score'] = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        
    chunks.sort(key=lambda x: x['score'], reverse=True)
    top_3 = chunks[:3]
    
    for tc in top_3:
        print(f"\nTop Chunk (index {tc['chunk_index']}, score {tc['score']}): {tc['text_content'][:50]}")
        ahead = [c for c in chunks if c['doc_id'] == tc['doc_id'] and c['chunk_index'] in [tc['chunk_index']+1, tc['chunk_index']+2]]
        for a in ahead:
            print(f"  Ahead Chunk (index {a['chunk_index']}, score {a['score']}): {a['text_content'][:50]}")
            
test()
