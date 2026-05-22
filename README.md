# Rapid Assistance & Mentorship System

## Embeddings

RAMS uses Gemini embeddings directly from the API process. Set these environment
variables before starting the app:

- `GEMINI_API_KEY`: required Google AI Studio / Gemini API key.
- `GEMINI_EMBEDDING_MODEL`: optional, defaults to `gemini-embedding-001`.
- `GEMINI_EMBEDDING_DIMENSIONS`: optional, defaults to `384` so the existing
  MongoDB Atlas vector index from `all-MiniLM-L6-v2` can keep working.
- `GEMINI_NORMALIZE_TRUNCATED_001`: optional, defaults to `true` for non-3072
  `gemini-embedding-001` vectors.

Chunk creation and document ingestion now start Gemini embedding jobs locally, so
the GitHub Actions embedder workflow and the separate sentence-transformers
dependency file are no longer needed.

To replace all existing sentence-transformer embeddings with Gemini embeddings (overwrite all),
run:

```bash
python scripts/run_embedder.py --force
```

Or call `POST /embed/all` with:

```json
{"force": true}
```

To **only** generate embeddings for new chunks that are currently missing them, run the separate script:

```bash
python scripts/embed_missing.py
```

Or call the dedicated endpoint on the deployed version: `POST /embed/missing`.
