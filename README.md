# MemBrain — Enterprise Neural Cortex

**Ending corporate amnesia through semantic knowledge graphs.**

MemBrain captures unstructured team knowledge (meeting notes, decisions, SOPs, voice notes, files), indexes it with a self-hosted hybrid retrieval engine, answers natural-language questions over it, and visualizes how memories relate in an interactive force-directed knowledge graph.

> Originally built for the AlphaNimble Hackathon (March 2026); now being developed into a production-grade system. See [History](#history) and the [Roadmap](#roadmap).

## Features

- **Multi-modal capture:** text, voice (on-device speech-to-text via the Web Speech API), and file import
- **Hybrid semantic search:** dense vector search (pgvector + local BGE embeddings) fused with Postgres full-text search using Reciprocal Rank Fusion
- **Chunked indexing:** long documents are split into overlapping chunks, so a fact buried deep in a long note is still retrievable
- **Knowledge graph:** a hand-written force-directed physics simulation on Canvas, with zoom and pan
- **Memory decay:** memories lose strength over time unless reinforced, so stale knowledge is surfaced for review
- **Offline-tolerant UI:** cached memories stay visible, and memories captured offline sync automatically when the backend returns
- **No external AI dependency:** embeddings run locally on CPU (ONNX); no API keys or per-request cost

## Architecture

```mermaid
flowchart LR
    UI["Frontend<br/>HTML/CSS/JS, Canvas graph"] -->|REST| API["FastAPI"]
    API --> EMB["fastembed<br/>BGE-small (ONNX, local)"]
    API -->|async SQLAlchemy| PG[("PostgreSQL 16<br/>pgvector + tsvector")]
    UI -.->|offline cache| LS[("localStorage")]
```

### How retrieval works

**Ingestion** (`POST /api/memories`):
1. The body is split into ~1,000-character chunks along paragraph and sentence boundaries, with 150 characters of overlap (`app/services/chunking.py`).
2. Each chunk is embedded together with the memory's title, so every vector carries its document's context.
3. Chunks are stored in a pgvector column with an **HNSW** index (cosine distance). Postgres maintains a weighted `tsvector` (title weighted above body) in a **GIN** index as a generated column.

**Search** (`POST /api/search`, `app/services/search.py`):
1. **Vector retrieval:** nearest chunks to the query embedding, collapsed to the best chunk per memory.
2. **Keyword retrieval:** `websearch_to_tsquery` full-text match, ranked by `ts_rank_cd`.
3. **Fusion:** Reciprocal Rank Fusion, `score(d) = Σ 1 / (k + rank_d)` with `k = 60`. RRF needs no score calibration between retrievers. Vector search catches paraphrases ("why did the site go down?" finds an outage postmortem); keyword search catches exact terms, names and IDs that embeddings blur.
4. Each result reports *why* it matched (`matched_by`, `vector_rank`, `keyword_rank`, `vector_similarity`). `mode` can be set to `vector` or `keyword` to compare retrievers.

### API

| Endpoint | Method | Description |
|---|---|---|
| `/api/memories` | POST | Chunk, embed and store a memory |
| `/api/memories` | GET | List memories (`limit`, `offset`) |
| `/api/memories/{id}` | DELETE | Delete a memory and its chunks |
| `/api/memories/{id}/reinforce` | POST | Boost a memory's strength |
| `/api/search` | POST | Hybrid search: `{ "query": "...", "k": 6, "mode": "hybrid" \| "vector" \| "keyword" }` |
| `/health` | GET | Liveness check, including database connectivity |

Interactive docs are at `http://127.0.0.1:8000/docs`.

## Quickstart

**Prerequisites:** Python 3.10+, Docker, and Chrome or Edge for voice capture.

```bash
git clone https://github.com/Sujalranjan/MemBrain.git
cd MemBrain

docker compose up -d db                   # PostgreSQL 16 + pgvector

python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cd backend
alembic upgrade head                      # create tables and indexes
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**. The first start downloads the embedding model (~130 MB), which is cached afterwards.

## Configuration

Every setting has a working local default. Override them through environment variables or `.env` (see [`.env.example`](.env.example)).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://membrain:membrain@localhost:5432/membrain` | Postgres connection |
| `WORKSPACE` | `engineering` | Workspace that all reads and writes are scoped to |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model (384-dim) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking parameters (characters) |
| `RRF_K` / `CANDIDATE_POOL` | `60` / `50` | Fusion constant and per-retriever candidate count |
| `CORS_ORIGINS` | `["http://localhost:8000","http://127.0.0.1:8000"]` | Allowed browser origins (JSON list) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest          # unit tests always run; API tests use the docker compose Postgres
ruff check .
```

The API tests run against a real Postgres (a separate `membrain_test` database, created automatically) with a deterministic fake embedder, so they're fast and don't need the model. They are skipped if Postgres isn't reachable.

## Project Structure

```
backend/
  app/
    main.py            # app factory, lifespan, health
    config.py          # pydantic-settings
    models.py          # Memory, Chunk (pgvector, tsvector)
    schemas.py         # request/response models
    routers/           # memories, search
    services/
      chunking.py      # paragraph/sentence-aware chunker
      embeddings.py    # Embedder protocol + fastembed implementation
      search.py        # vector + keyword retrieval, RRF fusion
      memories.py      # CRUD and reinforcement
      decay.py         # strength model
  alembic/             # migrations
  tests/
frontend/index.html    # UI and canvas graph engine
docker-compose.yml
```

## Tech Stack

- **Frontend:** HTML5, CSS3 (glassmorphism), vanilla JavaScript, Canvas API (custom force-directed graph), Web Speech API
- **Backend:** Python, FastAPI, Pydantic v2, SQLAlchemy 2 (async) + asyncpg, Alembic
- **Retrieval:** PostgreSQL 16, pgvector (HNSW), Postgres full-text search, fastembed (BGE-small, ONNX)
- **Tooling:** pytest, ruff, Docker Compose

## History

MemBrain began as a hackathon entry, where storage, vector search and answer generation were delegated to the hosted **Membrain API** provided by the organizers. After the event that hosted service was taken down, which left the original app unable to store or search anything.

The retrieval layer was then rebuilt in-house: local embeddings, pgvector, Postgres full-text search, and rank fusion. The app no longer depends on any third-party AI service. The hackathon submission is preserved under the `hackathon-v1` git tag.

## Roadmap

- [x] Hardening: input validation, async I/O, CORS lockdown, XSS fixes, config via env
- [x] Self-hosted retrieval: local embeddings, pgvector, hybrid full-text + vector search with RRF
- [ ] LLM answers with source citations
- [ ] Server-side knowledge graph (embedding-similarity edges)
- [ ] Forgetting-curve decay model used in ranking
- [ ] PDF/DOCX ingestion via a background worker
- [ ] Containerized API, GitHub Actions CI
- [ ] Auth and real multi-tenant workspaces
- [ ] Retrieval quality evaluation (recall@k / MRR) and live demo deployment

## Team

- **Sujal:** full-stack architecture and graph physics engine
- **Yashica:** UI/UX and brand identity
- **Melvin:** backend integration and data strategy
