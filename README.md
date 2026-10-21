# MemBrain — Enterprise Neural Cortex

**Ending corporate amnesia through semantic knowledge graphs.**

MemBrain captures unstructured team knowledge (meeting notes, decisions, SOPs, voice notes, files), indexes it with a self-hosted hybrid retrieval engine, answers natural-language questions over it, and visualizes how memories relate in an interactive force-directed knowledge graph.

> Originally built for the AlphaNimble Hackathon (March 2026); now being developed into a production-grade system. See [History](#history) and the [Roadmap](#roadmap).

## Features

- **Multi-modal capture:** text, voice (on-device speech-to-text via the Web Speech API), and file import
- **Hybrid semantic search:** dense vector search (pgvector + local BGE embeddings) fused with Postgres full-text search using Reciprocal Rank Fusion
- **Chunked indexing:** long documents are split into overlapping chunks, so a fact buried deep in a long note is still retrievable
- **Grounded Q&A with citations:** answers come only from your memories, and every claim links to the exact passage it came from (Claude with the Citations API)
- **Semantic knowledge graph:** links between memories come from embedding similarity computed in Postgres, drawn by a hand-written force-directed physics simulation on Canvas (stronger links pull nodes closer)
- **Forgetting-curve memory model:** retention decays exponentially. Every use (a manual review, or being cited in an AI answer) makes a memory last longer, with spaced-repetition-style gains. Fading knowledge is surfaced for review, and memories in active use get a small ranking boost
- **Offline-tolerant UI:** cached memories stay visible, and memories captured offline sync automatically when the backend returns
- **Local-first retrieval:** embeddings and search run locally on CPU (ONNX). Only answer generation calls an external model, and the app works without it

## Architecture

```mermaid
flowchart LR
    UI["Frontend<br/>HTML/CSS/JS, Canvas graph"] -->|REST| API["FastAPI"]
    API --> EMB["fastembed<br/>BGE-small (ONNX, local)"]
    API -->|"/api/ask"| LLM["Claude<br/>Citations API"]
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
4. A light forgetting-curve prior is applied (see [How memory decay works](#how-memory-decay-works)).
5. Each result reports *why* it matched (`matched_by`, `vector_rank`, `keyword_rank`, `vector_similarity`, `relevance`, `retention`). `mode` can be set to `vector` or `keyword` to compare retrievers.

### How answers work

`POST /api/ask` (`app/routers/ask.py`, `app/services/answers.py`):
1. Runs the hybrid search above and takes the top memories (6 by default).
2. Sends each memory to Claude as a separate plain-text **document with citations enabled**. Long memories are trimmed to a window around the retrieved chunk.
3. The system prompt restricts answers to those documents, asks the model to say when they don't contain the answer, and treats memory text as data rather than instructions, which guards against prompt injection from stored content.
4. The API returns text blocks with `char_location` citations that point at exact spans of the source documents. The backend turns them into `[n]` markers and a list of `{source, memory_id, title, cited_text}`. Because the citations are structured API output, never parsed from free text, every quote is guaranteed to exist in the cited memory.
5. An answer with no citations is flagged `grounded: false`. Without credentials, or if the model call fails, the endpoint still returns the retrieved memories with a `notice`, so the UI degrades instead of erroring.

### How memory decay works

`app/services/decay.py`, a simplified version of the model spaced-repetition systems like Anki/FSRS use:

- **Retention** follows Ebbinghaus' curve: `R = exp(-t / S)`, where `t` is days since last use and `S` is the memory's *stability* in days. New memories start at `S = 10`, so retention halves in about 7 days.
- **Every use is a review:** `S ← S × (1.1 + 2 × (1 − R))`, capped at 365 days. This is the spacing effect: re-reviewing something seen a minute ago gives only the base 1.1× gain, while rescuing a nearly forgotten memory roughly triples its stability.
- **What counts as a use:** the Reinforce button, and **being cited in an AI answer**. Knowledge the team actually relies on stays fresh automatically. Search results alone don't count, which avoids a rich-get-richer loop where top results reinforce themselves.
- **Ranking:** `score = relevance × (1 − w + w × R)` with `w = 0.15`, applied to a pool of 3×k candidates. Relevance dominates (a stale but clearly relevant memory still wins), but between near-equal matches the one in active use ranks first. Each result reports `relevance` and `retention` separately, so the effect is transparent.
- Memories below 40% retention appear under **Fading** in the Memory Decay view, with their current half-life.

### How the knowledge graph works

`GET /api/graph` (`app/services/graph.py`):
1. Every memory stores a document-level embedding: the mean of its chunk vectors, in its own HNSW index.
2. A single SQL query with a `CROSS JOIN LATERAL` finds each memory's *k* nearest neighbours. The inner `ORDER BY embedding <=> ...` lets the HNSW index serve every lookup.
3. Edges are selected **by rank, not by absolute threshold**. Each memory links to its nearest neighbour, plus any near-ties within 0.01 of that best score, and nothing below a 0.5 floor. Directed picks are merged into undirected edges, and shared tags are reported on each edge.

**Why rank-based?** I calibrated on 12 labelled memories in 4 topics (infra, sales, product, people) with the production embedding model:

| Rule | Edges | On-topic | Isolated |
|---|---|---|---|
| Absolute threshold ≥ 0.7 | 0 | — | 12 |
| Top-2 neighbours | 17 | 53% | 0 |
| Mutual top-2 | 7 | 71% | 2 |
| **Nearest + near-ties (chosen)** | **10** | **70%** | **0** |

Raw cosine scores from a small embedding model fall in a narrow band (0.42–0.66 here) where same-topic and cross-topic pairs overlap, so any fixed cutoff either drops every edge or links unrelated notes. Mean-centering the vectors didn't help either: top-1 precision went from 10/12 to 8/12. Ranking neighbours is robust to that compression. Several "cross-topic" links on inspection were real relationships (a delayed launch ↔ the hiring plan that explains it). The graph builds in about 50 ms for this set.

### API

| Endpoint | Method | Description |
|---|---|---|
| `/api/memories` | POST | Chunk, embed and store a memory |
| `/api/memories` | GET | List memories (`limit`, `offset`) |
| `/api/memories/{id}` | DELETE | Delete a memory and its chunks |
| `/api/memories/{id}/reinforce` | POST | Record a review (grows stability per the forgetting-curve model) |
| `/api/search` | POST | Hybrid search: `{ "query": "...", "k": 6, "mode": "hybrid" \| "vector" \| "keyword" }` |
| `/api/ask` | POST | Cited answer from your memories: `{ "question": "..." }` |
| `/api/graph` | GET | Knowledge graph: nodes and similarity-weighted edges (`neighbors`, `margin`, `min_similarity` optional) |
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

To enable AI answers in **Ask Anything**, put `ANTHROPIC_API_KEY=...` in a `.env` file at the repo root (see [`.env.example`](.env.example)). Everything else works without it.

## Configuration

Every setting has a working local default. Override them through environment variables or `.env` (see [`.env.example`](.env.example)).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://membrain:membrain@localhost:5432/membrain` | Postgres connection |
| `WORKSPACE` | `engineering` | Workspace that all reads and writes are scoped to |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model (384-dim) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking parameters (characters) |
| `RRF_K` / `CANDIDATE_POOL` | `60` / `50` | Fusion constant and per-retriever candidate count |
| `RETENTION_WEIGHT` | `0.15` | Weight of the forgetting-curve prior in ranking (`0` = pure relevance) |
| `CORS_ORIGINS` | `["http://localhost:8000","http://127.0.0.1:8000"]` | Allowed browser origins (JSON list) |
| `ANTHROPIC_API_KEY` | — | Enables generated answers; without it, Ask Anything lists the closest memories |
| `ANSWER_MODEL` / `ANSWER_EFFORT` | `claude-opus-5-5` / `low` | Model and effort level used for answers |
| `ANSWER_SOURCES` / `ANSWER_MAX_SOURCE_CHARS` | `6` / `6000` | Memories sent per question, and the per-memory size cap |
| `GRAPH_NEIGHBORS` / `GRAPH_MARGIN` / `GRAPH_MIN_SIMILARITY` | `3` / `0.01` / `0.5` | Graph edge selection (see above) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest          # unit tests always run; API tests use the docker compose Postgres
ruff check .
```

The API tests run against a real Postgres (a separate `membrain_test` database, created automatically) with a deterministic fake embedder and a fake answerer, so they're fast, free, and don't need the model or an API key. They are skipped if Postgres isn't reachable.

## Project Structure

```
backend/
  app/
    main.py            # app factory, lifespan, health
    config.py          # pydantic-settings
    models.py          # Memory, Chunk (pgvector, tsvector)
    schemas.py         # request/response models
    routers/           # memories, search, ask, graph
    services/
      chunking.py      # paragraph/sentence-aware chunker
      embeddings.py    # Embedder protocol + fastembed implementation
      search.py        # vector + keyword retrieval, RRF fusion
      answers.py       # grounded answers with citations (Claude)
      graph.py         # kNN graph over memory embeddings, rank-based edges
      memories.py      # CRUD and reinforcement
      decay.py         # forgetting-curve model (retention, spaced-repetition stability)
  alembic/             # migrations
  tests/
frontend/index.html    # UI and canvas graph engine
docker-compose.yml
```

## Tech Stack

- **Frontend:** HTML5, CSS3 (glassmorphism), vanilla JavaScript, Canvas API (custom force-directed graph), Web Speech API
- **Backend:** Python, FastAPI, Pydantic v2, SQLAlchemy 2 (async) + asyncpg, Alembic
- **AI:** Claude (Anthropic Python SDK, Citations API)
- **Retrieval:** PostgreSQL 16, pgvector (HNSW), Postgres full-text search, fastembed (BGE-small, ONNX)
- **Tooling:** pytest, ruff, Docker Compose

## History

MemBrain began as a hackathon entry, where storage, vector search and answer generation were delegated to the hosted **Membrain API** provided by the organizers. After the event that hosted service was taken down, which left the original app unable to store or search anything.

The retrieval layer was then rebuilt in-house: local embeddings, pgvector, Postgres full-text search, and rank fusion. The app no longer depends on any third-party AI service. The hackathon submission is preserved under the `hackathon-v1` git tag.

## Roadmap

- [x] Hardening: input validation, async I/O, CORS lockdown, XSS fixes, config via env
- [x] Self-hosted retrieval: local embeddings, pgvector, hybrid full-text + vector search with RRF
- [x] LLM answers with source citations
- [x] Server-side knowledge graph (embedding-similarity edges)
- [x] Forgetting-curve decay model used in ranking
- [ ] PDF/DOCX ingestion via a background worker
- [ ] Containerized API, GitHub Actions CI
- [ ] Auth and real multi-tenant workspaces
- [ ] Retrieval quality evaluation (recall@k / MRR) and live demo deployment

## Team

- **Sujal:** full-stack architecture and graph physics engine
- **Yashica:** UI/UX and brand identity
- **Melvin:** backend integration and data strategy
