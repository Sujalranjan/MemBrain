# MemBrain — Enterprise Neural Cortex

[![CI](https://github.com/Sujalranjan/MemBrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujalranjan/MemBrain/actions/workflows/ci.yml)

**Ending corporate amnesia through semantic knowledge graphs.**

MemBrain captures unstructured team knowledge (meeting notes, decisions, SOPs, voice notes, files), indexes it with a self-hosted hybrid retrieval engine, answers natural-language questions over it, and visualizes how memories relate in an interactive force-directed knowledge graph.

> Originally built for the AlphaNimble Hackathon (March 2026); now being developed into a production-grade system. See [History](#history) and the [Roadmap](#roadmap).

## Features

- **Multi-modal capture:** text, voice (on-device speech-to-text via the Web Speech API), and documents (PDF, DOCX, Markdown, text)
- **Background document ingestion:** uploads return immediately; a worker extracts, chunks and embeds them from a Postgres-backed job queue (`FOR UPDATE SKIP LOCKED`), with retries and crash recovery
- **Hybrid semantic search:** dense vector search (pgvector + local BGE embeddings) fused with **BM25** keyword ranking (an inverted index in Postgres) using Reciprocal Rank Fusion. On the public BEIR SciFact benchmark, hybrid raises Recall@5 from 0.765 to 0.806 over vector-only ([evaluation](#retrieval-evaluation))
- **Chunked indexing:** long documents are split into overlapping chunks, so a fact buried deep in a long note is still retrievable
- **Grounded Q&A with citations:** answers come only from your memories, and every claim links to the exact passage it came from (Claude with the Citations API)
- **Semantic knowledge graph:** links between memories come from embedding similarity computed in Postgres, drawn by a hand-written force-directed physics simulation on Canvas (stronger links pull nodes closer)
- **Forgetting-curve memory model:** retention decays exponentially. Every use (a manual review, or being cited in an AI answer) makes a memory last longer, with spaced-repetition-style gains. Fading knowledge is surfaced for review, and memories in active use get a small ranking boost
- **Multi-tenant API keys and rate limiting:** each key belongs to a workspace, and every query, document job and graph is scoped to it. Keys are stored as SHA-256 hashes. Token-bucket limits are enforced atomically in Postgres, so they hold across replicas, with a tighter bucket for the paid `/api/ask`
- **Offline-tolerant UI:** cached memories stay visible, and memories captured offline sync automatically when the backend returns
- **Local-first retrieval:** embeddings and search run locally on CPU (ONNX). Only answer generation calls an external model, and the app works without it

## Architecture

```mermaid
flowchart LR
    UI["Frontend<br/>HTML/CSS/JS, Canvas graph"] -->|REST| API["FastAPI"]
    API --> EMB["fastembed<br/>BGE-small (ONNX, local)"]
    API -->|"/api/ask"| LLM["Claude<br/>Citations API"]
    API -->|async SQLAlchemy| PG[("PostgreSQL 16<br/>pgvector + tsvector<br/>+ job queue")]
    W["Ingestion worker<br/>pypdf, python-docx"] -->|claim jobs| PG
    W --> EMB
    UI -.->|offline cache| LS[("localStorage")]
```

### How retrieval works

**Ingestion** (`POST /api/memories`):
1. The body is split into ~1,000-character chunks along paragraph and sentence boundaries, with 150 characters of overlap (`app/services/chunking.py`).
2. Each chunk is embedded together with the memory's title, so every vector carries its document's context.
3. Chunks are stored in a pgvector column with an **HNSW** index (cosine distance). Postgres maintains a weighted `tsvector` (title weighted above body) as a generated column, and its stemmed terms are written to a `memory_terms` **inverted index** (one posting per memory and term, with term frequency; title occurrences count double) for BM25.

**Search** (`POST /api/search`, `app/services/search.py`):
1. **Vector retrieval:** nearest chunks to the query embedding, collapsed to the best chunk per memory.
2. **Keyword retrieval:** **BM25** computed in SQL over the postings (`k1 = 1.2`, `b = 0.75`). Any query term can match, and IDF makes rare terms count for more than common ones. Queries using web-search syntax (`"exact phrase"`, `OR`, `-exclude`) are also filtered by that expression.
3. **Fusion:** Reciprocal Rank Fusion, `score(d) = Σ w_i / (k + rank_i(d))` with `k = 10` and equal weights, both selected on benchmark train splits (see below). RRF needs no score calibration between retrievers. Vector search catches paraphrases ("why did the site go down?" finds an outage postmortem); keyword search catches exact terms, names and IDs that embeddings blur.
4. A light forgetting-curve prior is applied (see [How memory decay works](#how-memory-decay-works)).
5. Each result reports *why* it matched (`matched_by`, `vector_rank`, `keyword_rank`, `vector_similarity`, `relevance`, `retention`). `mode` can be set to `vector` or `keyword` to compare retrievers.

### How document ingestion works

`POST /api/documents` (`app/routers/documents.py`, `app/services/ingestion.py`, `app/services/extraction.py`):
1. The upload is validated up front: size limit (10 MB), a supported extension, **and a matching file signature** (a `.pdf` must start with `%PDF-`, a `.docx` must be a zip), so a mislabelled file is rejected with `415`/`422` before any work is queued.
2. The file is stored in an `ingestion_jobs` row and the API answers **`202 Accepted`** with a job id. Extraction and embedding a long PDF take seconds, so they don't happen in the request.
3. A worker claims the oldest queued job with `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED LIMIT 1)`. Postgres is the queue, so no Redis or broker is needed, and **any number of workers can run concurrently without processing a job twice**. The worker runs inside the API process by default and wakes immediately on upload; set `INGESTION_WORKER=false` and run `python -m app.worker` to move it into separate processes.
4. Text is extracted (pypdf for PDFs, including empty-password encrypted ones; python-docx for Word, keeping tables in reading order), normalized, then chunked and embedded like any memory. The memory and the job's `succeeded` status are committed **in one transaction**, so a crash never leaves one without the other.
5. Failures are classified. A bad file (corrupt, password-protected, scanned with no text layer) fails at once with a readable error, since retrying can't help. An infrastructure error is requeued and retried up to 3 times. A job whose worker died mid-way is reclaimed after 10 minutes. The upload bytes are deleted once a job finishes.
6. Clients poll `GET /api/documents/jobs/{id}`. The UI shows a live list of recent imports.

### Retrieval evaluation

`backend/eval/` indexes a labelled dataset with the production pipeline (same chunker, embedding model, Postgres indexes and `search()` function), then scores each mode with standard IR metrics. Two public [BEIR](https://github.com/beir-cellar/beir) benchmarks are downloaded on demand, and an in-domain set ships in the repo:

```bash
cd backend
python -m eval.run --dataset scifact      # BEIR SciFact: 5,183 abstracts, 300 test queries
python -m eval.run --dataset nfcorpus     # BEIR NFCorpus: 3,633 medical documents, 323 test queries
python -m eval.run --dataset membrain     # 36 enterprise notes, 48 queries (eval/data/membrain.json)
python -m eval.run --dataset scifact --split train --sweep   # grid-search fusion settings
```

**Results** on held-out **test** splits (fusion settings were chosen on the separate train splits):

| Dataset | Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| SciFact | vector | 0.765 | 0.836 | 0.679 | 0.714 |
| | keyword (BM25) | 0.754 | 0.822 | 0.656 | 0.691 |
| | **hybrid** | **0.806** | **0.876** | **0.701** | **0.740** |
| NFCorpus | vector | 0.119 | 0.157 | 0.524 | 0.335 |
| | keyword (BM25) | 0.128 | 0.155 | 0.526 | 0.325 |
| | **hybrid** | **0.133** | **0.173** | **0.548** | **0.359** |
| In-domain | vector | 0.917 | 0.938 | 0.858 | 0.878 |
| | keyword (BM25) | 0.875 | 0.875 | 0.777 | 0.798 |
| | **hybrid** | **0.979** | **1.000** | **0.860** | **0.894** |

Hybrid beats either retriever alone on every dataset: **+5.3% Recall@5 over vector-only on SciFact** (+4.1 points), +11.3% on NFCorpus (+1.3 points; that dataset has many relevant documents per query, hence low recall@k for every mode) and +6.8% in-domain. In the in-domain set, vector search alone misses 25% of exact-identifier queries (`INC-2291`, `ERR_PAYMENT_DECLINED_51`); hybrid finds all of them, while still finding 96% of paraphrased questions that share no words with their answer. The vector-only SciFact nDCG@10 (0.714) matches the published figure for `bge-small-en-v1.5`, which is a sanity check on the harness. Per-dataset reports and raw numbers are in [`backend/eval/results/`](backend/eval/results/).

**What the evaluation changed.** The first run used Postgres' built-in ranking, and the numbers exposed two problems:

| Keyword retrieval | SciFact keyword R@5 | Hybrid vs vector R@5 |
|---|---:|---:|
| `websearch_to_tsquery` (all terms required) + `ts_rank_cd` | 0.073 | +0.4% |
| Any term + `ts_rank_cd` | 0.439 | **−11.8%** |
| **BM25 over an inverted index (current)** | **0.754** | **+5.3%** |

Requiring every query word matched almost nothing for natural-language questions. Matching any word fixed recall, but `ts_rank_cd` has no notion of term rarity, so the noisy keyword ranking made hybrid *worse* than vector alone. A sweep showed that no fusion weight could rescue it. BM25 (IDF plus length normalization) made keyword search competitive with the embeddings, and only then did fusion pay off. BM25 queries take about 20 to 85 ms at these corpus sizes, against about 8 ms for HNSW vector search.

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

#### How authentication and rate limiting work

`app/deps.py`, `app/services/auth.py`, `app/services/ratelimit.py`:

- **Workspaces are the tenant boundary.** Every memory, chunk, posting and ingestion job carries a workspace, and every query filters on it. With `AUTH_ENABLED=true` the workspace comes from the caller's API key, never from the request, so a client cannot ask for another tenant's data. Cross-tenant reads, deletes, reinforcements, graph views and job lookups are covered by tests.
- **API keys** look like `mb_<prefix>_<secret>`, with 256 random bits in the secret. Only a SHA-256 digest is stored. A fast hash is right here, unlike for passwords: there is nothing to brute-force in a random 256-bit secret, and per-request verification stays cheap. The public prefix finds the row, and the digest comparison is constant-time. Keys are sent as `Authorization: Bearer <key>` or `X-API-Key`, can be revoked instantly, and record when they were last used.
- **Rate limiting** is a token bucket per API key (or per client IP when auth is off). Refill-and-spend is a single `INSERT … ON CONFLICT DO UPDATE … WHERE tokens >= 1` statement, so concurrent requests on one bucket serialize on its row lock. With several API replicas, the last token still can't be spent twice: a test fires 20 concurrent requests at a 5-token bucket and exactly 5 succeed. The table is `UNLOGGED` (buckets are disposable, so the write-ahead log is skipped). Responses carry `X-RateLimit-Limit` / `X-RateLimit-Remaining`; rejections are `429` with `Retry-After`. `/api/ask` has its own, tighter bucket because each call costs a model request.

Keys are managed from the command line:

```bash
python -m app.cli create-key --workspace acme --name "ci pipeline"   # prints the key once
python -m app.cli list-keys
python -m app.cli revoke-key <prefix>
# with Docker: docker compose exec api python -m app.cli create-key --workspace acme --name laptop
```

Auth is off by default so local development needs no setup. When it's on, the UI asks for a key on first load and keeps it in the browser.

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
| `/api/documents` | POST | Upload a PDF/DOCX/text file (multipart `file`, optional `title`, `type`, `tags`); returns `202` with a job |
| `/api/documents/jobs` | GET | Recent ingestion jobs (`limit`) |
| `/api/documents/jobs/{id}` | GET | Job status: `queued`, `processing`, `succeeded` (with `memory_id`) or `failed` (with `error`) |
| `/api/me` | GET | The workspace and key the request is authenticated as |
| `/health` | GET | Liveness check, including database connectivity (public) |

With `AUTH_ENABLED=true`, every `/api/*` route needs an API key (`401` otherwise). All of them are rate limited (`429` with `Retry-After`).

Interactive docs are at `http://127.0.0.1:8000/docs`.

## Quickstart

### With Docker (everything)

```bash
git clone https://github.com/Sujalranjan/MemBrain.git
cd MemBrain
docker compose up --build
```

Open **http://localhost:8000**. Migrations run automatically on startup, and the embedding model is baked into the image, so the container needs no network at runtime.

To require API keys:

```bash
AUTH_ENABLED=true docker compose up --build -d
docker compose exec api python -m app.cli create-key --workspace acme --name laptop
```

To process documents in dedicated worker containers instead of the API process:

```bash
INGESTION_WORKER=false docker compose --profile workers up --build --scale worker=2
```

### For development (API on the host)

**Prerequisites:** Python 3.10+, Docker, and Chrome or Edge for voice capture.

```bash
docker compose up -d db                   # PostgreSQL 16 + pgvector only

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
| `AUTH_ENABLED` | `false` | Require API keys; the key then decides the workspace |
| `WORKSPACE` | `engineering` | Workspace used for every request while auth is disabled |
| `RATE_LIMIT_ENABLED` | `true` | Enforce per-key (or per-IP) token buckets |
| `RATE_LIMIT_PER_MINUTE` / `ASK_RATE_LIMIT_PER_MINUTE` | `120` / `10` | Bucket sizes for all API routes, and the extra bucket for `/api/ask` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model (384-dim) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking parameters (characters) |
| `RRF_K` / `KEYWORD_WEIGHT` / `CANDIDATE_POOL` | `10` / `1.0` / `50` | Fusion constant, keyword-vs-vector weight, and per-retriever candidate count (tuned with `eval/`) |
| `BM25_K1` / `BM25_B` | `1.2` / `0.75` | BM25 term-frequency saturation and length normalization |
| `RETENTION_WEIGHT` | `0.15` | Weight of the forgetting-curve prior in ranking (`0` = pure relevance) |
| `CORS_ORIGINS` | `["http://localhost:8000","http://127.0.0.1:8000"]` | Allowed browser origins (JSON list) |
| `ANTHROPIC_API_KEY` | — | Enables generated answers; without it, Ask Anything lists the closest memories |
| `ANSWER_MODEL` / `ANSWER_EFFORT` | `claude-opus-5-5` / `low` | Model and effort level used for answers |
| `ANSWER_SOURCES` / `ANSWER_MAX_SOURCE_CHARS` | `6` / `6000` | Memories sent per question, and the per-memory size cap |
| `GRAPH_NEIGHBORS` / `GRAPH_MARGIN` / `GRAPH_MIN_SIMILARITY` | `3` / `0.01` / `0.5` | Graph edge selection (see above) |
| `MAX_UPLOAD_BYTES` / `MAX_DOCUMENT_CHARS` | `10485760` / `300000` | Upload size limit, and the limit on extracted text per document |
| `INGESTION_WORKER` | `true` | Run the ingestion worker inside the API process; set `false` when running `python -m app.worker` separately |
| `INGESTION_MAX_ATTEMPTS` / `INGESTION_STALE_SECONDS` | `3` / `600` | Retries for transient failures, and when an abandoned job is reclaimed |
| `LOG_LEVEL` | `INFO` | Logging level |

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest          # unit tests always run; API tests use the docker compose Postgres
ruff check .
pytest --cov=app            # coverage report
../scripts/smoke_test.sh    # end-to-end check against a running stack
```

The API tests run against a real Postgres (a separate `membrain_test` database, created automatically) with a deterministic fake embedder and a fake answerer, so they're fast, free, and don't need the model or an API key. They are skipped if Postgres isn't reachable (CI sets `REQUIRE_TEST_DATABASE=1` so they fail instead). Ingestion tests generate real PDF and DOCX files, and simulate crashed workers and flaky embedders to cover reclaiming, retries and giving up.

### Continuous integration

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push to `main` and every pull request:

| Job | What it checks |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `test` | The full pytest suite against a real pgvector Postgres service, on Python 3.10 and 3.12, with a coverage report in the job summary (about 95%) |
| `docker` | Builds the production image (layer-cached), starts the full stack with `docker compose up --wait`, and runs [`scripts/smoke_test.sh`](scripts/smoke_test.sh) with the **real** embedding model: store, search, upload a document, and wait for the worker to index it |

Dependabot keeps pip packages, GitHub Actions and the Docker base image up to date.

## Project Structure

```
backend/
  app/
    main.py            # app factory, lifespan, health
    deps.py            # auth principal, workspace scoping, rate-limit dependencies
    cli.py             # API key management (python -m app.cli)
    worker.py          # standalone ingestion worker (python -m app.worker)
    config.py          # pydantic-settings
    models.py          # Memory, Chunk, MemoryTerm, IngestionJob, Workspace, ApiKey
    schemas.py         # request/response models
    routers/           # memories, search, ask, graph, documents
    services/
      chunking.py      # paragraph/sentence-aware chunker
      embeddings.py    # Embedder protocol + fastembed implementation
      search.py        # vector (HNSW) + BM25 keyword retrieval, weighted RRF fusion
      answers.py       # grounded answers with citations (Claude)
      graph.py         # kNN graph over memory embeddings, rank-based edges
      memories.py      # CRUD and reinforcement
      extraction.py    # PDF/DOCX/text extraction with signature checks
      ingestion.py     # Postgres job queue (SKIP LOCKED) and background worker
      auth.py          # API key generation, hashing, lookup, revocation
      ratelimit.py     # atomic token buckets in Postgres
      decay.py         # forgetting-curve model (retention, spaced-repetition stability)
  alembic/             # migrations
  eval/                # retrieval benchmark harness (BEIR + in-domain set), metrics, results
  tests/
frontend/index.html    # UI and canvas graph engine
Dockerfile             # multi-stage image: venv + baked-in model, non-root, healthcheck
docker/entrypoint.sh   # migrate, then run `api` or `worker`
docker-compose.yml     # db, api, optional scaled workers
scripts/smoke_test.sh  # end-to-end check used by CI
.github/               # CI workflow, Dependabot
```

## Tech Stack

- **Frontend:** HTML5, CSS3 (glassmorphism), vanilla JavaScript, Canvas API (custom force-directed graph), Web Speech API
- **Backend:** Python, FastAPI, Pydantic v2, SQLAlchemy 2 (async) + asyncpg, Alembic, pypdf, python-docx
- **AI:** Claude (Anthropic Python SDK, Citations API)
- **Retrieval:** PostgreSQL 16, pgvector (HNSW), Postgres full-text search, fastembed (BGE-small, ONNX)
- **Tooling:** pytest (+ coverage), ruff, Docker (multi-stage), Docker Compose, GitHub Actions, Dependabot

## History

MemBrain began as a hackathon entry, where storage, vector search and answer generation were delegated to the hosted **Membrain API** provided by the organizers. After the event that hosted service was taken down, which left the original app unable to store or search anything.

The retrieval layer was then rebuilt in-house: local embeddings, pgvector, Postgres full-text search, and rank fusion. The app no longer depends on any third-party AI service. The hackathon submission is preserved under the `hackathon-v1` git tag.

## Roadmap

- [x] Hardening: input validation, async I/O, CORS lockdown, XSS fixes, config via env
- [x] Self-hosted retrieval: local embeddings, pgvector, hybrid full-text + vector search with RRF
- [x] LLM answers with source citations
- [x] Server-side knowledge graph (embedding-similarity edges)
- [x] Forgetting-curve decay model used in ranking
- [x] PDF/DOCX ingestion via a background worker
- [x] Containerized API, GitHub Actions CI
- [x] API-key auth, multi-tenant workspaces, rate limiting
- [x] Retrieval quality evaluation (Recall@k, MRR, nDCG on BEIR + in-domain set) and BM25 keyword ranking
- [ ] Live demo deployment

## Team

- **Sujal:** full-stack architecture and graph physics engine
- **Yashica:** UI/UX and brand identity
- **Melvin:** backend integration and data strategy
