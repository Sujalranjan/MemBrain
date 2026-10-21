from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"), extra="ignore")

    database_url: str = "postgresql+asyncpg://membrain:membrain@localhost:5432/membrain"
    # With auth disabled (local development), every request uses this workspace. With auth enabled,
    # requests need an API key and the key decides the workspace; create keys with `python -m app.cli`.
    workspace: str = "engineering"
    auth_enabled: bool = False
    # Public demo mode: reads, search and Q&A work, but nothing can be created, changed or deleted.
    read_only: bool = False
    # Token-bucket limits per API key (or per client IP when auth is disabled). /api/ask has its own,
    # tighter bucket because each call costs a model request.
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 120
    ask_rate_limit_per_minute: int = 10
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    log_level: str = "INFO"

    # Embeddings run locally via fastembed (ONNX); the dimension must match the migration.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking: sized well under the model's 512-token window.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Hybrid retrieval
    rrf_k: int = 10
    # Weight of the keyword ranking relative to the vector ranking in fusion; tuned with eval/ (see README).
    keyword_weight: float = 1.0
    candidate_pool: int = 50
    # BM25 keyword ranking parameters (textbook defaults; results were insensitive to them on SciFact).
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # Weight of the forgetting-curve prior in ranking (0 = pure relevance).
    retention_weight: float = 0.15

    # Grounded answers (Claude). Read here so a key in .env works; when unset, the SDK falls back to
    # its own credential resolution (environment variables, `ant auth login` profile).
    anthropic_api_key: str | None = None
    answer_model: str = "claude-opus-5-5"
    answer_effort: str = "low"
    answer_max_tokens: int = 8000
    answer_timeout_seconds: float = 60.0
    answer_sources: int = 6
    answer_max_source_chars: int = 6000

    # Knowledge graph. Raw cosine scores from small embedding models sit in a narrow band where related
    # and unrelated pairs overlap, so no absolute threshold works; edges are rank-based instead. Each
    # memory links to its nearest neighbour, plus further neighbours (up to `graph_neighbors`) scoring
    # within `graph_margin` of that best match, and never below `graph_min_similarity`.
    graph_neighbors: int = 3
    graph_margin: float = 0.01
    graph_min_similarity: float = 0.5

    # Document ingestion (PDF, DOCX, text). Uploads are queued in Postgres and processed by a background
    # worker. It runs inside the API process by default; set `ingestion_worker=false` there and run
    # `python -m app.worker` to process jobs in separate processes instead.
    max_upload_bytes: int = 10 * 1024 * 1024
    max_document_chars: int = 300_000
    ingestion_worker: bool = True
    ingestion_poll_seconds: float = 2.0
    ingestion_max_attempts: int = 3
    # A job stuck in "processing" this long is assumed to belong to a crashed worker and is retried.
    ingestion_stale_seconds: int = 600

    @field_validator("database_url")
    @classmethod
    def use_asyncpg_driver(cls, v: str) -> str:
        """Accept the URLs hosting providers hand out (postgres://..., ?sslmode=require) for the asyncpg driver."""
        for scheme in ("postgres://", "postgresql://"):
            if v.startswith(scheme):
                v = "postgresql+asyncpg://" + v[len(scheme) :]
        return v.replace("sslmode=", "ssl=")


@lru_cache
def get_settings() -> Settings:
    return Settings()
