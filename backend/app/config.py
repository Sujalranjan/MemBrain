from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"), extra="ignore")

    database_url: str = "postgresql+asyncpg://membrain:membrain@localhost:5432/membrain"
    workspace: str = "engineering"
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    log_level: str = "INFO"

    # Embeddings run locally via fastembed (ONNX); the dimension must match the migration.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking: sized well under the model's 512-token window.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Hybrid retrieval
    rrf_k: int = 60
    candidate_pool: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
