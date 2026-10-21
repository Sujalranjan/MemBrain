import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryCreate(BaseModel):
    title: str = Field(default="Untitled", min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=50_000)
    type: str = Field(default="general", max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=50)


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    body: str
    type: str
    tags: list[str]
    created_at: datetime
    last_accessed_at: datetime
    strength: float
    current_strength: float


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    k: int = Field(default=6, ge=1, le=50)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"


class SearchResult(BaseModel):
    id: uuid.UUID
    title: str
    type: str
    tags: list[str]
    snippet: str
    score: float
    vector_similarity: float | None
    vector_rank: int | None
    keyword_rank: int | None
    matched_by: list[str]


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]
    # Populated once LLM answer synthesis lands; the frontend falls back to listing results.
    answer: str | None = None
