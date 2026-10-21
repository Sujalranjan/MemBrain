import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryCreate(BaseModel):
    title: str = Field(default="Untitled", min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=50_000)
    type: str = Field(default="general", max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("title", "body")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(t.strip() for t in v if t.strip()))


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    body: str
    type: str
    tags: list[str]
    created_at: datetime
    last_accessed_at: datetime
    stability: float  # days; grows with each review
    access_count: int
    retention: float  # 0..1, exp(-days_since_access / stability)
    half_life_days: float


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
    score: float  # final ranking score: relevance adjusted by retention
    relevance: float  # fused retrieval score before the retention prior
    retention: float
    vector_similarity: float | None
    vector_rank: int | None
    keyword_rank: int | None
    matched_by: list[str]


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    source: int  # 1-based index into `sources`, matching the [n] markers in `answer`
    memory_id: uuid.UUID
    title: str
    cited_text: str


class AskResponse(BaseModel):
    question: str
    answer: str | None  # None when answer generation is unavailable; `sources` are still returned
    grounded: bool
    citations: list[CitationOut]
    sources: list[SearchResult]
    model: str | None = None
    notice: str | None = None


class GraphNode(BaseModel):
    id: uuid.UUID
    title: str
    type: str
    tags: list[str]
    retention: float


class GraphEdge(BaseModel):
    source: uuid.UUID
    target: uuid.UUID
    similarity: float
    shared_tags: list[str]


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    neighbors: int
    margin: float
    min_similarity: float


class IngestionJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str | None
    size_bytes: int
    title: str
    type: str
    tags: list[str]
    status: Literal["queued", "processing", "succeeded", "failed"]
    attempts: int
    error: str | None
    chars: int | None  # extracted text length, once processed
    memory_id: uuid.UUID | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
