import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(50), default="general")
    tags: Mapped[list[str]] = mapped_column(ARRAY(String(100)), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Forgetting-curve state (see services/decay.py): retention is derived at read time from these.
    stability: Mapped[float] = mapped_column(Float, default=10.0)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    # Document length for BM25: total term frequency across title and body (see MemoryTerm).
    term_count: Mapped[int] = mapped_column(Integer, default=0)
    # Document-level embedding (mean of chunk vectors) used for memory-to-memory similarity in the graph.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    # Title matches outrank body matches in keyword search.
    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(body, '')), 'B')",
            persisted=True,
        ),
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_memories_search_tsv", "search_tsv", postgresql_using="gin"),
        Index(
            "ix_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class MemoryTerm(Base):
    """Inverted index for BM25: one posting per (memory, stemmed lexeme) with its term frequency.

    Derived from `Memory.search_tsv` when a memory is stored; title occurrences count double.
    """

    __tablename__ = "memory_terms"

    memory_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True)
    lexeme: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace: Mapped[str] = mapped_column(String(100))
    tf: Mapped[int] = mapped_column(Integer)

    __table_args__ = (Index("ix_memory_terms_workspace_lexeme", "workspace", "lexeme"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    memory: Mapped[Memory] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class IngestionJob(Base):
    """An uploaded document waiting for (or done with) background text extraction and indexing.

    The table doubles as the work queue: workers claim queued rows with FOR UPDATE SKIP LOCKED.
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace: Mapped[str] = mapped_column(String(100), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    # Raw upload; deferred so listing jobs never loads file bytes, and cleared once the job finishes.
    payload: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
    title: Mapped[str] = mapped_column(String(300))
    type: Mapped[str] = mapped_column(String(50))
    tags: Mapped[list[str]] = mapped_column(ARRAY(String(100)), default=list)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | processing | succeeded | failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    memory_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("memories.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Keeps the worker's "oldest claimable job" lookup cheap as finished jobs accumulate.
        Index(
            "ix_ingestion_jobs_pending",
            "created_at",
            postgresql_where="status IN ('queued', 'processing')",
        ),
    )
