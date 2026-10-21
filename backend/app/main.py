import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.config import REPO_DIR, Settings, get_settings
from app.db import make_engine, make_sessionmaker
from app.deps import PrincipalDep, SettingsDep, api_rate_limit
from app.models import EMBEDDING_DIM
from app.routers import ask, documents, graph, memories, search
from app.services.answers import Answerer, ClaudeAnswerer
from app.services.embeddings import Embedder, FastEmbedEmbedder
from app.services.ingestion import IngestionWorker

logger = logging.getLogger("membrain")


def create_app(
    settings: Settings | None = None, embedder: Embedder | None = None, answerer: Answerer | None = None
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        app.state.embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
        if app.state.embedder.dim != EMBEDDING_DIM:
            raise RuntimeError(f"Embedding model returns {app.state.embedder.dim} dims; schema expects {EMBEDDING_DIM}")
        app.state.answerer = answerer or ClaudeAnswerer(
            settings.answer_model,
            settings.answer_effort,
            settings.answer_max_tokens,
            settings.answer_timeout_seconds,
            api_key=settings.anthropic_api_key,
        )
        app.state.worker = None
        if settings.ingestion_worker:
            app.state.worker = IngestionWorker(app.state.sessionmaker, app.state.embedder, settings)
            app.state.worker.start()
        logger.info(
            "Ready: model=%s auth=%s",
            settings.embedding_model,
            "api-keys" if settings.auth_enabled else f"disabled (workspace {settings.workspace!r})",
        )
        yield
        if app.state.worker:
            await app.state.worker.stop()
        await engine.dispose()

    app = FastAPI(title="MemBrain API", version="0.5.0", lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    for router in (memories.router, search.router, ask.router, graph.router, documents.router):
        app.include_router(router, dependencies=[api_rate_limit])

    @app.get("/api/me", tags=["auth"])
    async def whoami(principal: PrincipalDep, settings: SettingsDep):
        """The workspace this request is scoped to, and the key used (useful to verify a key works)."""
        return {
            "workspace": principal.workspace,
            "key": principal.key_name,
            "auth_enabled": settings.auth_enabled,
            "read_only": settings.read_only,
        }

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(REPO_DIR / "frontend" / "index.html")

    @app.get("/health", tags=["ops"])
    async def health(request: Request):
        try:
            async with request.app.state.sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            db = "ok"
        except Exception:
            logger.exception("Health check: database unreachable")
            db = "unavailable"
        return {"status": "ok" if db == "ok" else "degraded", "database": db}

    return app


app = create_app()
