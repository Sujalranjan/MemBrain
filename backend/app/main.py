import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.config import REPO_DIR, Settings, get_settings
from app.db import make_engine, make_sessionmaker
from app.models import EMBEDDING_DIM
from app.routers import memories, search
from app.services.embeddings import Embedder, FastEmbedEmbedder

logger = logging.getLogger("membrain")


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        app.state.embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
        if app.state.embedder.dim != EMBEDDING_DIM:
            raise RuntimeError(f"Embedding model returns {app.state.embedder.dim} dims; schema expects {EMBEDDING_DIM}")
        logger.info("Ready: model=%s workspace=%s", settings.embedding_model, settings.workspace)
        yield
        await engine.dispose()

    app = FastAPI(title="MemBrain API", version="0.3.0", lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.include_router(memories.router)
    app.include_router(search.router)

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
