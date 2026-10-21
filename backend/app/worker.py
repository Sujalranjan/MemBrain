"""Standalone ingestion worker: `python -m app.worker`.

Run one or more of these alongside the API (with `ingestion_worker=false` on the API) to move document
processing out of the web process. Workers coordinate through the database, so they scale horizontally.
"""

import asyncio
import logging

from app.config import get_settings
from app.db import make_engine, make_sessionmaker
from app.services.embeddings import FastEmbedEmbedder
from app.services.ingestion import IngestionWorker


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = make_engine(settings.database_url)
    worker = IngestionWorker(make_sessionmaker(engine), FastEmbedEmbedder(settings.embedding_model), settings)
    try:
        await worker.run_forever()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
