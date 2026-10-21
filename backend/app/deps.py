from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.services.answers import Answerer, get_answerer
from app.services.embeddings import Embedder, get_embedder

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AnswererDep = Annotated[Answerer, Depends(get_answerer)]
