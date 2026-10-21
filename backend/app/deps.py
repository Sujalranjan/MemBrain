import math
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.services import auth, ratelimit
from app.services.answers import Answerer, get_answerer
from app.services.embeddings import Embedder, get_embedder

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AnswererDep = Annotated[Answerer, Depends(get_answerer)]


@dataclass
class Principal:
    workspace: str
    key_name: str | None = None  # None when auth is disabled
    key_prefix: str | None = None


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key")


async def get_principal(request: Request, session: SessionDep, settings: SettingsDep) -> Principal:
    if not settings.auth_enabled:
        return Principal(workspace=settings.workspace)
    key = _presented_key(request)
    api_key = await auth.authenticate(session, key) if key else None
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key" if key else "API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(workspace=api_key.workspace, key_name=api_key.name, key_prefix=api_key.prefix)


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def get_workspace(principal: PrincipalDep) -> str:
    return principal.workspace


WorkspaceDep = Annotated[str, Depends(get_workspace)]


def require_writable(settings: SettingsDep) -> None:
    if settings.read_only:
        raise HTTPException(status_code=403, detail="This is a read-only demo: search and Ask Anything still work")


writable = Depends(require_writable)


def rate_limit(scope: str, limit_setting: str):
    """Dependency enforcing a per-caller token bucket; the caller is the API key, or the client IP without auth."""

    async def dependency(request: Request, response: Response, principal: PrincipalDep, settings: SettingsDep):
        if not settings.rate_limit_enabled:
            return
        per_minute = getattr(settings, limit_setting)
        caller = f"key:{principal.key_prefix}" if principal.key_prefix else f"ip:{request.client.host}"
        # A separate short session: the bucket update commits on its own, independent of the endpoint's work.
        async with request.app.state.sessionmaker() as session:
            decision = await ratelimit.take(session, f"{scope}:{caller}", per_minute)
        headers = {"X-RateLimit-Limit": str(decision.limit), "X-RateLimit-Remaining": str(decision.remaining)}
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit of {per_minute} requests per minute exceeded",
                headers={**headers, "Retry-After": str(math.ceil(decision.retry_after))},
            )
        response.headers.update(headers)

    return Depends(dependency)


api_rate_limit = rate_limit("api", "rate_limit_per_minute")
ask_rate_limit = rate_limit("ask", "ask_rate_limit_per_minute")
