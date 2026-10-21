"""API keys: generation, hashing, lookup and revocation.

Keys look like `mb_<prefix>_<secret>`. Only a SHA-256 digest of the whole key is stored. A fast hash is
appropriate here, unlike for passwords: the secret is 256 random bits, so there is nothing to brute-force,
and verification stays cheap on every request. The public prefix finds the row; the digest comparison
is constant-time.
"""

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ApiKey, Workspace

KEY_PATTERN = re.compile(r"^mb_([0-9a-f]{12})_([A-Za-z0-9_-]{43})$")
WORKSPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
# Recording every use would add a write to each request; once per interval is enough to spot stale keys.
LAST_USED_RESOLUTION = timedelta(minutes=5)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> tuple[str, str]:
    """Returns (prefix, full key)."""
    prefix = secrets.token_hex(6)
    return prefix, f"mb_{prefix}_{secrets.token_urlsafe(32)}"


async def ensure_workspace(session: AsyncSession, slug: str) -> None:
    if not WORKSPACE_PATTERN.match(slug):
        raise ValueError("Workspace must be lowercase letters, digits and hyphens (max 63 chars)")
    if await session.get(Workspace, slug) is None:
        session.add(Workspace(slug=slug))
        await session.flush()


async def create_key(session: AsyncSession, workspace: str, name: str) -> tuple[ApiKey, str]:
    """Create a key (and its workspace, if new). The plaintext key is returned once and never stored."""
    await ensure_workspace(session, workspace)
    prefix, key = generate_key()
    api_key = ApiKey(workspace=workspace, name=name, prefix=prefix, key_hash=hash_key(key))
    session.add(api_key)
    await session.commit()
    return api_key, key


async def authenticate(session: AsyncSession, key: str) -> ApiKey | None:
    match = KEY_PATTERN.match(key)
    if not match:
        return None
    api_key = await session.scalar(select(ApiKey).where(ApiKey.prefix == match.group(1)))
    if api_key is None or api_key.revoked_at is not None:
        return None
    if not hmac.compare_digest(api_key.key_hash, hash_key(key)):
        return None
    now = datetime.now(timezone.utc)
    if api_key.last_used_at is None or now - api_key.last_used_at > LAST_USED_RESOLUTION:
        api_key.last_used_at = now
        await session.commit()
    return api_key


async def list_keys(session: AsyncSession, workspace: str | None = None) -> list[ApiKey]:
    query = select(ApiKey).order_by(ApiKey.workspace, ApiKey.created_at)
    if workspace:
        query = query.where(ApiKey.workspace == workspace)
    return list(await session.scalars(query))


async def revoke_key(session: AsyncSession, prefix: str) -> bool:
    api_key = await session.scalar(select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked_at.is_(None)))
    if api_key is None:
        return False
    api_key.revoked_at = datetime.now(timezone.utc)
    await session.commit()
    return True
