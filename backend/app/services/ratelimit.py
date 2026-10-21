"""Token-bucket rate limiting stored in Postgres, so limits hold across any number of API replicas.

Each bucket refills continuously at `rate` tokens per second up to `capacity`; a request spends one token.
Refill-and-spend is one atomic upsert: concurrent requests on the same bucket serialize on its row lock,
so two replicas can never both spend the last token.
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

REFILLED = """
    LEAST(CAST(:capacity AS float8),
          rate_limits.tokens + EXTRACT(EPOCH FROM clock_timestamp() - rate_limits.updated_at) * CAST(:rate AS float8))
"""

TAKE_SQL = text(
    f"""
    INSERT INTO rate_limits AS rate_limits (bucket, tokens, updated_at)
    VALUES (:bucket, CAST(:capacity AS float8) - 1, clock_timestamp())
    ON CONFLICT (bucket) DO UPDATE
        SET tokens = {REFILLED} - 1, updated_at = clock_timestamp()
        WHERE {REFILLED} >= 1
    RETURNING tokens
    """
)

PEEK_SQL = text(f"SELECT {REFILLED} FROM rate_limits WHERE bucket = :bucket")


@dataclass
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float  # seconds until one token is available; 0 when allowed


async def take(session: AsyncSession, bucket: str, per_minute: int) -> Decision:
    params = {"bucket": bucket, "capacity": per_minute, "rate": per_minute / 60.0}
    tokens = await session.scalar(TAKE_SQL, params)
    await session.commit()
    if tokens is not None:
        return Decision(True, per_minute, int(tokens), 0.0)
    available = await session.scalar(PEEK_SQL, params) or 0.0
    return Decision(False, per_minute, 0, (1 - available) / params["rate"])
