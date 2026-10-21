from datetime import datetime, timezone

DECAY_PER_DAY = 0.012
DECAY_FLOOR = 0.04
REINFORCE_BOOST = 0.4


def current_strength(strength: float, last_accessed_at: datetime, now: datetime | None = None) -> float:
    """Strength decays linearly from the value recorded at last access. Mirrors strength() in the frontend."""
    now = now or datetime.now(timezone.utc)
    days = max(0.0, (now - last_accessed_at).total_seconds() / 86400)
    return max(DECAY_FLOOR, strength - days * DECAY_PER_DAY)


def reinforced_strength(strength: float, last_accessed_at: datetime, now: datetime | None = None) -> float:
    return min(1.0, current_strength(strength, last_accessed_at, now) + REINFORCE_BOOST)
