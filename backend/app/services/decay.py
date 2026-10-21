"""Forgetting-curve memory model (a simplified form of what spaced-repetition systems use).

Retention follows Ebbinghaus' exponential curve, R(t) = exp(-t / S), where t is days since the memory was
last used and S is its stability in days. Every review multiplies stability, and by more the closer the
memory was to being forgotten (the spacing effect): re-reviewing something seen a minute ago barely helps,
while rescuing a fading memory makes it last much longer. Frontend mirror: strength() in index.html.
"""

import math
from datetime import datetime, timezone

INITIAL_STABILITY_DAYS = 10.0  # new memories: retention halves after ~7 days (S * ln 2)
MAX_STABILITY_DAYS = 365.0
BASE_GAIN = 1.1  # minimum stability multiplier per review, even when R ~ 1
SPACING_GAIN = 2.0  # extra multiplier per unit of forgetting at review time
FADING_THRESHOLD = 0.4  # below this a memory is flagged for review


def retention(stability: float, last_accessed_at: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    days = max(0.0, (now - last_accessed_at).total_seconds() / 86400)
    return math.exp(-days / max(stability, 1e-6))


def reviewed_stability(stability: float, last_accessed_at: datetime, now: datetime | None = None) -> float:
    r = retention(stability, last_accessed_at, now)
    return min(MAX_STABILITY_DAYS, stability * (BASE_GAIN + SPACING_GAIN * (1 - r)))


def half_life_days(stability: float) -> float:
    return stability * math.log(2)
