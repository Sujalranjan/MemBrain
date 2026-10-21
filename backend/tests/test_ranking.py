import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.services.decay import (
    INITIAL_STABILITY_DAYS,
    MAX_STABILITY_DAYS,
    half_life_days,
    retention,
    reviewed_stability,
)
from app.services.search import reciprocal_rank_fusion

A, B, C = (uuid.uuid4() for _ in range(3))
NOW = datetime(2026, 1, 31, tzinfo=timezone.utc)


def test_rrf_rewards_agreement_between_retrievers():
    scores = reciprocal_rank_fusion([[A, B, C], [B, C, A]], k=60)
    assert max(scores, key=scores.get) == B


def test_rrf_single_list_preserves_order():
    scores = reciprocal_rank_fusion([[A, B, C]], k=60)
    assert sorted(scores, key=scores.get, reverse=True) == [A, B, C]


def test_rrf_formula():
    assert reciprocal_rank_fusion([[A], [A]], k=60)[A] == pytest.approx(2 / 61)


def test_retention_follows_exponential_curve():
    s = INITIAL_STABILITY_DAYS
    assert retention(s, NOW, NOW) == 1.0
    assert retention(s, NOW - timedelta(days=s), NOW) == pytest.approx(math.exp(-1))
    assert retention(s, NOW - timedelta(days=half_life_days(s)), NOW) == pytest.approx(0.5)


def test_higher_stability_decays_slower():
    week_ago = NOW - timedelta(days=7)
    assert retention(30, week_ago, NOW) > retention(10, week_ago, NOW)


def test_spacing_effect_rewards_reviewing_fading_memories():
    just_seen = reviewed_stability(10, NOW, NOW)
    nearly_forgotten = reviewed_stability(10, NOW - timedelta(days=30), NOW)
    assert just_seen == pytest.approx(11.0)  # cramming: only the base gain
    assert nearly_forgotten > 2.5 * just_seen


def test_stability_is_capped():
    assert reviewed_stability(MAX_STABILITY_DAYS, NOW - timedelta(days=900), NOW) == MAX_STABILITY_DAYS
