import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.services.decay import DECAY_FLOOR, current_strength, reinforced_strength
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


def test_strength_decays_linearly_and_is_idempotent():
    accessed = NOW - timedelta(days=10)
    assert current_strength(1.0, accessed, NOW) == pytest.approx(0.88)
    assert current_strength(1.0, accessed, NOW) == current_strength(1.0, accessed, NOW)


def test_strength_has_floor():
    assert current_strength(1.0, NOW - timedelta(days=1000), NOW) == DECAY_FLOOR


def test_reinforce_boosts_from_current_strength_and_caps():
    assert reinforced_strength(1.0, NOW - timedelta(days=50), NOW) == pytest.approx(0.8)
    assert reinforced_strength(1.0, NOW, NOW) == 1.0
