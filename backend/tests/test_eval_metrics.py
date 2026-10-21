import math

import pytest

from eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k, summarize


def test_recall_counts_fraction_of_relevant_found():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0


def test_mrr_uses_first_relevant_rank_within_cutoff():
    assert mrr_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_ndcg_is_one_for_ideal_ranking_and_discounts_lower_ranks():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))


def test_hit_and_summary():
    assert hit_at_k(["x", "a"], {"a"}, 1) == 0.0 and hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    m = summarize({"q1": ["a"], "q2": ["x"]}, {"q1": {"a"}, "q2": {"b"}})
    assert m["recall@1"] == 0.5 and m["mrr@10"] == 0.5
