"""Standard ranked-retrieval metrics (binary relevance), matching the definitions used by BEIR/TREC tools."""

import math
import statistics


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 if any relevant document appears in the top k (a.k.a. success@k)."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top k, else 0."""
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def summarize(runs: dict[str, list[str]], qrels: dict[str, set[str]], k_values=(1, 5, 10)) -> dict[str, float]:
    """Mean metrics over all queries that have relevance judgements."""
    qids = [q for q in qrels if q in runs]
    out = {}
    for k in k_values:
        out[f"recall@{k}"] = statistics.fmean(recall_at_k(runs[q], qrels[q], k) for q in qids)
    out["hit@5"] = statistics.fmean(hit_at_k(runs[q], qrels[q], 5) for q in qids)
    out["mrr@10"] = statistics.fmean(mrr_at_k(runs[q], qrels[q], 10) for q in qids)
    out["ndcg@10"] = statistics.fmean(ndcg_at_k(runs[q], qrels[q], 10) for q in qids)
    return out
