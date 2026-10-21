### scifact: BEIR scifact (test split)

5,183 documents, 300 queries; `BAAI/bge-small-en-v1.5`, chunks 1000/150, RRF k=60.

| Mode | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.577 | 0.772 | 0.846 | 0.683 | 0.719 | 8.0 ms |
| keyword | 0.215 | 0.439 | 0.521 | 0.325 | 0.369 | 61.8 ms |
| hybrid | 0.452 | 0.681 | 0.792 | 0.567 | 0.616 | 64.9 ms |

Hybrid vs vector-only: Recall@5 0.772 → 0.681 (-11.8% relative, -9.1 points); MRR@10 0.683 → 0.567 (-17.0%).
