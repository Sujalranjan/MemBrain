### scifact: BEIR scifact (test split)

5,183 documents, 300 queries; `BAAI/bge-small-en-v1.5`, chunks 1000/150, RRF k=10.

| Mode | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.573 | 0.765 | 0.836 | 0.679 | 0.714 | 8.1 ms |
| keyword | 0.535 | 0.754 | 0.822 | 0.656 | 0.691 | 85.3 ms |
| hybrid | 0.588 | 0.806 | 0.876 | 0.701 | 0.740 | 88.1 ms |

Hybrid vs vector-only: Recall@5 0.765 → 0.806 (+5.3% relative, +4.1 points); MRR@10 0.679 → 0.701 (+3.3%).
