### nfcorpus: BEIR nfcorpus (test split)

3,633 documents, 323 queries; `BAAI/bge-small-en-v1.5`, chunks 1000/150, RRF k=10.

| Mode | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.057 | 0.119 | 0.157 | 0.524 | 0.335 | 7.9 ms |
| keyword | 0.056 | 0.128 | 0.155 | 0.526 | 0.325 | 20.7 ms |
| hybrid | 0.060 | 0.133 | 0.173 | 0.548 | 0.359 | 23.3 ms |

Hybrid vs vector-only: Recall@5 0.119 → 0.133 (+11.3% relative, +1.3 points); MRR@10 0.524 → 0.548 (+4.6%).
