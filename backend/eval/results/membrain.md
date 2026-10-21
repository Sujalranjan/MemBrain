### membrain: MemBrain in-domain set: 36 synthetic enterprise notes, 48 hand-labelled queries (written before any results were seen)

36 documents, 48 queries; `BAAI/bge-small-en-v1.5`, chunks 1000/150, RRF k=10.

| Mode | recall@1 | recall@5 | recall@10 | mrr@10 | ndcg@10 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.802 | 0.917 | 0.938 | 0.858 | 0.878 | 6.9 ms |
| keyword | 0.719 | 0.875 | 0.875 | 0.777 | 0.798 | 4.4 ms |
| hybrid | 0.781 | 0.979 | 1.000 | 0.860 | 0.894 | 8.2 ms |

Hybrid vs vector-only: Recall@5 0.917 → 0.979 (+6.8% relative, +6.2 points); MRR@10 0.858 → 0.860 (+0.2%).

| Query type | n | vector R@5 | keyword R@5 | hybrid R@5 |
|---|---:|---:|---:|---:|
| exact | 16 | 0.750 | 1.000 | 1.000 |
| mixed | 8 | 1.000 | 1.000 | 1.000 |
| paraphrase | 24 | 1.000 | 0.750 | 0.958 |
