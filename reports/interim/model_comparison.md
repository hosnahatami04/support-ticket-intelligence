# Model comparison

| task | model | macro_f1 | train_seconds | latency_p95_ms | size_mb |
| --- | --- | --- | --- | --- | --- |
| queue | tfidf | 0.5780 | 238.9 | 2.98 | 6.6 |
| queue | embeddings | 0.2915 | 47.4 | 0.88 | 0.0 |
| queue | transformer | 0.2448 | 891.8 | 43.69 | 268.6 |
| priority | tfidf | 0.6104 | 59.8 | 1.60 | 3.3 |
| priority | embeddings | 0.4368 | 95.4 | 0.46 | 0.0 |
| priority | transformer | 0.3949 | 646.2 | 16.77 | 268.5 |
