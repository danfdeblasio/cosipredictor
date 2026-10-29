# Single-label vs multi-label COSI classification

Train 2019–2024 → tune thresholds on 2025 (n=1598) → refit on 2019–2025 → test on 2026 with near-duplicates removed (n=1276).
Multi-COSI documents: train 11, val 3, test 0.

## Ranking quality

|              |   top-1 |   top-3 |   top-5 |
|:-------------|--------:|--------:|--------:|
| SL (softmax) |   0.654 |   0.914 |   0.958 |
| OvR          |   0.646 |   0.905 |   0.952 |

OvR top-1 minus SL top-1: -0.008 [95% paired bootstrap CI -0.019, +0.002]

## Suggested COSI sets

`hit_rate` = a correct COSI is among the suggestions; `avg_set_size` = COSIs suggested per submission.

|                           |   hit_rate |   avg_set_size |   sample_precision |   sample_recall |   sample_f1 |   exact_match |   macro_f1 |
|:--------------------------|-----------:|---------------:|-------------------:|----------------:|------------:|--------------:|-----------:|
| SL top-1 (current)        |      0.654 |          1.000 |              0.654 |           0.654 |       0.654 |         0.654 |      0.631 |
| SL top-3                  |      0.914 |          3.000 |              0.305 |           0.914 |       0.457 |         0.000 |      0.460 |
| SL threshold (p >= 0.17)  |      0.761 |          1.415 |              0.607 |           0.761 |       0.656 |         0.463 |      0.641 |
| OvR top-1                 |      0.646 |          1.000 |              0.646 |           0.646 |       0.646 |         0.646 |      0.620 |
| OvR threshold (s >= 0.48) |      0.706 |          1.215 |              0.615 |           0.706 |       0.645 |         0.525 |      0.638 |

## Hit rate at a matched number of suggestions (test)

| Avg. COSIs suggested | SL threshold | OvR threshold |
|---|---|---|
| 1.5 | 0.778 (t=0.15, size=1.49) | 0.775 (t=0.26, size=1.50) |
| 2.0 | 0.853 (t=0.08, size=1.98) | 0.844 (t=0.12, size=1.97) |
| 3.0 | 0.886 (t=0.05, size=2.49) | 0.900 (t=0.05, size=2.82) |

Runtime: 1.9 min.
