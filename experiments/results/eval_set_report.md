# Independent evaluation set

Near-duplicate threshold: cosine >= 0.8 (title + abstract TF-IDF) against all train/val documents.

| Split | Documents | Multi-COSI documents |
|---|---|---|
| test | 1276 | 0 |
| test_excluded_dup | 4 | 0 |
| train | 6224 | 11 |
| val | 1598 | 3 |

Test documents by max similarity to an earlier-year document:

| Similarity | Documents |
|---|---|
| 0.9–1.0 | 2 |
| 0.8–0.9 | 2 |
| 0.7–0.8 | 0 |
| 0.5–0.7 | 6 |
| 0.0–0.5 | 1270 |

Test COSI distribution (a multi-COSI document counts once per COSI):

| COSI | Documents |
|---|---|
| MLCSB | 263 |
| 3DSIG | 104 |
| TransMed | 100 |
| HiTSeq | 85 |
| RegSys | 76 |
| MICROBIOME | 68 |
| BOSC | 65 |
| EvolCompGen | 61 |
| SysMod | 60 |
| NetBio | 55 |
| iRNA | 51 |
| CSI | 51 |
| VarI | 46 |
| Function | 41 |
| BOKR | 33 |
| CAMDA | 29 |
| TextMining | 24 |
| BioVis | 19 |
| Education | 19 |
| CompMS | 19 |
| BioInfo-Core | 7 |
