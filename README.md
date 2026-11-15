# COSI Predictor

Predicts which ISMB Community of Special Interest (COSI) a submission belongs to from its title,
keywords and abstract, using TF-IDF features and multinomial logistic regression trained on accepted
ISMB submissions from 2019–2026. The web app is at https://cosipredictor.dandeblasio.com.

See [METHODS.md](METHODS.md) for the full methods and evaluation, and [web/README.md](web/README.md)
for deploying the web app.

## Layout

| Path | Purpose |
|---|---|
| `train.py` | Train the model from submission CSVs (`--holdout` evaluates on one conference year first) |
| `predict.py` | Predict COSIs for a CSV with `title`, `keywords`, `abstract` columns |
| `predict_wikipedia.py`, `wiki.py` | Predict COSIs for a list of English Wikipedia articles |
| `export_model.py` | Export `model.joblib` to the SQLite file used by the web app |
| `top_terms.py` | Top 10 stems per COSI by mean TF-IDF, for the web app's vocabulary page (needs `nltk`) |
| `compare.py`, `embed.py` | Comparison against SPECTER2 embeddings (optional; needs `transformers` and `adapters`) |
| `compare_stemming.py` | Comparison of the classifier with and without Porter stemming (needs `nltk`) |
| `explore.py` | Data-exploration report from the submission CSVs (aggregates only) |
| `common.py` | CSV loading, track-name normalization and the shared training-data filter |
| `cosis.txt` | The 21 COSIs the model is trained on |
| `model.joblib` | Trained model (scikit-learn 1.6.1) |
| `web/` | PHP web app, including the exported model in `web/data/` |
| `reports/eda.md` | Data-exploration report on the real submission data, written by `explore.py` |
| `experiments/` | Reproduction of METHODS.md and evaluation experiments (see `experiments/README.md`) |
| `tests/`, `requirements-dev.txt` | `pytest` tests on synthetic data, and the test dependency |

## Usage

Requires Python 3.9+ with scikit-learn, pandas and joblib.

```bash
python3 train.py <csv files or directories> --holdout ISMB2026   # evaluate, then train on everything
python3 predict.py abstracts.csv --out predictions.csv
python3 predict_wikipedia.py "BLAST (biotechnology)" AlphaFold --out wiki.csv
python3 export_model.py                                          # refresh web/data/cosi_model.sqlite
python3 top_terms.py <csv files or directories>                  # refresh web/data/top_terms.json
```

The training data (ISMB submission exports) is not included in this repository.

## Scripts

Every runnable script takes `--help` (`wiki.py`, `embed.py` and `common.py` are libraries). Scripts that read submission data accept CSV files and/or directories of
CSVs; files without a `trackname` column are skipped. `--model` defaults to `model.joblib`. In the scripts
that train or analyse (`train.py`, `top_terms.py`, `explore.py`, `compare.py`, `compare_stemming.py`),
`--cosis` defaults to `cosis.txt`; in `predict.py` and `predict_wikipedia.py` it only restricts the
suggestions to the COSIs given. Unless noted, the scripts need only scikit-learn, pandas and joblib.

| Script | Purpose | Inputs | Outputs | Extra dependencies |
|---|---|---|---|---|
| `train.py` | Train the COSI classifier (TF-IDF on title, keywords and abstract + logistic regression); with `--holdout`, first evaluate on one conference | Submission CSVs; `--cosis` (default `cosis.txt`); `--holdout` (e.g. `ISMB2026`) | `--out` model (default `model.joblib`); counts and held-out metrics on stdout | — |
| `predict.py` | Rank COSIs for new submissions | A CSV with `title`, `keywords`, `abstract` (other columns pass through); `--model`; `--cosis` to restrict to this year's COSIs; `--top` | `--out` CSV (default `predictions.csv`) | — |
| `predict_wikipedia.py` | Rank COSIs for English Wikipedia articles | Article titles or URLs as arguments and/or `--file`; `--scope lead\|full`; `--model`; `--cosis`; `--top`; `--delay` | `--out` CSV (default `wikipedia_predictions.csv`): one row per article with top suggestions and a probability per COSI | Internet access |
| `wiki.py` | Library used by `predict_wikipedia.py`: fetches an article and maps it to the model's fields (kept in sync with `web/lib/Wikipedia.php`) | — | — | Internet access |
| `export_model.py` | Export the trained model for the PHP web app | `--model` (default `model.joblib`) | `--out` SQLite file (default `web/data/cosi_model.sqlite`) | — |
| `top_terms.py` | Top stems per COSI by mean TF-IDF, for the web app's vocabulary page | Submission CSVs; `--cosis`; `--top`; `--min-df` | `--out` JSON (default `web/data/top_terms.json`) | `nltk` |
| `explore.py` | Data-exploration report: files, conferences, filtering steps, COSIs per conference, unmapped tracks, duplicate and placeholder titles, missing fields | Submission CSVs; `--cosis`; `--latest` (most recent ISMB conference; detected by default) | `--out` Markdown report (default `reports/eda.md`, aggregates only); `--details` per-record CSV (default `cache/eda_details.csv`, gitignored, contains titles) | — |
| `compare.py` | Compare the TF-IDF model with SPECTER2-embedding classifiers on a held-out conference | Submission CSVs; `--cosis`; `--holdout` (default `ISMB2026`); `--adapters` | Metrics on stdout; embeddings cached in `--cache-dir` (default `cache/`) | `transformers`, `adapters`, `torch` |
| `embed.py` | Library used by `compare.py`: SPECTER2 embeddings of title + keywords + abstract | — | — | `transformers`, `adapters`, `torch` |
| `compare_stemming.py` | Compare the classifier with and without Porter stemming, with a paired bootstrap | Submission CSVs; `--cosis`; `--holdout`; `--bootstrap` | Metrics and confidence intervals on stdout | `nltk` |
| `common.py` | Library: CSV loading, track-name normalization (`ALIASES`) and `filter_training_data`, the training filter shared by `train.py` and `explore.py` | — | — | — |

To run the tests: `pip install -r requirements-dev.txt` and then `pytest tests` (synthetic data only).
