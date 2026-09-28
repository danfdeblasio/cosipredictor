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
| `common.py` | CSV loading and track-name normalization |
| `cosis.txt` | The 21 COSIs the model is trained on |
| `model.joblib` | Trained model (scikit-learn 1.6.1) |
| `web/` | PHP web app, including the exported model in `web/data/` |

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
