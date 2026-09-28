# COSI Predictor web app

PHP front end for the TF-IDF + logistic regression COSI classifier. No Python is needed on the server:
the model is exported to `data/cosi_model.sqlite` and scored in PHP (`lib/CosiPredictor.php`). The
PHP results match scikit-learn's `predict_proba` to within 1e-7.

## Requirements

- PHP 8.1+ (built for 8.3) with `pdo_sqlite` and `mbstring`. Both are standard on most hosts; `intl` is not needed.
- For the Wikipedia page: outbound HTTPS to `en.wikipedia.org`, through the `curl` extension or `allow_url_fopen`.

## Deploy

Upload the contents of this folder (including the dot files) to the document root of
`cosipredictor.dandeblasio.com`:

```
index.php  wikipedia.php  bulk.php  terms.php  api.php  batch_api.php  config.php  .htaccess
assets/  lib/  data/
```

The `.htaccess` files (Apache/LiteSpeed) block web access to `lib/`, `data/` and `config.php`.
On nginx, add the equivalent:

```
location ~ ^/(lib|data)/ { deny all; }
location = /config.php   { deny all; }
```

## Configure

`config.php`:
- `active_cosis`: set to a list of COSIs to show only the ones running this year; percentages are
  renormalized over that list. `null` shows all 21.
- `names`: the full name shown next to each abbreviation.
- `wikipedia_user_agent`: how the site identifies itself to the Wikipedia API (Wikimedia asks for a
  contact URL or email).

## Wikipedia page

`wikipedia.php?article=<title or URL>&scope=lead|full` fetches an English Wikipedia article and maps it
to the model's fields: title → title; short description + visible categories → keywords; lead section
(default) or whole article → abstract. Result URLs are shareable. Disambiguation pages, missing
articles and non-English URLs get an error message.

## Bulk CSV page

`bulk.php` takes an uploaded CSV and returns it with extra columns: `wiki_title` and `wiki_url`
(Wikipedia files only), `status`, `predicted_cosi`, and a `p_<COSI>` probability column for every COSI.

- **Abstracts:** columns named `title`, `keywords`, `abstract` (any order; other columns are kept, so the
  conference export CSVs work as-is). With no header row, the first three columns are read in that order.
- **Wikipedia articles:** one URL or title per row, in the first column or a column named `url`,
  `article` or `title`.

The browser parses the file and sends rows to `batch_api.php` in small batches (25 abstracts or 5
articles per request), so large files never hit PHP's time limit and nothing is stored on the server.
Limits: 10,000 abstracts or 1,000 articles per file (set in `bulk.php`).

## COSI vocabulary page

`terms.php` shows each COSI's top 10 word stems with a bar for each stem's mean TF-IDF score across that
COSI's training submissions. It reads `data/top_terms.json`, produced by `top_terms.py` (see below).

## API

`POST /api.php` with form fields or a JSON body containing any of `title`, `keywords`, `abstract`:

```
curl -X POST https://cosipredictor.dandeblasio.com/api.php \
     -H 'Content-Type: application/json' \
     -d '{"title": "Deep learning for protein structure prediction"}'
```

Returns `{"predictions": [{"cosi": "3DSIG", "name": "...", "probability": 0.66}, ...]}`, sorted by
probability.

For a Wikipedia article, send `wikipedia` (title or URL) and optionally `scope` (`lead` or `full`):

```
curl -X POST https://cosipredictor.dandeblasio.com/api.php -d 'wikipedia=AlphaFold'
```

The response also includes `article` (title, url, description, categories). Errors return HTTP 422.

## Scripting (batch)

From `~/cosi_predictor`, the same predictions are available offline in Python:

```
python3 predict.py abstracts.csv --out predictions.csv                 # CSV of title/keywords/abstract
python3 predict_wikipedia.py "BLAST (biotechnology)" AlphaFold         # article titles or URLs
python3 predict_wikipedia.py --file articles.txt --scope full --out wiki.csv
```

`predict_wikipedia.py` writes one row per article with the resolved title and URL, a status (`ok` or the
error), the top-N suggestions, and a `p_<COSI>` probability column for every COSI. It accepts the
same `--cosis` option as `predict.py`. `wiki.py` and `lib/Wikipedia.php` map articles the same way, so
the script and the website give the same numbers.

## Updating the model

From `~/cosi_predictor`:

```
python3 train.py <csv dirs/files...>          # writes model.joblib
python3 export_model.py                       # writes web/data/cosi_model.sqlite
python3 top_terms.py <csv dirs/files...>       # writes web/data/top_terms.json
```

Then upload the new `data/cosi_model.sqlite`. If the COSI list changed, update `names` in `config.php`.
