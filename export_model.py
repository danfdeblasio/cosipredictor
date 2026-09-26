"""Export a trained model.joblib to a SQLite file the PHP web app can score with.

Example:
    python3 export_model.py --model model.joblib --out web/data/cosi_model.sqlite
"""
import argparse
import datetime
import json
import os
import sqlite3
import unicodedata

import joblib
import numpy as np


def accent_map():
    """Characters that sklearn's strip_accents='unicode' changes, for PHP hosts without intl."""
    out = {}
    for cp in range(0x80, 0x10000):
        if 0xAC00 <= cp <= 0xD7A3 or 0xD800 <= cp <= 0xDFFF:  # Hangul syllables, surrogates
            continue
        c = chr(cp)
        nfkd = unicodedata.normalize("NFKD", c)
        stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
        if stripped != c:
            out[c] = stripped
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--out", default="web/data/cosi_model.sqlite")
    args = ap.parse_args()

    model = joblib.load(args.model)
    features, clf = model.named_steps["features"], model.named_steps["clf"]
    coef = clf.coef_.astype("<f4")  # (n_classes, n_features)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    db = sqlite3.connect(args.out)
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE terms (field TEXT NOT NULL, term TEXT NOT NULL, idf REAL NOT NULL,
                            coef BLOB NOT NULL, PRIMARY KEY (field, term)) WITHOUT ROWID;
    """)

    fields, offset = [], 0
    for name, vec, column in features.transformers_:
        if name == "remainder":
            continue
        vocab = vec.vocabulary_
        # ColumnTransformer stacks each field's columns in order.
        rows = ((name, term, float(vec.idf_[j]), coef[:, offset + j].tobytes())
                for term, j in vocab.items())
        db.executemany("INSERT INTO terms VALUES (?, ?, ?, ?)", rows)
        fields.append({"name": name, "column": column, "ngram_range": list(vec.ngram_range),
                       "sublinear_tf": vec.sublinear_tf, "norm": vec.norm,
                       "lowercase": vec.lowercase, "strip_accents": vec.strip_accents})
        offset += len(vocab)
    assert offset == coef.shape[1]

    stop_words = sorted(features.transformers_[0][1].get_stop_words() or [])
    meta = {
        "classes": [str(c) for c in clf.classes_],
        "intercept": [float(b) for b in clf.intercept_],
        "fields": fields,
        "stop_words": stop_words,
        "token_pattern": features.transformers_[0][1].token_pattern,
        "accent_map": accent_map(),
        "exported": datetime.date.today().isoformat(),
        "n_features": int(coef.shape[1]),
    }
    db.executemany("INSERT INTO meta VALUES (?, ?)",
                   [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()])
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"Exported {offset} terms x {len(meta['classes'])} COSIs to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
