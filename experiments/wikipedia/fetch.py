"""Download every article named in sets.json and hidden.json, 20 per request (#25).

Writes to --cache (default cache/wikipedia, gitignored):
    articles.jsonl  one record per title in sets.json (pos, near, random, easy, discovery)
    hidden.jsonl    one record per title in hidden.json

Record: query, title, url, description, categories, text (lead section), length (page bytes), status
(ok, missing or disambiguation). Resumable: titles already in the output files are skipped, so an interrupted
run continues where it stopped. --limit N fetches at most N new titles per file (smoke run).

    python experiments/wikipedia/fetch.py
    python experiments/wikipedia/fetch.py --limit 20 --cache cache/wikipedia-smoke
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikidata as wd  # noqa: E402


def set_titles(sets):
    out = [t for v in sets["pos"].values() for t in v] + [t for v in sets["near"].values() for t in v]
    out += sets["random"] + sets["easy"] + sets["discovery"]
    return list(dict.fromkeys(out))


def download(titles, path, limit=0, api=wd.api, log=print):
    done = wd.read_jsonl(path)
    todo = [t for t in titles if t not in done]
    if limit:
        todo = todo[:limit]
    with open(path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), wd.BATCH):
            for r in wd.fetch_batch(todo[i:i + wd.BATCH], api=api):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n = min(i + wd.BATCH, len(todo))
            if n % 1000 < wd.BATCH or n == len(todo):
                log(f"  {os.path.basename(path)}: {len(done) + n}/{len(done) + len(todo)}")
    return len(done), len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=wd.DEFAULT_CACHE, help="directory with sets.json and hidden.json")
    ap.add_argument("--limit", type=int, default=0, help="fetch at most N new titles per file (smoke run)")
    args = ap.parse_args(argv)
    with open(os.path.join(args.cache, "sets.json")) as f:
        sets = json.load(f)
    hidden_path = os.path.join(args.cache, "hidden.json")
    hidden = json.load(open(hidden_path)) if os.path.exists(hidden_path) else []
    for titles, name in [(set_titles(sets), "articles.jsonl"), (hidden, "hidden.jsonl")]:
        had, new = download(titles, os.path.join(args.cache, name), args.limit)
        print(f"{name}: {had} already cached, {new} fetched")


if __name__ == "__main__":
    main()
