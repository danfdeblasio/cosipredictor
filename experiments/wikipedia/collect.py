"""List the WikiProject Computational Biology collection and draw the article sets for the experiments (#25).

Writes to --cache (default cache/wikipedia, gitignored):
    cats.json    collection {title: quality class}, seed subcategories, candidates, missing
    sets.json    pos (labelled sample by class), near (controls by field), random, easy (Lonnie's examples),
                 discovery (candidate pool) and source (category each candidate came from)
    hidden.json  collection articles of class B, C, Start or Stub that are not in the labelled sample

Needs network access. Sampling uses seed 2026, except list=random, which cannot be seeded: the drawn titles are
stored in sets.json and reused. --limit N caps every category listing at N members for a quick smoke run.

    python experiments/wikipedia/collect.py
    python experiments/wikipedia/collect.py --limit 20 --cache cache/wikipedia-smoke
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikidata as wd  # noqa: E402

POS_SIZES = [("GA", 12), ("B", 40), ("C", 40), ("Start", 40), ("Stub", 40)]
NEAR_PER_FIELD = 8
SUBCATS_PER_FIELD = 40
N_RANDOM = 60


def _limited(api, limit):
    """Wrap api so a listing stops after one page of at most `limit` members."""
    if not limit:
        return api

    def call(params):
        r = api(dict(params, cmlimit=str(limit)) if "cmlimit" in params else params)
        r = dict(r)
        r.pop("continue", None)
        return r
    return call


def seed_candidates(api=wd.api):
    """Subcategories of the seed categories (one level) and every article in the seeds and those subcategories."""
    subcats, cands = [], set()
    for cat in wd.SEED_CATEGORIES:
        c = f"Category:{cat}"
        cands |= {m["title"] for m in wd.members(c, api=api, cmnamespace="0")}
        for s in wd.members(c, api=api, cmtype="subcat"):
            if s["title"] not in subcats:
                subcats.append(s["title"])
    for s in subcats:
        cands |= {m["title"] for m in wd.members(s, api=api, cmnamespace="0")}
    return subcats, sorted(cands)


def field_members(api=wd.api, fields=wd.NEAR_FIELDS):
    """Direct article members of each nearby field, and the members of up to 40 of its subcategories."""
    direct, pools = {}, {}
    for f in fields:
        c = f"Category:{f}"
        direct[f] = [m["title"] for m in wd.members(c, api=api, cmnamespace="0")]
        pool = list(direct[f])
        for s in [m["title"] for m in wd.members(c, api=api, cmtype="subcat")][:SUBCATS_PER_FIELD]:
            pool += [m["title"] for m in wd.members(s, api=api, cmnamespace="0")]
        pools[f] = pool
    return direct, pools


def draw_random(coll, api=wd.api, n=N_RANDOM):
    out = []
    while len(out) < n:
        r = api({"action": "query", "list": "random", "rnnamespace": "0", "rnlimit": "20"})
        for m in r.get("query", {}).get("random", []):
            if m["title"] not in coll and m["title"] not in out and len(out) < n:
                out.append(m["title"])
    return out


def build_sets(coll, candidates, near_direct, near_pools, random_titles, seed=wd.SEED):
    """The labelled sample, controls and candidate pool, as in the 2026-11-20 prototype."""
    rng = random.Random(seed)
    pos = {}
    for cls, n in POS_SIZES:
        ts = sorted(t for t, c in coll.items() if c == cls)
        pos[cls] = rng.sample(ts, min(n, len(ts)))
    cand = set(candidates)

    def ok(t):
        return t not in coll and t not in cand and not t.startswith("List of")

    near, pool = {}, {}
    for f, ms in near_direct.items():
        ms = sorted(t for t in ms if ok(t))
        near[f] = rng.sample(ms, min(NEAR_PER_FIELD, len(ms)))
        pool[f] = [t for t in near_pools.get(f, ms) if ok(t)]
    easy = list(wd.LONNIE_EXAMPLES)
    negs = {t for v in near.values() for t in v} | set(random_titles) | set(easy)
    missing = [t for t in candidates if t not in coll and not t.startswith("List of")]
    disc = sorted((set(missing) | {t for v in pool.values() for t in v}) - negs)
    source = {t: wd.BIOINFO_SOURCE for t in missing}
    for f, v in pool.items():
        for t in v:
            source.setdefault(t, f)
    return {"pos": pos, "near": near, "random": list(random_titles), "easy": easy, "discovery": disc, "source": source}


def hidden_list(coll, pos):
    used = {t for v in pos.values() for t in v}
    return sorted(t for t, c in coll.items() if c in ("B", "C", "Start", "Stub") and t not in used)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=wd.DEFAULT_CACHE, help="output directory (default: cache/wikipedia)")
    ap.add_argument("--limit", type=int, default=0, help="cap every category listing at N members (smoke run)")
    args = ap.parse_args(argv)
    api = _limited(wd.api, args.limit)
    os.makedirs(args.cache, exist_ok=True)

    coll = wd.collection(api=api)
    subcats, cands = seed_candidates(api=api)
    missing = [t for t in cands if t not in coll and not t.startswith("List of")]
    with open(os.path.join(args.cache, "cats.json"), "w") as f:
        json.dump({"collection": coll, "subcats": subcats, "candidates": cands, "missing": missing}, f)
    print(f"collection {len(coll)}, seed subcategories {len(subcats)}, candidates {len(cands)}, missing {len(missing)}")

    direct, pools = field_members(api=api)
    rnd = draw_random(coll, api=wd.api, n=min(N_RANDOM, args.limit) if args.limit else N_RANDOM)
    sets = build_sets(coll, cands, direct, pools, rnd)
    hidden = hidden_list(coll, sets["pos"])
    with open(os.path.join(args.cache, "sets.json"), "w") as f:
        json.dump(sets, f, indent=0)
    with open(os.path.join(args.cache, "hidden.json"), "w") as f:
        json.dump(hidden, f)
    print({k: len(v) for k, v in sets["pos"].items()}, "near", sum(len(v) for v in sets["near"].values()),
          "random", len(sets["random"]), "discovery", len(sets["discovery"]), "hidden", len(hidden))
    print(f"Wrote cats.json, sets.json and hidden.json to {args.cache}")


if __name__ == "__main__":
    main()
