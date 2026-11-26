#!/usr/bin/env python3
"""Score the public JevBench split against a local /v1/systemone server,
broken down by input length.

Two jobs:
  1. accuracy / Brier / ECE per length bucket  (<1k, 1-2k, 2-4k, >4k)
     -> this is the gate B measurement: does the score actually fall with
        length? If it does not, the "long input is the weak point" hypothesis
        is wrong and must be written down as wrong.
  2. the same numbers overall, so the smoke run has a single headline.

Scoring follows jevbench/scoring.py: distributions must cover the exact label
set and sum to 1 (tolerance 1e-3, with a 2e-2 renormalisation band), argmax
ties break lexicographically, ordinal score uses the expected value.

Usage:
  python scripts/eval_buckets.py --eval <dir with pub_*.jsonl> \
      --url http://127.0.0.1:8000 --out results/base.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.error
import urllib.request

BUCKETS = [(0, 1000, "<1k"), (1000, 2000, "1-2k"),
           (2000, 4000, "2-4k"), (4000, 10 ** 9, ">4k")]


def bucket_of(tokens: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= tokens < hi:
            return name
    return ">4k"


def post(url: str, body: dict, timeout=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}, time.perf_counter() - t0


def load_eval(paths):
    items = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    items.append(json.loads(line))
    return items


def score_one(item, ans):
    """Return (correct, confidence_of_gold, probs_or_None)."""
    qtype = item["question"]["type"] if isinstance(item["question"], dict) else item["question_type"]
    expected = str(item["expected"])
    labels = [str(x) for x in item["labels"]]

    if qtype == "noul":
        p_yes = float(ans["noul"])
        probs = {"yes": p_yes, "no": 1.0 - p_yes}
    else:
        probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}

    if expected not in probs and expected.isdigit():
        idx = int(expected)
        if 0 <= idx < len(labels):
            # Older calib wire files wrote the option INDEX as expected while
            # the evaluator keys probs by option text. Resolve it instead of
            # silently discarding the row (408 choice rows died this way and
            # the temperature fit reported n=0 for the whole type).
            expected = labels[idx]

    if set(probs) != set(labels):
        return None, None, probs          # invalid distribution -> counts as wrong

    if qtype == "score":
        ev = sum(float(k) * v for k, v in probs.items())
        correct = str(int(round(ev))) == expected
        gold_p = probs.get(expected)
    else:
        best = max(sorted(probs), key=lambda k: probs[k])
        correct = best == expected
        gold_p = probs.get(expected)
    return correct, gold_p, probs


def brier(rows):
    """Multiclass Brier over the items that had a valid distribution."""
    vals = []
    for r in rows:
        p = r["probs"]
        if not p:
            continue
        labels = r["labels"]
        y = r["expected"]
        vals.append(sum((p.get(l, 0.0) - (1.0 if l == y else 0.0)) ** 2 for l in labels))
    return sum(vals) / len(vals) if vals else None


def ece(rows, n_bins=10):
    pairs = [(r["conf"], r["correct"]) for r in rows
             if r["conf"] is not None and r["correct"] is not None]
    if not pairs:
        return None
    total = len(pairs)
    acc = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        seg = [p for p in pairs if lo <= p[0] < hi or (b == n_bins - 1 and p[0] == 1.0)]
        if not seg:
            continue
        conf = sum(p[0] for p in seg) / len(seg)
        hit = sum(1 for p in seg if p[1]) / len(seg)
        acc += len(seg) / total * abs(conf - hit)
    return acc


def state_to_text(st) -> str:
    """Public-split states are sometimes structured; flatten to readable text.

    A dict state is not a formatting nicety: sending it raw is what the server
    rejects with "need string state", and silently json.dumps-ing it buries the
    policy list inside one unbroken blob. Render list-valued fields as bullets.
    """
    if isinstance(st, str):
        return st
    if isinstance(st, list):
        return "\n".join(state_to_text(x) for x in st)
    if isinstance(st, dict):
        out = []
        for k, v in st.items():
            head = str(k).replace("_", " ").strip().capitalize()
            if isinstance(v, (list, tuple)):
                if not v:
                    continue
                out.append(f"{head}:")
                out.extend(f"- {state_to_text(x)}" for x in v)
            elif isinstance(v, dict):
                out.append(f"{head}:")
                out.append(state_to_text(v))
            elif v is not None:
                out.append(f"{head}: {v}")
        return "\n".join(out)
    return str(st or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="dir containing pub_*.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --out")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.eval, "pub_*.jsonl")))
    if not paths:
        raise SystemExit(f"no pub_*.jsonl under {args.eval}")
    items = load_eval(paths)
    if args.limit:
        items = items[: args.limit]
    print(f"eval items: {len(items)} from {len(paths)} files")

    rows = []
    errs = 0
    seen = set()
    if args.resume and os.path.exists(args.out):
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
            rows = prev.get("rows") or []
            seen = {r["id"] for r in rows if not r.get("error")}
            print(f"resume: {len(rows)} items already done")
        except Exception as e:
            print(f"resume skipped, unreadable cache: {e}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    def checkpoint():
        """Write partial results so an interrupted CPU run is never a total loss."""
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"report": {"partial": True, "done": len(rows),
                                  "items": len(items)}, "rows": rows},
                      fh, indent=2, ensure_ascii=False)

    for i, it in enumerate(items):
        if it["id"] in seen:
            continue
        q = it["question"] if isinstance(it["question"], dict) else {
            "type": it.get("question_type"), "instructions": it.get("question", "")}
        body = {"state": state_to_text(it["state"]), "model": "lightemerge-decide-v1",
                "questions": {"decision": q}}
        status, payload, lat = post(args.url, body)
        if status != 200:
            errs += 1
            rows.append({"id": it["id"], "bucket": bucket_of(
                (it.get("provenance") or {}).get("approx_state_tokens", 0)),
                "correct": None, "conf": None, "probs": None,
                "error": str(payload)[:200], "latency_s": lat})
            continue
        ans = payload["answers"]["decision"]
        correct, gold_p, probs = score_one(it, ans)
        rows.append({
            "id": it["id"], "family": it.get("family"),
            "bucket": bucket_of((it.get("provenance") or {}).get("approx_state_tokens", 0)),
            "tokens": (it.get("provenance") or {}).get("approx_state_tokens"),
            "expected": str(it["expected"]), "labels": [str(x) for x in it["labels"]],
            "correct": correct, "conf": gold_p, "probs": probs,
            "latency_s": lat,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)
            checkpoint()

    def summarize(sel):
        n = len(sel)
        scored = [r for r in sel if r["correct"] is not None]
        acc = sum(1 for r in scored if r["correct"]) / len(sel) if n else 0.0
        lat = sorted(r["latency_s"] for r in sel if r.get("latency_s"))
        return {
            "n": n, "scored": len(scored),
            "accuracy": round(acc, 4),
            "brier": round(brier(sel), 4) if brier(sel) is not None else None,
            "ece": round(ece(sel), 4) if ece(sel) is not None else None,
            "p50_latency_s": round(lat[len(lat)//2], 2) if lat else None,
        }

    report = {
        "url": args.url, "items": len(items), "errors": errs,
        "overall": summarize(rows),
        "by_bucket": {name: summarize([r for r in rows if r["bucket"] == name])
                      for _, _, name in BUCKETS},
        "by_family": {},
    }
    fams = sorted({r.get("family") for r in rows if r.get("family")})
    for f in fams:
        report["by_family"][f] = summarize([r for r in rows if r.get("family") == f])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "rows": rows}, fh, indent=2, ensure_ascii=False)

    print(json.dumps(report["overall"], indent=2))
    print("\nby bucket:")
    for k, v in report["by_bucket"].items():
        print(f"  {k:6s} n={v['n']:4d} acc={v['accuracy']:.3f} "
              f"brier={v['brier']} ece={v['ece']}")
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
