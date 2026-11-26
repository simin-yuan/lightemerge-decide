"""Run the benchmark in-process, with no HTTP server in the middle.

The HTTP path proved unreliable on this box: the serving process died silently
during long-sequence requests (no traceback in its log, exit 127), which made the
evaluator fail with ConnectionReset and write nothing. Loading the model once and
looping in the same process removes the socket and the second process entirely,
and any crash now surfaces as a real traceback instead of a vanished port.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_buckets import bucket_of, ece, score_one, state_to_text  # noqa: E402
from serve_systemone import Reader  # noqa: E402
import promptfmt  # noqa: E402
import torch  # noqa: E402


def load_items(paths):
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="dir containing pub_*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dtype", default="float32",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--prompt-style", default="native")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir; empty means evaluate the raw base model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=0,
                    help="stop after this many newly-scored items (batch mode)")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    files = sorted(os.path.join(args.eval, f) for f in os.listdir(args.eval)
                   if f.startswith("pub") and f.endswith(".jsonl"))
    items = load_items(files)
    if args.limit:
        items = items[:args.limit]
    print(f"eval items: {len(items)} from {len(files)} files", flush=True)
    if not items:
        # A run that loads nothing used to exit 0 with an empty report, which read
        # like success. Refuse instead: n=0 is a broken input path, not a score.
        print("ERROR: 0 items loaded - check --eval dir and the pub* file names", flush=True)
        sys.exit(2)

    rows, done = [], set()
    if args.resume and os.path.exists(args.out):
        prev = json.load(open(args.out, encoding="utf-8"))
        rows = prev.get("rows", [])
        done = {r["id"] for r in rows if r.get("correct") is not None}
        print(f"resume: {len(done)} items already done", flush=True)

    reader = Reader(args.base, args.adapter, args.max_len, args.dtype, args.device)
    promptfmt.set_style(args.prompt_style)
    if args.threads:
        torch.set_num_threads(args.threads)
    print("model loaded", flush=True)

    lat = []
    new_done = 0
    for i, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        if args.max_new and new_done >= args.max_new:
            json.dump({"rows": rows}, open(args.out, "w", encoding="utf-8"),
                      ensure_ascii=False)
            print(f"batch limit reached ({new_done} new items), exiting (batch mode)", flush=True)
            break
        new_done += 1
        t0 = time.time()
        try:
            state = state_to_text(item["state"])
            q = item["question"]
            prompt, labels, cand_ids = reader.render(state, q)
            p, ntok = reader.probs(prompt, labels, cand_ids)
            if q["type"] == "noul":
                ans = {"noul": p[labels.index("yes")]}
            else:
                ans = {"probabilities": dict(zip(labels, p))}
            correct, gold_p, probs = score_one(item, ans)
            row = {"id": item["id"], "family": item.get("family"),
                   "question_type": q["type"], "tokens": ntok,
                   "bucket": bucket_of(ntok), "expected": str(item["expected"]),
                   "correct": correct, "conf": gold_p, "probs": probs,
                   "latency_s": round(time.time() - t0, 2)}
        except Exception as e:  # keep going; a bad item must not kill the run
            row = {"id": item["id"], "family": item.get("family"),
                   "question_type": item["question"].get("type"),
                   "error": f"{type(e).__name__}: {e}",
                   "latency_s": round(time.time() - t0, 2)}
        rows.append(row)
        lat.append(row["latency_s"])
        if i % 5 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)
        if i % args.checkpoint_every == 0:
            json.dump({"rows": rows}, open(args.out, "w", encoding="utf-8"),
                      ensure_ascii=False)

    good = [r for r in rows if r.get("correct") is not None]
    report = {"n": len(rows), "scored": len(good),
              "errors": sum(1 for r in rows if r.get("error"))}
    if good:
        report["accuracy"] = round(sum(1 for r in good if r["correct"]) / len(good), 4)
        # Multiclass Brier (summed over every label), matching eval_buckets.
        # A gold-only term is a different quantity and ~4-5x smaller, so the two
        # must not be compared across scripts.
        bvals = [sum((v - (1.0 if k == r["expected"] else 0.0)) ** 2
                     for k, v in r["probs"].items())
                 for r in good if r.get("probs")]
        report["brier"] = round(sum(bvals) / len(bvals), 4) if bvals else None
        report["ece"] = round(ece(good), 4)
    if lat:
        report["p50_latency_s"] = round(statistics.median(lat), 2)

    by_bucket, by_family = {}, {}
    for r in good:
        by_bucket.setdefault(r["bucket"], []).append(r)
        by_family.setdefault(r.get("family") or "unknown", []).append(r)
    report["by_bucket"] = {k: {"n": len(v),
                               "acc": round(sum(1 for r in v if r["correct"]) / len(v), 4)}
                           for k, v in sorted(by_bucket.items())}
    report["by_family"] = {k: {"n": len(v),
                               "acc": round(sum(1 for r in v if r["correct"]) / len(v), 4)}
                           for k, v in sorted(by_family.items(),
                                              key=lambda kv: sum(1 for r in kv[1]
                                                                 if r["correct"]) / len(kv[1]))}

    json.dump({"rows": rows, "report": report},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
