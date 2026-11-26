"""Score an evaluation run with JevBench's published axes.

IMPORTANT — what this is and is not:
  These are JevBench's *formulas* applied to our own 231-item validation set.
  They are NOT leaderboard scores. The official benchmark draws its items from
  a sealed pool we do not have; our set imitates the public split's shape but
  the questions are different. Report these as "official-formula estimate on our
  validation set", never as a leaderboard figure.

Formulas (JevBench methodology + arXiv 2610.17205):
  Intelligence I = 100 * (acc - chance) / (1 - chance), clipped at 0, per
                   difficulty tier, tier-weighted.
  Calibration  C = 100 * (1 - ECE / 0.5), averaged per question type:
                   choice -> top-answer ECE
                   noul   -> ECE of P(yes)
                   score  -> mean(top-answer ECE, 100 * (1 - RPS))
  Capability     = (I + C) / 2

Our tier weights are renormalised: we have easy/standard/hard but no judge tier,
so the judge weight is redistributed rather than silently scored as zero.

  python scripts/score_official.py --result results/base_qwen35_08b.json
"""
import argparse
import collections
import glob
import json
import os
import re

TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "hard": 0.30, "judge": 0.28}
N_BINS = 10


def tier_of(item_id: str) -> str:
    """Difficulty from the id prefix; unlabelled authored items are standard."""
    m = re.match(r"^(easy|standard|judge|hard)-", item_id)
    if m:
        return m.group(1)
    return "standard"  # UNVERIFIED for the 72 'original-authored' items


def load_types(eval_dir=None):
    """Map id -> question type from the source eval files.

    The directory must be passed in: this script runs on a remote box where the
    authoring machine's scratch path does not exist, and a silent empty map
    would make every item default to 'standard' and wreck the tier weights.
    """
    out = {}
    dirs = [eval_dir] if eval_dir else [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")]
    for d in dirs:
        for f in glob.glob(os.path.join(d, "pub_*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                out[r["id"]] = r["question"]["type"]
    return out


def item_chance(row, qtype):
    """Probability of a correct answer by guessing."""
    if qtype == "noul":
        return 0.5
    n = len(row["probs"]) if row.get("probs") else 2
    return 1.0 / n


def ece_rows(pairs, n_bins=N_BINS):
    """Expected calibration error over (confidence, correct) pairs."""
    if not pairs:
        return None
    buckets = collections.defaultdict(list)
    for conf, hit in pairs:
        i = min(int(conf * n_bins), n_bins - 1)
        buckets[i].append((conf, hit))
    total = len(pairs)
    e = 0.0
    for b in buckets.values():
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, h in b if h) / len(b)
        e += (len(b) / total) * abs(avg_conf - avg_acc)
    return e


def rps(row):
    """Ranked probability score for an ordinal answer, normalised to 0..1.

    Standard RPS divides the squared CDF difference by (K-1) so the scale does
    not depend on how many grades the rubric has; without that division a
    5-grade item scores up to 4 and drives the calibration term negative.
    """
    labels = sorted(row["probs"], key=lambda k: float(k))
    gold_idx = labels.index(str(row["expected"]))
    cum_p = cum_g = 0.0
    total = 0.0
    for i, lab in enumerate(labels[:-1]):  # last category contributes nothing
        cum_p += row["probs"][lab]
        cum_g += 1.0 if i >= gold_idx else 0.0
        total += (cum_p - cum_g) ** 2
    return total / (len(labels) - 1) if len(labels) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--json", action="store_true", help="emit machine-readable only")
    ap.add_argument("--eval", default=None,
                    help="dir holding pub_*.jsonl; needed off the authoring machine")
    args = ap.parse_args()

    types = load_types(args.eval)
    data = json.load(open(args.result))
    rows = [r for r in data["rows"] if r.get("correct") is not None and r.get("probs")]
    # Result rows carry their own question_type; the id lookup only covers the
    # public set, so relying on it alone silently mislabels every heldout item as
    # "choice" and applies the wrong calibration branch.
    for r in rows:
        r["_qtype"] = r.get("question_type") or types.get(r["id"]) or "choice"
    if not rows:
        raise SystemExit("no scored rows found")

    # ---------------------------------------------------------------- intelligence
    by_tier = collections.defaultdict(list)
    for r in rows:
        by_tier[tier_of(r["id"])].append(r)

    tier_rows = []
    num = den = 0.0
    for tier, group in sorted(by_tier.items()):
        acc = sum(1 for r in group if r["correct"]) / len(group)
        chance = sum(item_chance(r, r["_qtype"])
                     for r in group) / len(group)
        intel = max(0.0, 100.0 * (acc - chance) / (1.0 - chance))
        w = TIER_WEIGHTS.get(tier, 0.0)
        num += w * intel
        den += w
        tier_rows.append((tier, len(group), acc, chance, intel, w))
    intelligence = num / den if den else 0.0

    # ---------------------------------------------------------------- calibration
    cal_parts = {}
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["_qtype"]].append(r)

    choice = by_type.get("choice", [])
    if choice:
        pairs = [(max(r["probs"].values()), bool(r["correct"])) for r in choice]
        e = ece_rows(pairs)
        cal_parts["choice"] = 100.0 * (1.0 - e / 0.5)

    noul = by_type.get("noul", [])
    if noul:
        # ECE of P(yes) against the actual yes/no outcome.
        pairs = [(r["probs"].get("yes", 0.5),
                  1 if str(r["expected"]) == "yes" else 0) for r in noul]
        e = ece_rows(pairs)
        cal_parts["noul"] = 100.0 * (1.0 - e / 0.5)

    score = by_type.get("score", [])
    if score:
        pairs = [(max(r["probs"].values()), bool(r["correct"])) for r in score]
        e = ece_rows(pairs)
        r_mean = sum(rps(r) for r in score) / len(score)
        cal_parts["score"] = 0.5 * (100.0 * (1.0 - e / 0.5)) + 0.5 * (100.0 * (1.0 - r_mean))

    calibration = sum(cal_parts.values()) / len(cal_parts) if cal_parts else 0.0
    capability = (intelligence + calibration) / 2.0

    # ---------------------------------------------------------------- local extras
    acc_all = sum(1 for r in rows if r["correct"]) / len(rows)

    noul_bal = None
    if noul:
        yes = [r for r in noul if str(r["expected"]) == "yes"]
        no = [r for r in noul if str(r["expected"]) == "no"]
        yes_rec = (sum(1 for r in yes if r["correct"]) / len(yes)) if yes else None
        no_rec = (sum(1 for r in no if r["correct"]) / len(no)) if no else None
        if yes_rec is not None and no_rec is not None:
            noul_bal = (yes_rec + no_rec) / 2

    score_mae = None
    if score:
        def expected_val(r):
            return sum(float(k) * p for k, p in r["probs"].items())
        score_mae = sum(abs(expected_val(r) - float(r["expected"])) for r in score) / len(score)

    brier = sum(sum((p - (1.0 if k == str(r["expected"]) else 0.0)) ** 2
                    for k, p in r["probs"].items()) for r in rows) / len(rows)

    out = {
        "source": args.result,
        "n": len(rows),
        "official_formula": {
            "intelligence": round(intelligence, 2),
            "calibration": round(calibration, 2),
            "capability": round(capability, 2),
            "calibration_by_type": {k: round(v, 2) for k, v in cal_parts.items()},
            "tiers": [
                {"tier": t, "n": n, "acc": round(a, 4), "chance": round(c, 4),
                 "intelligence": round(i, 2), "weight": w}
                for t, n, a, c, i, w in tier_rows
            ],
        },
        "local": {
            "accuracy": round(acc_all, 4),
            "noul_balanced_accuracy": round(noul_bal, 4) if noul_bal is not None else None,
            "brier_multiclass": round(brier, 4),
            "score_mae": round(score_mae, 4) if score_mae is not None else None,
            "by_type": {t: {"n": len(v), "acc": round(sum(1 for r in v if r["correct"]) / len(v), 4)}
                        for t, v in sorted(by_type.items())},
        },
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"result: {args.result}   n={out['n']}")
    print("\nJevBench-formula estimate (our validation set, NOT a leaderboard score)")
    print(f"  Intelligence {out['official_formula']['intelligence']:6.2f}")
    print(f"  Calibration  {out['official_formula']['calibration']:6.2f}"
          f"   {out['official_formula']['calibration_by_type']}")
    print(f"  Capability   {out['official_formula']['capability']:6.2f}")
    print("\n  per tier:")
    for t in out["official_formula"]["tiers"]:
        print(f"    {t['tier']:9s} n={t['n']:3d} acc={t['acc']:.3f} "
              f"chance={t['chance']:.3f} -> I={t['intelligence']:6.2f} (w={t['weight']})")
    print("\nlocal diagnostics:")
    loc = out["local"]
    print(f"  accuracy               {loc['accuracy']:.4f}")
    print(f"  noul balanced accuracy {loc['noul_balanced_accuracy']}")
    print(f"  brier (multiclass)     {loc['brier_multiclass']:.4f}")
    print(f"  score MAE              {loc['score_mae']}")
    print(f"  by type                {loc['by_type']}")


if __name__ == "__main__":
    main()
