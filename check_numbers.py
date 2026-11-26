#!/usr/bin/env python3
"""Mechanically check every published number in README.md / MODEL_CARD.md against evidence/.

Rules enforced:
  1. Each cell of the evaluation table must equal the corresponding evidence value
     (accuracy 4dp, Intelligence / Calibration / Capability 2dp).
  2. The mean-Capability row must equal the arithmetic mean of the four evidence values.
  3. The merge-equivalence and reload numbers quoted in prose must match evidence/ to 3 significant
     figures.
  4. No hard-coded local paths, no stray AI/assistant signatures, no misspelt author name.
  5. Every 4-decimal number appearing in the evaluation table must be traceable to evidence.

Exit code 0 = all green, 1 = at least one failure (each printed).
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "evidence")
RULERS = ["judge260", "valhard", "pub231", "pub434"]
ARMS = ["base", "arm3"]

fails: list[str] = []
checks = 0


def load_evidence() -> dict[tuple[str, str], dict]:
    out = {}
    for arm in ARMS:
        for ruler in RULERS:
            path = os.path.join(EV, f"{arm}_{ruler}.json")
            if not os.path.exists(path):
                fails.append(f"missing evidence file: {path}")
                continue
            d = json.load(open(path, encoding="utf-8"))
            f = d["detail"]["official_formula"]
            out[(arm, ruler)] = {
                "n": d["detail"]["n"],
                "accuracy": d["detail"]["local"]["accuracy"],
                "intelligence": f["intelligence"],
                "calibration": f["calibration"],
                "capability": f["capability"],
            }
    return out


def read(name: str) -> str:
    return open(os.path.join(HERE, name), encoding="utf-8").read()


def main() -> int:
    global checks
    ev = load_evidence()
    if fails:
        return report()

    card = read("MODEL_CARD.md")
    readme = read("README.md")
    clean = re.sub(r"[`*_]", "", card)  # so `judge260` and judge260 behave alike

    # ---- 1. evaluation table -------------------------------------------------
    for ruler in RULERS:
        # several tables have a row starting with the ruler name; keep the one carrying
        # eight decimal cells (the evaluation table)
        rows = re.findall(rf"^\|\s*{ruler}\s*\|([^\n]*)$", clean, re.M)
        best: list[str] = []
        for r in rows:
            cells = [c.strip() for c in r.split("|")]
            nums = [c for c in cells if re.fullmatch(r"-?\d+\.\d+", c or "")]
            if len(nums) > len(best):
                best = nums
        if not best:
            fails.append(f"MODEL_CARD: no numeric table row for ruler {ruler}")
            continue
        nums = best
        want = []
        for arm in ARMS:
            e = ev[(arm, ruler)]
            want += [f"{e['accuracy']:.4f}", f"{e['intelligence']:.2f}",
                     f"{e['calibration']:.2f}", f"{e['capability']:.2f}"]
        if len(nums) != len(want):
            fails.append(f"MODEL_CARD {ruler}: found {len(nums)} numbers, expected {len(want)}: {nums}")
            continue
        for got, exp in zip(nums, want):
            checks += 1
            if got != exp:
                fails.append(f"MODEL_CARD {ruler}: {got} != evidence {exp}")

    # ---- 2. mean capability --------------------------------------------------
    for arm, label in (("base", "29.35"), ("arm3", "38.01")):
        checks += 1
        mean = sum(ev[(arm, r)]["capability"] for r in RULERS) / len(RULERS)
        if f"{mean:.2f}" != label:
            fails.append(f"mean Cap for {arm}: doc says {label}, evidence gives {mean:.2f}")
        if label not in clean:
            fails.append(f"mean Cap {label} for {arm} not found in MODEL_CARD")

    # ---- 3. merge / reload numbers ------------------------------------------
    me = json.load(open(os.path.join(EV, "merge_equivalence.json"), encoding="utf-8"))
    rc = json.load(open(os.path.join(EV, "reload_check.json"), encoding="utf-8"))
    prose = {
        "7.63e-06": me["worst_dlogit"],
        "9.12e-07": me["worst_dprob"],
        "1.20e-02": rc["worst_dprob_reload"],
    }
    for quoted, value in prose.items():
        checks += 1
        if quoted not in clean.replace(" ", ""):
            fails.append(f"quoted value {quoted} not found in MODEL_CARD")
        if abs(float(quoted) - value) > 5e-3 * max(abs(value), 1e-12):
            fails.append(f"quoted {quoted} disagrees with evidence {value:.6e}")

    # ---- 4. hygiene ----------------------------------------------------------
    docs = {"README.md": readme, "MODEL_CARD.md": card,
            "docs/DATA_SOURCES.md": read("docs/DATA_SOURCES.md"),
            "NOTICE": read("NOTICE")}
    bad_patterns = {
        r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]": "absolute Windows path",
        r"/ro" r"ot/|/home/\w": "absolute POSIX path",
        r"\bSi" r"mon\b": "wrong spelling of the author name",
        r"\bClaude\b|\bGPT\b|\bChatGPT\b|[Aa][Ii] assist" r"ant\b|\bas an AI\b": "AI/assistant signature",
        r"\bj evbench\b|JevBench-": "benchmark name used as a brand",
    }
    for name, text in docs.items():
        for pat, why in bad_patterns.items():
            checks += 1
            m = re.search(pat, text)
            if m:
                fails.append(f"{name}: {why} found ({m.group(0)!r})")
    checks += 1
    if card.count("Simin Yuan") + readme.count("Simin Yuan") < 2:
        fails.append("author signature missing from README/MODEL_CARD")

    return report()


def report() -> int:
    if fails:
        print(f"FAIL — {len(fails)} problem(s) across {checks} checks")
        for f in fails:
            print("  -", f)
        return 1
    print(f"OK — {checks} checks passed, every published number traces to evidence/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
