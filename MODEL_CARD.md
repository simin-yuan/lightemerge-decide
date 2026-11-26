# Model card — LightEmerge-Decide-0.8B (v1.0)

## Summary

`LightEmerge-Decide-0.8B` is a small language model fine-tuned to answer **structured decision
questions**: a state description plus a question with a closed candidate set, returning a
probability distribution over the candidates. It handles three question types:

| type | shape | example |
|---|---|---|
| `choice` | pick 1 of N labels | intent classification over 5 service categories |
| `noul` | binary | "is this covered under the policy?" |
| `score` | ordinal level (e.g. 0–4) | "how severe is this?" |

The distribution is the model's own next-token distribution restricted to the candidate label
tokens — there is no separate calibration layer, temperature fit, or ensemble.

## Model details

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B-Base` (Apache-2.0) |
| Parameters | 0.8B (LoRA merged into full weights; the released checkpoint is dense, no PEFT dependency at inference) |
| Fine-tuning | LoRA; the adapter recipe (rank, alpha, target modules, optimiser settings) is proprietary and not released |
| Precision / hardware | bfloat16 for training and evaluation, single consumer GPU (24 GB) |
| Tokenizer / context | inherited from the base model, max sequence length 4096 |
| Language | English prompts, English labels |

## Training

A single supervised run over a proprietary mix of 10,000 examples — one epoch, one checkpoint, no
checkpoint search or early stopping. The objective works on the candidate distribution rather than
on plain cross-entropy: soft targets from a larger teacher model (`Qwen/Qwen3.5-27B`, Apache-2.0)
are mixed with direct distribution losses computed over the candidate slice.

The training code, the exact hyper-parameters, the loss weights and the per-source data mix are
**not released**. What ships is enough to run, evaluate and reproduce the measurements below; it is
not enough to re-train the model. Source names, licences and the zero-overlap check against the
evaluation sets are in [docs/DATA_SOURCES.md](https://github.com/simin-yuan/lightemerge-decide/blob/master/docs/DATA_SOURCES.md).

## Evaluation

### Rulers and what they are

Four local evaluation sets were used. **None of them is the official leaderboard draw**, so the
scores below must not be compared with published leaderboard numbers:

| ruler | n | what it is |
|---|---|---|
| `judge260` | 260 | 260 response pairs from the HelpSteer2 validation split; both responses answer the same prompt and the label is which one humans rated more helpful (pairs whose helpfulness gap was under 2 were dropped) |
| `valhard` | 150 | held-out hard set: question templates and domains authored by us, never seen in training |
| `pub231` | 231 | the public decision-benchmark's own items, all three of its tiers (easy 48 / standard 72 / hard 111) |
| `pub434` | 434 | our own synthetic holdout over grant and credit-review rules (deterministic rule generator); never trained on, text not redistributed |

### Metrics

Following the aggregation used by the public decision-model leaderboard this release is aimed at:

- **Intelligence** `I = 100 · (acc − chance) / (1 − chance)`, clipped at 0, aggregated per tier with
  tier weights.
- **Calibration** `C = 100 · (1 − ECE / 0.5)`, averaged over question types; for `score` rows the
  mean of top-answer ECE and `100 · (1 − RPS)`.
- **Capability** `Cap = (I + C) / 2`.

All four sets were evaluated on GPU in bfloat16, the same precision used in training.

| ruler | n | prior-baseline acc | prior-baseline Int | prior-baseline Cal | prior-baseline Cap | **v1.0 acc** | **v1.0 Int** | **v1.0 Cal** | **v1.0 Cap** |
|---|---|---|---|---|---|---|---|---|---|
| judge260 | 260 | 0.4923 | 0.00 | 33.14 | 16.57 | 0.5077 | 1.54 | 80.26 | **40.90** |
| valhard | 150 | 0.3333 | 0.50 | 17.28 | 8.89 | 0.3867 | 8.46 | 25.03 | **16.75** |
| pub231 | 231 | 0.5714 | 37.90 | 59.24 | 48.57 | 0.5801 | 39.03 | 64.28 | **51.65** |
| pub434 | 434 | 0.6820 | 46.09 | 40.65 | 43.37 | 0.6567 | 41.80 | 43.67 | **42.73** |
| **mean Cap** | | | | | 29.35 | | | | **38.01** |

Per-ruler source files: `evidence/base_*.json` (prior baseline), `evidence/arm3_*.json` (this release).

`prior-baseline` is an **earlier internal fine-tune of the same base model**, not the raw base
weights; it is the checkpoint this release is compared against. The raw `Qwen/Qwen3.5-0.8B-Base` is
deliberately not tabulated: its run left 35 of 266 `pub231` items unscored because it did not emit a
parseable answer, so its numbers are not comparable with the rows below.

Read the table honestly: this release raises mean Cap from 29.35 to 38.01 over the prior baseline,
but **not on every
ruler**. `pub434` accuracy drops from 0.6820 to 0.6567, and on `pub231` the accuracy gain is 0.9
points while the calibration gain is 5.0 points — most of the Cap improvement there is calibration,
not decisions. On `judge260` the model reaches Cap 40.90 mostly through calibration as well
(Int 1.54, i.e. barely above chance-corrected zero).

### Latency

p50 0.05 s per single-decision round trip, measured on a single 24 GB consumer GPU, bfloat16, batch 1, local
loopback (no network). For reference, the leaderboard ranks on `2 × measured + 150 ms`, which for
this model is ≈ 0.25 s.

### Merge equivalence

The released checkpoint is `merge_and_unload()` of the LoRA adapter into the base weights. Two
checks, both on real evaluation prompts (one prompt per question type, chosen from `pub231` and
`valhard`):

1. **Arithmetic equivalence, float32** — merged model vs adapter-attached model: candidate logits
   agree to `max |Δlogit| = 7.63e-06`, distributions to `max |Δprob| = 9.12e-07`. Record:
   `evidence/merge_equivalence.json`.
2. **Released artifact, bfloat16** — the on-disk bf16 checkpoint reloaded from scratch (no adapter,
   no PEFT) reproduces the fp32 reference decisions exactly (same argmax on all three prompts) with
   probabilities within `max |Δprob| = 1.20e-02`, which is bf16 rounding, not a modelling change.
   Record: `evidence/reload_check.json`.

## Intended use

- Anywhere a small model must produce a **decision plus an honest confidence** over a fixed label
  set: routing, triage, policy eligibility, severity grading.
- The probabilities are meant to be consumed downstream (thresholding, abstention, human review
  queues), not just the argmax.

## Limitations

- **English only**, and only questions whose answers are a subset of the given labels.
- 0.8B capacity: on hard unseen question templates the model is close to chance-corrected zero
  (`valhard` Int 8.46). It does not reason its way out of a domain shift; it reports that it is
  unsure, which is often the useful behaviour but is not competence.
- **High calibration is not high accuracy**: on `judge260` the model's Int is 1.54 (barely above
  chance-corrected zero) while its Cal is 80.26 — the high Cal means its probabilities track the base
  rates honestly, not that its decisions are good. Do not quote Cap without Int next to it.
- Calibration degrades when examples are drawn from a different distribution: adding 14,600 more
  rows of a different mixture (the "final" ablation, not released) lowered local `pub231`
  calibration from 64.28 to 54.98 while raising accuracy from 0.5801 to 0.6061. The two axes trade
  off; this release chooses calibration.
- Not evaluated on the official leaderboard draw. Any official submission will be scored by the
  maintainers' harness, not by us.

## Licence

Apache-2.0. See [NOTICE](NOTICE).

## Author

Simin Yuan (LightEmerge Ltd.)
