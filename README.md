# LightEmerge-Decide-0.8B

A 0.8B-parameter model that makes **structured decisions with calibrated probabilities**.
Give it a state description and a question with a fixed set of labels or levels; it returns a
probability distribution over those candidates plus the chosen answer.

- 0.8B parameters (fine-tuned from `Qwen/Qwen3.5-0.8B-Base` with LoRA, merged into full weights)
- Three question types: categorical choice (N labels), binary yes/no, ordinal score (N levels)
- Native distributions — the probabilities come from the model's own next-token distribution over
  the candidate labels, not from a post-hoc temperature fit
- p50 latency 0.05 s per decision on a single RTX 4090D in bfloat16

This is the v1.0 release: one training run, one checkpoint. Nothing was tuned against the
evaluation sets, and the training code and recipe are not part of this release.

## Install

```bash
pip install torch transformers accelerate
```

`accelerate` is not optional on transformers 5.x: the loader imports it in
`transformers/core_model_loading.py` even when you never pass `device_map`. Tested on torch 2.15.0
(CPU) and transformers 5.24.0 with accelerate 1.16.0; the model itself is bf16 weights, no code is
fetched from the hub at load time (`trust_remote_code=False` everywhere).

## Use it as a library

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "simin-yuan/LightEmerge-Decide-0.8B"    # Hugging Face weights repository
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
# Prompt construction and candidate scoring: see scripts/promptfmt.py and scripts/serve_systemone.py
```

## Use it as a service

`scripts/serve_systemone.py` exposes a single-endpoint HTTP service that takes a state string plus
a question object and returns `{"label": ..., "probs": {...}}`. See the docstring at the top of the
file for the wire format.

## Evaluation

All numbers in [MODEL_CARD.md](MODEL_CARD.md) come from `evidence/*.json` and are checked
mechanically by `check_numbers.py`:

```bash
python check_numbers.py            # exits non-zero if any published number lacks evidence
```

Four local rulers were used. **They are our own sets, not the official leaderboard draw**, so the
scores here are not comparable to published leaderboard numbers; see the ruler notes in the model
card before quoting any figure.

## Files

```
scripts/serve_systemone.py   HTTP service (one endpoint, batched candidate scoring)
scripts/promptfmt.py         prompt rendering for the three question types
scripts/score_official.py    Int / Cal / Cap aggregation over an evaluation dump
scripts/eval_local.py        run the four rulers against a checkpoint
evidence/                    raw per-ruler evaluation summaries + merge equivalence record
check_numbers.py             mechanical number checker over README/MODEL_CARD
docs/DATA_SOURCES.md         training data sources, licences, and overlap checks
```

## License and attribution

Code and weights: Apache-2.0. Base model: `Qwen/Qwen3.5-0.8B-Base` (Apache-2.0). See
[NOTICE](NOTICE) for the full attribution list, including the teacher model used to generate soft
targets and the public benchmark whose interface contract this code follows.

## Author

Simin Yuan
