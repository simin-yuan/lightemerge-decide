"""Single source of truth for prompt rendering.

Both the training loss and the inference readout go through this module.
If these ever drift apart, training optimises one question and evaluation
measures a different one — the numbers would still look plausible, which is
exactly what makes that failure mode dangerous. Import from here, never
re-implement the template.
"""
from __future__ import annotations

from typing import Any

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def letter(i: int) -> str:
    """A..Z, then AA..AZ, BA..: single-column labels stop at 26, and the wire
    contract allows up to 255 options per question."""
    out, n = "", i + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out

CHOICE_HEADER = "Options:"
CHOICE_TAIL = "Answer with the option letter only."
SCORE_HEADER = "Levels:"
SCORE_TAIL = "Answer with the level number only."
NOUL_TAIL = "Answer yes or no."

# Two prompt styles.
#   native   - the flat format this repo started with
#   sections - the layout used by the public Qwen3.5-0.8B decision model that is
#              already on the leaderboard: markdown sections, choice options show
#              both key and description, score levels are lettered rather than
#              numeric. Kept switchable so the difference can be measured rather
#              than argued about.
STYLE = "sections"


def set_style(style: str) -> None:
    global STYLE
    if style not in ("native", "sections"):
        raise ValueError(style)
    STYLE = style


def _render_choice_sections(state: str, instructions: str, options: list[str],
                            descriptions: list | None = None):
    parts = ["### State", state, "### Question", instructions,
             "### Options (pick exactly one)"]
    labs = [letter(i) for i in range(len(options))]
    for i, key in enumerate(options):
        desc = descriptions[i] if descriptions else None
        line = f"{labs[i]}. {key}"
        if desc is not None and str(desc).strip():
            line += f": {desc}"
        parts.append(line)
    parts.append("Answer:")
    return "\n".join(parts), list(options), [f" {lab}" for lab in labs]



def render_choice(state: str, instructions: str, options: list[str],
                  descriptions: list | None = None):
    """options: ordered label list. Returns (prompt, labels, letter tokens)."""
    if STYLE == "sections":
        return _render_choice_sections(state, instructions, options, descriptions)
    labs = [letter(i) for i in range(len(options))]
    lines = [f"{labs[i]}. {lab}" for i, lab in enumerate(options)]
    prompt = (f"{state}\n\n{instructions}\n\n{CHOICE_HEADER}\n"
              + "\n".join(lines) + f"\n{CHOICE_TAIL}\nAnswer:")
    return prompt, list(options), [f" {lab}" for lab in labs]


def render_noul(state: str, instructions: str, true_desc: str | None,
                false_desc: str | None):
    """Returns (prompt, ['yes','no'], [' yes',' no'])."""
    if STYLE == "sections":
        parts = ["### State", state, "### Question", instructions,
                 "### Answer yes or no"]
        if true_desc is not None:
            parts.append(f"yes: {true_desc}")
        if false_desc is not None:
            parts.append(f"no: {false_desc}")
        parts.append("Answer:")
        return "\n".join(parts), ["yes", "no"], [" yes", " no"]
    lines = []
    if true_desc:
        lines.append(f"Answer yes if: {true_desc}")
    if false_desc:
        lines.append(f"Answer no if: {false_desc}")
    body = ("\n".join(lines) + "\n") if lines else ""
    prompt = f"{state}\n\n{instructions}\n\n{body}{NOUL_TAIL}\nAnswer:"
    return prompt, ["yes", "no"], [" yes", " no"]


def render_score(state: str, instructions: str, levels: list):
    """levels: ordered list of level descriptions, index == level number.

    Labels stay numeric (that is what the wire contract expects for keys); only
    the surface tokens differ between styles.
    """
    labels = [str(i) for i in range(len(levels))]
    if STYLE == "sections":
        parts = ["### State", state, "### Question", instructions,
                 "### Levels (pick the one that best describes the state)"]
        for i, lv in enumerate(levels):
            parts.append(f"{LETTERS[i]}. {lv}")
        parts.append("Answer:")
        return ("\n".join(parts), labels,
                [f" {LETTERS[i]}" for i in range(len(levels))])
    lines = [f"{lab}. {levels[i]}" for i, lab in enumerate(labels)]
    # The prompt must end with a trailing space so the model's next token is the
    # digit itself. With "Answer:" and candidates " 0".." 4" the tokenizer splits
    # each into [220, 15..19] and taking the first id gives 220 for all five —
    # five identical candidates, a uniform 0.2 softmax, and a score item that can
    # never be answered correctly.
    prompt = (f"{state}\n\n{instructions}\n\n{SCORE_HEADER}\n"
              + "\n".join(lines) + f"\n{SCORE_TAIL}\nAnswer: ")
    return prompt, labels, [f"{lab}" for lab in labels]


def render_wire(state: str, question: dict[str, Any]):
    """Render straight from a /v1/systemone question object.

    Returns (prompt, labels, candidate_tokens).
    """
    qtype = question.get("type")
    instructions = question.get("instructions") or ""
    crit = question.get("criteria")
    if qtype == "noul":
        true_d = crit.get("true") if isinstance(crit, dict) else None
        false_d = crit.get("false") if isinstance(crit, dict) else None
        return render_noul(state, instructions, true_d, false_d)
    if qtype == "score":
        levels = crit if isinstance(crit, list) else list(range(len(crit or [])))
        return render_score(state, instructions, levels)
    if not isinstance(crit, dict) or not crit:
        raise ValueError("choice question needs a criteria map")
    keys = list(crit.keys())
    return render_choice(state, instructions, keys, [crit[k] for k in keys])


def render_item_cands(state: str, item: dict[str, Any]):
    """Render from a training item, returning the candidate token strings too.

    Returns (prompt, labels, cand_tokens, target_token). ``cand_tokens`` is in
    the same order as ``labels``: the trainer must gather logits at these token
    ids. Slicing ``logits[:, :w]`` instead takes the first w *vocabulary*
    entries (special tokens), which makes the KD term meaningless.
    """
    qtype = item["question_type"]
    instr = item.get("question") or ""
    if qtype == "choice":
        opts = list(item["options"])
        prompt, labels, toks = render_choice(state, instr, opts)
    elif qtype == "noul":
        prompt, labels, toks = render_noul(state, instr, None, None)
    elif qtype == "score":
        levels = item["levels"] if "levels" in item else item["criteria"]
        prompt, labels, toks = render_score(state, instr, levels)
    else:
        raise ValueError(f"unsupported question_type {qtype!r}")
    gold = labels.index(str(item["expected"]))
    return prompt, labels, toks, toks[gold]


def render_item(state: str, item: dict[str, Any]):
    """Render straight from a training item. Returns (prompt, labels, target_token)."""
    prompt, labels, _toks, target = render_item_cands(state, item)
    # Take the target from the same token list the evaluator will score
    # against. Hardcoding " {expected}" here is how train and eval drift
    # apart by a leading space and quietly destroy a question type.
    return prompt, labels, target
