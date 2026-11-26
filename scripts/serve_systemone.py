#!/usr/bin/env python3
"""JevBench-compatible /v1/systemone server backed by single-pass logit readout.

Wire contract (verified against jevbench adapters/typesafe.py on 2026-11-24):

  POST /v1/systemone
    {"state": str, "model": str,
     "questions": {"decision": {"type": "choice|score|noul",
                                "instructions": str,
                                "criteria": <map|list|null>}}}
  -> {"answers": {"decision": {...}}, "model": str, "usage": {...}}

  answer shapes:
    noul   {"type":"noul","noul": p_yes}
    choice {"type":"choice","choice": <label>, "probabilities": {label: p}}
    score  {"type":"score","probabilities": {"<level index>": p}}

Readout: one forward pass, no generation. The state and instructions are rendered,
options are mapped to single letters A/B/C..., and the next-token distribution over
those letters is renormalised. noul uses the ` yes` / ` no` tokens instead. This is
the same family of method as the published Qwen3.5-0.8B entry on JevBench, which
makes it a fair baseline for what we train.

Nothing here truncates: if the rendered prompt exceeds --max-len the request fails
loudly rather than silently cutting the state in half.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Reader:
    """Holds the model and turns one request into a probability distribution."""

    def __init__(self, base: str, adapter: str | None, max_len: int,
                 dtype: str = "bfloat16", device: str = "cuda"):
        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        # No device_map here on purpose: device_map needs `accelerate` installed, and a
        # clean container that only pip-installed torch+transformers fails at import time
        # (transformers 5.x raises ImportError). .to(device) has no such dependency.
        self.model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch_dtype,
            trust_remote_code=False).to(device).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.max_len = max_len
        self.base = base
        self.adapter = adapter

    def _resolve(self, tokens: list[str]) -> list[int]:
        """Map candidate surface strings to single token ids (first token wins)."""
        return self._resolve_ids(tokens)

    def _resolve_ids(self, tokens: list[str]) -> list[int]:
        """Map candidate strings to single token ids, asserting both invariants.

        Two failure modes have already cost a full eval round:
          - a candidate that encodes to several tokens silently uses the first
            one (" 0" -> [220, 15] -> 220, the space), and
          - several candidates collapsing onto the same id, which makes softmax
            return a uniform distribution so the item cannot be answered at all.
        Both are cheap to check and fatal to miss, so they are hard errors here
        rather than something to notice later in the scores.
        """
        ids = []
        for t in tokens:
            enc = self.tok.encode(t, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(
                    f"candidate {t!r} encodes to {len(enc)} tokens {enc}; "
                    "every candidate must be exactly one token, otherwise the "
                    "logit readout scores the wrong position")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"candidate ids collide: {list(zip(tokens, ids))}; "
                "colliding ids make the distribution uniform by construction")
        return ids

    # ------------------------------------------------------------ rendering

    def render(self, state: str, q: dict[str, Any]):
        """Delegate to the shared formatter so train and serve cannot drift.

        Returns (prompt, labels, cand_ids). ``cand_ids`` is ``None`` when some
        label needs more than one token (any question above 26 options): those
        items are scored per option by :meth:`probs_multi` instead.
        """
        from promptfmt import render_wire
        prompt, labels, cand_tokens = render_wire(state, q)
        try:
            return prompt, labels, self._resolve(cand_tokens)
        except ValueError:
            return prompt, labels, None, cand_tokens

    @torch.inference_mode()
    def probs_multi(self, prompt: str, token_strings: list[str]):
        """Per-option scoring for labels that are not a single token.

        The prompt is encoded once and every extra label token is fed through
        the cache, so the cost is one long forward plus a few one-token steps.
        """
        import math
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        n = enc["input_ids"].shape[1]
        if n > self.max_len:
            raise ValueError(f"prompt is {n} tokens, above --max-len {self.max_len}; "
                             "refusing to truncate")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        try:
            out = self.model(**enc, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = self.model(**enc, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        scores = []
        for text in token_strings:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            total, cur, pk = 0.0, logits, past
            for j, tid in enumerate(ids):
                total += torch.log_softmax(cur.float(), dim=-1)[tid].item()
                if j + 1 < len(ids):
                    step = torch.tensor([[tid]], device=self.model.device)
                    o = self.model(input_ids=step, past_key_values=pk, use_cache=True)
                    cur, pk = o.logits[0, -1], o.past_key_values
            scores.append(total)
        m = max(scores)
        ex = [math.exp(x - m) for x in scores]
        ssum = sum(ex)
        return [e / ssum for e in ex], n

    # ------------------------------------------------------------ inference

    @torch.inference_mode()
    def probs(self, prompt: str, labels: list[str], cand_ids: list[int]):
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        n = enc["input_ids"].shape[1]
        if n > self.max_len:
            raise ValueError(f"prompt is {n} tokens, above --max-len {self.max_len}; "
                             "refusing to truncate")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        # Only materialise the last position: full [vocab x seq] logits cost
        # ~2.5 GB per request at 4096 tokens and OOM-thrashed this box.
        try:
            out = self.model(**enc, logits_to_keep=1)
        except TypeError:
            # transformers < 4.50 has no logits_to_keep; fall back to full logits.
            out = self.model(**enc)
        logits = out.logits[0, -1]
        picked = torch.tensor([logits[i] for i in cand_ids], device=logits.device)
        p = torch.softmax(picked.float(), dim=-1).tolist()
        return p, n


def make_handler(reader: Reader):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # keep the console clean
            pass

        def do_GET(self):
            if self.path in ("/health", "/"):
                self._json(200, {"ok": True, "model": reader.base,
                                 "adapter": reader.adapter})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/systemone"):
                self._json(404, {"error": "use /v1/systemone"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:
                self._json(400, {"error": f"bad json: {e}"})
                return

            t0 = time.perf_counter()
            state = body.get("state")
            questions = body.get("questions") or {}
            # `state` may be a string or a JSON object (Decision Index format), never a guess.
            if isinstance(state, dict):
                # A raw json.dumps() buries list-valued fields (policy lists, tool
                # schemas, turn histories) in one unbroken blob, which is not the
                # shape this model was trained on. state_to_text renders dict
                # states as readable bullets — the same conversion eval_local.py
                # uses, so the served path and the evaluated path agree.
                from eval_buckets import state_to_text
                state = state_to_text(state)
            if not isinstance(state, str) or not isinstance(questions, dict):
                self._json(400, {"error": "need state (string or object) and a questions map"})
                return
            if not questions:
                # A wrong key name would otherwise return 200 with zero answers and
                # zero tokens, which a benchmark harness reads as "scored nothing,
                # no error" and then reports as a result. Fail loudly instead.
                self._json(400, {"error": "empty questions map; expected "
                                          '{"state": ..., "questions": {"<id>": {"type": ...}}}'})
                return

            answers, ntok = {}, 0
            for key, q in questions.items():
                if not isinstance(q, dict):
                    continue
                try:
                    rendered = reader.render(state, q)
                    if len(rendered) == 4:          # labels need >1 token each
                        prompt, labels, _, token_strings = rendered
                        p, n = reader.probs_multi(prompt, token_strings)
                    else:
                        prompt, labels, cand = rendered
                        p, n = reader.probs(prompt, labels, cand)
                except Exception as e:
                    self._json(400, {"error": f"{type(e).__name__}: {e}"})
                    return
                ntok += n
                qtype = q.get("type")
                if qtype == "noul":
                    answers[key] = {"id": key, "type": "noul", "noul": float(p[0])}
                    continue
                probs = {lab: float(x) for lab, x in zip(labels, p)}
                if qtype == "score":
                    answers[key] = {"id": key, "type": "score", "probabilities": probs}
                else:
                    best = labels[max(range(len(p)), key=lambda i: p[i])]
                    answers[key] = {"id": key, "type": "choice", "choice": best,
                                    "probabilities": probs}

            if not answers:
                self._json(400, {"error": "no question produced an answer; each entry needs "
                                          'a "type" of choice|score|noul'})
                return

            self._json(200, {
                "answers": answers,
                "model": f"{reader.base}+{reader.adapter or 'base'}",
                "usage": {"input_tokens": ntok, "output_tokens": 0},
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            })

        def _json(self, code: int, payload: dict):
            blob = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir, optional")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads; 0 leaves torch's default")
    ap.add_argument("--prompt-style", default="sections",
                    choices=["native", "sections"])
    args = ap.parse_args()

    import promptfmt
    promptfmt.set_style(args.prompt_style)

    if args.threads:
        torch.set_num_threads(args.threads)
    # One request at a time: CPU logit readout already saturates every core, so
    # ThreadingHTTPServer only piled up blocked request threads (2953 live on
    # 2026-11-24, ~27 GB private, pages/sec 15k). Serialise instead.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    reader = Reader(args.base, args.adapter, args.max_len, args.dtype,
                    device=args.device)
    srv = HTTPServer(("0.0.0.0", args.port), make_handler(reader))
    print(f"serving {args.base} adapter={args.adapter} on :{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
