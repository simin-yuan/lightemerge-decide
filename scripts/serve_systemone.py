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
        """Delegate to the shared formatter so train and serve cannot drift."""
        from promptfmt import render_wire
        prompt, labels, cand_tokens = render_wire(state, q)
        return prompt, labels, self._resolve(cand_tokens)

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
            q = questions.get("decision")
            if not isinstance(state, str) or not isinstance(q, dict):
                self._json(400, {"error": "need string state and questions.decision"})
                return
            try:
                prompt, labels, cand = reader.render(state, q)
                p, ntok = reader.probs(prompt, labels, cand)
            except Exception as e:
                self._json(400, {"error": f"{type(e).__name__}: {e}"})
                return

            qtype = q.get("type")
            if qtype == "noul":
                ans = {"id": "decision", "type": "noul", "noul": float(p[0])}
            elif qtype == "score":
                ans = {"id": "decision", "type": "score",
                       "probabilities": {lab: float(x) for lab, x in zip(labels, p)}}
            else:
                best = labels[max(range(len(p)), key=lambda i: p[i])]
                ans = {"id": "decision", "type": "choice", "choice": best,
                       "probabilities": {lab: float(x) for lab, x in zip(labels, p)}}

            self._json(200, {
                "answers": {"decision": ans},
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
