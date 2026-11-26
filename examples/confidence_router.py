#!/usr/bin/env python3
"""Confidence-gated routing, end to end.

The model returns a distribution over the options you gave it. This example turns
that distribution into a routing decision: above the threshold the answer goes
straight through, below it the item is handed to a human or a larger model.

It talks to `scripts/serve_systemone.py`, so start that first:

    python scripts/serve_systemone.py \
        --base simin-yuan/LightEmerge-Decide-0.8B --device cpu --dtype float32 --port 8000

then:

    python examples/confidence_router.py

Nothing here claims a cost reduction. How much a gate saves depends on your mix of
easy and hard cases and on what escalation costs you — measure that on your own
traffic before you quote a number.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

TICKET_ROUTING = dict(
    state=("用户留言：我刚下单的咖啡机，能改成明天下午送到公司前台吗？"
           "订单号尾号 8821，前台 17:30 下班。"),
    question=dict(
        type="choice",
        instructions="这条工单应该分派给哪个团队？",
        criteria={
            "物流催单": "查发货进度、联系承运商、给用户新的到达时间",
            "退款处理": "订单取消与退款流程",
            "投诉升级": "涉及服务态度、承诺未兑现或重复投诉",
            "产品咨询": "询问功能、参数、使用方法",
            "其他": "无法归入以上任一类",
        },
    ),
)

EXPENSE_CHECK = dict(
    state=("报销单：2026-11-20 因客户拜访产生的市内出租车费 86 元，附电子发票一张，"
           "拜访记录已填写。无出差审批单。"),
    question=dict(
        type="noul",
        instructions="这笔费用是否符合差旅报销政策？",
        criteria={"true": "市内交通费，单据齐全，属于差旅报销范围",
                  "false": "缺少出差审批单，或不属于差旅范围"},
    ),
)

SEVERITY = dict(
    state=("核心支付链路全量失败，所有渠道订单 100% 无法支付，已持续 12 分钟，"
           "客服收到大量用户投诉，重试无恢复迹象。"),
    question=dict(
        type="score",
        instructions="这个线上异常应该定为几级？",
        criteria=["无影响，可忽略", "轻微，需要观察", "中等，当日处理",
                  "严重，需要立即响应", "灾难性，需要全量止损"],
    ),
)


def ask(base_url: str, state: str, question: dict, timeout: float = 120.0) -> dict:
    body = json.dumps({"state": state, "model": "systemone",
                       "questions": {"decision": question}}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def top(answer: dict) -> tuple[str, float]:
    """(label, probability) of the most likely option, whatever the question type."""
    if answer["type"] == "noul":
        p_yes = float(answer["noul"])
        return ("yes", p_yes) if p_yes >= 0.5 else ("no", 1.0 - p_yes)
    probs = answer["probabilities"]
    label = max(probs, key=lambda k: probs[k])
    return str(label), float(probs[label])


def route(answer: dict, threshold: float) -> dict:
    label, p = top(answer)
    return {"label": label, "p": p, "gate": "auto" if p >= threshold else "escalate"}


def escalate(item: dict, result: dict) -> str:
    """Where a below-threshold item goes.

    In production this is usually one of: a big-model call, a human queue, or a
    rule-based fallback. Keep the branch explicit so the low-confidence path is
    observable — silently answering anyway defeats the point of the gate.
    """
    return (f"escalate: p={result['p']:.3f} < threshold, "
            f"model suggested {result['label']!r} but does not stand behind it")


def main() -> int:
    ap = argparse.ArgumentParser(description="confidence-gated routing demo")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="serve_systemone.py base url")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="below this, hand the item to a human or a larger model")
    args = ap.parse_args()

    cases = [("工单分流", TICKET_ROUTING), ("报销核查", EXPENSE_CHECK), ("事故定级", SEVERITY)]
    print(f"threshold={args.threshold:.2f}   server={args.url}\n")

    for name, case in cases:
        try:
            resp = ask(args.url, case["state"], case["question"])
        except urllib.error.URLError as e:
            print(f"{name}: cannot reach {args.url} ({e.reason}) — is serve_systemone.py running?")
            return 2

        answer = resp["answers"]["decision"]
        result = route(answer, args.threshold)
        dist = {k: round(v, 3) for k, v in
                ({"yes": float(answer["noul"]), "no": 1.0 - float(answer["noul"])}
                 if answer["type"] == "noul" else answer["probabilities"]).items()}
        print(f"{name}  [{answer['type']}]")
        print(f"  distribution: {json.dumps(dist, ensure_ascii=False)}")
        print(f"  top: {result['label']!r} p={result['p']:.3f}")
        if result["gate"] == "auto":
            print(f"  -> auto-pass, p >= {args.threshold:.2f}")
        else:
            print("  -> " + escalate(case, result))
        print()

    print("The gate is the product. A model that answers everything at p=0.5 is worse")
    print("than one that says p=0.92 here and hands the rest back to you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
