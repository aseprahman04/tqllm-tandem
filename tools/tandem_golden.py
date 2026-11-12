#!/usr/bin/env python3
"""Tandem (multi-GPU fork of tqLLM): token-equality check for the layer split.

Record greedy answers from a running tqserve, once on one GPU and once split across two, then compare them. The split
must not change a single token. Standard library only.

    python tools/tandem_golden.py --out one.json            # server on one GPU
    python tools/tandem_golden.py --out two.json            # same model and options, TQLLM_SPLIT=K on two GPUs
    python tools/tandem_golden.py --compare one.json two.json
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list. Output only code.",
    "Explain in plain English why the sky is blue, in about 150 words.",
    "List the first 20 prime numbers, comma separated, then explain the sieve of Eratosthenes in three sentences.",
    "Write a TypeScript class for a min-heap with push, pop and peek, with JSDoc comments.",
]


def ask(url, prompt, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def record(args):
    out = []
    for p in PROMPTS:
        j = ask(args.url, p, args.max_tokens)
        t = j.get("timings", {})
        out.append({"prompt": p, "content": j["choices"][0]["message"]["content"], "timings": t})
        print(f"{p[:44]}... {t.get('predicted_per_second', 0):.1f} tok/s, "
              f"draft {t.get('draft_n_accepted')}/{t.get('draft_n')}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def compare(a_path, b_path):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    same = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x["content"] == y["content"]:
            same += 1
            continue
        k = next((j for j, (c, d) in enumerate(zip(x["content"], y["content"])) if c != d),
                 min(len(x["content"]), len(y["content"])))
        print(f"DIFF in answer {i} at character {k}: {x['content'][k:k + 60]!r} vs {y['content'][k:k + 60]!r}")
    print(f"{same}/{len(a)} identical")
    return same == len(a) == len(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8086")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("ONE", "TWO"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    if not args.out:
        ap.error("--out FILE to record, or --compare ONE TWO")
    record(args)


if __name__ == "__main__":
    main()
