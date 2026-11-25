"""Compare seeded TP replies against its full-vocabulary head on the same checkpoint.

Start tqserve with TQLLM_TP=1 and TQLLM_DEBUG=tp_nohead. Run this script with
--record reference.json. Restart without tp_nohead and pass --compare reference.json.
Both runs use the same model, drafter, device order, flags and server URL.
"""
import argparse
import json
import urllib.request
from pathlib import Path


def complete(base, messages, temperature, top_k):
    body = dict(model="tqllm", messages=messages, temperature=temperature, top_k=top_k,
                top_p=0.95, seed=42, reasoning_effort="none", max_tokens=192, stream=False)
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        choice = json.load(response)["choices"][0]
    if choice["finish_reason"] != "stop" or not choice["message"]["content"]:
        raise AssertionError(f"Incomplete response with temperature={temperature}, top_k={top_k}")
    return choice["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of a running TP server")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=Path)
    mode.add_argument("--compare", type=Path)
    args = parser.parse_args()
    replies = {}
    for name, prompt in [
        ("copy", "Copy the uppercase word exactly once and nothing else: PELICAN"),
        ("sum", "Output only the decimal integer equal to 37 plus 28."),
        ("code", "Write a Python function that returns the larger of two integers. Output only the function."),
    ]:
        messages = [{"role": "user", "content": prompt}]
        for label, temperature, top_k in [("greedy", 0, 20), ("sampled", 0.6, 20), ("top1", 0.6, 1)]:
            replies[name + "/" + label] = complete(args.url.rstrip("/"), messages, temperature, top_k)
    if args.record:
        args.record.write_text(json.dumps(replies, indent=2), encoding="utf-8")
        return
    reference = json.loads(args.compare.read_text(encoding="utf-8"))
    if replies != reference:
        changed = sorted(k for k in set(reference) | set(replies) if reference.get(k) != replies.get(k))
        raise AssertionError("Replies differ from full-vocabulary reference: " + ", ".join(changed))
    print(f"Matched all {len(replies)} reference replies.")


if __name__ == "__main__":
    main()
