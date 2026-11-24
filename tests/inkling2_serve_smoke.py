"""Check a running native Inkling tqserve via its public chat API."""
import argparse
import json
import time
import urllib.request


def assess(base):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/health", timeout=5) as response:
        health = json.load(response)
    prompt = "Reply with the single uppercase word PELICAN."
    tasks = [
        ("short", prompt, "PELICAN", 0, 0),
        ("long", "Archive of repeated entries:\n" + " pad" * 1300
         + "\nApproved code is BAY-328. Return only the approved code.", "BAY-328", 1024, 0),
        ("fresh", prompt, "PELICAN", 0, 0),
        ("sampled", prompt, "PELICAN", 0, 0.6),
        ("thinking", "What is five plus seven? Reply with the number only.", "12", 0, 0),
    ]
    report = {"health": health, "cases": []}
    for name, text, expected, minimum, temperature in tasks:
        data = {"model": "tqllm", "messages": [{"role": "user", "content": text}], "temperature": temperature,
                "top_k": 20, "top_p": 0.95, "seed": 42, "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 32, "stream": True,
                "stream_options": {"include_usage": True}}
        if name == "thinking":
            data.update(reasoning_effort="xhigh", reasoning_budget_tokens=192, max_tokens=256,
                        chat_template_kwargs={"enable_thinking": True})
        request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(data).encode(),
                                         {"Content-Type": "application/json"})
        start = time.monotonic()
        parts, usage, timings, finish = [], {}, {}, None
        reasoning_chars = 0
        with opener.open(request, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                wire = line[5:].strip()
                if wire == b"[DONE]":
                    break
                event = json.loads(wire)
                if "error" in event:
                    raise RuntimeError(event["error"])
                usage = event.get("usage") or usage
                timings = event.get("timings") or timings
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    parts.append(delta.get("content") or "")
                    reasoning_chars += len(delta.get("reasoning_content") or "")
                    finish = choice.get("finish_reason") or finish
        answer = "".join(parts).strip()
        row = dict(case=name, answer=answer, expected=expected, usage=usage, timings=timings, finish=finish,
                   seconds=round(time.monotonic()-start, 2), reasoning_chars=reasoning_chars,
                   ok=(answer == expected and usage.get("prompt_tokens", 0) > minimum and finish == "stop"
                       and (reasoning_chars > 0) == (name == "thinking")))
        report["cases"].append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="HTTP URL of a running Inkling server")
    args = parser.parse_args()
    results = assess(args.base.rstrip("/"))
    raise SystemExit(0 if all(case["ok"] for case in results["cases"]) else 1)
