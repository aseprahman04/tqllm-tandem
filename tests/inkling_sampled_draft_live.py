"""Compare single-stream sampled Inkling tokens with the DSpark verifier off and on.

Start the same pack with --mtp in both runs. Record with TQLLM_INKLING2_SAMPLED_DRAFT=0,
restart with it set to 1, and verify against the recorded file using the same arguments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument("reference", type=Path)
    parser.add_argument("--url", required=True, help="Base URL of the dedicated Tandem server")
    parser.add_argument("--model", default="inkling")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text()) if args.mode == "verify" else {}
    cases = {}
    prompts = (
        "Write a Python function that merges overlapping half-open intervals and preserves adjacent intervals.",
        "Implement a small JSON Lines parser in Python and add tests for blank lines and invalid JSON.",
    )
    for thinking, prompt in enumerate(prompts):
        for seed, penalty in [(1, 0.0), (817, 0.0), (817, 0.2)]:
            key = f"thinking-{thinking}-seed-{seed}-penalty-{penalty}"
            body = dict(model=args.model, messages=[dict(role="user", content=prompt)],
                        max_tokens=160, temperature=0.6, top_k=20, top_p=0.95, seed=seed,
                        presence_penalty=penalty, return_token_ids=True,
                        chat_template_kwargs={"enable_thinking": bool(thinking)})
            req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            ids = result["choices"][0]["token_ids"]
            timing = result["timings"]
            count = timing["draft_n"]
            expected_drafts = args.mode == "verify" and penalty == 0
            if not ids or (count > 0) != expected_drafts:
                raise RuntimeError(f"{key}: {len(ids)} output tokens, {count} proposals")
            cases[key] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            print(key, len(ids), count, timing["predicted_per_second"], cases[key][:12], flush=True)
    if args.mode == "record":
        args.reference.write_text(json.dumps(cases, indent=2) + "\n")
    elif cases != reference:
        raise RuntimeError(f"Token mismatch across the two runs: {[k for k in cases if cases[k] != reference.get(k)]}")


if __name__ == "__main__":
    main()
