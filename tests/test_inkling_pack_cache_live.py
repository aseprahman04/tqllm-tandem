"""Real-server check for Inkling pack cache correctness across compaction."""
import json
import os
import unittest
import urllib.request


URL = os.environ.get("TQSERVE_INKLING_PACK_TEST_URL", "").rstrip("/")
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(route, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + route, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=300) as response:
        return json.load(response)


def ask(messages):
    result = api("/v1/chat/completions", {
        "model": "inkling", "messages": messages, "temperature": 0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    row = max(api("/api/monitor/v1/snapshot")["requests"], key=lambda r: r["id"])
    if row["phase"] != "complete" or row["prompt_tokens"] != result["usage"]["prompt_tokens"]:
        raise AssertionError("Expected the idle test server to report this completed request")
    return result["choices"][0]["message"]["content"], row["prefill_cached_tokens"]


def user(text):
    return {"role": "user", "content": text}


@unittest.skipUnless(URL, "requires a dedicated Inkling pack server with prompt caching")
class InklingPackCache(unittest.TestCase):
    def test_compacted_tail_matches_cold_and_changed_prefix_is_invalidated(self):
        self.assertIn("Basalt CUDA in-process", str(api("/health")))
        filler = "The shipment register records the date and the weight of each pallet.\n" * 400
        reference = [user("The pallet code is 5172.\n" + filler),
                     {"role": "assistant", "content": "Archive received."}]
        question = user("Return only the four digits of the pallet code.")
        old = reference + [user("Review the shipment archive."),
                           {"role": "assistant", "content": "Review pending."}, question]
        compacted = reference + [user("Summary: shipment review pending."), question]
        ask(old)
        warm, reused = ask(compacted)
        self.assertGreaterEqual(reused, 4096)
        changed = [user(reference[0]["content"].replace("5172", "9246", 1)), *compacted[1:]]
        other, reused = ask(changed)
        self.assertEqual(reused, 0)
        self.assertIn("9246", other)
        cold, reused = ask(compacted)
        self.assertEqual(reused, 0)
        self.assertIn("5172", cold)
        self.assertEqual(warm, cold)


if __name__ == "__main__":
    unittest.main()
