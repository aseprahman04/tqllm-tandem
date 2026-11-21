#!/usr/bin/env python3
"""Parity test: Inkling-2-Mini tokenizer (tqllm GGUF) vs llama-tokenize reference.

Compares token IDs from tqllm's tokenizer.cpp (via tqtokenize --text-file) against
llama-tokenize.exe (via -f) reading the same GGUF. Both must produce identical IDs.

Text is passed via UTF-8 files to avoid Windows command-line encoding issues.

Usage:
    python inkling_tokenizer_parity.py \
        --gguf F:/models/inkling-2-mini/Inkling-2-Mini-UD-Q8_K_XL.gguf \
        --llama-tokenize F:/tools/llama-unsloth-b12588/bin/llama-tokenize.exe \
        --tqtokenize <path-to-tqtokenize>
"""
import argparse
import subprocess
import sys
import tempfile
import os

TEXTS = [
    # English
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm testing contractions like don't, isn't, they're, we've.",
    "It's a test of 's, 't, 're, 've, 'm, 'll, 'd suffixes.",
    "Multiple   spaces and\ttabs here.",
    "Line one\nLine two\n\nLine four after blank.",
    "Trailing spaces   ",
    "  Leading spaces.",
    # Numbers
    "1234567 and 2026-10-24, pi 3.14159, 0007",
    "Numbers: 1, 23, 456, 7890, 12345",
    "Mixed: a1 b22 c333 d4444 e55555",
    # Code (TypeScript)
    "function foo(x: number): number {\n    return x * 2;\n}\n",
    "const items = [1, 2, 3].map((x) => x * 2);\nconsole.log(items);\n",
    "class Foo {\n  constructor(private bar: string) {}\n  getBar(): string { return this.bar; }\n}\n",
    # Code (Python)
    "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n\nif __name__ == '__main__':\n    print(is_palindrome('racecar'))\n",
    "import os\nfrom pathlib import Path\n\ndef main() -> None:\n    for p in Path('.').rglob('*.py'):\n        print(p)\n",
    # Indonesian
    "Halo dunia! Ini adalah tes tokenizer untuk model Inkling-2-Mini.",
    "Saya sedang menulis kode dalam Bahasa Indonesia dan English secara bersamaan.",
    "Kata-kata dengan tanda baca: koma, titik, tanya?, seru!, petik 'single', \"double\".",
    # Unicode / emoji
    "Café — 東京 😊",
    "Emoji test: 🚀🎉💡✅❌",
    "Mixed: café naïve résumé fjord",
    "Devanagari: नमस्ते दुनिया",
    "Arabic: مَرْحَبًا بِالْعَالَمِ",
    "Vietnamese: Tiếng Việt có dấu",
    "Cyrillic: Привет мир",
    # Special tokens (should be matched atomically)
    "<|begin_of_text|>",
    "<|message_user|>hello<|end_message|>",
    "text with <|content_text|> inside",
    "EOS test: <|content_model_end_sampling|>",
    # Edge cases
    "a",
    " ",
    "\n",
    "\r\n",
    "\t",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "0123456789",
    "!@#$%^&*()_+-=[]{}|;:',.<>?/",
    "x\u0301y",  # letter + combining mark
    "\u00f1",  # ñ precomposed
    "n\u0303",  # n + combining tilde (decomposed)
]


def write_utf8(text: str) -> str:
    """Write text to a temp UTF-8 file, return path."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="toktest_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def llama_tokenize_file(gguf: str, llama_bin: str, text_file: str) -> list[int]:
    """Get token IDs from llama-tokenize.exe using -f (file input)."""
    result = subprocess.run(
        [llama_bin, "-m", gguf, "-f", text_file, "--ids"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {result.stderr}")
    line = result.stdout.strip()
    start = line.find("[")
    end = line.rfind("]") + 1
    if start < 0 or end <= start:
        raise RuntimeError(f"Cannot parse llama-tokenize output: {line!r}")
    ids_str = line[start:end]
    return [int(x) for x in ids_str.strip("[]").split(",") if x.strip()]


def tqtokenize_file(tqtokenize_bin: str, gguf: str, text_file: str) -> list[int]:
    """Get token IDs from tqtokenize using --text-file."""
    result = subprocess.run(
        [tqtokenize_bin, "--tokenizer", gguf, "--text-file", text_file],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tqtokenize failed: {result.stderr}")
    return [int(x) for x in result.stdout.split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--llama-tokenize", required=True)
    parser.add_argument("--tqtokenize", required=True)
    args = parser.parse_args()

    passed = 0
    failed = 0
    for text in TEXTS:
        tf = write_utf8(text)
        try:
            try:
                ref = llama_tokenize_file(args.gguf, args.llama_tokenize, tf)
            except Exception as e:
                print(f"SKIP (llama-tokenize error) {text!r}: {e}")
                continue
            try:
                actual = tqtokenize_file(args.tqtokenize, args.gguf, tf)
            except Exception as e:
                print(f"FAIL (tqtokenize error) {text!r}: {e}")
                failed += 1
                continue
            if ref == actual:
                passed += 1
                print(f"PASS ({len(ref):3d} ids) {text!r}")
            else:
                failed += 1
                print(f"FAIL {text!r}")
                print(f"  ref    ({len(ref)}): {ref[:20]}{'...' if len(ref)>20 else ''}")
                print(f"  actual ({len(actual)}): {actual[:20]}{'...' if len(actual)>20 else ''}")
        finally:
            os.unlink(tf)

    total = passed + failed
    print(f"\nParity: {passed}/{total} texts identical ({failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
