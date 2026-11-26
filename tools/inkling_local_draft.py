"""Copy an Inkling 2 Mini Basalt pack with a 1,024-token window in every DSpark proposal layer."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original .basalt file")
    parser.add_argument("output", type=Path, help="new .basalt file on a volume supporting hard links")
    parser.add_argument("--basalt-source", type=Path, required=True, help="Basalt checkout containing tools/model_file.py")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    sys.path.insert(0, str(args.basalt_source / "tools"))
    from model_file import ModelFile, ModelWriter

    source = ModelFile(str(args.source))
    if not source.verify("meta"):
        raise ValueError("Invalid source metadata checksum")
    meta = json.loads(source.read("meta"))
    pattern = "inkling2.dspark.sliding_window_pattern"
    window = "inkling2.dspark.sliding_window"
    kv = meta["gguf_kv"]
    if meta.get("arch") != "inkling2" or kv.get(pattern) != [True, False, True] or kv.get(window) != 1024:
        raise ValueError("Expected an Inkling 2 Mini pack with two local DSpark layers and one global DSpark layer")
    kv[pattern] = [True, True, True]

    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=args.output.name + ".", delete=False) as file:
        temporary = Path(file.name)
    try:
        writer = ModelWriter(str(temporary))
        try:
            for name, (offset, size, _) in source.sections.items():
                if name == "meta":
                    writer.add_bytes(name, json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode())
                else:
                    writer.add_file(name, str(args.source), offset, size)
            writer.close()
        finally:
            if not writer.f.closed:
                writer.f.close()

        derived = ModelFile(str(temporary))
        if set(derived.sections) != set(source.sections) or not derived.verify("meta"):
            raise ValueError("Invalid output pack metadata or section table")
        for name, section in source.sections.items():
            if name != "meta" and derived.sections[name][1:] != section[1:]:
                raise ValueError(f"Payload section differs from the source: {name}")
        # Publish a complete file. A concurrent writer wins or loses without overwriting the other result.
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
