"""Replace the embedded DSpark drafter in an Inkling pack, preserving its target quantization."""
import argparse
from hashlib import sha256
import json
from multiprocessing.dummy import Pool
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def convert(args):
    sys.path[:0] = [str(args.basalt_source / "tools/convert"), str(args.basalt_source / "tools")]
    from checkpoint_inkling2 import Dflash2
    from model_file import ModelFile, ModelWriter, Stream, TENSOR_ALIGN

    source = ModelFile(str(args.source))
    if not source.verify("meta"):
        raise ValueError("Invalid source metadata digest")
    meta = json.loads(source.read("meta"))
    kv = meta["gguf_kv"]
    if meta.get("arch") != "inkling2" or not kv.get("inkling2.dspark.block_count"):
        raise ValueError("Expected Inkling with an embedded DSpark drafter")
    drafter = Dflash2(args.drafter, SimpleNamespace(text=meta["hf_config"]["text_config"]))
    target = {k: v for k, v in meta["tensors"].items() if not k.startswith("dspark.")}
    order = sorted(target, key=lambda k: target[k]["offset"])
    offset = source.sections["gguf"][0]
    parts = [(str(args.source), offset + target[n]["offset"], target[n]["bytes"]) for n in order]
    meta["gguf_kv"] = {k: v for k, v in kv.items() if not k.startswith("inkling2.dspark.")}
    meta["gguf_kv"].update(drafter.kv())
    meta["dspark"] = None
    files = []
    for path in drafter.files:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        files.append({"path": path.name, "sha256": digest.hexdigest()})
    meta["dflash2"] = {"source": args.drafter.name, "files": files}
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        writer = ModelWriter(str(temporary))
        try:
            with Pool(args.workers) as pool:
                for name, tensor in drafter.tensors.items():
                    size = tensor.expected_bytes()
                    parts.append(Stream(size, drafter.chunks(name, pool, depth=args.workers))
                                 if tensor.type_name == "Q8_0" else drafter.float_bytes(name))
                    target[name] = dict(type=tensor.type_name, type_id=tensor.type_id,
                                        shape=list(tensor.ne), bytes=size)
                    order.append(name)
                offsets = writer.add_parts("gguf", parts, TENSOR_ALIGN)
            for name, position in zip(order, offsets, strict=True):
                target[name]["offset"] = position
            meta["tensors"] = target
            writer.add_bytes("meta", json.dumps(meta, separators=(",", ":")).encode())
            for section, (start, size, _) in source.sections.items():
                if section not in ("meta", "gguf"):
                    writer.add_file(section, str(args.source), start, size)
            writer.close()
        finally:
            if not writer.f.closed:
                writer.f.close()
        output = ModelFile(str(temporary))
        for section, entry in source.sections.items():
            if section not in ("meta", "gguf") and output.sections[section][1:] != entry[1:]:
                raise ValueError(f"Section differs from source: {section}")
        if not output.verify("gguf") or not output.verify("meta"):
            raise ValueError("Invalid output pack digest")
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("drafter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--basalt-source", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1:
        parser.error("Specify a new output path and at least one worker")
    convert(args)
