"""Generate a Ninja manifest that targets SM120a only for Basalt's expert CUDA object."""
import argparse
from pathlib import Path
import re


def specialize(manifest):
    # CMake emits a separate build statement with its own FLAGS for every object.
    statements = re.split(r"(?=^build )", manifest, flags=re.MULTILINE)
    generic = "--generate-code=arch=compute_120,code=[compute_120,sm_120]"
    native = "--generate-code=arch=compute_120a,code=[sm_120a]"
    matches = []
    for index, statement in enumerate(statements):
        header = statement.partition("\n")[0].replace("\\", "/")
        if re.match(r"build [^:]*kernels\.dir/src/kernels/cuda/iq_kernels\.cu\.(?:o|obj):", header):
            matches.append(index)
    if len(matches) != 1 or statements[matches[0]].count(generic) != 1:
        raise ValueError("Expected one SM120 CMake Ninja rule for the Basalt iq_kernels object")
    index = matches[0]
    statements[index] = statements[index].replace(generic, native)
    return "".join(statements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    original = args.build_dir / "build.ninja"
    output = args.build_dir / "inkling-native.ninja"
    output.write_bytes(specialize(original.read_text(encoding="utf-8")).encode("utf-8"))
    print(output)
