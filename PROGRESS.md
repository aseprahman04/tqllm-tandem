# Inkling-2-Mini (inkling-entry branch) — Progress

## 2026-11-22 — Tensor-name mismatch fixed (dry-run 0 mismatches, tqserve BUILD-OK)

**Symptom.** After the tokenizer + template + 2-GPU layer split passed, loading the model
`F:\models\inkling-2-mini\Inkling-2-Mini-UD-Q8_K_XL.gguf` (root/Unsloth layout, arch `inkling2`,
734 tensors, 24 physical layers, 48 visits) failed with:

```
inkling2: missing tensor blk.5.attn_norm.weight
```

**Root cause (two bugs, both in `src/models/inkling2/inkling2.cpp` `configure()`):**

1. **`sliding_` never set from the GGUF.** The loader initialised `sliding_.resize(n_layers_, true)`
   (every layer sliding) and never read `inkling2.attention.sliding_window_pattern`. The root layout
   stores a 48-element bool array (one per visit); full layers (visits 5, 11, 17, 23 → physical
   layers 5, 11, 17, 23) carry `attn_{q,k,v}_inp_norm.weight` instead of `attn_norm.weight`. With
   `sliding_[5]` stuck true, the loader looked for `blk.5.attn_norm.weight`, which does not exist.

2. **`head_count` / `head_count_kv` read as scalars but stored as arrays.** The root layout stores
   `inkling2.attention.head_count` and `head_count_kv` as 48-element arrays (one value per visit).
   `GgufFile::num()` returns `.num`, which is 0 for an array, so `n_head_` and `n_kv_` were 0 —
   every attention dimension would have been wrong at runtime (silent, not caught by the name check).

**Fix.** `configure()` now:
- reads `head_count` / `head_count_kv` via a new `arr0()` helper (first element of the array, falls
  back to the scalar);
- reads `inkling2.attention.sliding_window_pattern` and sets `sliding_[p] = false` for any physical
  layer `p` that has a `false` visit, using the existing `phys_` visit→layer map.

Result: 20 sliding layers + 4 full layers (5, 11, 17, 23), `n_head=24`, `n_kv=8`.

**Dry-run tool.** `tools/inkling2_names_check.cpp` (target `inkling2-names-check`, links `tqllm`,
no GPU) parses the GGUF header, dumps all 734 tensor names + type + shape to
`F:\cache\tmp\inkling-research\gguf-tensors.txt`, enumerates the exact tensor set the loader expects
(24 physical layers + head + embedding + loop queries + expert stacks), and reports every mismatch
(missing / unused / shape / type) at once. After the fix it reports **0 mismatches**:

```
config: H=2304 n_visits=48 n_pass=3 core=[6,18) n_layers=24 n_head=24 n_kv=8 hd=128 ...
sliding: 20 sliding, 4 full (full layers: 5 11 17 23)
missing(required): 0   shape mismatch: 0   type mismatch: 0   unused: 0
TOTAL mismatches: 0
```

**Build.** `cmd /c F:\cache\tmp\build-inkling-tqserve.bat` → `BUILD-OK` (tqserve.exe links).

**Not done (out of scope, no GPU run):** the forward pass has not been executed on the GPUs; the
loader now finds every tensor and the shapes/types match, but numerical parity against the float64
reference (`ref/inkling2.py`) is still pending. The MoE host round-trip and the 2-GPU split are
unchanged.
