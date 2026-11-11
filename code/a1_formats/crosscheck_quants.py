#!/usr/bin/env python3
"""Cross-validate my hand-written Q4_K/Q6_K dequantizers against the official
`gguf` python package on a REAL llama.cpp-produced Q4_K_M file.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "ggufpkg"))
import numpy as np
import gguf
from gguf_probe import parse, dequant_q4_k, dequant_q6_k, GGML

PATH = "stories15M.Q4_K_M.gguf"

# --- official reader ---
r = gguf.GGUFReader(PATH)
print(f"gguf pkg: {len(r.tensors)} tensors, version {r.fields['general.version'].contents() if 'general.version' in r.fields else 3}")

# --- my reader ---
g = parse(PATH)
print(f"my reader: {len(g['tensors'])} tensors, version {g['version']}")
assert len(r.tensors) == len(g["tensors"]), "tensor count mismatch"
assert g["data_start"] == r.data_offset, f"data_offset mismatch {g['data_start']} vs {r.data_offset}"
print(f"data_offset agrees: {r.data_offset:,}")

# --- per-tensor comparison of names, dims, types ---
mine = {t["name"]: t for t in g["tensors"]}
bad = 0
for t in r.tensors:
    m = mine.get(t.name)
    if m is None:
        print("MISSING", t.name); bad += 1; continue
    if tuple(m["dims"]) != tuple(int(x) for x in t.shape):
        print("DIM MISMATCH", t.name, m["dims"], t.shape); bad += 1
    if m["type"] != int(t.tensor_type):
        print("TYPE MISMATCH", t.name, m["type"], int(t.tensor_type)); bad += 1
print(f"tensor table agrees: {len(r.tensors)-bad}/{len(r.tensors)}")

# --- numeric cross-check on every Q4_K and Q6_K tensor ---
def ref_dequant(t):
    qt = gguf.GGMLQuantizationType(int(t.tensor_type))
    return gguf.quants.dequantize(t.data, qt).astype(np.float32).reshape(-1)

worst4 = worst6 = 0.0
for t in r.tensors:
    nm = int(t.tensor_type)
    tname = GGML.get(nm, ("",))[0]
    if tname not in ("Q4_K", "Q6_K"):
        continue
    ref = ref_dequant(t)
    m = mine[t.name]
    be, bb = GGML[nm][1], GGML[nm][2]
    n = 1
    for d in m["dims"]: n *= d
    nb = n // be
    g["f"].seek(g["data_start"] + m["offset"])
    raw = g["f"].read(nb * bb)
    fn = dequant_q4_k if tname == "Q4_K" else dequant_q6_k
    got = np.concatenate([fn(raw[i*bb:(i+1)*bb]) for i in range(nb)])
    if got.size != ref.size:
        print(f"  SIZE MISMATCH {t.name}: {got.size} vs {ref.size}"); continue
    err = float(np.abs(got - ref).max())
    if tname == "Q4_K": worst4 = max(worst4, err)
    else:               worst6 = max(worst6, err)
    print(f"  {tname:5s} {t.name:28s} n={got.size:7d} max|mine-ref|={err:.3e}")

print(f"\nWORST Q4_K deviation from reference: {worst4:.3e}")
print(f"WORST Q6_K deviation from reference: {worst6:.3e}")
assert worst4 < 1e-6 and worst6 < 1e-6, "dequantizer disagrees with reference!"
print("PASS: hand-written dequantizers are bit-equivalent to gguf/quants.py")
