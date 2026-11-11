#!/usr/bin/env python3
"""Measure ACTUAL information loss in Q4_K_M vs the f16 original of the same
model (llama2.c stories15M). Real numbers, not theory.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "ggufpkg"))
import numpy as np, gguf
from gguf_probe import parse, dequant_q4_k, dequant_q6_k, GGML

Q, F = "stories15M.Q4_K_M.gguf", "stories15M.f16.gguf"

def load_all(path):
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        qt = gguf.GGMLQuantizationType(int(t.tensor_type))
        out[t.name] = gguf.quants.dequantize(t.data, qt).astype(np.float32).reshape(-1)
    return out

q, f = load_all(Q), load_all(F)
print(f"tensors: q4km={len(q)}  f16={len(f)}")
print(f"weights: {sum(v.size for v in f.values()):,}\n")

hdr = f"{'tensor':28s} {'type':6s} {'n':>8s} {'relFro':>8s} {'maxAbs':>9s} {'cos':>8s} {'SNR dB':>7s}"
print(hdr); print("-" * len(hdr))
rows = []
for name in sorted(q):
    a, b = q[name], f.get(name)
    if b is None or a.size != b.size:
        print(f"{name:28s} MISSING/MISMATCH"); continue
    err = a - b
    rel = np.linalg.norm(err) / max(np.linalg.norm(b), 1e-12)
    maxa = np.abs(err).max()
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    snr = 10 * np.log10(max(np.sum(b**2), 1e-30) / max(np.sum(err**2), 1e-30))
    tname = [t for t in parse(Q)["tensors"] if t["name"] == name][0]["type"]
    tn = GGML[tname][0]
    rows.append((name, tn, a.size, rel, maxa, cos, snr))
    if not name.endswith("norm.weight"):
        print(f"{name:28s} {tn:6s} {a.size:8d} {rel:8.3%} {maxa:9.5f} {cos:8.5f} {snr:7.2f}")

print("\n=== aggregate (weight matrices only, excluding norms) ===")
mats = [r for r in rows if not r[0].endswith("norm.weight")]
for tn in sorted(set(r[1] for r in mats)):
    sub = [r for r in mats if r[1] == tn]
    print(f"  {tn:6s} {len(sub):2d} tensors  mean relFro={np.mean([r[3] for r in sub]):7.3%}  "
          f"mean SNR={np.mean([r[6] for r in sub]):6.2f} dB  min cos={min(r[5] for r in sub):.6f}")

allerr = np.concatenate([q[n] - f[n] for n in q if n in f])
allref = np.concatenate([f[n] for n in q if n in f])
print(f"\nWHOLE MODEL: relFro={np.linalg.norm(allerr)/np.linalg.norm(allref):.3%}  "
      f"SNR={10*np.log10(np.sum(allref**2)/np.sum(allerr**2)):.2f} dB")

# what fraction of weights are bit-identical to the f16 original?
exact = np.mean(np.abs(allerr) < 1e-7)
print(f"weights reproduced exactly (|err|<1e-7): {exact:.2%}")
print(f"distinct values in f16 original: {np.unique(allref).size:,} of {allref.size:,}")

# ---- the key structural question: does the SIGN structure survive? ----
sign_flip = np.mean(np.sign(q[next(iter(q))]) != np.sign(f[next(iter(q))]))
print(f"\nsign flips on first tensor: {sign_flip:.3%}  "
      f"(a sign flip in a gate/up projection changes the SwiGLU branch)")
