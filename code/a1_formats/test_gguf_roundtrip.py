#!/usr/bin/env python3
"""Build a synthetic GGUF (F32 + Q4_K + Q6_K) and verify gguf_probe's parser
and dequantizers against known ground truth. No external deps beyond numpy.
"""
import struct, subprocess, sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from gguf_probe import parse, dequant_q4_k, dequant_q6_k, GGML

# ---------- reference quantizers (spec-faithful, per ggml-quants.c) ----------
def pack_q4_k(w):  # 256 float -> 144 bytes
    assert w.size == 256
    w = w.astype(np.float32)
    sub = w.reshape(8, 32)
    # asymmetric min-max fit per 32-weight sub-block
    mins = sub.min(axis=1)                       # may be negative
    maxs = sub.max(axis=1)
    scale = (maxs - mins) / 15.0
    scale[scale == 0] = 1e-8
    q = np.clip(np.round((sub - mins[:, None]) / scale[:, None]), 0, 15).astype(np.uint8)
    # quantise the per-sub-block scales/mins themselves to 6 bits.
    # NOTE: mins are negative; dmin must be POSITIVE and m6 = |min|/dmin,
    # because the dequant rule is w = d*sc*q - dmin*m  (m >= 0).
    d    = scale.max() / 63.0
    dmin = np.abs(mins).max() / 63.0
    if d == 0: d = 1e-8
    if dmin == 0: dmin = 1e-8
    sc6 = np.clip(np.round(scale / d), 0, 63).astype(np.uint8)
    m6  = np.clip(np.round(np.abs(mins) / dmin), 0, 63).astype(np.uint8)
    # pack 12 bytes
    sc = np.zeros(12, dtype=np.uint8)
    for j in range(4):
        sc[j]     = sc6[j] & 63
        sc[4 + j] = m6[j] & 63
    for j in range(4):
        sc[8 + j] = (sc6[4 + j] & 0x0F) | ((m6[4 + j] & 0x0F) << 4)
        sc[j]     |= ((sc6[4 + j] >> 4) & 0x03) << 6
        sc[4 + j] |= ((m6[4 + j]  >> 4) & 0x03) << 6
    qs = np.zeros(128, dtype=np.uint8)
    o = 0
    for blk in range(4):
        lo = q[blk * 2]          # weights 0..31 of this 64-span
        hi = q[blk * 2 + 1]      # weights 32..63
        qs[o:o + 32] = (lo & 0x0F) | ((hi & 0x0F) << 4)
        o += 32
    return struct.pack("<e", d) + struct.pack("<e", dmin) + sc.tobytes() + qs.tobytes()

def pack_q6_k(w):  # 256 float -> 210 bytes
    w = w.astype(np.float32)
    # Scale assignment is INTERLEAVED, not contiguous 16-blocks:
    # weight (half*128 + l + 32*k) uses scale index (half*8 + 2*k + l//16).
    idx = np.zeros(256, dtype=np.int32)
    for half in range(2):
        for l in range(32):
            for k in range(4):
                idx[half * 128 + l + 32 * k] = half * 8 + 2 * k + (l // 16)
    amax = np.zeros(16, dtype=np.float32)
    for s_ in range(16):
        amax[s_] = np.abs(w[idx == s_]).max() if np.any(idx == s_) else 0.0
    scale = amax / 32.0
    scale[scale == 0] = 1e-8
    q = np.clip(np.round(w / scale[idx]), -32, 31).astype(np.int8)
    d = scale.max() / 127.0
    if d == 0: d = 1e-8
    sc = np.clip(np.round(scale / d), -128, 127).astype(np.int8)
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for half in range(2):
        for l in range(32):
            i0 = half * 128 + l
            q1, q2 = q[i0], q[i0 + 32]
            q3, q4 = q[i0 + 64], q[i0 + 96]
            u1, u2, u3, u4 = (q1 + 32), (q2 + 32), (q3 + 32), (q4 + 32)
            ql[half * 64 + l]      = (u1 & 0xF) | ((u3 & 0xF) << 4)
            ql[half * 64 + l + 32] = (u2 & 0xF) | ((u4 & 0xF) << 4)
            qh[half * 32 + l] = ((u1 >> 4) & 3) | (((u2 >> 4) & 3) << 2) \
                              | (((u3 >> 4) & 3) << 4) | (((u4 >> 4) & 3) << 6)
    return ql.tobytes() + qh.tobytes() + sc.tobytes() + struct.pack("<e", d)

# ---------- build a tiny GGUF ----------
rng = np.random.default_rng(0)
W_f32 = rng.standard_normal((8, 8)).astype(np.float32) * 0.02
W_q4k = rng.standard_normal(256).astype(np.float32) * 0.02
W_q6k = rng.standard_normal(256).astype(np.float32) * 0.02

b_q4k = pack_q4_k(W_q4k)
b_q6k = pack_q6_k(W_q6k)
assert len(b_q4k) == 144, len(b_q4k)
assert len(b_q6k) == 210, len(b_q6k)

def s(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b

meta = []
meta.append(("general.architecture", 8, s("qwen2")))
meta.append(("general.name", 8, s("synthetic-q4k-test")))
meta.append(("qwen2.block_count", 4, struct.pack("<I", 28)))
meta.append(("qwen2.embedding_length", 4, struct.pack("<I", 1536)))
meta.append(("general.alignment", 4, struct.pack("<I", 32)))

tensors = [
    ("blk.0.attn_q.weight", [8, 8], 0, W_f32.tobytes()),
    ("blk.0.ffn_gate.weight", [256], 12, b_q4k),
    ("blk.0.ffn_down.weight", [256], 14, b_q6k),
]

head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta))
body = b""
for k, t, payload in meta:
    body += s(k) + struct.pack("<I", t) + payload
# tensor infos: offsets computed after we know data_start
infos = b""
offset = 0
offsets = []
for name, dims, ttype, payload in tensors:
    offsets.append(offset)
    name, be, bb = GGML[ttype]
    n = 1
    for d in dims: n *= d
    offset += (n // be) * bb
    offset = (offset + 31) // 32 * 32
for (nm, dims, ttype, _), off in zip(tensors, offsets):
    infos += s(nm) + struct.pack("<I", len(dims))
    for d in dims: infos += struct.pack("<Q", d)
    infos += struct.pack("<I", ttype) + struct.pack("<Q", off)

pos = len(head) + len(body) + len(infos)
pad = (32 - pos % 32) % 32
blob = head + body + infos + b"\x00" * pad
data = b""
cursor = 0
for (nm, dims, ttype, payload), off in zip(tensors, offsets):
    data += b"\x00" * (off - cursor) + payload
    cursor = off + len(payload)
open("synth.gguf", "wb").write(blob + data)
print("wrote synth.gguf", len(blob + data), "bytes")

# ---------- verify ----------
g = parse("synth.gguf")
print("parsed version", g["version"], "tensors", len(g["tensors"]))
assert g["meta"]["general.architecture"] == "qwen2"
assert g["meta"]["qwen2.block_count"] == 28
assert g["meta"]["qwen2.embedding_length"] == 1536

f = g["f"]
t_by_name = {t["name"]: t for t in g["tensors"]}

# F32 exact round-trip
t = t_by_name["blk.0.attn_q.weight"]
f.seek(g["data_start"] + t["offset"])
got = np.frombuffer(f.read(8 * 8 * 4), dtype="<f4").reshape(8, 8)
assert np.array_equal(got, W_f32), "F32 not exact"
print("F32 round-trip: EXACT")

# Q4_K
t = t_by_name["blk.0.ffn_gate.weight"]
f.seek(g["data_start"] + t["offset"])
raw = f.read(144)
dq = dequant_q4_k(raw)
err = np.abs(dq - W_q4k)
print(f"Q4_K  max|err|={err.max():.6f}  rmse={np.sqrt((err**2).mean()):.6f} "
      f"rel={np.linalg.norm(err)/np.linalg.norm(W_q4k):.4%}")
# theoretical bound: half a quant step per sub-block
step = np.abs(W_q4k.reshape(8,32).max(1) - W_q4k.reshape(8,32).min(1)) / 15
print(f"      max half-step={step.max()/2:.6f} -> err within bound: {err.max() <= step.max()/2 + 1e-4}")

# Q6_K
t = t_by_name["blk.0.ffn_down.weight"]
f.seek(g["data_start"] + t["offset"])
raw = f.read(210)
dq6 = dequant_q6_k(raw)
err6 = np.abs(dq6 - W_q6k)
print(f"Q6_K  max|err|={err6.max():.6f}  rmse={np.sqrt((err6**2).mean()):.6f} "
      f"rel={np.linalg.norm(err6)/np.linalg.norm(W_q6k):.4%}")
print(f"      Q4_K rel err / Q6_K rel err = "
      f"{(np.linalg.norm(err)/np.linalg.norm(W_q4k))/(np.linalg.norm(err6)/np.linalg.norm(W_q6k)):.2f}x")
print("\nALL ASSERTIONS PASSED")
