#!/usr/bin/env python3
"""Minimal dependency-light GGUF reader + Q4_K/Q6_K dequantizer.
Verifies: header, metadata, tensor table, ggml type histogram.
Usage: python3 gguf_probe.py <file.gguf>
"""
import sys, struct, collections
import numpy as np

# ggml type enum -> (name, block_size_elems, block_bytes)
GGML = {
    0:("F32",1,4), 1:("F16",1,2), 2:("Q4_0",32,18), 3:("Q4_1",32,20),
    6:("Q5_0",32,22), 7:("Q5_1",32,24), 8:("Q8_0",32,34), 9:("Q8_1",32,36),
    10:("Q2_K",256,84), 11:("Q3_K",256,110), 12:("Q4_K",256,144),
    13:("Q5_K",256,176), 14:("Q6_K",256,210), 15:("Q8_K",256,292),
    16:("IQ2_XXS",256,66), 17:("IQ2_XS",256,74), 18:("IQ3_XXS",256,98),
    19:("IQ1_S",256,50), 20:("IQ4_NL",32,18), 21:("IQ3_S",256,110),
    22:("IQ2_S",256,82), 23:("IQ4_XS",256,136), 24:("I8",1,1), 25:("I16",1,2),
    26:("I32",1,4), 27:("I64",1,8), 28:("F64",1,8), 29:("IQ1_M",256,56),
    30:("BF16",1,2), 34:("TQ1_0",256,54), 35:("TQ2_0",256,66),
}

# GGUF metadata value types
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
SCALAR = {T_U8:("<B",1),T_I8:("<b",1),T_U16:("<H",2),T_I16:("<h",2),
          T_U32:("<I",4),T_I32:("<i",4),T_F32:("<f",4),T_BOOL:("<?",1),
          T_U64:("<Q",8),T_I64:("<q",8),T_F64:("<d",8)}

class Reader:
    def __init__(self, f): self.f = f
    def read(self, n):
        b = self.f.read(n)
        if len(b) != n: raise EOFError(f"want {n} got {len(b)}")
        return b
    def unpack(self, fmt, size): return struct.unpack(fmt, self.read(size))[0]
    def string(self):
        n = self.unpack("<Q", 8)
        return self.read(n).decode("utf-8", "replace")

def read_meta_value(r, vtype):
    if vtype == T_STR: return r.string()
    if vtype == T_ARR:
        etype = r.unpack("<I", 4)
        n = r.unpack("<Q", 8)
        # don't materialise huge tokenizer arrays; sample small ones
        if n > 64:
            for _ in range(n): read_meta_value(r, etype)
            return f"<array of {n} type={etype}>"
        return [read_meta_value(r, etype) for _ in range(n)]
    fmt, size = SCALAR[vtype]
    return r.unpack(fmt, size)

def parse(path):
    f = open(path, "rb")
    r = Reader(f)
    magic = r.read(4)
    assert magic == b"GGUF", magic
    version = r.unpack("<I", 4)
    n_tensors = r.unpack("<Q", 8)
    n_kv = r.unpack("<Q", 8)
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.unpack("<I", 4)
        meta[key] = read_meta_value(r, vtype)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.unpack("<I", 4)
        dims = [r.unpack("<Q", 8) for _ in range(nd)]
        ttype = r.unpack("<I", 4)
        off = r.unpack("<Q", 8)
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = meta.get("general.alignment", 32)
    pos = f.tell()
    data_start = (pos + align - 1) // align * align
    return dict(f=f, version=version, meta=meta, tensors=tensors,
                data_start=data_start, align=align, size=None)

def f16(b):  # 2 bytes -> float
    return np.frombuffer(b, dtype="<f2").astype(np.float32)

def dequant_q4_k(block):  # 144 bytes -> 256 float32
    assert len(block) == 144
    d    = float(f16(block[0:2])[0])
    dmin = float(f16(block[2:4])[0])
    sc = block[4:16]; qs = block[16:144]
    # unpack 8 x 6-bit scales and 8 x 6-bit mins  (get_scale_min_k4)
    scales = [0]*8; mins = [0]*8
    for j in range(4):
        scales[j] = sc[j] & 63
        mins[j]   = sc[4+j] & 63
    for j in range(4):
        scales[4+j] = (sc[8+j] & 0x0F) | ((sc[j]   >> 6) << 4)
        mins[4+j]   = (sc[8+j] >> 4)   | ((sc[4+j] >> 6) << 4)
    out = np.empty(256, dtype=np.float32)
    is_ = 0; qi = 0; o = 0
    for _ in range(4):  # four 64-weight spans
        d1, m1 = d*scales[is_], dmin*mins[is_]
        d2, m2 = d*scales[is_+1], dmin*mins[is_+1]
        chunk = np.frombuffer(qs[qi:qi+32], dtype=np.uint8)
        out[o:o+32]      = d1*(chunk & 0x0F) - m1
        out[o+32:o+64]   = d2*(chunk >> 4)   - m2
        o += 64; qi += 32; is_ += 2
    return out

def dequant_q6_k(block):  # 210 bytes -> 256 float32
    assert len(block) == 210
    ql = np.frombuffer(block[0:128], dtype=np.uint8)
    qh = np.frombuffer(block[128:192], dtype=np.uint8)
    sc = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float32)
    d  = float(f16(block[208:210])[0])
    out = np.empty(256, dtype=np.float32)
    ql = ql.astype(np.int32); qh = qh.astype(np.int32)
    for half in range(2):
        qlb = ql[half*64:half*64+64]; qhb = qh[half*32:half*32+32]
        scb = sc[half*8:half*8+8]; base = half*128
        for l in range(32):
            is_ = l // 16
            q1 = ((qlb[l]      & 0xF) | (((qhb[l] >> 0) & 3) << 4)) - 32
            q2 = ((qlb[l+32]   & 0xF) | (((qhb[l] >> 2) & 3) << 4)) - 32
            q3 = ((qlb[l]   >> 4)     | (((qhb[l] >> 4) & 3) << 4)) - 32
            q4 = ((qlb[l+32]>> 4)     | (((qhb[l] >> 6) & 3) << 4)) - 32
            out[base+l]      = d*scb[is_+0]*q1
            out[base+l+32]   = d*scb[is_+2]*q2
            out[base+l+64]   = d*scb[is_+4]*q3
            out[base+l+96]   = d*scb[is_+6]*q4
    return out

def main(path):
    g = parse(path)
    import os
    size = os.path.getsize(path)
    print(f"file        {path}")
    print(f"bytes       {size:,}")
    print(f"version     {g['version']}   alignment {g['align']}")
    print(f"tensors     {len(g['tensors'])}")
    print(f"data_start  {g['data_start']:,}  (header+meta = {g['data_start']/size*100:.2f}% of file)")
    print("\n--- key metadata ---")
    for k in sorted(g["meta"]):
        if any(k.startswith(p) for p in ("general.", "llama.", "qwen2.", "tokenizer.ggml.model",
                                          "tokenizer.ggml.tokens")):
            v = g["meta"][k]
            s = str(v)
            print(f"  {k:42s} {s[:70]}")
    print("\n--- ggml type histogram ---")
    hist = collections.Counter()
    byt = collections.Counter()
    for t in g["tensors"]:
        hist[t["type"]] += 1
        name, be, bb = GGML.get(t["type"], ("?",1,1))
        n = 1
        for dm in t["dims"]: n *= dm
        byt[name] += (n//be)*bb
    for t, c in hist.most_common():
        name, be, bb = GGML.get(t, ("?",1,1))
        print(f"  {name:8s} (id {t:2d}) {c:3d} tensors  {byt[name]/1e6:8.1f} MB")
    print(f"  {'TOTAL':8s}         {sum(hist.values()):3d} tensors  {sum(byt.values())/1e6:8.1f} MB")

    # dequantise one Q4_K and one Q6_K tensor for real
    f = g["f"]
    for want in ("Q4_K", "Q6_K"):
        cand = [t for t in g["tensors"] if GGML.get(t["type"], ("",))[0] == want]
        if not cand: continue
        t = cand[0]
        name, be, bb = GGML[t["type"]]
        n = 1
        for dm in t["dims"]: n *= dm
        nb = n // be
        f.seek(g["data_start"] + t["offset"])
        raw = f.read(nb * bb)
        fn = dequant_q4_k if want == "Q4_K" else dequant_q6_k
        vals = np.concatenate([fn(raw[i*bb:(i+1)*bb]) for i in range(min(nb, 4))])
        print(f"\n--- dequantised {want} tensor: {t['name']} dims={t['dims']} ---")
        print(f"  blocks={nb}  bytes={nb*bb:,}  sampled={vals.size} values")
        print(f"  min={vals.min():+.5f} max={vals.max():+.5f} mean={vals.mean():+.6f} "
              f"std={vals.std():.5f}  zeros={(vals==0).mean()*100:.1f}%")
        u = np.unique(np.round(vals, 7))
        print(f"  distinct values in sample: {u.size} / {vals.size}")

if __name__ == "__main__":
    main(sys.argv[1])
