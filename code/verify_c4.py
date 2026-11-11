#!/usr/bin/env python3
"""
verify_c4.py — re-run C4 with the CORRECTED steering estimator.

The first C4 result was measured with a last-token vector that was numerically
an embedding difference (cos 1.00000). This script uses
`ModelAdapter.build_steer_vectors()`: position-averaged, with contrast pairs that
share an identical suffix. Same test, same controls, fixed instrument.

Test: does the real vector beat 20 random vectors of matched norm?
  T1 selectivity     real target effect  > p95 of random
  T2 collateral      real control effect <= median of random
  T3 behaviour       generation changes more than p95 of random
  T4 held-out sign   direction consistent on prompts never used to build it

Usage: PYTHONPATH=code .venv/bin/python code/verify_c4.py [--model PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from mw.adapter import ModelAdapter

HELDOUT = ["def merge(a, b):\n", "class Cache:\n", "async def fetch(url):\n",
           "def parse(x):\n", "with open(path) as fh:\n", "while queue:\n",
           "def to_json(obj):\n", "for k, v in d.items():\n", "def main():\n", "try:\n"]
COMMENT_TOKS = ["#", "# ", " #", "##", "# this", "# return"]
CODE_TOKS = ["return", "    return", "def", "for", "if", "class", "x", "self"]
YES_TOKS = [" yes", "yes", " Yes"]
NO_TOKS = [" no", "no", " No"]


def ids(tok, ss):
    out = []
    for s in ss:
        i = tok(s, add_special_tokens=False).input_ids
        if i:
            out.append(i[0])
    return sorted(set(out))


def pct(xs, p):
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".hf/bf16-hf")
    ap.add_argument("--randoms", type=int, default=20)
    args = ap.parse_args()

    a = ModelAdapter(args.model).load(dtype=torch.float32)
    ct, cd = ids(a.tok, COMMENT_TOKS), ids(a.tok, CODE_TOKS)
    yt, nt = ids(a.tok, YES_TOKS), ids(a.tok, NO_TOKS)

    t0 = time.time()
    vectors = a.build_steer_vectors()

    # CALIBRATION (S1 finding, 2026-11-12): the raw vectors were ~335 units while
    # the residual norm at that layer is ~5, so coef=2 was a ~130x perturbation
    # and both real and random vectors saturated. Rescale every layer to its mean
    # residual norm so coef=1 means "one residual norm".
    a._capturing = True
    a._capture_full = False
    scale_acc: dict[int, list] = {}
    for p in HELDOUT:
        i = a.tok(p, return_tensors="pt").input_ids
        a._pre.clear()
        with torch.no_grad():
            a.model(input_ids=i)
        for l, h in a._pre.items():
            scale_acc.setdefault(l, []).append(float(h.norm()))
    a._capturing = False
    scale = {l: statistics.mean(v) for l, v in scale_acc.items() if v}
    vectors = {l: v / max(float(v.norm()), 1e-8) * scale.get(l, 1.0)
               for l, v in vectors.items()}
    print(f"vectors: {len(vectors)} layers, calibrated to residual norm, "
          f"{time.time()-t0:.1f}s", flush=True)
    print(f"  scales: L8={scale.get(8,0):.2f} L16={scale.get(16,0):.2f}", flush=True)

    # sanity: is the vector distinguishable from a token embedding?
    emb = a.model.get_input_embeddings().weight
    for l in (0, 8, 16):
        v = vectors[l]
        sims = torch.nn.functional.cosine_similarity(v.unsqueeze(0), emb[:2000], dim=1)
        print(f"  L{l}: norm={float(v.norm()):.1f} max cos to token emb={float(sims.abs().max()):.4f}")

    @torch.no_grad()
    def fwd(prompt):
        i = a.tok(prompt, return_tensors="pt").input_ids
        return a.model(input_ids=i).logits[0, -1, :].float()

    def effect(lg, idset, other):
        return float(torch.logsumexp(lg[idset], 0) - torch.logsumexp(lg[other], 0))

    # Precompute baselines ONCE. The old loop recomputed them for every vector,
    # tripling the forward count and blowing the time budget.
    base = {}
    for p in HELDOUT:
        lg = fwd(p)
        base[p] = (effect(lg, ct, cd), abs(effect(lg, yt, nt)))

    def measure(vec, layer, coef):
        a.set_steer(layer, vec, coef)
        te, ce, sign = [], [], []
        for p in HELDOUT:
            lg = fwd(p)
            t = effect(lg, ct, cd) - base[p][0]
            te.append(t)
            ce.append(abs(effect(lg, yt, nt)))
            sign.append(1 if t > 0 else 0)
        a.set_steer(None, None, 0.0)
        return statistics.mean(te), statistics.mean(ce), sum(sign) / len(sign)

    results = {}
    for layer in (8, 16):
        te, ce, sg = measure(vectors[layer], layer, 1.0)
        results[layer] = {"te": te, "ce": ce, "sign": sg}
        print(f"  L{layer} coef+1: target={te:+.3f} control={ce:.3f} sign={sg:.2f}", flush=True)

    layer = max(results, key=lambda l: results[l]["te"])
    print(f"best layer: L{layer}")

    rng = torch.Generator().manual_seed(7)
    rnd = []
    for k in range(args.randoms):
        rv = torch.randn(vectors[layer].shape, generator=rng)
        rv = rv / rv.norm() * vectors[layer].norm()
        te, ce, sg = measure(rv, layer, 1.0)
        rnd.append({"te": te, "ce": ce, "sign": sg})
        print(f"  random {k+1}/{args.randoms}: target={te:+.2f} control={ce:.2f} sign={sg:.2f}",
              flush=True)

    real = results[layer]
    t1 = real["te"] > pct([r["te"] for r in rnd], 95)
    t2 = real["ce"] <= pct([r["ce"] for r in rnd], 50)
    t4 = real["sign"] >= 0.8 and real["sign"] > pct([r["sign"] for r in rnd], 95)

    out = {"layer": layer, "real": real,
           "random_p95_target": pct([r["te"] for r in rnd], 95),
           "random_median_control": pct([r["ce"] for r in rnd], 50),
           "random_p95_sign": pct([r["sign"] for r in rnd], 95),
           "T1_selectivity": t1, "T2_collateral": t2, "T4_heldout_sign": t4,
           "T3_behaviour": "not run (needs generation; run steering_test.py)"}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
