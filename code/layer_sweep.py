#!/usr/bin/env python3
"""
layer_sweep.py — find out whether ANY layer gives a specific, monotone steering
effect, or whether diff-in-means is just producing a generic perturbation.

Why this exists: C4 at layer 14 produced a V-shaped curve and random vectors
with LARGER spans than the real vector. That is the signature of a non-specific
direction. Before concluding "steering doesn't work", sweep layers and use
enough random controls to make the comparison meaningful.

Usage: python code/layer_sweep.py [--model PATH] [--pairs 12] [--randoms 10]
"""
from pathlib import Path
from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(4)

PAIRS = [
    ("# add two numbers and return the result\n", "def add(a, b):\n"),
    ("# read the file and parse each line as json\n", "for line in f:\n"),
    ("# raise an error if the input is empty\n", "if not data:\n"),
    ("# loop over every item in the list\n", "for x in items:\n"),
    ("# this class stores user credentials\n", "class User:\n"),
    ("# return the cached value if present\n", "return cache.get(k)\n"),
    ("# validate the request before processing\n", "def handle(req):\n"),
    ("# write the output to disk\n", "with open(p, 'w') as f:\n"),
    ("# handle the case where the key is missing\n", "try:\n"),
    ("# compute the mean of the array\n", "total = sum(xs)\n"),
    ("# connect to the database\n", "conn = sqlite3.connect(p)\n"),
    ("# send the payload to the server\n", "requests.post(url, json=body)\n"),
]

COMMENT_TOKENS = ["#", "# ", " #", "##", "# Compute", "# this", "# return"]
CODE_TOKENS = ["return", "    return", "def", "for", "if", "class", "x", "self"]
PROBE = "def add(a, b):\n"


def first_ids(tok, strings):
    out = []
    for s in strings:
        ids = tok(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/.hf/q4km-hf")
    ap.add_argument("--randoms", type=int, default=10)
    ap.add_argument("--layers", default="4,8,12,16,20,22,24,26")
    ap.add_argument("--grid", default="-3,-2,-1,0,1,2,3")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    grid = [float(x) for x in args.grid.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    cap: dict[int, torch.Tensor] = {}
    steer = {"layer": None, "vec": None, "coef": 0.0}

    def mk_cap(i):
        def hook(mod, a, o):
            t = o[0] if isinstance(o, tuple) else o
            cap[i] = t[:, -1, :].detach()
        return hook

    def mk_steer(i):
        def hook(mod, a):
            if steer["layer"] != i or steer["coef"] == 0.0:
                return None
            h = a[0].clone()
            h[:, -1, :] = h[:, -1, :] + steer["coef"] * steer["vec"]
            return (h,) + tuple(a[1:])
        return hook

    for i, L in enumerate(model.model.layers):
        L.register_forward_hook(mk_cap(i))
        L.register_forward_pre_hook(mk_steer(i))

    comment_ids = first_ids(tok, COMMENT_TOKENS)
    code_ids = first_ids(tok, CODE_TOKENS)
    probe_ids = tok(PROBE, return_tensors="pt").input_ids

    @torch.no_grad()
    def logits_for(prompt_ids):
        return model(input_ids=prompt_ids).logits[0, -1, :].float()

    def effect(lg):
        return float(torch.logsumexp(lg[comment_ids], 0)
                     - torch.logsumexp(lg[code_ids], 0))

    @torch.no_grad()
    def resid_at(text, layer):
        ids = tok(text, return_tensors="pt").input_ids
        model(input_ids=ids)
        return cap[layer][0]

    base_lg = logits_for(probe_ids)
    base_eff = effect(base_lg)

    # random controls, matched norm per layer
    rng = torch.Generator().manual_seed(1234)
    rand_vecs = {L: [] for L in layers}
    for L in layers:
        h = resid_at(PAIRS[0][0], L)
        for _ in range(args.randoms):
            r = torch.randn(h.shape, generator=rng)
            rand_vecs[L].append(r / r.norm() * h.norm())

    results = []
    for L in layers:
        pos = torch.stack([resid_at(a, L) for a, _ in PAIRS])
        neg = torch.stack([resid_at(b, L) for _, b in PAIRS])
        v = pos.mean(0) - neg.mean(0)
        resid_norm = float(pos.norm(dim=-1).mean())
        v = v / max(float(v.norm()), 1e-8) * resid_norm

        def curve(vec):
            xs, ys = [], []
            for c in grid:
                steer.update(layer=L, vec=vec, coef=c)
                lg = logits_for(probe_ids)
                xs.append(c)
                ys.append(effect(lg) - base_eff)
            steer.update(layer=None, vec=None, coef=0.0)
            return xs, ys

        xs, real = curve(v)
        rand_spans, rand_curves = [], []
        for rv in rand_vecs[L]:
            _, ry = curve(rv)
            rand_curves.append(ry)
            rand_spans.append(max(ry) - min(ry))

        span = max(real) - min(real)
        # monotone fraction: how often effect increases with coef
        inc = sum(1 for i in range(len(real) - 1) if real[i + 1] > real[i])
        mono = inc / (len(real) - 1)
        better_than = sum(1 for s in rand_spans if span > s) / len(rand_spans)
        results.append({
            "layer": L, "curve": [round(x, 3) for x in real],
            "span": round(span, 3), "monotone_frac": round(mono, 2),
            "beats_random_frac": round(better_than, 2),
            "rand_span_max": round(max(rand_spans), 3),
            "rand_span_median": round(sorted(rand_spans)[len(rand_spans) // 2], 3),
        })
        print(f"L{L:>2} span={span:7.3f} mono={mono:.2f} "
              f"beats_rand={better_than:.2f} rand_med={sorted(rand_spans)[len(rand_spans)//2]:7.3f} "
              f"curve={[round(x,2) for x in real]}", flush=True)

    best = max(results, key=lambda r: (r["beats_random_frac"], r["monotone_frac"]))
    print("\nBEST:", json.dumps(best))
    json.dump(results, open(f"{ROOT}/proof/layer_sweep.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
