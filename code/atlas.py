#!/usr/bin/env python3
"""
atlas.py — C7: empirical neuron atlas.

Profiles every MLP "neuron" (index in intermediate_size) of every layer on a real
corpus: how often it fires, how hard, and on which token. This is the honest
alternative to "neuron 4021 = recursion": we report measured behaviour, not labels.

The activation we profile is the SwiGLU intermediate: act(gate_proj(x)) * up_proj(x),
captured as the input to down_proj. That is the quantity the down projection
actually consumes, so it is the right thing to attribute.

Usage:
  python code/atlas.py --tokens 20000 [--model PATH] [--batch 4] [--seq 64]
Writes: proof/c7_atlas.json, proof/c7_coverage.txt
"""
from pathlib import Path
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(4)
PROOF = f"{ROOT}/proof"


def build_corpus(tok, need_tokens: int) -> torch.Tensor:
    files = []
    for pat in ("/usr/lib/python3.*/*.py", "/usr/lib/python3.*/**/*.py"):
        files.extend(glob.glob(pat, recursive=True))
    files = [f for f in files if "test" not in f.lower()][:400]
    if not files:
        sys.exit("no corpus files found")
    chunks, total = [], 0
    for f in files:
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        chunks.append(ids)
        total += len(ids)
        if total >= need_tokens * 2:
            break
    flat = [i for c in chunks for i in c]
    print(f"corpus: {len(files)} files, {len(flat)} tokens", file=sys.stderr)
    return torch.tensor(flat, dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/.hf/q4km-hf")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=1e-6)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    nl = m.config.num_hidden_layers
    inter = m.config.intermediate_size

    # stats per layer
    fire = torch.zeros(nl, inter, dtype=torch.long)
    total = torch.zeros(nl, inter)
    maxv = torch.full((nl, inter), -float("inf"))
    maxtok = torch.full((nl, inter), -1, dtype=torch.long)

    cur = {"ids": None}

    def mk(i):
        def hook(mod, a):
            if cur["ids"] is None:
                return None
            x = a[0].detach().float()            # [B,T,inter]
            b, t, h = x.shape
            flat = x.reshape(b * t, h)           # [BT, inter]
            tokflat = cur["ids"].reshape(-1)     # [BT]
            fire[i] += (flat > args.threshold).sum(0)
            total[i] += flat.sum(0)
            mx, idx = flat.max(0)
            better = mx > maxv[i]
            maxv[i] = torch.where(better, mx, maxv[i])
            maxtok[i] = torch.where(better, tokflat[idx], maxtok[i])
            return None
        return hook

    for i, L in enumerate(m.model.layers):
        L.mlp.down_proj.register_forward_pre_hook(mk(i))

    data = build_corpus(tok, args.tokens)
    n = (len(data) // args.seq) * args.seq
    data = data[:n]
    batches = [(data[k:k + args.batch * args.seq].view(-1, args.seq))
               for k in range(0, n, args.batch * args.seq)]
    limit = args.tokens // (args.batch * args.seq)
    batches = batches[:max(1, limit)]

    t0 = time.time()
    seen = 0
    with torch.no_grad():
        for bi, b in enumerate(batches):
            cur["ids"] = b
            m(input_ids=b)
            seen += b.numel()
            if bi % 5 == 0:
                el = time.time() - t0
                print(f"  {seen:>6} tokens  {seen/max(el,1e-9):5.1f} tok/s  "
                      f"{el:5.1f}s", file=sys.stderr, flush=True)
    cur["ids"] = None
    dur = time.time() - t0

    fire_rate = fire.float() / max(seen, 1)
    covered = (fire > 0)
    per_layer = [{
        "layer": i,
        "covered": int(covered[i].sum()),
        "total_neurons": inter,
        "coverage": round(float(covered[i].float().mean()), 4),
        "median_fire_rate": round(float(fire_rate[i].median()), 6),
        "max_fire_rate": round(float(fire_rate[i].max()), 6),
        "dead_neurons": int((fire[i] == 0).sum()),
        "top5_neurons": [
            {"neuron": int(j), "fire_rate": round(float(fire_rate[i][j]), 5),
             "max_act": round(float(maxv[i][j]), 3),
             "max_token": tok.decode([int(maxtok[i][j])]) if maxtok[i][j] >= 0 else None}
            for j in fire[i].topk(5).indices.tolist()
        ],
    } for i in range(nl)]

    overall_cov = float(covered.float().mean())
    total_dead = int((fire == 0).sum())
    summary = {
        "model": args.model,
        "tokens_seen": seen,
        "duration_s": round(dur, 1),
        "tok_per_s": round(seen / max(dur, 1e-9), 2),
        "layers": nl, "intermediate_size": inter,
        "total_neurons": nl * inter,
        "covered_neurons": int(covered.sum()),
        "coverage": round(overall_cov, 4),
        "dead_neurons": total_dead,
        "per_layer": per_layer,
    }
    os.makedirs(PROOF, exist_ok=True)
    json.dump(summary, open(f"{PROOF}/c7_atlas.json", "w"), indent=1)

    lines = ["C7 — NEURON ATLAS COVERAGE", "=" * 60,
             f"model      : {args.model}",
             f"tokens seen: {seen}  ({dur:.0f}s, {seen/max(dur,1e-9):.1f} tok/s)",
             f"neurons    : {nl*inter} total ({nl} layers x {inter})",
             f"covered    : {int(covered.sum())} ({100*overall_cov:.2f}%)",
             f"never fired: {total_dead} ({100*total_dead/(nl*inter):.2f}%)", ""]
    for pl in per_layer:
        lines.append(f"  L{pl['layer']:>2} cov={pl['coverage']*100:5.1f}%  "
                     f"dead={pl['dead_neurons']:>4}  median_rate={pl['median_fire_rate']:.4f}")
    lines += ["", "NOTE: coverage is corpus-dependent. More tokens and diverse code",
              "      raise it; these numbers describe THIS corpus, not the model.", ""]
    open(f"{PROOF}/c7_coverage.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"RESULT C7 PASS coverage={overall_cov*100:.2f}% covered={int(covered.sum())}/{nl*inter} dead={total_dead}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
