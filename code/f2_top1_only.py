#!/usr/bin/env python3
"""
f2_top1_only.py <O20|O1L|O1I> — one arm per invocation, results merged into JSON.

Simulated API views of the same local model at an identical query budget:
  O20  top-20 tokens + logprobs
  O1L  top-1 token + its logprob
  O1I  top-1 token identity only (label-only)

Search = greedy coordinate ascent, suffix of 2 tokens over 8 single-token
candidates, 1 pass = 16 queries.

Why split into arms: this box takes ~10 s to load and ~9 s per batch of 8 short
sequences; a single arm fits in one short shell window, a three-arm run does not.
"""
from pathlib import Path
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(4)
M = f"{ROOT}/.hf/bf16-hf"
OUT = f"{ROOT}/proof/f2_top1_only.json"
BASE = "def parse(x):\n"
POOL_STR = ["#", "note", "\n", " ", "def", "return", "this", "here"]
SUF = 2
PASSES = 1
COMMENT_STR = ["#", "# ", " #", "##", "# this", "# return"]
CODE_STR = ["return", "    return", "def", "for", "if", "class", "x", "self"]


def first_ids(tok, ss):
    return sorted({tok(s, add_special_tokens=False).input_ids[0]
                   for s in ss if tok(s, add_special_tokens=False).input_ids})


def main():
    kind = sys.argv[1]
    assert kind in ("O20", "O1L", "O1I")
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(
        M, dtype=torch.float32, attn_implementation="eager").eval()
    comment_ids = first_ids(tok, COMMENT_STR)
    code_ids = first_ids(tok, CODE_STR)
    pool = [(s, tok(s, add_special_tokens=False).input_ids[0])
            for s in POOL_STR
            if len(tok(s, add_special_tokens=False).input_ids) == 1]
    base_ids = tok(BASE, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def logits_of(seqs):
        return model(input_ids=torch.tensor(seqs)).logits[:, -1, :].float()

    def te(lg):
        return float(torch.logsumexp(lg[comment_ids], 0)
                     - torch.logsumexp(lg[code_ids], 0))

    def scores(lg):
        out = []
        for row in lg:
            if kind == "O20":
                top = torch.topk(row, 20)
                vis = {int(i): float(p) for i, p in zip(top.indices, top.values)}
                hits = [p for i, p in vis.items() if i in comment_ids]
                out.append(max(hits) if hits else min(vis.values()) - 1.0)
            elif kind == "O1L":
                p, i = torch.max(row, 0)
                out.append(float(p) if int(i) in comment_ids else float(p) - 10.0)
            else:
                _, i = torch.max(row, 0)
                out.append(1.0 if int(i) in comment_ids else 0.0)
        return out

    base_lg = logits_of([base_ids])
    te0 = te(base_lg[0])
    nl = tok("\n", add_special_tokens=False).input_ids[0]
    sfx = [nl] * SUF
    info, steps, cand_tables = 0, 0, []
    for _ in range(PASSES):
        for pos in range(SUF):
            cands = []
            for _, tid in pool:
                t = list(sfx)
                t[pos] = tid
                cands.append(t)
            lg = logits_of([base_ids + c for c in cands])
            sc = scores(lg)
            steps += 1
            if len(set(sc)) > 1:
                info += 1
            cand_tables.append({"pos": pos,
                                "scores": [round(float(x), 4) for x in sc]})
            sfx = list(cands[max(range(len(sc)), key=lambda k: sc[k])])
    lg = logits_of([base_ids + sfx])
    top1 = int(torch.argmax(lg[0]))

    rec = {
        "suffix": tok.decode(sfx),
        "TE_final": round(te(lg[0]), 4),
        "TE_base": round(te0, 4),
        "TE_gain": round(te(lg[0]) - te0, 4),
        "informative_step_frac": round(info / steps, 3),
        "n_queries": PASSES * SUF * len(pool),
        "argmax_is_comment": top1 in comment_ids,
        "top1_str": tok.decode([top1]),
        "candidate_scores": cand_tables,
    }
    cur = {}
    if os.path.exists(OUT):
        cur = json.load(open(OUT))
    cur.setdefault("meta", {
        "base": BASE, "pool": [p[0] for p in pool], "suffix_len": SUF,
        "passes": PASSES, "queries_per_arm": PASSES * SUF * len(pool),
        "model": "Qwen2.5-Coder-1.5B bf16 (fp32 compute), eager",
    })
    cur.setdefault("arms", {})[kind] = rec
    json.dump(cur, open(OUT, "w"), indent=1)
    print("ARM", kind, json.dumps({k: v for k, v in rec.items()
                                   if k != "candidate_scores"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
