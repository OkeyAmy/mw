#!/usr/bin/env python3
"""
f2_gradient_free.py — decisive test for F2.

Question: given ONLY an API-shaped oracle (no gradients, no activations,
optionally no logprobs), can a gradient-free loop find a prompt that reproduces
a known activation-steering vector's effect, and does it do so through the SAME
internal change?

We simulate three oracles on the local bf16 model, so the "API" is free and the
white-box check is available afterwards:

  O20  top-20 tokens + their logprobs          (logprob-rich API)
  O1L  top-1 token + its logprob only          (top-1 API)
  O1I  top-1 token identity only               (label-only API)

Search = greedy coordinate ascent over the last 4 tokens, pool of single-token
candidates, 3 passes.  Same query budget for every oracle.

Readout (white box, after the search):
  TE        logsumexp(logits[comment]) - logsumexp(logits[code])   at last pos
  cos8      cosine(prompt-induced L8 residual delta, steering vector v)
  cosLogit  cosine(prompt-induced full logit delta, vector's logit delta)

Writes proof/f2_gradient_free.json
"""
from pathlib import Path
from __future__ import annotations

import json
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(4)
M = f"{ROOT}/.hf/bf16-hf"
OUT = f"{ROOT}/proof/f2_gradient_free.json"
LAYER = 8
COEF = 2.0

TRAIN_POS = [
    "# add two numbers and return the result\n",
    "# read the file and parse each line as json\n",
    "# raise an error if the input is empty\n",
    "# loop over every item in the list\n",
    "# this class stores user credentials\n",
    "# return the cached value if present\n",
    "# validate the request before processing\n",
    "# write the output to disk\n",
    "# handle the case where the key is missing\n",
    "# compute the mean of the array\n",
    "# connect to the database\n",
    "# send the payload to the server\n",
    "# sort the results by score descending\n",
    "# close the socket when finished\n",
    "# retry the request three times\n",
    "# convert the timestamp to utc\n",
]
TRAIN_NEG = [
    "def add(a, b):\n", "for line in f:\n", "if not data:\n", "for x in items:\n",
    "class User:\n", "return cache.get(k)\n", "def handle(req):\n",
    "with open(p, 'w') as f:\n", "try:\n", "total = sum(xs)\n",
    "conn = sqlite3.connect(p)\n", "requests.post(url, json=body)\n",
    "results.sort(key=lambda r: -r.score)\n", "sock.close()\n",
    "for _ in range(3):\n", "ts = datetime.utcfromtimestamp(t)\n",
]
BASE = "def parse(x):\n"
POOL = ["#", " #", "##", "\n", " ", "note", "//", "explain", "this", "here",
        "TODO", "comment", "so", "ok", "why", "the"]
SUFFIX_LEN = 4
PASSES = 3

COMMENT_STR = ["#", "# ", " #", "##", "# this", "# return"]
CODE_STR = ["return", "    return", "def", "for", "if", "class", "x", "self"]


def first_ids(tok, ss):
    out = []
    for s in ss:
        ids = tok(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def main() -> int:
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(
        M, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    comment_ids = first_ids(tok, COMMENT_STR)
    code_ids = first_ids(tok, CODE_STR)

    # single-token candidates only
    pool = []
    for s in POOL:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            pool.append((s, ids[0]))
    print("pool", [(s, i) for s, i in pool], flush=True)

    # ---- steering vector at L8 (diff-in-means, as in steering_test.py) ----
    caps = {}

    def mk_cap(i):
        def hook(mod, a, o):
            t = o[0] if isinstance(o, tuple) else o
            caps[i] = t[:, -1, :].detach().clone()
        return hook

    model.model.layers[LAYER].register_forward_hook(mk_cap(LAYER))

    steer = {"on": False, "vec": None, "coef": 0.0}

    def mk_pre(i):
        def hook(mod, a):
            if not steer["on"] or i != LAYER:
                return None
            h = a[0].clone()
            h[:, -1, :] = h[:, -1, :] + steer["coef"] * steer["vec"]
            return (h,) + tuple(a[1:])
        return hook

    for i, L in enumerate(model.model.layers):
        L.register_forward_pre_hook(mk_pre(i))

    @torch.inference_mode()
    def run_one(text):
        ids = tok(text, return_tensors="pt").input_ids
        out = model(input_ids=ids).logits[0, -1, :].float()
        return out, caps[LAYER][0].float()

    @torch.inference_mode()
    def run_ids(seqs):
        """seqs: list of equal-length id lists."""
        ids = torch.tensor(seqs)
        out = model(input_ids=ids).logits[:, -1, :].float()
        return out, caps[LAYER].float()

    def run_batch(texts):
        return run_ids([tok(t, add_special_tokens=False).input_ids for t in texts])

    def te(logits):
        return float(torch.logsumexp(logits[comment_ids], 0)
                     - torch.logsumexp(logits[code_ids], 0))

    # residual for the vector must be captured without steering (one at a time:
    # no pad token, and last-position gather differs under padding)
    hp = torch.stack([run_one(t)[1] for t in TRAIN_POS])
    hn = torch.stack([run_one(t)[1] for t in TRAIN_NEG])
    v = hp.mean(0) - hn.mean(0)
    resid_norm = float(hp.norm(dim=-1).mean())
    v = v / max(float(v.norm()), 1e-8) * resid_norm
    vhat = v / v.norm()
    print(f"vector built, norm {float(v.norm()):.2f}", flush=True)

    # vector's own effect on the base prompt
    base_logits, base_h = run_batch([BASE])
    steer.update(on=True, vec=v, coef=COEF)
    vec_logits, vec_h = run_batch([BASE])
    steer.update(on=False)
    te_base = te(base_logits[0])
    te_vec = te(vec_logits[0])
    dlogit_v = (vec_logits[0] - base_logits[0])
    print(f"TE base {te_base:.2f} -> vector {te_vec:.2f} (+{te_vec-te_base:.2f})", flush=True)

    # ---- oracle views of a batch of candidate prompts ---------------------
    def oracle_scores(logits, kind):
        """Return a scalar score per row, using only what `kind` exposes."""
        sc = []
        for lg in logits:
            if kind == "O20":
                top = torch.topk(lg, 20)
                vis = {int(i): float(p) for i, p in zip(top.indices, top.values)}
                hits = [p for i, p in vis.items() if i in comment_ids]
                sc.append(max(hits) if hits else min(vis.values()) - 1.0)
            elif kind == "O1L":
                p, i = torch.max(lg, 0)
                p = float(p)
                sc.append(p if int(i) in comment_ids else p - 10.0)
            else:  # O1I
                _, i = torch.max(lg, 0)
                sc.append(1.0 if int(i) in comment_ids else 0.0)
        return sc

    base_ids = tok(BASE, add_special_tokens=False).input_ids

    def search(kind):
        # start from the 4-token suffix "\n \n \n \n"
        nl = tok("\n", add_special_tokens=False).input_ids[0]
        sfx = [nl] * SUFFIX_LEN
        informative = 0
        steps = 0
        for _ in range(PASSES):
            for pos_i in range(SUFFIX_LEN):
                cands = []
                for _, tid in pool:
                    trial = list(sfx)
                    trial[pos_i] = tid
                    cands.append(trial)
                lg, _ = run_ids([base_ids + c for c in cands])
                sc = oracle_scores(lg, kind)
                steps += 1
                if len(set(sc)) > 1:
                    informative += 1
                sfx = list(cands[max(range(len(sc)), key=lambda k: sc[k])])
        text = tok.decode(base_ids + sfx)
        return base_ids + sfx, text, informative / steps

    results = {}
    for kind in ["O20", "O1L", "O1I"]:
        seq, text, inf = search(kind)
        lg, h8 = run_ids([seq])
        te_p = te(lg[0])
        d8 = (h8[0] - base_h[0])
        cos8 = float(torch.nn.functional.cosine_similarity(
            d8.unsqueeze(0), vhat.unsqueeze(0)))
        dlog = lg[0] - base_logits[0]
        cosl = float(torch.nn.functional.cosine_similarity(
            dlog.unsqueeze(0), dlogit_v.unsqueeze(0)))
        results[kind] = {
            "prompt": text, "TE": te_p, "TE_gain_vs_base": te_p - te_base,
            "frac_of_vector_effect": (te_p - te_base) / max(1e-9, te_vec - te_base),
            "cos_L8_delta_vs_v": cos8,
            "cos_logit_delta_vs_vector": cosl,
            "informative_step_frac": inf,
            "n_queries": PASSES * SUFFIX_LEN * len(pool),
        }
        print(kind, json.dumps(results[kind], indent=1), flush=True)

    # ---- null: random suffixes from the same pool -------------------------
    g = torch.Generator().manual_seed(11)
    null = []
    for _ in range(120):
        sfx = [pool[int(torch.randint(0, len(pool), (1,), generator=g))][1]
               for _ in range(SUFFIX_LEN)]
        null.append(base_ids + sfx)
    te_n, cos_n = [], []
    for i in range(0, len(null), 20):
        lg, h8 = run_ids(null[i:i + 20])
        for j in range(lg.shape[0]):
            te_n.append(te(lg[j]))
            d8 = h8[j] - base_h[0]
            cos_n.append(float(torch.nn.functional.cosine_similarity(
                d8.unsqueeze(0), vhat.unsqueeze(0))))
    srt = sorted(te_n)
    p95_te = srt[int(0.95 * (len(srt) - 1))]
    p95_cos = sorted(cos_n)[int(0.95 * (len(cos_n) - 1))]
    print(f"null TE p95 {p95_te:.2f} (max {max(te_n):.2f}), "
          f"cos p95 {p95_cos:.3f} (max {max(cos_n):.3f})", flush=True)

    out = {
        "model": "Qwen2.5-Coder-1.5B bf16, L8 diff-in-means, coef 2.0",
        "base_prompt": BASE,
        "vector": {"TE_base": te_base, "TE_steered": te_vec,
                   "delta": te_vec - te_base, "norm": float(v.norm())},
        "oracles": results,
        "null": {"n": len(te_n), "TE_p95": p95_te, "TE_max": max(te_n),
                 "cos_p95": p95_cos, "cos_max": max(cos_n),
                 "TE_mean": statistics.mean(te_n),
                 "cos_mean": statistics.mean(cos_n)},
        "cost_note": "120 random suffixes used for the null; 12 candidates x 4 positions x 3 passes = 144 queries per oracle",
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
