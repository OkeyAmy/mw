#!/usr/bin/env python3
"""
mw.certify — quantisation decision-flip certificate (v2, post-audit).

Answers one question with numbers: **which next-token decisions does this
quantisation change, and by how much?** Not a benchmark average.

Two audits shaped this file (agents/C1_certificate_thesis.md,
agents/C2_certificate_code.md):

  C2-D3  the cache had no fingerprint: a different model pair with different
         prompts could silently reuse old arrays and print a false certificate.
         Fixed: every cache carries a hash of (code version, model, prompts,
         dtype) and is ignored on mismatch.
  C2-D1  argmax was taken from the float16 cache; on a near-tie fp16 rounds two
         log-probs equal and numpy picks the lower token id. Fixed: the float32
         argmax is recorded at measurement time and compare() uses that.
  C2-D5  no tokenizer equality check. Fixed: meta carries a vocab fingerprint.
  C2-D9  no uncertainty. Fixed: Wilson 95% interval, plus unique-prompt count,
         because repeating 41 templates to reach 120 is not 120 samples.
  C1     a 5% gate sits inside the dtype noise floor (bf16 vs f16 flips a few
         percent by itself). Fixed: `--null-control` measures the same model in
         a second dtype and prints that floor beside the result.

Usage:
  PYTHONPATH=code python -m mw.certify \
      --reference .hf/bf16-hf --candidate .hf/q4km-hf \
      --builtin 41 --null-control --out proof/cert_qwen_q4

Exit code: 0 if flip rate <= threshold, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import guard

CODE_VERSION = "certify-v2"
TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16,
                "bfloat16": torch.bfloat16}

PROMPTS_CODE = [
    "def add(a, b):\n", "def merge(a, b):\n", "class Cache:\n", "async def fetch(url):\n",
    "def parse(x):\n", "with open(path) as fh:\n", "while queue:\n", "def to_json(obj):\n",
    "for k, v in d.items():\n", "def main():\n", "try:\n", "import os\n",
    "return {\n", "if __name__ == '__main__':\n", "def __init__(self, x):\n",
    "raise ValueError(\n", "for i in range(10):\n", "print(f\"{x}\")\n",
    "def fib(n):\n    return n if n < 2 else\n", "sorted(items, key=lambda x:\n",
    "with lock:\n", "assert isinstance(x, int)\n", "yield from gen()\n",
    "lambda x: x + 1\n", "def wrapper(*args, **kwargs):\n",
]
PROMPTS_TEXT = [
    "The capital of France is", "The largest planet is", "Water boils at",
    "The author of Hamlet was", "Two plus two equals", "The sun rises in the",
    "Machine learning is a field of", "The first president of the United States was",
    "A triangle has", "The chemical symbol for gold is", "The year 1969 saw",
    "In Python, a list is", "The quick brown fox", "Once upon a time",
    "The theory of relativity was proposed by", "Photosynthesis converts",
]


def build_prompts(n: int) -> list[str]:
    """Cycles the base set; callers must report the UNIQUE count (see result)."""
    base = PROMPTS_CODE + PROMPTS_TEXT
    return [base[i % len(base)] for i in range(n)]


def tokenizer_fingerprint(tok) -> str:
    vocab = tok.get_vocab()
    h = hashlib.sha256()
    for k in sorted(vocab)[:5000]:
        h.update(str(k).encode())
        h.update(b"\x00")
        h.update(str(vocab[k]).encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def cache_fingerprint(model: str, prompts: list[str], dtype: str) -> str:
    h = hashlib.sha256()
    h.update(CODE_VERSION.encode())
    h.update(b"\x00")
    h.update(str(Path(model).resolve()).encode())
    h.update(b"\x00")
    h.update(dtype.encode())
    h.update(b"\x00")
    h.update("\x00".join(prompts).encode())
    return h.hexdigest()[:16]


def log_probs(model, tok, prompts: list[str], model_path: str, dtype: str,
              topk: int = 20) -> tuple[np.ndarray, list[dict]]:
    rows, meta = [], []
    tok_fp = tokenizer_fingerprint(tok)
    t0 = time.time()
    for k, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0, -1, :].float()
        lp = lg.log_softmax(-1)
        rows.append(lp.half().numpy())
        v, i = lp.topk(topk)
        argmax = int(lg.argmax())          # float32 decision, not the fp16 cache
        meta.append({
            "prompt": p,
            "argmax": argmax,
            "argmax_tok": tok.decode([argmax]),
            "margin": float(v[0] - v[1]),
            "tok_hash": tok_fp,
            "model": str(model_path),
            "dtype": dtype,
            "top": [[int(j), tok.decode([int(j)]), round(float(q), 4)]
                    for q, j in zip(v.tolist(), i.tolist())],
        })
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(prompts)}  {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return np.stack(rows), meta


def measure(path: str, prompts: list[str], dtype: str, cache: Path):
    fp = cache_fingerprint(path, prompts, dtype)
    if cache.exists():
        try:
            d = np.load(cache, allow_pickle=True)
            if str(d["fp"]) == fp:
                print(f"[cert] cache hit {cache.name}", file=sys.stderr)
                return d["logp"], json.loads(str(d["meta"]))
            print(f"[cert] cache stale ({cache.name}); recomputing", file=sys.stderr)
        except Exception:  # noqa: BLE001 - unreadable cache is simply ignored
            print(f"[cert] cache unreadable ({cache.name}); recomputing", file=sys.stderr)

    guard.require(path, dtype)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=TORCH_DTYPES[dtype], low_cpu_mem_usage=True).eval()
    lp, meta = log_probs(m, tok, prompts, path, dtype)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logp=lp, meta=json.dumps(meta), fp=fp)
    del m
    return lp, meta


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compare(ref_lp, ref_meta, cand_lp, cand_meta) -> dict:
    n = len(ref_meta)
    if len(cand_meta) != n or ref_lp.shape[0] != n or cand_lp.shape[0] != n:
        raise ValueError(
            f"row count mismatch: ref={len(ref_meta)} cand={len(cand_meta)} "
            f"ref_lp={ref_lp.shape[0]} cand_lp={cand_lp.shape[0]}")
    if ref_meta and ref_meta[0].get("tok_hash") != cand_meta[0].get("tok_hash"):
        raise ValueError(
            "tokenizer mismatch: reference and candidate map token ids to different "
            "strings, so a 'flip' would be a tokenizer artefact, not a model change")

    flips, risky, rows = [], [], []
    for a, b, ra, rb in zip(ref_lp, cand_lp, ref_meta, cand_meta):
        ia, ib = int(ra["argmax"]), int(rb["argmax"])   # float32 decisions
        pa = torch.from_numpy(a.astype(np.float32))
        pb = torch.from_numpy(b.astype(np.float32))
        kl = float((pa.exp() * (pa - pb)).sum())
        row = {"prompt": ra["prompt"], "ref": ia, "cand": ib,
               "ref_tok": ra["argmax_tok"], "cand_tok": rb["argmax_tok"],
               "margin_ref": round(float(ra["margin"]), 4), "kl": round(kl, 4),
               "flip": ia != ib}
        rows.append(row)
        if ia != ib:
            flips.append(row)
            if float(ra["margin"]) > 1.0:
                risky.append(row)

    lo, hi = wilson(len(flips), n)
    unique = len({r["prompt"] for r in rows})
    return {
        "n_prompts": n,
        "n_unique": unique,
        "duplicates": n - unique,
        "flips": len(flips),
        "flip_rate": round(len(flips) / max(n, 1), 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "confident_flips": len(risky),
        "max_kl": round(max((r["kl"] for r in rows), default=0.0), 4),
        "mean_kl": round(float(np.mean([r["kl"] for r in rows])) if rows else 0.0, 4),
        "rows": rows,
    }


def render(res: dict, ref: str, cand: str, threshold: float,
           null_res: dict | None = None) -> str:
    within = res["flip_rate"] <= threshold
    certifiable = res["ci95_high"] <= threshold
    lines = [
        "# Quantisation decision-flip certificate",
        "",
        f"- reference : `{ref}`",
        f"- candidate : `{cand}`",
        f"- prompts   : {res['n_prompts']} measured, **{res['n_unique']} unique**"
        + (f" ({res['duplicates']} repeats)" if res["duplicates"] else ""),
        f"- gate      : flip rate <= {threshold:.0%}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| decisions flipped | **{res['flips']} / {res['n_prompts']}** ({res['flip_rate']:.1%}) |",
        f"| 95% CI on flip rate | [{res['ci95_low']:.1%}, {res['ci95_high']:.1%}] |",
        f"| confident flips (ref margin > 1 nat) | **{res['confident_flips']}** |",
        f"| mean KL(ref‖cand) | {res['mean_kl']:.4f} |",
        f"| max KL | {res['max_kl']:.4f} |",
    ]
    if null_res is not None:
        lines += [
            ("| **null control** (same model, other dtype) | "
             f"**{null_res['flip_rate']:.1%}** flips "
             f"[{null_res['ci95_low']:.1%}, {null_res['ci95_high']:.1%}] |"),
            "",
            ("> The null control is the floor: a same-model dtype conversion already "
             f"flips {null_res['flip_rate']:.1%} of decisions. Any candidate flip rate "
             "at or below this is indistinguishable from storage format."),
        ]
    lines += [
        "",
        "## Gate",
        "",
        ("- flip rate is **within** the threshold." if within
         else "- flip rate **exceeds** the threshold."),
        ("- the result is **certifiable** at this threshold (95% upper bound "
         f"{res['ci95_high']:.1%} <= {threshold:.0%})." if certifiable else
         f"- **not certifiable** at {threshold:.0%}: with {res['n_unique']} unique prompts "
         f"the 95% upper bound is {res['ci95_high']:.1%}. Add prompts or widen the gate."),
        "",
        ("> A low flip rate is not a statement that the model is safe. It is a "
         "statement about the decisions in this prompt set."),
        "",
        "## Flips",
        "",
    ]
    flips = [r for r in res["rows"] if r["flip"]]
    if not flips:
        lines.append("_none_")
    else:
        lines += ["| prompt | reference | candidate | ref margin | KL |",
                  "|---|---|---|---|---|"]
        for r in flips:
            p = r["prompt"].replace("\n", "\\n").replace("|", "\\|")[:40]
            lines.append(f"| `{p}` | `{r['ref_tok']}` | `{r['cand_tok']}` | "
                         f"{r['margin_ref']:.2f} | {r['kl']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--builtin", type=int, default=41)
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--dtype", default="float32", choices=list(TORCH_DTYPES))
    ap.add_argument("--null-control", action="store_true",
                    help="also measure the reference in --null-dtype to establish "
                         "the dtype noise floor")
    ap.add_argument("--null-dtype", default="bfloat16", choices=list(TORCH_DTYPES))
    ap.add_argument("--out", default="proof/cert")
    args = ap.parse_args()

    if args.prompts:
        prompts = [p for p in Path(args.prompts).read_text().splitlines() if p.strip()]
    else:
        prompts = build_prompts(args.builtin)
    if not prompts:
        raise SystemExit("refusing to certify an empty prompt set")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[cert] {len(prompts)} prompts "
          f"({len(set(prompts))} unique)", file=sys.stderr)
    print("[cert] reference pass", file=sys.stderr)
    ref_lp, ref_meta = measure(args.reference, prompts, args.dtype, out / "ref.npz")
    print("[cert] candidate pass", file=sys.stderr)
    cand_lp, cand_meta = measure(args.candidate, prompts, args.dtype, out / "cand.npz")

    res = compare(ref_lp, ref_meta, cand_lp, cand_meta)

    null_res = None
    if args.null_control and args.null_dtype != args.dtype:
        print(f"[cert] null control: reference in {args.null_dtype}", file=sys.stderr)
        n_lp, n_meta = measure(args.reference, prompts, args.null_dtype,
                               out / f"null_{args.null_dtype}.npz")
        null_res = compare(ref_lp, ref_meta, n_lp, n_meta)
        res["null_control"] = {k: null_res[k] for k in
                               ("flip_rate", "ci95_low", "ci95_high", "flips")}

    (out / "report.md").write_text(
        render(res, args.reference, args.candidate, args.threshold, null_res))
    (out / "report.json").write_text(json.dumps(res, indent=1))
    print(render(res, args.reference, args.candidate, args.threshold, null_res))
    return 0 if res["flip_rate"] <= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
