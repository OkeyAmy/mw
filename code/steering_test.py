#!/usr/bin/env python3
"""
steering_test.py — the decisive test for C4.

Pre-registered before running. The question: does a diff-in-means steering
vector do something a RANDOM vector of matched norm cannot?

Four independent tests, all required:

  T1 SELECTIVITY      real target effect > 95th pct of 20 random vectors
  T2 COLLATERAL       real control effect <= median of random controls
                      (a real direction should NOT disturb unrelated logits)
  T3 BEHAVIOUR        generating with the vector produces more comment lines
                      than without, by more than 95th pct of random vectors
  T4 HELD-OUT SIGN    the effect direction is consistent on prompts never used
                      to build the vector (>=80%, and above random p95)

If T1-T4 pass -> steering is real and specific.
If they fail  -> diff-in-means steering at this scale is not distinguishable
                 from noise, and every product based on "control" is unfounded.

The vector is built from mw.adapter.DEFAULT_POS / DEFAULT_NEG (8 contrast
pairs). TRAIN_POS / TRAIN_NEG, the 16 hand-written pairs from an earlier draft,
were never used and have been deleted.

Usage: python code/steering_test.py [--model PATH] [--quick]
Writes: proof/c4_decisive.{json,txt} (decisive)
        proof/c4_quick.{json,txt}    (--quick: SCREEN ONLY, not decisive)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from mw.adapter import DEFAULT_NEG, DEFAULT_POS
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(4)
PROOF = f"{ROOT}/proof"
LAYER = 8
EXPECT_LAYERS = 40

# ---- HELD-OUT prompts (never used to build the vector) ---------------------
HELDOUT = [
    "def merge(a, b):\n", "class Cache:\n", "async def fetch(url):\n",
    "def parse(x):\n", "with open(path) as fh:\n", "while queue:\n",
    "def to_json(obj):\n", "for k, v in d.items():\n", "def main():\n",
    "try:\n",
]

COMMENT_TOKS = ["#", "# ", " #", "##", "# this", "# return"]
CODE_TOKS = ["return", "    return", "def", "for", "if", "class", "x", "self"]
YES_TOKS = [" yes", "yes", " Yes"]
NO_TOKS = [" no", "no", " No"]
DIGIT_TOKS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def first_ids(tok, ss):
    out = []
    for s in ss:
        ids = tok(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids[0])
    return sorted(set(out))


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


# ---- hard guards (H12) -----------------------------------------------------
# A zero / non-finite vector, or a steering hook that changes nothing, must
# never produce an artifact. Both failures are silent no-ops otherwise.
def guard_vector(vec, name="steering vector") -> float:
    """Raise RuntimeError unless `vec` is finite and has norm > 1e-6."""
    if vec is None:
        raise RuntimeError(f"{name} is None (no vector was built)")
    if not bool(torch.isfinite(vec).all()):
        raise RuntimeError(f"{name} contains NaN/Inf")
    norm = float(vec.norm())
    if not (norm > 1e-6):
        raise RuntimeError(f"{name} has norm {norm:.3e} <= 1e-6 (no-op steering)")
    return norm


def guard_steering_changes_logits(base_logits, steered_logits, tol=1e-6) -> float:
    """Raise RuntimeError unless steering moved at least one logit by > tol."""
    if base_logits is None or steered_logits is None:
        raise RuntimeError("guard needs base and steered logits")
    delta = float((steered_logits - base_logits).abs().max())
    if not (delta > tol):
        raise RuntimeError(
            f"steering changed no logit (max|steered-base| = {delta:.3e} <= {tol:.0e}); "
            "hook is inactive or the vector cancels"
        )
    return delta


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_model_config(model_type, n_layers, path) -> None:
    """Raise RuntimeError unless this is the pre-registered Granite model."""
    if model_type != "granite" or n_layers != EXPECT_LAYERS:
        raise RuntimeError(
            f"unsupported model for the C4 test: model_type={model_type!r}, "
            f"layers={n_layers}, expected 'granite' / {EXPECT_LAYERS} "
            f"(got --model {path}). The pre-registered run is IBM Granite 3.3 2B "
            f"at layer {LAYER}; use --model .hf/granite-hf."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/.hf/granite-hf")
    ap.add_argument("--randoms", type=int, default=20)
    ap.add_argument("--behav-randoms", type=int, default=10)
    ap.add_argument("--behav-tokens", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.randoms, args.behav_randoms, args.behav_tokens = 6, 4, 6

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    # ---- model guard (H11): this harness only speaks Granite L8 -------------
    model_type = getattr(model.config, "model_type", None)
    n_layers = len(model.model.layers)
    try:
        check_model_config(model_type, n_layers, args.model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    steer = {"layer": None, "vec": None, "coef": 0.0}

    def mk_pre(i):
        def hook(mod, a):
            if steer["layer"] != i or steer["coef"] == 0.0:
                return None
            h = a[0].clone()
            h[:, -1, :] = h[:, -1, :] + steer["coef"] * steer["vec"]
            return (h,) + tuple(a[1:])
        return hook

    for i, L in enumerate(model.model.layers):
        L.register_forward_pre_hook(mk_pre(i))

    caps: dict[int, torch.Tensor] = {}

    def mk_cap(i):
        def hook(mod, a, o):
            t = o[0] if isinstance(o, tuple) else o
            caps[i] = t.detach()
        return hook

    model.model.layers[LAYER].register_forward_hook(mk_cap(LAYER))

    comment_ids = first_ids(tok, COMMENT_TOKS)
    code_ids = first_ids(tok, CODE_TOKS)
    yes_ids = first_ids(tok, YES_TOKS)
    no_ids = first_ids(tok, NO_TOKS)
    digit_ids = first_ids(tok, DIGIT_TOKS)

    @torch.no_grad()
    def last_logits(text):
        ids = tok(text, return_tensors="pt").input_ids
        return model(input_ids=ids).logits[0, -1, :].float()

    @torch.no_grad()
    def resid(text):
        ids = tok(text, return_tensors="pt").input_ids
        model(input_ids=ids)
        return caps[LAYER][0].clone()

    def target_eff(lg):
        return float(torch.logsumexp(lg[comment_ids], 0) - torch.logsumexp(lg[code_ids], 0))

    def ctrl_eff(lg):
        a = float(torch.logsumexp(lg[yes_ids], 0) - torch.logsumexp(lg[no_ids], 0))
        b = float(torch.logsumexp(lg[digit_ids], 0) - torch.logsumexp(lg[comment_ids], 0))
        return abs(a) + abs(b)

    # ---- build vector ------------------------------------------------------
    # Same estimator the app ships (adapter.build_steer_vectors): contrast pairs
    # share a prefix and suffix so only the in-body comment differs, and pooling
    # excludes token 0 (the attention-sink spike). Calibrated to the mean
    # last-position residual norm on HELDOUT, matching verify_c4.py.
    def pooled(text):
        row = resid(text)                          # [T, d]
        return (row[1:] if row.shape[0] > 1 else row).mean(0)

    pos = torch.stack([pooled(t) for t in DEFAULT_POS])
    neg = torch.stack([pooled(t) for t in DEFAULT_NEG])
    v = pos.mean(0) - neg.mean(0)
    resid_norm = statistics.mean(float(resid(t)[-1].norm()) for t in HELDOUT)
    v = v / max(float(v.norm()), 1e-8) * resid_norm

    # ---- guards (H12): never score a no-op --------------------------------
    try:
        unit = guard_vector(v)
    except RuntimeError as exc:
        print(f"FATAL: {exc}\nRefusing to run or write any proof artifact.", file=sys.stderr)
        return 1
    print(f"vector built: {len(DEFAULT_POS)} pairs, layer {LAYER}, norm {unit:.2f}"
          f" (residual unit {resid_norm:.2f})", file=sys.stderr)

    # ---- baselines (H3): compute ONCE, steering OFF, before any steer.update
    # The old loop recomputed `base` inside the prompt loop while steering was
    # already active, so prompt 1 always had delta 0. Fixed form, as in
    # verify_c4.py:103-108.
    @torch.no_grad()
    def compute_baselines():
        steer.update(layer=LAYER, vec=None, coef=0.0)
        lgs, tes = {}, []
        for p in HELDOUT:
            lg = last_logits(p)
            lgs[p] = lg
            tes.append(target_eff(lg))
        return lgs, tes

    base_logits, BASE_TE = compute_baselines()

    # ---- guard: the first steered forward must actually move a logit ------
    try:
        with torch.no_grad():
            steer.update(layer=LAYER, vec=v, coef=1.0)
            first_steered = last_logits(HELDOUT[0])
        moved = guard_steering_changes_logits(base_logits[HELDOUT[0]], first_steered)
    except RuntimeError as exc:
        print(f"FATAL: {exc}\nRefusing to run or write any proof artifact.", file=sys.stderr)
        return 1
    finally:
        steer.update(layer=LAYER, vec=None, coef=0.0)
    print(f"steering guard: max|steered-base| on prompt 1 = {moved:.3f}", file=sys.stderr)

    def measure(vec, coef=1.0):
        """Return (mean target effect, mean control effect, sign consistency).

        Target deltas are taken against the precomputed, steering-off baselines;
        control effects stay absolute (unchanged T2 metric, out of scope here).
        """
        te, ce, signs = [], [], []
        try:
            steer.update(layer=LAYER, vec=vec, coef=coef)
            for i, p in enumerate(HELDOUT):
                lg = last_logits(p)          # one forward serves both metrics
                d = target_eff(lg) - BASE_TE[i]
                te.append(d)
                signs.append(1 if d > 0 else 0)
                ce.append(ctrl_eff(lg))
        finally:
            steer.update(layer=LAYER, vec=None, coef=0.0)
        return (statistics.mean(te), statistics.mean(ce), sum(signs) / len(signs))

    real_te, real_ce, real_sign = measure(v)

    rng = torch.Generator().manual_seed(7)
    rand = []
    for _ in range(args.randoms):
        rv = torch.randn(v.shape, generator=rng)
        rv = rv / rv.norm() * v.norm()
        te, ce, sg = measure(rv)
        rand.append({"te": te, "ce": ce, "sign": sg})

    # ---- behaviour ---------------------------------------------------------
    @torch.no_grad()
    def gen_comment_count(vec, coef, prompt):
        try:
            steer.update(layer=LAYER, vec=vec, coef=coef)
            ids = tok(prompt, return_tensors="pt").input_ids
            out_ids = model.generate(ids, max_new_tokens=args.behav_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            steer.update(layer=LAYER, vec=None, coef=0.0)
        txt = tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
        return txt.count("#")

    base_counts = [gen_comment_count(None, 0.0, p) for p in HELDOUT]
    real_counts = [gen_comment_count(v, 1.0, p) for p in HELDOUT]
    behav_delta = statistics.mean(real_counts) - statistics.mean(base_counts)

    rng2 = torch.Generator().manual_seed(99)
    rand_deltas = []
    for _ in range(args.behav_randoms):
        rv = torch.randn(v.shape, generator=rng2)
        rv = rv / rv.norm() * v.norm()
        c = [gen_comment_count(rv, 1.0, p) for p in HELDOUT]
        rand_deltas.append(statistics.mean(c) - statistics.mean(base_counts))

    # ---- verdicts ----------------------------------------------------------
    te_p95 = pct([r["te"] for r in rand], 95)
    ce_med = pct([r["ce"] for r in rand], 50)
    sign_p95 = pct([r["sign"] for r in rand], 95)
    bd_p95 = pct(rand_deltas, 95)

    T1 = real_te > te_p95
    T2 = real_ce <= ce_med
    T3 = behav_delta > bd_p95 and behav_delta > 0
    T4 = real_sign >= 0.8 and real_sign > sign_p95
    ok = T1 and T2 and T3 and T4

    out = {
        # ---- provenance (H16) ---------------------------------------------
        "model": args.model,
        "model_id": getattr(model.config, "_name_or_path", None) or args.model,
        "arch": model_type,
        "dtype": str(next(model.parameters()).dtype),
        "n_layers": n_layers,
        "layer": LAYER,
        "coef": 1.0,
        "unit": float(v.norm()),
        "unit_definition": "mean last-position resid_post L8 norm on HELDOUT; "
                           "vector scaled so coef 1 == this norm",
        "prompt_render": "raw",
        "vector_source": "mw.adapter.DEFAULT_POS/NEG via daemon",
        "n_vector_pairs": len(DEFAULT_POS),
        "behav_randoms": args.behav_randoms,
        "behav_tokens": args.behav_tokens,
        "quick": bool(args.quick),
        "decisive": not args.quick,
        "git_sha": git_sha(),
        # ---- configuration / results --------------------------------------
        "randoms": args.randoms,
        "real": {"target_eff": real_te, "ctrl_eff": real_ce, "sign_consistency": real_sign,
                 "behav_delta": behav_delta, "behav_base": statistics.mean(base_counts),
                 "behav_real": statistics.mean(real_counts)},
        "random": {"target_eff_p95": te_p95, "target_eff_mean": statistics.mean([r["te"] for r in rand]),
                   "ctrl_eff_median": ce_med, "sign_p95": sign_p95, "behav_delta_p95": bd_p95,
                   "target_eff_all": [r["te"] for r in rand]},
        "tests": {"T1_selectivity": T1, "T2_collateral": T2,
                  "T3_behaviour": T3, "T4_heldout_sign": T4},
        "status": "PASS" if ok else "FAIL",
    }
    if args.quick:
        out["SCREEN_ONLY"] = "not decisive: quick mode uses max-of-4/6 gates"
    os.makedirs(PROOF, exist_ok=True)
    tag = "quick" if args.quick else "decisive"
    with open(f"{PROOF}/c4_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)

    banner = "SCREEN ONLY — not decisive" if args.quick else "C4 DECISIVE TEST — is the steering vector special?"
    lines = [
        banner,
        "=" * 62,
        (f"model {out['model_id']} | {model_type} | {out['dtype']} | {n_layers} layers"
         f" | steer L{LAYER} | coef {out['coef']:.1f} | unit {out['unit']:.3f}"),
        (f"{len(DEFAULT_POS)} vector pairs (DEFAULT_POS/NEG) | {len(HELDOUT)} held-out prompts"
         f" | {args.randoms} random controls | render {out['prompt_render']}"),
        (f"git {out['git_sha']} | behav_randoms {args.behav_randoms} | "
         f"behav_tokens {args.behav_tokens} | quick {out['quick']}"),
        "",
        f"{'metric':<26}{'real':>12}{'random ref':>16}{'pass':>8}",
        f"{'T1 target effect':<26}{real_te:>12.3f}{te_p95:>16.3f}{T1!s:>8}",
        f"{'T2 control effect':<26}{real_ce:>12.3f}{ce_med:>16.3f}{T2!s:>8}",
        f"{'T3 behaviour delta':<26}{behav_delta:>12.3f}{bd_p95:>16.3f}{T3!s:>8}",
        f"{'T4 sign consistency':<26}{real_sign:>12.2f}{sign_p95:>16.2f}{T4!s:>8}",
        "",
        f"baseline comment lines per completion: {statistics.mean(base_counts):.2f}",
        f"with real vector                     : {statistics.mean(real_counts):.2f}",
        "",
        f"VERDICT: {'PASS — steering is specific' if ok else 'FAIL — steering is not distinguishable from noise'}",
    ]
    if args.quick:
        lines += ["", "SCREEN ONLY — not decisive. Not written to proof/c4_decisive.*"]
    if not ok:
        lines += ["", "If this fails, the product cannot be 'control'. It must be measurement."]
    with open(f"{PROOF}/c4_{tag}.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"RESULT C4DEC {out['status']} T1={T1} T2={T2} T3={T3} T4={T4} "
          f"real_te={real_te:.3f} rand_p95={te_p95:.3f} behav={behav_delta:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
