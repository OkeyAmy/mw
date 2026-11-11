#!/usr/bin/env python3
"""
sanity.py — C8: the gate that must pass before any result is reported.

Every check here is a way the instrument could lie. If any fails, every other
number produced this session is suspect.

Usage: python code/sanity.py [--model PATH]
Emits:  RESULT C8 PASS|FAIL <summary>
Writes: proof/c8_sanity.txt
"""
from pathlib import Path
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

torch.set_num_threads(4)
PROOF = f"{ROOT}/proof"

PROMPT = "def add(a, b):\n    return a + b\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".hf" / "bf16-hf"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    cap: dict[int, torch.Tensor] = {}
    pre_cap: dict[int, torch.Tensor] = {}
    steer = {"layer": None, "vec": None, "coef": 0.0}
    patch = {"layer": None, "vec": None}

    def mk_cap(i):
        def h(mod, a, o):
            cap[i] = (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().clone()
        return h

    def mk_pre(i):
        def h(mod, a):
            x = a[0]
            # capture resid_PRE: this is the site the patch and steer hooks write to,
            # so it is the only correct reference for a self-patch no-op test.
            pre_cap[i] = x[:, -1, :].detach().clone()
            if patch["layer"] == i and patch["vec"] is not None:
                x = x.clone()
                x[:, -1, :] = patch["vec"]
            if steer["layer"] == i and steer["coef"] != 0.0:
                x = x + steer["coef"] * steer["vec"]
            return (x,) + tuple(a[1:])
        return h

    for i, L in enumerate(m.model.layers):
        L.register_forward_hook(mk_cap(i))
        L.register_forward_pre_hook(mk_pre(i))

    ids = tok(PROMPT, return_tensors="pt").input_ids
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    @torch.no_grad()
    def fwd():
        return m(input_ids=ids, output_attentions=True)

    # 1 determinism
    with torch.no_grad():
        a = m(input_ids=ids).logits
        b = m(input_ids=ids).logits
    check("determinism_bitidentical", torch.equal(a, b), "same input twice")

    out = fwd()
    real = out.logits
    nl = m.config.num_hidden_layers

    # 2 lens identity
    dl = float((m.lm_head(m.model.norm(cap[nl - 1])) - real[:, -1, :]).abs().max())
    check("lens_identity_at_final_layer", dl < 1e-4, f"max|diff|={dl:.2e}")

    # 3 coeff=0 steering is a true no-op
    steer.update(layer=8, vec=torch.randn(1, m.config.hidden_size), coef=0.0)
    with torch.no_grad():
        z = m(input_ids=ids).logits
    steer.update(layer=None, vec=None)
    check("steer_coef0_bitidentical", torch.equal(z, real), "torch.equal")

    # 4 self-patch is a no-op (must use resid_PRE, the site the patch writes to)
    own = pre_cap[8][0]
    patch.update(layer=8, vec=own)
    with torch.no_grad():
        p = m(input_ids=ids).logits
    patch.update(layer=None, vec=None)
    check("self_patch_noop", torch.equal(p, real), "torch.equal")

    # 5 attention rows normalise
    a0 = out.attentions[0]
    sums = a0.sum(-1)
    check("attention_rows_sum_to_1", bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-4)),
          f"max|sum-1|={float((sums-1).abs().max()):.2e}")

    # 6 no non-finite values anywhere we rely on
    finite = all(bool(torch.isfinite(cap[i]).all()) for i in range(nl)) and bool(torch.isfinite(real).all())
    check("all_captured_tensors_finite", finite, f"{nl} layers")

    # 7 tokenizer round-trip on code
    rt = tok.decode(tok(PROMPT, add_special_tokens=False).input_ids) == PROMPT
    check("tokenizer_roundtrip", rt, "decode(encode(x))==x")

    # 8 eval mode (no dropout)
    check("model_in_eval_mode", not m.training, f"training={m.training}")

    # 9 wrong-layer patch differs from right-layer patch
    other = cap[3][0]
    patch.update(layer=8, vec=other)
    with torch.no_grad():
        w = m(input_ids=ids).logits
    patch.update(layer=None, vec=None)
    check("wrong_vector_changes_output", not torch.equal(w, real),
          f"max|diff|={float((w-real).abs().max()):.2e}")

    # 10 capture shapes
    shapes_ok = all(tuple(cap[i].shape) == (1, m.config.hidden_size) for i in range(nl))
    check("capture_shapes", shapes_ok, f"{nl} x [1,{m.config.hidden_size}]")

    passed = sum(1 for _, ok, _ in checks if ok)
    status = "PASS" if passed == len(checks) else "FAIL"

    lines = ["C8 — SANITY GATE", "=" * 60,
             f"model: {args.model}", f"dtype: {next(m.parameters()).dtype}", ""]
    for name, ok, d in checks:
        lines.append(f"  [{'ok' if ok else 'FAIL'}] {name:38} {d}")
    lines += ["", f"passed {passed}/{len(checks)}", f"VERDICT: {status}", ""]
    if status != "PASS":
        lines.append("!! DO NOT REPORT ANY RESULT FROM THIS SESSION UNTIL FIXED.")
    import os
    os.makedirs(PROOF, exist_ok=True)
    open(f"{PROOF}/c8_sanity.txt", "w").write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"RESULT C8 {status} checks={passed}/{len(checks)} lens_diff={dl:.2e}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
