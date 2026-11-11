#!/usr/bin/env python3
"""
gguf_to_hf.py — convert a Qwen2 GGUF (any K-quant) back into a HuggingFace
safetensors directory, so we can hook it with transformers.

This is deliberately Qwen2-specific and explicit: no name-map magic, no silent
guesses. If a tensor name is unknown it aborts rather than dropping weights.

Shape rule: GGUF stores 2-D weights as [in_features, out_features]; HF wants
[out_features, in_features]. 1-D tensors (norms, biases) are untouched.

Usage:
  python code/gguf_to_hf.py <gguf> <outdir> [--template-hf <snapshot_dir>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

try:
    from gguf import GGUFReader, quants
except ImportError:
    sys.exit("gguf not installed: .venv/bin/python -m pip install gguf")


def map_name(name: str) -> str | None:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if not name.startswith("blk."):
        return None
    _, idx, rest = name.split(".", 2)
    p = f"model.layers.{idx}."
    table = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    return p + table[rest] if rest in table else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("outdir")
    ap.add_argument("--template-hf", default=None,
                    help="HF snapshot dir to copy config/tokenizer from")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    r = GGUFReader(args.gguf)
    print(f"gguf tensors: {len(r.tensors)}", file=sys.stderr)

    tensors: dict[str, np.ndarray] = {}
    unknown: list[str] = []
    for t in r.tensors:
        hf = map_name(t.name)
        if hf is None:
            unknown.append(t.name)
            continue
        d = quants.dequantize(t.data, t.tensor_type)
        # NOTE: quants.dequantize already returns HF-compatible [out, in] shapes.
        # `t.shape` reports the GGUF [in, out] convention, which is NOT what the
        # dequantized numpy array uses. Transposing here silently corrupts every
        # 2-D weight (verified: it produced a model that loads but is wrong).
        d = d.astype(np.float16 if args.dtype == "float16" else np.float32, copy=False)
        # keep norms/biases in fp32 for numerical fidelity
        if d.ndim == 1:
            d = d.astype(np.float32, copy=False)
        tensors[hf] = np.ascontiguousarray(d)

    if unknown:
        sys.exit(f"UNKNOWN TENSOR NAMES (refusing to drop weights): {unknown[:10]}")

    print(f"mapped {len(tensors)} tensors", file=sys.stderr)
    save_file(tensors, str(out / "model.safetensors"))

    # copy config + tokenizer from the reference snapshot
    if args.template_hf:
        src = Path(args.template_hf)
        for f in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt", ".gitattributes"):
            if (src / f).exists():
                shutil.copy2(src / f, out / f)
        print(f"copied tokenizer/config from {src}", file=sys.stderr)

    cfg = json.loads((out / "config.json").read_text())
    cfg["torch_dtype"] = "float16"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"OK -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
