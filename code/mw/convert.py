"""
mw.convert — GGUF -> HF safetensors for any architecture in registry.SPECS.

Design rules:
  * refuse unknown architectures instead of guessing
  * refuse fused tensors (phi3 qkv) instead of silently producing garbage
  * carry architecture scaling factors from GGUF metadata into config.json
  * verify the output by re-parsing its own header before reporting success

Usage:
  python -m mw.convert --gguf <file> --out <dir> --hf-repo ibm-granite/granite-3.3-2b-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .registry import resolve

SMALL_FILES = ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
               ".gitattributes")


def map_tensor(name: str, spec) -> str | None:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if not name.startswith("blk."):
        return None
    _, idx, rest = name.split(".", 2)
    hf = spec.tensor_map.get(rest)
    return f"model.layers.{idx}.{hf}" if hf else None


def fetch_small_files(repo: str, out: Path) -> None:
    """Fetch config/tokenizer files. Failures are reported, not swallowed (G1 D19)."""
    from huggingface_hub import hf_hub_download
    failed = []
    for f in SMALL_FILES:
        try:
            p = hf_hub_download(repo_id=repo, filename=f)
            (out / f).write_bytes(Path(p).read_bytes())
            print(f"  fetched {f}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - network/404/rate-limit all mean "absent"
            failed.append(f"{f}: {type(e).__name__}")
    if failed:
        print(f"  not fetched (may be optional): {failed}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--allow-scaling-default", action="store_true",
                    help="permit missing scaling metadata to fall back to 1.0 "
                         "(almost always wrong; G1 D4)")
    args = ap.parse_args()

    from gguf import GGUFReader, quants

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(args.gguf)
    meta = {}
    for f in r.fields.values():
        if f.name.startswith("tokenizer.ggml.tokens"):
            continue
        try:
            meta[f.name] = f.contents()
        except (KeyError, TypeError, ValueError):
            continue  # metadata field we cannot interpret; not fatal

    gguf_arch = str(meta.get("general.architecture", "?"))
    spec = resolve(gguf_arch)
    if spec is None:
        sys.exit(f"UNSUPPORTED ARCH '{gguf_arch}'. Add it to mw/registry.py. "
                 f"Refusing to guess.")
    if spec.fused:
        sys.exit(f"ARCH '{gguf_arch}' uses fused tensors ({spec.notes}). "
                 f"Conversion not implemented; refusing to produce a wrong model.")
    # G1 D4: scaling factors are load-bearing. A missing key must abort.
    missing_scaling = [gk for gk in spec.config_from_gguf if gk not in meta]
    if missing_scaling and spec.requires_config_from_gguf and not args.allow_scaling_default:
        sys.exit(
            f"REFUSING: {gguf_arch} requires scaling metadata that is missing: "
            f"{missing_scaling}. Assuming 1.0 would produce a numerically wrong "
            f"model (logits can be off by 8x). Pass --allow-scaling-default only "
            f"if you have verified the GGUF genuinely has no scaling.")

    print(f"arch={gguf_arch} -> {spec.hf_arch} ({len(r.tensors)} tensors)", file=sys.stderr)

    tensors: dict[str, np.ndarray] = {}
    unknown = []
    nonfinite = []
    for t in r.tensors:
        hf = map_tensor(t.name, spec)
        if hf is None:
            unknown.append(t.name); continue
        d = quants.dequantize(t.data, t.tensor_type)
        # quants.dequantize returns HF-compatible [out, in]; do NOT transpose.
        if not np.isfinite(d).all():
            nonfinite.append(t.name)
        if args.dtype == "float16" and d.ndim == 2:
            # G1 D20: fp16 overflows above 65504 and silently becomes inf.
            if float(np.abs(d).max()) > 65504.0:
                sys.exit(f"REFUSING fp16: {t.name} max|w|={float(np.abs(d).max()):.0f} > 65504")
            d = d.astype(np.float16, copy=False)
        else:
            d = d.astype(np.float32, copy=False)
        tensors[hf] = np.ascontiguousarray(d)

    if nonfinite:
        sys.exit(f"NON-FINITE WEIGHTS after dequantisation: {nonfinite[:5]}")

    if unknown:
        sys.exit(f"UNKNOWN TENSORS (refusing to drop weights): {unknown[:8]}")
    save_file(tensors, str(out / "model.safetensors"))

    if args.hf_repo:
        fetch_small_files(args.hf_repo, out)

    cfg_path = out / "config.json"
    if not cfg_path.exists():
        sys.exit("no config.json: pass --hf-repo so we can fetch the architecture "
                 "config (scaling factors cannot be guessed)")
    # G1 D19: a model dir without a tokenizer is a broken model, not a success.
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing_files = [f for f in required if not (out / f).exists()]
    if missing_files:
        sys.exit(f"conversion produced an unloadable model dir; missing {missing_files}. "
                 f"Pass --hf-repo for a repo that has them.")

    cfg = json.loads(cfg_path.read_text())

    # G1 D21: tied config + an output tensor alias the same Parameter and one
    # matrix is silently dropped at load. Refuse the mismatch.
    tied = bool(cfg.get("tie_word_embeddings", False))
    has_out = "lm_head.weight" in tensors
    if tied and has_out:
        sys.exit("REFUSING: config says tie_word_embeddings=true but the GGUF has an "
                 "output.weight; loading would alias embed_tokens and lm_head and "
                 "silently drop one matrix.")
    if (not tied) and (not has_out):
        sys.exit("REFUSING: config says tie_word_embeddings=false but the GGUF has no "
                 "output.weight; lm_head would be randomly initialised.")

    # carry scaling factors from GGUF metadata
    applied = {}
    for gk, ck in spec.config_from_gguf.items():
        if gk in meta:
            cfg[ck] = float(meta[gk])
            applied[ck] = float(meta[gk])
    for k, v in spec.defaults.items():
        cfg.setdefault(k, v)
    cfg["torch_dtype"] = args.dtype
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # self-verify
    with open(out / "model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    actual = os.path.getsize(out / "model.safetensors")
    ok = actual == 8 + n + end

    print(f"wrote {len(tensors)} tensors, {actual/1e9:.2f} GB", file=sys.stderr)
    print(f"scaling applied from GGUF: {applied or '{}'}", file=sys.stderr)
    print(f"header self-check: {'OK' if ok else 'MISMATCH'}", file=sys.stderr)
    if not ok:
        sys.exit("output file failed its own header check")

    print(json.dumps({"arch": spec.hf_arch, "gguf_arch": gguf_arch,
                      "tensors": len(tensors), "bytes": actual,
                      "scaling": applied, "ok": ok}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
