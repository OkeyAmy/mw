#!/usr/bin/env python3
"""
mod_wei microscope — activation tracing + live steering for Qwen2 on CPU.

Core idea: every generated token is a fresh forward pass. A hook that edits the
residual stream before that token's logits are computed is therefore *live*
steering. This file implements the whole loop with no magic.

Usage:
  python microscope.py trace   --prompt "def add(a,b):"
  python microscope.py steer   --prompt "def add(a,b):" --layer 14 --coef 2.0
  python microscope.py lens    --prompt "The capital of France is" --topk 5
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DTYPE = torch.float32  # i7-8665U has no AVX-512/AMX/bf16: fp32 is the only fast dtype

# 4 physical cores. SMT measurably hurts (12.2 GB/s at 4 threads vs 11.4 at 8).
torch.set_num_threads(4)


# ----------------------------------------------------------------------------- trace
@dataclass
class Trace:
    step: int
    token_id: int
    token_str: str
    logits_topk: list[tuple[int, str, float]]
    layers: list[dict] = field(default_factory=list)


class Microscope:
    """Wraps a HF causal LM with residual-stream capture and pre-hook steering."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            attn_implementation="eager",  # needed for attention weights
        ).to(device).eval()
        self.device = device
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self.load_s = time.time() - t0

        self.capturing = False
        self._resid: dict[int, torch.Tensor] = {}

        # steering state, mutated between tokens = mid-generation control
        self.steer_layer: Optional[int] = None
        self.steer_vector: Optional[torch.Tensor] = None
        self.steer_coef: float = 0.0
        self.steer_positions: str = "last"  # last | all

        self._install_hooks()

    # -- hooks ---------------------------------------------------------------
    def _install_hooks(self) -> None:
        layers = self.model.model.layers

        for i, layer in enumerate(layers):
            # capture resid_post (output of layer i)
            layer.register_forward_hook(self._make_capture(i))

        # steering: forward_pre_hook on layer L edits its *input* (resid_pre L)
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._make_steer(i))

    def _make_capture(self, i: int):
        def hook(module, args, output):
            if not self.capturing:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            self._resid[i] = hs[:, -1, :].detach().float()
            return None
        return hook

    def _make_steer(self, i: int):
        def hook(module, args):
            if self.steer_layer != i or self.steer_coef == 0.0 or self.steer_vector is None:
                return None
            h = args[0]
            v = self.steer_vector.to(device=h.device, dtype=h.dtype)
            if self.steer_positions == "all":
                h = h + self.steer_coef * v
            else:  # last position only
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + self.steer_coef * v
            return (h,) + tuple(args[1:])
        return hook

    # -- logit lens ----------------------------------------------------------
    @torch.no_grad()
    def logit_lens(self, h: torch.Tensor, topk: int = 5):
        """Project residual vector(s) through final norm + unembedding.

        h: [..., d_model] -> list of top-k [(id, str, prob)] per vector.
        """
        logits = self.model.lm_head(self.model.model.norm(h))  # [..., vocab]
        probs = logits.softmax(-1)
        p, idx = probs.topk(topk, dim=-1)
        return [
            [(int(j), self.tok.decode([int(j)]), float(q)) for q, j in zip(pp, ii)]
            for pp, ii in zip(p.reshape(-1, p.shape[-1]), idx.reshape(-1, idx.shape[-1]))
        ]

    def _topk_from_logits(self, logits: torch.Tensor, topk: int):
        """Top-k over REAL logits (not a lens projection)."""
        p, idx = logits.float().softmax(-1).topk(topk, dim=-1)
        return [
            [(int(j), self.tok.decode([int(j)]), float(q)) for q, j in zip(pp, ii)]
            for pp, ii in zip(p.reshape(-1, p.shape[-1]), idx.reshape(-1, idx.shape[-1]))
        ]

    @torch.no_grad()
    def logit_lens_layers(self, resid: dict[int, torch.Tensor], topk: int = 5):
        """ONE GEMM for all layers.

        The unembedding is 151936x1536 (933 MB fp32). Calling it 28 times
        re-reads that matrix 28x = ~2.5 s/token. Stacking the residuals into
        one [L, d] @ [d, vocab] matmul reads it once = ~0.49 s/token (measured).
        """
        layers = sorted(resid)
        hs = torch.stack([resid[l].reshape(-1) for l in layers])  # [L, d]
        all_lens = self.logit_lens(hs, topk)                      # [L][topk]
        return {l: all_lens[i] for i, l in enumerate(layers)}

    # -- single forward trace ------------------------------------------------
    @torch.no_grad()
    def trace(self, prompt: str, topk: int = 5) -> dict:
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        self.capturing = True
        self._resid.clear()
        out = self.model(input_ids=ids, output_attentions=True)
        self.capturing = False

        lens = self.logit_lens_layers(self._resid, topk)

        # attention entropy per layer, averaged over heads, last query position
        attn_entropy = {}
        for i, a in enumerate(out.attentions):  # a: [B, heads, T, T]
            p = a[:, :, -1, :].clamp_min(1e-9)
            ent = -(p * p.log()).sum(-1).mean().item()
            attn_entropy[i] = round(ent, 3)

        final = self._topk_from_logits(out.logits[:, -1, :], topk)[0]
        return {
            "prompt": prompt,
            "n_tokens": int(ids.shape[1]),
            "layers": {str(k): v for k, v in lens.items()},
            "attn_entropy": attn_entropy,
            "next_token_topk": final,
            "resid_norms": {str(k): round(float(v.norm()), 3) for k, v in self._resid.items()},
        }

    # -- steering vector extraction -----------------------------------------
    @torch.no_grad()
    def diff_in_means(self, pairs: list[tuple[str, str]], layer: int) -> torch.Tensor:
        """pairs = [(positive_text, negative_text), ...]; returns unit vector."""
        self.capturing = True
        pos, neg = [], []
        for a, b in pairs:
            for text, bucket in ((a, pos), (b, neg)):
                ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
                self._resid.clear()
                self.model(input_ids=ids)
                bucket.append(self._resid[layer].mean(0))
        self.capturing = False
        v = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
        return v / v.norm().clamp_min(1e-8)

    def set_steer(self, layer: Optional[int], vector: Optional[torch.Tensor], coef: float,
                  positions: str = "last") -> None:
        self.steer_layer, self.steer_vector, self.steer_coef = layer, vector, coef
        self.steer_positions = positions

    # -- live generation -----------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.0,
        on_token: Optional[Callable[[Trace], None]] = None,
        schedule: Optional[Callable[[int, Trace], None]] = None,
        force_token: Optional[int] = None,
        topk: int = 5,
    ) -> list[Trace]:
        """
        Manual loop: full control between tokens. `schedule(step, trace)` is called
        BEFORE sampling each token, so it can mutate steering coefficients live.
        """
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        past = None
        traces: list[Trace] = []
        self.capturing = True

        for step in range(max_new_tokens):
            self._resid.clear()
            out = self.model(
                input_ids=ids if past is None else ids[:, -1:],
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,  # only need the last position: big CPU saving
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

            if force_token is not None:
                logits[:, force_token] += 100.0

            if temperature <= 0:
                nxt = int(logits.argmax(-1))
            else:
                probs = (logits / temperature).softmax(-1)
                nxt = int(torch.multinomial(probs, 1))

            topk_vals = self._topk_from_logits(logits, topk)[0]
            lens_by_layer = self.logit_lens_layers(self._resid, 3)
            tr = Trace(
                step=step,
                token_id=nxt,
                token_str=self.tok.decode([nxt]),
                logits_topk=topk_vals,
                layers=[
                    {"layer": l, "resid_norm": round(float(h.norm()), 3),
                     "lens": lens_by_layer[l]}
                    for l, h in sorted(self._resid.items())
                ],
            )
            traces.append(tr)

            if schedule is not None:
                schedule(step, tr)  # <-- mid-generation tuning happens here
            if on_token is not None:
                on_token(tr)

            if nxt == self.tok.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device)], dim=1)

        self.capturing = False
        return traces


# ----------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trace", "lens", "steer"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="def add(a, b):\n")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--coef", type=float, default=2.0)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    m = Microscope(args.model)
    print(f"loaded {args.model} in {m.load_s:.1f}s | layers={m.n_layers} d={m.d_model}",
          file=__import__("sys").stderr)

    if args.cmd == "lens":
        r = m.trace(args.prompt, args.topk)
        print(json.dumps(r["next_token_topk"], indent=2, ensure_ascii=False))
        for l in sorted(map(int, r["layers"])):
            if l % 4 == 0 or l == m.n_layers - 1:
                print(f"L{l:>2} norm={r['resid_norms'][str(l)]:>7} "
                      f"-> {r['layers'][str(l)][0][1]!r}")
        return

    if args.cmd == "trace":
        r = m.trace(args.prompt, args.topk)
        print(json.dumps(r, indent=2, ensure_ascii=False) if args.json
              else json.dumps(r["next_token_topk"], indent=2, ensure_ascii=False))
        return

    # steer: baseline vs steered, same prompt, greedy
    print("=== baseline ===", file=__import__("sys").stderr)
    base = m.generate(args.prompt, max_new_tokens=args.max_new, topk=3)
    base_text = "".join(t.token_str for t in base)
    print(base_text)

    # toy vector: the model's own "more comments" direction is not known yet;
    # use the mean residual of a contrast pair as a demonstration of the mechanism.
    vec = m.diff_in_means([
        ("# explain the code\n", "def f():\n"),
        ("# write documentation\n", "x = 1\n"),
    ], args.layer)
    m.set_steer(args.layer, vec, args.coef)
    print(f"=== steered L{args.layer} coef={args.coef} ===", file=__import__("sys").stderr)
    st = m.generate(args.prompt, max_new_tokens=args.max_new, topk=3)
    print("".join(t.token_str for t in st))


if __name__ == "__main__":
    main()
