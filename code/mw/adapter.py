"""
mw.adapter — one interface, many models.

Audited fixes (G1, 2026-11-11):
  D1  import guard
  D6  capture BOTH resid_pre and resid_post; steering and diff_in_means share a site
  D10 self.cfg IS self.model.config (no divergent config objects)
  D11 logits_scaling: explicit None handling, never silently 1.0
  D12 lm_head resolution raises instead of masking None; verifies tying
  D13 layer index validated
  D14 hooks are idempotent
  D15 trace() can return the full per-token residual stream
  D16 every trace carries a provenance block
  D17 capability report can actually report a limitation
  D25 no import-time thread reconfiguration
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import guard
from .registry import HF_SUPPORTED

# Default contrast pairs: identical suffix, comment inserted. This removes the
# "which token ends the string" confound that made the first vector meaningless.
_SUFFIX = "def f(x):\n    return x\n"
DEFAULT_POS = [
    "# add the two numbers\n" + _SUFFIX,
    "# read the file and parse it\n" + _SUFFIX,
    "# raise if the input is empty\n" + _SUFFIX,
    "# loop over every item\n" + _SUFFIX,
    "# this class stores credentials\n" + _SUFFIX,
    "# return the cached value\n" + _SUFFIX,
    "# validate the request first\n" + _SUFFIX,
    "# write the output to disk\n" + _SUFFIX,
]
DEFAULT_NEG = [_SUFFIX for _ in DEFAULT_POS]

VIS_FULL = "FULL"
VIS_PARTIAL = "PARTIAL"
VIS_BLACKBOX = "BLACKBOX"

SITES = ("pre", "post")


class CapabilityError(RuntimeError):
    pass


@dataclass
class Capabilities:
    model: str
    arch: str
    visibility: str
    hidden_states: bool = True
    logit_lens: bool = True
    steering: bool = True
    attention: bool = True
    logprobs: bool = True
    d_model: int = 0
    n_layers: int = 0
    vocab: int = 0
    scaling: dict = field(default_factory=dict)

    def report(self) -> str:
        return (
            f"{self.model}\n"
            f"  arch        : {self.arch}\n"
            f"  visibility  : {self.visibility}\n"
            f"  hidden states / logit lens / steering / attention : "
            f"{self.hidden_states} / {self.logit_lens} / {self.steering} / {self.attention}\n"
            f"  shape       : {self.n_layers} layers, d={self.d_model}, vocab={self.vocab}\n"
            f"  scaling     : {self.scaling or 'none'}"
        )


@dataclass
class ModelAdapter:
    path: str
    model: object = None
    tok: object = None
    cfg: object = None
    _pre: dict = field(default_factory=dict)
    _post: dict = field(default_factory=dict)
    _mlp: dict = field(default_factory=dict)   # top-N MLP neuron indices per layer
    neuron_k: int = 32                         # captured; the wire payload sends 3
    # layer -> {neuron_index: value}. Written into the MLP intermediate before
    # the down projection, so this is direct neuron-level intervention.
    _neuron_steer: dict = field(default_factory=dict)
    # shared 2-D PCA basis for the trajectory view
    _pca_basis: torch.Tensor | None = None
    _pca_mu: torch.Tensor | None = None
    _pca_sd: torch.Tensor | None = None
    _pca_buf: list = field(default_factory=list)
    _pca_n: int = 0
    _capturing: bool = False
    _capture_full: bool = False
    _hooks: list = field(default_factory=list)
    _steer: dict = field(default_factory=lambda: {
        "layer": None, "vec": None, "coef": 0.0, "positions": "last", "site": "pre"})
    dtype_actual: str = ""

    # ---- lifecycle ---------------------------------------------------------
    def load(self, dtype: torch.dtype | None = None, n_threads: int | None = None,
             attn_implementation: str = "sdpa") -> ModelAdapter:
        name = {torch.float32: "float32", torch.float16: "float16",
                torch.bfloat16: "bfloat16"}.get(dtype) if dtype else None
        chosen = guard.require(self.path, name)
        if dtype is None:
            dtype = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}[chosen]
            warnings.warn(
                f"dtype not specified; guard selected {chosen} from available memory. "
                f"Pass dtype= explicitly for reproducible numerics.", stacklevel=2)
        if n_threads:
            torch.set_num_threads(n_threads)

        # BF16 -> FP16 silently overflows (65504) and yields NaN logits.
        if guard.unsafe_downcast(self.path, name or chosen):
            raise ValueError(
                f"{self.path} is BF16; casting to float16 overflows and produces NaN. "
                f"Use dtype=torch.float32 or torch.bfloat16.")

        self.tok = AutoTokenizer.from_pretrained(self.path)
        # SDPA is measurably faster than eager on CPU (Granite 2B: 8.1 s vs
        # 8.5 s prefill, 1.35 vs 1.76 s/token). `trace()` needs attention
        # weights, so it degrades gracefully when SDPA is active.
        self.attn_impl = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, dtype=dtype, attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).eval()
        self.cfg = self.model.config          # D10: one config object, not two
        self.dtype_actual = str(next(self.model.parameters()).dtype)
        self._install_hooks()
        return self

    @property
    def layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise CapabilityError(f"cannot locate decoder layers on {type(self.model).__name__}")

    @property
    def final_norm(self):
        m = getattr(self.model, "model", None)
        if m is not None and hasattr(m, "norm"):
            return m.norm
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        raise CapabilityError("cannot locate final norm")

    @property
    def lm_head(self):
        oe = self.model.get_output_embeddings()
        if oe is None:
            raise CapabilityError(
                f"{type(self.model).__name__} has no output embedding; "
                f"refusing to guess between lm_head and a tied head")
        return oe

    @property
    def logits_scaling(self) -> float:
        v = getattr(self.model.config, "logits_scaling", None)
        if v is None:
            # Arch declares scaling but config lost it -> that is a conversion bug,
            # not a licence to assume 1.0.
            if getattr(self.model.config, "model_type", "") in ("granite",):
                raise CapabilityError(
                    "granite config has no logits_scaling; conversion dropped it")
            return 1.0
        return float(v)

    # ---- hooks -------------------------------------------------------------
    def _install_hooks(self) -> None:
        if self._hooks:                      # D14: idempotent
            return
        for i, layer in enumerate(self.layers):
            self._hooks.append(layer.register_forward_pre_hook(self._mk_pre(i)))
            self._hooks.append(layer.register_forward_hook(self._mk_post(i)))
            mlp_out = self._mlp_out_module(layer)
            if mlp_out is not None:
                self._hooks.append(mlp_out.register_forward_pre_hook(self._mk_mlp(i)))

    @staticmethod
    def _mlp_out_module(layer):
        """The projection that consumes the MLP intermediate. Names differ by arch."""
        mlp = getattr(layer, "mlp", None)
        for name in ("down_proj", "c_proj", "dense_4h_to_h"):
            mod = getattr(mlp, name, None)
            if mod is not None:
                return mod
        return None

    def _mk_mlp(self, i: int):
        def hook(mod, args):
            if not args:
                return None
            x = args[0]
            edits = self._neuron_steer.get(i)
            if edits:
                x = x.clone()
                for j, val in edits.items():
                    if 0 <= j < x.shape[-1]:
                        x[:, -1, j] = val
                if self._capturing:
                    self._record_neurons(i, x)
                return (x,) + tuple(args[1:])
            if self._capturing:
                self._record_neurons(i, x)
            return None
        return hook

    def _record_neurons(self, i: int, x) -> None:
        """Store top-N neurons (N = neuron_k) so the UI can drill into a layer."""
        y = x[:, -1, :].detach().float()[0]
        k = min(self.neuron_k, y.shape[-1])
        v, idx = y.topk(k)
        self._mlp[i] = [(int(j), round(float(q), 3)) for q, j in zip(v, idx)]

    def set_neuron_steer(self, layer: int, neuron: int, value: float) -> None:
        """Force one MLP neuron to a value at the last position, from now on."""
        if not (0 <= layer < len(self.layers)):
            raise ValueError(f"layer {layer} out of range")
        self._neuron_steer.setdefault(layer, {})[int(neuron)] = float(value)

    def clear_neuron_steer(self, layer: int | None = None) -> None:
        if layer is None:
            self._neuron_steer.clear()
        else:
            self._neuron_steer.pop(int(layer), None)

    def top_neurons(self, layer: int, k: int = 32) -> list[tuple[int, float]]:
        """Last captured top-N neurons for a layer (drill-down view)."""
        return list(self._mlp.get(layer, []))[:k]

    def _mk_pre(self, i: int):
        def hook(mod, args):
            h = args[0]
            s = self._steer
            if s["site"] == "pre" and s["layer"] == i and s["coef"] != 0.0 and s["vec"] is not None:
                v = s["vec"].to(device=h.device, dtype=h.dtype)
                if s["positions"] == "all":
                    h = h + s["coef"] * v
                else:
                    h = h.clone()
                    h[:, -1, :] = h[:, -1, :] + s["coef"] * v
            if self._capturing:
                self._pre[i] = (h[:, :, :] if self._capture_full else h[:, -1, :]).detach().float()
            return (h,) + tuple(args[1:])
        return hook

    def _mk_post(self, i: int):
        def hook(mod, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            s = self._steer
            if s["site"] == "post" and s["layer"] == i and s["coef"] != 0.0 and s["vec"] is not None:
                v = s["vec"].to(device=hs.device, dtype=hs.dtype)
                if s["positions"] == "all":
                    hs = hs + s["coef"] * v
                else:
                    hs = hs.clone()
                    hs[:, -1, :] = hs[:, -1, :] + s["coef"] * v
            if self._capturing:
                self._post[i] = (hs[:, :, :] if self._capture_full else hs[:, -1, :]).detach().float()
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return hook

    # ---- lens --------------------------------------------------------------
    @torch.no_grad()
    def logit_lens(self, h: torch.Tensor, topk: int = 5):
        # Captures are stored in float32 for analysis, but the head/norm may be
        # fp16/bf16. Cast at the boundary or the matmul raises a dtype error
        # (found by running the fp16 Granite path).
        w = self.lm_head.weight
        if h.dtype != w.dtype:
            h = h.to(dtype=w.dtype)
        logits = self.lm_head(self.final_norm(h)) / self.logits_scaling
        logits = logits.float()
        p, idx = logits.softmax(-1).topk(topk, dim=-1)
        p = p.reshape(-1, p.shape[-1]); idx = idx.reshape(-1, idx.shape[-1])
        return [[(int(j), self.tok.decode([int(j)]), float(q)) for q, j in zip(pp, ii)]
                for pp, ii in zip(p, idx)]

    @torch.no_grad()
    def logit_lens_layers(self, site: str = "post", topk: int = 5,
                          only: list[int] | None = None):
        """One GEMM for the requested layers. `only` lets the live server lens a
        subset (the unembedding read is the per-token bottleneck)."""
        store = self._post if site == "post" else self._pre
        keys = sorted(only) if only is not None else sorted(store)
        hs = torch.stack([store[k].reshape(-1) if store[k].ndim == 2
                          else store[k][:, -1, :].reshape(-1) for k in keys])
        out = self.logit_lens(hs, topk)
        return {k: out[n] for n, k in enumerate(keys)}

    # ---- trace -------------------------------------------------------------
    @torch.no_grad()
    def trace(self, prompt: str, topk: int = 5, full: bool = False) -> dict:
        ids = self.tok(prompt, return_tensors="pt").input_ids
        self._capturing = True
        self._capture_full = full
        self._pre.clear(); self._post.clear()
        out = self.model(input_ids=ids,
                         output_attentions=(self.attn_impl == "eager"))
        self._capturing = False
        lens = self.logit_lens_layers("post", topk)
        return {
            "prompt": prompt,
            "model": {                        # D16: provenance
                "path": self.path,
                "arch": type(self.model).__name__,
                "dtype": self.dtype_actual,
                "logits_scaling": self.logits_scaling,
                "site": "resid_post",
                "positions": "all" if full else "last",
            },
            "n_tokens": int(ids.shape[1]),
            "layers": {str(k): v for k, v in lens.items()},
            "resid_norms": {str(k): round(float(v.norm()), 3) for k, v in self._post.items()},
            "next_topk": self._topk(out.logits[:, -1, :], topk)[0],
        }

    def _topk(self, logits, topk):
        p, idx = logits.float().softmax(-1).topk(topk, dim=-1)
        return [[(int(j), self.tok.decode([int(j)]), float(q)) for q, j in zip(pp, ii)]
                for pp, ii in zip(p, idx)]

    # ---- steering ----------------------------------------------------------
    def set_steer(self, layer: int | None, vector: torch.Tensor | None,
                  coef: float, positions: str = "last", site: str = "pre") -> None:
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}")
        if layer is not None and not (0 <= layer < len(self.layers)):
            raise ValueError(f"layer {layer} out of range 0..{len(self.layers)-1}")
        self._steer.update(layer=layer, vec=vector, coef=coef, positions=positions, site=site)

    @torch.no_grad()
    def diff_in_means(self, pairs, layer: int, site: str = "pre") -> torch.Tensor:
        """D6: default site is 'pre' — the same site set_steer writes to, so
        diff_in_means(L) -> set_steer(L) composes without an off-by-one block."""
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}")
        pos, neg = [], []
        self._capturing = True
        self._capture_full = False
        for a, b in pairs:
            for text, bucket in ((a, pos), (b, neg)):
                ids = self.tok(text, return_tensors="pt").input_ids
                self._pre.clear()
                self.model(input_ids=ids)
                store = self._pre if site == "pre" else self._post
                bucket.append(store[layer].mean(0))
        self._capturing = False
        v = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
        return v / max(float(v.norm()), 1e-8) * float(torch.stack(pos).norm(dim=-1).mean())

    @torch.no_grad()
    def build_steer_vectors(self, pos: list[str] | None = None,
                            neg: list[str] | None = None,
                            site: str = "pre") -> dict[int, torch.Tensor]:
        """Contrast-pair vectors for EVERY layer, built at the steering site.

        Two fixes forced by the H3 audit:
          1. Average over ALL token positions, not the last one. Last-token
             vectors were measuring "which token ends the string" (cos 1.0 with
             an embedding difference), not the contrast.
          2. Default pairs share an identical suffix, so the only difference is
             the inserted comment. That removes the final-token confound by
             construction.
        """
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}")
        pos = pos if pos is not None else list(DEFAULT_POS)
        neg = neg if neg is not None else list(DEFAULT_NEG)
        if len(pos) != len(neg):
            raise ValueError("pos and neg must be the same length")

        acc: dict[int, dict[str, list]] = {l: {"p": [], "n": []}
                                           for l in range(len(self.layers))}
        self._capturing = True
        self._capture_full = True      # keep [T, d] so we can mean over positions
        for text, key in [(t, "p") for t in pos] + [(t, "n") for t in neg]:
            ids = self.tok(text, return_tensors="pt").input_ids
            self._pre.clear()
            self.model(input_ids=ids)
            store = self._pre if site == "pre" else self._post
            for l in range(len(self.layers)):
                h = store.get(l)
                if h is None:
                    continue
                # mean over token positions -> [d]
                acc[l][key].append(h[0].mean(0) if h.ndim == 3 else h.mean(0))
        self._capturing = False
        self._capture_full = False

        out: dict[int, torch.Tensor] = {}
        for l, d in acc.items():
            if not d["p"] or not d["n"]:
                continue
            p = torch.stack(d["p"])
            n = torch.stack(d["n"])
            v = p.mean(0) - n.mean(0)
            norm = float(p.norm(dim=-1).mean())
            out[l] = v / max(float(v.norm()), 1e-8) * norm
        return out

    # ---- residual trajectory projection ------------------------------------
    def _ensure_pca(self, X: torch.Tensor) -> None:
        """Maintain a SHARED 2-D basis over recent trajectories (J1/J2 spec).

        X: [n_layers, d] residuals for one token. A shared basis is required or
        ribbons from different tokens are not comparable. Re-fit every 8 tokens
        over a 32-token rolling buffer, mean-centred and standardised first —
        raw distances are dominated by 1-3 dims and activations up to 1e5x larger.
        """
        self._pca_buf.append(X)
        self._pca_buf = self._pca_buf[-32:]
        need = self._pca_basis is None or (self._pca_n % 8 == 0)
        if need and len(self._pca_buf) >= 1:
            Y = torch.cat(self._pca_buf, 0).float()
            mu = Y.mean(0, keepdim=True)
            sd = Y.std(0, keepdim=True).clamp_min(1e-6)
            Z = (Y - mu) / sd
            q = min(2, Z.shape[0] - 1) if Z.shape[0] > 1 else 1
            basis = torch.zeros(X.shape[1], 2)
            try:
                _, _, V = torch.pca_lowrank(Z, q=max(q, 1))
                k = min(2, V.shape[1])
                basis[:, :k] = V[:, :k]
            except RuntimeError:
                # degenerate input (e.g. all-zero variance); keep the zero basis
                pass
            self._pca_basis = basis
            self._pca_mu, self._pca_sd = mu, sd
        self._pca_n += 1

    @torch.no_grad()
    def project_trajectory(self, layers: list[int]) -> tuple[list[list[float]], float]:
        """Return (xy per layer, variance explained) for the current residuals."""
        X = torch.stack([self._post[l].reshape(-1) for l in layers]).float()
        self._ensure_pca(X)
        Z = (X - self._pca_mu) / self._pca_sd
        P = Z @ self._pca_basis
        # variance explained by the 2 retained components
        var = float(P.var(0).sum() / Z.var(0).sum().clamp_min(1e-9))
        return [[round(float(a), 4), round(float(b), 4)] for a, b in P], round(var, 4)

    # ---- live streaming ----------------------------------------------------
    @torch.no_grad()
    def stream(self, prompt: str, max_new: int = 48,
               lens_layers: list[int] | None = None):
        """Yield one protocol-shaped trace per generated token.

        `lens_layers` restricts the logit-lens computation; probabilities per
        layer are the dominant cost because the unembedding is ~933 MB.
        """
        ids = self.tok(prompt, return_tensors="pt").input_ids
        past = None
        t0 = time.time()
        want = lens_layers if lens_layers is not None else list(range(len(self.layers)))
        for step in range(max_new):
            self._capturing = True
            self._capture_full = False
            out = self.model(
                input_ids=ids if past is None else ids[:, -1:],
                past_key_values=past, use_cache=True, logits_to_keep=1,
            )
            self._capturing = False
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            nxt = int(logits.argmax(-1))
            vals, idxs = logits.softmax(-1).topk(5, -1)
            lens_map = self.logit_lens_layers("post", 3, only=want)
            ls = sorted(self._post)
            xy, var_expl = self.project_trajectory(ls)
            yield {
                "step": step,
                "token": self.tok.decode([nxt]),
                "token_id": nxt,
                "ms": round((time.time() - t0) * 1000),
                # provenance: the UI cannot label honestly without this
                "steered": bool(self._steer["coef"]) or bool(self._neuron_steer),
                "coef": round(float(self._steer["coef"]), 4),
                "steer_layer": self._steer["layer"],
                "topk": [[self.tok.decode([int(i)]).replace("\n", "\\n"),
                          round(float(p), 4)]
                         for p, i in zip(vals[0].tolist(), idxs[0].tolist())],
                # real coordinates for the 3-D view: shared PCA of the
                # standardised residual trajectory (J1/J2 spec)
                "pca": {"var_explained": var_expl, "n_layers": len(ls)},
                "layers": [
                    {"l": l,
                     "norm": round(float(self._post[l].norm()), 2),
                     "xy": xy[n],
                     "lens": lens_map.get(l, []),
                     # wire payload stays small; the drill-down endpoint serves
                     # the full top-32 stored in self._mlp
                     "neurons": self._mlp.get(l, [])[:3]}
                    for n, l in enumerate(ls)
                ],
            }
            if nxt == self.tok.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)

    # ---- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 24, temperature: float = 0.0,
                 schedule: Callable[[int, dict], None] | None = None,
                 force_token: int | None = None) -> list[dict]:
        ids = self.tok(prompt, return_tensors="pt").input_ids
        past, out_tokens = None, []
        for step in range(max_new_tokens):
            out = self.model(
                input_ids=ids if past is None else ids[:, -1:],
                past_key_values=past, use_cache=True, logits_to_keep=1,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if force_token is not None:
                logits[:, force_token] += 100.0
            nxt = int(logits.argmax(-1)) if temperature <= 0 else int(
                torch.multinomial((logits / temperature).softmax(-1), 1))
            rec = {"step": step, "token_id": nxt, "token_str": self.tok.decode([nxt]),
                   "topk": self._topk(logits, 5)[0]}
            out_tokens.append(rec)
            if schedule is not None:
                schedule(step, rec)
            if nxt == self.tok.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        return out_tokens

    # ---- capability report -------------------------------------------------
    def capabilities(self) -> Capabilities:
        arch = type(self.model).__name__
        supported = arch in HF_SUPPORTED
        scaling = {k: getattr(self.cfg, k) for k in
                   ("embedding_multiplier", "residual_multiplier",
                    "attention_multiplier", "logits_scaling")
                   if getattr(self.cfg, k, None) is not None}
        return Capabilities(
            model=self.path, arch=arch,
            visibility=VIS_FULL if supported else VIS_PARTIAL,
            hidden_states=True, logit_lens=True, steering=supported,
            attention=True, logprobs=True,
            d_model=int(getattr(self.cfg, "hidden_size", 0)),
            n_layers=len(self.layers),
            vocab=int(getattr(self.cfg, "vocab_size", 0)),
            scaling=scaling,
        )
