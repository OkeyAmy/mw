# mod_wei

**A local-first microscope for open-weight language models.** Trace what a model
does inside, prove which activations are causal, and intervene — with controls
that are allowed to fail.

Every claim in this repository is paired with a test that can falsify it. Where a
test failed, the failure is recorded rather than deleted.

```
prompt ──► transformer ──► next token
             ▲  ▲  ▲
             │  │  └─ steering: add coef·v at layer L, live, per token
             │  └──── logit lens: what does layer L want to say?
             └─────── trace: every residual, every attention pattern
```

## What it does

- **Trace** every layer's residual stream, per token.
- **Logit lens** — project any layer through the final norm + unembedding.
- **Causal patching** — prove an activation matters, with layer/random controls.
- **Steering** — add a direction to the residual stream mid-generation.
- **Capability report** — states plainly what a given model supports.

## Known boundaries — and what is still open

These are stated as **measured facts** or **UNVERIFIED**, never as settled
negatives. See `STATE_OF_PROJECT.md` for the live list.

**Measured:**
- APIs do not return hidden states. Activation-level tracing requires weights.
- Superposition and polysemanticity are real; "neuron 4021 = recursion" is not a
  thing in the weights. (Evidence: `agents/A6_limits.md`.)

**Open — do not treat as settled:**
- **Can internals be recovered from an API?** Partly, yes — Carlini et al. 2024
  (arXiv:2403.06634) recovered a production model's embedding projection layer
  from logits alone. How far this generalises is unknown; agent F2 is on it and
  has not reported.
- **Does a distilled surrogate transfer mechanism?** F1 says behaviourally yes,
  mechanistically mostly no — but that is one setting and its decisive test is
  unrun.
- **Does steering work?** Our current vector fails our own specificity test on
  bf16 (C4). That is a fact about *this vector* and *this test*, not a proof that
  steering is impossible. The criterion changed twice during development and is
  under audit (G1, pending).

> Until G1 returns, treat C4's failure as **provisional**, and treat every
> "cannot" above as **UNVERIFIED**.

## Supported architectures

| architecture | GGUF → HF | trace/lens | notes |
|---|---|---|---|
| Qwen2 / Qwen2.5 | ✅ | ✅ verified `0.000e+00` | q/k/v biases handled |
| Llama / Mistral | ✅ | ✅ | Mistral uses the `llama` GGUF arch string |
| Granite 3.x | ✅ | ✅ verified `9.5e-06` | **four scaling factors**; ignoring them is 1.4e+02 wrong |
| Gemma 1/2/3 | partial | ✅ | per-version check required |
| Phi-3 | ❌ refused | — | fused QKV tensors; conversion not implemented |

Adding a model is a **data change, not a code change**: add an `ArchSpec` to
`code/mw/registry.py` and a slow test. Unknown architectures are **refused**,
never silently mapped to a default.

## Live app

Two processes, one contract (`app/PROTOCOL.md`):

```bash
make backend    # FastAPI + model on :8000
make frontend   # Next.js on :3000
```

Open http://localhost:3000. The brain is the product; chat is an input box.

**Activation steering vs context injection — they are different:**

| | changes | model sees |
|---|---|---|
| activation steering | residual stream / neuron values | same tokens |
| context injection | the token sequence | new text, from that point |

Measured on Granite 3.3 2B (2026-11-11): appending **raw text** mid-stream has
**no effect** (`' Paris.'` with and without an inserted sentence). Formatting the
insertion as a **user turn** through the model's chat template **does** change the
continuation. The server therefore renders injections as turns; use
`--no-chat-template` to disable.

Injection also had a cache bug worth remembering: after forwarding the injected
tokens, re-feeding the last one duplicates a token already in the KV cache and
corrupts every later step. The loop now keeps the injection forward's logits
instead of re-feeding.

## Memory safety (read this)

Loading a model can freeze a laptop if it does not fit. This package refuses
unsafe loads instead of hanging.

- `load()` calls `mw.guard.require(path)` first and raises `MemoryError` with
  advice rather than starting a doomed allocation.
- **zram swap** (Linux) is detected as a risk: it is RAM-backed, so memory
  pressure spirals instead of degrading. It is never *required*; macOS and
  Windows use different mechanisms and the guard handles them via `psutil`.
- **Never cast a BF16 checkpoint to FP16.** BF16 shares fp32's exponent range;
  FP16 tops out at 65504, so the cast silently produces `NaN` logits. The
  adapter refuses this and tells you to use `float32` or `bfloat16`.
- `low_cpu_mem_usage=True` is always set, to avoid duplicating the state dict.

Recommended:

```python
a = ModelAdapter("Qwen/Qwen2.5-Coder-1.5B").load(dtype=torch.float32)  # BF16 source
a = ModelAdapter(".hf/granite-hf").load(dtype=torch.float16)           # FP16 artefact
```

## Install

```bash
uv venv --python 3.12 .venv
.venv/bin/python -m pip install -e ".[dev]"
```

CPU-only is the primary target. A 1.5B model runs at ~2–3 tok/s single-stream
fp32, ~49 tok/s batched on 4 physical cores.

## Quickstart

```python
from mw.adapter import ModelAdapter

a = ModelAdapter("Qwen/Qwen2.5-Coder-1.5B").load()
print(a.capabilities().report())

trace = a.trace("def add(a, b):\n")
print(trace["next_topk"][:3])

v = a.diff_in_means([("# add two numbers\n", "def add(a,b):\n")], layer=8)
a.set_steer(8, v, coef=2.0)
print("".join(t["token_str"] for t in a.generate("def add(a, b):\n", max_new_tokens=16)))
```

Convert a GGUF you already have (no re-download):

```bash
PYTHONPATH=code python -m mw.convert \
  --gguf ~/.ollama/models/blobs/sha256-... \
  --out .hf/granite-hf \
  --hf-repo ibm-granite/granite-3.3-2b-instruct
```

## The validation regime

This is the part that matters more than the features.

| file | what it enforces |
|---|---|
| `VALIDATION.md` | hypothesis → test → result → verdict. Citations must be fetched, numbers recomputed. |
| `PROOF_CHAIN.md` | eight links (C1–C8), each with a pass criterion and a negative control. |
| `tests/` | fast tests need no model (CI); slow tests load real weights (release gate). |
| `STATE_OF_PROJECT.md` | what is settled vs contested, including our own errors. |

Rules learned the hard way, now enforced:

- **Monotonicity is not specificity.** Any `α·v` moves logits roughly linearly,
  so a monotone dose-response is produced by random vectors too. (`VALIDATION.md` L1)
- **Never validate steering on a quantised artefact.** The same vector passed on
  Q4_K_M and failed on bf16. (L2)
- **Replacement is not addition.** A "random replacement" control should destroy
  the signal; asserting it does nothing is a category error. (L3)

```bash
make fast    # no model, no network — what CI runs
make slow    # real weights — what a release requires
make prove   # the full C1–C8 chain
```

## Status (honest)

| link | result |
|---|---|
| C1 trace | PASS — `0.000e+00` |
| C2 lens identity | PASS — `0.000e+00` |
| C3 causal patching | PASS — L22–27 ≈ 1.0, L0–20 ≈ 0 |
| C4 steering specificity | **FAIL** on bf16 — provisional, criterion under audit |
| C5 mid-generation | PASS — divergence begins after the switch |
| C6 force vs steer | PASS — force 1.00, semantic 0.00 |
| C7 neuron atlas | PASS at threshold 1e-6 (coverage, not meaning) |
| C8 sanity gate | PASS — 10/10 |

The instrument is trustworthy. The *control* claim is **not yet established** —
which is different from "disproved". C5 shows intervention changes generation;
C4/C6 say our current vector does not do so in a way random vectors cannot.

## Contributing

The most useful contribution is a model we cannot load. Open an issue with the
GGUF arch string and a tensor listing; we add an `ArchSpec` and a slow test.

Every PR must keep `make fast` green and must not weaken a control to make a
test pass. If a test needs its criterion changed, the change goes in
`PROOF_CHAIN.md` with the reason.

## License

MIT. Model weights keep their own licences.
