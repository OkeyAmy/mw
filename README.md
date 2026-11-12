# mw

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
negatives. The live list is tracked locally, not published.

**Measured:**
- APIs do not return hidden states. Activation-level tracing requires weights.
- Superposition and polysemanticity are real; "neuron 4021 = recursion" is not a
  thing in the weights.

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

The first implementation appended a **whole rendered template**, which re-inserted
the system preamble and reopened the assistant turn without closing the previous
one — 67 tokens per injection. It now appends only the template diff (18 tokens):
`<|end_of_text|>` + user turn + assistant prompt. Live check:
injecting "What is the capital of Japan?" mid-stream yields
`The capital of Japan is Tokyo, Japan.`

**Steering units are calibrated.** `coef = 1` means *one residual norm* at that
layer. The original vectors were normalised on raw text but applied to
chat-formatted prompts, so a unit was 2.6 residual norms at L8 and 59 at L9 —
most slider positions were 5–60× too strong. `residual_scale()` measures
`||h_last||` per layer on chat-rendered text and `calibrate_vectors()` rescales
direction-preserving. Live: the same prompt at L8 gives
`coef 0 → "It seems like you"`, `+2 → "I'm sorry for"`, `−2 → "\n\nI'm"`.

Injection also had a cache bug worth remembering: after forwarding the injected
tokens, re-feeding the last one duplicates a token already in the KV cache and
corrupts every later step. The loop now keeps the injection forward's logits
instead of re-feeding.

## Certificate: what did quantisation change?

```bash
PYTHONPATH=code python -m mw.certify \
  --reference .hf/bf16-hf --candidate .hf/q4km-hf \
  --builtin 41 --null-control --out proof/cert_qwen_q4
```

It compares **next-token decisions**, not benchmark averages, and prints the
null control (same model, other dtype) so the result has a floor.

First run, Qwen2.5-Coder-1.5B bf16 vs Q4_K_M, 41 unique prompts:

| metric | value |
|---|---|
| decisions flipped | **2 / 41 (4.9%)** |
| 95% CI | [1.4%, 16.1%] |
| confident flips (ref margin > 1 nat) | **0** |
| mean KL / max KL | 0.042 / 0.218 |
| null control (fp32 vs bf16) | **0.0%** [0%, 8.6%] |

The two flips:

| prompt | bf16 | Q4_K_M | ref margin |
|---|---|---|---|
| `Two plus two equals` | ` four` | ` three` | 0.19 |
| `The author of Hamlet was` | ` a` | ` born` | 0.21 |

**What this does not say.** It is *not certifiable* at a 5% gate: with 41 unique
prompts the 95% upper bound is 16.1%. And because the null control's upper bound
(8.6%) is wider than the observed flip rate (4.9%), we **cannot claim
quantisation caused those flips** rather than precision noise. Both flipped on
near-ties (margin < 0.25 nat).

The null control here is fp32-vs-bf16 because this checkpoint is bf16 and a
bf16→fp16 cast overflows; the fp16 storage floor C1 measured (3.4–6.7%) cannot
be reproduced on this model. That limitation is part of the certificate.

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
| C4 steering specificity | **FAIL** — re-run with the corrected estimator, T1/T4 still fail |
| C5 mid-generation | PASS — divergence begins after the switch |
| C6 force vs steer | **PROVISIONAL** — force 1.00 works; the semantic arm used the broken vector |
| C7 neuron atlas | PASS at threshold 1e-6 (coverage, not meaning) |
| C8 sanity gate | PASS — 10/10 |

The instrument is trustworthy. The *control* claim is **not yet established** —
which is different from "disproved".

**C4, re-run with the corrected estimator (2026-11-11).** The first run used a
last-token vector that was numerically identical to an embedding difference
(cosine 1.00000) — it was steering "ends with this token", not comment mode.
The estimator is now position-averaged over pairs with an identical suffix, and
the artifact is gone:

```
L0  norm=  0.3   max cos to any token embedding = 0.4669
L8  norm=335.1   max cos = 0.0925
L16 norm=338.5   max cos = 0.0920
```

**The scale mattered more than the direction.** The first re-run used raw
vectors of norm ~335 against a residual norm of ~5 — `coef=2` was a ~130×
perturbation. S1 found the unit bug: vectors were normalised on raw text but
applied to chat-formatted prompts, so one unit was 2.6 residual norms at L8 and
**59 at L9**. After `residual_scale()`/`calibrate_vectors()`, `coef=1` means one
residual norm, which is the regime the UI actually operates in.

C4 re-run at that calibrated scale (L8, coef +1, 20 random controls):

| test | result | numbers |
|---|---|---|
| T1 selectivity | **FAIL** | real target **−0.38** vs random p95 **+1.12** |
| T2 collateral | **FAIL** | real control 2.87 vs random median 2.28 |
| T4 held-out sign | **FAIL** | real **0.10** — only 1 of 10 prompts moved the right way |

At a realistic perturbation the vector does not just fail to beat random: it
moves the target **slightly the wrong way** and is almost entirely inconsistent
across held-out prompts. At 130× it produced +4.4 — that was **saturation**, not
semantics. Every large perturbation moves logits; that is not steering.

**Honest status: the steering slider changes output, but it is a perturbation
knob, not semantic control.** Any coefficient changes the text; the diff-in-means
direction does not carry the intended semantics at a usable scale. Measurement
and forcing (C6) are unaffected.

## Contributing

The most useful contribution is a model we cannot load. Open an issue with the
GGUF arch string and a tensor listing; we add an `ArchSpec` and a slow test.

Every PR must keep `make fast` green and must not weaken a control to make a
test pass. If a test needs its criterion changed, the change goes in
`PROOF_CHAIN.md` with the reason.

## License

MIT. Model weights keep their own licences.
