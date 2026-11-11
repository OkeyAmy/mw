# D1 — Executable proof chain: C1..C6 as a runnable harness

Deliverable: `code/proof_chain.py`. It turns `PROOF_CHAIN.md` from a table of claims into
six tests that emit pass/fail artifacts. Nothing in this file is a finding about the model —
no model was runnable when it was written (see F6). This file is about **the instrument**.

---

## TL;DR (5 bullets)

- **`code/proof_chain.py` implements C1–C6 and writes all six artifacts** (`proof/c1_trace.json`
  … `proof/c6_force.txt`) plus `proof/summary.tsv`. Every link prints exactly one
  `RESULT <id> PASS|FAIL <metric>` line, so the orchestrator can parse without reading prose.
- **It is precondition-gated, not broken.** On this box right now torch is absent and the
  weights are 0.20 of ~3.1 GB. The harness prints a `PRECONDITION MISSING` banner, writes
  `proof/PRECONDITION_MISSING.txt`, emits `RESULT C1 FAIL precondition_missing:<reason>` for
  each requested link and **exits 2** (distinct from exit 1 = a test actually ran and failed).
- **`--selftest` proves the instrument without the weights.** It builds a 3-layer random
  `Qwen2ForCausalLM` (no download, ~2 s) and asserts the seven plumbing properties C1–C6 rely
  on, including the C2 identity. Run it the moment `transformers` lands — it does not need the
  3 GB model and it separates "my hooks are wrong" from "the model does something unexpected".
- **C2 ships a diagnostic ladder.** If the lens identity fails, the artifact reports the max
  diff for four candidate identities (correct / no-norm / double-norm / off-by-one-layer). A
  failure therefore *names its own cause* instead of just being a number above threshold.
- **Building this surfaced two real bugs in `code/microscope.py`** (F1): its `logit_lens` is
  applied to logits (a second projection through norm+lm_head) and its capture hook stores
  only the last position, so it can never satisfy C1's `[T,1536]` requirement. Do not reuse
  `Microscope.trace()` for any proof artifact.

---

## Findings

### F1. `code/microscope.py` cannot produce a valid proof artifact — two independent bugs  [OBS, code read]

Both are in the file we already have, both are silent (no exception, plausible-looking numbers).

1. **Double projection.** `microscope.py:133` does
   `final = self.logit_lens(out.logits[:, -1, :], topk)` where `logit_lens` is
   `self.model.lm_head(self.model.model.norm(h))`. It applies **RMSNorm + lm_head to logits**.
   That is a shape-legal, numerically meaningless operation (norm over the vocabulary axis).
   Any "top-k next token" printed by `trace()` is therefore wrong unless it happens to be
   computed from `out.logits` directly.
2. **Wrong capture site for C1.** `microscope.py:83` stores `hs[:, -1, :]` — last position only,
   shape `[B,1536]`. C1 requires `[T,1536]` for every layer. The hook cannot be fixed by
   reshaping; it has to store the full sequence.

**Why it matters:** C1/C2 are exactly the links that certify the instrument. A prototype that
prints plausible top-k tokens while projecting through the wrong function is the worst failure
mode available to this project — it looks like it works.
**Confidence:** high (both verified by reading the code).
**Cost:** zero; `proof_chain.py` does not import `microscope.py`.

### F2. The C2 identity is algebraic, so 1e-4 is a *loose* threshold, and a near-miss is diagnostic  [OBS, arithmetic]

Qwen2's output is `logits = lm_head(norm(resid_post(27)))` — `model.model.norm` then
`model.lm_head`, both applied to the tensor our layer-27 forward hook returns. The computation is
straight-line: no reduction across the sequence axis, no softmax, no `T`-dependent term. In fp32
the only difference between our recomputation and the model's is the order of the final
matmul, i.e. **the same kernel on the same values → a diff of 0**, or at worst a few ulp of the
linear layer (~1e-6 at logit magnitudes of ~10).

So: `1e-4` is not a tight tolerance, it is a **bug detector**. Three regimes:

| observed max\|diff\| | meaning |
|---|---|
| `< 1e-5` | correct. the lens is exact. |
| `1e-3 … 1e-1` | almost always **dtype**: the model was loaded in bf16/fp16 and our recomputation ran in fp32 (or vice versa). bf16 has ~3 decimal digits; a bf16 matmul vs an fp32 one differs at ~1e-2 on logits of magnitude 10. |
| `> 0.1` | **wrong tensor**: resid_pre instead of resid_post, layer 26 instead of 27, or norm applied twice/not at all. The ladder identifies which. |

**Why it matters:** a C2 failure is not ambiguous. It has exactly four causes, and the harness
tests all four in one pass.
**Confidence:** high.
**Cost:** one forward pass.

### F3. `coef=0` bit-identity is a property of the *code path*, not of the arithmetic  [OBS]

`h + 0.0*v` is numerically the identity for every finite `h` (and `torch.equal` treats `-0.0`
as equal to `0.0`), so the arithmetic path is *probably* fine. But "probably" is not what
`VALIDATION.md` rule 5 asks for: it asks for bit-identity. The safe construction is to make
`coef == 0` **return `None` from the pre-hook** so no new tensor is ever produced and the
downstream op sees the identical buffer.

`proof_chain.py` implements the inert path as the real behaviour and *also* runs the arithmetic
path (`force_arith=True`) as a second assertion, recording both:
`controls.coef0_inert_path_bitidentical` and `controls.coef0_arith_path_bitidentical`.
If the arithmetic path ever stops being bit-identical (a different BLAS blocking decision on a
freshly allocated buffer), the JSON shows it instead of hiding it.

**Why it matters:** this is the cheapest negative control in the whole chain and the one most
likely to be faked by accident (e.g. by comparing `allclose` instead of `equal`).
**Confidence:** high.

### F4. Mid-generation divergence is *exactly* one step, and that is a property of the KV cache  [OBS/INF]

Steps 0–5 run with `coef=0`, which is a no-op (F3), so the residual stream and therefore the
K/V cache are bit-identical to the unsteered run. At step 6 the forward pass sees a 1-token
chunk and the hook edits position `-1` of layer 14's input. Nothing before step 6 can differ,
and step 6 must differ. This makes C5 an **exact** assertion, not a statistical one:
`max|Δlogits| == 0.0` for steps `<6` and `> 0.0` at step 6.

Caveat the harness records separately: the *logits* diverge at step 6, but the greedy **token**
may not flip until a later step. Those are different claims and the artifact reports both
(`first_token_divergence`). Do not report "steering worked at step 6" on the strength of a
non-zero logit delta.

**Confidence:** high on the mechanism; the token-flip step is model-dependent.

### F5. "Patch a different layer, expect ~no effect" is not true for layer 0  [OBS, mechanics]

`PROOF_CHAIN.md` C3 asks for a control patch at a different layer. Taken literally at layer 0
this is a weak control: layer 0's input is the **token embedding**, so copying the clean
embedding into the corrupt run injects the entity itself. It will move the output, sometimes a
lot — that is information flow working, not a broken control.

The harness therefore runs three controls and only two of them are absolute:

- **NC1 self-patch** (patch the corrupt run with its own residual) → must be `torch.equal`. Absolute.
- **NC3 random direction of matched norm** at the best layer → recovery must be ≤0.20. Absolute:
  it isolates "the content of the activation" from "the magnitude of the perturbation".
- **NC2 layer 0** → reported as a number, judged only *relative* to the best layer
  (`recovery[best] > recovery[L0]`).

**Why it matters:** a control that is expected to fail but doesn't will get a correct harness
marked FAIL, and then someone will "fix" the harness.
**Confidence:** high.

### F6. Current environment state — the chain cannot run yet  [OBS, measured 2026-11-10]

```
torch          : not installed (.venv has pip only)
transformers   : not installed
weights        : 0.201 GB in blobs/*.incomplete, no *.safetensors linked into the snapshot
HF_HOME        : /home/okey/Desktop/Projects/mod_wei/.hf
```
Measured by `python3 code/proof_chain.py --preflight`. The harness exits 2 and writes
`proof/PRECONDITION_MISSING.txt`. **No C1–C6 result exists yet. Nothing in this file claims one.**

---

## Concrete implementation

### Files

| path | role |
|---|---|
| `code/proof_chain.py` | the harness (single file, stdlib-only imports at module level) |
| `proof/c1_trace.json` | 28 layer records: shape, finite, norm, mean/std/min/max |
| `proof/c1_resid_last.npy` | `[28,1536]` last-position residuals — reusable by A9/C4 |
| `proof/c2_lens_identity.txt` | the identity test **plus the 4-way diagnostic ladder** |
| `proof/c3_patching.txt` | per-layer recovery curve + 3 negative controls |
| `proof/c4_dose_response.json` | full sweep, controls, Spearman, KL per point |
| `proof/c5_midgen.txt` | per-step logit deltas table |
| `proof/c6_force.txt` | forced vs semantic, per prompt, with coherence |
| `proof/summary.tsv` / `.json` | machine-readable roll-up |
| `proof/selftest.txt` | plumbing proof on a random tiny Qwen2 |

### Commands

```bash
cd /home/okey/Desktop/Projects/mod_wei

# 0. does the box have what we need? (never downloads, never loads the model)
.venv/bin/python code/proof_chain.py --preflight

# 1. prove the instrument, no weights required (~2 s)
.venv/bin/python code/proof_chain.py --selftest

# 2. the chain (needs the full 3.1 GB model)
.venv/bin/python code/proof_chain.py

# 3. targeted reruns
.venv/bin/python code/proof_chain.py --only C2
.venv/bin/python code/proof_chain.py --only C1,C2 --stop-on-fail
```

**Exit codes:** `0` all requested links PASS · `1` at least one ran and FAILED · `2` blocked on a
precondition (torch/weights missing) · `3` harness crash. A `FAIL precondition_missing:…` line is
a *blocked* run — do not read it as a refutation.

### Runtime (estimated, not measured — no torch yet)

~230 forward passes total on prompts of 5–10 tokens. At the A9-measured ~55 GFLOPS fp32 and
~18 ms/token-forward, plus ~30–60 s model load: **≈2–5 min end to end**. C6 dominates (~135
passes). This is a *batch* job, consistent with CONTEXT's "design around batch traces".

### What each test proves, and what a failure would mean

**C1 — we can see inside.** Captures `resid_post(L)` for all 28 layers on one prompt; asserts
shape `[1,T,1536]` (i.e. `[T,1536]` per sequence) and `torch.isfinite(...).all()`. It also
cross-checks our hook output against HF's own `output_hidden_states` and requires
**max diff == 0.0** — that is what proves we captured the tensor HF actually feeds to the final
norm, not a lookalike.
*Failure means:* the hook is on the wrong module or stores the wrong slice. Everything downstream
is then measuring the wrong tensor. Fix before anything else.
*Note:* `[T,1536]` is per-sequence; the hook keeps the batch dim, so the stored shape is
`[1,T,1536]` — the JSON records the literal shape so nobody has to trust this sentence.

**C2 — the lens is faithful (the load-bearing link).** Asserts
`logits == lm_head(model.model.norm(resid_post(27)))` over all `T×151936` entries, max abs diff
`< 1e-4`. The artifact prints the diff for four candidate identities.
*Failure means:* **the lens is invalid and C3–C6 are uninterpretable.** The ladder localises it:
`no-norm` matching → wrong hook site; `2x-norm` matching → double application; `off-by-one`
matching → layer 26 / resid_pre; nothing matching → dtype (check the `dtype` line, it must say
`torch.float32`). If C2 fails, the harness prints a loud banner and says so in the artifact.
It does **not** stop C3–C6 by default (they use raw logits, not the lens) but every downstream
artifact is then a measurement of *something else*.

**C3 — activations are causally used.** Clean `"The capital of France is"` vs corrupt
`"The capital of Germany is"`; copy the clean `resid_post(L)` at the last position into the
corrupt run for **every** layer; report
`recovery = (patched − corrupt) / (clean − corrupt)` for the target token. Pass = best layer
`≥ 0.5` **and** self-patch exact **and** random-direction recovery `≤ 0.20` **and**
`recovery[best] > recovery[L0]`.
*Failure means:* either activations at that site are not causally load-bearing for this contrast,
or the patch lands at the wrong position. Check the recorded `contrast` first — if the clean and
corrupt logits for the target token are nearly equal, the test is underpowered, not refuted.

**C4 — we can steer semantics.** Diff-in-means vector at layer 14 over 8 comment/code contrast
pairs, normalised per **A3's pre-registered convention** (`v/‖v‖ · E‖h^(14)‖`, so `coef` is in
units of one residual norm — `agents/A3_steering.md` F1). Sweep `coef ∈ {-3..+3}`; metric is
`logsumexp(logits[comment tokens]) − logsumexp(logits[code tokens])` minus its `coef=0` value.
Pass = Spearman `ρ ≥ 0.8` with the right signs **and** `coef=0` bit-identical **and** three
matched-norm random vectors show `|ρ| < 0.8` and a span `< 0.75×` the real span.
*Failure means:* the vector is not causal at that layer, or the metric is too crude to see it.
Both are informative. A vector that fails at layer 14 may pass at another layer — but that is a
new pre-registered test, not a retry.

**C5 — steering works mid-generation.** 12 greedy tokens; `coef=0` for steps 0–5, `+2` for 6–11.
Pass = `max|Δlogits| == 0.0` exactly for steps 0–5 **and** `> 0.0` at step 6.
*Failure means:* steps <6 differing → hook state is leaking between calls or the schedule is read
at the wrong time; step 6 bit-identical → the hook never fires under cached decoding (wrong
module, or a dtype/device mismatch on the vector).

**C6 — we can force a token.** `logits[target] += 100` (deterministic) vs `coef=+2` semantic
steering, over 5 held-out prompts. Records hit rate at step 0, the **logit push in nats**, and a
coherence score (mean log p of the generated continuation under the clean model).
Pass = forced rate 100% (that is all the forcing path claims).
*Failure means:* the plumbing is broken — forcing is arithmetic on the logits, it cannot
half-work.
*What the numbers are for:* forcing is +100 nats and produces an incoherent continuation;
steering is O(1) nats and only flips the token when the state supports it. That gap **is** the
honest statement of what activation steering can and cannot do.

### Design decisions worth knowing

- `torch.set_num_threads(4)` — 4 physical cores. Using 8 logical threads makes fp32 GEMM
  *slower* here (A9 measured it) and makes timing claims non-comparable.
- `attn_implementation="eager"` — required for attention-weight access; also the only path whose
  numerics are easy to reason about.
- `lm_head` is called **as a module** (`h.lm_head(x)`). `tie_word_embeddings=true` in this
  checkpoint, so there is no separate `lm_head.weight` file to look up; a raw-weight version of
  the same call is how you silently get the wrong matrix.
- The final norm is `model.model.norm` (a `Qwen2RMSNorm`), asserted by `type()` and printed with
  its epsilon in the C2 artifact.
- The model is loaded `local_files_only=True` unless `MOD_WEI_ONLINE=1`. The harness **never
  downloads**.
- `--selftest` uses `Qwen2ForCausalLM._from_config(Qwen2Config(...))` with 3 layers, `d=32`,
  `vocab=64` — the same class as the real model, so the hook contract is identical.

---

## What NOT to do (traps)

1. **Do not apply `model.norm` to `logits`.** That is the `microscope.py` bug (F1). The norm goes
   on the *residual*, never on the output of `lm_head`.
2. **Do not read `lm_head.weight` directly.** With tied embeddings it may be absent, aliased, or
   re-tied at load time. Call the module.
3. **Do not compare with `torch.allclose` for the negative controls.** `VALIDATION.md` says
   bit-identical; `allclose` will happily pass a broken hook.
4. **Do not load in bf16 "because the checkpoint is bf16".** The checkpoint is bf16, the *compute*
   must be fp32: bf16 has ~3 decimal digits and this CPU has no bf16 matmul, so it would be
   emulated and slower. bf16 also makes C2's 1e-4 threshold unreachable for a reason that has
   nothing to do with correctness.
5. **Do not run C4–C6 before C2 passes.** A broken lens makes every steering result
   uninterpretable — that is the ordering rule in `PROOF_CHAIN.md`, and it exists because a
   steering curve looks convincing regardless of whether the instrument is calibrated.
6. **Do not call a monotone curve "steering works"** from a single coefficient. The artifact
   carries all 7 points, the KL per point, and the random-vector controls precisely so the claim
   has to survive the whole curve.
7. **Do not treat `RESULT … FAIL precondition_missing:…` as a refutation.** It is exit code 2:
   the test did not run.

---

## Open questions for other agents

- **A3:** is `v/‖v‖ · E‖h^(L)‖` really the convention you want for `coef ∈ [-3,3]`? At `coef=3`
  that is three residual norms of push; the C4 artifact will show whether the curve saturates or
  collapses at the ends. If it collapses, the pre-registered grid should be `[-2..+2]` and that
  decision belongs in your file, not in mine.
- **A6:** is "recovery ≥ 0.5 at some layer" the right C3 bar, or should C3 require the recovery
  *curve* to have a peak (single-layer, not monotone)? A flat curve and a peaked curve mean
  different things.
- **A2:** C4's metric is a hand-built token set (`#` vs `return/def/for`). If you have a
  better-justified behavioural metric for a *code* model, the metric is a one-function swap.
- **A9:** `proof/c1_resid_last.npy` is `[28,1536]` on one prompt — a free fixture for the atlas
  work if you want a known-good residual to validate your own hooks against.
- **Orchestrator:** should C2's 1e-4 be tightened to 1e-5 once we see the real fp32 number? The
  observed value should be ~1e-6; 1e-4 is a bug detector, not a measurement.

---

## SCHEMA CHANGE

Proposed addition to the canonical trace schema in `CONTEXT.md` — two fields that make a trace
self-describing about *where* it was measured:

```json
{"step": 0, "token_id": 123, "token_str": "def",
 "hook_site": "resid_post",          // NEW: resid_pre | resid_post | post_norm
 "coef_in_residual_norms": 0.0,      // NEW: coefficient in units of E||h^(L)||
 "layers": [...], "logits_topk": [...], "chosen": [...]}
```

Reason: a trace without `hook_site` cannot be compared to another trace, and a coefficient
without its normalisation cannot be compared to another coefficient. `c4_dose_response.json`
already emits `coef_in_residual_norms` per point for this reason.

---

## Build next (ordered, with effort estimate)

1. **`--selftest` as soon as `transformers` is importable** — 2 s, no weights. If it passes, the
   instrument is sound and any later C2 failure is science, not plumbing. *(2 min)*
2. **`--preflight` again after the 3.1 GB download completes** — confirms the snapshot has a
   complete safetensors file linked. *(10 s)*
3. **C1 + C2 only**, `--only C1,C2 --stop-on-fail`. Read `proof/c2_lens_identity.txt` and the
   `dtype` line. *(~2 min incl. load)*
4. **C3**, then read the recovery curve before running C4. If the peak is at layer 0–2 the
   contrast is too shallow and the prompts need work. *(~1 min)*
5. **C4**, and look at the *KL* column as well as the effect column: a rising effect with a
   collapsed KL means the metric is saturating, not that steering is working. *(~2 min)*
6. **C5, C6.** *(~2 min)*
7. **Only then** touch `LEDGER.md` — and only with numbers copied from the artifacts, each
   labelled `[OBS]`.

Total: **~10 min of compute once the preconditions land**, plus however long the download takes.
