# STATE OF PROJECT — 2026-11-11

Purpose of this file: separate what is **settled** from what is **still contested**
so we stop re-litigating the same questions and can see what actually blocks a decision.

---

## 1. What is settled (verified, reproducible)

| claim | evidence | confidence |
|---|---|---|
| We can read every layer's exact state | C1: max diff 0.000e+00 vs HF internals | high |
| The logit lens is faithful at the final layer | C2: max diff 0.000e+00 | high |
| Activations are causally used | C3: L22-27 recovery ~1.0, L0-20 ~0, controls clean | high |
| We can intervene mid-generation | C5: divergence begins after the switch, never before | high |
| Forcing a token works | C6: forced rate 1.00 on bf16 | high |
| A random direction also produces a monotone dose-response | C4 bf16: 2/10 randoms, |rho| 0.86-0.89, larger span | high |
| "Neuron = concept" is false | superposition literature + A6/A9 | high |
| Quantisation hides specificity failures | C4 passed on Q4_K_M, failed on bf16 | high |
| The model is Q4_K_M-capable; bf16 weights are exact and verified | 338 tensors, all BF16, 3,087,467,144 B | high |
| Hardware: i7-8665U, 4 physical cores, AVX2 only, ~17 GB free | lscpu + measured | high |
| ~49 tok/s batched, ~2-3 tok/s single-stream fp32 | measured | high |

## 2. What is NOT settled

### 2.1 The core claim: can we steer semantics?
**Status: PROVISIONAL — not established, not disproved.**
- C4 (specificity) fails on bf16 — but the criterion changed twice mid-session
  and has not been audited (G1 pending). See VALIDATION.md L5.
- C6 (behavioural) shows semantic rate 0.00 on bf16; C5 shows intervention does
  change generation. These point in different directions.
- **Unresolved:** is the failure (a) the vector, (b) the layer, (c) the metric,
  or (d) the claim is false at 1.5B scale?
- The decisive test (`code/steering_test.py`) was run in --quick mode only
  (T1 pass, T2/T3/T4 fail) and the full run was interrupted. It must be rerun
  with 20 controls and a stable criterion before any conclusion is published.

### 2.1b Can we see inside an API model?
**Status: PARTIALLY, not never.**
- Direct hidden states: not exposed by any documented API.
- Partial parameter recovery from logits: **demonstrated** — Carlini et al. 2024
  (arXiv:2403.06634) recovered a production embedding projection layer.
- Full mechanistic insight via surrogates: F1 says mostly no (behavioural
  mimicry), with one favourable-setting audit at 50/50.
- **Unresolved:** how far logit-based recovery generalises. F2 pending.

### 2.2 What are we building?
Four candidates, none chosen:

| candidate | source | requires | state |
|---|---|---|---|
| Research instrument (microscope) | the original idea | nothing more | built, validated |
| Quantisation decision-flip certificate | E3 | fp16 + Q4 pairs; decision margins | not started |
| Prefix-stable context compressor | E2 | cache-hit analysis | not started |
| Safety regression triage for fine-tunes | E4 | an instruct model, not base | not started |

The instrument is the **engine** for all three. That is the only thing they share.

### 2.3 Is this a business at all?
E3's uncomfortable question is unanswered: interpretability buyers are researchers
who expect everything free and open-source. `diffing-toolkit`, nnsight,
TransformerLens, Neuronpedia and circuit-tracer are all free. D3 confirmed 6 of 7
proof links are already publicly demonstrated. **The tool is not a moat.**

---

## 3. Live contradictions (not yet resolved)

| # | contradiction | who says what | status |
|---|---|---|---|
| C-1 | Does steering work? | Q4 run passed C4; bf16 failed; A3 says the literature is solid; E1/AxBench says steering is weak (0.409 vs prompt 0.905) | **open** — bf16 is the honest artefact |
| C-2 | Is "save tokens" the goal? | User's thesis vs E1's arithmetic ($38/mo, prompt = 0.5-6% of bill) | **open** — user has not conceded or reframed |
| C-3 | Which product? | E3: quantisation certificate. E2: prefix-stable compressor. E4: safety triage | **open** — three valid answers |
| C-4 | Does 32B fit? | A8: "fits comfortably at 4.5 bpw" (18.45 GB). D2: only 16 GB available, so false. A7: fits but 0.3 tok/s | **resolved against A8** — D2 correct |
| C-5 | SAE compute cost | A2: "~25 CPU-h, weekend". D2: based on an impossible 150 tok/s constant; real ~150-200 h | **resolved against A2** |
| C-6 | Is the atlas done? | C7 "PASS, 99.99% coverage" vs threshold 1e-6 being trivially loose | **open** — coverage is real, meaning is not |
| C-7 | Which model? | base coder 1.5B (working) vs instruct (needed for refusal) vs Qwen3-1.7B (unlocks circuit-tracer) | **open** — strategic |
| C-8 | What counts as "steering success"? | Criterion changed twice during the session (layer, pairs, span->structure) | **open** — no stable pre-registered criterion |

---

## 4. Unanswered questions, ranked by whether they block a decision

**Blocking (cannot choose a product without these):**
1. Does a behavioural steering effect survive 20 random controls on bf16?
   *If no: the product cannot be "control". It must be "measurement".*
2. Is there a buyer who pays for a certificate rather than a tool?
   *Answerable in 5 days with 3 models x 3 bit-widths and one public report.*
3. What is the one-sentence claim we can defend? Today it is not "we can
   control the model"; it is "we can prove what changes inside it."

**Important but not blocking:**
4. Does the atlas need features (SAE) or is neuron profiling enough? (A9 says
   features; C7 shows neurons are cheap but nearly all fire.)
5. Is C5 genuinely novel (D3 says the public literature never localises the
   divergence step) or is our criterion just weaker?
6. Should the science model switch to Qwen3-1.7B for circuit-tracer support?

**Housekeeping:**
7. A1's per-type GGUF histogram is wrong (57 tensors, not 338); conclusion stands.
8. A5's CircuitsVis API names are CamelCase but the real Python API is snake_case.
9. `fetch_model.sh` produced a corrupt 3.3 GB file; deprecated in favour of `pget.sh`.

---

## 4b. The "backward loop" question (F1/F2/F3) — answered, with a caveat

The owner's hypothesis: closed/API models might still be controllable or readable
through an indirect loop. Three agents investigated.

- **F1 (surrogate distillation): behavioural mimicry, not mechanistic knowledge.**
  - The one published surrogate-vs-teacher circuit study **was not black-box**:
    DistilGPT-2 used a cosine loss on the teacher's *hidden states* plus layer
    initialisation, i.e. it had the internals. Even so, IOI died
    (94.8% -> 35.2%, zero name-movers) and Greater-Than only survived by
    remapping L9H1 -> L4H1.
  - **CKA cannot be the bridge.** Randomly-initialised-then-distilled ResNet
    students already score CKA 0.885-0.939 (Stanton 2021, Table 1), and a student
    with a *dead* IOI circuit still scores CKA 0.921 (Faumi 2026). High CKA is
    compatible with zero functional preservation.
  - Output-only distillation diverges systematically: simplicity bias selects
    mechanisms the teacher lacks (Wu 2024); induction heads are impaired
    (Goyal 2025); unique feature directions (Baek & Tegmark 2025).
  - Cost: a plausible 27% student costs ~88 CPU-h per 5M tokens. A 30M student
    is 1.9% of the 1.54B teacher.
  - Falsifier available if we want to test it ourselves: E0, a ~1.5-hour
    symbolic IOI test specified in agents/F1_surrogate.md.
- **F3 (transfer & limits): the control loop needs target-side information.**
  Cross-model steering only works with shared weights or a map fitted on the
  target's activations. Without that, transfer efficiency is 0.03-0.17. Black-box
  observation cannot distinguish a steered model from a different checkpoint.
- **F2 (gradient-free loops): behaviour, not mechanism — structurally.**
  - Top-k logprobs expose 20 of 151,936 coordinates; the rest are censored
    (-9999.0), so **no token-space gradient exists** to propose candidates.
  - Logit leakage is real (Carlini: full projection matrix for <$20; Finlayson:
    full-vocabulary outputs) but it needs `logit_bias` and recovers only the
    **readout** (final layer, up to RMSNorm scale). Steering vectors live at
    L8/L14, which is not the readout.
  - Steering directions are **formally non-identifiable from behaviour**
    (arXiv:2602.06801: "any direction in this null space can be added …
    without altering any observable output"). Behaviour returns an equivalence
    class, not a direction.
  - What GCG finds is a large-norm perturbation with a substantial orthogonal
    component — **the same object our C4 rejected.**
  - Cost: 2.56e5 queries ≈ $2-$207 per behaviour, ~8.5 h at 500 RPM.
  - Measured local throughput on fp32: 19.7 tok/s (not the 49 tok/s batched
    figure from the atlas run — different batching, both real).

### 2.1c LIVE API RESULT (measured, 2026-11-11) — the loop is closed
- `responseLogprobs` returns **400** on every reachable generation model
  (gemini-3.5-flash, gemini-3.8-flash; control call 200).
- The new **Interactions API removed `candidates` by design** — it returns "the
  single best answer". Per-token distributions are not merely disabled, the
  endpoint shape forbids them.
- Responses **begin with thinking tokens**, so even the argmax of the first real
  token is unobservable.
- The only surviving surface is **embeddings**: 8/8 linear separation of
  comment-vs-code prompts, but pooled, unit-norm, produced by a *different*
  model, and non-injectable. Useful for measurement, not control.
- **Behaviour is cheap; mechanism is not.** 16 one-bit queries reached +7.97 nats
  on the target behaviour, exceeding the local L8 vector's +5.81.
  *Caveat: those are different models and different metrics — not a like-for-like
  comparison. What it does show is that prompt search is a stronger behavioural
  lever than activation steering, consistent with AxBench.*
- Cheapest remaining proof if anyone wants it: 2,000 queries (~$0.09), score a
  prompt's L8 delta against 200 random suffixes.

**Verdict so far:** a loop around an API can produce *behaviour* and *falsifiable
hypotheses*, not mechanism. This does not block the project — it defines the
open-source boundary: open weights = full instrument; API = measurement only.

## 4c. Live app status (2026-11-11)

Running: FastAPI on :8000, Next.js on :3000, Granite 3.3 2B (40 layers, d=2048,
fp16, scaling 8.0).

**Verified working**
- per-token payload: 40 layers, real `xy` from a shared PCA of the standardised
  residual (`var_explained` ~0.63), top-3 neurons, top-3 lens tokens
- provenance: every token carries `steered` / `coef` / `steer_layer`
- activation steering reaches the logits; `coef=0` is a no-op
- mid-generation injection: turn-formatted text changes the continuation
- layer drill-down endpoint returns top-32 neurons

**Defects found by looking at the running app (not by reading code)**
- the header and empty state hardcoded **28 layers** while the model has 40
- the footer still said **"summary space (proxy)"** after real PCA shipped
- the 3D plane stack rendered as a faint wireframe, no layer labels
- `/api/health` reported `busy: true` for 160 s with `generations: 0`: a
  generation thread held the lock — the app was unusable until restart
- injection had two faults: raw text was ignored, and the loop re-fed a token
  already in the KV cache (see README)
- the client cleared traces on every send, which reads as "it reloads each time"

**Lesson.** Every one of these was invisible to the test suite and obvious within
60 seconds of running the app and taking a screenshot. Green tests are not
evidence of a usable product.

### Verified by driving the real UI (headless Chrome, CDP)
Screenshots: `/tmp/opencode/shots/ui4.png` (rest), `ui6.png` (streaming).

- Sending `def add(a, b):` produced a reply and a live trace:
  `HUD: TOKENS 4 · ACTIVE 10 · LAYER L39 · TOK/S 0.6`
  `panel: T3 " you" · steered false · coef 0.00 · steer_layer null · layers 40`
- 40 layer columns render with a real node trajectory sized by `norm`.
- `inject` and `Cancel` are both present during streaming.
- Time-to-first-token after the fp32+SDPA switch: **9.0 s** (was 27.7 s).

**Defects found only by clicking:**
- example chips filled the composer but never sent (copy said "try one of these")
- the client ignored the `prefill` frame, leaving 9 s of silent dead air
- each token replaced the graph instead of animating from the previous state

The method matters more than any single fix: **render it, click it, screenshot
it.** Reading the code and trusting green tests found none of the above.

## 5. What is true today, in one sentence

> We have a locally-running, causally validated instrument that can prove what
> changes inside a 1.5B model — and we do not yet have evidence that it can
> control the model in a way a random perturbation cannot.
