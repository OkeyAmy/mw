# D3 — Precedent and independent evidence

**Scope:** for each link C1–C7 in `PROOF_CHAIN.md`, find ≥2 *existing, working, independently
verifiable* public artifacts. Every URL below was fetched during this session; HTTP status and
extracted text are recorded in the Verification Log (§11). Nothing here is cited from memory.

**Rule applied:** an artifact counts only if (a) the repo/page resolves, (b) the code or weights
exist at that URL, and (c) I could quote the specific line that proves the capability.

---

## TL;DR (5 bullets)

1. **Six of seven links are already proven in public.** C1, C2, C3, C4, C6 are routine;
   C7 is proven for code models at 7B and for Qwen2.5-Coder-**1.5B-Instruct**, not for the base 1.5B.
2. **C5 is proven as a mechanism, not as our test.** Per-token intervention APIs exist and are
   documented (nnsight `tracer.iter`, EasySteer `generation_window`), and two papers recompute the
   coefficient every token. **No public artifact does the step-exact experiment** ("change coef at
   token 6 → divergence starts at token 6"). That is the one genuinely original experiment.
3. **C6 has a real hole:** there is a published, all-layer SAE atlas for Gemma 2 2B and Gemma 3 1B,
   and **none for Qwen2.5-Coder-1.5B**. Our atlas is new work, not reuse.
4. **The lens tolerance in our proof chain is stricter than the reference implementation's own
   check.** TransformerLens ships a logit-lens tool whose internal identity check is *relative*
   (`0.1 × logit spread`), while its unit tests use `atol=1e-4`. We must write our own absolute
   assertion; the library will not enforce ours.
5. **Our negative control is statistically unsafe as written.** A 2026 paper reproduces ActAdd on
   GPT-2-XL and finds that **9 of 20 fresh random vectors produce larger mean KL than the steering
   arm**. "Random vector no effect" must be a distribution test over ≥20 seeds, not one vector.

---

## Findings

### F1. Per-layer hidden states from a small open model, extracted and visualised
**Verdict: CONFIRMED, 3 independent proofs.**

| artifact | what it proves | licence | last commit | CPU? | reuse |
|---|---|---|---|---|---|
| `TransformerLensOrg/TransformerLens` `demos/Exploratory_Analysis_Demo.ipynb` | `cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1)` returns every layer's residual, then plots a logit-lens curve | MIT | 2026-11-09 | yes | **direct** |
| `poloclub/transformer-explainer` (live: poloclub.github.io/transformer-explainer) | runs GPT-2-small in-browser (ONNX), renders real per-layer internals + attention for user text; CHI'26 paper, 490k users | MIT | 2026-06-06 | yes (WASM) | UI reference only |
| `ndif-team/nnsight` `/documentation/intervention/iterator/` | `with model.generate(...) as tracer: for step in tracer.iter[:3]` — per-step hidden-state capture | MIT | 2026-11-06 | yes | **direct** |

Quote (TransformerLens notebook, cell 41):
```python
accumulated_residual, labels = cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache)
```
Why it matters: this is exactly our C1 artifact, already written, for GPT-2. Porting to Qwen is a
config change, not new science.

---

### F2. A faithful logit lens on a modern model, with published code
**Verdict: CONFIRMED, 3 independent proofs.**

| artifact | what it proves | licence | last commit | CPU? | reuse |
|---|---|---|---|---|---|
| `TransformerLensOrg/TransformerLens` `transformer_lens/tools/analysis/logit_lens.py` (in PyPI **v4.3.0**) | applies the real `ln_final` → `W_U` → `b_U` → adapter output transform; refuses to run if that path doesn't reproduce the model's logits | MIT | 2026-11-09 | yes | **direct** |
| `tests/integration/test_logit_lens.py` | `TOL = dict(atol=1e-4, rtol=1e-4)`; `test_final_entry_reproduces_logits` asserts the final entry equals real logits | MIT | same | yes | **direct** |
| `AlignmentResearch/tuned-lens` (PyPI `tuned-lens`) | learned affine probe per layer; paper states "All code needed to reproduce our results can be found at github.com/AlignmentResearch/tuned-lens" | MIT | 2025-08-07 | yes | fallback |
| nnsight `/vllm/logit-lens/` tutorial | second, independent logit-lens implementation | MIT | 2026-11-06 | GPU (vLLM page) | reference |

Quote (`logit_lens.py:411`):
```python
def _require_readout_identity(...):
    """Refuse when ``ln_final -> unembed`` on the last block's output does not give ``logits``."""
```
**Important nuance:** the tool's own guard uses `_READOUT_IDENTITY_RTOL = 0.1` against the *spread*
of the logits (`logit_lens.py:66`, line 451). That is much looser than our `max abs diff < 1e-4`.
The 1e-4 number lives only in the test suite. **We must implement C2's assertion ourselves.**

---

### F3. Causal activation patching on a real model, reproducible notebook
**Verdict: CONFIRMED, 3 independent proofs.**

| artifact | what it proves | licence | last commit | CPU? | reuse |
|---|---|---|---|---|---|
| `TransformerLensOrg/TransformerLens` `demos/Activation_Patching_in_TL_Demo.ipynb` | clean/corrupted pairs → patch each (layer, position) → metric heatmap; 27 uses of "patching", 16 `imshow` | MIT | 2026-11-09 | yes (notebook asks for GPU; GPT-2 runs on CPU) | **direct** |
| `stanfordnlp/pyvene` `tutorials/advanced_tutorials/Causal_Tracing.ipynb` (+ `IOI_Replication.ipynb`, `IOI_with_DAS.ipynb`) | replicating ROME causal tracing with an intervention framework | Apache-2.0 | 2026-03-05 | yes | fallback |
| nnsight `/tutorials/causal_mediation_analysis/activation_patching/` | third independent patching tutorial | MIT | 2026-11-06 | yes | reference |

Quote (TL notebook): *"This demonstrates how to use the Activation Patching utils in TransformerLens."*
Caveat: the TL notebook header says *"go to Runtime > Change Runtime Type and select GPU"*. That is a
Colab convenience, not a hard requirement for GPT-2-small; for Qwen-1.5B on our i7 it will be slow but
legal.

---

### F4. Activation steering that visibly changes behaviour, with repo + reported effect sizes
**Verdict: CONFIRMED, 6 independent proofs.**

| artifact | effect size (quoted) | licence | last commit | CPU? | reuse |
|---|---|---|---|---|---|
| `vgel/repeng` | README shows strength −2.2 / +1 / +2.2 producing visibly different completions; `ControlModel.set_control(vector, coeff)` | MIT | 2025-09-24 | yes | **direct (vendor)** |
| `nrimsky/CAA` (Rimsky et al., ACL 2024) | CAA steering vectors from contrast pairs; eval scripts + `plot_results.py --layers $(seq 0 31) --multipliers -1 0 1` | MIT | 2024-05-23 | GPU (Llama-2) | method |
| `steering-vectors/steering-vectors` (PyPI) | library + Colab notebook (`examples/caa_sycophancy.ipynb`), docs at steering-vectors.github.io | MIT | 2025-02-21 | yes | **direct** |
| `likenneth/honest_llama` (ITI, NeurIPS'23) | **truthfulness 32.5% → 65.1%** on Alpaca | MIT | 2025-01-28 | GPU (Llama-7B) | method |
| `andyrdt/refusal_direction` | directional ablation removes refusal; Apache-2.0 | Apache-2.0 | 2024-10-01 | GPU | method |
| Turner et al., *ActAdd*, arXiv 2308.10248 | "SOTA on negative-to-positive sentiment shift and detoxification using models including LLaMA-3 and OPT" | paper | — | — | method |

Note: `nrimsky/Conditional-Activation-Additions` does **not** exist — the repo is `nrimsky/CAA`
(verified: `git ls-remote` → `ERROR: Repository not found` for the long name, `5dabbbd...` for CAA).
Do not cite the wrong URL.

---

### F5. Mid-generation / per-token intervention
**Verdict: CONFIRMED as mechanism, NOT confirmed as a step-exact experiment.**

| artifact | what it proves | licence | last commit | CPU? | reuse |
|---|---|---|---|---|---|
| nnsight `/vllm/conditional/` docs | per-decode-step conditional intervention **with printed output**: inside `for step in tracer.iter[:8]:` it reads the step's logits and rewrites `model.samples` / `model.logits` | MIT | 2026-11-06 | API is GPU/vLLM on that page; core `tracer.iter` works with HF on CPU | **direct** |
| `ZJU-REAL/EasySteer` (EMNLP'26 demo) | `ApplySpec(prompt_positions=[-1], generation="all")` and `generation_window=(0, 4)` — steer only decode steps 0–3 | Apache-2.0 | 2026-09-15 | **no** (vLLM + CUDA) | design reference |
| `vgel/repeng` `repeng/control.py` | control applied inside `ControlModule.forward`, i.e. every decode step; `set_control()` can be called between steps | MIT | 2025-09-24 | yes | **direct** |
| SAPS (EMNLP'26 Findings) | "autoregressively recomputing projection coefficients during decoding"; measures JS divergence **per generation step** | paper | — | — | method |
| DSEM (ACL'25 Findings) | "dynamically determines steering vectors for generating the next few tokens" | paper | — | — | method |
| `MaxBelitsky/cache-steering` | one-shot **KV-cache** intervention before generation (SmolLM2-360M, `examples.ipynb`) | **NO LICENSE FILE** | 2026-08-18 | yes | **do not copy** |

Quote (nnsight conditional docs, verified code block):
```python
with model.trace(PROMPT, temperature=0.0, max_tokens=8) as tracer:
    for step in tracer.iter[:8]:
        h = model.model.norm(sum(model.model.layers[28].output)[-1:])
        guess = model.logits_processor(model.lm_head, h).argmax(-1).item()
        if guess == paris:
            model.samples[:] = berlin
```
That is a *conditional, per-token* edit with a published expected output. It is not the same as
"coefficient schedule changes at step 6", but it is one line away.

---

### F6. A neuron / feature atlas for a ~1–2B model
**Verdict: CONFIRMED for Gemma, ABSENT for Qwen.**

| artifact | what it proves | licence | date | CPU? | reuse |
|---|---|---|---|---|---|
| Gemma Scope (arXiv 2408.05147) `google/gemma-scope-2b-pt-res` | "JumpReLU SAEs trained on **all layers and sub-layers** of Gemma 2 2B and 9B"; 321 files in the residual-stream repo | CC-BY-4.0 (weights) | 2025-01-19 | yes (SAE inference) | **direct** |
| Gemma Scope 2 `google/gemma-scope-2-1b-pt` | "SAEs on three different sites ... for **every layer** of the pretrained and instruction-tuned models of 270M, **1B**, 4B, 12B, 27B" — tree API returns layers **0–25** and sites `attn_out, attn_out_all, clt, crosscoder, mlp_out, mlp_out_all, resid_post, resid_post_all, transcoder, transcoder_all` | CC-BY-4.0 | 2026-01-07 | yes | **direct** |
| Neuronpedia (live API) | `GET /api/feature/gemma-2-2b/12-gemmascope-res-16k/1000` → `"description": "references to software packages, libraries, or APIs"` | site ToS | live | yes | UI reference |
| `decoderesearch/SAELens` | loader for the above (`library_name: saelens` on the model card) | MIT | 2026-11-10 | yes | **direct** |

**Gap:** searching HF for Qwen SAEs returns nothing at 1.5B for Qwen2.5-Coder. The only Qwen-side
atlas hit was a generic `Qwen2.5-1.5B` representation-engineering toy repo, not a feature atlas.

---

### F7. A base (non-instruct) **code** model responds to activation steering
**Verdict: CONFIRMED at 7B base and 1.5B-instruct; NOT confirmed at 1.5B base.**

| artifact | model | claim | licence | date | code? |
|---|---|---|---|---|---|
| `MechCode1/Correct_Codegen` (ICML'26 workshop) | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `-7B-Instruct` | code correctness is a 1-D subspace; **+15.6% / +11.0% pass@1** forward, −12.9% / −11.7% backward | MIT | 2026-05-21 | **yes, 4 scripts** |
| Rahman, Guha, Menon, arXiv **2603.23629** | CodeGemma-7B, **Qwen2.5-Coder-7B**, Llama3.1-8B | difference-in-means layer-wise vectors added to hidden states during generation; "substantially increase generation toward the target ecosystem ... even when prompts explicitly request the opposite" | CC-BY-4.0 | 2026-03-26 | **no repo found** |
| *When Does Activation Steering Change What a Model Computes From?* arXiv **2606.29522** | "primary three arms are built from **Qwen2.5-Coder-7B** ... The base arm is the unmodified pretrained model" | mean-activation steering works but does **not** recreate the natural activation; needs 26–77× natural-scale norms | paper | 2026 | no |
| `microsoft/llm-steer-instruct` (ICLR'25) | four models | abstract: "steering vectors computed on instruction-tuned models **can transfer to improve base models**" | MIT (`LICENSE.txt`) | 2025-05-27 | **yes** |

Caveat: 2603.23629 names "Qwen2.5-Coder-7B" without "-Instruct", which is the base HF ID, but the
paper never states the training stage. **Mark base/instruct for that paper `UNVERIFIED`.** 2606.29522
is explicit ("unmodified pretrained model"), so base-code steering is proven — at 7B.

---

## Evidence strength table

| link | independent public proofs | strongest artifact | confidence | residual risk |
|---|---|---|---|---|
| **C1** see inside, 28 layers | **3** (TransformerLens, nnsight, Transformer Explainer) | TL `accumulated_resid` + `run_with_cache` | **very high** | none material |
| **C2** lens faithful, <1e-4 | **3** (TL logit_lens + tests, tuned-lens, nnsight) | TL `logit_lens.py` + `atol=1e-4` test | **high** on technique, **medium** on our tolerance | TL's own guard is *relative* 0.1×spread; 1e-4 on a 151,936-vocab Qwen is unmeasured publicly |
| **C3** causal patching | **3** (TL demo, pyvene causal tracing, nnsight) | TL `Activation_Patching_in_TL_Demo.ipynb` | **very high** | none material |
| **C4** steer semantics, dose-response | **6** (repeng, CAA, steering-vectors, ITI, refusal_direction, ActAdd) | ITI 32.5→65.1%; Correct_Codegen +15.6% pass@1 | **very high** | "monotone dose-response" as a *criterion* is unpublished |
| **C5** mid-generation coefficient change | **4 mechanisms, 0 step-exact demos** (nnsight, EasySteer, repeng, SAPS/DSEM) | nnsight `tracer.iter` conditional edit | **medium-high** | our exact pass criterion has no precedent |
| **C6** atlas at 1–2B | **3 for Gemma, 0 for Qwen** | Gemma Scope 2 (Gemma 3 1B, all 26 layers) | **high** that it's possible, **low** that we can reuse | no Qwen2.5-Coder atlas exists |
| **C7** base code model steers | **4** (1.5B-instruct code, 7B base paper ×2, base transfer) | Correct_Codegen (MIT, 1.5B, runnable) | **high** at 7B, **medium** at 1.5B base | interpolation, not a published result |

---

## What is genuinely NOT proven anywhere — the honest gaps

This is the section that should drive budgeting. Each gap is a place where we cannot cite, only measure.

**G1. Step-exact mid-generation causality (C5).**
Nobody publishes "coefficient changes at token 6, therefore the first divergent token is token 6, and
tokens 0–5 are bit-identical". SAPS and DSEM change the coefficient per token but report aggregate
metrics (JS divergence averaged over steps, benchmark scores). nnsight and EasySteer give the
*mechanism*; neither gives the *localisation proof*. **This is our one genuinely novel experiment.**
Budget: 1 day, and it may fail for a boring reason — the first affected token's logits may not change
the argmax for several steps because the effect is small relative to the margin. Design the test on a
prompt where the steered direction flips a close call.

**G2. Absolute `max abs diff < 1e-4` lens identity on Qwen2.5-Coder-1.5B.**
TransformerLens tests 1e-4 on GPT-2 (vocab 50,257). Our vocab is 151,936 and the model uses **tied
embeddings**. Nobody has published this number for Qwen. Expect ~1e-6 in fp32, but if it lands at
3e-4 we must either relax the criterion (and say so) or switch the readout to float64. **Do not
discover this late.**

**G3. Any feature/neuron atlas for Qwen2.5-Coder-1.5B.** Does not exist. Gemma Scope is a different
model family with a different tokenizer; its features do not transfer. If C7 in our chain means
"250,880 MLP neurons profiled", that is a *neuron* atlas (cheap: forward passes + statistics), not an
SAE atlas (expensive). **Nobody has published either for this model.** Budget the neuron atlas at
days; budget an SAE at "probably not on this CPU".

**G4. Base (non-instruct) 1.5B code model steering.** We have 7B-base (paper, no code) and
1.5B-**instruct** (code, MIT). The 1.5B-base case is an interpolation. Risk is low (the direction is
a property of the pretrained residual stream, and 2606.29522 shows base arms steer), but it is not a
citation.

**G5. CPU-only interactive mid-generation steering at ~1.9 tok/s.** Every artifact above assumes a
GPU. No public demo does per-token inspection + intervention on a 4-core AVX2 laptop. This is an
engineering gap, not a scientific one, but it is the one most likely to make the product feel broken.

**G6. "Monotone dose-response" as a pass criterion.** CAA/ITI/ActAdd show coefficient sweeps in
figures; none asserts monotonicity as a formal test. Published behaviour is often non-monotone at
large |coef| (repeng's own README shows `strength=2.2` degenerating into repeated garbage). **Our C4
criterion may be falsified by the literature before we run it.** Expect monotonicity only in a band,
e.g. |coef| ≤ 2.

**G7. Single-random-vector negative controls are not null.**
*Stress-Testing Random-Vector Controls for Activation Steering* (BlackboxNLP 2026,
aclanthology.org/2026.blackboxnlp-1.73) reproduces ActAdd on GPT-2-XL to **1.74e-6** max abs error,
then reports: *"At coefficient 10, 9 of 20 fresh random-control realizations have greater mean
normalized KL divergence than the steering arm."* Our VALIDATION.md rule "a random vector of matched
norm must produce no systematic effect" is therefore **wrong as written**. Replace with: 20 random
vectors, compare the steering arm against the random distribution (e.g. steering arm must exceed the
95th percentile), and report the p-value.

**G8. Nothing proves the UI is novel.** Transformer Explainer (490k users, CHI'26), Neuronpedia, and
CircuitsVis already own "pretty brain view". Any novelty claim must come from the measurement, not
the visualisation.

---

## Reusable code shortlist (permissive licences only)

| # | repo | licence | why | integration cost |
|---|---|---|---|---|
| 1 | **`TransformerLensOrg/TransformerLens` v4.3.0** | **MIT** | already has: `run_with_cache`, `accumulated_resid`, `logit_lens.py`, patching utils, and a `Qwen2ForCausalLM` adapter (`transformer_lens/model_bridge/supported_architectures/qwen2.py`, present in the v4.3.0 tag) | low; one config |
| 2 | **`ndif-team/nnsight` 0.8.2** | **MIT** | the only clean way to intervene *inside* `generate()` per step (`tracer.iter`); no weight conversion, wraps plain HF | low |
| 3 | **`vgel/repeng`** | **MIT** | ~300 LOC, `ControlModel` + `set_control(coeff)` between forwards; `export_gguf` if we ever need it | very low (vendor the file) |
| 4 | **`steering-vectors/steering-vectors`** | **MIT** | vector training + evaluation harness, clean API, Colab example | low |
| 5 | **`decoderesearch/SAELens`** | **MIT** | only sane loader for Gemma Scope / Gemma Scope 2 if we do SAE work | medium |

**Fallbacks:** `stanfordnlp/pyvene` (**Apache-2.0**) if TransformerLens' Qwen bridge misbehaves —
it is hook-based and framework-agnostic. `AlignmentResearch/tuned-lens` (**MIT**) if the raw logit
lens proves too brittle at early layers.

**Explicitly NOT reusable:**
- `MaxBelitsky/cache-steering` — **no LICENSE file** (default all-rights-reserved). Read it, don't copy it.
- `goodfire-ai/causalab` — no LICENSE file found on the repo page.
- Gemma Scope weights — **CC-BY-4.0**, attribution required; fine for research, but it is a different
  model family so it buys us nothing for Qwen.

---

## Novelty statement (blunt)

**The microscope is not novel. Every component is published and most of it is MIT-licensed code we
can import this week.** Logit lens (2020/2023), activation patching (2022–2023), difference-in-means
steering (2023–2024), per-token intervention APIs (2024–2026), all-layer SAE atlases (2024–2026) —
none of this is ours. If the deliverable were "a nicer UI over logit lens + steering", the honest
description would be **a nicer UI on known techniques**, and it would be judged as such.

What is defensibly new, in descending order of strength:

1. **A step-localised causality test for mid-generation steering (C5).** Not "steering changed the
   output" but "the output changed at exactly the step where the coefficient changed, and the prefix
   is bit-identical". No public artifact does this. It is a *measurement*, and it is falsifiable.
2. **An empirical per-layer sensitivity map for Qwen2.5-Coder-1.5B *base*.** Two papers steer 7B base
   code models; nobody publishes layer-band → behaviour maps for the 1.5B base. This is real data,
   cheap to produce, and it is the input A8 needs for interpretability-guided compression.
3. **An integrated CPU-only instrument with a canonical trace schema.** Not a technique, an
   engineering contribution: one schema, one adapter interface, runs on 4 cores. Defensible only if
   it actually runs at a usable rate.
4. **Possibly** the compression result (A8): using per-layer sensitivity to allocate bits. That is
   the only place where a *new claim about models* could come from, and it depends on 2 and 3.

**If C5's step-exact test fails and A8's compression shows no gain, what remains is a well-built
teaching tool.** Say that out loud before spending three weeks on the UI.

---

## Concrete implementation

### C2 — do not trust the library's guard; assert absolutely
```python
# our own check, not TransformerLens'
with torch.no_grad():
    resid = cache["resid_post", 27]              # [T,1536]
    ours  = model.unembed(model.ln_final(resid)) # [T,151936]
    diff  = (ours.float() - logits.float()).abs().max().item()
assert diff < 1e-4, f"C2 FAIL: {diff:.3e}"       # record the number, not a boolean
```
Also record `diff` for **every** layer's lens output vs a full re-run truncated at that layer, so a
broken middle layer cannot hide behind a correct last one.

### C5 — the experiment nobody has published
```python
# nnsight, HF backend, CPU
V = steer_vector[14]                     # [1536]
with model.generate(prompt, max_new_tokens=20, do_sample=False) as tracer:
    toks = []
    for step in tracer.iter[:]:
        if step >= 6:                    # coefficient turns on here
            model.model.layers[14].output[0][:, -1, :] += 2.0 * V
        toks.append(model.samples.item())
# pass: toks[0:6] == baseline[0:6]  (bit-identical)
#       toks[6]   != baseline[6]     (diverges at the step, not before)
# control: same code with coef=0.0 -> all 20 identical
# control: same code with V=random(seed i), i=1..20 -> report how many diverge
```
Report `first_divergence_index` for 20 prompts × 5 coefficients. That table is the artifact.

### Cost arithmetic for the lens on CPU (do this before promising a UI)
```
per-layer lens MACs = 151936 * 1536            = 233,373,696
28 layers                                      = 6,534,463,488 MAC  = 13.07 GFLOP
at 60 GFLOPS fp32 (A9 measurement)             = 0.218 s per generated token
```
`[INF]` Generation itself is ~0.53 s/token at 1.9 tok/s. **A full 28-layer logit lens per token
roughly adds 40% to per-token latency.** Therefore: lens every layer only on *paused* tokens; during
streaming, lens 3–4 sampled layers or every 5th token. This is a design constraint, not an
optimisation.

---

## What NOT to do (traps)

1. **Don't cite `nrimsky/Conditional-Activation-Additions`.** It 404s. The repo is `nrimsky/CAA`.
2. **Don't copy `MaxBelitsky/cache-steering` or `goodfire-ai/causalab`.** No licence file = no rights.
3. **Don't use a single random vector as the C4 negative control.** 9/20 random vectors beat the
   steering arm on KL in a published reproduction. Use a distribution.
4. **Don't assume TransformerLens' `logit_lens` tool enforces 1e-4.** It enforces `0.1 × spread`.
5. **Don't claim "mid-generation steering is novel".** It is a documented nnsight feature and an
   EasySteer API flag. Only the *localisation test* is novel.
6. **Don't quote Gemma Scope as evidence about Qwen.** Different family, different tokenizer,
   different feature basis. It proves the *method* scales to 1B, not that our atlas will work.
7. **Don't run the patching notebook on CPU for Qwen without batching.** It patches (layer × position)
   cells one at a time; at 1.9 tok/s that is hours. Batch the patch positions.

---

## Open questions for other agents

- **A2:** does `logit_lens` at layers 0–4 of Qwen2.5-Coder produce anything but noise? If the first
  quartile is uninterpretable, our trace schema wastes 1/4 of its bytes.
- **A3:** what is the *measured* monotonicity band for difference-in-means on this model? My guess
  from repeng's README is |coef| ≤ 2, but that is `[SPEC]`.
- **A4:** can nnsight's `tracer.iter` and a `generate()` loop coexist at 1.9 tok/s without re-running
  the prefill? Measure prefill cost separately from decode cost.
- **A6:** replace VALIDATION.md rule 5 bullet 2 with a distributional control. Cite G7.
- **A8:** the sensitivity map from C7 is your input. If C7 only produces a *neuron* atlas, does that
  give you layer sensitivity, or do you need per-head ablation too?
- **A9:** is 250,880 the right denominator? 28 layers × 8960 intermediate = 250,880 — yes, but only if
  we profile `mlp.gate_proj`/`up_proj` pre-activation. Confirm the hook point.

---

## SCHEMA CHANGE (proposed)

Add three fields, because C5 and C8 need them and the current schema cannot express either:

```json
{"step": 6, "token_id": 123, "token_str": "def",
 "steer": {"layer": 14, "coef": 2.0, "applied": true, "vector_id": "style_v1"},
 "layers": [...],
 "logits_topk": [...], "chosen": [id, tok, prob],
 "diverged_from_baseline": false}
```
- `steer` — without it, a trace cannot be audited for "was the hook on at this step?".
- `diverged_from_baseline` — the C5 artifact is a *table of first-divergence indices*; the schema
  should carry it so the UI can mark the step where the model changed its mind.

---

## Build next (ordered, with effort estimate)

| # | task | effort | kills what if it fails |
|---|---|---|---|
| 1 | Load Qwen2.5-Coder-1.5B through TransformerLens v4.3.0 bridge; dump 28 × `[T,1536]`; assert finite | 0.5 d | C1, and therefore everything |
| 2 | Logit lens + **absolute** 1e-4 identity check; log the number | 0.5 d | C2; if >1e-4, switch readout to fp64 before proceeding |
| 3 | Port `Activation_Patching_in_TL_Demo` to Qwen; one clean/corrupt pair; random-layer control | 1 d | C3 |
| 4 | Build diff-in-means vector (style: comment-heavy vs bare code); sweep −3..+3; 20 random controls | 1 d | C4; expect non-monotonicity above ~2 |
| 5 | **Step-exact mid-generation test** (nnsight); 20 prompts × 5 coefs | 1.5 d | C5 — the novel claim |
| 6 | Neuron atlas: 28 × 8960 activation statistics over a code corpus | 3–5 d | C7 (the cheap version) |
| 7 | Only then: UI. Reuse Transformer Explainer's layout logic, not its code | 1 wk | nothing scientific |

Do **not** start at 7. Items 1–5 are ~4.5 days and decide whether the project has a claim.

---

## Verification log (every URL fetched this session)

| URL / command | status | what came back |
|---|---|---|
| `api.github.com/repos/TransformerLensOrg/TransformerLens` | 200 | MIT, pushed 2026-11-10, 4059 stars |
| `api.github.com/repos/ndif-team/nnsight` | 200 | MIT, pushed 2026-11-10, 1144 stars |
| `pypi.org/pypi/transformer-lens/json` | 200 | version **4.3.0**, license MIT, python >=3.10 |
| `pypi.org/pypi/nnsight/json` | 200 | version **0.8.2**, license MIT, python >=3.10 |
| `raw.githubusercontent.com/.../v4.3.0/transformer_lens/tools/analysis/logit_lens.py` | 200 | file exists in the released tag |
| `raw.githubusercontent.com/.../v4.3.0/.../supported_architectures/qwen2.py` | 200 | Qwen2 adapter ships in v4.3.0 |
| clone + `git log -1` `TransformerLensOrg/TransformerLens` | ok | 2026-11-09T11:26:57-06:00 |
| `tests/integration/test_logit_lens.py` line 18 | ok | `TOL = dict(atol=1e-4, rtol=1e-4)` |
| `logit_lens.py` line 66 / 451 | ok | `_READOUT_IDENTITY_RTOL = 0.1`, compared to spread |
| `demos/Exploratory_Analysis_Demo.ipynb` cell 41 | ok | `cache.accumulated_resid(...)`, logit-lens line plot |
| `demos/Activation_Patching_in_TL_Demo.ipynb` | ok | clean/corrupted patching, 16 `imshow` |
| `demos/Qwen.ipynb` | ok | Qwen demo + `assert_hf_and_tl_model_are_close` |
| `github.com/vgel/repeng` commits atom | 200 | 2025-09-24, "qwen attention_type fix (#73)" |
| `raw.../vgel/repeng/main/repeng/control.py` | 200 | `set_control(coeff)`, add inside `ControlModule.forward` |
| `raw.../nrimsky/CAA/main/LICENSE` | 200 | MIT, "Copyright (c) 2024 Nina Rimsky" |
| `git ls-remote nrimsky/CAA` | ok | `5dabbbd…` 2024-05-23 |
| `git ls-remote nrimsky/Conditional-Activation-Additions` | **error** | Repository not found |
| `raw.../steering-vectors/steering-vectors/main/LICENSE` | 200 | MIT |
| `git ls-remote steering-vectors` | ok | 2025-02-21 |
| `raw.../likenneth/honest_llama/master/LICENSE` | 200 | MIT, 2025-01-28 |
| `raw.../andyrdt/refusal_direction/main/LICENSE` | 200 | Apache-2.0, 2024-10-01 |
| `raw.../AlignmentResearch/tuned-lens/main/LICENSE` | 200 | MIT, 2025-08-07 |
| `arxiv.org/abs/2308.10248` | 200 | ActAdd abstract, LLaMA-3 / OPT sentiment SOTA |
| `arxiv.org/abs/2306.03341` (ITI) | 200 | "32.5% to 65.1%" |
| `nnsight.net/documentation/intervention/iterator/` | 200 | `for step in tracer.iter[:3]` |
| `nnsight.net/vllm/conditional/` | 200 | per-step conditional edit + printed output |
| `nnsight.net/vllm/steering/`, `/vllm/patching/`, `/vllm/logit-lens/` | 200 | all exist |
| `poloclub.github.io/transformer-explainer/` | 200 | "powered by the GPT-2 (small) model which has 124 million parameters" |
| `github.com/poloclub/transformer-explainer` | ok | MIT, 2026-06-06 |
| `arxiv.org/abs/2408.04619` | 200 | CHI'26, live GPT-2 in browser, 490k users |
| `arxiv.org/abs/2408.05147` (Gemma Scope) | 200 | "all layers and sub-layers of Gemma 2 2B and 9B" |
| `huggingface.co/api/models/google/gemma-scope-2b-pt-res` | 200 | 321 files, CC-BY-4.0, 2025-01-19 |
| `huggingface.co/api/models/google/gemma-scope-2-1b-pt/tree/main?recursive=true` | 200 | layers **0–25**, 10 site types, CC-BY-4.0, 2026-01-07 |
| `raw.../google/gemma-scope-2-1b-pt/main/README.md` | 200 | "every layer of ... 270M, 1B, 4B, 12B and 27B" |
| `neuronpedia.org/api/feature/gemma-2-2b/12-gemmascope-res-16k/1000` | 200 | `"references to software packages, libraries, or APIs"` |
| `github.com/decoderesearch/SAELens` | ok | MIT, 2026-11-10 |
| `arxiv.org/html/2603.23629v2` | 200 | Qwen2.5-Coder-7B / CodeGemma-7B / Llama3.1-8B steered; no repo link |
| `arxiv.org/abs/2606.29522` | 200 | "primary three arms are built from Qwen2.5-Coder-7B ... base arm is the unmodified pretrained model" |
| `icml.cc/virtual/2026/82743` | 200 | Correct_Codegen, +15.6%/+11.0% pass@1, repo `MechCode1/Correct_Codegen` |
| `raw.../MechCode1/Correct_Codegen/main/README.md` | 200 | 4 scripts, both Qwen2.5-Coder-Instruct sizes, MIT, 2026-05-21 |
| `raw.../microsoft/llm-steer-instruct/main/README.md` + `LICENSE.txt` | 200 | MIT, 2025-05-27 |
| `arxiv.org/abs/2410.12877` | 200 | "steering vectors computed on instruction-tuned models can transfer to improve base models" |
| `github.com/ZJU-REAL/EasySteer` + docs `user-guide/steering` | 200 | Apache-2.0, 2026-09-15, `generation_window=(0,4)` |
| `raw.../MaxBelitsky/cache-steering/main/README.md` | 200 | SmolLM2-360M demo; **no LICENSE file** |
| `github.com/MaxBelitsky/cache-steering` tree | 200 | no `LICENSE` in listing |
| `github.com/goodfire-ai/causalab` | 200 | no licence badge / no LICENSE found |
| `raw.../graySwanAI/circuit-breakers/main/LICENSE` | 200 | MIT, 2024-09-24 |
| `aclanthology.org/2026.blackboxnlp-1.73/` | 200 | "9 of 20 fresh random-control realizations have greater mean normalized KL" |
| `huggingface.co/Qwen/Qwen2.5-Coder-1.5B` | 200 | 28 layers, 12 Q / 2 KV GQA, 32k ctx, "Training Stage: Pretraining", apache-2.0 |

`[OBS]` = read from the page above. `[INF]` = my arithmetic or inference. `[SPEC]` = my guess.
