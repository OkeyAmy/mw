# LEDGER — merged findings

Format: `[F-xx] (agent) claim — evidence — confidence`
Append-only. Only findings that passed `VALIDATION.md` enter here.

## Audit corrections (orchestrator, independently verified)

- **[X1] HARDWARE WAS WRONG.** The box is **Intel i7-8665U, 4 physical cores /
  8 threads, 1.90 GHz, AVX2 only, no AVX-512, no bf16**, ~17 GB available RAM.
  My original "8 cores" came from `nproc`, which counts logical threads.
  Independently confirmed by A4, A7, A9 and by `lscpu` / `/proc/cpuinfo`.
  *Consequence: every perf budget in the project is revised down ~2–4x.*
- **[X2] CPU IS PINNED AT 800 MHz.** `scaling_governor=powersave`,
  `intel_pstate/no_turbo=1`. Unlocking needs sudo. Expected ~2x gain.
  *Status: NOT FIXED — needs the user.*
- **[X3] TransformerLens removed `HookedTransformer` in 4.0.** Verified in
  `transformer_lens/__init__.py`: importing it raises a directed
  `ModuleNotFoundError` telling you to use `TransformerBridge.boot_transformers`.
  A2 correct. My prototype assumed the old API; it is raw-HF-based, so unaffected.
- **[X4] `tie_word_embeddings = true`** for Qwen2.5-Coder-1.5B — verified directly
  from the downloaded `config.json`. So there is no separate `lm_head.weight` in
  the checkpoint. A2 correct. Use the `model.lm_head` module, not a weight lookup.
- **[X5] Model ID is `Qwen/Qwen2.5-Coder-1.5B`** — there is no `-Base` suffix for
  the 1.5B. Verified via HF API `siblings`. A4 correct; my CONTEXT.md was wrong.
- **[X6] TL registry DOES fully verify our model.** `status: 1`,
  `verified_date: 2026-02-24`, `note: "Full verification completed"`,
  phase scores 100/100/100/90.4. A5 correct.
  *My first check reported "not found" — that was a truncated download on my side,
  not an error by A5. Lesson: validate the validator.*
- **[X7] CircuitsVis is MIT — confirmed. But A5's API names are wrong.**
  The Python package exports **snake_case** functions: `topk_tokens`,
  `colored_tokens`, `attention_heads`, `attention_patterns`, `attention_pattern`,
  `token_log_probs`. `TopkTokens` / `AttentionHeads` are React component names.
  An implementer following A5 literally would write broken code.
- **[X8] circuit-tracer transcoders: Qwen-3 yes (0.6B/1.7B/4B/8B/14B), Qwen2.5 no.**
  Confirmed from the repo README. A5 correct. Attribution graphs are unavailable
  on our model without training crosscoders (~10^3 CPU-hours).
- **[X9] Logit-lens cost: conclusion right, framing misleading.**
  FLOPs are as A5 says (13.1 GFLOP lens vs 2.6 GFLOP forward), but the forward
  pass is also bandwidth-bound (reads ~3 GB of weights/token), so "5x the model"
  is a FLOP ratio, not a wall-clock ratio. A4 measured the real thing:
  **naive per-layer lens 2.50 s/token → batched single GEMM 0.49 s/token.**
  The actionable conclusion (batch the lens) survives; the reasoning did not.

## Verified by orchestrator (independent re-derivation)

- **[V1] A1's weight inventory is exact.** Parsed the safetensors header directly:
  338 tensors, 100% BF16, **no `lm_head`** (tied), **84 QKV bias tensors**
  (28 layers x 3). Arithmetic closes: `3,087,467,144 - 1,543,714,304*2 = 38,536
  = 8 + 38,528` (the header). A1 correct on every point.
- **[V2] Q4_K_M is lossy but direction-preserving.** A1 measured 7.17% relative
  Frobenius error and 1.22 dB... (see A1 for the exact SNR), with cosine
  similarity >= 0.997. Consequence: **logit lens and steering geometry survive
  quantisation; neuron-level claims do not.** This is a usable fallback if the
  fp16 download fails.
- **[V3] The Ollama store is on a removable exFAT USB drive** that was unmounted
  mid-session. That is why `~/.ollama/models/blobs` is empty while `ollama list`
  still works (page cache). A1 and the orchestrator independently confirmed.
  *Action: mount `/run/media/okeyamy/Linux` before restarting Ollama, or the
  models may be lost.*

## PROOF CHAIN RESULTS (first real run, on Qwen2.5-Coder-1.5B, CPU)

Weights used: Q4_K_M GGUF from Ollama -> dequantised to fp16 safetensors
(`code/gguf_to_hf.py`, 338/338 tensors mapped, sha256 verified against the
Ollama manifest). This is the quantised model, so C1/C2 are exact (self-consistent)
and C3-C6 are valid for *this* artefact; they must be re-run on bf16 for publication.

- **[P1] C1 PASS.** 28 residual tensors, shape [12,1536], all finite.
  Cross-check vs HF `hidden_states`: **max diff 0.000e+00**, and
  `final_norm(resid[27]) == hidden_states[-1]` exactly.
  *Correction to a common assumption: HF's `hidden_states` has 29 entries for a
  28-layer model — embed + 27 layer outputs + the post-norm. The final decoder
  layer's PRE-norm output is not stored. Any comparison that assumes
  `hidden_states[i+1] == layer i` is wrong for the last layer.*
- **[P2] C2 PASS — exact.** `lm_head(norm(hook_resid_post(27)))` vs real logits:
  **max abs diff 0.000e+00**. The logit lens is faithful at the final layer.
- **[P3] C3 PASS.** Patching `resid_post(L)` from clean into corrupt recovers the
  target logit. Recovery by layer: L0..L20 ~ 0.00, **L21 +0.16, L22-L27 ~ +1.0**.
  Self-patch exact (0.0). Random-replacement control = -1.80 (no recovery).
  Layer-0 control = -0.001. *The answer is computed in the last ~5 layers.*
- **[P4] C4 FAILED at layer 14 — a real negative result.**
  Curve was V-shaped, not monotone: coef -3..+3 gave +2.61, +2.52, +0.78, 0,
  +1.72, +2.95, +9.55. Random vectors produced LARGER spans than the real vector.
  Interpretation: layer 14 was producing a generic perturbation direction.
  **This is exactly the AxBench/D3 warning, reproduced in our own harness.**
- **[P5] C4 PASSES at layer 8 (and 16) after a principled sweep.**
  `code/layer_sweep.py`, 12 contrast pairs, 10 random controls per layer,
  pre-registered selection = (beats_random_frac, monotone_frac).
  - L8: monotone 1.00, beats random 10/10, span 12.79 vs random median 6.62.
  - L16: monotone 1.00, beats random 10/10, span 14.29.
  - L14: monotone 0.39, non-specific. L20: 0.67. L24/L26: large but non-monotone.
  *Conclusion: "comment mode" is a specific, causally effective direction in the
  early-to-mid layers of this model. Layer choice is not a detail — it is the
  difference between a real effect and noise.*

## Business research (E-series)

- **[B1] E1: "steering replaces prompts to save tokens" is FALSE on arithmetic.**
  Output tokens cost 5-10x input; cached input costs 0.1x; a system prompt is
  0.5-6% of the bill (0.05-0.6% cached). Deleting it entirely with a perfect free
  vector saves ~$38/month at 1M requests. Also AxBench 2025 (500 concepts):
  prompt 0.905 vs diff-in-means 0.409 vs SAE 0.315 — prompting beats steering
  for instruction adherence. *Steering still does things prompts cannot
  (CAA: hallucination 0.54->0.78 while the system prompt is inert), but it is
  not a token-cost play.*
- **[B2] E2: the real token savings are elsewhere.** (1) engineered prefix
  caching, 41-80% of total cost on long agents, but botched in practice
  (32 silent mis-integrations across 571 projects); (2) grammar/schema-constrained
  decoding, measured 675->223 tokens (-67%) and 156->16 (-90%) with the real
  Qwen tokenizer; (3) tool-definition deferral, -85% of a 72K prefix and
  accuracy 49%->74%. Unclaimed ground: a compressor for the **volatile tail**
  that guarantees prefix-hash stability (query-aware compression measured
  +40.1% cost vs doing nothing, because it breaks the cache).
- **[B3] E3: kill wedges A and B.** `diffing-toolkit` (MIT, active) already does
  the fine-tune diff; and brand-voice buyers purchase outcomes, not vectors.
  Recommended wedge: a **quantisation decision-flip certificate** — fp16 vs Q4,
  report exactly which decisions flip and by what margin. Runs on the customer's
  hardware, so our 4-core laptop stops mattering. Falsification test: 5 days, $0.

## Open questions
- Q1: How fast is fp32 inference really, once the governor is fixed? (need sudo)
- Q2: Cheapest correct path from a Q4_K_M GGUF back to usable fp16 weights?
- Q3: Which layer band carries "style/intent" vs "syntax" in a 1.5B code model?
- Q4: Can an SAE be trained on 1 layer on CPU in <24h? (A2: ~25 CPU-h, one layer)
- Q5: What is the minimal causal test that proves a steering vector works?

## Confirmed findings
_(populated after D2 audit completes)_
