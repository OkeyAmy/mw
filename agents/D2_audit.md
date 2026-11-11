# D2 — Adversarial audit of A2–A9

**Role:** independent auditor. I re-derived numbers, fetched citations, read upstream source,
and looked for contradictions. I did **not** try to summarise the reports or agree with them.

**Verdict up front: 6 of 8 reports contain at least one error that changes a build decision.**
The arithmetic is mostly careful; the *throughput assumptions* and the *memory budget* are not.
Two reports (A4, A8) contradict themselves in the same paragraph or table.

---

## 0. Verification environment (so this is auditable)

Run on 2026-11-10, on the box, while the other agents' processes were running.

```
$ lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core\(s\)|L3"
Model name:  Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
CPU(s):      8      Thread(s) per core: 2      Core(s) per socket: 4
L3 cache:    8 MiB (1 instance)
$ grep -o -E 'avx[0-9a-z_]*|amx[a-z_]*' /proc/cpuinfo | sort -u
avx avx2            # no avx512*, no amx*
$ free -g
              total  used  free  shared  buff/cache  available
Mem:              31    14     4       1          14         16
$ cat /sys/.../intel_pstate/no_turbo ; cat .../scaling_governor ; cat .../scaling_cur_freq
1
powersave
799965              # 0.80 GHz = 42% of the 1.9 GHz base
```

numpy 2.5.2 / OpenBLAS, this session:

| measurement | result |
|---|---|
| GEMM 1536³ fp32, 1 thread | **80.7 GFLOPS** |
| GEMM 1536³ fp32, 2 / 4 / 8 threads | 84.2 / 76.7 / 65.2 GFLOPS |
| GEMM 512×8960×1536 (MLP-shaped), default | **73.7 GFLOPS** |
| streaming read, 256 MB, single process | **7.2 GB/s** |

So the GEMM figures in A9 (54.5–67.4) and A4 (50–70) are **correct for a contended 8-thread
run**, and slightly pessimistic for a quiet box. Bandwidth is the number that moves most:
A9 9.7 / A4 11.6 (8 procs, aggregate) / A7 12.2 (4 threads, triad) / **7.2 single-stream now**.
None of those is wrong; they are four different measurements being compared as if they were one.

---

## 1. The three errors that matter most

1. **A2 plans the whole project at 150 tok/s. The machine cannot exceed ~21 tok/s.**
   150 tok/s × 3.09 GFLOP/token = **463 GFLOPS**; the box delivers 65–83. Every CPU-hour in
   A2 is therefore ~7× optimistic, including "SAE is a weekend" (25 h → ~200 h).
2. **A4 says a 512-token prefill lands at "0.3–1 s" after computing that it costs 29 s of
   compute.** 512 × 3.09 GFLOP ÷ 65 GFLOPS = **24 s**. The "prompt length is nearly free"
   conclusion, which A4 hands to A5's interaction design, is false.
3. **A8 budgets against 28 GB of RAM; the box has 16 GB available.** Its headline
   "32B fits comfortably at 4.5 bpw" (18.45 GB) is false on the measured machine.

---

## 2. Audit table — A2 (interpretability)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A2 | "150 tok/s batched fp32 forward on 8 cores (≈4.6× the 16 tok/s single-stream)" | 150 × 3.087 GFLOP = 463 GFLOPS required; measured 65–83. Also 4.6 × 16 = **73.6**, not 150 | **WRONG** | Use 8–12 tok/s (A9/A7, batched) or ~17 tok/s roofline at batch 16. Re-scale every CPU-hour by ~1/8. |
| A2 | "62 ms/token … consistent with CONTEXT.md's 5–20 tok/s" | 3.087/50 GFLOPS = 62 ms is the *compute* bound; batch-1 is bandwidth-bound: 6.175 GB ÷ 7.2–11.6 GB/s = **0.53–0.86 s** | **MISLEADING** | 62 ms is the ceiling, not the decode time. Single-stream is 1.2–1.9 tok/s. |
| A2 | "896 forwards × 0.21 s ≈ 190 s ≈ 0.05 CPU-h" | 0.21 s/forward at T=32 needs 32×3.087/0.21 = **470 GFLOPS** | **WRONG** | Real cost 1.5–1.8 s/forward → 0.45 CPU-h batch-1. Only ~0.02 CPU-h if the 32 positions are batched (they are independent) — but the stated derivation is impossible. |
| A2 | "HF `resolve` returns 401 for Qwen repos from this host" | `GET /api/models/Qwen/Qwen2.5-Coder-1.5B` → **200** (385,980 downloads); `…-Base` → 401 | **WRONG** | The 401 is specific to the nonexistent `-Base` name, not host-wide auth. A4 is right: the unsuffixed repo *is* the base model. |
| A2 | "Coder-1.5B-Base config could not be fetched … tie_word_embeddings UNVERIFIED" | local `.hf/…/config.json`: `"tie_word_embeddings": true`; safetensors header contains only `model.embed_tokens.weight`, no `lm_head` | **CONFIRMED** (claim) / **WRONG** (the excuse) | The claim is true and was verifiable all along at the unsuffixed ID. |
| A2 | "There is no `lm_head.weight` tensor in the checkpoint; `W_U = embed_tokens.weight`" | parsed the safetensors JSON header via HTTP Range: 338 tensors, `lm_head` absent, `model.embed_tokens.weight` shape `[151936,1536]` bf16 | **CONFIRMED** | — |
| A2 | "Qwen2 has QKV biases and none elsewhere" | header inventory: per layer `q/k/v_proj.bias` present, `o_proj`/MLP biases absent | **CONFIRMED** | — |
| A2 | "`HookedTransformer.from_pretrained` was removed in 4.0 … raises `ImportError`" | read `transformer_lens/__init__.py` at main; PyPI latest = **4.3.0** | **CONFIRMED** | Source comment: bare `from transformer_lens import HookedTransformer` → `ImportError`; submodule path → `ModuleNotFoundError`. Both observations (A2's and the orchestrator's) are right. |
| A2 | "`TransformerBridge.boot_transformers(...)` is the replacement" | source: `if name == "TransformerBridge": from .model_bridge import TransformerBridge` | **CONFIRMED** | — |
| A2 | "`enable_compatibility_mode()` centers `W_U`/`W_E`" | `bridge_core.py`: signature `center_writing_weights=True, center_unembed=True, fold_ln=True` | **CONFIRMED, imprecise** | It centers `W_U` (unembed) and *writing* weights (attn/MLP `W_out`), not `W_E`. |
| A2 | "do **not** call `enable_compatibility_mode()` … then use the shipped `logit_lens`" | TL's own docstring says compat mode is "**required** for analyses that reason in HookedTransformer's post-processed coordinate system: **logit lens**, direct logit attribution, residual-stream norms" | **MISLEADING** | Internally tense: the shipped `logit_lens` reads "through the real final norm, unembedding and adapter output transform", so raw mode works — but A2 should have flagged the docstring conflict rather than asserting silent breakage. |
| A2 | "`tools.analysis` ships … `direct_path_patching`, `attribution_patching`" | `tools/analysis/__init__.py` exports `get_act_patch_direct_path`, `get_act_patch_direct_path_all_sources`, and `attribution_patch` from module `attribution_patching` | **WRONG (names)** | There is no `.direct_path_patching`. Correct names: `get_act_patch_direct_path`, `attribution_patch`. |
| A2 | `detect_head(model, seq, detection_pattern)` with `error_measure ∈ {"mul","abs"}` | `head_detector.py`: `HeadName = Literal["previous_token_head","duplicate_token_head","induction_head"]`, `error_measure: ErrorMeasure = "mul"` | **CONFIRMED** | — |
| A2 | Qwen2 adapter `_qkvo_weight_conversions(include_biases=True)` | `supported_architectures/qwen2.py` matches exactly | **CONFIRMED** | — |
| A2 | exp-16 SAE: "453 MFLOP/token → 9 ms/token → 25 h of training alone" for 30 M tokens | 9.06 ms × 30e6 = 271,800 s = **75.5 h** | **WRONG** | 25 h corresponds to ~10 M tokens. |
| A2 | SAE one layer, 10 M tokens ≈ **25 CPU-h** (18.5 h harvest) | 18.5 h assumes 150 tok/s; at 17 tok/s harvest alone is 163 h | **WRONG** | ~150–200 CPU-h at realistic batched throughput. The "do it, it's a weekend" verdict does not survive. |
| A2 | storage 10 M × 1536 × 2 B = 30.7 GB; lens 418 GFLOP at T=32 ≈ 8 s | 30.72 GB ✓; 418/50 = 8.4 s ✓ | **CONFIRMED** | — |

---

## 3. Audit table — A3 (steering)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A3 | CAA "is scaled to the *mean residual-stream norm at that layer* … The released code does this" | CAA paper §"Vector normalization choices": "we choose **not** to normalize over the layers to preserve a 'natural norm'"; repo README: "we normalize **across behaviors** for each layer" | **WRONG** | CAA normalizes vector magnitude across behaviours, not to `E‖h‖`. Coefficients are *not* "one residual-norm worth of push". The whole coef-scale narrative and the cross-paper comparability claim rest on this. |
| A3 | post-norm steering is "provably equivalent to a rank-1 logit bias `Δlogits = W_U·v`" and the code implements it | `SteeringHook` registers `target.register_forward_pre_hook(self._pre)` on `model.model.norm`, i.e. **adds `v` before RMSNorm** | **WRONG (implementation)** | Adding before a norm is not linear in `v`. To get `Δlogits = W_U·v` you must add after the norm: `register_forward_hook` returning `out + v`. As written, `assert_postnorm_equivalence` will fail. This is PROOF_CHAIN C6's mechanism. |
| A3 | "`tie_word_embeddings` … true for the 1.5B Qwen2.5 family but **UNVERIFIED** here" | local `config.json`: `true`; A2 and A4 also verified it | **OUTDATED** | Verified true. Remove the hedge. |
| A3 | extraction "200 pairs × ~120 tokens ≈ 24k token-forwards ≈ **10–20 CPU-minutes** per layer batched at 8" | 24,000 ÷ (8–12 tok/s) = 2,000–3,000 s = **33–50 min** | **WRONG (>2×)** | 33–50 min per pass, not per layer. |
| A3 | "A 5-layer sweep ≈ 1–2 h" | `output_hidden_states=True` returns **all 29 layers in one pass** | **MISLEADING** | A 5-layer sweep costs the same as a 1-layer sweep (~35–50 min). The overestimate is harmless but reveals a misunderstanding of the extraction path. |
| A3 | Tan et al. "explicitly note degradation is particularly pronounced for Qwen" | paper: "some points far above or below the x=y line, and this is more noticeable for the Qwen model" — about OOD-vs-ID scatter, on **Qwen-1.5-14b-Chat** | **MISLEADING** | Not "degradation", and not our model. The "Qwen is a hard case" prior is still reasonable, but the citation overstates it. |
| A3 | "Rimsky/Panickssery et al., 2023/ACL 2024" for CAA | arXiv 2312.06681 authors: Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner — **no Rimsky** | **WRONG (authorship)** | Panickssery et al. |
| A3 | "FASB … Gan et al., 2025, arXiv:2508.17621" | first author is **Zifeng Cheng** (Gan is 2nd) | **WRONG (authorship)** | Cheng et al. |
| A3 | CAST: Qwen 1.5 Chat 1.8B; 100% / 96.4% unconditional; 90.7% CAST | paper HTML: "Model: Qwen 1.5 Chat 1.8B"; "Refusal 100.00% 96.40%"; "90.7 ← 45.8" | **CONFIRMED** | — |
| A3 | `gen` positions leave token 1 unsteered; schedules are path-dependent through the KV cache | correct: prefill K/V computed before any decode-step edit | **CONFIRMED** | — |
| A3 | `out.hidden_states[L] == blocks.L.hook_resid_pre` | correct HF contract (index 0 = embedding output) | **CONFIRMED** | — |
| A3 | negative steering stronger than positive | not independently checked | **UNVERIFIED** | Do not put this in the ledger without a source line. |

---

## 4. Audit table — A4 (runtime)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A4 | "512-token prompt costs 512 × 3.09 GFLOP / ~55 GFLOPS ≈ **29 s** of compute … so it lands around **0.3–1 s** in practice — i.e. prompt length is nearly free" | 512 × 3.087 = 1,580 GFLOP; ÷ 65 GFLOPS = **24 s** | **WRONG (self-contradictory)** | Prefill is compute-bound at ~25–29 s. "Prompt length is nearly free" is false and propagates into A5's interaction design. A6's 1.2 s per 20-token forward agrees with me, not with A4. |
| A4 | `Qwen/Qwen2.5-Coder-1.5B-Base` does not exist; real ID is `Qwen/Qwen2.5-Coder-1.5B` | HF API: `-Base` → 401; unsuffixed → 200, single `model.safetensors` | **CONFIRMED** | — |
| A4 | 1,543,714,304 params | derived: 28×(5,505,024+41,287,680+3,072+2,048) + 233,373,696 + 1,536 = **1,543,714,304** | **CONFIRMED (exact)** | — |
| A4 | fp32 6.17 GB; lens 0.933 GB; 28 lenses = 26.1 GB; 13.07 GFLOP = 4.2× model | 6.175 GB ✓; 933.5 MB ✓; 26.14 GB ✓; 13.07/3.087 = 4.23 ✓ | **CONFIRMED** | — |
| A4 | measured lens: 1 layer 94 ms, naive 2504 ms, stacked-28 493 ms, stacked-7 283 ms | 493 ms → 13.07/0.493 = 26.5 GFLOPS at 0.8 GHz — consistent | **CONFIRMED** | — |
| A4 | "fix the governor … it is worth ~2x" | DRAM bandwidth does not scale with core clock; triad already 12.2 GB/s at 0.9 GHz vs ~38 GB/s theoretical | **MISLEADING** | Expect 1.2–1.4× on decode, not 2×. A4's own "15–20 GB/s at full clock" is a 1.3–1.7× claim and contradicts its "2×" headline. |
| A4 | "decode ~2–3 tok/s at full clock (≈1–2 as configured)" | 6.175/11.6 = 532 ms → 1.88 tok/s now; 6.175/17 = 363 ms → 2.76 | **CONFIRMED** | — |
| A4 | "bf16/fp16 CPU matmul is 4×–50× slower than fp32" | upstream issues not re-checked; direction corroborated by A9/A7 | **UNVERIFIED (plausible)** | Fine as a prior; do not quote the 50× without the issue link in hand. |
| A4 | transformers 5.22.2, torch 2.15.0, numpy 2.5.4, fastapi 0.162.4 | not verifiable from this host (pip index not reachable); local env is numpy 2.5.2 | **UNVERIFIED** | Pin at install time, not from the report. |
| A4 | "`Qwen2DecoderLayer.forward` returns a bare tensor (line 290)" | version-specific line number not checked | **UNVERIFIED** | The defensive `isinstance(out, tuple)` advice is correct regardless. |
| A4 | attention at T=512 = 352 MB | 28×12×512²×4 B = 352.3 MB ✓ (fp32) | **CONFIRMED** | A5's 176 MB is the fp16 figure — both right, dtype unstated in both. |

---

## 5. Audit table — A5 (UI)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A5 | batched lens "one weight pass, ~467 MB → **~25–60 ms**" | arithmetic intensity = 13.07 GFLOP / 0.467 GB = **28 FLOP/byte** > ridge ~9 → compute-bound; 13.07/0.065 = **201 ms** (157 ms at 83 GFLOPS). A4 measured 493 ms at 0.8 GHz. | **WRONG (4–10×)** | The batched lens is *not* free: ~160–250 ms at full clock. This is why A4's 8-layer subset (≈57–90 ms) is mandatory rather than an optimisation. |
| A5 | "**Lens ≈ 5× the model**" (TL;DR) | 13.07/2.62 = 4.99 by FLOPs, but 13.1 GB / 3.09 GB = **4.2×** by bytes, and the box is bandwidth-bound | **MISLEADING** | Say "4.2× the model's bytes; ~5× its FLOPs". The 19% gap matters because the paragraph uses the 5× as a *time* argument. |
| A5 | "there are **no public SAEs for the 1.5B Qwen2.5 family**" | HF model search: ≥15 hits, incl. `huypn16/sae-qwen-2.5-coder-codeforces-8x` (all 28 layers, exp 8, k=64), `Resa-Yi/Pre-trained-SAE-Qwen2.5-1.5B-65k`, `ilyasoulk/sae-qwen2.5-1.5b-layer20-exp16`, `Solshine/…qwen25-1p5b` | **WRONG** | SAEs exist, including for **Coder**. "Don't ship a features panel in v1" may still be right, but not for this reason. |
| A5 | "`HookedTransformer` gives us named hook points (`blocks.{L}.hook_resid_post` …)" | `transformer_lens/__init__.py`: HookedTransformer **removed in 4.0**, directed `ImportError` | **WRONG** | Use `TransformerBridge.boot_transformers` (A2/A4 correct). A5's whole "~50-line hook loop" plan names a dead API. |
| A5 | exBERT = arXiv:2010.11994 | that ID is **"Thresholded Lasso Bandit"** (Ariu, Abe, Proutière) | **WRONG (citation)** | exBERT is arXiv:1910.05276. |
| A5 | Neuronpedia `qwen2.5-1.5b-it` has `sourceSets: []` | `POST /api/model/list`: entry exists, `visibility: UNLISTED`, `layers: 28`, `defaultSourceSetName: null`, `defaultSourceId: null` | **CONFIRMED (substance) / WRONG (field)** | There is no `sourceSets` field in that payload. The conclusion (no SAE set) holds. |
| A5 | Transformer Explainer, "CHI 2026" | arXiv v2 revised 2026-08; venue not stated on the abs page | **UNVERIFIED** | Likely CHI 2025 (v1 Aug 2024). Check before citing. |
| A5 | LLM-TT licence CC BY-NC 4.0 | `facebookresearch/llm-transparency-tool/LICENSE`: "Attribution-NonCommercial 4.0 International" | **CONFIRMED** | — |
| A5 | Neuronpedia Apache-2.0; tuned-lens MIT; Transformer Explainer MIT; LIT Apache-2.0; circuit-tracer MIT | each LICENSE file fetched and read | **CONFIRMED** | — |
| A5 | circuitsvis MIT, exports `TopkTokens`/`AttentionPattern`/… | LICENSE and `package.json` both 404 on `main` and `master` | **UNVERIFIED** | The claim is probably right but was not demonstrable at the path given. |
| A5 | 176 MB attention at T=512; 14,336 heatmap cells; 512×28×4 = 57 KB ImageData; 1.5 GB int8 for 1.5B | 28·12·512²·2 = 176.2 MB ✓; 28×512 = 14,336 ✓; 57,344 B ✓; 1.5e9 B ✓ | **CONFIRMED** | — |

---

## 6. Audit table — A6 (limits)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A6 | tuned lens is "1536 × 151936 ≈ **233 M params per layer** × 28 = **6.5 B** … not a CPU workload" | `tuned-lens` source `tuned_lens/nn/lenses.py`: `translator = th.nn.Linear(config.d_model, config.d_model)` — **d→d**, then the model's own unembed | **WRONG (~99×)** | 1536² + 1536 = **2.36 M/layer, 66 M total**. A2's figure is right. The tuned lens *is* affordable; A6's "skip it" advice is based on a false cost. |
| A6 | "**Gupta** et al. 2025, arXiv:2502.16681" | authors: **Kantamneni**, Engels, Rajamanoharan, Tegmark, Nanda | **WRONG (authorship)** | Kantamneni et al. A9 cites it correctly. |
| A6 | "Does Circuit Analysis Interpretability Scale?, **Conmy** et al., 2023, arXiv:2307.09458" | authors: **Lieberum**, Rahtz, Kramár, Nanda, Irving, Shah, Mikulik | **WRONG (authorship)** | Lieberum et al. The later sentence "Conmy et al. reached Chinchilla 70B" inherits the error. |
| A6 | "Extending Activation Steering…, **Stolfo** et al., 2024, arXiv:2403.05767" | authors: **van der Weij, Poesio, Schoots** | **WRONG (authorship)** | van der Weij et al. |
| A6 | check 2: "logit lens at last layer == real logits, max \|Δ\| < **1e-3**" | PROOF_CHAIN C2 requires "**max abs diff < 1e-4** (must be exact, not approximate)" | **WRONG (inconsistent)** | A6's own gate is 10× looser than the project's acceptance test. |
| A6 | n=100, α=0.05/56, power 0.8 ⇒ MDE d_z ≈ 0.42 | (z_{8.93e-4/2} + z_0.8)/√100 = (3.33+0.84)/10 = **0.417** | **CONFIRMED (exact)** | — |
| A6 | ACDC: 400 components ⇒ 1.6e5 forwards ≈ 53 CPU-h; 10³ pairs ≈ 6 CPU-years | 400² = 160,000 × 1.2 s = 53.3 h ✓; ×1,000 = 6.1 yr ✓ | **CONFIRMED** | — |
| A6 | "A forward at 1.5B / 20 tokens / fp32 / 8 cores ≈ **1.2 s**" | 20 × 3.087 = 61.7 GFLOP ÷ 55 GFLOPS = 1.12 s | **CONFIRMED** | This is the *correct* forward-cost figure; A2 and A4 should adopt it. |
| A6 | 151,936 directions in 1536 dims = "99× overcompleteness" | 151936/1536 = 98.9 | **CONFIRMED** | — |
| A6 | "Not All Language Model Features Are **One-Dimensionally** Linear" (title correction) | arXiv 2405.14860 title matches exactly | **CONFIRMED** | A6's correction is right. |
| A6 | "An Interpretability Illusion for BERT" 2104.07143; Leask 2502.04878; Chanin 2409.14507; 2501.17727 | all four titles and IDs match; 2501.17727 abstract confirms random-vs-trained parity | **CONFIRMED** | — |

---

## 7. Audit table — A7 (CPU scaling)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A7 | "Kumar et al. 2022, arXiv:2212.09720" (35k experiments, 4-bit optimal) | arXiv 2212.09720 = **"The case for 4-bit precision"**, authors **Tim Dettmers, Luke Zettlemoyer** | **WRONG (authorship)** | Dettmers & Zettlemoyer, ICML 2023. A8 makes the identical error — the same bad citation propagated. |
| A7 | "usable budget = 31 − 3 GB OS = 28 GB" | `free -g`: 16 GB available, 14 GB already used | **WRONG (~75%)** | Budget against 16 GB unless the desktop is closed. |
| A7 | "**Everything up to 32B fits in 31 GB**" (TL;DR) | 32B Q4_K_M = 19.9 GB > 16 GB available | **MISLEADING** | True of total RAM, false of available RAM. A7 then says "almost none of it is usable", which softens it, but the headline is the sentence people will quote. |
| A7 | "fixing clocks … worth roughly **2×**" | bandwidth-bound decode; triad already 12.2 GB/s at 0.9 GHz | **MISLEADING** | 1.2–1.4× is defensible. Same overstatement as A4. |
| A7 | KV/token: 28/56/192/256/320 KiB; 32k = 0.94/1.88/6.44/8.59/10.74 GB | recomputed 2·L·H_kv·d_head·ctx·2 B for every row | **CONFIRMED (all rows)** | — |
| A7 | nominal bpw: Q2_K 2.5625, Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625, Q8_0 8.5 | matches llama.cpp block structs | **CONFIRMED** | — |
| A7 | 6.37 tok/s for 0.99 GB Q4_K_M ⇒ 6.3 GB/s effective | 0.99×6.37 = 6.31 GB/s ✓ | **CONFIRMED** | — |
| A7 | 14B fp16 = 29.6 GB | 14.7e9 × 2 = 29.4 GB ✓ | **CONFIRMED** | — |
| A7 | "70B never fits" | 70.6e9 × 2.0625/8 = 18.2 GB (IQ2 nominal) but the real file is 25.5 GB; either way > 16 GB available | **CONFIRMED** | — |

---

## 8. Audit table — A8 (guided compression)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A8 | "Usable RAM after OS + PyTorch/llama.cpp overhead: **~28 GB**" | `free -g` → **16 GB available** | **WRONG** | Everything in F11's fit table is computed against the wrong budget. |
| A8 | TL;DR: "**32B fits**, 70B does not … 32B @ 4.5 bpw ≈ 18 GB + ~1 GB KV → comfortable" | 32.8e9 × 4.5/8 = 18.45 GB; +KV → 19.5 GB > 16 GB available | **WRONG** | 32B@4.5 does not fit while the desktop is running. 32B@3.5 (14.35 GB) is the only variant with headroom, and A8 itself calls its quality UNVERIFIED. |
| A8 | "KV cache for a 32B GQA model at 8k ctx ≈ **1 GB**" | 2·64·8·128·8192·2 = **2.15 GB** f16 (A7's table agrees) | **WRONG (2×)** | 1 GB is the q8_0 figure; unstated. |
| A8 | "Kumar et al. 2022/ICML'23, arXiv:2212.09720" | Dettmers & Zettlemoyer | **WRONG (authorship)** | Same error as A7. |
| A8 | "per-layer sensitivity-guided mixed precision is NOT novel" — HAWQ 2019 → ZeroQ → BRECQ → LLM-MQ → Zhang et al. ICLR 2026 | all five IDs resolve to the named papers with the right years: 1905.03696, 2005.03642, 2102.05426, 2410.13056, 2504.02010 | **CONFIRMED** | — |
| A8 | Zhang et al. "protecting just **2%** of all weights … raise average accuracy by **6.57%**"; "dynamically quantized **2.51-bit** R1 reaches close-to-R1"; "difference of means and attribution patching … every linear component" | paper HTML, three separate passages match almost verbatim | **CONFIRMED** | A8's central "this is already published" claim is accurate. |
| A8 | Gromov: "Llama-2 45–55%, Mistral 35%, Phi-2 25%, Qwen **20%**" | paper: "45%-55% … 35% for Mistral 7B, 25% for Phi-2, and **20% for models from the Qwen family**" | **CONFIRMED (verbatim)** | Caveat A8 already states: Qwen-1, not Qwen2.5. |
| A8 | ShortGPT: XSum 19.40 → 0.67 at 28% removal | both numbers appear in the ShortGPT tables | **CONFIRMED (direction)** | Exact row not isolated; low risk. |
| A8 | "uniform 4-bit leaves only **12–15%** of SAE features intact (Duan 2026)" | Duan abstract: "INT7 improves perplexity while degrading 18.7% of active SAE features"; "62.4% / 51.3% survive at INT6" | **CONFIRMED (direction), UNVERIFIED (INT4 %)** | The INT4 14.5/12.4% row is body data I could not fetch. Do not quote those two numbers without the table. |
| A8 | Gupte et al. 2026: "middle layers are significantly more sensitive to pruning than early or late layers" | arXiv 2608.25941 abstract: exact sentence present | **CONFIRMED (verbatim)** | — |
| A8 | CGC 2026: "orthogonal to Wanda importance (Spearman −0.054 over 384 heads)", "negative result on GPT-2 Medium" | arXiv 2603.16440 abstract: "Spearman rho = -0.054, n = 384 heads"; "negative result on PPL-based compression comparison"; "GPT-2 Medium as an insufficient test bed" | **CONFIRMED** | Author is Rishaank **Gupta** (CGC) — distinct from **Gupte** (pruning). A8 keeps them apart correctly. |
| A8 | MLP = 88.2% of block params; attention 11.8%; embedding = 15% of 1.5B | 41.29/46.79 = 88.24% ✓; 233.4/1543.7 = 15.1% ✓ | **CONFIRMED** | — |
| A8 | "70B×2.1 bpw = 32B×4.6 bpw = ~147 Gbit" | 70e9×2.1 = 147e9 bits; 32e9×4.6 = 147.2e9 ✓ | **CONFIRMED** | — |
| A8 | E2 layer-ablation sweep = **55 min** (28 × 32 × 96 tok) | 896 forwards × 96 tok × 3.087 GFLOP = 265 TFLOP ÷ 65 GFLOPS = **68 min** (batch-1 96-token forwards) | **WRONG (~25%)** | ~1.1–1.3 h. Minor, but it is one of nine rows that sum to "fits in a day". |
| A8 | vocab/lens and 2 in Llama-2-7B at dims 1415, 2533 | Massive Activations paper: "two fixed feature dimensions (1415, 2533)" | **CONFIRMED** | — |

---

## 9. Audit table — A9 (neuron atlas)

| agent | claim | check performed | verdict | correction |
|---|---|---|---|---|
| A9 | "hard compute cap = 40/3.08 = **13 tok/s**" | A9 itself measured 54.5–67.4 GFLOPS; 67.4/3.087 = **21.8**; my run: 83/3.087 = **26.9** | **WRONG** | The cap is **17–27 tok/s**. A9 contradicts its own measurement by using 40 GFLOPS. |
| A9 | "full 28-layer forward, batch 1, fp32 ⇒ **0.54 s/token ≈ 1.9 tok/s**" | 6.175 GB ÷ 9.7 GB/s = **0.637 s → 1.57 tok/s**. 0.54 s requires 11.4 GB/s | **WRONG (internal inconsistency)** | Either 9.7 GB/s → 1.57 tok/s, or 11.4 GB/s → 1.85. A9 silently switches between the two. |
| A9 | "~3.3 s/neuron (batched) → … 343 h = 14.3 days" for all 250,880 | 343 h × 3600 ÷ 250,880 = **4.92 s/neuron**; and 158 s/pass ÷ B=32 = 4.94 | **WRONG (1.5×)** | 4.9 s/neuron. The 343 h figure is the consistent one; the 3.3 s is not. |
| A9 | "Literature sparsity figures (≈90% for ReLU-ified Llama, **Liu et al. 2023**, arXiv:2312.11514)" | arXiv 2312.11514 = **"LLM in a flash: Efficient LLM Inference with Limited Memory"**, Alizadeh et al. | **WRONG (wrong paper)** | That ID has nothing to do with sparsity. Cite the actual sparsity paper or drop the claim. |
| A9 | "vocabulary promotion … 1.2×10¹⁴ FLOP … one big dense GEMM, so it runs near peak, 35–50 min" | 250,880 × 1536 × 151,936 × 2 = 1.171e14 ✓; ÷ 54 GFLOPS = 36 min ✓ — **but** the output is 250,880 × 151,936 floats = **152 GB** | **MISLEADING** | Must chunk over neurons (e.g. 1024 at a time → 622 MB). As written it OOMs on a 31 GB box. |
| A9 | batch 8–16 ⇒ **8–12 tok/s** | batch-16 roofline: bytes 6.175/16 = 0.386 GB ÷ 7.2 GB/s = 54 ms → 18.6 tok/s; compute 47 ms → 21 | **CONFIRMED (conservative)** | 8–12 is a safe planning number; the ceiling is ~18. Do not "correct" it upward for a thermally-throttled 15 W part. |
| A9 | Gurnee et al.: "**1–5%** of neurons are universal" | arXiv 2401.12181 abstract: "find that **1-5% of neurons are universal**" | **CONFIRMED (verbatim)** | — |
| A9 | "Bills et al.: 0.33% of GPT-2 small's neurons got explanations > 0.8" | OpenAI neuron-explainer page not fetched | **UNVERIFIED** | 1,000/307,200 = 0.33% is arithmetically consistent; the 1,000 figure is unverified. |
| A9 | "Qwen-Scope covers Qwen3 / Qwen3.5 only — not Qwen2.5" | arXiv 2605.11887: "14 groups of SAEs across 7 model variants from the **Qwen3 and Qwen3.5** series" | **CONFIRMED (verbatim)** | — |
| A9 | "community SAE for Qwen2.5-1.5B-Instruct (HuggingAnalist/…), 16× = 24,576, 100 M tokens, layers 17–21" | repo exists (Apache-2.0, `layer_17/18/19/21`); width and token count not checked | **CONFIRMED (existence) / UNVERIFIED (width, tokens)** | — |
| A9 | "Gemma Scope widths 16,384 / 65,536 at d_model 2304 ≈ 7× / 28×" | 16384/2304 = 7.11; 65536/2304 = 28.4 | **CONFIRMED** | — |
| A9 | 250,880 = 28 × 8960; class matrix 100 × 250,880 fp16 = 50 MB; atlas < 250 MB | 28×8960 = 250,880 ✓; 50.2 MB ✓ | **CONFIRMED** | — |
| A9 | hardware: i7-8665U, 4P/8T, AVX2 only, 8 MB L3, ~15 GB free, 1.9 tok/s batch 1 | `lscpu`, `/proc/cpuinfo`, `free -g` all match; tok/s matches at 11.4 GB/s | **CONFIRMED** | This is the most accurate hardware section of any report. |

---

## 10. Contradictions between agents (highest-value output)

| # | topic | who says what | adjudication |
|---|---|---|---|
| C-1 | **Model ID** | A4: `-Base` does not exist, use `Qwen/Qwen2.5-Coder-1.5B`. A2: "HF returns 401 for Qwen repos from this host". CONTEXT/A3/A5/A6/A8/A9 all use `-Base`. | **A4 correct.** `-Base` → 401; unsuffixed → 200. Six reports and CONTEXT carry the wrong ID. |
| C-2 | **`tie_word_embeddings`** | A2: true for 1.5B/Instruct, UNVERIFIED for Coder. A3: UNVERIFIED. A4: true. | **True.** Local `config.json` and the safetensors header confirm. A2/A4 right; A3 stale. |
| C-3 | **`HookedTransformer`** | A2 & A4: removed in 4.0. A5: "`HookedTransformer` gives us named hook points". | **A2/A4 correct.** A5's UI plan names a removed API. |
| C-4 | **TL model registry** | A2: tested list has Qwen2.5-1.5B and -Instruct, **not** Coder. A5: Coder-1.5B is "status 1, full verification completed, 2026-02-24". | **Unresolved.** `supported_models.json` is not at any path I could find in the current tree. Both are UNVERIFIED. Do not put either in the ledger. |
| C-5 | **Public SAEs** | A5: "no public SAEs for the 1.5B Qwen2.5 family". A2: Solshine repo + FAST suite. A9: HuggingAnalist repo. A8: "community SAEs for Qwen2.5-1.5B-Instruct". | **A2/A9/A8 correct; A5 wrong.** ≥15 public SAEs on HF, including a Coder set. |
| C-6 | **Tuned-lens size** | A2: 2.36 M params/layer (66 M total), diagonal-first. A6: 233 M/layer (6.5 B total), "not affordable". | **A2 correct** (repo: `nn.Linear(d_model, d_model)`). A6's conclusion is void. |
| C-7 | **Usable RAM** | A8 & A7: ~28 GB. A9: ~15 GB. CONTEXT: ~17 GB. | **16 GB measured.** A8/A7 wrong; A9/CONTEXT right. |
| C-8 | **Throughput** | A2: 150 tok/s batched. A9: 8–12 batched / 1.9 batch-1. A4: 2–3. A7: 6.4 (Q4 llama.cpp). A6: 1.2 s per 20-token forward (~17 tok/s prefill). | **A2 wrong by ~8×.** A6's forward cost is the only figure that survives a roofline check. |
| C-9 | **512-token prefill** | A4: "0.3–1 s … prompt length is nearly free". A6: 1.2 s for **20** tokens. | **A6 correct.** Prefill is ~25–29 s for 512 tokens. A4's sentence contradicts A4's own arithmetic. |
| C-10 | **32B@8k KV** | A8: ~1 GB. A7: 2.15 GB. | **A7 correct** (2.147 GB f16). A8 quoted the q8_0 figure. |
| C-11 | **70B feasibility** | A7: 25.5 GB, "does not fit". A8: 70B@2.5 = 21.9 GB, "yes". | Both assume >21 GB. **With 16 GB available, neither fits.** A7's conclusion is right, its budget is wrong. |
| C-12 | **Lens/model ratio** | A5: 5×. A4: 4.2×. | Both right, different denominators (2.62 vs 3.09 GFLOP). A5's 5× is FLOPs; the byte ratio is 4.2×. |
| C-13 | **Attention memory at T=512** | A5: 176 MB. A4: 352 MB. | Both right — fp16 vs fp32. Neither states the dtype. |
| C-14 | **Acceptance threshold for C2** | A6 check 2: <1e-3. PROOF_CHAIN: <1e-4. | **PROOF_CHAIN wins** (it is the gate). A6 must match it. |
| C-15 | **Same paper, two authors** | A6: "Gupta et al." (2502.16681). A9: "Kantamneni et al." (2502.16681). | **A9 correct.** |
| C-16 | **Same paper, two authors** | A7 & A8: "Kumar et al." (2212.09720). | **Both wrong** — Dettmers & Zettlemoyer. |

---

## 11. UNVERIFIED / SPECULATION presented as fact

These appear in TL;DR or conclusion sections but are not established:

| agent | claim in TL;DR/conclusion | status |
|---|---|---|
| A2 | "150 tok/s batched" | **wrong**, and it is the planning constant |
| A2 | "SAE … ≈25 CPU-h … do it" | wrong by ~8× |
| A3 | "coef = 1 is a large perturbation" | **wrong** (CAA does not normalise that way) |
| A4 | "prompt length is nearly free" | **wrong** |
| A5 | "only `circuitsvis` is a genuine drop-in" | licence/API unverified at the path given |
| A5 | "no public SAEs" | **wrong** |
| A6 | "tuned lens is not affordable" | **wrong** (cost model wrong) |
| A7 | "~2× from fixing clocks" | estimate, not measured; probably 1.2–1.4× |
| A8 | "32B fits comfortably" | **wrong** on 16 GB available |
| A8 | "uniform 4-bit leaves 12–15%" | direction confirmed, exact figures unverified |
| A9 | "hard compute cap 13 tok/s" | **wrong** (17–27) |
| A9 | "exhaustive ablation 343 h" | consistent; the 3.3 s/neuron input is not |

---

## 12. Ranked top-10 errors

1. **A2 — 150 tok/s planning constant.** 463 GFLOPS required vs 65–83 available. Every A2 CPU-hour
   figure is ~7× optimistic; the SAE goes from "a weekend" to "a week and a half".
2. **A4 — 512-token prefill "0.3–1 s".** The arithmetic in the same sentence says 29 s. This is
   the number A5's pause/preview interaction design is built on.
3. **A8 — 28 GB RAM budget.** 16 GB measured. "32B fits comfortably" is false; the report's
   headline recommendation does not hold on the actual machine.
4. **A6 — tuned lens 233 M params/layer.** 99× too high; the "not affordable, skip it" verdict is
   wrong and would delete a cheap, useful tool (66 M params total).
5. **A5 — batched lens 25–60 ms.** Compute-bound at 157–250 ms. The UI cannot run a 28-layer
   lens at 10 Hz; the 8-layer subset is mandatory, not an optimisation.
6. **A3 — CAA residual-norm normalisation.** False in the paper and in the released code. Every
   coefficient interpretation inherits the error.
7. **A3 — post-norm hook adds `v` before RMSNorm.** The "steering == rank-1 logit bias" bridge
   (PROOF_CHAIN C6) is not implemented as described and its own assertion test would fail.
8. **A5 — "no public SAEs for the 1.5B Qwen2.5 family".** False, including a Coder set. Changes
   the v1 features-panel decision and A8's audit plan.
9. **A9 — 13 tok/s compute cap.** Contradicts A9's own 54.5–67.4 GFLOPS measurement; the real cap
   is 17–27, which changes the harvest budget by ~2×.
10. **Citation-attribution cluster (6 papers):** 2212.09720 called Kumar (A7+A8) → Dettmers;
    2502.16681 called Gupta (A6) → Kantamneni; 2307.09458 called Conmy (A6) → Lieberum;
    2403.05767 called Stolfo (A6) → van der Weij; 2312.11514 called a sparsity paper (A9) →
    "LLM in a flash"; exBERT cited as 2010.11994 (A5) → "Thresholded Lasso Bandit".
    This is the exact failure mode `VALIDATION.md` rule 1 exists to prevent.

**Honourable mentions:** A5 building on a removed `HookedTransformer`; A4's "2× from the
governor"; A8's 1 GB KV for 32B; A6's 1e-3 vs 1e-4 lens tolerance; A2's wrong `tools.analysis`
function names.

---

## 13. What this means for the build

**Do first, before any other code:**

1. **One measured throughput table, owned by A4, quoted by everyone.** Batch 1 / 4 / 8 / 16,
   prefill at 32 / 128 / 512 tokens, at the current governor *and* after `cpupower`. Until it
   exists, every CPU-hour in every report is ±8×. My prediction: **1.5–2 tok/s decode,
   ~17–21 tok/s batched ceiling, ~25 s per 512-token prefill.**
2. **Adopt the roofline as a gate.** Compute ceiling = 3.087 GFLOP/token ÷ 65 GFLOPS ≈ 21 tok/s.
   Bandwidth floor (batch 1) = 6.175 GB ÷ measured GB/s. Any claimed tok/s outside
   [bandwidth floor, compute ceiling] is a bug, not a result.
3. **Budget memory from `free -g`, not from 31 GB.** 16 GB available. That kills 32B as a target
   while the desktop runs, and it means A8's compression study should be re-scoped to
   8B→5B (its own F11 already says this is the better target).

**Design consequences:**

4. **Prompts must be short.** ~25 s per 512 tokens means the interaction budget is
   *prompt* length, not output length — the opposite of A4's advice. Cap prompts at 128 tokens
   (~6 s) and keep generations short.
5. **The lens is a 2-tier feature.** 8 layers live (~60–90 ms), 28 layers offline (~200–500 ms).
   A5's "batch it and stream" is right in spirit and wrong in budget.
6. **Keep the tuned lens.** It is 66 M params, not 6.5 B. A2's diagonal-first plan is the
   correct cheap version; A6's "skip it" was based on a wrong cost.
7. **Fix the post-norm hook before claiming C6.** `register_forward_hook` on `model.model.norm`
   returning `out + v`, not a pre-hook. Then assert `logits == base + W_U·v` numerically.
8. **Re-derive steering-vector scaling locally.** Do not inherit CAA's coefficient semantics;
   measure `E‖h^(L)‖` and report `‖Δh‖/‖h‖` (A3's recommendation is still right even though its
   premise was wrong).
9. **Evaluate a public SAE before training one.** `huypn16/sae-qwen-2.5-coder-codeforces-8x`
   (28 layers, exp 8, k=64) is a free baseline; so is `Solshine/…qwen25-1p5b` (k=32, L14/16/21).
   A2's 25 h training estimate is really ~150–200 h.
10. **Add a citation gate to the ledger.** Six misattributions in eight reports. The check is
   mechanical: resolve the arXiv ID, compare title *and* first author, fail closed. `VALIDATION.md`
   already requires it; nothing enforced it.

**What survives the audit intact:**

- A2's mechanistic content (logit lens, DLA, patching, path patching, SAE math, GQA/RoPE traps,
  the TransformerLens 4.x migration, the tied-embedding trap) is correct and verified against
  source.
- A6's statistical protocol is the strongest artefact in the project: the d_z = 0.42 power
  calculation, the 8 controls, the certificate gate, and the golden-model idea are all sound.
- A7's KV-cache and quantisation arithmetic is exact in every row I checked.
- A9's hardware section and A8's compression-prior-art section (including the "already
  published" verdict) are accurate and well-cited.
- A4's measured lens benchmark is the single most useful number anyone produced.

---

## 14. What I could not verify

- Body-text claims in arXiv HTML that timed out: the INT4 survival row in Duan 2026; the exact
  ShortGPT row; "negative steering is stronger" (A3); "1,000 neurons > 0.8" (A9); "CHI 2026"
  for Transformer Explainer (A5); TransformerLens' `supported_models.json` (A5 vs A2).
- Package versions in A4 (`transformers 5.22.2`, `torch 2.15.0`, …): no package index reachable
  from this host.
- Whether `transformers` v5.22.2 really returns a bare tensor from `Qwen2DecoderLayer.forward`.
- CircuitsVis licence/API: `LICENSE` and `package.json` 404 on both `main` and `master`.
- All timing figures are under contention (load average ≈ 14). Absolute tok/s will improve on an
  idle box; the *ratios* I corrected (compute vs bandwidth, FLOPs vs bytes) will not.
