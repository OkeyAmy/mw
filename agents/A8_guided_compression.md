# A8 — Interpretability-guided compression: can the microscope decide where the bits go?

Scope: use measured activations / attribution / per-layer sensitivity to allocate precision and
decide what to delete, instead of blind uniform quantization. Question: *can the microscope tell us
how to shrink a model so a large one fits on this CPU box without destroying it?*

All cost figures assume the hardware in `CONTEXT.md` (8 cores, 31 GB, no GPU). Anything that needs
a GPU is marked **[GPU]**. Anything I could not verify is marked **UNVERIFIED**. Anything that is my
inference rather than a published result is marked **SPECULATION**.

---

## TL;DR (5 bullets)

- **The only proven lever is activation-aware weight quantization, and it already ships in the CPU
  toolchain.** AWQ's mechanism (0.1–1% salient channels identified by *activation* magnitude, then
  per-channel rescaling so the quantizer's group range stops being set by the outlier) is ~40 lines
  of PyTorch and is exactly what llama.cpp's `imatrix` approximates (it weights by `E[a_j²]`). This is
  where 90% of the achievable win is. Everything else on my list is 1–5% on top.
- **Per-layer sensitivity-guided mixed precision is NOT novel.** HAWQ (2019, Hessian) → ZeroQ (2020,
  data-free Pareto) → BRECQ (2021, block reconstruction) → LLM-MQ (2023, first-order + integer
  programming) → **Zhang et al., ICLR 2026, which uses attribution patching + difference-of-means to
  locate critical weights and gets +6.57% average accuracy by protecting the 2% of weights that
  quantizers over-compress.** Our honest novelty is narrower and is stated in F7.
- **Layer pruning is a trap for a code model.** Llama-2 tolerates 45–55% of layers removed *on
  multiple-choice QA*, but the Qwen family's phase transition is at **20%**, and ShortGPT reports
  generative tasks (XSum 19.40 → 0.67, GSM8K → ~0) collapsing at 25–28% even though the average
  multiple-choice score stays at ~92%. Code generation is a generative task. Budget: ≤2–3 of 28 layers.
- **MoE-ification, BTX, MoD and early exit do not reduce resident bytes.** Experts and skipped layers
  still occupy RAM; they reduce FLOPs and bandwidth, not capacity. On this box the binding constraint
  is capacity. Upcycling/BTX/MoD additionally need GPU-cluster training (100B-token scale) and are
  simply unavailable to us.
- **Arithmetic verdict: 32B fits, 70B does not.** 32B @ 4.5 bpw ≈ 18 GB + ~1 GB KV → comfortable, at
  Q4_K_M-to-Q5 quality. 70B needs ≤3.0 bpw to fit (~26 GB) and at equal total bits a 32B@4.6bpw beats
  a 70B@2.1bpw (Kumar et al. 2022/ICML'23). **Do not chase 70B on 31 GB.** And: uniform 4-bit leaves
  only 12–15% of SAE features intact (Duan 2026), so the microscope must stay bf16 and any compressed
  model needs a feature audit, not a perplexity check.

---

## Findings

### F1. Activation outliers are the whole story — and the fix is a diagonal rescale
[source: *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, Lin et al., 2023, https://arxiv.org/abs/2306.00978 · *SmoothQuant*, Xiao et al., 2022/ICML'23, https://arxiv.org/abs/2211.10438 · *GPTQ*, Frantar et al., 2022/ICLR'23, https://arxiv.org/abs/2210.17323 · *SpQR*, Dettmers et al., 2023, https://arxiv.org/abs/2306.03078 · *LLM.int8()*, Dettmers et al., 2022, https://arxiv.org/abs/2208.07339 · *Massive Activations in LLMs*, Sun et al., 2024/COLM, https://arxiv.org/abs/2402.17762]

**Mechanism, precisely.** Uniform asymmetric group quantization with group size `g`:
`Δ = (max−min)/(2^b−1)`, `q = clip(round(W/Δ) − z)`, `Ŵ = Δ(q + z)`, and per-group MSE ≈ `Δ²/12`.
The range is set by the *largest* element in the group, so one element 100× the median inflates `Δ`
by ~100× and every normal element rounds to ~0. That is the entire failure mode of naive 4-bit.
Empirically this is not diffuse: `LLM.int8()` finds ~0.1% of *feature dimensions* carry the outliers;
`Massive Activations` shows a handful of channels (2 in Llama-2-7B: dims 1415, 2533) reach ~1000×
the median and act as implicit attention biases — i.e. they are functionally load-bearing, not noise.

**AWQ's fix, and why it is not just "keep outliers in fp16".** Keeping 1% of weights in fp16 is
hardware-hostile (mixed per-channel types) and, measured on OPT-6.7B INT3/g128, gives WikiText PPL
43.2 (RTN) → 13.0. AWQ instead applies an *equivalent* per-input-channel transform
`W → W·diag(s)`, `X → diag(s)⁻¹·X` (output unchanged), so the group range is no longer dominated by
the salient channel. Their derivation: error for channel `j` scales as `Δ'·|a_j|/s_j`; if the salient
weight dominates its group then `Δ' ≈ s_j·|w_j|` and the error is unchanged, but if it does not, `Δ'`
stays put and the error drops by `1/s_j`. `s` is grid-searched over a small set (α ∈ {0, 0.5, 1} in
practice) to minimise `‖Q(W·diag(s))·diag(s)⁻¹X − WX‖`. **No backprop, no reconstruction** — this is
why it generalises across domains instead of overfitting the calibration set.

**SmoothQuant** is the same algebra applied to *activations*: `s_j = max|X_j|^α / max|W_j|^(1−α)`,
migrating quantization difficulty from activations into weights, enabling W8A8. **For us it is mostly
conceptual**: CPU decode is weight-bandwidth bound and ggml does weight-only quantization well, so
W8A8 buys little; but the "equivalent transform that rebalances range" trick is the reusable idea.

**GPTQ** is a different family: layer-wise OBQ with `H = 2XXᵀ`, quantize column `j`, then compensate
the remaining columns with `w_F ← w_F − ((w_j − q_j)/[H⁻¹]_jj)·H⁻¹_{:,j}`. It fixes error *after*
rounding by exploiting input-channel correlations, and reaches 175B at 3–4 bits in ~4 GPU-hours.
**SpQR** isolates *individual outlier weights* (not channels) into a sparse high-precision side
structure plus bilevel quantization; <1% relative PPL loss at 3–4 bits, 33B in 24 GB. SpQR is the
right mental model for "the tail is sparse"; AWQ is the right model for "the tail is channel-aligned".

**Why it matters for us:** a 1.5B code model at 4-bit is already in the flat part of the curve, so the
headroom is in *which* tensors get the bits, not in a smarter quantizer per se.
**Confidence:** high (four independent papers, replicated in every GGUF/AWQ/GPTQ release).
**Cost:** trivial. ~40 lines + a calibration pass. Runs on CPU in minutes for 1.5B, ~1–2 h for 32B.

---

### F2. Layer pruning works on QA benchmarks and fails on generation — Qwen is the worst case
[source: *The Unreasonable Ineffectiveness of the Deeper Layers*, Gromov et al., 2024/ICLR'25, https://arxiv.org/abs/2403.17887 · *ShortGPT*, Men et al., 2024, https://arxiv.org/abs/2403.03853]

**Method.** Gromov: for a block of `n` layers, compute the angular distance
`d(x^ℓ, x^{ℓ+n}) = (1/π)·arccos(cos sim)` between the block's input and output hidden states on the
final token, averaged over ~10k samples; delete the block minimising `d`; heal with QLoRA. ShortGPT:
`BI_ℓ = 1 − cos(x^ℓ, x^{ℓ+1})`; delete the lowest-BI layers.

**How much can be lost (measured, MMLU 5-shot, phase transition to random):**
| family | transition |
|---|---|
| Llama-2 (7B/13B/70B) | **45–55%** of layers |
| Mistral-7B | 35% |
| Phi-2-2.7B | 25% |
| **Qwen (1st gen)** | **20%** |

That Qwen number is the single most important line in this report for us: our model is Qwen2.5.
Caveat: measured on Qwen-1, not Qwen2.5 — **UNVERIFIED** for our exact model, treat 20% as the prior
and measure it.

**What breaks, and it breaks first.** The paper is explicit: GSM8K and HellaSwag "exhibit immediate
degradation in performance with any amount of pruning", correlating with the next-token loss, while
MMLU/BoolQ stay flat. ShortGPT likewise: Llama-2-7B with 9/32 layers (28%) removed keeps ~88–92% of
the *multiple-choice average* (69 → 60.9) while XSum goes 19.40 → **0.67** and GSM8K to ~0. Their
"ShortGPT-gen" variant preserves generation but **does not reduce parameters** (it merges layers), so
it is useless for our capacity problem.

**Interpretation that matters for a code model:** the deep layers are not storage, they are *iterated
computation*. Multiple-choice QA needs one lookup; generation needs a maintained state machine. Code
needs bracket matching, variable binding across lines, and multi-token planning. Therefore:
**expect short completions to survive and long function bodies to break, with damage growing in the
number of tokens generated.** That is testable in our harness (exact-match on teacher-forced body
tokens as a function of distance from the prompt) and it is the honest framing for the "how many
layers can an 8B lose" question: *on QA, 45%; on generation, ~10% (3 of 32); on Qwen, less.*

**Confidence:** high for the direction and the Llama numbers; medium for extrapolating to Qwen2.5-Coder.
**Cost:** free to measure (one forward pass per prompt), free to apply (delete modules), but healing
needs **QLoRA on a GPU [GPU]** — without healing the simple "drop the deepest block" heuristic
degrades rapidly (Gromov Fig. 5, left panels).

---

### F3. Head pruning is the wrong lever on a GQA code model — 88% of the params are MLP
[source: *Are Sixteen Heads Really Better than One?*, Michel et al., 2019, https://arxiv.org/abs/1905.10650 · *Analyzing Multi-Head Self-Attention*, Voita et al., 2019, https://arxiv.org/abs/1905.09418 · *High-Layer Attention Pruning with Rescaling (HARP)*, Liu & Liu, 2025, https://arxiv.org/abs/2507.01900 · *Retrieval Head*, Wu et al., 2024, https://arxiv.org/abs/2404.14469]

**What is known.** Michel: many heads are prunable *at test time* (up to ~40% in BERT) but the
prunability is extremely layer-heterogeneous — some layers need all their heads, and the same head can
be critical for one input and useless for another. Voita: ~38% of encoder heads prunable; 3 heads carry
positional/linguistic function. HARP (2025, the current best training-free structured method) prunes
heads preferentially in **higher** layers plus an adaptive rescaling of the residual magnitude: on
Llama-3.1-8B it prunes heads in 8 layers at a 1/4 head ratio, on Qwen2-7B 4 layers at 1/7, and it beats
FLAP / LLM-Pruner / Wanda-structured on generation tasks. Counter-evidence: "retrieval heads" — a small
identifiable set — are load-bearing for long-context factuality, and pruning them causes hallucination.
So "low mean attention" ≠ "unimportant".

**Scoring options** (cheapest first): (a) mean ‖attn output‖ or mean attention entropy per head;
(b) `‖W_O·W_V‖` (OV-circuit norm) — a weights-only proxy; (c) gradient×activation (Michel's "importance");
(d) **activation patching on the head's output** — the only causal one, and the one our harness already
supports via `forward_pre_hook`.

**Why it is the wrong lever here — do the arithmetic on our model.**
Qwen2.5-Coder-1.5B: `d=1536`, 12 query heads, **2 KV heads**, `d_ff=8960`, 28 layers.
Per block: `q` 1536² = 2.36M, `k` 1536×256 = 0.39M, `v` 0.39M, `o` 1536² = 2.36M → **5.51M**;
MLP: `gate` 1536×8960 = 13.76M, `up` 13.76M, `down` 13.76M → **41.29M**. Total 46.79M/block, so
**MLP = 88.2% of block parameters**, attention = 11.8%. With only 2 KV heads, KV-cache head pruning
would mean deleting 50% of the KV cache per head removed — almost certainly catastrophic — while query-head
pruning saves no KV memory at all.

**Conclusion:** head pruning is at best a ~3% parameter saving on this architecture and a ~12% saving
on an 8B-class GQA model. Do it only if a head is *causally* dead (patching shows zero effect), and
never as the primary strategy. Direct the effort at `gate`/`up`/`down`.
**Confidence:** high on the arithmetic; medium on HARP's transfer to a 1.5B code model.
**Cost:** measurement free; applying it needs `num_attention_heads` surgery + a rescale (HARP) and
usually light finetuning **[GPU]** to recover.

---

### F4. Early exit / dynamic depth buys speed, never memory, and needs training to be reliable
[source: *Confident Adaptive Language Modeling (CALM)*, Schuster et al., 2022, https://arxiv.org/abs/2207.07061 · *LayerSkip*, Elhoushi et al., 2024/ACL, https://arxiv.org/abs/2404.16710]

**What it is.** CALM trains per-layer exit classifiers and a conformal-style threshold so that a token
exits as soon as its prediction is "confident enough" relative to the full model; it reports up to 3×
compute reduction with a provable sequence-level quality guarantee, and it must *retrain* the model with
the early-exit loss (and it needs to attend back to missing representations). LayerSkip trains with
layer dropout (low rate early, high rate late) + a shared early-exit loss, then uses the early layers as
a *draft* and the remaining layers to verify (self-speculative decoding): 2.16× on CNN/DM, **1.82× on
coding**, 2.0× on TOPv2.

**Can per-token depth be decided from the residual stream?** Yes, and cheaply — this is the one place
where the microscope gives a *free* signal. The logit-lens/tuned-lens trajectory already tells you how
converged a token is: if `KL(p_final ‖ p_lens_ℓ) < ε` for several consecutive layers, later layers are
not changing the answer for that token. Our `trace()` already computes per-layer top-k lens outputs, so
a "convergence depth" per token costs nothing. **UNVERIFIED** whether an untrained 1.5B code model's
lens trajectory is calibrated enough to gate on without the CALM/LayerSkip training — that is exactly
what experiment E4 below measures.

**Why it does not solve our problem.** Exit-at-layer-L still requires layers L+1..N to be *resident* for
the tokens that do not exit, so peak RSS is unchanged. And on CPU, decode is bandwidth-bound: skipping
layers does reduce bytes read per token (so it does help latency), but the model must still fit.
**Confidence:** high. **Cost:** measurement free; training the exit heads is **[GPU]**.

---

### F5. MoE-ification reduces FLOPs, not resident bytes — upcycling is out of reach
[source: *MoEfication*, Zhang et al., 2021/ACL Findings'22, https://arxiv.org/abs/2110.01786 · *Sparse Upcycling*, Komatsuzaki et al., 2022/ICLR'23, https://arxiv.org/abs/2212.05055 · *Branch-Train-MiX (BTX)*, Sukhbaatar et al., 2024, https://arxiv.org/abs/2403.07816 · *Mixture-of-Depths*, Raposo et al., 2024, https://arxiv.org/abs/2404.02258]

**MoEfication (the only CPU-feasible one).** Split each FFN into `k` experts by clustering the *neurons*
(balanced k-means on `W1` columns, or a co-activation graph partitioned with METIS), then train a small
router (an MLP, ~100 examples/layer) to pick the top-`n` experts per token. Reported: **10–30% of FFN
parameters active while keeping >95% of original performance**, and **2× CPU speedup at 25% FFN
parameters**. Training required: routers only — hours on CPU for a 1.5B, plausible for 8B.

**The catch, and it is decisive for our goal.** All `k` experts must be in RAM. MoE-ification is a
*compute/bandwidth* transformation, not a capacity one. It converts "8B dense at 5 tok/s" into "8B
sparse at maybe 9 tok/s", which is genuinely useful — but it does nothing for "32B does not fit".
**SPECULATION:** if routing is strongly skewed, an mmap'd expert store could keep only the hot experts
resident; but a 20 GB model with random 4 KB–10 MB reads from SSD per token will be far slower than RAM,
and page-cache thrash is likely. Not worth pursuing before the simpler wins.

**Sparse Upcycling / BTX / MoD — all need GPU-scale training.** Upcycling copies the dense FFN into `N`
experts and continues pretraining; the reported win costs ~40–50% of the *original pretraining* budget.
BTX additionally branch-trains each expert on a different corpus, then averages attention weights and
MoE-finetunes the router — three training stages at 100B-token scale. MoD requires training with a
top-k router from the start (it matches baseline at equal FLOPs but does not retrofit). None of these
are available on 8 CPU cores; a single epoch over 1B tokens at 1.5B scale is ~months here.
**Confidence:** high on the memory argument (it is arithmetic); high on the training-cost argument.
**Cost:** MoEfication routers ~hours CPU; upcycling/BTX/MoD **[GPU cluster]**.

---

### F6. Low-rank is real but small (10–30%), and only the activation-aware variants are worth it
[source: *ASVD*, Yuan et al., 2023/24, https://arxiv.org/abs/2312.05821 · *SVD-LLM*, Wang et al., 2024/ICLR'25, https://arxiv.org/abs/2403.07378 · *MoDeGPT*, Lin et al., 2024/ICLR'25, https://arxiv.org/abs/2408.09632 · *Palu*, Chang et al., 2024/ICLR'25, https://arxiv.org/abs/2407.21118]

**Weights.** Vanilla SVD on `W` is bad because activations are anisotropic: the directions that matter
are not the directions with the largest singular values. **ASVD** whitens by the input activation
covariance (`W·Σ_x^{1/2}` before decomposing) and searches rank per layer by measured sensitivity;
10–30% compression at ~1% MMLU loss, training-free, 4 h on an A6000 for 7B. **SVD-LLM** adds
truncation-aware whitening plus a sequential closed-form update of the factors; it dominates ASVD at
≥40% compression but both degrade badly there (at 80% reduction, PPL 31.8 vs ASVD's 80k — both are
unusable, the comparison is only relative). **MoDeGPT** decomposes *modules* (not tensors) into
coupled low-rank + diagonal structures with cross-layer alignment, no backprop: 90–95% of zero-shot at
25–30% compression.

**Palu is not a weight method** — it low-rank-projects the **KV cache** (50% cache compression, 91%
combined with quantization). Relevant to context length, not to fitting weights. Do not conflate.

**Does our calibration help?** Yes, and this is a real, cheap contribution of the harness: ASVD's
whitening matrix is literally `E[x xᵀ]` over calibration activations, which we can compute from our own
traces on *code* rather than wikitext. The risk is overfitting the calibration domain (ASVD's own
ablation shows sensitivity-ordered rank allocation is what saves it). Expect +1–3% over wikitext-calibrated
ranks on code; **SPECULATION**, testable in an afternoon.
**Confidence:** high on the 10–30% envelope, high that >40% is not training-free-viable.
**Cost:** SVD of 1536×8960 and 8960×1536 matrices is seconds on CPU; 32B has 8192×28672 → still
feasible (~1–2 h total). Calibration pass is the same one quantization needs.

---

### F7. Sensitivity-guided mixed precision is already published. Here is exactly what is and is not novel.
[source: *HAWQ*, Dong et al., 2019, https://arxiv.org/abs/1905.03696 · *ZeroQ*, Cai et al., 2020, https://arxiv.org/abs/2005.03642 · *BRECQ*, Li et al., 2021, https://arxiv.org/abs/2102.05426 · *LLM-MQ*, Li et al., 2023, https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/5c805adc-b555-499f-9882-5ca35ce674b5.pdf · *CMPQ*, 2024, https://arxiv.org/abs/2410.13056 · *When Reasoning Meets Compression*, Zhang et al., ICLR 2026, https://arxiv.org/abs/2504.02010 · *Capability-Guided Compression*, Gupta, 2026, https://arxiv.org/abs/2603.16440 · *The case for 4-bit precision*, Kumar et al., 2022/ICML'23, https://arxiv.org/abs/2212.09720]

**Prior art, in order, so we do not re-invent it:**
1. **HAWQ (2019)** — per-layer Hessian trace (top eigenvalue, then trace in HAWQ-V2) as sensitivity,
   then bit allocation by integer programming. The original "allocate bits by measured sensitivity".
2. **ZeroQ (2020)** — data-free, uses a Pareto frontier over layer-wise quantization error to pick
   bit-widths without calibration data.
3. **BRECQ (2021)** — block-wise reconstruction error (Fisher-weighted) as the allocation signal;
   the "measure the damage of quantizing this block, then allocate" pattern.
4. **LLM-MQ (2023)** — first-order (gradient) sensitivity per layer + integer programming under a
   *memory budget*, plus protecting 0.5% outlier weights in fp16; reaches **2.8-bit average** with
   custom CUDA kernels.
5. **CMPQ (2024)** — channel-wise mixed precision using only per-channel activation L2 norms; beats
   layer-wise LLM-MQ, i.e. *granularity mattered more than the sensitivity estimator*.
6. **Zhang et al., ICLR 2026 (the closest to our synthesis)** — uses **difference-of-means and
   attribution patching on every linear component** of compressed reasoning models to find which
   weights matter, then protects them. Two headline results: (a) dynamically-quantized **2.51-bit R1
   reaches close-to-R1 performance**; (b) current quantizers **over-compress the final layer's MLP-up
   and the MLP gate projections**, and protecting just **2% of all weights raises average accuracy by
   6.57%**, beating SOTA. This is our idea, done, with better baselines than we can run.
7. **CGC (2026)** — SAE-derived "capability density" to allocate compression budgets; finds the signal
   is **orthogonal to Wanda importance** (Spearman −0.054 over 384 heads) but reports a **negative
   result** on GPT-2 Medium because perplexity-based comparison cannot see the effect.

**What is genuinely left for us (three claims, in decreasing strength):**
- **(N1) A free sensitivity proxy.** Replace attribution patching (1 forward + 1 backward) with
  **logit-lens divergence per layer** (0 extra compute — we already compute the lens in `trace()`).
  Nobody in the list above uses lens divergence as the allocation signal. If it rank-correlates with
  measured quantization damage at ρ ≳ 0.8, we get per-layer sensitivity for a 32B model at the cost of
  one forward pass. **This is the one clean, cheap, novel measurement.** Testable in ~1 h.
- **(N2) A dual-objective budget.** All prior work allocates to protect *outputs* (loss/PPL/accuracy).
  Gupte et al. 2026 show SAE features are most fragile in the **middle** layers, while Gromov/ShortGPT
  show the **deep** layers are the output-redundant ones. Those two orderings point in *different
  directions*. A compressor that protects outputs will therefore preferentially damage the layers a
  microscopist cares about. Measuring that disagreement, and allocating under a joint
  (behaviour + feature) budget, is not in the literature I found. See F9.
- **(N3) Steering fidelity as the objective.** Bhandari & Wade 2026 measured steering under
  quantization post hoc; nobody has used *steering-vector cosine + effect slope* as the loss function
  for bit allocation. It is a small but real gap, and it is the metric our project actually cares about.

**What is NOT novel and should not be claimed:** "use interpretability to guide compression",
"sensitivity-guided per-layer mixed precision", "protect important weights at higher precision",
"attribution patching to find critical weights". All published, some of it five years old.
**Confidence:** high (I read the abstracts of 6 and 7 directly). **Cost:** zero to cite, and it saves us
from writing a paper that already exists.

---

### F8. Compression damages interpretability *before* it damages perplexity
[source: *Perplexity Can Miss SAE Feature Damage Under Quantization*, Duan, 2026, https://arxiv.org/abs/2606.03002 · *When Pruning Meets Interpretability*, Gupte et al., 2026, https://arxiv.org/abs/2608.25941 · *Steering Under Compression*, Bhandari & Wade, 2026, https://arxiv.org/abs/2609.06473]

This is the counter-risk in my brief, and the answer is unambiguous: **yes, 4-bit destroys the
interpretability substrate, and perplexity will not tell you.**

**SAE feature survival** (frozen fp16-trained SAE, per-feature Pearson correlation > 0.9 on identical
tokens, RTN per-output-channel):

| bits | Pythia-70M survived | Gemma-2-2B survived | mean corr (P/G) |
|---|---|---|---|
| INT8 | 98.3% | 99.1% | 0.981 / 0.966 |
| INT6 | 62.4% | 51.3% | 0.888 / 0.891 |
| INT5 | 38.0% | 27.7% | 0.777 / 0.764 |
| **INT4** | **14.5%** | **12.4%** | **0.583 / 0.494** |

And the dissociation: on Gemma-2-2B, **INT7 *improved* perplexity by 5.6% while degrading 18.7% of
active features**; INT6 improved sliding-window perplexity while only 51.3% of features survived. Survival
is *predictable from fp16 statistics alone* (AUC 0.92–0.97), with **peak activation** the strongest
marginal predictor — i.e. weak-signal features are the ones that blur. Damage is *graded* (blurred, not
destroyed) between INT7 and INT5, then collapses.

**Pruning:** Gupte et al. derive that SAE damage is governed by **perturbation energy**
`ε² = tr(ΔW Σ_x ΔWᵀ)` — a covariance-weighted norm — which explains two things: magnitude pruning is
bad because it ignores activation geometry, and **activation-aware pruning (Wanda, SparseGPT) implicitly
controls `ε²` and is therefore much more robust for SAE preservation**. Their structural finding:
**middle layers are significantly more sensitive to pruning than early or late layers**, and
layer-wise sparsity allocation beats uniform at equal average sparsity.

**Steering:** the *direction* survives quantization remarkably well — cosine 0.989–0.998 (INT8) and
0.945–0.990 (NF4) against the fp16 vector — and sentiment steering survives at matched behavioural
effect. But the *cost structure* does not: Mistral-NF4 lost 0.545 → 0.365 GSM8K **at α = 0**, i.e. the
compression moved the model further than the intervention did, and reasoning-length shortening becomes a
step function that fails discontinuously beyond 12–30%.

**Implications for this project (three rules):**
1. **D2 in `CONTEXT.md` is correct and now has evidence.** Science on bf16. Do not "just use Q4_K_M
   because it's faster" — you would be measuring a different model's features.
2. **A quantized model needs its own feature audit.** Any claim of the form "feature X causes
   behaviour Y" must be re-verified on the quantized weights, or the claim is about a model nobody runs.
3. **Steering vectors must be re-extracted, not transferred.** Cosine 0.95 sounds fine until you
   notice the coefficient needs recalibration and the side-effect budget has changed.
**Confidence:** high for SAE survival and steering cosine; medium for generalisation beyond the two
models tested (both papers say so explicitly).
**Cost:** the audit is cheap *if* you have an SAE. We do not have one for Qwen2.5-Coder-1.5B
(**UNVERIFIED**; Qwen-Scope covers Qwen3/3.5 only, arXiv 2605.11887). Proxy: per-layer residual cosine
+ logit-lens top-1 agreement vs fp32, which costs one forward pass and needs no SAE.

---

### F9. The two sensitivity rankings disagree — and that disagreement is the actual research opportunity
[source: synthesis of F2/F8: Gromov et al. 2024 (deep layers output-redundant) vs Gupte et al. 2026 (middle layers feature-fragile)]

Put the two results side by side:

- **Output sensitivity** to deleting a block is *lowest in the deep layers* (Gromov's angular distance is
  smallest there; ShortGPT's BI ranking deletes 21–29 of 32).
- **Feature sensitivity** to compression is *highest in the middle layers* (Gupte et al.).

If both hold on our model, then **the layers that are cheapest to compress for accuracy are not the
layers that are cheapest to compress for interpretability**, and a single sensitivity ranking is
strictly insufficient. That is a concrete, falsifiable, cheap claim, and it reframes the project's
novel axis from "guided compression" (done) to **"compression that preserves the microscope"** (not done).

**Test (E3 below, ~40 min):** on Qwen2.5-Coder-1.5B compute four per-layer rankings —
(a) ShortGPT BI `1−cos(x^ℓ,x^{ℓ+1})`, (b) Gromov angular distance for n=1, (c) logit-lens KL to final,
(d) **measured** ΔKL from actually quantizing that one layer to 4-bit — plus (e) a feature-fragility
proxy (per-layer activation cosine under the same 4-bit round-trip, which is Duan's metric without the
SAE). Then report the Spearman matrix. Any ρ < 0.5 between (d) and (e) is a publishable negative-then-
positive result: "output-guided and feature-guided compression orderings diverge; here is the joint
allocation that fixes it".
**Confidence:** the disagreement is likely but **UNVERIFIED** on our model; the experiment is cheap
enough that it does not matter.
**Cost:** ~40 min CPU, no GPU, no SAE.

---

### F10. CPU-specific constraints decide what is implementable (and they are not the GPU constraints)
[source: llama.cpp `tools/imatrix/README.md` and discussion #5263 (E[a_j²]), https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md · `llama-quantize --target-bpw` discussion #18531 · *Kumar et al. 2022*, https://arxiv.org/abs/2212.09720]

Four facts that change the design:

1. **ggml quantization granularity is a 32-weight block with a shared scale**, and the *type* is chosen
   **per tensor**. So the practical allocation unit is the tensor (7 per block × 28 = 196 for our 1.5B),
   **not** the channel. Any scheme requiring per-channel mixed types is not implementable in GGUF without
   new kernels. Design for per-tensor allocation.
2. **`Q4_K_M` is already a static mixed-precision policy**: "Q6_K for half of `attention.wv` and
   `feed_forward.w2`, else Q4_K" (~4.85 bpw for FFN tensors). `Q3_K_M` bumps `wv`/`wo`/`w2` to Q4_K.
   So the baseline we must beat is not "uniform Q4", it is a hand-tuned heuristic that has been iterated
   on for years. **This is the most important competitive fact in this report.**
3. **`imatrix` is activation-aware quantization, already shipped.** It computes `E[a_j²]` per input
   channel from a calibration corpus and uses it to weight the quantization error inside each block.
   That is the same signal as AWQ/Wanda, applied at weight-selection granularity. Its weakness is that
   it does **not** change the *type* allocation — that is `--target-bpw`, which allocates by tensor
   *size*, not importance. **The gap we can fill is exactly: type allocation by measured sensitivity,
   on top of an existing imatrix.** Nobody in the llama.cpp ecosystem does that automatically (there are
   community scripts that sweep per-layer KLD by hand).
4. **Decode on CPU is bandwidth-bound, so bytes ≈ time.** Mixed precision costs nothing extra in compute
   (each tensor gets its own kernel); a 4.9 bpw model is simply ~22% slower than 4.0 bpw and ~22% smaller
   than 6.0 bpw. This makes the trade-off unusually clean: quality ↔ bytes, with no kernel penalty. It
   also means **the whole value of guided allocation is "more quality for the same bytes"**, which is
   exactly what we can measure without ever deploying.

**Also from Kumar et al. 2022 (35k experiments, 19M–176B):** 4-bit is *almost universally optimal* for a
fixed total-bit budget; the relationship reverses at 3 bits. Consequence: **mixed precision has the most
to gain in the 3.0–4.0 bpw band and the least at 6+ bpw.** If our 32B target lands at 4.5 bpw, expect a
small win. If it lands at 3.5 bpw, expect a real one.
**Confidence:** high (all four facts are checkable in the llama.cpp tree / the paper).
**Cost:** zero — this is a design constraint, not a work item.

---

### F11. Honest verdict: what actually gets 32B / 70B onto 31 GB
[source: arithmetic from `CONTEXT.md` hardware + F1/F2/F6/F8 + Kumar et al. 2022]

Usable RAM after OS + PyTorch/llama.cpp overhead: **~28 GB**. KV cache for a 32B GQA model at 8k ctx
≈ 1 GB; for 70B ≈ 1.5–2 GB. Activations: negligible.

| target | bpw needed | weight bytes | fits? | quality reality |
|---|---|---|---|---|
| 8B | 8.0 | 8.0 GB | yes, easy | lossless-ish |
| 8B | 4.5 | 4.5 GB | yes | Q4_K_M, good |
| 32B | 6.0 | 24.0 GB | **yes, tight** (~25 GB w/ KV) | Q6-class |
| 32B | 4.5 | 18.0 GB | **yes, comfortable** | Q4_K_M/Q5, good |
| 32B | 3.5 | 14.0 GB | yes | degrades, usable for code? **UNVERIFIED** |
| 70B | 4.5 | 39.4 GB | **no** | — |
| 70B | 3.0 | 26.3 GB | marginal (~28 GB w/ KV) | Q2_K-class, poor |
| 70B | 2.5 | 21.9 GB | yes | Q2_K/IQ2, bad |

**The decisive argument against 70B is not memory, it is the scaling law.** At equal total bits,
70B×2.1 bpw = 32B×4.6 bpw = ~147 Gbit. Kumar et al. (35k experiments) find 4-bit is almost universally
optimal for a fixed bit budget and the relationship *reverses* at 3 bits. So the 32B@4.6 model is
expected to beat the 70B@2.1 model at the same RAM. **Recommendation: target 32B at 4.5–5.5 bpw, not 70B.**

**What each technique contributes to that goal, ranked by expected value per hour of work:**

| technique | byte saving | generation quality risk | CPU-feasible? | verdict |
|---|---|---|---|---|
| AWQ/imatrix activation-aware 4-bit | 4× | low | yes, minutes | **do it, it is the whole game** |
| per-tensor type allocation (sensitivity-guided) | 0 (reallocates) | low if validated | yes, ~2 h | **do it, it is our thesis, expect +2–8%** |
| low-rank (ASVD/SVD-LLM) 20% | 1.25× | low–medium | yes, ~1–2 h | do it after quantization; stack it |
| layer pruning 10% | 1.11× | **high for code** | yes to apply, no to heal | only if E2 says it survives |
| layer pruning 25–50% | 1.3–2× | **catastrophic for generation** | no (needs QLoRA healing) | **no** |
| head pruning | 1.03× (1.5B) / 1.12× (8B) | medium | yes | not worth it |
| MoE-ification | **1.0×** | medium | yes (routers) | speed only; do it for tok/s, not capacity |
| early exit / LayerSkip | **1.0×** | low–medium | apply yes, train no | speed only |
| upcycling / BTX / MoD | 1.0× (or negative) | — | **no [GPU]** | out of scope |

**One-sentence verdict:** *the microscope can tell us where to put the bits — but the honest expected
gain over a well-tuned `Q4_K_M` + `imatrix` is single-digit percent, and the only way to make a 32B run
on this box is 4.5-bit weight-only quantization, which works today without any of our machinery. Our
contribution is to make that 4.5 bits land better on a code model, and to keep the model interpretable
while doing it — not to make a 70B fit.*

---

## Concrete implementation (code sketches, real APIs)

### 10.1 Toy AWQ + GPTQ, ~60 lines, no dependencies beyond torch

```python
import torch

def group_quant(W, bits=4, g=128, scale=None):
    """W: [out, in] fp32. Asymmetric, per-group of g input channels.
    scale: optional per-input-channel multiplier (AWQ's diag(s))."""
    if scale is not None:
        W = W * scale[None, :]
    out, inn = W.shape
    Wg = W.reshape(out, inn // g, g)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    qmax = 2**bits - 1
    d = (mx - mn).clamp_min(1e-8) / qmax
    q = ((Wg - mn) / d).round().clamp(0, qmax)
    Wq = (q * d + mn).reshape(out, inn)
    return (Wq / scale[None, :]) if scale is not None else Wq   # dequantized, fp32

def awq_search(X, W, bits=4, g=128, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """X: [n, in] calibration activations. Returns best per-channel scale + loss."""
    a = X.abs().mean(0).clamp_min(1e-8)          # salient = large activation
    best = (None, float("inf"))
    for alpha in alphas:                          # AWQ searches a small grid, not gradients
        s = a.pow(alpha)
        s = s / s.mean()                          # keep magnitudes comparable
        err = (group_quant(W, bits, g, s) @ X.T - W @ X.T).pow(2).mean().item()
        if err < best[1]:
            best = (s, err)
    return best

def gptq_layer(W, X, bits=4, g=128, damp=0.01):
    """Column-wise OBQ with Cholesky. O(d_in^3) once, then O(d_in^2) per column."""
    W = W.clone().float()
    H = 2 * X.T @ X                                # [in, in]
    H += damp * torch.eye(H.shape[0]) * H.diag().mean()
    Hinv = torch.linalg.cholesky(torch.linalg.inv(H)).T   # upper-triangular factor
    qmax = 2**bits - 1
    for j in range(W.shape[1]):
        w = W[:, j]
        d = w.abs().max().clamp_min(1e-8) / qmax
        q = (w / d).round().clamp(-qmax // 2, qmax // 2) * d
        err = (w - q) / Hinv[j, j].clamp_min(1e-8)
        W[:, j] = q
        W[:, j + 1:] -= torch.outer(err, Hinv[j, j + 1:])   # compensate the rest
    return W
```
**Reimplementation note (the part that actually matters):** the *only* thing that makes 4-bit work on a
1.5B is the activation-derived scale `s`. Weight-magnitude salience does not find the right channels
(AWQ Table 1: selecting by weight magnitude is no better than random). Our `X` should come from *code*,
not wikitext, and from ~128 sequences × 512 tokens.

### 10.2 Three-tier sensitivity measurement (this is the novel instrument)

```python
# Tier A — free, from the existing trace (microscope.py already caches resid_post per layer)
#   a) BI_l      = 1 - cos(x_l, x_{l+1})                    [ShortGPT]
#   b) ang_l(n)  = arccos(cos(x_l, x_{l+n})) / pi           [Gromov]
#   c) lensKL_l  = mean_t KL(p_final(t) || softmax(lm_head(norm(x_l(t)))))   [ours]
#
# Tier B — attribution patching, 1 fwd + 1 bwd for ALL layers at once (Syed et al. 2023,
#   arXiv 2310.10348; best-practice caveats: Zhang & Nanda 2024, arXiv 2404.14082)
#   With h_l clean and h_l' = h_l + delta (simulating quantization noise):
#       score_l ~= sum_t (h_l' - h_l) . grad_{h_l} NLL(t)
#   -> one backward gives every layer's linearised damage. This is what scales to 32B.
#
# Tier C — ground truth on a subset: actually round-trip ONE tensor at a time and measure
#   dKL = mean_t KL(p_fp32 || p_q4) and dNLL, plus act_cos_l = cos(x_l^fp32, x_l^q4).
#   Cost per layer = one full forward over the probe set. Use 8 layers to validate Tier A/B.
```

**Why Tier C is per-*tensor*, not per-layer**: the allocation unit is the tensor (F10), and Zhang et al.
2026 show the damage is very unevenly distributed *within* a layer (`mlp.up` of the last layer is the
single most critical component). So sweep 7 tensors × 8 layers × 2 bit-widths = 112 forwards on a 1024-
token probe set. At ~8 tok/s that is 112 × 128 s ≈ **4 h** — too slow. Compromise: sweep all 7 tensors at
4-bit for 8 layers (56 forwards, 2 h) and use Tier B for the full 196-tensor ranking.

### 10.3 Allocation: multiple-choice knapsack, exact, milliseconds

```python
# err_t(b) = err_t(4) * 2**(4 - b)     # heuristic; calibrate the exponent on the Tier-C layers
# minimise sum_t err_t(b_t)  s.t.  sum_t n_t * b_t <= B_bits,  b_t in {2,3,4,5,6,8}
# DP over (tensor index, budget in 1e6-bit units). 197 tensors x ~7500 budget steps x 6 = 9M ops.
```
Report **actual bytes**, not "average bits" — the two differ because of the 233M-parameter embedding
(~15% of a 1.5B, assuming tied embeddings — 28×46.79M + 233M ≈ 1.54B matches the published size, but
verify `tie_word_embeddings` in `config.json`; if untied there is a second 233M tensor). It must not go
below 6-bit: errors there hit every token.

### 10.4 The experiment: Qwen2.5-Coder-1.5B, CPU, one day

**Fixed setup.** Model `Qwen/Qwen2.5-Coder-1.5B-Base`, fp32 (per `microscope.py`; CPU has no fast bf16
matmul). Calibration: 128 Python files from the local repo, 512 tokens each. Held-out: 32 disjoint files.
Probe set for sensitivity: 32 prompts × 96 tokens.

**Steps and cost** (assumes ~8 tok/s forward; ±2× uncertainty; measured on this box before committing):

| # | step | what it produces | est. time |
|---|---|---|---|
| E0 | load fp32, cache `resid_post` for all 28 layers on 2048 tokens | hidden-state bank (352 MB) | 8 min |
| E1 | Tier A metrics (BI, angular, lens KL) | 3 per-layer rankings, free | 2 min |
| E2 | layer *ablation* sweep: zero each block's output, measure ΔKL/ΔNLL | output-sensitivity ground truth | 55 min (28 × 32 × 96 tok) |
| E3 | **rank-correlation matrix** between E1/E2 and the feature proxy | F9's answer: do the rankings disagree? | 5 min |
| E4 | convergence-depth calibration: per-token `KL(p_final‖p_lens_ℓ)` vs whether early-exit would have been correct | is F4's free signal usable? | 15 min |
| E5 | Tier C: 4-bit round-trip, 7 tensors × 8 layers | err_t(4) ground truth, calibrates 10.3 | 2 h |
| E6 | Tier B: attribution patching over 1024 tokens | full 196-tensor sensitivity ranking | 15 min |
| E7 | allocate at B = 4.0 and 4.5 bpw; build variants | 2 allocations | 2 min |
| E8 | build + evaluate 7 variants (below) | the result | 3.2 h |
| E9 | HumanEval-20 generation smoke test on V0/V1/V3 | qualitative sanity only | 20 min |
| **total** | | | **≈ 7 h** (fits in a day with margin) |

**The 7 variants** (all compared at *matched bytes*, which is the only fair comparison):

| id | model | purpose |
|---|---|---|
| V0 | fp32 | reference |
| V1 | uniform Q4, group 128, RTN | naive baseline |
| V2 | uniform Q4 + AWQ scaling (imatrix-equivalent) | **the real baseline to beat** |
| V3 | sensitivity-guided mixed at 4.0 bpw (V2 + per-tensor types) | **our thesis** |
| V4 | uniform Q6 | upper bound on quality at 1.15 GB |
| V5 | V2 + drop the 2 lowest-sensitivity layers | does pruning survive on code? |
| V6 | V3 + drop the same 2 layers | do the two stack? |
| V7 | *feature-preserving* allocation at 4.0 bpw (protect middle layers) | N2, the novel claim |

**Exact metrics (definitions, so A6 can audit them):**
1. `ΔNLL_code` — mean nats/token on 6k held-out tokens, teacher-forced. Report as Δ vs V0 and as
   *bits-normalised*: `ΔNLL / (bytes_saved)`.
2. `KL_code` — `mean_t KL(p_V0 ‖ p_V)` on the same tokens. This is the most sensitive metric we have
   (Zhang et al. 2026 use KLD-style tails for exactly this reason); report mean, p95, p99.
3. `exact_match_body` — top-1 agreement with V0 on teacher-forced body tokens, **binned by distance from
   the prompt** (0–8, 8–32, 32+ tokens). This is the layer-pruning hypothesis test from F2: damage should
   grow with distance.
4. `steering_cos` — cosine between the variant's diff-in-means vector and V0's, at the same layer, for
   3 concepts (comment-density, error-handling, type-annotation).
5. `steering_slope` — for each concept, `Δlogsumexp(p[target_tokens])` at α = 2 divided by V0's, on 1024
   teacher-forced tokens. 1.0 = the same coefficient still works; 0.5 = needs recalibration.
6. `steering_side_effect` — ΔNLL on 512 tokens of unrelated prose at α = 2.
7. `interp_proxy` — per-layer `cos(x_l^V, x_l^V0)` on the probe set (Duan's metric without the SAE),
   and logit-lens top-1 agreement rate per layer.
8. `bps` — actual bytes × 8 / params, measured from the serialized tensors, not assumed.

**Pre-registered kill criterion.** If V3 does not beat V2 by **>5% relative** on both `KL_code` and
`steering_cos` at matched bytes, guided allocation does not pay at 4-bit and we report a negative result
(this is a live possibility: CGC 2026 published exactly such a negative result, and Kumar et al. say
4-bit is already near-optimal). In that case the fallback claim is N1 (free lens proxy) + N2 (ranking
disagreement), both of which we will already have measured in E3.

**Expected result — SPECULATION, stated in advance so it can be falsified:** per-layer quantization error
varies by 2–3× across layers (HAWQ/LLM-MQ's premise), so V3 should beat V2 by roughly 10–25% relative on
`KL_code`, with most of the gain from protecting the final block's `mlp.up`/`mlp.gate` and the embedding.
`steering_cos` should move less (vectors are robust, per Bhandari & Wade). V5 should show `exact_match_body`
collapsing at 32+ tokens while `ΔNLL` moves only slightly — the "generation breaks first" signature.

### 10.5 From the experiment to a runnable GGUF

The experiment measures *allocation*, not runtime. To ship it: emit a per-tensor type map
(`blk.{i}.ffn_down.weight -> Q6_K`, etc.) and drive `llama-quantize` with
`--imatrix code-imatrix.dat --target-bpw 4.5` plus explicit `--include-weights`/`--exclude-weights`
overrides (llama.cpp already supports per-tensor types; the missing piece is *our* map). Validate by
running the same `ΔNLL`/`steering_cos` metrics through the GGUF path, because a per-tensor map can
silently produce a different bpw than the DP assumed.

---

## What NOT to do (traps)

1. **Do not evaluate compression on perplexity alone.** INT7 *improved* perplexity while degrading 18.7%
   of SAE features (F8). Use KLD tails + behavioural + steering metrics.
2. **Do not prune layers of a Qwen model and expect Llama headroom.** 20% vs 45–55% phase transition
   (F2). And never prune for a *generative* workload without measuring generation length-dependently.
3. **Do not prune heads to save memory on a 2-KV-head GQA model.** Attention is 11.8% of block params;
   MLP is 88.2% (F3).
4. **Do not use unstructured sparsity (Wanda/SparseGPT) on CPU expecting a speedup.** ggml has no sparse
   kernels; 50% zeros is 0% faster and 100% of the memory. (Wanda is still useful as *analysis*, and its
   importance scores are a good baseline for our rankings.)
5. **Do not chase 70B on 31 GB.** At equal total bits, 32B@4.6 beats 70B@2.1 (F11).
6. **Do not expect MoE-ification / early exit to reduce resident memory.** They reduce FLOPs (F4/F5).
7. **Do not design per-channel mixed precision for GGUF.** ggml blocks are 32 weights with one scale;
   allocation is per tensor (F10).
8. **Do not reuse a steering vector across precisions.** Cosine 0.95 is not "the same vector"; the
   coefficient and the side-effect budget both move (F8). Re-extract and re-fit α.
9. **Do not calibrate on wikitext for a code model.** AWQ's own claim is that activation-aware scaling
   generalises across domains — but the *salient channel set* is domain-dependent, and our whole thesis
   is that per-layer sensitivity is measurable. Measure it on code.
10. **Do not compare a 4.9 bpw guided model against a 4.0 bpw uniform model and declare victory.** Matched
    bytes or it means nothing. (This is the most common error in the mixed-precision literature.)
11. **Do not quantize the embedding to 2–4 bits on a 1.5B.** 233M params = 15% of the model; every token
    reads it; keep it ≥6-bit and account for it in the budget.
12. **Do not use bf16 on this CPU for the science runs.** No AMX; fp32 is faster in PyTorch and is what
    `microscope.py` already does. bf16 only for the GGUF/llama.cpp path.
13. **Do not claim novelty for sensitivity-guided mixed precision.** It is HAWQ 2019 → LLM-MQ 2023 →
    Zhang et al. 2026 (F7). Cite it and differentiate on N1/N2/N3.

---

## Open questions for other agents

- **A1 (formats):** can we emit a *custom* per-tensor type map into GGUF (not just a preset), and does
  `llama-quantize` accept explicit per-tensor overrides without breaking the K-quant block layout? If
  yes, our allocation is directly shippable; if no, the experiment stays a PyTorch-only result.
- **A2 (interp):** is there any usable SAE for Qwen2.5-Coder-1.5B-Base? Qwen-Scope covers Qwen3/3.5 only.
  There are community SAEs for Qwen2.5-1.5B-**Instruct**; using them on a Base-Coder model is a basis
  mismatch. If not, what is the cheapest *causal* feature proxy — activation cosine, or a tiny TopK SAE
  trained on one layer? (This decides whether F8's audit is possible at all for us.)
- **A3 (steering):** does a vector extracted at layer L from a 4-bit model still work when injected into
  the fp16 model at the same layer (cross-basis transfer), or must everything be re-extracted? Our
  `steering_slope` metric answers the reverse direction; the forward direction is untested.
- **A4 (runtime):** can the runtime load a per-tensor type map and stream/dequant per tensor without
  materialising the whole model in fp32? (If not, our PyTorch evaluation path costs 6 GB for a 1.5B and
  is unusable for a 32B — we would need llama.cpp for the eval too.)
- **A6 (skeptic):** power analysis. With 30 prompts and greedy decoding, what is the minimum detectable
  effect on `steering_slope` and `exact_match_body`? If the MDE is larger than the effect we expect
  (F7 says single-digit percent), the experiment is underpowered and we should switch to teacher-forced
  KLD as the primary endpoint.
- **A7 (CPU scale):** measured tok/s for Qwen2.5-Coder-1.5B in fp32 with `attn_implementation="eager"`
  on this exact box, and measured RSS for a 20 GB mmap'd model. Both numbers feed E0–E9 directly; my
  estimates (8 tok/s, ~28 GB usable) are assumptions, not measurements.

---

## SCHEMA CHANGE

Proposal: add an optional `quant` block to the canonical trace so a trace carries its own compression
provenance. Without it, a trace from a 4-bit model is indistinguishable from a trace of a different
model, which is exactly the confusion F8 warns about.

```json
{"step": 0, "token_id": 123, "token_str": "def",
 "quant": {"bpw": 4.37, "scheme": "awq+g128+per-tensor",
           "layer_bits": {"0": 4, "1": 4, "27": 6},
           "dq_error": {"0": 0.031, "27": 0.008},
           "ref": "fp32"},
 "layers": [{"layer": 0, "resid_norm": 12.3,
             "logit_lens_topk": [[id, tok, prob]],
             "attn_entropy": 1.7, "mlp_out_norm": 4.1,
             "resid_cos_vs_ref": 0.9987}],
 "logits_topk": [[id, tok, prob]], "chosen": [id, tok, prob]}
```

`dq_error` = relative Frobenius error of that layer's dequantized weights; `resid_cos_vs_ref` = cosine of
this layer's residual against the same position in the reference model (only present when a reference
trace exists). Both are cheap and make "did compression change the computation?" a first-class query
instead of a separate script.

---

## Build next (ordered, with effort estimate)

| # | task | effort | why now |
|---|---|---|---|
| 1 | **Implement `group_quant` + `awq_search` + `gptq_layer`** (10.1) and validate against a known-good AWQ on 1.5B (target: uniform-Q4 PPL vs AWQ-Q4 PPL gap reproduces the published direction) | 3 h | everything downstream needs a trustworthy quantizer; if our toy is wrong, every conclusion is wrong |
| 2 | **E0–E1: hidden-state bank + Tier A rankings** (BI, angular, lens KL) | 15 min | free, and it already answers "which layers look redundant on *this* model" |
| 3 | **E2: layer-ablation sweep** — the output-sensitivity ground truth | 1 h | directly tests F2's 20%-Qwen prior on Qwen2.5-Coder; highest information per minute in the whole plan |
| 4 | **E5 (subset) + E3: 4-bit round-trip on 8 layers, then the rank-correlation matrix** | 2.5 h | this is N2 — the actual novel claim (do output- and feature-sensitivity rankings diverge?) |
| 5 | **E6 + E7: attribution patching, then allocation at 4.0/4.5 bpw** | 30 min | produces the deliverable artifact (a bit-allocation map) |
| 6 | **E8: build + evaluate V0–V4** (the core comparison) | 2.5 h | answers the thesis: does guided beat `imatrix` at matched bytes? |
| 7 | **V5–V7** (pruning + feature-preserving allocation) | 1.5 h | the stretch claims; V7 only if E3 shows the rankings disagree |
| 8 | **Ship the per-tensor map through `llama-quantize` and re-measure** (10.5) | 2 h | converts a research result into something that runs on the box |
| 9 | *(only if 6 fails the kill criterion)* **Write up N1 + N2 as the result** — "a free lens-based sensitivity proxy, and evidence that output-guided compression is feature-hostile" | 4 h | a clean negative-then-positive result is publishable; a weak positive is not |

**Sequencing rule:** steps 2–4 are worth doing even if the thesis fails, because they produce the
sensitivity map that A2/A3 need anyway. Step 6 is the go/no-go. Do not start step 8 before step 6 passes.
