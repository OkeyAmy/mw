# A2 — Mechanistic interpretability: what to compute inside a 28-layer, d=1536 CPU model

Scope: the *measurement* layer. Given hook access to `Qwen/Qwen2.5-Coder-1.5B-Base`
(28 layers, d_model 1536, 12 Q heads / 2 KV heads, vocab 151936), which interpretability
computations do we implement, in what order, and with exactly what math.
Steering *injection* is A3. Runtime is A4.

**Planning constant used for every cost below** (UNVERIFIED until measured on the box):
`150 tok/s` batched fp32 forward on 8 cores (≈4.6× the 16 tok/s single-stream in CONTEXT.md,
because batched matmuls amortise memory traffic). All CPU-hour figures scale as `1/throughput`;
measure this on day 1 and re-scale the table.

Model FLOPs/token ≈ `2·N·T = 2 × 1.54e9 ≈ 3.1 GFLOP`. At 50 GFLOPS sustained fp32 that is
62 ms/token single-stream — consistent with CONTEXT.md's 5–20 tok/s. Batched is where the win is.

---

## TL;DR (5 bullets)

- **Logit lens, attention patterns, and direct logit attribution are nearly free** (< 0.1 CPU-h
  per prompt) and give 80% of the visible value — but they are **readouts, not causes**. Build
  them first because they are also the debugging harness for everything else.
- **Activation patching is affordable here and is the first honest causal claim.** A full
  28-layer × 32-position denoising sweep is ~900 forward passes ≈ 0.05 CPU-h. This is the
  single highest value-per-hour method in the whole list for a 1.5B model.
- **`TransformerLens` 4.3.0 changed the API**: `HookedTransformer.from_pretrained` is *removed*.
  Use `TransformerBridge.boot_transformers(...)` + `bridge.run_with_cache(...)`, and **do not**
  call `enable_compatibility_mode()` — it centers `W_U`/`W_E` and silently breaks logit lens.
  Its `head_detector.detect_head` already implements previous-token / duplicate-token / induction
  head detection, and `tools.analysis` ships logit_lens, attribution_patching, direct_path_patching,
  sparse_probing. Do not re-implement these.
- **SAEs are the only method here that is genuinely marginal on CPU.** One layer, expansion 4,
  TopK k=32, 10M tokens ≈ 25 CPU-h (18 h of that is *harvesting activations*, not training).
  Verdict: do it, but treat features as hypotheses that must be causally validated, never as
  established concepts.
- **Attribution graphs / crosscoders (Anthropic 2025) are out of reach.** They require
  per-layer transcoders trained on billions of tokens; none exist for Qwen2, and training them
  on 8 CPU cores is ~10³ CPU-hours. Substitute head-level path patching, and say so plainly in
  the UI rather than implying we reproduce the paper.

---

## Findings

### F1. The residual stream is a shared read/write bus, and that is the whole justification for
### doing patching at `hook_resid_post`
[source: Elhage et al., "A Mathematical Framework for Transformer Circuits", Transformer Circuits
Thread, 2021 — https://transformer-circuits.pub/2021/framework/index.html]
*(page is >5 MB and could not be fetched here; URL and content are canonical and stable. Confidence
high from prior knowledge; body text UNVERIFIED by direct fetch this session.)*

**What it is.** Every component reads from and writes to the same `d_model`-dimensional residual
stream by addition. Attention head `h` in layer `l` has `W_OV = W_V W_O` (writes) and
`W_QK = W_Q W_K^T` (reads, for the score). Nothing else couples components; the "circuit" is the
set of paths through this bus. Because writes are additive and reads are linear, you can
*decompose* the stream exactly: `resid_post(l) = resid_pre(0) + Σ_{j≤l} (attn_out_j + mlp_out_j)`.
That identity is what makes direct logit attribution exact rather than heuristic, and it is why
patching one hook point is a meaningful intervention rather than an arbitrary perturbation.

**Why it matters for us.** It tells us the right hook point and the right decomposition. It also
tells us what *not* to expect: there is no "layer that holds the answer" — there is a sum.

**Two Qwen2-specific corrections to the clean story:**
1. **GQA.** 12 query heads share 2 KV heads (group size 6). `W_K, W_V` are `[1536, 2·128]`, not
   `[1536, 12·128]`. Any per-head `W_OV` analysis must use the *shared* KV slice: heads 0–5 use
   KV head 0, heads 6–11 use KV head 1. Consequence: you **cannot** cleanly patch one head's K/V
   path; patch its query side (`hook_q_input`) or its output (`hook_z`) instead.
2. **RoPE.** Scores are `(R_θ(i) q_i)·(R_θ(j) k_j)`, which depends only on `i−j`. So the
   "QK circuit matrix" is not a fixed matrix — it is a family indexed by relative offset. Do not
   eigendecompose `W_QK` on this model and call the eigenvectors "what the head looks for"; that
   trick is only valid for learned-absolute-position models. Work with the **attention pattern**
   directly.
3. Qwen2 has **QKV biases** (`b_Q, b_K, b_V`) and none elsewhere. Any weight-based path-patching
   math must include them or it is wrong by a constant.

**Confidence:** high. **Cost:** zero (it is a framing, not a computation).

---

### F2. Logit lens — exact math, exact shapes, and its one fatal misuse
[source: nostalgebraist, "interpreting GPT: the logit lens", LessWrong, 2020 —
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens]

**What it is.** Apply the model's *own* final norm and unembedding to each intermediate residual
and read off a distribution. Exact computation for Qwen2.5-Coder-1.5B:

```
h_l   = cache[f"blocks.{l}.hook_resid_post"]        # [B, T, 1536]   l = 0..27
g     = ln_f.weight                                  # [1536]         (model.norm.weight)
eps   = 1e-6                                         # config.rms_norm_eps
rms   = sqrt(mean(h_l**2, dim=-1, keepdim=True) + eps)      # [B, T, 1]
x̂_l   = (h_l / rms) * g                              # [B, T, 1536]   RMSNorm, weight AFTER normalising
W_U   = embed_tokens.weight                          # [151936, 1536]  <-- TIED, see below
b_U   = None                                         # lm_head has bias=False in Qwen2
logits_l = x̂_l @ W_U.T                               # [B, T, 151936]
p_l      = softmax(logits_l, dim=-1)
```

`h_27` through this path reproduces the model's own logits **exactly** (up to fp error). That
identity is the acceptance test for the whole pipeline: if `logit_lens(l=27)` ≠ `model(...)`
within ~1e-4, your norm or your `W_U` is wrong.

**Tied embeddings — this is a real trap, verified.** `Qwen2.5-1.5B` and `Qwen2.5-1.5B-Instruct`
both ship `"tie_word_embeddings": true` (verified from published `config.json`). There is **no
`lm_head.weight` tensor** in the checkpoint; `W_U = embed_tokens.weight`, shape `[151936, 1536]`.
*(The Coder-1.5B-Base config could not be fetched — HF `resolve` returns 401 for Qwen repos from
this host — so Coder is UNVERIFIED but is the same size tier. Assert `W_U.shape == (151936, 1536)`
at load and print `config.tie_word_embeddings`.)* Practical consequence: A1 must materialise the
tied matrix when converting to GGUF (llama.cpp requires an explicit `output.weight`), and any
analysis that treats `W_E` and `W_U` as independent objects is wrong on this model.

**Why it matters.** It is the cheapest possible per-layer view and it is the "brain view" the UI
wants. But read F3 before trusting it.

**Confidence:** high on math; high on tying for 1.5B/Instruct, medium on Coder. **Cost:**
~1 forward + 28 GEMMs of `[T,1536]×[1536,151936]` = `T × 13 GFLOP`; at T=32 that is 418 GFLOP
≈ 8 s. Per-token in the live trace, restrict to the last position and top-k via chunked matmul.

---

### F3. Logit lens is biased and early layers are noise — report *rank trajectories*, not probabilities
[source: Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens",
arXiv:2303.08112, 2023 — https://arxiv.org/abs/2303.08112]

**What it is.** The tuned lens trains a per-layer **affine translator** `A_l, b_l` and decodes
`softmax(W_U · (A_l h_l + b_l))`. Trained by minimising `KL(p_final ‖ p_l)` against the frozen
model's own final distribution — no labels, pure distillation. The paper's claims: more predictive,
more reliable, and **less biased** than logit lens, tested up to 20B; and causally, the tuned lens
uses features similar to the model's own.

**Why logit lens fails without it.** The residual stream is not in a fixed basis: later layers
rotate and rescale it. Logit lens assumes the final-layer readout basis is already correct at layer
5, which is false. The symptom is characteristic: the lens appears to "know" the answer early and
then *un-knows* it, or produces a flat/uniform distribution. Belrose et al. show the logit lens is
systematically biased, not merely noisy.

**Cost on our box, honestly.** A full-rank translator is `1536² + 1536 = 2.36M` params **per
layer** (66M total). With ~10–50 samples/param needed for a linear map, that wants ≥25M tokens;
at 150 tok/s that is 46 CPU-h *just for the forward passes* (the translators themselves are cheap
— 0.79 GFLOP/token for all 28). Recommendation: **start with a diagonal translator** (1536 params
per layer, `A_l = diag(a_l)`), trained on 1–2M tokens (~2–4 CPU-h, translators train in the same
forward pass since the model is frozen and `h_l` are detached). Diagonal captures the dominant
failure mode — per-dimension scale drift — and is 1500× cheaper to fit. Only go full-rank if the
diagonal lens still looks broken.

**Confidence:** high on the method and its claims; the diagonal-first adaptation is **our
engineering choice, not from the paper** — mark it as such in any writeup.

---

### F4. Head taxonomy is detectable with three cheap pattern statistics, and TL ships the detector
[source: Olsson et al., "In-context Learning and Induction Heads", arXiv:2209.11895, 2022 —
https://arxiv.org/abs/2209.11895; Gould et al., "Successor Heads: Recurring, Interpretable
Attention Heads In The Wild", arXiv:2312.09230, 2023 — https://arxiv.org/abs/2312.09230;
Wang et al., "Interpretability in the Wild" (IOI), arXiv:2211.00593, 2022 —
https://arxiv.org/abs/2211.00593]

**The heads, and how each is *measured* (not just named):**

| head type | definition | detection metric |
|---|---|---|
| previous-token | query at `i` attends to `i−1` | mean over `i` of `A[i, i-1]` on random tokens |
| duplicate-token | query at `i` attends to an earlier *copy of the same token* | mean attention to positions `j<i` with `tok_j == tok_i` |
| induction | query at `i` attends to `j+1` where `j<i` is the previous occurrence of `tok_i` | fraction of attention on the "prefix-match successor" positions |
| successor | head output boosts `tok+1` for ordered sets (numbers, months, days) | DLA of head output onto the successor token, averaged over `"Monday"→"Tuesday"`-style prompts |

The **induction score** in the paper's appendix is a *prefix-matching score*: build a sequence of
random tokens with a repeated prefix, then for each position measure the attention mass placed on
the token that *followed* the earlier occurrence of the current token. A pure induction head scores
≈1.0; a previous-token head scores ≈0 on this metric but ≈1.0 on the previous-token metric, so the
two are separable.

**Why it matters.** Induction heads are the only circuit in a 1.5B model that we can *fully*
explain and *causally* verify cheaply: ablate them and in-context learning degrades (Olsson et al.
give strong causal evidence for attention-only models, correlational for larger ones — be precise
about which claim we are entitled to). They are also the natural demo: "the model is doing
pattern-matching, here are the heads, watch them light up."

**Verified API.** `transformer_lens.head_detector.detect_head(model, seq, detection_pattern)` where
`detection_pattern ∈ {"previous_token_head", "duplicate_token_head", "induction_head"}` returns a
`[n_layers, n_heads]` score tensor, with `error_measure="mul"` (default; fraction of attention on
the predicted pairs) or `"abs"` (mean abs difference from the ideal pattern, mapped to `[-1,1]`).
This is exactly the three detectors we want, already written.

**Confidence:** high (API verified in source). **Cost:** one forward on a ~100-token prompt;
patterns are `[12, T, T]` per layer ≈ 13 MB total. < 0.01 CPU-h.

---

### F5. Activation patching — the math, the two directions, and why zero-ablation is a trap
[source: Meng et al., "Locating and Editing Factual Associations in GPT" (ROME), arXiv:2202.05262,
2022 — https://arxiv.org/abs/2202.05262; Wang et al. (IOI), arXiv:2211.00593, 2022;
Heimersheim & Nanda, "How to use and interpret activation patching", arXiv:2404.15255, 2024 —
https://arxiv.org/abs/2404.15255]

**What it is.** Pick two prompts that differ in exactly one thing (`clean` has the answer,
`corrupt` does not). Run both. Then re-run one of them with a single activation swapped in from
the other, and measure how much of the behavioural gap closes.

**Exact math.** Metric `M = logit(t) − logit(t')` at the final position (logit *difference*, never
probability — see traps). Clean gap `Δ = M_clean − M_corrupt`.

- **Denoising (restoring):** run on `corrupt`, patch in the *clean* activation at `(l, p)`.
  `score = (M_patched − M_corrupt) / Δ ∈ [0,1]`. Asks: *is this activation sufficient to restore
  the behaviour?* High score ⇒ the information is present here.
- **Noising (corrupting):** run on `clean`, patch in the *corrupt* activation.
  `score = (M_clean − M_patched) / Δ`. Asks: *is it necessary?* High score ⇒ the model needs it.

The two give different pictures, and Heimersheim & Nanda's central point is that reporting one and
calling it "the circuit" is the standard error. **Run both.** A component that scores high on
denoising but low on noising is redundant (a backup path exists) — that is a finding, not a bug.

**Why zero-ablation is a trap (four independent reasons):**
1. `0` is off-manifold. No real residual stream value is the zero vector, so you are measuring the
   model's response to an out-of-distribution input, not the component's role.
2. **RMSNorm actively punishes it.** With `x = 0`, `rms = sqrt(eps) = 1e-3`, so the *normalised*
   vector is `0/1e-3 · g = 0` — fine — but ablating one component leaves the residual norm
   *smaller*, and RMSNorm then divides by a smaller RMS, **amplifying every other component** by
   the same factor. Zero-ablating head 3 in layer 10 changes the effective scale of everything else
   in layer 10. That is a confound, not a measurement.
3. It conflates "removed information" with "injected large perturbation" — downstream nonlinearities
   respond to magnitude.
4. It cannot distinguish a component that matters from one that merely sits on the path.

Use instead: (a) patch from a corrupted prompt (in-distribution), (b) **mean-ablation** over a
dataset of prompts (in-distribution, preserves scale), or (c) symmetric token replacement / random
token substitution as the corruption. If you must ablate, ablate to the *dataset mean* and say so.

**Sweep design.** `layer × position` grid: 28 × T forwards. At T=32, 896 forwards × 0.21 s ≈
190 s ≈ **0.05 CPU-h per prompt pair**. Average over ≥10 prompt pairs; report the mean heatmap and
the per-pair variance (a finding that only exists on one prompt pair is not a finding).

**Confidence:** high. **Cost:** 0.05–0.5 CPU-h per experiment. **Claim type: CAUSAL.**

---

### F6. Path patching — the edge-level version, and the exact projection you need
[source: Wang et al., IOI, arXiv:2211.00593, 2022 (introduces path patching for circuits);
Goldowsky-Dill et al., "Localizing Model Behavior with Path Patching", arXiv:2304.05969, 2023]

**What it is.** Activation patching tells you *which node* matters. Path patching tells you *which
edge*. You patch only the sender's contribution to the receiver's input, holding everything else
at the corrupt value — so the measured effect is the `sender → receiver` edge, not "everything
upstream of the receiver".

**Exact math (this is where people get it wrong).** The receiver's query at position `p` is
`q_p = resid_p W_Q + b_Q`. The sender `s` writes `a_s` additively into `resid_p`. So the sender's
*exact* contribution to `q_p` is `a_s W_Q`. Path-patch the edge `s → q`:

```
q_p^patched = q_p^corrupt + (a_s^clean − a_s^corrupt) @ W_Q[h]        # [128]
```
and identically for `k` and `v` (with `W_K`, `W_V` — **remember the GQA slice and the bias**).
For an MLP receiver, the analogous projection is through `W_in` (`gate_proj`/`up_proj`).

Implementation: hook `blocks.{Lr}.attn.hook_q_input` (or `hook_k_input` / `hook_v_input`) — TL 4
exposes these as first-class hook aliases — and add the delta *after* the original projection has
been computed, or equivalently hook `hook_q` and add `delta @ W_Q[h]`.

**Combinatorics.** Full edge set is `(28·12)² ≈ 113k` edges — do not sweep it. Rank candidate
senders by direct logit attribution first (F7), keep the top ~100–200, then path-patch only those.
~200 forwards ≈ 40 s.

**Why it matters.** It is the difference between "layer 14 matters" (useless for a UI) and
"head 14.9 → head 19.2 → `return` token" (a story). It is also the honest head-level substitute
for Anthropic-style feature graphs (F8).

**Confidence:** high on math (it is just linear algebra on the residual decomposition). **Cost:**
0.1–0.5 CPU-h per experiment. **Claim type: CAUSAL.**

---

### F7. Direct logit attribution — free, exact, and only *half* causal
[source: Wang et al., IOI, arXiv:2211.00593, 2022 (direct logit attribution); Elhage et al., 2021
for the decomposition it relies on]

**What it is.** Because `resid_final = Σ components` and the unembed is linear (after the final
norm), you can attribute the target logit to each component:

```
contribution(component c, token t) = ( W_U[t] · ln_f(c_out) )      # scalar, per position
```
Summed over all components this equals `logit(t)` minus the bias term. For heads, use
`hook_z` unrolled through `W_O` (i.e. `hook_result`) so each head is attributed separately rather
than the whole attention block.

**Why it matters.** It is the ranking function that makes patching affordable (F6), and it is the
only per-head signal cheap enough to render live in the trace schema.

**Honest limitation.** DLA is a *decomposition of the output*, not an intervention. A component can
have large DLA and zero causal effect (its contribution is cancelled downstream), or zero DLA and
large causal effect (it gates something else). **Claim type: correlational.** Always pair a DLA
ranking with a patching confirmation before putting it in a report.

**Confidence:** high. **Cost:** ~0.01 CPU-h (it is a matmul on cached activations).

---

### F8. Attribution graphs / circuit tracing — correctly out of scope, and what to do instead
[source: Lindsey et al., "On the Biology of a Large Language Model", Transformer Circuits Thread,
2025 — https://transformer-circuits.pub/2025/attribution-graphs/biology.html; Ameisen et al.,
"Circuit Tracing: Revealing Computational Graphs in Language Models", 2025 —
https://transformer-circuits.pub/2025/attribution-graphs/methods.html; open-source library:
https://github.com/decoderesearch/circuit-tracer (moved from `safety-research/circuit-tracer`,
announced by Anthropic 2025-05-29 — https://www.anthropic.com/research/open-source-circuit-tracing)]

**What it is.** Replace each MLP (or, in the cross-layer variant, each layer's computation) with a
sparse **transcoder/CLT** trained to reproduce that sublayer's output from its input. Build a
"replacement model" from the transcoders, run it on a prompt, and compute a pruned **attribution
graph**: nodes are active features + error nodes + token embeddings, edges are direct linear effects
between them, computed by backprop through the replacement model.

**What the paper actually claims** (worth quoting accurately, because the popular summary
overstates it): the authors report parallel mechanisms and modularity, cross-domain abstraction,
plan formation (features for "rabbit"/"habit" active *before* the line that ends in them),
backward chaining from a goal, and a limited form of meta-cognition. They also state plainly that
their own graphs are sometimes **inconsistent with perturbation experiments**, that transcoders are
not guaranteed causally faithful, that feature splitting is unresolved, and that their method would
plausibly *miss* safety-relevant computation. Do not cite it as "we can now read the model's mind".

**Feasibility for us: no.** Three independent blockers:
1. **No transcoders exist for Qwen2.5-Coder.** The released weights cover Gemma-2-2B, Llama-3.2-1B,
   and (newer) Qwen3. Nothing for our model. Transcoder weights are model-specific — not portable.
2. **Training them is a GPU-scale job.** Per-layer transcoders trained on ~10⁹ tokens; ~28 layers.
   On our box that is ~10³ CPU-hours *for one model*, with no guarantee of quality.
3. `circuit-tracer` runs on an **nnsight** backend and expects a transcoder set + feature files for
   visualisation. Even a hand-trained CLT would land in the "cannot visualise" bucket the repo
   itself documents for most models.

**What to do instead — and be explicit in the UI about the downgrade.** Build a **head-level
attribution graph**: nodes = `(layer, head)` and `(layer, MLP)`; edges = path-patching effects (F6);
prune by effect magnitude; render with the same left-to-right layered layout. It is coarser
(12 heads/layer instead of ~10⁴ features) and it inherits none of the cross-layer-feature
machinery, but every edge in it is a *measured intervention*, which is arguably a stronger
epistemic status than a transcoder-derived attribution. Label it "head-level circuit", never
"attribution graph".

**Confidence:** high on infeasibility (repo, model list, and compute all check out).
**Cost:** the substitute is 0.5–2 CPU-h per prompt. The real thing: don't.

---

### F9. Sparse autoencoders — the math, and an honest CPU budget
[source: Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary
Learning", Transformer Circuits Thread, 2023 —
https://transformer-circuits.pub/2023/monosemantic-features/index.html; Templeton et al., "Scaling
Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", 2024 —
https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html; Lieberum et al.,
"Gemma Scope", arXiv:2408.05147, 2024 — https://arxiv.org/abs/2408.05147; Rajamanoharan et al.,
"Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs", arXiv:2407.14435, 2024 —
https://arxiv.org/abs/2407.14435; Gao et al., "Scaling and evaluating sparse autoencoders" (TopK),
arXiv:2406.04093, 2024]

**The vanilla (L1) SAE, exact:**

```
f(x)   = ReLU( (x − b_d) W_enc + b_enc )          # W_enc [d, d_sae],  f ∈ R^{d_sae}
x̂(x)   = f(x) W_dec + b_d                          # W_dec [d_sae, d]
L      = ‖x − x̂‖₂²  +  λ ‖f(x)‖₁
```
with `d = 1536`, `d_sae = expansion × d`. Expansion 4 → 6144; 8 → 12288; 16 → 24576.

**The improved loss Anthropic actually uses** (verified, April 2024 update): decoder columns are
*not* constrained to unit norm; instead the sparsity penalty is weighted by the column norm:
```
L = (1/|X|) Σ_x ‖x − x̂‖₂²  +  λ Σ_i |f_i(x)| · ‖W_d,i‖₂
```
and activations are pre-scaled so their mean squared L2 norm equals `d` before training (verified,
Scaling Monosemanticity). Pre-scaling matters: it makes `λ` transferable across layers.

**Why we should use TopK instead of L1** (engineering judgement, well-supported): TopK
(`keep the k largest f_i, zero the rest`) removes `λ` from the hyperparameter search entirely —
`k` *is* the sparsity target, L0 is exact and constant, and there is no L1 shrinkage bias.
`k=32` with `d_sae=6144` is the configuration used by the public Qwen2.5-1.5B SAEs
(e.g. `Solshine/deception-v4-saes-qwen25-1p5b`, and the FAST SAE suite which covers
`Qwen2.5-1.5B-Instruct` at layers 4/14/24 — arXiv:2506.07691). JumpReLU is better still on
reconstruction-at-fixed-sparsity but needs straight-through estimators and a tuned threshold; not
worth the debugging time on CPU.

**Realistic CPU budget for ONE layer of our model** (d=1536, expansion 4, d_sae=6144):

| step | math | cost |
|---|---|---|
| harvest 10M token activations | 3.1 GFLOP/token ÷ 50 GFLOPS | **18.5 h** |
| storage | 10M × 1536 × 2 B (bf16) | **30.7 GB** (fits in 293 GB; 20M tokens = 61 GB) |
| SAE training | ~113 MFLOP/token (fwd+bwd, dense) → 2.3 ms/token | **6.4 h** |
| **total** | | **≈ 25 CPU-h** |

At expansion 16 it becomes ~75.5M params and 453 MFLOP/token → 9 ms/token → 25 h of training
alone, and 100 GB of activations for 30M tokens. **Expansion 4–8 is the sweet spot on this box.**

**The honest caveat, stated plainly:** published SAEs see **10⁸–10⁹** tokens. We can afford
**10⁷**. That is 1–2 orders of magnitude short. Expect: many dead features (the public 6144-wide
Qwen SAEs report only ~500–590 alive features at k=32 — a warning sign about undertrained
dictionaries generally), poor reconstruction on rare tokens, and unstable feature identities across
seeds. **Therefore: an SAE feature is a hypothesis generator, never a finding.** Promote a feature
to a finding only after (a) it fires on a held-out probe set, and (b) ablating/scaling it moves the
target logit in the predicted direction (F5-style). Claim type: **correlational until ablated.**

**Confidence:** high on the math (both losses verified against primary sources); medium on the
throughput-derived hours. **Cost:** ~25 CPU-h per layer. **Do one layer. Not three.**

---

### F10. Sparse linear probes — the cheapest way to get *named* features on a code model
[source: Gurnee et al., "Finding Neurons in a Haystack: Case Studies with Sparse Probing",
arXiv:2305.01610, 2023 — https://arxiv.org/abs/2305.01610]

**What it is.** Train a linear classifier on `resid_post(l)` to predict a binary feature of the
input, with an L1 penalty (or a hard top-k mask) on the *weights* so that only a few dimensions are
used:
```
L = BCE(σ(w·x + b), y) + λ‖w‖₁        # k-sparse variant: keep top-k |w_i|, zero the rest
```
With `k=1` you localise individual neurons; the paper's finding is that early layers use sparse
*combinations* (superposition), middle layers have near-dedicated neurons for high-level contextual
features.

**Why this is unusually good for a *code* model.** Code has features that are (a) syntactically
unambiguous, (b) trivially labellable with a tokenizer/parser, and (c) genuinely interesting:
*inside a string literal*, *inside a comment*, *indentation depth == 2*, *inside an f-string*,
*bracket nesting depth*, *this identifier was defined earlier in the file*, *current token is a
Python keyword*. You can generate 10⁵ labelled positions in minutes with `ast`/`tokenize`, no human
labelling. This is the highest signal-per-CPU-hour item in the entire document after patching.

**Cost.** One cached forward over the corpus + logistic regression on `[N, 1536]`. For N=200k
positions, fitting is seconds in sklearn/torch. **1–2 CPU-h including data generation.**

**Claim type: correlational.** A probe that decodes "indentation depth" at layer 12 does not prove
the model *uses* indentation there. Upgrade to causal by ablating the probe direction and checking
the logit (which is also exactly A3's steering-vector pipeline — the probe direction and the
steering vector are the same object, so this work is shared).

**Confidence:** high. **Cost:** 1–2 CPU-h.

---

### F11. Summary table: cost, value, and what kind of claim each method earns

| # | method | CPU-h (this box) | value | claim type |
|---|---|---|---|---|
| 1 | logit lens (rank trajectory) | ~0.01 / prompt | high — the UI's spine | readout |
| 2 | attention patterns + entropy + max-weight | ~0.01 | high | correlational |
| 3 | head taxonomy (`detect_head`) | ~0.01 | high — cheap real structure | structural |
| 4 | direct logit attribution | ~0.01 | high — ranks everything else | correlational |
| 5 | activation patching (denoise + noise, 28×T) | 0.05–0.5 | **highest** | **CAUSAL** |
| 6 | path patching (top-200 edges) | 0.1–0.5 | high | **CAUSAL** |
| 7 | sparse probes (code features) | 1–2 | high | correlational |
| 8 | tuned lens, diagonal | 2–4 | medium | readout |
| 9 | tuned lens, full-rank | 40–120 | low (marginal over #8) | readout |
| 10 | SAE, 1 layer, exp 4, 10M tok | ~25 | medium (hypotheses only) | correlational |
| 11 | SAE, 1 layer, exp 16, 30M tok | ~60 | low (diminishing) | correlational |
| 12 | attribution graphs / CLTs | ~10³ | **infeasible** | (would be causal) |

Rule of thumb that falls out: **everything above row 6 is affordable in an afternoon; row 10 is a
weekend; row 12 is a different machine.**

---

### F12. The library situation changed in 2026 — pin the version and use the bridge API
[source: TransformerLens README + "Migrating to TransformerLens 4.0" docs, v4.3.0, 2026 —
https://github.com/TransformerLensOrg/TransformerLens;
https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v4.html]

**Verified facts:**
- `HookedTransformer.from_pretrained` was **removed in 4.0**. `from transformer_lens import
  HookedTransformer` now raises `ImportError`.
- Replacement: `TransformerBridge.boot_transformers(name, device=..., dtype=...)` then
  `bridge.run_with_cache(...)`, `bridge.run_with_hooks(...)`, `bridge.to_tokens(...)`.
- **`enable_compatibility_mode()`** reproduces HookedTransformer's weight processing (LayerNorm
  folding, `center_writing_weights`, `center_unembed`). **Do not call it.** We want raw HF weights
  so that `logit_lens(l=27) == model logits`. Centering `W_U` shifts every logit by a per-position
  constant and silently invalidates logit lens comparisons across layers.
- Qwen2 has a dedicated adapter (`transformer_lens/model_bridge/supported_architectures/qwen2.py`)
  with `_qkvo_weight_conversions(include_biases=True)` — i.e. it explicitly handles Qwen2's QKV
  biases and GQA K/V layout. The officially-tested list includes `Qwen/Qwen2.5-1.5B` and
  `-Instruct` but **not** the `-Coder-` variants (same architecture, so it should load — verify).
- Built-in analyses (use these, don't rewrite): `transformer_lens.tools.analysis.logit_lens`,
  `.attribution_patching`, `.direct_path_patching`, `.sparse_probing`, `.direct_logit_attribution`,
  `.jacobian_lens`; plus `transformer_lens.head_detector.detect_head`.
- Hook aliases confirmed in source: `blocks.{l}.hook_resid_pre` → `hook_in`,
  `hook_resid_post` → `hook_out`, `hook_attn_out` → `attn.hook_out`,
  `hook_mlp_out` → `mlp.hook_out`, `attn.hook_q/k/v` → `q/k/v.hook_out`,
  `attn.hook_z` → `o.hook_in`, plus `attn.hook_q_input` / `hook_k_input` / `hook_v_input` and
  `attn.hook_pattern`.
- Patching-while-caching gotcha (from the `run_with_cache` docstring): hook lists passed directly to
  `run_with_cache` raise `TypeError`; wrap the call in `with bridge.hooks(fwd_hooks=[...]):`.

**Confidence:** high (read from source + docs this session). **Cost:** zero, but getting this wrong
costs a day.

---

## Concrete implementation (code sketches, real APIs)

### 0. Setup and the acceptance test (do this before anything else)

```python
import torch
from transformer_lens.model_bridge import TransformerBridge

torch.set_grad_enabled(False)
bridge = TransformerBridge.boot_transformers(
    "Qwen/Qwen2.5-Coder-1.5B-Base",
    device="cpu",
    dtype=torch.float32,          # bf16 weights for memory, fp32 for matmul on CPU
)
# DO NOT call bridge.enable_compatibility_mode()

cfg = bridge.cfg
assert cfg.n_layers == 28 and cfg.d_model == 1536 and cfg.d_vocab == 151936
W_U = bridge.unembed.W_U
assert W_U.shape == (151936, 1536), W_U.shape      # v4 raw mode stores [d_vocab, d_model]
assert getattr(bridge.cfg, "tie_word_embeddings", True) is True   # see F2
```

### 1. Logit lens — manual (shows the math) and via the shipped helper

```python
import torch.nn.functional as F

def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x / rms) * weight

def logit_lens_all_layers(cache, ln_f_w, W_U, top_k=5):
    """cache: ActivationCache from bridge.run_with_cache(). Returns [L+1] list of dicts."""
    out = []
    for l in range(28):
        h = cache[f"blocks.{l}.hook_resid_post"]        # [B, T, 1536]
        x = rms_norm(h, ln_f_w)                          # [B, T, 1536]
        logits_l = x @ W_U.T                             # [B, T, 151936]  (b_U = None)
        p = logits_l[:, -1, :].softmax(-1)               # last position only
        vals, idx = p.topk(top_k, dim=-1)
        out.append({"layer": l, "topk": list(zip(idx[0].tolist(), vals[0].tolist())),
                    "resid_norm": h[:, -1, :].norm(dim=-1).item()})
    return out

# Acceptance test: layer 27 through the lens must equal the model's own logits.
tokens = bridge.to_tokens("def fibonacci(n):\n    return")
logits, cache = bridge.run_with_cache(tokens)
lens27 = rms_norm(cache["blocks.27.hook_resid_post"], bridge.ln_final.weight) @ W_U.T
assert torch.allclose(lens27, logits, atol=1e-3), (lens27 - logits).abs().max()

# Or use the shipped implementation (chunked, handles model-specific readout transforms):
from transformer_lens.tools.analysis import logit_lens
res = logit_lens(bridge, "def fibonacci(n):\n    return", targets=" n")
print(res.rank_trajectory(" n"))     # rank of " n" at every layer — the real signal
```

**Report rank, not probability.** `rank_trajectory` is what belongs in the UI: "the answer is rank
40,000 at layer 8, rank 900 at layer 14, rank 3 at layer 22, rank 1 at layer 27" is a readable
story. A probability curve is not, and it is confounded by residual-norm drift that RMSNorm hides.

### 2. Attention patterns, entropy, and head taxonomy

```python
from transformer_lens.head_detector import detect_head

pattern = cache["blocks.14.attn.hook_pattern"]       # [B, 12, T, T]  (GQA-expanded)
entropy = -(pattern.clamp_min(1e-9).log() * pattern).sum(-1)     # [B, 12, T]
max_w   = pattern.max(-1).values                                  # [B, 12, T]

# One call gives [n_layers, n_heads] scores for each head class:
prev  = detect_head(bridge, seq, "previous_token_head")   # random-token string
dup   = detect_head(bridge, seq, "duplicate_token_head")
ind   = detect_head(bridge, seq, "induction_head")        # repeated random prefix
print((ind > 0.5).nonzero())        # candidate induction heads (layer, head)
```
Then **confirm causally**: patch/ablate the candidate head's `hook_z` and check that the
induction-completion logit drops (F5). A pattern score is a hypothesis; the patch is the finding.

### 3. Activation patching — denoising sweep with a logit-difference metric

```python
def logit_diff(logits, t_ok, t_bad, pos=-1):
    return (logits[0, pos, t_ok] - logits[0, pos, t_bad]).item()

def denoise_sweep(bridge, clean_ids, corrupt_ids, t_ok, t_bad, layers=range(28), positions=None):
    _, ccache = bridge.run_with_cache(clean_ids)
    base = bridge(corrupt_ids)
    m_corrupt = logit_diff(base, t_ok, t_bad)
    m_clean   = logit_diff(bridge(clean_ids), t_ok, t_bad)
    delta = m_clean - m_corrupt
    T = corrupt_ids.shape[1]
    positions = range(T) if positions is None else positions
    grid = torch.zeros(28, T)
    for l in layers:
        src = ccache[f"blocks.{l}.hook_resid_post"]          # [1, T, 1536]
        for p in positions:
            def hook(resid, hook, l=l, p=p):
                resid = resid.clone()
                resid[:, p, :] = src[:, p, :]
                return resid
            with bridge.hooks(fwd_hooks=[(f"blocks.{l}.hook_resid_post", hook)]):
                out = bridge(corrupt_ids)
            grid[l, p] = (logit_diff(out, t_ok, t_bad) - m_corrupt) / delta
    return grid        # [28, T] heatmap, 1.0 = fully restored the behaviour
```
Run the **noising** twin (patch corrupt→clean) and show both. They disagree, and the disagreement
is the interesting part.

### 4. Path patching — the exact edge projection

```python
def path_patch_edge(bridge, clean_ids, corrupt_ids, s_out_clean, s_out_corrupt,
                    Lr, head, which="q", t_ok=0, t_bad=0):
    """Patch only the sender->receiver edge by projecting the delta through W_{q,k,v}."""
    attn = bridge.blocks[Lr].attn
    W = {"q": attn.W_Q, "k": attn.W_K, "v": attn.W_V}[which]   # [n_heads, 1536, 128]
    W_h = W[head]                                              # [1536, 128]
    delta = (s_out_clean - s_out_corrupt)                      # [1, T, 1536]
    dq = delta @ W_h                                           # [1, T, 128]

    def hook(x, hook):
        x = x.clone()
        x[:, :, head, :] += dq                                 # hook_q is [B, T, 12, 128]
        return x

    with bridge.hooks(fwd_hooks=[(f"blocks.{Lr}.attn.hook_q", hook)]):
        out = bridge(corrupt_ids)
    return logit_diff(out, t_ok, t_bad)
```
Note: for `which="k"` or `"v"` with GQA, `W_h` must index the *shared* KV head
(`kv = head // 6` for 12/2), and the hook tensor is `[B, T, 2, 128]`. Patch `q` when you can —
it is the only cleanly per-head path on this model.

### 5. TopK SAE (one layer), with the honest budget in comments

```python
import torch.nn as nn

class TopKSAE(nn.Module):
    def __init__(self, d_in=1536, d_sae=6144, k=32):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / d_in**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) / d_sae**0.5)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        f = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        v, i = f.topk(self.k, dim=-1)
        f = torch.zeros_like(f).scatter(-1, i, v)
        return f @ self.W_dec + self.b_dec, f

# Pre-scale so mean squared L2 norm == d_model (verified from Scaling Monosemanticity):
scale = x.mean(0).pow(2).sum().sqrt() / (1536 ** 0.5)   # fit on a sample, apply to all

# L1 variant instead, if you prefer the published loss:
#   loss = mse + lam * (f.abs() * W_dec.norm(dim=1)).sum(-1).mean()
# (weighted by decoder column norm — the April-2024 update; unit-norm columns is the older form)
```
Harvest once, reuse forever: `torch.save` the `[N, 1536]` bf16 block to
`/tmp/opencode/acts/layer14.pt` (10M tokens ≈ 31 GB), then every SAE experiment is training-only.

### 6. Sparse probe on a code feature

```python
# labels from Python's own tokenizer/ast — no human labelling
import tokenize, io
def label_in_string(src: str) -> list[bool]:
    """True for positions inside a string literal."""
    flags = [False] * len(src.split())   # (tokenise properly in the real impl)
    ...
# then: logistic regression on cached resid_post(l) with L1, k-sparse variant keeps top-k |w_i|
```
Probe directions double as steering vectors → hand the top probe directions to A3 directly.

---

## What NOT to do (traps)

1. **Do not zero-ablate and call it patching.** Zero is off-manifold, and RMSNorm *rescales every
   remaining component* when you shrink the residual norm. Use corrupt-prompt patching or
   mean-ablation. (F5)
2. **Do not report probabilities from the logit lens.** Report rank trajectories. The lens is
   scale-invariant by construction (RMSNorm), so absolute probability is not comparable across
   layers, and early-layer distributions are noise. (F3)
3. **Do not call `enable_compatibility_mode()`** and then do logit lens. Centered `W_U` breaks the
   `lens(27) == logits` identity. (F12)
4. **Do not treat `W_E` and `W_U` as independent on this model.** They are tied (verified for
   Qwen2.5-1.5B/-Instruct). Anything that reads `lm_head.weight` from the checkpoint gets nothing.
5. **Do not eigendecompose `W_QK` on a RoPE model** and interpret eigenvectors as "what the head
   looks for". The score depends on relative offset, not on a fixed matrix. (F1)
6. **Do not path-patch K/V per-head without handling GQA.** 12 query heads share 2 KV heads; a
   "per-head K patch" actually hits 6 heads. Patch queries, or patch `hook_z`. (F1/F6)
7. **Do not forget Qwen2's QKV biases** in weight-space path patching. Off by a constant otherwise.
8. **Do not read attention weights as explanations.** Attention is a *routing* signal computed
   before the value/output path; high attention ≠ high contribution. Use DLA/patching to check.
9. **Do not report a single-prompt-pair patching result.** Average ≥10 pairs and show variance.
   Heimersheim & Nanda's whole point is that a patch is evidence about *one contrast*, not a map.
10. **Do not claim "denoising found the circuit".** Denoising shows sufficiency at a cross-section;
    a cut through a circuit is not the circuit. Run noising too. (F5)
11. **Do not train an SAE on <1M tokens and publish features.** Dead features and feature splitting
    will dominate. Our ceiling (~10⁷ tokens) is already 1–2 orders below published work. (F9)
12. **Do not call a head-level graph an "attribution graph".** Different object, different
    guarantees, and the Anthropic paper explicitly warns its own graphs are sometimes unfaithful —
    ours would be a different method entirely. Label it "head-level circuit". (F8)
13. **Do not run a full 113k-edge path-patching sweep.** Rank by DLA, take the top ~200. (F6)
14. **Do not let the UI show a "confidence" derived from `resid_norm`.** Residual norm grows
    monotonically with depth and is not a confidence signal.

---

## Open questions for other agents

- **A4 (runtime):** does `bridge.run_with_cache` on a *single decode step* (KV-cached, `past_key_values`)
    still populate `blocks.l.attn.hook_pattern` with shape `[1, 12, 1, T]`? The per-token trace
    depends entirely on this. Also: does `bridge.hooks(...)` compose with generation loops, or do we
    need a manual forward per token (CONTEXT D5 says we can afford one forward per token)?
- **A4:** what is the *measured* batched throughput? Every CPU-hour number in this file scales
    linearly with it; 150 tok/s is an assumption, not a measurement.
- **A3 (steering):** a difference-in-means steering vector and a sparse-probe direction are the same
    mathematical object. Let us share one `probe/steer` module: A2 fits the direction, A3 injects it,
    and the *validation* is a shared logit-diff metric. Please adopt `logit(t) − logit(t')` as the
    project-wide causal metric so our numbers are comparable.
- **A3:** for dose–response, the measurement side is mine — tell me the exact injection schedule and
    I will produce the (coef → Δlogit) curve with error bars.
- **A1 (formats):** confirm whether `Qwen2.5-Coder-1.5B-Base` really has `tie_word_embeddings: true`
    (HF returned 401 from this host, so I could only verify the non-Coder 1.5B). If tied, the GGUF
    conversion must materialise `lm_head` from `embed_tokens`, and any Q4_K_M round-trip must
    preserve it or the logit lens will be wrong in a way that looks like "quantisation damage".
- **A5 (UI):** the trace schema below assumes the UI can render a `[28, T]` heatmap for patching.
    Patching is *not* per-token-live (it costs ~900 forwards); it is an offline artifact per prompt.
    Please design two views: live trace (cheap, per token) and causal panel (offline, per prompt).
- **A6 (validation):** propose the acceptance test for a "finding". My suggestion, for you to
    harden: (a) patching effect > 2σ over prompt-pair shuffle, (b) reproduces on ≥5 independent
    prompt pairs, (c) an intervention in the predicted direction changes the output. All three, or
    it is a correlation.

---

## SCHEMA CHANGE

The canonical schema in `CONTEXT.md` is per-token and cheap; that is right. Two changes:

**(1) Add three cheap per-layer fields** (all free, computed from the same cache):

```json
{"step": 0, "token_id": 123, "token_str": "def",
 "layers": [{"layer": 14,
             "resid_norm": 12.3,
             "logit_lens_topk": [[id, tok, prob], ...],
             "chosen_rank": 812,
             "attn_entropy": 1.7,
             "attn_max_weight": 0.62,
             "mlp_out_norm": 4.1,
             "dla": {"attn": 0.31, "mlp": -0.12}}],
 "logits_topk": [[id, tok, prob], ...], "chosen": [id, tok, prob]}
```

- `chosen_rank` — rank of the token the model *actually emits* at this layer. One int, and it is the
  single most informative number in the whole trace (it turns the lens into a readable trajectory).
- `attn_max_weight` — mean over heads of the max attention weight. Cheap, and unlike entropy it is
  comparable across layers.
- `dla` — direct logit attribution of the chosen token, split attention vs MLP. Two floats; this is
  what makes "which layer is doing the work" answerable at a glance.

**Definitions to pin down** (currently ambiguous): `resid_norm` = L2 norm of `hook_resid_post` at the
**last position**; `attn_entropy` = mean over heads of `−(Σ A log A)` at the last query position;
`mlp_out_norm` = L2 norm of `blocks.l.mlp.hook_out` at the last position.

**(2) Causal results are a separate artifact, not part of the per-token stream.**
Patching costs ~900 forwards; it cannot be emitted per token. Add a sibling file per prompt:

```json
{"trace_id": "...", "kind": "activation_patching", "direction": "denoise",
 "metric": "logit_diff", "targets": ["return", "print"],
 "grid": {"layers": 28, "positions": 12, "values": [[...]]},
 "n_prompt_pairs": 10, "std": [[...]]}
```
This keeps the live trace cheap and the causal claims honestly separated from the readouts.

---

## Build next (ordered, with effort estimate)

Ordered by (value ÷ cost) on this specific model, with the dependency chain respected.

| # | step | effort | why here |
|---|---|---|---|
| 1 | **Harness + acceptance test**: load via `TransformerBridge`, assert shapes/tie, prove `lens(27) == logits` to 1e-3. Measure batched tok/s. | 0.5 day | everything else is invalid if this is wrong; also replaces the 150 tok/s assumption with a real number |
| 2 | **Logit lens + `chosen_rank` trajectory** into the trace schema. | 0.5 day | the UI spine; ~free |
| 3 | **Attention patterns, entropy, `attn_max_weight`** per layer. | 0.5 day | ~free, and needed for #4 |
| 4 | **Head taxonomy** via `detect_head` (prev-token / duplicate-token / induction) on synthetic prompts. | 0.5 day | first real structure; validates the hook layer end-to-end |
| 5 | **DLA** (per head, per MLP) → ranking. | 0.5 day | makes #6 affordable; feeds the UI |
| 6 | **Activation patching** — denoise *and* noise, 28×T grid, ≥10 prompt pairs, variance reported. | 1–2 days | **the first honest causal claim.** Do not skip to anything below before this ships |
| 7 | **Path patching** on the top ~200 DLA-ranked edges → head-level circuit graph. | 1 day | the "story" layer; substitutes for attribution graphs |
| 8 | **Sparse probes** on 5–8 code features (string literal, comment, indent depth, bracket depth, f-string, keyword, previously-defined identifier). | 1 day | highest signal per hour after #6; directions feed A3 directly |
| 9 | **Tuned lens, diagonal** (1536 params/layer, 1–2M tokens). Only if #2's trajectories still look broken. | 1 day | cheap fix for lens bias |
| 10 | **SAE, layer 14** (exp 4, TopK k=32, 10M tokens, harvested once). | ~25 CPU-h ≈ 2–3 days wall | hypotheses only; must be causally validated per feature before it appears in any finding |
| 11 | *(optional)* SAE layer 21 or expansion 8, if #10 produces anything worth chasing. | +25 CPU-h | only if #10 pays |
| — | **Do not build**: attribution graphs / CLTs, full-rank tuned lens, per-neuron "concept" maps, SAE sweeps across all 28 layers. | — | see F8, F3, F9 and CONTEXT.md's "what transparency cannot mean" |

**Critical path to a defensible demo:** 1 → 2 → 4 → 6. That is roughly one week and it yields a
system that can say, with evidence: *"at layer 14 the model's residual stream already contains
'return' as the top-1 prediction (rank 1), and patching that position from a corrupted prompt
restores 87% of the behaviour gap, averaged over 10 prompt pairs, while patching layer 22 restores
nothing."* That is a real mechanistic claim. Everything above #6 is decoration on it.
