# A3 — Steering & control: build a vector, inject it live, prove it worked

## TL;DR (5 bullets)

- **The vector is the easy part.** `v = mean(h⁺) − mean(h⁻)` at one layer, unit-normalized, then scaled to the layer's mean residual norm, gives a usable steering vector in ~30 lines and ~15 CPU-minutes for a 1.5B model. CAA, RepE and Arditi's "refusal direction" are all this same estimator with different framing.
- **The hard part is the harness, not the math.** A steering effect is only believable if (a) `coef=0` is *bit-identical* to no hook, (b) a matched-norm random vector does nothing, (c) the dose–response curve is reported in full, and (d) the metric is measured by something other than the probe that built the vector. Without these four, 100% of "it worked" claims are unfalsifiable.
- **Mid-generation control is a coefficient scheduler, and it is path-dependent.** HF's KV cache bakes the coefficient that was active when each token was computed into that token's K/V for every layer ≥ L. Changing `coef` between tokens is therefore *history*, not a re-parameterization: `coef=[1,1,2]` ≠ `coef=[2,2,2]`. This must be an explicit design decision, not an accident.
- **Three intervention sites with three different physics:** `resid_pre(L)` (semantic, generalizes, contaminates cache), `resid_post(27)` (mostly local, still caches), and post-`model.norm` (zero cache effect, and *provably equivalent to a rank-1 logit bias* `Δlogits = W_U·v`). That last equivalence is the bridge between "steering" and "forcing a token".
- **For this project, steer code behaviours you can measure** — comment density, language mode, verbosity, docstring presence — not "refusal". `Qwen2.5-Coder-1.5B-Base` is a *base* model: it has no chat template and no RLHF refusal circuit to ablate, so the Arditi result is a method to learn from, not a demo to reproduce.

---

## Findings

### F1. CAA gives an exact, cheap recipe: mean activation difference over minimal pairs, at one layer, normalized to the residual norm [source: "Steering Llama 2 via Contrastive Activation Addition", Rimsky/Panickssery et al., 2023/ACL 2024, https://arxiv.org/abs/2312.06681 · https://aclanthology.org/2024.acl-long.828/]

**What it is.** CAA (Contrastive Activation Addition) computes, for a behaviour *b*:

```
v_b^(L) = mean_i( h_i^(L)(x_i^+) ) − mean_i( h_i^(L)(x_i^-) )
```

where `(x_i^+, x_i^-)` are *minimal pairs* — prompts identical except for the behaviour-bearing span — and `h^(L)` is the residual stream at layer `L` at a **fixed, matched token position** (CAA uses the answer-token position, e.g. the `A`/`B` token in a multiple-choice item). At inference the vector is added at **every token position of the generated text, after the prompt ends**, with a signed coefficient. It is the same estimator as Turner et al.'s ActAdd [source: "Activation Addition: Steering Language Models Without Optimization", Turner et al., 2023, https://arxiv.org/abs/2308.10248 — later retitled "Steering Language Models With Activation Engineering"], packaged with a contrastive-dataset methodology and an evaluation suite.

Three details that are load-bearing and usually dropped in summaries:

1. **Normalization.** The vector is scaled to the *mean residual-stream norm at that layer*: `v ← v/‖v‖ · E‖h^(L)‖`. Coefficients are therefore in units of "one average residual-norm worth of push". `coef = 1` is a large perturbation; `coef = ±0.25` is subtle. Raw, unnormalized coefficients are not comparable across layers or models. (The released code does this; the exact sweep grid used per experiment is **UNVERIFIED** here — re-read the repo before quoting a number. Working range in practice: ±0.5 to ±3.)
2. **Negative steering is stronger than positive.** The paper reports this explicitly across 7B/13B/70B. Asymmetry is the norm, not a bug: pushing *away* from a behaviour tends to collapse the manifold faster than pushing toward it. Any dose–response curve must be run on both sides of zero.
3. **It is "effective over and on top of" finetuning and system-prompt design, while "minimally reducing capabilities"** — i.e. the paper's own claim is *small but nonzero* damage, which is exactly the quantity we must measure (F9).

**Why it matters for us.** This is our primary extractor. It needs no gradients, no training, no GPU, and ~200 pairs. It is also the method with the most independent replication, including a critical one (F9).

**Confidence:** high on the mechanism and normalization; medium on exact hyperparameters (layer choice is model-specific and must be swept).
**Cost:** extraction is one forward pass per prompt with `output_hidden_states=True`. 200 pairs × ~120 tokens ≈ 24k token-forwards ≈ **10–20 CPU-minutes per layer** batched at 8. A 5-layer sweep ≈ 1–2 h.

---

### F2. RepE formalizes "reading vectors" and "control vectors" and adds LoRRA; the reading step is PCA on paired differences, not the mean difference [source: "Representation Engineering: A Top-Down Approach to AI Transparency", Zou et al., 2023, https://arxiv.org/abs/2310.01405 · code https://github.com/andyzoujm/representation-engineering]

**What it is.** RepE's pipeline is LAT (Linear Artificial Tomography): present paired stimuli (e.g. "pretend you're happy" vs "pretend you're sad"), collect layer activations, **mean-center within each pair, then take the first principal component across all pairs**. That first PC is the *reading vector* (used to monitor) and, added with a coefficient, the *control vector* (used to steer). LoRRA (Low-Rank Representation Adaptation) is the trained variant: a LoRA adapter fit so that the model's representations align with a target direction — this one *does* need gradient steps.

**Why it matters.** Three things we should steal:
- The **PCA-vs-diff-in-means distinction is real but small in practice.** Diff-in-means is the class-mean difference; PCA-on-paired-differences finds the direction of maximal variance in the difference distribution. They coincide when the difference distribution is roughly isotropic around its mean, which is often true and sometimes not. Computing both and reporting their cosine similarity is a 5-line diagnostic that tells you whether your concept is "one direction" at all.
- RepE is explicit that **the same vector serves reading and writing**. That is the conceptual basis for the probe-gated scheduler in F7: one object, two uses.
- **LoRRA is out of scope for us.** It requires backprop through a 1.5B model on CPU. Say no.

**Confidence:** high.
**Cost:** free — PCA on a `[N, 1536]` matrix is milliseconds.

---

### F3. ITI is the per-head variant: probe each attention head, intervene on the top-K, shift by α·σ along the probe direction [source: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model", Li et al., NeurIPS 2023, https://arxiv.org/abs/2306.03341]

**What it is.** Train one logistic probe per attention head per layer on the head's contribution to the residual stream (`p = σ(⟨θ, x_h^l⟩)`), rank heads by validation probe accuracy, keep the top-K, and at inference add `α · σ_h^l · θ̂_h^l` to each selected head's output, where `σ_h^l` is the standard deviation of that head's projection onto `θ̂` over the calibration set. Reported result: Alpaca truthfulness 32.5% → 65.1% on TruthfulQA, with an explicit **truthfulness/helpfulness tradeoff tuned by α**.

**Why it matters.** Two transferable ideas:
- **Per-head granularity is a real alternative to whole-residual steering**, and it is *more surgical* — the paper's own ablation finds that intervening on all heads at once performs worse than a sparse top-K set. If whole-layer steering damages fluency on our 1.5B model (likely — small models have less redundancy), per-head is the fallback.
- **`σ`-scaling** is the per-head analogue of CAA's residual-norm scaling: the intervention is expressed in units of natural variation along that direction, which makes α comparable across heads.

Implementation note specific to us: in HF `Qwen2DecoderLayer`, the concatenated per-head outputs are the **input to `self_attn.o_proj`** (`[B, T, 12·128] = [B,T,1536]`). A `forward_pre_hook` on `o_proj` therefore gives clean per-head slicing with no surgery on attention internals. With GQA (12 query heads, 2 KV heads) only the 12 query heads are intervenable.

**Confidence:** high on method and result; **UNVERIFIED** on the exact released defaults (top-K and α) — the paper's Llama-7B configuration uses a few tens of heads and α in the 10–20 range; read `honest_llama` before quoting.
**Cost:** probe training is sklearn-scale (seconds). K probes × layers is the only real cost.

---

### F4. "Refusal is a single direction" is the cleanest causal demo in the literature — and a published jailbreak primitive [source: "Refusal in Language Models Is Mediated by a Single Direction", Arditi et al., NeurIPS 2024, https://arxiv.org/abs/2406.11717]

**What it is.** For 13 chat models up to 72B: compute a difference-in-means vector between mean activations on harmful vs harmless instructions, per layer, at the last token position; pick the single layer/direction with the best validation effect. Then:

- **Directional ablation** — `h ← h − (h·r̂)r̂` at all layers and all positions — removes refusal on harmful prompts.
- **Addition** — `h ← h + r̂` (swept) — induces refusal on harmless prompts.
- **Weight orthogonalization** — the same edit folded into the weights, so the model is permanently "abliterated" by a rank-one edit.

**Why it works across layers.** The residual stream is additive and shared: a direction written at layer *l* persists into *l+1…27* unless a component actively removes it, and every layer's read/write matrices see the same basis. Arditi et al. show the per-layer difference-in-means directions are highly correlated after mid-stack, which is why one direction generalizes up and down the stack. This is also the argument for *why any* mid-stack steering vector works at all, and it is the mechanistic justification for `resid_pre` being the right hook point.

**Note on the estimator.** Directional ablation is *steering with an adaptive coefficient*: `h − (h·r̂)r̂ ≡ h + c·r̂` with `c = −(h·r̂)`. Ablation and addition are the same operation with a per-position coefficient; that unification is worth keeping in the code (`mode="ablate"` is `coef="auto"`).

**Dual-use and honesty.** This is a *known, published, trivially reproducible* jailbreak primitive, and the paper says so in its own abstract. We describe the mechanism and its defensive implications; we do **not** operationalize it, and we do not build an "abliteration" tool. Defensive reading: (i) refusal behaviour is a rank-one feature of current safety tuning, hence brittle — which motivates representation-level defenses such as Representation Rerouting / circuit breakers [source: "Improving Alignment and Robustness with Circuit Breakers", Zou et al., NeurIPS 2024, https://arxiv.org/abs/2406.04313], which *train* the model to reroute harmful representations rather than relying on a single linear feature; (ii) probe-based monitoring of the same direction is a cheap detector; (iii) the technique transfers across layers, so defending one layer is not enough.

**Two corrections the field has since made, which we must not ignore:**
- Refusal is **not** strictly one-dimensional. Gradient-based search finds multiple independent directions and multi-dimensional *concept cones* [source: "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence", Wollschläger et al., ICML 2025, https://arxiv.org/abs/2502.17420]. Orthogonality ≠ independence under intervention.
- Across refusal *categories*, directions are geometrically distinct; different directions change *how* the model refuses more than *whether* it does [source: "There Is More to Refusal in Large Language Models than a Single Direction", Joad et al., EMNLP 2026, https://aclanthology.org/2026.emnlp-main.2606].

**Project-specific caveat (important).** Our target is `Qwen2.5-Coder-1.5B-**Base**`. A base model has no instruction tuning and no refusal circuit. Extracting "the refusal direction" from it is meaningless. The *method* — diff-in-means, validate by ablation and addition, check cross-layer transfer — is what we adopt.

**Confidence:** high on mechanism and on the critiques; high that it does not apply to our base model as a refusal demo.
**Cost:** identical to F1.

---

### F5. Alternative vector sources, ranked for a CPU-only 1.5B: diff-in-means ≻ linear probe ≈ LDA/LEACE ≻ PCA ≻ SAE (and SAEs lose to simple baselines for steering) [sources: "Sparse Autoencoders Find Highly Interpretable Features in Language Models", Cunningham et al., 2023, https://arxiv.org/abs/2309.08600 · "Scaling Monosemanticity", Templeton et al., 2024, https://transformer-circuits.pub/2024/scaling-monosemanticity · "Gemma Scope", Lieberum et al., 2024, https://arxiv.org/abs/2408.05147 · "Improving Steering Vectors by Targeting Sparse Autoencoder Features" (SAE-TS), Chalnev/Siu/Conmy, 2024, https://arxiv.org/abs/2411.02193 · "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders", 2025, https://arxiv.org/abs/2501.17148]

| source | formula | needs | verdict for us |
|---|---|---|---|
| **diff-in-means (CAA)** | `μ⁺ − μ⁻` | labelled pairs | **default**; best effect/effort ratio |
| **linear probe normal** | `w` from logistic regression; use `w/‖w‖` | labels + fitting | near-identical to diff-in-means when classes are balanced; use when you also want a *gate* (F7) |
| **LDA / whitened diff** | `Σ_w⁻¹(μ⁺−μ⁻)` | labels + covariance | better when within-class variance is anisotropic; 1536×1536 solve is trivial on CPU |
| **LEACE** | closed-form concept erasure, minimal damage | labels | the principled version of "ablate this concept"; use for *removal* experiments [source: "LEACE: Perfect linear concept erasure in closed form", Belrose et al., NeurIPS 2023, https://arxiv.org/abs/2306.03819] |
| **PCA on activations** | first PC of centred acts | unlabelled | diagnostic only — the first PC of raw activations is usually a norm/frequency direction, not the concept. PCA on *paired differences* (RepE) is the usable form |
| **SAE feature / decoder column** | `d_i` from the SAE decoder | a trained SAE | **out of scope** |

**The SAE story, honestly.** SAE features are interpretable and *clampable* (`h ← h + (target − f_i(h))·d_i`), and Anthropic's monosemanticity work shows feature steering at scale. But: (a) **AxBench (2025) finds diff-in-means and linear probes beat SAE-based steering on both concept detection and steering**, with SAEs underperforming despite far higher cost; (b) SAE-TS exists precisely because raw SAE decoder directions have unpredictable side effects; (c) no public SAE suite for Qwen2.5-Coder-1.5B is known to us (**UNVERIFIED**), and training one on 8 CPU cores is not a round-1 task. Recommendation: **do not put SAEs on the critical path.** If A2 wants them for *interpretation*, that is a separate, later track.

**Cross-vector interference is measurable and must be measured.** With a set of k vectors at the same layer, compute the `k×k` cosine Gram matrix. Pairs with `|cos| > 0.5` are not independent concepts, and their effects will leak. Report this matrix next to the dose–response curves.

**Confidence:** high on the ranking (AxBench is a direct head-to-head); high that SAE training is infeasible here.
**Cost:** LDA/LEACE: seconds. SAE: GPU-scale, rejected.

---

### F6. Injection mechanics: which hook, which positions, what the KV cache does, and why `‖h‖` is the only sane unit [source: synthesized from F1–F4 plus HF/TransformerLens semantics; hook-index equivalence verified against `transformers` `output_hidden_states` contract]

**Which hook.** Three sites, three physics:

| site | HF | TransformerLens | affects | caches? |
|---|---|---|---|---|
| mid-stack residual, pre-layer | `model.model.layers[L].register_forward_pre_hook` | `blocks.L.hook_resid_pre` | layers L…27, attention *and* MLP | **yes** — K/V for layers ≥ L are computed from the edited `h` |
| last-layer residual, post-layer | forward hook on `layers[27]` | `blocks.27.hook_resid_post` | final norm + `lm_head`, plus layer-27 attention output | yes (layer 27 K/V) |
| post-final-norm | forward hook on `model.model.norm` | hook on `ln_final` | `lm_head` only | **no** |

**Index equivalence (use this as a test, not a belief):** in HF, `out.hidden_states[0]` is the embedding output and `out.hidden_states[L]` is the output of layer `L−1`, i.e. **`out.hidden_states[L] == cache["blocks.L.hook_resid_pre"]`**. Assert it once in CI; if it ever fails, the extraction and the injection disagree about what "layer 14" means and every result is garbage.

**Positions.** Four modes, all of which we need:

- `all` — every position, prefill and decode. Strongest; also steers the prompt, so the *prompt's* K/V are contaminated and the first token is already affected.
- `prompt_last` — only the final prompt position during prefill, then every decode step. This steers the **first** generated token.
- `gen` — prefill untouched; steering begins on the first *decode* step. This is CAA's semantics and the right default for "steer the generation, not the question". **Consequence: the first sampled token is always unsteered.** That is architectural, not a bug — the state that produced token 1 was computed before any generated token existed.
- `post_norm` — cache-free (see below).

**What happens to the KV cache (the part that matters for mid-generation tuning).** At decode step *t*, only one token is computed (`[B,1,D]`). The hook edits `h_t` at `resid_pre(L)`; layer `L`'s attention then computes `K_t, V_t` **from the edited `h_t`** and appends them to the cache. Layers `0…L−1` for that token were already computed and cached unsteered. Every future token attends to the edited K/V.

Therefore:

- A coefficient schedule is **path-dependent**. `[1,1,2]` and `[2,2,2]` end at the same coefficient but produce different sequences, because the early tokens' cached K/V differ. Steering is *history*, like a stain in the residual stream.
- You **cannot** retroactively change a past token's contribution without recomputing it.
- To get an *a-temporal* (stateless) intervention — "what would the model say if this coefficient had been active all along?" — you must either recompute the whole prefix each step (`O(T²)`, unusable past ~100 tokens on CPU) or intervene at `post_norm`, where nothing is cached.

**Recommendation:** expose `cache_semantics ∈ {"history", "stateless", "output_only"}` as an explicit parameter of `steer()`. Default `history` (cheap, and honestly the more interesting object: it is what "tuning it mid-generation" actually means). Provide `stateless` only for short sequences and for the ablation study that quantifies the difference.

**Post-norm steering is a logit bias in disguise — and this is the key theoretical bridge.** Adding `v` after `model.norm` gives `logits = W_U(h_norm + v)`, i.e.

```
Δlogits = W_U · v
```

a **fixed additive bias on all 151,936 token logits**, of rank ≤ d_model and confined to the row space of `W_U` (and if embeddings are tied — `config.tie_word_embeddings`, true for the 1.5B Qwen2.5 family but **UNVERIFIED** here — `W_U = E`, so the bias is `⟨e_j, v⟩`, a dot product with each token embedding). So "activation steering" and "logit bias" are the *same operation* at different depths; the difference is that mid-stack steering is re-processed by the remaining layers and therefore *generalizes*, while post-norm steering is a one-shot distributional nudge that cannot.

This gives a free audit: `implied_logit_bias(model, v)` prints the top-20 tokens the post-norm vector favours. If a "concise" vector's implied bias is dominated by `"\n"` and `" "`, you have learned a formatting artefact, not a concept.

**Why normalize by `‖h‖`.** Residual norms grow with depth and vary per token and per prompt. A fixed absolute `coef` therefore means a *different relative perturbation* at every layer and every position, and results do not transfer. Two normalizations, both defensible:
- `dataset`: `coef · E‖h^(L)‖ · v̂` — CAA's choice, reproducible, comparable to the literature.
- `per_token`: `coef · ‖h_t‖ · v̂` — keeps the relative perturbation `‖Δh‖/‖h‖` constant across tokens; better for heterogeneous prompts.

Either way, **report `‖Δh‖/‖h‖` per step** as the interpretable x-axis. Rules of thumb: 0.01–0.05 subtle, 0.05–0.2 strong but usually coherent, >0.5 expect degeneration. (Thresholds are empirical and model-specific — treat as priors to be measured, not constants.)

**One layer or all layers?** Mid-stack single-layer steering works because of residual-stream persistence (F4). Arditi et al. ablate at *all* layers because they want the direction gone everywhere, including re-injection by later layers. For *addition*, single-layer at a swept-best layer is the standard and is far easier to attribute causally. Do not steer at multiple layers in round 1 — you lose the ability to say which layer did it.

**Confidence:** high on all mechanics; the post-norm equivalence is elementary linear algebra and is exactly checkable numerically (do it: `allclose(model(...).logits, base_logits + F.linear(v, W_U))`).
**Cost:** zero — it is the same forward pass.

---

### F7. Conditional / dynamic steering is the actual answer to "tune it mid-generation" — and there is now a real literature on it [sources: CAST, Lee et al., ICLR 2025, https://arxiv.org/abs/2409.05907 · FASB "Steering When Necessary", Gan et al., 2025, https://arxiv.org/abs/2508.17621 · DSEM "Dynamic Steering with Episodic Memory", ACL Findings 2025, https://aclanthology.org/2025.findings-acl.706.pdf · SADI, Wang et al., ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/c4d26a95fd83f8e590f81c54ae670b5d-Abstract-Conference.html · IDS, Vogels et al., 2025, https://arxiv.org/abs/2510.13285]

Five designs, in increasing order of what they actually control:

1. **CAST — condition vector gates behaviour vector (prompt-level).** A *behaviour vector* `v` (e.g. refusal) plus a *condition vector* `c` (e.g. "harmful"). At inference: `if cos(h, ĉ) > τ then h ← h + α·v`. Threshold τ, layer, and comparison direction are grid-searched to maximize F1 on held-out condition examples. Their headline numbers, on **Qwen 1.5 Chat 1.8B** (same family as our target — useful): unconditional refusal steering gives 100% refusal on harmful *and* 96.4% on harmless prompts; CAST gives 90.7% / 2.2%. The gate is a **thresholded cosine similarity, not a semantic policy** — and their own ablation shows it is fragile to threshold choice.
2. **FASB — per-token state tracking with backtracking.** Probe the *generated* token's activation each step; if the deviation probability exceeds β, **rewind s tokens**, compute an intervention strength from the probe value, and regenerate. This is the only design in the literature that explicitly handles "we noticed too late", and it is the closest thing to the user's stated goal. Cost: a probe forward per token plus regeneration on trigger.
3. **DSEM — token-level vectors from episodic memory.** Rather than one vector for the whole generation, retrieve a steering vector per token-step from a memory of demonstration activations, interpolating between a local (memory) and global vector. Training-free.
4. **SADI — semantics-adaptive dynamic vector.** Identify the critical heads/hidden states/neurons from contrastive pairs, then scale *element-wise* along the input-semantics direction at inference. i.e. the vector itself becomes input-dependent.
5. **IDS — adapt strength to distributional distance.** Scale `α` by how far the current representation sits inside the training distribution: intervene hard when off-manifold, back off when in-distribution. Motivated exactly by the coherence collapse that fixed-α steering causes.

**What this means for our design.** None of these is "a scheduler over `coef`" in the trivial sense; all of them are *closed-loop*: something is measured during generation and the intervention responds. The minimal closed loop we can afford on CPU is:

```
gate_t = σ(⟨θ, h_t⟩)                       # one dot product, at the steered layer
g_t    = λ·g_{t-1} + (1−λ)·gate_t          # low-pass, avoids per-token flicker
coef_t = c_max · 1[g_t > τ]                # Schmitt trigger
```

Three engineering details that separate a working loop from a noisy one:
- **Compute the gate on the *unsteered* `h`**, before adding. Otherwise the gate sees its own output and latches.
- **Hysteresis / minimum on-duration.** A per-token gate with no memory produces alternating steered/unsteered tokens, which is often worse than either extreme. A low-pass filter plus `min_run` (e.g. 4 tokens) fixes it.
- **Log the gate.** Every step's `gate_t`, `coef_t`, and `‖Δh‖/‖h‖` go into the trace. Without them you cannot distinguish "the gate never fired" from "the vector did nothing" — the two most common false conclusions in this literature.

**Also: coefficient *schedules* are a degenerate special case** and are what most papers actually do (constant, linear ramp, cosine decay, step). Turner et al. used decay for style transfer; CAA uses constant. Keep them as `schedule="constant"|"linear"|"cosine"|"step"`, but understand that a schedule is open-loop and cannot respond to the model.

**Confidence:** high on CAST/FASB/DSEM/SADI/IDS as described; medium on how well any of them transfers to a 1.5B base model (all were evaluated on chat/instruct models ≥1.8B).
**Cost:** gate = 1 dot product per token ≈ free. Probe training = seconds. Backtracking = `s` extra forwards per trigger.

---

### F8. Steering is not control: logit bias, banned tokens, grammars and forcing are a different instrument with different guarantees [source: HF `LogitsProcessor` API; `outlines`/`lm-format-enforcer`/`xgrammar`; plus the post-norm equivalence from F6]

| goal | instrument | guarantee | generalizes? | notes |
|---|---|---|---|---|
| never emit token X | `SuppressTokensLogitsProcessor` / additive bias `−inf` | **exact** | n/a | per-token-id; brittle if X is a substring of a needed token |
| answer must parse as JSON/SQL | grammar-constrained decoding (FSM over vocab) | **exact** (by construction) | n/a | `outlines`, `lm-format-enforcer`, `xgrammar`; cost ≈ negligible per token |
| must start with ```` ```python ```` | `PrefixConstrainedLogitsProcessor` / forced prefix | **exact** | n/a | the model may produce incoherent continuation after the forced span ("forced-token whiplash") |
| prefer a *style*: concise, commented, cautious, formal | **activation steering** | **soft, statistical** | **yes** | needs the full validation harness (F9); no per-token guarantee |
| suppress a *category* of content | classifier-gated steering (CAST) or output filter | soft | partly | defense-in-depth, never a guarantee |

**The distinction that matters.** Forcing a token changes **one categorical draw**. Steering changes **the state that generates every subsequent draw**. Consequences:

- Forcing is auditable, exact, and locally brittle: it can make the model say a word it has no state to continue from.
- Steering is general (it survives paraphrase, affects tokens you never enumerated) and **unverifiable without an eval harness** — there is no per-token predicate that says "this was steered correctly".
- They **compose**: grammar masks the support, steering reweights within it. "Structured output with a controlled style" is `steering + grammar`, and that combination is strictly more powerful than either.

**The formal bridge.** Post-norm steering *is* a logit bias (F6): `Δlogits = W_U·v`. So the spectrum is continuous — early-layer steering → late-layer steering → additive logit bias → hard mask → forced token — and the only thing that changes along it is how much nonlinear computation re-processes the intervention. Frame the tool that way in the UI: one slider (depth) and one slider (strength), with "hard constraint" at the end of the depth axis.

**Practical API note.** In `transformers`, pass `logits_processor=LogitsProcessorList([...])` to `generate()`; processor names and signatures drift between minor versions (`NoBadWordsLogitsProcessor` → `SuppressTokensLogitsProcessor`/`SequenceBiasLogitsProcessor`), so pin the version and assert behaviour in a test.

**Confidence:** high.
**Cost:** free.

---

### F9. Validation: the field's own reliability paper says steering is *less* reliable than the demos suggest, and Qwen was among the worst [source: "Analysing the Generalisation and Reliability of Steering Vectors", Tan et al., NeurIPS 2024, https://arxiv.org/abs/2407.12404 · code https://github.com/dtch1997/steering-bench]

**What it found.** Across a broad behaviour suite: (i) **in-distribution steerability varies wildly per input** — the same vector at the same coefficient moves some prompts and not others, and spurious dataset biases explain much of the apparent effect; (ii) **steerability is mostly a property of the dataset, not the model**; (iii) out-of-distribution, vectors often generalize but for several concepts are **brittle to reasonable prompt changes**; (iv) they explicitly note degradation is **particularly pronounced for Qwen**. So: expect our target model to be a *hard* case, and design the eval to detect null results rather than to produce positive ones.

**The protocol I will hold A6 to.** Ordered by how much each item protects against self-deception:

1. **Harness identity test (non-negotiable).** Hook installed, `coef = 0` ⇒ logits **bit-identical** to no hook (`torch.equal`, not `allclose`). This catches dtype, in-place, and position-masking bugs that otherwise masquerade as effects. Run it in CI on every commit.
2. **Negative controls, all four:**
   - matched-norm Gaussian random vector, same layer, same positions;
   - the true vector at a random *layer*;
   - a vector extracted from **shuffled labels** (should be null — this is the placebo that catches leakage in the pair construction);
   - the true vector with the target direction projected out (should be null).
3. **Dose–response, full curve, both signs.** `coef ∈ {−3…+3}`, ≥7 points, report the whole curve. Non-monotonicity is a finding, not something to smooth over. Also report the curve for the random control on the same axes.
4. **Held-out prompts, two ways.** (a) extract on split A, evaluate on split B; (b) evaluate on a *distribution shift* (paraphrase, different length, prose vs code, different language). Report both. The gap between them is the honest generalization number.
5. **Damage.** Perplexity on held-out text (and on the model's own domain — code), plus one capability probe (e.g. 50–100 unit-test-passing completions, or a small MMLU slice). Report Δ at every coefficient. "Minimally reduces capabilities" (CAA's phrase) is a number, so measure it.
6. **Off-target matrix.** ~8–10 unrelated behaviours measured with the *same* instrument (length, sentiment, language, refusal, hedging, code correctness, comment density, …). Steering should be selective; the full matrix is the evidence.
7. **Non-circular measurement.** If the vector came from probe `P`, the outcome must be measured by something that is not `P`: a different probe trained on different data, a programmatic check (AST, regex, unit test), or human/LLM-judge labels. Otherwise the result is a tautology.
8. **Causality, three ways.** (a) add → effect; (b) ablate/negate → opposite effect; (c) **patching**: transplant `h` from a positive prompt into a negative one at layer L and show the behaviour follows the transplant. (c) is what separates "correlated direction" from "causal mediator".
9. **Variance.** ≥30 prompts, ≥3 seeds where sampling is on; report CIs, not point estimates. On CPU at 5–20 tok/s this is affordable for short generations and is the difference between a result and an anecdote.
10. **Baseline headroom.** Choose prompts where the unsteered model does the target behaviour ~20–50% of the time. If the baseline is at ceiling, no intervention can show an effect; if at floor, any change looks like success.

**Confidence:** high — this is a direct, published, adversarial evaluation of the exact technique we plan to use.
**Cost:** the dominant cost. Budget ~3–4 CPU-hours for a complete round-1 study (extraction + sweep + controls).

---

## Concrete implementation (code sketches, real APIs)

All sketches target `Qwen2ForCausalLM` (28 layers, d_model 1536, 12 q-heads / 2 kv-heads, head_dim 128) and are written for `transformers` ≥4.40 with `DynamicCache`. They are deliberately dependency-light so they run on CPU.

### 1. Vector extraction — diff-in-means, with the residual-norm constant

```python
# code/steering/extract.py
import torch

@torch.no_grad()
def extract_diffmean(model, tok, pairs, layer, batch_size=8, max_len=256):
    """pairs: list[(pos_text, neg_text)] minimal pairs.
    Returns (v_unit[d_model], resid_norm: float).

    Uses out.hidden_states[layer] == blocks.{layer}.hook_resid_pre
    (hidden_states[0] is the embedding output, so index L == input to layer L).
    """
    pos_acts, neg_acts, norms = [], [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        for texts, sink in (([p for p, _ in chunk], pos_acts),
                            ([n for _, n in chunk], neg_acts)):
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer]                      # [B, T, D]
            last = enc["attention_mask"].sum(1) - 1           # pad-safe
            v = h[torch.arange(h.shape[0]), last].float()     # [B, D]
            sink.append(v)
            norms.append(v.norm(dim=-1))
    P, N = torch.cat(pos_acts), torch.cat(neg_acts)
    assert P.shape == N.shape, "unbalanced contrast sets: use mean-of-differences"
    mu = P.mean(0) - N.mean(0)
    return mu / mu.norm(), torch.cat(norms).mean().item()


@torch.no_grad()
def extract_lda(pos_acts, neg_acts, ridge=1e-3):
    """Whitened difference: Σ_w⁻¹(μ⁺−μ⁻). Better when within-class variance
    is anisotropic. 1536×1536 solve is milliseconds on CPU."""
    mu = pos_acts.mean(0) - neg_acts.mean(0)
    X = torch.cat([pos_acts - pos_acts.mean(0), neg_acts - neg_acts.mean(0)])
    S = X.T @ X / (len(X) - 1)
    S = S + ridge * S.diag().mean() * torch.eye(S.shape[0])
    w = torch.linalg.solve(S, mu)
    return w / w.norm()


@torch.no_grad()
def extract_repE_pca(pos_acts, neg_acts):
    """RepE/LAT: first PC of the *paired difference* distribution."""
    D = pos_acts - neg_acts
    _, _, V = torch.pca_lowrank(D - D.mean(0), q=1)
    return V[:, 0]
```

Diagnostics to run immediately after extraction (all cheap, all informative):

```python
print("cos(diffmean, lda)  =", float(v_dm @ v_lda))
print("cos(diffmean, repE) =", float(v_dm @ v_pca))
print("resid_norm          =", resid_norm)   # measure, don't assume
```

### 2. `SteeringHook` — HF `forward_pre_hook` on a decoder layer

```python
# code/steering/hook.py
import torch

class SteeringHook:
    """h[:, pos, :] += coef * scale * v   at the input of decoder layer `layer`.

    positions: "all" | "prompt_last" | "gen" | "post_norm"
    norm:      "dataset" (coef * resid_norm) | "per_token" (coef * ||h_t||)
    gate:      optional callable(h_last[D]) -> float in [0,1]; multiplies coef.
    """

    def __init__(self, model, layer, v, coef=1.0, resid_norm=1.0,
                 positions="gen", norm="dataset", gate=None,
                 threshold=0.5, min_run=1, ema=0.0):
        self.model, self.layer = model, layer
        self.v = v.detach().float()
        self.v = self.v / self.v.norm()
        self.coef = float(coef)
        self.resid_norm = float(resid_norm)
        self.positions, self.norm = positions, norm
        self.gate, self.threshold = gate, threshold
        self.min_run, self.ema = min_run, ema

        self.enabled = True
        self.last_gate = 1.0
        self._g_ema, self._on_run = 1.0, 0
        self._delta_ratio = 0.0

        target = (model.model.norm if positions == "post_norm"
                  else model.model.layers[layer])
        self._handle = target.register_forward_pre_hook(self._pre)

    # ---- internals -------------------------------------------------
    def _gate_value(self, h_last):
        if self.gate is None:
            return 1.0
        raw = float(self.gate(h_last))
        self._g_ema = (self.ema * self._g_ema + (1 - self.ema) * raw) if self.ema else raw
        if self._g_ema > self.threshold:
            self._on_run += 1
        else:
            self._on_run = 0
        return float(self._g_ema > self.threshold and self._on_run >= self.min_run)

    def _delta(self, h):                      # h: [B,T,D]
        if self.norm == "dataset":
            scale = self.coef * self.resid_norm          # python float
        else:
            scale = self.coef * h.norm(dim=-1, keepdim=True)   # [B,T,1]
        return (scale * self.v).to(dtype=h.dtype)

    def _pre(self, module, args):
        if not self.enabled or self.coef == 0.0:
            return None                          # bit-identical to no hook
        h = args[0]
        if h.shape[1] == 0:
            return None

        if self.positions == "gen" and h.shape[1] > 1:
            return None                          # prefill untouched (CAA semantics)
        if self.positions == "prompt_last" and h.shape[1] > 1:
            g = self._gate_value(h[:, -1, :])
            if g == 0.0:
                return None
            delta = torch.zeros_like(h)
            delta[:, -1:, :] = self._delta(h[:, -1:, :]) * g
            return (h + delta,) + tuple(args[1:])

        g = self._gate_value(h[:, -1, :])
        if g == 0.0:
            self.last_gate = 0.0
            return None
        self.last_gate = g
        delta = self._delta(h) * g
        with torch.no_grad():
            self._delta_ratio = float(delta.norm() / h.norm().clamp_min(1e-6))
        return (h + delta,) + tuple(args[1:])

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
```

TransformerLens equivalent (TransformerLens 3 `TransformerBridge` supports Qwen and keeps the legacy alias; [source: TransformerLens 3 docs, 2025, https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html]):

```python
def tl_hook(layer, v, coef, resid_norm):
    def fn(value, hook):                       # value: [B,T,D] at resid_pre
        return value + (coef * resid_norm) * v.to(value.dtype)
    return f"blocks.{layer}.hook_resid_pre", fn

# bridge.run_with_hooks(tokens, fwd_hooks=[tl_hook(14, v, 1.0, rn)])
```

### 3. Scheduler + generation loop (coefficient changes between tokens)

```python
# code/steering/generate.py
import math, torch

class Scheduler:
    """coef(t) where t = number of tokens already generated.
    kind: constant | linear | cosine | step | probe
    """
    def __init__(self, kind="constant", c0=1.0, c1=0.0, n=32,
                 start=0, until=None):
        self.kind, self.c0, self.c1, self.n = kind, c0, c1, max(n, 1)
        self.start, self.until = start, until

    def __call__(self, t, hook=None):
        if t < self.start or (self.until is not None and t >= self.until):
            return 0.0
        if self.kind == "constant":
            return self.c0
        if self.kind == "linear":
            return self.c0 + (self.c1 - self.c0) * min(t / self.n, 1.0)
        if self.kind == "cosine":
            return self.c1 + (self.c0 - self.c1) * 0.5 * (1 + math.cos(math.pi * min(t / self.n, 1.0)))
        if self.kind == "step":
            return self.c0 if t < self.n else self.c1
        if self.kind == "probe":
            # NB: hook.last_gate was written by the *previous* forward pass, so this
            # form lags by one token. Prefer passing `gate=` to SteeringHook, which
            # gates inside the same forward pass and has no lag.
            return self.c0 * (hook.last_gate if hook else 1.0)
        raise ValueError(self.kind)


@torch.no_grad()
def generate_steered(model, tok, prompt, hook=None, sched=None,
                     max_new_tokens=64, temperature=0.0, eos_id=None,
                     trace=None):
    """Per-token loop. `hook.coef` is rewritten before every forward pass, so the
    coefficient genuinely changes mid-generation. Cache semantics are 'history':
    each token's K/V is computed under the coefficient active at that step."""
    enc = tok(prompt, return_tensors="pt")
    ids, mask = enc.input_ids, enc.attention_mask
    past, out_ids = None, []

    for t in range(max_new_tokens):
        if hook is not None and sched is not None:
            hook.coef = float(sched(t, hook))

        step_ids = ids if past is None else ids[:, -1:]
        out = model(input_ids=step_ids, attention_mask=mask,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values

        logits = out.logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial((logits / temperature).softmax(-1), 1)

        ids = torch.cat([ids, nxt], 1)
        mask = torch.cat([mask, mask.new_ones((1, 1))], 1)
        out_ids.append(int(nxt))

        if trace is not None:
            trace.append({
                "step": t,
                "token_id": int(nxt),
                "steer": None if hook is None else {
                    "layer": hook.layer,
                    "coef": hook.coef,
                    "gate": hook.last_gate,
                    "delta_norm_ratio": hook._delta_ratio,
                },
            })
        if eos_id is not None and int(nxt) == eos_id:
            break
    return ids, out_ids
```

**The `stateless` variant**, for the ablation that measures how much the path-dependence matters (short sequences only — this is `O(T²)`):

```python
@torch.no_grad()
def generate_stateless(model, tok, prompt, hook, sched, max_new_tokens=32):
    ids = tok(prompt, return_tensors="pt").input_ids
    for t in range(max_new_tokens):
        hook.coef = float(sched(t, hook))
        # no past_key_values: recompute the whole prefix under the current coef
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :].float()
        ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], 1)
    return ids
```

**Post-norm audit** — verify the logit-bias equivalence numerically, and read the implied bias:

```python
@torch.no_grad()
def implied_logit_bias(model, v, topk=20):
    W_U = model.lm_head.weight                      # [vocab, D]
    bias = W_U.float() @ v.float()                  # [vocab]
    vals, idx = bias.topk(topk)
    return [(int(i), float(x)) for x, i in zip(vals, idx)]

def assert_postnorm_equivalence(model, batch, v):
    base = model(**batch, use_cache=False).logits[:, -1, :]
    with SteeringHook(model, layer=None, v=v, coef=1.0, resid_norm=1.0,
                      positions="post_norm"):
        steered = model(**batch, use_cache=False).logits[:, -1, :]
    expect = base + (model.lm_head.weight.float() @ v.float())
    assert torch.allclose(steered, expect, atol=1e-3), "post-norm != logit bias"
```

### 4. Validation harness — the four controls and the dose–response

```python
# code/steering/validate.py
import torch, statistics

def identity_test(model, tok, hook, prompt):
    """coef=0 must be BIT-identical. Catches dtype / in-place / mask bugs."""
    batch = tok(prompt, return_tensors="pt")
    hook.enabled = False
    a = model(**batch, use_cache=False).logits
    hook.enabled, hook.coef = True, 0.0
    b = model(**batch, use_cache=False).logits
    return torch.equal(a, b)


def random_control(v, seed=0):
    g = torch.randn_like(v, generator=torch.Generator().manual_seed(seed))
    return g / g.norm() * v.norm()      # matched norm, random direction


def dose_response(model, tok, hook, prompts, coefs, metric_fn, gen_kw=None):
    gen_kw = gen_kw or {}
    rows = []
    for c in coefs:
        hook.coef = c
        texts = [generate_steered(model, tok, p, hook, Scheduler("constant", c),
                                  **gen_kw)[1] for p in prompts]
        rows.append({"coef": c,
                     "metric": metric_fn(texts),
                     "delta_ratio": hook._delta_ratio})
    return rows


@torch.no_grad()
def perplexity(model, tok, texts, max_len=512):
    nll, ntok = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        out = model(**enc, labels=enc.input_ids)
        n = enc.input_ids.shape[1] - 1
        nll += float(out.loss) * n
        ntok += n
    return float(torch.exp(torch.tensor(nll / max(ntok, 1))))
```

A complete round-1 experiment is then a 5×7 grid: `layer ∈ {8,12,16,20,24} × coef ∈ {−2,−1,−0.5,0,0.5,1,2}`, each with the random control on the same axes, plus PPL and the off-target matrix at the best cell. ~2–3 CPU-hours.

---

## What NOT to do (traps)

1. **Do not use in-place `h[:, pos, :] += ...`.** Views, autocast, bf16 and cached tensors make in-place mutation either silently ineffective or quietly corrupting. Build `delta` out-of-place and return `h + delta`.
2. **Do not skip the `coef=0` bit-identity test.** Every "steering works" result that later evaporates is, in our experience of this literature, a harness bug: wrong position mask, wrong layer index, or a hook that fired during prefill.
3. **Do not compare coefficients across layers.** Norms differ by an order of magnitude. Report `‖Δh‖/‖h‖` and the dataset-normalized coefficient, never a raw absolute one.
4. **Do not steer the prompt when you mean to steer the generation** (and vice versa). `positions` is a semantic choice with a visible off-by-one: `gen` leaves token 1 unsteered by construction.
5. **Do not report only the best coefficient.** Report the curve, both signs, and the random control on the same axes. Peak-picking is how non-effects become findings.
6. **Do not evaluate with the probe that built the vector.** Circular. Use a programmatic check, a held-out probe, or judge labels.
7. **Do not attribute an effect when the baseline is at ceiling or floor.** Check baseline behaviour rate first; target 20–50% headroom.
8. **Do not assume the KV cache is a detail.** A schedule is path-dependent. If you claim "coef 2 at step 20 gives X", that claim is about a *history*, not a *state*. Either say so or use `stateless`/`post_norm`.
9. **Do not steer at multiple layers in round 1.** You lose causal attribution. One layer, swept.
10. **Do not extract from a base model and call it "refusal".** `Qwen2.5-Coder-1.5B-Base` has no refusal circuit. Pick behaviours the model actually has: language mode, comment density, verbosity, docstring presence, error-handling style.
11. **Do not train an SAE on CPU.** AxBench already shows simple baselines beat SAE steering; the cost/benefit is negative by an order of magnitude.
12. **Do not build or ship an abliteration tool.** The refusal-direction result is a documented jailbreak primitive. Use it as evidence about brittleness of linear safety features and as a defence rationale (probe monitoring, representation rerouting), not as a capability.
13. **Do not trust a single seed / single prompt.** Tan et al. show per-input steerability is highly variable and Qwen is a particularly brittle case.

---

## Open questions for other agents

- **A2 (interp):** Which probe should drive the per-token gate — same layer as the steering vector, or an earlier one (cheaper, less contaminated by the intervention)? Can `attn_entropy` and `mlp_out_norm` at the steered layer be emitted *for the steered forward pass* so the trace shows the perturbation's footprint?
- **A2:** Does the logit-lens trajectory visibly bend at the steering layer? That is the strongest visual evidence that the intervention took effect where we claim, and it is nearly free.
- **A4 (runtime):** Hook lifecycle — one persistent handle mutated per step (as above) vs attach/detach per forward? Does the streaming generator expose `step` to a scheduler callback, or must the scheduler live inside the loop? Who owns cache invalidation for `stateless` mode?
- **A4:** Batched generation with per-sequence coefficients (needed for dose–response throughput). `forward_pre_hook` sees `[B,T,D]`; the scale must then be `[B,1,1]`. Does the runtime support that, or do we serialize?
- **A5 (UI):** The two-slider model from F8 (depth × strength) plus a live `‖Δh‖/‖h‖` readout and a gate trace strip. The dose–response curve should be a *plotted artefact*, not a number.
- **A6 (limits):** Pre-register the success threshold per behaviour before the sweep, and specify the minimum sample size for a "confirmed" result. I propose: ≥30 held-out prompts, effect ≥ 2× the random-control 95th percentile, monotone-ish over ≥3 coefficients.
- **A1 (formats):** Does a Q4_K_M model steer with the *same* vector and coefficient? The vector lives in the fp16 residual basis, but the quantized model's residual stream is computed differently (dequantized matmuls, possibly different accumulation). Prediction: same direction works, effective magnitude differs. Worth one experiment — it decides whether steering is a "format-independent" feature of the ModelAdapter.

---

## SCHEMA CHANGE

Two additive changes. Both are optional fields, so existing emitters stay valid.

**1. Per-step `steer` block** — without it, a trace cannot distinguish "gate never fired" from "vector did nothing", which is the single most common false negative in this literature.

```json
{"step": 12, "token_id": 4021, "token_str": "def",
 "steer": {"layer": 14, "coef": 1.25, "gate": 0.83,
           "delta_norm_ratio": 0.041, "vector_id": "v_comments_l14"},
 "layers": [...], "logits_topk": [...], "chosen": [...]}
```

**2. Top-level `steering` metadata block** — needed by A6 to verify and by A5 to render.

```json
"steering": {
  "vector_id": "v_comments_l14",
  "sha256": "…",                 // of the fp32 vector bytes
  "extractor": "diffmean",       // diffmean | lda | pca | probe | sae
  "layer": 14,
  "norm": "dataset",             // dataset | per_token
  "resid_norm": 42.7,
  "positions": "gen",            // all | prompt_last | gen | post_norm
  "schedule": "cosine",
  "cache_semantics": "history"   // history | stateless | output_only
}
```

**3. Proposed `steer()` signature refinement** (backward compatible with CONTEXT.md):

```python
steer(layer=14, vector=v, coef=2.0,
      positions="gen",              # all | prompt_last | gen | post_norm
      schedule="constant",          # or a callable coef(t) -> float
      norm="dataset",               # dataset | per_token
      start_step=0, until_step=None,
      gate=None,                    # callable(h_last) -> float in [0,1]
      gate_threshold=0.5, gate_min_run=1, gate_ema=0.0,
      cache_semantics="history")
```

---

## Build next (ordered, with effort estimate)

| # | deliverable | why first | effort |
|---|---|---|---|
| 1 | `code/steering/extract.py` — diff-in-means + resid-norm + LDA/PCA variants, with the `hidden_states[L] == hook_resid_pre` assertion | everything downstream depends on a correct extractor | 2 h |
| 2 | `code/steering/hook.py` — `SteeringHook` with all four `positions` modes + `identity_test()` in CI | the identity test is the cheapest possible insurance | 2 h |
| 3 | `code/steering/generate.py` — per-token loop + `Scheduler` (constant/linear/cosine/step/probe) | this *is* the mid-generation capability | 3 h |
| 4 | Contrast-pair datasets for two **measurable** code behaviours: comment density (same function ± comments) and language mode (same algorithm in Python vs JS). 100–200 pairs each | measurable targets make validation programmatic, not judgemental | 2 h |
| 5 | `code/steering/validate.py` — dose–response + PPL + 4 negative controls + off-target matrix | turns effects into findings | 3 h |
| 6 | Layer sweep: 5 layers × 7 coefficients × 20 prompts, both behaviours, random control on same axes | the first real result; also picks the production layer | 2–3 h **compute** |
| 7 | `code/steering/bias.py` — post-norm equivalence test + `implied_logit_bias()` audit | proves the steering↔forcing bridge; catches formatting artefacts | 1 h |
| 8 | Probe-gated per-token scheduler (EMA + hysteresis + `min_run`), compared against constant and cosine | the actual "tune it mid-generation" experiment | 3 h |
| 9 | Logit-bias / grammar comparison arm (same behaviour, hard-constrained) | answers "when should we force instead of steer" with data | 1 h |
| 10 | Write-up: 2 dose–response figures, 1 off-target heatmap, 1 gate trace, 1 vector Gram matrix | the deliverable that A5 renders and A6 audits | 2 h |

**Total: ~19 h engineering + ~4 h CPU compute.** Steps 1–3 are the critical path; 4–6 produce the first publishable result; 7–9 are the parts that make it *science* rather than a demo.
