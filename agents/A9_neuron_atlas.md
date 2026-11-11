# A9 — Exhaustive neuron & head enumeration ("the neuron atlas") for Qwen2.5-Coder-1.5B

Scope: can we profile **every** MLP neuron (250,880) and **every** attention head (336) of a
1.5B model on 8 CPU cores, and what does that buy us? Target: `Qwen/Qwen2.5-Coder-1.5B-Base`,
28 layers × 8960 intermediate = 250,880 neurons; 28 × 12 = 336 query heads (2 KV heads, 6:1 GQA).

**All compute numbers below were measured on this box, not estimated.** See F11 — the CPU is a
4-core 15 W mobile i7-8665U with AVX2 only, which changes the budget by roughly 5× versus what
`CONTEXT.md` assumes. That is the single most important finding in this file.

---

## TL;DR (5 bullets)

- **The census is achievable; the concept dictionary is not.** We can give every one of the
  250,880 neurons a statistically-grounded, corpus-relative profile in ~17 h of compute, and every
  one of the 336 heads a *causally verified* profile in ~30 min. But expect only **1–5 %** of
  neurons to earn a clean, causally-checked single-concept label — that number is measured in prior
  work (Gurnee et al. 2024: 1–5 % "universal"; Bills et al. 2023: 0.33 % of GPT-2 small's neurons
  got auto-explanations scoring > 0.8).
- **Exhaustive causal ablation of all 250,880 neurons is ~14 days of continuous compute.** Do not
  do it. Instead: one **backward pass per token yields an attribution for all 250,880 neurons at
  once** (gradient × activation / AtP-style), giving a full causal *ranking* in ~4.5 h, then
  *verify* the top 2,000–5,000 with real ablations in 3–7 h.
- **Label neurons objectively, not with an LLM.** A 1.5B local labeler is far too weak to do
  Bills-et-al-style explain-a-neuron. But this is a *code* model, so we have a free, exact,
  non-hallucinating label channel: **tree-sitter AST context** (`for_stmt`, `call.func`,
  `subscript.slice`, …) plus **vocabulary promotion** (`W_U · W_down[:, j]`, Geva-style). Those two
  give a defensible label with zero LLM in the loop.
- **Heads are the opposite case: exhaustive is cheap and should be done exhaustively.** 336
  mean-ablations, batched, ≈ 30 min. Induction score (Olsson et al. 2022), attention-sink mass
  (Xiao et al. 2023), entropy, prev-token score, and Δlogit — all measured for every head, no
  sampling.
- **Storage is a non-issue; the bottleneck is tokens-per-second.** The finished atlas is
  **~250 MB** on disk (one parquet of 250,880 rows ≈ 150 MB raw / 50–70 MB compressed, plus a
  100 × 250,880 fp16 context-class matrix = 50 MB). But we can only afford **~500 k corpus
  tokens** in a 2-day budget, which means neurons firing on < 0.1 % of tokens are statistically
  under-sampled and must be flagged as such rather than silently labelled.

---

## Findings

### F1. The "1 neuron = 1 concept" map is empirically falsified, and the falsification is quantified
[source: Universal Neurons in GPT2 Language Models, 2024, https://arxiv.org/abs/2401.12181]
[source: Language models can explain neurons in language models, 2023, https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html]

- **What it is.** Gurnee et al. computed pairwise correlations of *every* neuron pair across five
  GPT-2 models trained from different seeds, over **100 M tokens**, and found **1–5 %** of neurons
  are "universal" (consistently activate on the same inputs across seeds). They argue universality
  is a *proxy for interpretability* — i.e. only that 1–5 % slice is even a candidate for a clean
  concept. Bills et al. explained all **307,200** neurons of GPT-2 small with GPT-4 and scored the
  explanations by simulating the neuron from the explanation; only **≈1,000 (0.33 %)** reached
  score > 0.8.
- **Why it matters.** These are the two best available base rates for "what fraction of neurons can
  be named". They set the ceiling on the deliverable. Any plan promising a concept per neuron is
  promising a 30–300× overestimate.
- **Confidence:** high. Two independent groups, two independent methods, same order of magnitude.
- **Cost:** free — this is a planning constraint, not a compute item.
- **Design consequence:** the atlas must be a **census with graded confidence**, not a dictionary.
  Every row carries `label`, `label_method`, `label_confidence`, `n_fire`. A row with
  `label_confidence = 0.05` is *still useful* (it tells you the neuron is active and roughly when)
  and is honest, whereas a confident wrong label poisons everything downstream.

### F2. Prior art: four real atlases, four different failure modes
[source: OpenAI Microscope, 2020, https://microscope.openai.com/]
[source: Exploring Neural Networks with Activation Atlases, Distill, 2019, https://distill.pub/2019/activation-atlas/]
[source: Zoom In: An Introduction to Circuits, Distill, 2020, https://distill.pub/2020/circuits/zoom-in]
[source: Neuronpedia, 2024–2026, https://www.neuronpedia.org/]
[source: Neuron-level Interpretation of Deep NLP Models: A Survey, TACL 2022, https://arxiv.org/abs/2108.13138]

| Atlas | Unit | What it taught us | Where it stops |
|---|---|---|---|
| **Activation Atlas** (Carter et al. 2019) | 2-D projection of activation space | Global "shape" of feature space; you can see concept manifolds | Projects away individual neurons — no per-unit claims, no causality |
| **OpenAI Microscope** (2020) | individual neurons + circuits, vision models | That per-neuron browsing *works* and is genuinely illuminating; spawned the whole "neuron viewer" genre | Vision only; qualitative; no causal validation; neurons hand-picked because they looked interesting (selection bias) |
| **Neuronpedia** (2024–) | **SAE features**, plus head dashboards | Feature dashboards (top activations, logits, explanations, steering) are the right *UX*; we should copy it | Features ≠ neurons. Feature sets are non-canonical and model-specific. Head dashboards are metrics, not causal profiles |
| **Universal Neurons** (Gurnee 2024) | neurons, GPT-2 small | The only published *exhaustive-ish* per-neuron catalogue for an LLM; produced neuron "families" (entropy neurons, attention-deactivating neurons, token-set predictors) | 1–5 % coverage; GPT-2 small only; needs 100 M tokens; correlation-based, not causal |

- **Why it matters.** Nobody has published a full per-neuron + per-head atlas for a *code* model,
  and nobody has published one that is **causally verified at the population level**. That is a real
  gap and it is the honest framing of "novel contribution" for this project.
- **Confidence:** high on the inventory; the "nobody has done it" claim is **medium** (absence of
  evidence in a fast-moving field).
- **Cost:** zero.
- **Reusable asset:** Neuronpedia's feature-dashboard information architecture is the right model
  for A5 (UI). Steal the layout, not the data.

### F3. There is no paper called "A Walk Through the Layers" — do not cite it
[source: (negative result) searches for the title return nothing matching]

- **What it is.** The brief asked me to check Anthropic's "A Walk Through the Layers" style work. I
  could not verify any such paper. **UNVERIFIED — treat as non-existent** until someone produces a
  link. The real Anthropic lineage that plays this role is:
  *A Mathematical Framework for Transformer Circuits* (2021),
  *Towards Monosemanticity* (2023, https://transformer-circuits.pub/2023/monosemantic-features),
  *Scaling Monosemanticity* (2024, https://transformer-circuits.pub/2024/scaling-monosemanticity/),
  and *On the Biology of a Large Language Model* (2025,
  https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
- **Why it matters.** `CONTEXT.md` rule: no unsourced claims. If this title leaks into the
  deliverable it is an instant credibility loss.
- **Confidence:** high that it does not exist under that title.
- **Cost:** zero.

### F4. The profiling methodology: a "response profile" is 6 statistics + 3 evidence channels, computed in one streaming pass
[source: Transformer Feed-Forward Layers Are Key-Value Memories, 2021, https://arxiv.org/abs/2012.14913]
[source: Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space, 2022, https://arxiv.org/abs/2203.14680]

- **What it is.** For neuron `j` in layer `L`, define the response profile over corpus `D`:
  1. **Moments** — `n`, `n_fire`, mean, std, skew, kurtosis (streaming, Welford-style).
  2. **Tail shape** — p50/p90/p99/p99.9/max. This is more trustworthy than kurtosis (see below).
  3. **Top-k contexts** — the k=32 highest-activating token positions, with `(doc_id, byte_offset,
     token_id, position, ast_class)`.
  4. **Vocabulary promotion** — `argtop_k( W_U · W_down[:, j] )`: the tokens this neuron's write
     vector pushes *up*, and the bottom-k it pushes *down*. This is the Geva "key-value memory"
     view, and it is exact, cheap, and hallucination-free.
  5. **Context-class distribution** — normalised histogram over ~100 AST/syntactic classes.
  6. **Causal effect** — Δlogit / Δloss under mean-ablation, plus gradient attribution rank.
- **Critical technical point specific to Qwen2 (SwiGLU).** The "neuron" is
  `a_j = silu(W_gate x)_j · (W_up x)_j`, which is **signed and heavy-tailed**, not ReLU-like. So:
  - "Sparsity" must be defined as `P(|a_j| < τ)` with an explicit `τ`, **not** `a_j == 0`.
    Qwen2 has essentially no exact zeros.
  - "Firing" is not sign-consistent: a neuron can be strongly *inhibitory* on some tokens. Report
    `mean_signed` and `mean_abs` separately, and `frac_negative`.
  - Literature sparsity figures (≈90 % for ReLU-ified Llama, Liu et al. 2023,
    https://arxiv.org/abs/2312.11514) **do not transfer**. Measure it; expect 60–90 % at τ = 1 % of
    the per-neuron max.
- **Why it matters.** Getting this wrong produces an atlas that silently mis-describes half the
  model. This is the most likely place to fool ourselves, and it is cheap to get right.
- **Confidence:** high on the algebra; medium on the sparsity range (must be measured).
- **Cost:** zero extra — it's the same forward pass.

### F5. The multiple-comparisons problem is real but it is *not* the main threat; document-level confounding is
[source: Controlling the False Discovery Rate, Benjamini & Hochberg, 1995, JRSS-B]

- **What it is.** 250,880 neurons × ~100 classes ≈ 2.5 × 10⁷ association tests. Naive BH-FDR at
  q = 0.05 is the textbook fix and is trivially applicable. **But** the dominant error mode is
  different: all tokens from one Python file share a topic, style, and indentation, so a neuron that
  fires on "this repo's docstring style" will appear to fire on "function definitions" purely because
  of document clustering. A p-value that assumes token independence is meaningless here.
- **The fix (three parts, all cheap):**
  1. **Block permutation null.** Shuffle context labels *within document*, not globally. Recompute
     the neuron × class matrix 200× to get an empirical null for each cell. Cost: 200 × (a 100 ×
     250,880 reduction) ≈ minutes.
  2. **Effect size, not p-value, for ranking.** Use `lift = P(active | class) / P(active)`, require
     `lift ≥ 3` **and** `n_support ≥ 100` **and** permutation p < 0.001. Report the p-value but
     rank by lift.
  3. **Double-dipping guard — discovery/verification split.** Labels derived from corpus A must be
     scored on a *held-out* corpus B (different repos, same distribution). This is the single
     cheapest defence against the "we found a pattern in the noise we selected" failure, and it gives
     A6 a hard number to test (label precision@k on B).
- **Why it matters.** Without (1) and (3), the atlas will be full of beautiful, confident,
  wrong labels — and this is exactly the failure mode the project is trying to avoid.
- **Confidence:** high.
- **Cost:** +5 % wall clock (the permutation pass), plus a 15 % corpus hold-out.

### F6. Automatic neuron labelling: the method works, the reliability is poor, and the failure modes are known
[source: Language models can explain neurons in language models, 2023, https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html]
[source: SAEBench, 2025, https://arxiv.org/abs/2503.09532]
[source: Open Problems in Mechanistic Interpretability, 2025, https://arxiv.org/abs/2501.16496]

- **What it is.** Bills et al.'s pipeline: (1) show a strong LLM the top-activating contexts, (2) have
  it write an explanation, (3) have a *simulator* LLM predict the neuron's activation on fresh text
  from the explanation alone, (4) score = correlation between predicted and actual. Known failure
  modes, all documented in that paper or in the follow-up literature:
  - **Base-rate neglect.** The labeler only sees positives, so a neuron firing on `the` 90 % of the
    time gets labelled "the" — technically true, useless.
  - **Overly broad labels.** Bills et al. explicitly note this (their "don't stop" neuron).
  - **Confirmation bias in the labeler.** Shown 10 similar contexts, an LLM will *always* find a
    pattern; there is no negative evidence in the prompt.
  - **Corpus-artefact labels.** "Python code", "PEP-8 formatting" — the label describes our corpus,
    not the neuron.
  - **Score ≠ faithfulness.** SAEBench (2025) shows auto-interp scores correlate imperfectly with
    human judgement and with downstream usefulness; Leask et al. (ICLR 2025,
    https://proceedings.iclr.cc/paper_files/paper/2025/hash/84ca3f2d9d9bfca13f69b48ea63eb4a5-Abstract-Conference.html)
    go further and argue SAE/feature units are not canonical at all.
- **Our constraint.** The only LLM we can run locally is a 1.5B. Using a 1.5B to explain a 1.5B is
  not the Bills setup (they used GPT-4 to explain GPT-2) and will produce mostly noise. **Do not
  build the atlas on an LLM labeler.**
- **The constructive substitute — two objective channels that need no LLM:**
  - **AST-grounded labels.** Parse with tree-sitter-python; for each token position record the
    innermost AST node type and the token's *role* in it (function name / arg / operator / decorator
    / …). ~100 classes. A neuron whose activation distribution is concentrated on `for_stmt.target`
    is labelled `for-loop target`, with a lift and a permutation p-value. This is objective,
    reproducible, and directly answers the user's actual question.
  - **Vocabulary promotion.** `W_U · W_down[:, j]` gives the top-10 promoted and top-10 suppressed
    tokens for every neuron in ~35–50 min of matmul (250,880 × 1536 × 151,936 × 2 ≈ 1.2 × 10¹⁴ FLOP;
    it is one big dense GEMM, so it runs near peak).
- **Contrastive labelling** (if an LLM labeler is used at all): always show top activations **and**
  matched random activations, and require the label to *discriminate*, not just describe.
- **Why it matters.** This is the difference between an atlas that is 5 % labelled and trustworthy
  and one that is 100 % labelled and worthless.
- **Confidence:** high.
- **Cost:** AST channel ≈ 2 h of CPU (parsing 500 k tokens) + minutes of reduction; vocab channel
  ≈ 35–50 min.

### F7. Causal profiling at scale: gradient attribution screens all 250,880 neurons in one backward pass
[source: AtP*: An efficient and scalable method for localizing LLM behaviour to components, 2024, https://arxiv.org/abs/2403.00745]
[source: Knowledge Neurons in Pretrained Transformers, 2022, https://arxiv.org/abs/2104.08696]
[source: Massive Activations in Large Language Models, 2024, https://arxiv.org/abs/2402.17762]

- **What it is.** Choose a behaviour objective `L = logit(target) − logit(anti-target)` (or
  `−log p(target)`). Then `∂L/∂a_j` is available for **all 250,880 neurons simultaneously** from a
  single backward pass. Score `attr_j = a_j · ∂L/∂a_j` (gradient × activation) — the AtP family.
  AtP* (Kramar et al. 2024) shows this ranks components well but is a *first-order* approximation
  that can misestimate magnitude, so it must be used for **ranking only** and verified by ablation.
- **Cost asymmetry that makes the whole design work:**
  - Screening: forward+backward ≈ 2.5× forward → 50 k probe tokens ≈ **4.5 h** → attribution for
    **all** 250,880 neurons.
  - Verification by real ablation: ~3.3 s/neuron (batched) → 2,000 neurons ≈ **3 h**, 5,000 ≈ **7 h**.
  - Exhaustive ablation of all 250,880 ≈ **343 h = 14.3 days**.
- **The batching trick (essential).** Different neurons need different interventions, which normally
  means one forward pass each. Instead put **B different neurons in one batch**: same prompt repeated
  B times, batch element `b` has neuron `j_b` mean-ablated. One pass → B causal measurements. Throughput
  improves ~8–15× over naive per-neuron looping. Measured planning figure: B = 32, S = 64 → 158 s per
  pass → 7,840 passes for full coverage.
- **Mean-ablation, not zero-ablation.** Zeroing a neuron removes both its signal *and* its baseline
  offset, which is a much larger perturbation than the neuron's actual contribution. Replace `a_j`
  with its corpus mean (or the conditional mean for that token class). Zero-ablation systematically
  overstates importance; this is the standard error in casual "neuron ablation" studies.
- **Published causal-neuron studies exist but are narrow.** Dai et al. 2022 (knowledge neurons —
  factual recall, single-relation prompts); Geva et al. 2021/2022 (FFN as key-value memory);
  Meng et al. 2022 (ROME causal tracing). **None** does a population-level census. That gap is real.
- **Why it matters.** It converts "250 k ablations = 2 weeks" into "1 day", and it is the only way
  this project gets causal claims for every neuron rather than a hand-picked handful.
- **Confidence:** high on the mechanism; medium on AtP fidelity (hence the verification step).
- **Cost:** 4.5 h screen + 3–7 h verify.

### F8. Heads: exhaustive causal profiling is cheap, so do it exhaustively — with one GQA caveat
[source: In-context Learning and Induction Heads, 2022, https://arxiv.org/abs/2209.11895]
[source: Efficient Streaming Language Models with Attention Sinks, 2023, https://arxiv.org/abs/2309.17453]
[source: Are Sixteen Heads Really Better than One?, 2019, https://arxiv.org/abs/1905.10650]
[source: Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, 2019, https://arxiv.org/abs/1908.08498]

- **What it is.** For each of the 336 query heads, measure:
  - **Attention pattern type** — mass on position 0 (attention sink), mass on previous token, mass on
    the current token, entropy, and the position of the argmax (binned).
  - **Induction score** — Olsson et al.'s prefix-matching score: on `[random tokens] ++ [same random
    tokens]`, measure attention from the second occurrence to the token *following* the first
    occurrence. This is a validated, cheap, quantitative induction detector.
  - **Copy score** — attention to the same token elsewhere in context.
  - **Causal effect** — mean-ablate the head's output slice and measure Δloss on a probe suite, plus
    Δ on specific behaviours (next-token, syntax, long-range copy).
- **GQA caveat nobody mentions.** Qwen2.5-1.5B has 12 query heads and **2 KV heads** (6:1). A query
  head's output is a well-defined 128-dim slice of `o_proj`'s input, so **per-query-head ablation is
  clean**. But the key/value subspace is *shared by 6 heads*, so "ablate a KV head" is not a
  per-head operation — it ablates 6 query heads at once. Consequence for A8 (compression): query
  heads are prunable individually, KV heads only in groups of 6, and any head-importance ranking that
  ignores this will mis-allocate bits.
- **Cost:** 336 ablations, batched at B = 32, S = 64 → 11 passes × 158 s ≈ **29 min**. Even the naive
  loop is only ~3 h. There is no reason to sample here.
- **Prior art limit:** Michel et al. 2019 and Voita et al. 2019 both showed most heads can be pruned
  with little loss — but they measured *task* loss, not *which behaviour each head implements*. Our
  atlas should do the latter.
- **Confidence:** high.
- **Cost:** 0.5 h.

### F9. SAE vs neurons: do the neuron census first, then one SAE as a re-basis — not the other way round
[source: Gemma Scope, 2024, https://arxiv.org/abs/2408.05147]
[source: Jumping Ahead: JumpReLU SAEs, 2024, https://arxiv.org/abs/2407.14435]
[source: Scaling and evaluating sparse autoencoders, 2024, https://arxiv.org/abs/2406.04093]
[source: Are Sparse Autoencoders Useful? A Case Study in Sparse Probing, ICML 2025, https://arxiv.org/abs/2502.16681]

- **The tradeoff, stated plainly:**

| | Neuron atlas | SAE feature atlas |
|---|---|---|
| Coverage | **100 %** — every unit gets a row | Partial — dead latents, absorption, arbitrary width |
| Cost | 1 forward pass per token | + 10–20 h training per layer, + activation cache |
| Unit validity | Basis-dependent (whatever SGD made) | Also non-canonical (Leask et al. 2025) |
| Monosemanticity | Low (superposition) | Higher, but not guaranteed |
| Interpretability ceiling | 1–5 % clean labels | Higher per-feature, but features may not be causal |
| Evidence of usefulness | Correlational unless ablated | Kantamneni et al. 2025: SAEs often **don't** beat simple baselines on downstream probing |

- **The honest read on SAEs.** They are a *re-basis*, not a ground truth. Two 2025 papers
  (SAEBench; "Are SAEs Useful?") independently find that SAE features are weaker than advertised as
  practical tools. So an SAE is worth having — it makes *some* directions legible — but it cannot be
  the backbone of a transparency claim.
- **Concrete config for one layer of Qwen2.5-1.5B (d_model = 1536):**
  - **JumpReLU** (Rajamanoharan et al. 2024), width **24,576 (16×)**, target **L0 ≈ 50–100**.
    Gemma Scope used widths 16,384 and 65,536 on Gemma-2-2B (d_model 2304, i.e. ≈7× and ≈28×), so
    16× is squarely in-distribution for a model this size.
  - Alternative: TopK SAE (Gao et al. 2024) with k = 64 — simpler, no L1 tuning, and it removes the
    L1↔L0 shrinkage bias. **Recommend TopK for a first attempt**, JumpReLU once the pipeline works.
  - **Dead-latent resampling** is mandatory (Gao et al. 2024). With our small token budget expect
    **40–70 % dead latents**; without resampling the SAE will look much worse than it is.
- **CPU training cost (measured-derived):** fwd+bwd ≈ 3 × 2 × d × W FLOPs = 227 MFLOP/token-step
  for W = 24,576 → 5.7 ms at 40 GFLOPS → **177 steps/s**.
  - 1 M activations × 10 epochs = 10 M steps → **15.7 h**.
  - W = 16,384, 20 M steps → **21 h**.
- **Verdict on Q4 of the LEDGER ("can an SAE be trained on 1 layer in <24 h?"):** **Yes — one layer,
  ~16 h, at 16× expansion.** But the quality will be demo-grade, not research-grade: Gemma Scope
  trained on orders of magnitude more tokens. Expect fuzzy features and many dead latents. Do not
  promise "Gemma Scope for Qwen".
- **Shortcut worth checking first:** a community SAE exists for **Qwen2.5-1.5B-Instruct**
  (https://huggingface.co/HuggingAnalist/sae-qwen2.5-1.5B-res): L1 SAE, 16× = 24,576 features,
  **100 M training tokens**, layers 17–21, residual stream, `sae_lens` compatible. It is the wrong
  model variant (Instruct, not Base) and its quality is **unverified** (0 likes, no downloads
  tracked), but it proves the community-standard budget is 100 M tokens at 16×, and it is a free
  baseline. **Qwen-Scope** (2026, https://arxiv.org/abs/2605.11887) is the official suite but covers
  **Qwen3 / Qwen3.5 only — not Qwen2.5**, so it does not help us.
- **Confidence:** high on cost; medium on the quality ceiling.
- **Cost:** 16 h for one layer, or 0 h to try the community SAE.

### F10. Data engineering: stratified corpus beats raw token count, and the statistics say 500 k tokens is the honest floor
[source: The Stack v2 / StarCoder2, 2024, https://arxiv.org/abs/2402.19173]
[source: The Stack: 3 TB of permissively licensed source code, 2022, https://arxiv.org/abs/2211.15533]

- **Corpus design (two tiers):**
  - **Tier A — generic Python, ~300 k tokens.** `bigcode/the-stack-v2` Python subset, or
    `the-stack-dedup`, filtered to files of 50–500 lines with a docstring (so there is both code and
    natural language). Deduplicate by MinHash. Sample **by repository**, not by file, and hold out
    entire repositories for corpus B (the verification split from F5).
  - **Tier B — targeted construct probes, ~30 k tokens.** Synthetic snippets, 200 each, for every
    Python construct we want to be able to query: `for`, `while`, `if/elif/else`, `try/except/finally`,
    `with`, `lambda`, list/dict/set comprehension, generator expression, decorator, class, dataclass,
    `async def`/`await`, `yield`, `match`, f-string, slicing, chained attribute, chained comparison,
    walrus, `*args`/`**kwargs`, type hints, recursion, closures. Vary identifiers and literals so the
    neuron can't latch onto a name.
  - **Why Tier B matters more than its token count suggests.** A neuron that only fires on `yield`
    fires on ~0.05 % of generic code — statistically invisible at 300 k tokens. But it fires on
    **100 %** of our 200 `yield` probes. **Stratified sampling buys us coverage of rare constructs
    for free**, which is exactly the user's actual query ("show me the neurons for a `for` loop").
- **The statistical argument for how many tokens.** Let `p_j` be the fire rate and
  `σ_fire/μ_fire ≈ 2.5` (heavy-tailed code activations). Relative error of the corpus mean is
  `ε = (σ/μ)/√(n · p_j)`. For ε = 5 %: `n · p_j ≥ 2,500`.

| fire rate `p_j` | tokens needed for ±5 % mean | feasible? |
|---|---|---|
| 10 % | 25 k | yes |
| 1 % | 250 k | yes |
| 0.1 % | 2.5 M | no (69 h) |
| 0.01 % | 25 M | no |

  → **500 k tokens covers neurons firing ≥ 1 %; 0.1 % is out of reach.** Therefore every row must
  carry `n_fire`, and the atlas must expose a `stat_confidence` derived from it. Never present a
  top-32 context list built from 8 observations as if it were a profile.
  - **On kurtosis:** under normality `SE(kurt) ≈ √(24/n)`, but code activations are heavy-tailed
    enough that the sample 4th moment converges very slowly and is not robust. **Report it, rank by
    p99/p99.9/max ratios instead.** This is a real trap — kurtosis looks scientific and is the least
    stable statistic in the table.
- **Why 100 M tokens (Gurnee) is not a counter-argument.** They needed 100 M because they were
  estimating *pairwise correlations between all neuron pairs across 5 seeds* — a vastly harder
  estimand. Marginal per-neuron moments converge far faster.
- **Confidence:** high on the scaling argument (it's a standard effective-sample-size derivation);
  medium on the 2.5 dispersion constant (measure it in the calibration pass).
- **Cost:** ~1 h to build the corpus (download + filter + parse).

### F11. **The hardware is 5× worse than CONTEXT.md assumes.** This is the finding that governs everything
[source: measured on this machine, 2026-11-10 — see commands in "Concrete implementation"]

- **What it is.** `lscpu` reports: **Intel Core i7-8665U, 4 physical cores / 8 threads, CPU max
  1900 MHz, AVX2 only (no AVX-512, no AVX512-BF16, no VNNI), 8 MiB L3.** This is a 2019 15 W
  *mobile* U-series part, not a workstation or server CPU. `CONTEXT.md` says "8 cores, 31 GB RAM" —
  it is 4 cores + SMT, and only **~15 GB is actually free** (15 GB already in use).

  Measured with numpy 2.5.2 / OpenBLAS, 8 threads:

  | measurement | result |
  |---|---|
  | GEMM 1536³ fp32 | **67.4 GFLOPS** |
  | GEMM 512 × 8960 × 1536 (MLP-shaped) | **54.5 GFLOPS** |
  | `memcpy` 1 GiB | **7.4 GB/s** (read+write) |
  | One MLP layer, batch-1 GEMV (2 matrices) | 11.3 ms → **9.7 GB/s effective** |
  | ⇒ full 28-layer forward, batch 1, fp32 | **≈ 0.54 s/token ≈ 1.9 tok/s** |
  | ⇒ batch 8–16 (amortise weight reads) | **≈ 8–12 tok/s**; hard compute cap = 40/3.08 = **13 tok/s** |

- **Why it matters.** `CONTEXT.md` says "expect 5–20 tok/s". At **batch 1** the truth is **~2 tok/s**,
  because the model is memory-bandwidth-bound (6.2 GB of fp32 weights streamed per token at 9.7 GB/s).
  Batching is not an optimisation, it is *the* precondition for this project existing. Any agent
  planning around 20 tok/s is planning a 4× overrun.
- **Second consequence:** bf16 will **not** help. There is no native bf16 arithmetic on this chip, so
  bf16 weights either get upcast (no gain) or hit an emulated path (slower). **Plan fp32.**
- **Third consequence:** 15 W TDP means sustained load throttles. Derate the 8–12 tok/s figure by
  ~20 % for runs longer than a few minutes → **plan with 8 tok/s**.
- **Confidence:** high — measured, not estimated.
- **Cost:** zero (already paid).
- **Action for the orchestrator:** correct `CONTEXT.md`'s hardware line. It propagates into A7's and
  A8's budgets too, and A7 in particular may be over-optimistic about 8B feasibility.

### F12. Concrete budget: the 2-day version, and the 2-week version
- **2-day version (recommended, 35 h of compute, 13 h slack):**

  | step | wall clock |
  |---|---|
  | corpus build (Tier A + Tier B) + AST annotation | 1.5 h |
  | calibration pass (25 k tokens → per-neuron τ) | 0.9 h |
  | **harvest 500 k tokens** (batch 16, seq 512) | **17.4 h** |
  | vocabulary-promotion matmul (all neurons) | 0.7 h |
  | gradient attribution, 50 k probe tokens | 4.5 h |
  | head profiling, all 336, batched | 0.5 h |
  | **ablation verification, top 2,000 neurons** | **3.0 h** |
  | co-activation clustering + parquet write | 0.3 h |
  | SAE, 1 layer, 16×, 1 M acts × 10 epochs (optional) | 15.7 h |

  Without the SAE: **28 h**. With it: **44 h** — still inside 48 h, but only just. **Recommendation:
  run the atlas first, launch the SAE overnight only if the atlas finishes early.**

- **2-week version (the "exhaustive" claim, honestly):**
  harvest 2 M tokens (69 h) + exhaustive ablation of all 250,880 neurons (343 h) + SAEs on 4 layers
  (63 h) = **≈ 475 h ≈ 20 days of continuous compute**, and the machine is a laptop that will
  thermal-throttle. **Exhaustive causal coverage of all neurons is not a 2-day deliverable and
  should not be promised as one.**

### F13. Verdict: what an atlas actually explains, and where it stops

| layer of claim | coverage | confidence |
|---|---|---|
| Every neuron has a row with moments + top contexts | **100 %** | high |
| …that is statistically meaningful (±5 % mean) | ~60–75 % (fire rate ≥ 1 %) | high |
| …with an objective AST / vocab-promotion label | ~50–70 % | medium-high |
| …with a *clean single-concept* label | **1–5 %** (2.5 k–12.5 k neurons) | medium — this is the prior-work base rate |
| …with a *causally verified* effect (ablated) | 2,000–5,000 neurons (**1–2 %**) | high |
| …with a full causal ranking (gradient-screened) | **100 %** | medium (first-order approximation) |
| Every head causally profiled | **100 %** | high |
| Behaviour-level explanation ("ablating these 12 neurons breaks `for`-loop emission") | achievable for ~20–40 named behaviours | high |

- **What the atlas genuinely explains:** *which* units are involved in *which* behaviour, with a
  measured effect size, for any behaviour we can write a probe for. That is a **causal lookup table**,
  and it is what the microscope actually needs.
- **Where it stops, concretely:**
  1. **Composition.** A neuron's effect measured in isolation ≠ its role in a circuit. Two neurons
     that each have 2 % effect can jointly have 60 %. The atlas is a *marginal* map, not a circuit
     diagram. Circuit discovery is a different (and much more expensive) job.
  2. **Corpus-relativity.** Every label means "on our 500 k tokens of Python". A neuron labelled
     `for-loop target` may do something else entirely on prose. There is no corpus-independent label.
  3. **The long tail.** 25 % of neurons fire too rarely for us to say anything, and the most
     *interesting* neurons are often exactly those. Honest answer: unknown, not "unimportant".
  4. **Superposition is not dissolved.** Profiling neurons does not undo superposition; it just
     documents its symptoms. The SAE re-basis helps locally and does not fix it globally.
  5. **Behaviour ≠ mechanism.** Knowing that ablating neurons 14.{3021, 4402} breaks `for`-loop
     emission tells you *where*, not *how*. The "how" requires the circuit layer.
- **Honest headline for the user:** *"We can tell you, for any Python construct, which ~20 neurons
  matter and prove it by ablation. We cannot tell you what most of the 250,880 neurons mean, and
  neither can anyone else."*

---

## Concrete implementation (code sketches, real APIs)

### 0. Calibrate the hardware before planning anything
```bash
lscpu | grep -E "Model name|CPU\(s\)|Core|Thread|CPU max"
grep -o -E 'avx[0-9a-z_]*' /proc/cpuinfo | sort -u     # expect: avx avx2  (NO avx512)
free -g                                                 # expect ~15 GB free, not 31
```
```python
# measure, don't assume — ~10 s
import numpy as np, time
n=1536; a=np.random.rand(n,n).astype(np.float32); b=a.copy(); a@b
t=time.time(); [a@b for _ in range(10)]; dt=(time.time()-t)/10
print("GEMM GFLOPS:", 2*n**3/dt/1e9)
# full-model tok/s estimate (batch 1, memory bound):
#   1.54e9 params * 4 B / (9.7 GB/s) ~= 0.54 s/token
```

### 1. The exact hook point — `down_proj`'s *input* is the neuron vector
```python
# Qwen2MLP.forward:  self.down_proj( self.act_fn(self.gate_proj(x)) * self.up_proj(x) )
# => the tensor entering down_proj IS a[:, 0:8960] == the 8960 neurons. No recomputation needed.
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Base", torch_dtype=torch.float32,   # fp32: no bf16 on this CPU
    attn_implementation="eager",                                  # need real attention weights
).eval()
torch.set_num_threads(8)

captured = {}
def make_hook(L):
    def hook(mod, args):                     # forward_pre_hook -> args[0] is the neuron vector
        captured[L] = args[0].detach()       # (B, T, 8960)
    return hook
for L in range(28):
    model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L))
```

### 2. Single-pass streaming profiler (the core of the atlas)
Two-phase: a short **calibration** pass sets a per-neuron threshold, then the **main** pass only
captures what clears it. This keeps memory at ~600 MB instead of 1 TB.

```python
import numpy as np

class Atlas:
    def __init__(self, n_layers=28, d_ff=8960, k=32, nbins=48):
        N = n_layers * d_ff
        self.k, self.nbins, self.N = k, nbins, N
        self.n      = np.zeros(N, np.int64)
        self.n_fire = np.zeros(N, np.int64)
        self.s1 = np.zeros(N); self.s2 = np.zeros(N)
        self.s3 = np.zeros(N); self.s4 = np.zeros(N)      # float64 accumulators
        self.n_neg = np.zeros(N, np.int64)
        self.max  = np.zeros(N, np.float32)
        self.hist = np.zeros((N, nbins), np.uint32)       # log-spaced |a| bins
        self.thr  = None
        # top-k: values + (doc_id, byte_off, token_id, pos) packed into int64
        self.tk_v = np.full((N, k), -np.inf, np.float32)
        self.tk_c = np.zeros((N, k), np.int64)

    def calibrate(self, a):            # a: (T, N) float32, 25k tokens
        self.n += a.shape[0]
        self.s1 += a.sum(0, dtype=np.float64)
        self.s2 += (a.astype(np.float64) ** 2).sum(0)
        self.max = np.maximum(self.max, np.abs(a).max(0))
        self.thr = np.maximum(np.quantile(np.abs(a), 0.99, axis=0), 1e-6) * 0.25

    def update(self, a, ctx):          # ctx: (T,) int64 packed context ids
        self.n      += a.shape[0]
        self.n_fire += (np.abs(a) > self.thr).sum(0)
        self.n_neg  += (a < 0).sum(0)
        af = a.astype(np.float64)
        self.s1 += af.sum(0); self.s2 += (af**2).sum(0)
        self.s3 += (af**3).sum(0); self.s4 += (af**4).sum(0)
        self.max = np.maximum(self.max, np.abs(a).max(0))
        # vectorised top-k merge: only rows that can change are touched
        cand = np.abs(a) > self.tk_v.min(1)[None, :]
        if cand.any():
            rows, cols = np.nonzero(cand)
            for r in np.unique(cols):
                sel = rows[cols == r]
                vals = np.concatenate([self.tk_v[r], np.abs(a)[sel, r]])
                cids = np.concatenate([self.tk_c[r], ctx[sel]])
                o = np.argpartition(vals, -self.k)[-self.k:]
                self.tk_v[r], self.tk_c[r] = vals[o], cids[o]

    def finalize(self):
        n = np.maximum(self.n, 1)
        mu  = self.s1 / n
        var = np.maximum(self.s2 / n - mu**2, 0)
        sd  = np.sqrt(var)
        return dict(mean=mu, std=sd, skew=(self.s3/n - 3*mu*sd**2 - mu**3)/sd**3,
                    kurt=(self.s4/n - 4*mu*self.s3/n + 6*mu**2*self.s2/n - 3*mu**4)/var**2,
                    fire_rate=self.n_fire/n, frac_neg=self.n_neg/n,
                    absmax=self.max, n=self.n, n_fire=self.n_fire)
```
> `cand` + per-column `argpartition` is the hot loop. If it is too slow, the fallback is:
> pass 1 = moments + histogram only (fast), pass 2 = threshold from the histogram, capture sparse
> `(neuron, ctx)` pairs into a COO list, top-k at the end. That costs 2× tokens but 1/5 the CPU per
> token, and is the safer engineering choice.

### 3. Corpus + AST annotation (this is the labelling engine, not an LLM)
```python
import ast, io, tokenize
# tree-sitter-python gives exact byte spans per token; the stdlib `ast` gives the node types.
# For each (line, col) build: innermost_node_type, role, depth, enclosing_function_len
CLASSES = ["for_stmt.target","for_stmt.iter","for_stmt.body","while_stmt.test","if_stmt.test",
           "call.func","call.arg","call.kwarg","attribute.value","attribute.attr","subscript.value",
           "subscript.slice","binop.left","binop.right","binop.op","assign.target","assign.value",
           "return.value","yield.value","lambda.params","comprehension.elt","comprehension.cond",
           "decorator","class_def.name","func_def.name","func_def.params","annassign.annotation",
           "fstring.expr","with_stmt.item","except_handler.type","match_case.pattern",
           "import.module","import.name","docstring","comment","newline","indent","number","string"]
```
Output: `ctx_class: int8[T]` and `ctx_pack: int64[T]` = `(doc_id << 32) | byte_offset`.

### 4. Gradient attribution — one backward pass for all 250,880 neurons
```python
def attribution(model, ids, target, anti):
    a_store = {}
    hooks = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
                 lambda m, args, L=L: a_store.__setitem__(L, args[0])) for L in range(28)]
    logits = model(ids).logits[:, -1, :]
    obj = logits[0, target] - logits[0, anti]          # behaviour objective
    obj.backward()
    # grads land on a_store[L] because it is the down_proj input; capture via retain_grad
    ...
    # attr[L, j] = a[L, j] * dL/da[L, j]   -> rank all 250,880 at once
```
> Implementation detail: use `args[0].requires_grad_()` inside the hook plus `.retain_grad()`, or
> register the hook on `down_proj` as a *forward* hook that stores `output` and take
> `grad` from the module output. The second is cleaner in HF.

### 5. Batched ablation — B neurons per forward pass (the 15× speedup)
```python
def ablate_batch(model, prompt_ids, neuron_ids, means):
    """neuron_ids: list of B (layer, j). Same prompt replicated B times."""
    B = len(neuron_ids)
    ids = prompt_ids.repeat(B, 1)
    hooks = []
    for b, (L, j) in enumerate(neuron_ids):
        def h(mod, args, b=b, j=j, L=L):
            x = args[0].clone()
            x[b, :, j] = means[L, j]          # MEAN-ablate, not zero-ablate
            return (x,) + args[1:]
        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(h))
    out = model(ids).logits
    for hk in hooks: hk.remove()
    return out                                   # (B, T, V) -> per-neuron delta
```

### 6. Parquet schema (the atlas artifact)

`atlas/neurons.parquet` — 250,880 rows:

| column | type | notes |
|---|---|---|
| `layer` | uint8 | 0–27 |
| `index` | uint16 | 0–8959 |
| `n_tokens`, `n_fire` | uint32 | sample support |
| `fire_rate`, `frac_neg` | float32 | |
| `mean_signed`, `mean_abs`, `std` | float32 | |
| `p50,p90,p99,p999,absmax` | float32 | **rank by these, not kurtosis** |
| `skew`, `kurtosis` | float32 | present, low trust |
| `top_ctx` | list<struct<ctx_id:int64, tok_id:uint32, pos:uint16, act:float16, ast_class:uint8>> | k=32 |
| `promote_top10` | list<struct<tok_id:uint32, dlogit:float16>> | from `W_U·W_down[:,j]` |
| `suppress_top10` | list<struct<tok_id:uint32, dlogit:float16>> | |
| `ast_dist` | list<struct<cls:uint8, prob:float16>> | top-8 |
| `label` | string | ≤ 80 chars |
| `label_method` | uint8 | 0=none 1=voclift 2=ast 3=both 4=llm 5=human |
| `label_confidence` | float16 | 0–1, **derived from n_fire + lift + holdout precision** |
| `lift`, `perm_p` | float32 | |
| `attr_rank`, `attr_mean` | uint32, float32 | gradient screening |
| `ablate_dlogit`, `ablate_verified` | float32, bool | only for the verified subset |
| `cluster_id`, `top_partners` | uint16, list<uint16>×8 | co-activation |
| `flags` | uint8 | bit0 dead, bit1 massive-activation, bit2 sink, bit3 under-sampled |

`atlas/heads.parquet` — 336 rows: `layer, head, kv_group, sink_mass, prev_tok_mass,
self_mass, entropy_mean, argmax_pos_mode, induction_score, copy_score, ablate_dloss,
ablate_dlogit, pattern_type, flags`.

`atlas/class_matrix.npy` — `float16[100, 250880]`: mean activation per AST class. **This is the
query index.** 50 MB.

**Sizes (measured arithmetic, not guesswork):** neurons.parquet ≈ 151 MB raw → 50–70 MB zstd;
heads.parquet < 100 KB; class_matrix 50 MB; context dictionary (200 k strings) ≈ 40 MB.
**Total < 250 MB.** Disk is 293 GB — irrelevant.

### 7. Query design: "show me the neurons active when the model writes a `for` loop"

Three query shapes, all cheap, all over precomputed artifacts:

```python
# (a) Direct class query — O(|C| x N) dot, milliseconds
M = np.load("atlas/class_matrix.npy")          # (100, 250880) fp16
v = M[CLASSES.index("for_stmt.target")]
hits = np.argsort(-v)[:200]                    # then filter by n_fire / lift from the parquet

# (b) CONTRASTIVE query — the one that actually answers the question.
#     "for" specifically, not "code" in general:
v = M[CLASSES.index("for_stmt.target")] - M[CLASSES.index("while_stmt.target")] \
  - 0.5 * M[CLASSES.index("docstring")]
```
```sql
-- (c) SQL over the parquet, via duckdb — no server needed
SELECT layer, index, label, label_confidence, fire_rate, lift, ablate_dlogit
FROM 'atlas/neurons.parquet'
WHERE list_contains(CAST(ast_dist AS VARCHAR), 'for_stmt')
  AND fire_rate BETWEEN 0.001 AND 0.30
  AND n_fire >= 100
ORDER BY lift DESC LIMIT 50;
```
```python
# (d) LIVE query — "what is firing right now?"  (1 forward pass, ~1 s)
acts = run_and_capture(user_snippet)      # (T, 250880)
top = np.argsort(-acts[-1])[:32]          # last position
show(atlas_lookup(top))                   # join to parquet rows
```
- **Why the contrastive form is the whole trick.** A naive query returns "neurons active on code".
  The user wants "neurons *specific to* `for` loops". Subtracting the sibling class is what turns a
  census into an answer, and it costs one vector subtraction.
- **Precompute, don't scan.** Because `class_matrix` is materialised, every categorical query is a
  single matvec. Only free-text/live queries need a forward pass.

### 8. Minimal SAE (one layer, TopK, CPU)
```python
# 1) cache activations: 1M tokens x 1536 fp16 = 3.07 GB  (fits)
acts = np.lib.format.open_memmap("acts_L14.npy", mode="w+", dtype=np.float16, shape=(1_000_000, 1536))
# 2) TopK SAE: k=64, W=16*1536=24576  (TopK first; JumpReLU once it works)
#    loss = ||x - decode(relu(encode(x)))||^2 ; keep top-k by value ; resample dead latents
# 3) 10 epochs x 1M = 10M steps at ~177-265 steps/s => 10-16 h
```
Config provenance: Gemma Scope used widths 16,384 / 65,536 at d_model 2304 (≈7× / 28×) with
JumpReLU and L0 in the 20–100 band (https://arxiv.org/abs/2408.05147); TopK + dead-latent resampling
follows Gao et al. 2024 (https://arxiv.org/abs/2406.04093).

---

## What NOT to do (traps)

1. **Do not zero-ablate.** Zeroing removes the neuron's mean offset as well as its signal and
   overstates importance by a large factor. Mean-ablate (replace with the corpus mean, or the
   conditional mean for that token class). Every "neuron X is critical" claim built on zero-ablation
   is suspect.
2. **Do not compute exact sparsity on SwiGLU.** `a_j = silu(gate)·up` is signed and never exactly
   zero. "Sparsity" without a stated `τ` is meaningless. Report `P(|a| < τ)` and always state `τ`.
3. **Do not trust kurtosis.** It is the least stable statistic here and it *looks* the most
   scientific. Rank by p99/p99.9/absmax ratios.
4. **Do not label from top activations only.** That is the classic double-dip: you select the
   extremes, then describe them, then "verify" on the same extremes. Use corpus A to discover and
   corpus B (held-out repositories) to score. Report precision@k on B or don't report a label.
5. **Do not use an LLM labeler you can actually run.** A 1.5B explaining a 1.5B is not Bills et al.
   It will produce fluent, confident, unfalsifiable nonsense at scale. Use AST + vocab promotion;
   add an LLM only as a *fourth* channel with a distinct `label_method` value.
6. **Do not run batch-1 forward passes.** At ~1.9 tok/s you will conclude the project is infeasible.
   Batch 8–16 gets 8–12 tok/s. The whole budget depends on this.
7. **Do not plan on bf16.** No AVX512-BF16 on this chip. fp32 or nothing.
8. **Do not store per-token activations for all neurons.** 250,880 × 4 B = 1 MB/token → 500 GB for
   500 k tokens. Stream and aggregate. The only things that go to disk are moments, top-k, and the
   class matrix.
9. **Do not promise exhaustive causal coverage.** 343 h for all 250,880 ablations. Screen with
   gradients, verify a sample, and *say which neurons were not verified*.
10. **Do not treat the atlas as a circuit diagram.** Marginal effects do not compose additively. Two
    neurons with 2 % individual effect can have 60 % joint effect (redundancy and backup circuits are
    documented in the head-pruning literature). Say "marginal", always.
11. **Do not compare heads without accounting for GQA.** 12 query heads share 2 KV heads. Ablating a
    KV head silently ablates 6 query heads. Rank query heads via the `o_proj` input slice.
12. **Do not cite "A Walk Through the Layers".** See F3. It does not appear to exist.

---

## Open questions for other agents

- **A2 (interp methods):** my atlas assumes `down_proj`'s input is the canonical neuron unit. Do you
  want an additional per-layer *transcoder* view (input→output dictionary), which is what Anthropic's
  2025 attribution graphs actually use? It would change the schema's `unit` field.
- **A3 (steering):** the atlas gives you `promote_top10` per neuron, i.e. a ready-made bank of
  single-neuron steering candidates with a measured Δlogit. Do you want me to emit a
  `steering_candidates.parquet` (layer, index, direction = `W_down[:,j]`, predicted Δlogit,
  measured Δlogit)? That seems strictly more useful than a hand-built steering vector.
- **A4 (runtime):** can the runtime expose a **batched** `forward_trace` (B > 1)? My whole budget
  assumes batch 8–16. If the runtime is batch-1-only, every number in F12 must be multiplied by ~4.
- **A5 (UI):** the atlas is a 250 k-row parquet + a 100×250 k matrix. The right UI is a **faceted
  search over neurons** (filter by layer / AST class / fire rate / verified-causal) plus a live
  "what fires now" panel — essentially Neuronpedia's dashboard, but keyed on neurons and AST classes
  instead of SAE features. Do you want the atlas to pre-render top-100 lists per AST class as JSON
  so the UI needs no query engine?
- **A6 (validation):** I propose a falsifiable atlas-quality metric: **train a tiny classifier that
  predicts, from AST class + position features alone, which neurons will fire on a held-out file;
  report AUC.** If AUC ≈ 0.5 the atlas is describing nothing. Also: please independently re-run 20
  random ablations from my verified set — if we disagree, the ablation harness is broken, not the
  neuron.
- **A7 (CPU scale):** **your budget is wrong too.** This box is a 4-core 15 W i7-8665U with AVX2
  only and ~15 GB free RAM, not "8 cores / 31 GB". Measured: 54–67 GFLOPS fp32 GEMM, 9.7 GB/s
  effective streaming bandwidth, ~1.9 tok/s at batch 1. Please re-derive your 8B feasibility from
  these numbers before we commit.
- **A8 (guided compression):** the head atlas gives you a GQA-aware importance ranking (query heads
  individually, KV heads in groups of 6) plus per-neuron `ablate_dlogit`. Note the redundancy caveat
  (trap #10): marginal importance underestimates the cost of pruning *redundant pairs*. Consider
  ranking by *cluster* importance, not neuron importance.

---

## SCHEMA CHANGE

**Proposal 1 — extend the canonical trace with an optional sparse neuron field.**
Do **not** put 250,880 floats per step into the trace (that is 1 MB/step and would swamp the schema).
Add an optional, capped field:

```json
{"step": 0, "token_id": 123, "token_str": "for",
 "layers": [{"layer": 14, "resid_norm": 12.3,
             "logit_lens_topk": [[id, tok, prob]],
             "attn_entropy": 1.7, "mlp_out_norm": 4.1,
             "neurons_topk": [[3021, 4.82], [4402, 3.10]],     // NEW, optional, k<=32
             "heads_topk":   [[3, 0.61, "sink"], [7, 0.22, "prev"]]}],  // NEW, optional
 "logits_topk": [[id, tok, prob]], "chosen": [id, tok, prob]}
```
Rationale: the live microscope needs *which* neurons fired, not all 8960. `k ≤ 32` keeps the trace
under ~10 KB/step.

**Proposal 2 — a separate artifact spec (not part of the trace).** The atlas is a build product, not
a per-token record. Specify it as `atlas/{neurons.parquet, heads.parquet, class_matrix.npy,
contexts.parquet}` with the schema in §6, versioned by `atlas_meta.json`:
```json
{"model":"Qwen/Qwen2.5-Coder-1.5B-Base","dtype":"float32","corpus":{"tokens":500000,
 "tierA":"the-stack-v2/python","tierB":"synthetic-construct-probes-v1","holdout_repos":37},
 "tau_policy":"per-neuron 25% of p99 of |a| (calibration pass, 25k tokens)",
 "ablation":"mean","verified_neurons":2000,"harness":"A9/v1","date":"2026-11-10"}
```

**Proposal 3 — `CONTEXT.md` hardware correction.** It currently says "8 CPU cores, 31 GB RAM" and
"expect 5–20 tok/s". Measured truth: **4 cores / 8 threads, i7-8665U @ 1.9 GHz, AVX2 only, ~15 GB
free RAM, ~1.9 tok/s at batch 1 / 8–12 tok/s at batch 8**. This propagates into A7 and A8 and should
be fixed before they finalise.

---

## Build next (ordered, with effort estimate)

| # | task | effort | why first |
|---|---|---|---|
| 1 | **Re-benchmark the box** with the §0 snippet; correct `CONTEXT.md`. | 15 min | Every downstream estimate depends on it. Currently 5× off. |
| 2 | **Hook `down_proj`, capture one batch, verify shape `(B,T,8960)`** and that `n_fire > 0` for ~5–30 % of neurons. | 1 h | Confirms the hook point and measures real SwiGLU sparsity — kills F4's main uncertainty. |
| 3 | **Build the corpus**: Tier A (300 k Python tokens, repo-held-out split) + Tier B (~30 k synthetic construct probes) + AST annotation. | 1.5 h | The AST classes *are* the labelling engine. |
| 4 | **Calibration pass** (25 k tokens) → per-neuron τ + histogram; report the fire-rate distribution. | 1 h | Sets the threshold for the whole harvest; also validates the 60–90 % sparsity guess. |
| 5 | **Harvest 500 k tokens** into the streaming Atlas (batch 16, seq 512). | **17.4 h** | The long pole. Run it and go do something else. |
| 6 | **Vocabulary-promotion matmul** (`W_U · W_down`) for all 250,880 neurons; top-10 promote/suppress. | 0.7 h | First real labels, zero LLM, exact. |
| 7 | **Head atlas, exhaustive**: 336 mean-ablations batched + induction/sink/entropy/prev-token. | 0.5 h | Cheapest high-value deliverable; 100 % coverage. Do it while the harvest runs. |
| 8 | **Gradient attribution** over 50 k probe tokens → `attr_rank` for all neurons. | 4.5 h | Turns "250 k ablations = 2 weeks" into a ranked list. |
| 9 | **Ablation-verify the top 2,000** neurons (batched, mean-ablation). | 3 h | The causal claims. Only these get `ablate_verified = true`. |
| 10 | **Co-activation clustering** + write `neurons.parquet` / `class_matrix.npy` / `atlas_meta.json`. | 0.3 h | Ships the artifact. |
| 11 | **Hold-out evaluation** on corpus B: label precision@20, and the A6 AUC test. | 1 h | Without this the atlas is unfalsifiable. |
| 12 | *Optional overnight:* **TopK SAE, layer 14, 16×, 1 M acts × 10 epochs.** | 15.7 h | Adds a monosemantic re-basis for one layer. Only if 1–11 finished early. |

**Total without the SAE: ~30 h.** With it: ~46 h. Fits the 2-day window with ~15 % slack for
thermal throttling and the inevitable hook bugs.

**Do not attempt:** exhaustive ablation of all 250,880 neurons (343 h), SAEs on all 28 layers
(~440 h), or 2 M-token harvests (69 h) inside this window.
