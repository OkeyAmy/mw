# F3 — Transfer limits: can a direction cross a model boundary?

**Question owned:** the owner's claim that "this is a backward loop that can allow us" —
i.e. that a closed / API model might still be controllable (or readable) *indirectly*,
through a loop that goes local-model → direction → closed-model.
**Verdict up front:** the belief is **defensible for one narrow loop and false for the one
that matters**. Behaviour can be pushed on a closed model through the *input* channel
(prompt / image / logit bias). A *direction* cannot cross the boundary, because fitting
the map requires the target's activations, and injecting the vector requires a runtime
hook the API does not expose. Details and the cheapest falsifier below.

**Reading conventions:** `[OBS]` = measured and quoted from a fetched source
(URL in §Sources). `[INF]` = my inference. `[SPEC]` = speculation.
Preprints from 2026 are marked `[preprint]` — several are not peer-reviewed and two of
them contradict each other; I say so where it matters.

---

## TL;DR

1. **PRH is real but much weaker than its reputation.** Huh et al. report that CKA shows
   "a very weak trend of alignment between models, **even when comparing models within
   their own modality**" and had to switch to mutual-kNN. `[OBS]` Later work shows the
   global CKA trend largely disappears under width/depth calibration, and that mutual-kNN
   alignment collapses when the probe set grows from 1k to 15M samples. What survives is
   **coarse neighbourhood structure**, not a shared coordinate system.
2. **Every credible cross-model *steering* result uses target-side information.** Either
   the two models share weights (base↔instruct), or a map is fitted on paired activations
   from **both** models. I found **no** peer-reviewed result where a direction from model A
   changes behaviour in an independently trained model B with no target-side fitting.
3. **Where transfer is measured, the numbers are modest and task-dependent.** Language
   steering survives an affine stitch "for some languages but not for others"; instruction
   steering does **not** ("in no case did transfer steering improve upon ground truth
   steering"). Cross-family zero-shot refusal transfer: TE 0.03–0.17, cosine 0.019.
4. **Tokenisers are the hard wall.** Qwen 151,936 / GPT-2 50,257 / SmolLM2 49,152. The
   residual stream is indexed by token *position*; different tokenisers mean position *i*
   is a different span of text. The LRT paper calls tokenisation transfer "an open
   problem"; the transfer paper that *does* cross tokenisers had to resize attention masks
   and admits imperfect mapping.
5. **Identifiability is worse than people think, in all three regimes.** With weights +
   activations, a steering vector is *provably* non-identifiable up to the null space of
   the activation→logit Jacobian (infinitely many behaviourally equivalent vectors). With
   behaviour only, you cannot even distinguish "activation-steered model" from "different
   checkpoint".
6. **The strongest steelman is an input-space loop, not an activation loop.** Prompt↔weight
   duality is a theorem (a prompt's effect *is* a rank-structured weight update), and
   optimised inputs can emulate directions on closed models (VISOR++). But the emulation is
   measurably lossy and, on Claude 3.5, produced almost no effect.
7. **Decision:** kill the closed-model leg now (not falsifiable under our constraints), run
   **one** open-model cross-tokeniser test as the falsifier, and keep the same-lineage
   transfer as a positive control. Total ≈ 1.6 GB download, ≈ 45 min CPU.

---

## Method note (so the reader can discount appropriately)

I searched for, and fetched, the primary sources listed at the end. Where only a search
snippet was available I say so. I did **not** run any model — every claim about
*our* Qwen2.5-Coder-1.5B is `UNVERIFIED — needs runtime`, per `VALIDATION.md` rule 3.
Two things I explicitly failed to find and will not pretend exist:

- A paper titled or equivalent to **"every model is many models"**. The concept is real
  under other names: functional equivalence / permutation symmetry, and *predictive
  multiplicity*. I cite the actual literature instead of the phrase.
- Any peer-reviewed demonstration of **zero-shot** steering-vector transfer between
  independently trained models of **different families and tokenisers** that works. The
  one systematic claim I found in that direction is a preprint with n=1 model per size
  tier below 7B, and a blog with two model pairs.

---

## Findings

### F1. Representational convergence — what PRH actually measured

**What it is.** Huh, Cheung, Wang, Isola, *The Platonic Representation Hypothesis*,
arXiv:2405.07987 (2024; ICML 2024). Claim: as models get more capable they measure
distance between datapoints more similarly, converging on a shared statistical model of
reality.

**The measured evidence, quoted.**
- `[OBS]` "Our initial efforts to measure alignment with CKA revealed a very weak trend of
  alignment between models, even when comparing models within their own modality." They
  therefore introduce mutual-kNN as the primary metric.
- `[OBS]` Vision panel: 78 models, Places-365. Alignment rises with VTAB competence.
- `[OBS]` Cross-modal panel: vision vs language models on WIT — **1024 image–caption
  pairs**, one-to-one pairing.
- `[OBS]` Their framing admits the limit: "all strong models are alike, each weak model is
  weak in its own way"; the mechanism is a *Capacity Hypothesis*, i.e. bigger function
  classes approximate the same optimum better.

**Three caveats that bite us.**
1. **It is measured on pooled datapoint embeddings, not token-level residual streams.**
   A 1024-sample sentence-embedding similarity statement says nothing about whether
   `resid_post[8]` at token position 37 of model A corresponds to anything in model B.
   Our steering vectors live in exactly the space PRH does not measure. `[INF]`
2. **The global-geometry claim is contested.** `[OBS]` Gröger et al., *Revisiting the
   Platonic Representation Hypothesis: An Aristotelian View*, arXiv:2602.14486
   `[preprint]`: re-running Huh's protocol with a permutation null-calibration against
   width/depth "removes the confound"; calibrated CKA shows **no** systematic increase with
   model size, while neighbourhood metrics keep their trend.
   `[OBS]` Köpke et al., *Back into Plato's Cave*, arXiv:2604.18572 `[preprint]`: mutual-kNN
   on WIT-1024 is 0.135 (k=10) but falls to **0.008** on LAION-15M at fixed k; same-modality
   pairs stay at 0.35–0.62. Their conclusion: models "share coarse structure without
   evidence of convergence to a shared representation".
3. **The metrics are dangerously easy to fool.** `[OBS]` arXiv:2608.16347 `[preprint]`,
   Table 1: for Qwen2↔FLAN-T5 at middle layers, **CKA is 0.013–0.015 for trained models
   and 0.81–0.82 for randomly initialised ones**; orthogonal Procrustes is 0.11–0.12
   trained vs 0.95 untrained. Anisotropic activation magnitudes dominate. Rank-based
   mutual-kNN is the only metric that behaves (trained > untrained by 3–6×).
   `[INF]` **Consequence for us: any CKA or Procrustes number we report without a shuffled
   / random-init null is noise.** This is the same failure mode as L1 in `VALIDATION.md`
   (monotonicity is not specificity), one level up.

**Confidence:** high on the *caveats*, medium on the *positive* claim.
**Why it matters:** PRH is the load-bearing citation for "the geometry is universal", and
it does not say what the slogan says. It says *coarse neighbourhoods* are shared under a
specific 1k-sample protocol; it explicitly does not claim a shared coordinate system.

---

### F2. Steering-vector transfer — the effect sizes, ranked by how much they actually show

I ordered these by evidential strength, because the literature routinely conflates them.

| # | Setup | Shared weights? | Shared tokeniser? | Target-side fitting? | Measured effect |
|---|---|---|---|---|---|
| 1 | Gemma-2 2B **base → it** (arXiv:2410.12877) | **yes** (it = fine-tune of base) | yes | none | cross-model steering **outperforms** same-model steering on Gemma 2B `[OBS]` |
| 2 | Qwen2.5 base/instruct/math, 1.5B & 7B (arXiv:2509.06608) | **yes** | yes | none | relative gain 0.94 (base→instr) down to **0.05** and **−0.34** (math→instr 7B); LLaMA-3.1 base↔instr **0.01 / −0.01** `[OBS]` |
| 3 | Gemma-2 2B → 9B, affine stitch (arXiv:2506.06609) | no | yes | **yes** (paired activations) | language steering works for some languages, bimodal gap; **IFEval instruction steering fails** `[OBS]` |
| 4 | Qwen 0.5B → 1.5B, autoencoder map (arXiv:2503.04429) | no | yes | **yes** (nonlinear AE) | LLM-judge text similarity 2.6/5; KL 7.86 `[OBS]` |
| 5 | Qwen→Llama "similar tokenizer" (arXiv:2503.04429) | no | no | yes | 3.0/5 text similarity — *better* than same-family 2.6 `[OBS]`, which tells you the metric is noisy |
| 6 | Gemma 2B → Llama 3B, different tokeniser (arXiv:2503.04429) | no | no | yes | 1.2/5 text, 2.40 coherence; −150% vs the Qwen→Llama row `[OBS]` |
| 7 | Qwen 7B → Gemma 9B, **zero-shot**, matched hidden dim 3584 (blog, not peer-reviewed) | no | no | **none** | transfer efficiency **0.17 / 0.03**, cross-cosine **0.019**; same-family Qwen 14B↔32B TE **1.25 / 1.00** `[OBS]` |

**The pattern.** Effect size tracks exactly one variable: *how much of the target model you
were allowed to touch*. Shared weights → near-full transfer. Fitted map + shared tokeniser →
partial, task-dependent. No fitting, no shared tokeniser → collapse.

**The disagreement you must not paper over.** arXiv:2506.00653 (LRT hypothesis) reports that
an affine-mapped vector from Gemma-2-2b has "the same effect as one learned natively" in
Gemma-2-9b (high correlation, low MSE across α). arXiv:2506.06609 (the NeurIPS version of
the stitching work) reports that for IFEval instruction steering "in no case did transfer
steering improve upon ground truth steering". Both can be true: the *response-language*
task transfers, the *instruction-following* task does not. The difference is whether the
behaviour is a low-dimensional output bias or a compositional one. `[INF]`
Note also 2506.00653's own remark on the concurrent work: Oozeer et al. "report mixed or
negative results for **affine** transformations" and needed a nonlinear autoencoder.

**A distinction that gets conflated, and shouldn't.** `[OBS]` *Refusal Direction is
Universal Across Safety-Aligned Languages* (arXiv:2505.17306, NeurIPS 2025) shows that a
refusal vector extracted in English transfers to 13 other languages "with near-perfect
effectiveness", and that vectors from German/Chinese/Thai all collapse refusal across each
other (compliance → 90%+). That is a genuinely large universality result — **but it is
universality across languages inside one model's residual stream, not across models.**
Arditi et al. (arXiv:2406.11717) are explicit that each of the 13 chat models gets its own
separately estimated, model-specific direction. `[INF]` The strongest "universal direction"
evidence in the literature is therefore evidence for a shared *concept geometry within a
model*, which is a different claim from a shared *coordinate system between models*.

**A trap in reading #2.** The LLaMA-3.1 rows show base→instruct = 0.01 and instruct→base =
−0.01 **despite identical weights and the same tokeniser**. The authors attribute it to chat
templates. `[INF]` This is the cheapest possible transfer setting and it still produced a
null, because the *tokenised prompt distribution* changed. Any transfer experiment must
match prompt formatting before it can claim a geometric result.

---

### F3. Mapping one space onto another with paired activations

**What exists, and how well it works.**

- **Linear identifiability is a theorem, with a caveat.** Roeder, Metz, Kingma, *On Linear
  Identifiability of Learned Representations*, arXiv:2007.00810 (ICML 2021): a large family
  of discriminative models is identifiable **in function space, up to a linear
  indeterminacy** `[OBS]`. That is the justification for Procrustes/affine maps being the
  natural hypothesis class — not for those maps succeeding.
- **Orthogonal Procrustes, measured.** `[OBS]` arXiv:2608.27115 `[preprint]`, Table 4, 36
  language pairs of *monolingual* models: Procrustes P@1 = **0.887**, affine 0.814, MLP
  0.809, **identity 0.002**. Procrustes beats affine/MLP on retrieval despite worse MSE.
  Their causal follow-up: patching a rotated residual into another model flips a factual
  cloze prediction in **66–85%** of cases against a 76–98% within-model ceiling.
- **Five architectures, one laptop.** `[OBS]` arXiv:2610.33495 `[preprint]`: 5 independently
  trained models (4 transformer labs + Mamba), final layer, no weight merging. Pairwise
  least-squares maps: held-out retrieval **0.82–0.85** (chance 0.031). Generalized
  Procrustes to a shared space: **0.91–0.93**. Runs on *one laptop, no GPU*.
- **The depth structure is the killer for us.** Same paper: "in the middle of the network
  the pairwise maps are far better (0.56–0.67 against 0.26–0.40 at 25–75% depth), in line
  with the earlier finding that **the middle layers of these models are individual**."
  `[OBS]` The GPA space also "fails on mathematical and logical notation" `[OBS]` — a
  direct problem for a *coder* model.
- **Stitching preserves function, not steering.** arXiv:2506.06609: affine stitches in
  Pythia, GPT-2, Gemma-2; next-token loss after big→small stitch is within **<1%** of
  ground truth, small→big costs ~8–13%; but the down-stitch is bounded by the weaker model
  ("downstream performance is consistently bottlenecked by the worse model A").
- **Causal injection is the weakest link.** `[OBS]` arXiv:2608.16347 `[preprint]`: a learned
  MLP projection retrieved the correct target state at 45–50% top-1 (chance 5%) for three
  decoder pairs, yet **injecting** the projected state produced a significant effect for
  only **1 of 3** pairs (23.3% vs 0.0% negative control, p_FDR = 0.047) and for the other
  two the real condition scored **0.0%**, *below* the negative control. Their summary line
  is the one to keep: "evidence for causal transfer of the representational **vehicle**, not
  of meaning".

**Answer to the question as asked.** A map fitted on paired activations reaches high
*retrieval* accuracy (0.82–0.93) at final layers. It retains **causal** effect only for
behaviours that are approximately a linear output bias, and even then partially. At
mid-depth — where our own working vector lives — alignment is much weaker. `[INF]`

---

### F4. Tokeniser barriers — what actually breaks

Verified configs (HF `config.json`, fetched):

| model | hidden | layers | vocab |
|---|---|---|---|
| Qwen2.5-Coder-1.5B (ours) | 1536 | 28 | **151,936** |
| Qwen2.5-Coder-0.5B | 896 | 24 | **151,936** |
| HuggingFaceTB/SmolLM2-360M | 960 | 32 | **49,152** |
| openai-community/gpt2 | 768 | 12 | **50,257** |

**What breaks, concretely.**

1. **The index set changes.** A residual-stream tensor is `[seq_len, d_model]`. `seq_len` is
   a function of the tokeniser. The same 40-word prompt can be 52 tokens in Qwen and 61 in
   GPT-2. There is no canonical position `i` to align. `[INF]`
2. **The units change.** A concept can be one token in one model and 3 tokens in the other.
   A direction that operates "at the concept token" has no single target position. `[INF]`
3. **The embedding matrices are not comparable at all.** Different vocabulary → different
   row sets. Only the *residual* space is a candidate for a map, and only at aligned spans.
4. **Measured cost of getting this wrong.** `[OBS]` arXiv:2503.04429, Table 2: same-family
   Qwen0.5B→Qwen1.5B scores 2.6 text / 4.60 coherence; Gemma2B→Llama3B (different
   tokeniser) scores **1.2 / 2.40**, a −150% / −92% gap. Their method for crossing
   tokenisers is to **resize the smaller attention mask to the larger one**, and they
   concede "this doesn't guarantee perfect token-level mappings, some non-padded tokens in
   the smaller sequence map to padded tokens."
5. **The field agrees it is unsolved.** `[OBS]` arXiv:2506.00653: "A major obstacle here is
   the lack of a reliable method for transferring tokenization schemes. This challenge is
   also observed in the literature on cross-tokenizer model distillation … and **remains an
   open problem**."

**Is the residual stream even comparable?** Partially, and only statistically. `[INF]`
Two models with different tokenisers can be compared at the level of *pooled span
embeddings* (mean over the character span), and then Procrustes alignment works (that is
exactly what arXiv:2608.27115 does across languages/scripts). But a *steering vector* is not
a pooled span embedding. Applying it requires per-position injection, and per-position is
exactly what tokenisers destroy.

---

### F5. Identifiability limits, by access regime

First a correction to the question's framing: **(b) white-box weights without activations is
not a distinct regime** — if you have the weights you can compute activations by running the
model. Informationally (b) ⊇ (c). The only real distinction is *behaviour-only* vs
*weights+activations*. `[INF]`

**(a) Black-box, logprob access only.**
- What you *can* get: `[OBS]` Carlini et al., *Stealing Part of a Production Language Model*,
  arXiv:2403.06634 — recovers the **embedding projection layer up to symmetries** from
  typical API access; Ada/Babbage projection matrices for under $20; gpt-3.5-turbo hidden
  dim recovered, ~$2k for the full matrix. So black-box is not information-free.
- What you *cannot* get: the residual stream at any interior layer. The API exposes a
  distribution over vocabulary, not a state. `[INF]`
- Formal limit: `[OBS]` arXiv:2608.24662 `[preprint]`, Proposition 1 — for any target
  distribution Q with supp(Q) ⊆ supp(P_θ), there exists a logit-level inference policy I
  with P_{θ,I} = Q. Hence "a base model plus a steering policy" and "a different model" are
  **observationally identical**. Detection of a behavioural shift ≠ attribution of where it
  came from. This is the precise sense in which a black-box loop cannot close.

**(b/c) White-box, weights + activations.**
- **Identifiability of representations is only up to linear indeterminacy** (arXiv:2007.00810),
  and **parameter identifiability is only up to the symmetry group** — permutation and
  scaling/affine symmetries. `[OBS]` Kaba & Ravanbakhsh, *Affine symmetries and neural
  network identifiability*, arXiv:2006.11727: the set of networks realising a given function
  obtained via symmetries is "in fact exhaustive" for tanh-type nonlinearities. `[OBS]`
  Entezari et al., arXiv:2209.04836: permutation symmetry means independently trained nets
  live in one basin up to a permutation you must *find*.
- **Steering vectors specifically are non-identifiable.** `[OBS]` *On the
  Non-Identifiability of Steering Vectors in Large Language Models*, arXiv:2602.06801
  `[preprint]`, Proposition 1: under white-box single-layer access, "for any steering vector
  v there exist **infinitely many** vectors v′ ∝̸ v that are observationally equivalent
  under linearization", because the activation→logit Jacobian has a large null space.
  Empirically, orthogonal perturbations achieve near-equivalent efficacy. Their own
  conclusion is careful and correct: "This does not mean steering is ineffective … The
  limitation is representational, not functional."
- **Model multiplicity.** `[OBS]` Marx et al., *Predictive Multiplicity in Classification*,
  arXiv:2002.06694: many near-equally-accurate models disagree on individual predictions.
  `[INF]` So even a *perfect* behavioural clone of a closed model is one member of a large
  equivalence class; recovering "the" direction from behaviour is ill-posed, not merely hard.

**Bottom line.** The impossibility is not "we lack data". It is that the object we want
(the direction) is not uniquely defined even with full access, and the object we can observe
(behaviour) cannot distinguish the mechanisms that produce it.

---

### F6. Steelman vs rebuttal

I will state the owner's position in its strongest form.

#### Steelman (best case that a backward loop works)

1. **The geometry is genuinely shared at the coarse level, and this survives null
   calibration.** Mutual-kNN alignment persists after permutation calibration (2602.14486),
   and five independently trained models from four labs plus a state-space model can be put
   into one space with Procrustes at **0.91 retrieval on a laptop** (2610.33495). This is
   not nothing.
2. **Transfer demonstrably works when the loop is given a little target-side signal.**
   Base→instruct transfers at 0.94 relative gain in Qwen2.5-1.5B (2509.06608); affine
   stitched language vectors work across a 4.5× size gap (2506.06609); patching a rotated
   residual flips a factual prediction 66–85% of the time (2608.27115).
3. **Prompt and activation are not different kinds of thing.** `[OBS]` *Transmuting prompts
   into weights*, arXiv:2510.08734: a prompt's effect on a transformer block is exactly a
   token-dependent rank-structured weight update; averaging gives the "thought vector",
   which is provably the least-squares analogue of a diff-in-means steering vector. If a
   prompt is a weight update and a steering vector is a mean weight update, then **a prompt
   is a legal, if lossy, encoding of a direction** — and prompts work on closed APIs.
4. **Input-space emulation of a direction has been demonstrated on closed models.**
   `[OBS]` VISOR++ (arXiv:2509.25533): a single optimised image per model ensemble
   "emulat[es] each of their steering vectors", with directional consistency on 6 of 7
   unseen models including GPT-4 variants; 99.9% MMLU preservation. This is the existence
   proof that the boundary is permeable in principle.
5. **Therefore**: hypothesis direction on the local model → compile it into an input
   perturbation → measure the behavioural shift on the API model → update the hypothesis.
   That is a real loop, and it terminates in a real measurement.

#### Rebuttal (why that loop does not deliver what the owner wants)

1. **Every success in (2) used target-side activations or shared weights.** Strip those and
   the effect size goes to the values in F2 row 7: TE 0.17 / 0.03, cosine 0.019. The loop
   in (5) is not the loop in (2); it is a different method with a different (much weaker)
   evidence base.
2. **The input-space encoding is measurably lossy, and the loss is large.** `[OBS]`
   Subramani et al., *Extracting Latent Steering Vectors from Pretrained LMs*,
   arXiv:2205.05124, Table 5: injecting **one** vector into the hidden states beats
   concatenating **50** prompt-space vectors (reconstruction BLEU 81.7 at k=1 rising to
   98.5 at k=50 — i.e. you need ~50 slots and it still trails direct injection). "Even with
   k=50 … recovery fails to match that of a single z_steer injected into the hidden states."
3. **On the one closed model where it was tried, emulation failed.** `[OBS]` VISOR++:
   "We observe that the steering images have almost no effect on Claude Sonnet 3.5." The
   GPT-4 deltas are "modest"; only the *sign* is consistent.
4. **The ceiling is low before transfer even starts.** `[OBS]` AxBench (arXiv:2501.17148):
   prompting outperforms all representation-based steering; diff-in-means 0.409 vs prompt
   0.905 (as recorded in our `LEDGER.md` [B1]). `[OBS]` Tan et al. (arXiv:2407.12404):
   steerability is "highly variable across different inputs" and brittle to "reasonable
   changes in the prompt". A transferred vector is a *worse* version of an already-weak
   instrument.
5. **The measurement problem is fatal, not inconvenient.** Per (a) above, even a positive
   behavioural result on the API model cannot be attributed to the injected direction rather
   than to the prompt that encoded it. And a prompt that reliably changes behaviour is
   called prompting. The loop "works" only in the sense that it degenerates into prompt
   engineering with extra steps. `[INF]`
6. **Practical:** the vendor can change the served model or its inference policy underneath
   you, and black-box observation cannot tell you that it happened (arXiv:2608.24662
   Prop. 1). The blog title is blunt and correct: *You can't steer what you can't open*
   (worldagrees.com, 2026-10-11) — the UK AISI reproduced Anthropic's steering result on
   open models because "the ability to reproduce results using open source models was a key
   enabler"; the same experiment could not be run on Claude because "there is no knob for
   that on the API, and there never will be."

#### Where the two sides actually agree

Both agree that (i) coarse representational structure is shared, (ii) a fitted map can carry
a direction, and (iii) an input can approximate a direction. They disagree only on whether
(iii) is *the same thing* as (ii). It is not: it is strictly weaker, and the weakening is
large enough to fall below the noise floor of prompting.

---

## Concrete implementation — the two tests worth running, and their arithmetic

Hardware per `CONTEXT.md`: 4 physical cores, AVX2, ~17 GB free, ~50 GFLOPS fp32,
~49 tok/s batched generation, ~2.6 GFLOP/token at 1.5B.

### T1 — Positive control: same-tokeniser small→large (must pass, or the whole idea dies early)

- **Pair:** `Qwen/Qwen2.5-Coder-0.5B` (L6/24 ≈ 25% depth) → `Qwen2.5-Coder-1.5B` (L8/28 ≈
  29% depth). Same tokeniser (151,936), same pretraining lineage, same code domain.
- **Map:** ridge least squares `h_tgt ≈ h_src W + b`, fit on 192 prompts, test on 64.
  Vectors transferred with `W` only, no bias (matches 2506.06609 Eq. 6).
- **Vector:** our existing comment-mode diff-in-means, computed on the 0.5B.
- **Readout, cheap first:** logit-level only. For held-out prompts, measure
  `Δ = log P(comment tokens) − log P(baseline)` at the first generated position.
  Compare against **20 random vectors of matched norm** — the L1 criterion from
  `VALIDATION.md`, not a dose-response curve.
- **Only if it passes:** batched generation, comment-rate behavioural check.
- **Pass criterion, pre-registered:** mapped vector beats `k/20` random vectors on held-out
  sign consistency, `k ≥ 19`. Anything less is a kill.
- **Cost:** 0.98 GB download; forward passes 2×24,576 tokens ≈ 8 min (0.5B) + 25 min (1.5B);
  activation cache 84 MB + 144 MB per layer; Procrustes is a 896×1536 ridge solve (≈ms);
  generation only on the pass branch ≈ 65 s/condition × 21 ≈ 23 min.
  **≈ 40 min total, ≈ 1 GB.**

### T2 — The falsifier: different tokeniser *and* different family, no shared lineage

- **Pair:** `openai-community/gpt2` (768, 12 layers, vocab 50,257) →
  `Qwen2.5-Coder-0.5B` (896, 24, vocab 151,936).
- **Alignment:** character-offset span alignment (as arXiv:2603.18908 does), then pool each
  character span to one vector before fitting the map. This deliberately gives the
  hypothesis its best shot by removing the positional mismatch from the *similarity*
  measurement, and leaves it in place for the *causal* test.
- **Mandatory nulls (this is the part everyone skips):** shuffled-prompt CKA null, and a
  random-init GPT-2 null. Per arXiv:2608.16347, untrained models can score *higher* on CKA
  and Procrustes than trained ones. Report retrieval P@1, not CKA, as the headline.
- **Causal test:** same logit-level readout as T1, same 20 random controls.
- **Pre-registered prediction (I am stating this before the run):** retrieval above chance
  on pooled spans (0.2–0.5), **causal transfer at or below random**. If causal transfer
  clears 19/20, the owner is right and I am wrong; that is the falsifier.
- **Cost:** 0.55 GB download; GPT-2 forwards ≈ 2 min; total **≈ 15 min, 0.55 GB**.

### What I would *not* build

A "closed-model loop" harness. Under our constraints (no API budget) the terminal
measurement cannot be made even once, so any result would be `UNVERIFIED` by construction —
see the decision rule.

---

## Decision rule

Ranked by *information per CPU-minute* under our constraints.

| Rank | Loop | What it would settle | Prior from literature | Cheapest falsification | Verdict |
|---|---|---|---|---|---|
| 1 | **T2** cross-tokeniser, cross-family, no fitting of target at inference | whether a direction is *portable* without target data | negative (TE 0.03–0.17; −150% text quality) | logit-level Δ vs 20 randoms, 15 min | **RUN FIRST.** It is the actual question. |
| 2 | **T1** same-lineage 0.5B→1.5B affine | whether *any* transfer machinery works here | positive (0.94 relative; stitch works) | same protocol, 40 min | **RUN SECOND** as positive control. If T1 fails, T2 is uninterpretable. |
| 3 | Distil API → local surrogate, steer the surrogate | gives us a steerable stand-in | feasible but well-trodden | n/a | **VALID but reframe.** It is not a backward loop; it is distillation. Do not call it control of the closed model. |
| 4 | Image/input-space universal steering | closed-model control | weak, VLM-only | needs a VLM + API | **DEAD END** for a text-only coder model. |
| 5 | Prompt/logit-bias emulation of a discovered direction | "we can steer the API model" | degrades to prompting (Subramani; AxBench) | free tier, but attribution impossible | **DEAD END as a scientific claim.** Fine as a demo; never as a finding. |
| 6 | Fit a map *to* a closed model | transfer across the boundary | method undefined (needs target activations) | — | **DEAD END, definitionally.** Not a measurement problem. |

**One-sentence rule:** *Only run loops whose terminal measurement is inside a model we can
open; treat every closed-model loop as a prompting study wearing a steering costume.*

**Corollary for the product question (C-3 in `STATE_OF_PROJECT.md`):** the transfer result
is *not* the wedge, because the only setting where it works (same lineage, fitted map) is
also the setting where the customer already has both models. The reusable artefact is the
**protocol**: retrieval + null-calibrated CKA + random-vector controls. That is the same
machinery the quantisation certificate (E3) needs.

---

## What NOT to do (traps)

1. **Do not report CKA or Procrustes without a null.** Untrained models score 0.81 vs 0.013
   for trained (2608.16347). A raw CKA of 0.7 is not evidence of anything.
2. **Do not call base→instruct transfer "cross-model".** It is a fine-tune of the same
   weights. The literature does this constantly; it inflates the apparent universality.
3. **Do not transfer a vector and report only the mean effect.** Report the per-prompt
   distribution; the stitching paper's language result is explicitly *bimodal* — "works for
   some languages but not others", concentrated at 0 and 1.
4. **Do not validate on generation alone.** A changed output can come from degraded fluency.
   Use matched-norm random controls and a coherence check (the 2503.04429 table shows how:
   text similarity *and* distribution *and* coherence).
5. **Do not assume mid-depth transfer.** Our working vector is L8/28 (29%). That is inside
   the band where cross-model maps are worst (0.26–0.40 vs 0.82–0.93 at final layer). If a
   final-layer map works and L8 does not, that is a real result, not a bug.
6. **Do not treat a 2026 preprint as settled.** Two of the load-bearing 2026 papers here
   contradict each other on whether local alignment survives scale (2602.14486 vs
   2604.18572), and 2608.05164's "1.7B threshold" rests on n=1 model per tier below 7B.

---

## Open questions for other agents

- **A3 (steering):** does our comment-mode vector survive at fractional depth 25% in the
  0.5B? T1 needs a matched source layer, and we have never swept the 0.5B.
- **A6 (limits):** is the logit-level readout a legitimate proxy for the behavioural claim,
  or does it smuggle in the "correlation is not causation" error? I believe it is legitimate
  *as a screen* because it is still an intervention with a random control, but it should be
  labelled a screen.
- **E3 (wedge):** the null-calibrated retrieval protocol is reusable. Does the certificate
  need cross-model transfer at all, or only fp16-vs-Q4 within one model?
- **C-7 (model choice):** if we switch to an instruct model for refusal work, T1 and T2 get
  cheaper to interpret (refusal has the strongest published transfer evidence).

---

## SCHEMA CHANGE

One addition proposed to the canonical trace schema, needed for anything in this document:
a top-level `"nulls": {"cka_shuffled": float, "procrustes_random_init": float,
"retrieval_chance": float}` block. Without it a trace cannot be audited for the
anisotropy failure in F1.3.

---

## Build next (ordered, effort)

1. `code/pair_activations.py` — run two models over the same strings, emit aligned span
   activations + a `spans.json` character-offset map. **~150 lines, 2 h.**
2. `code/transfer.py` — ridge/Procrustes fit, retrieval P@1, shuffled null, random-init
   null. **~120 lines, 2 h.** No model needed to write it; test on synthetic rotations.
3. `code/transfer_probe.py` — logit-level Δ vs 20 matched-norm randoms, pre-registered
   criterion in a config file. **~100 lines, 2 h.**
4. Run **T1** (40 min), then **T2** (15 min). Write the result into `agents/F3_*` and hand
   to the orchestrator for the audit sample.

Total: **~6 h engineering + ~1 h compute + 1.6 GB disk.**

---

## The three-way split

### POSSIBLE (measured, reproducible, or a theorem)

- Reading and causally steering a model we can load. Settled for this project (C1–C6).
- Transferring a direction between models that **share weights** (base↔instruct) or **share
  a tokeniser + pretraining lineage** with a fitted affine map. Effect: 0.94 relative gain
  (Qwen2.5-1.5B base→instruct); partial and task-dependent across a size gap.
- Fitting an affine/Procrustes map from paired activations: retrieval **0.82–0.93** at final
  layers, **0.26–0.67** at mid-depth. Zero-shot latent communication across independently
  trained models at the final layer (Procrustes P@1 0.887).
- Recovering a closed model's **projection layer up to symmetries** from a full-logit API
  (Carlini et al.), for tens to thousands of dollars.
- Emulating *some* of a direction's effect through optimised inputs (VISOR++), with modest
  magnitude and only directional consistency.

### IMPOSSIBLE (proof or definition, not "we haven't tried")

- Injecting a vector into a closed API model's residual stream. No interface exists; this is
  the whole reason the open-model literature exists.
- Fitting a source→target map **without target activations**. The map's domain is
  `R^{d_tgt}`; you cannot estimate it from strings.
- Identifying **the** steering vector even with full white-box access: the Jacobian null
  space makes infinitely many vectors behaviourally equivalent (2602.06801 Prop. 1).
- Attributing an observed behavioural change to activation steering rather than prompting,
  a different checkpoint, or an inference policy, from black-box observation alone
  (2608.24662 Prop. 1).
- Exact token-level transfer across different tokenisers. No reliable tokenisation-transfer
  method exists; the field calls it an open problem (2506.00653).

### UNKNOWN (testable by us, cheaply, and not yet settled)

- Whether a direction transfers **zero-shot** across independently trained families at
  <2B. Cross-family evidence is negative but thin: two model pairs in a blog, and a preprint
  with n=1 per size tier.
- Whether an affine map preserves **causal** effect for a **style/syntax** direction in a
  **code** model. Published causal successes are language-ID and refusal; published code
  results are probing-only ("code and language domains again seem hard to directly transfer"
  — 2506.06609).
- Whether our **L8 / 29%-depth** band is transferable at all. Mid-depth is the least aligned
  band in the only multi-architecture study, but nobody has run a random-vector-controlled
  steering test there.
- Whether a 1.5B target sits above or below the claimed ~1.7B capacity threshold
  (2608.05164 `[preprint]`, n=1 per tier — treat the threshold as a hypothesis, not a fact).
- Whether the owner's loop, redefined as *input-space emulation*, ever beats a plain prompt
  on a modern text-only model. Nobody has published that comparison; VISOR++ only shows
  directional consistency on VLMs.

---

## Sources (all fetched unless marked)

| cited as | title | year | link | status |
|---|---|---|---|---|
| Huh et al. | The Platonic Representation Hypothesis | 2024 | arXiv:2405.07987 | peer-reviewed (ICML) |
| Gröger et al. | Revisiting the PRH: An Aristotelian View | 2026 | arXiv:2602.14486 | preprint |
| Köpke et al. | Back into Plato's Cave | 2026 | arXiv:2604.18572 | preprint |
| Chen et al. | Transferring Linear Features Across LMs With Model Stitching | 2025 | arXiv:2506.06609 | NeurIPS 2025 |
| Chen et al. | Linear Representation Transferability Hypothesis | 2025 | arXiv:2506.00653 | preprint |
| Sinii et al. | Small Vectors, Big Effects | 2025 | arXiv:2509.06608 | preprint |
| Stolfo et al. | Improving Instruction-Following via Activation Steering | 2024 | arXiv:2410.12877 | peer-reviewed |
| Oozeer et al. | Activation Space Interventions Can Be Transferred Between LLMs | 2025 | arXiv:2503.04429 | ICML 2025 |
| Arditi et al. | Refusal in LMs Is Mediated by a Single Direction | 2024 | arXiv:2406.11717 | NeurIPS 2024 |
| Wang et al. | Refusal Direction is Universal Across Safety-Aligned Languages | 2025 | arXiv:2505.17306 | NeurIPS 2025 |
| Tan et al. | Analyzing the Generalization and Reliability of Steering Vectors | 2024 | arXiv:2407.12404 | NeurIPS 2024 |
| Wu et al. | AxBench | 2025 | arXiv:2501.17148 | peer-reviewed |
| Subramani et al. | Extracting Latent Steering Vectors from Pretrained LMs | 2022 | arXiv:2205.05124 | peer-reviewed |
| Roeder et al. | On Linear Identifiability of Learned Representations | 2021 | arXiv:2007.00810 | ICML 2021 |
| Kaba & Ravanbakhsh | Affine symmetries and neural network identifiability | 2020 | arXiv:2006.11727 | peer-reviewed |
| Entezari et al. | The Role of Permutation Invariance in LMC | 2021 | arXiv:2209.04836 | ICLR 2022 |
| Carlini et al. | Stealing Part of a Production Language Model | 2024 | arXiv:2403.06634 | peer-reviewed |
| Marx et al. | Predictive Multiplicity in Classification | 2020 | arXiv:2002.06694 | ICML 2020 |
| — | On the Non-Identifiability of Steering Vectors | 2026 | arXiv:2602.06801 | preprint |
| — | Architecture-Dependent Causal Transfer of Activation States | 2026 | arXiv:2608.16347 | preprint |
| — | Cross-Architecture Steering Transfer (systematic) | 2026 | arXiv:2608.05164 | preprint, n=1/tier |
| — | Five Architectures, One Shared Space | 2026 | arXiv:2610.33495 | preprint |
| — | Cross-Lingual Alignment Without Joint Training | 2026 | arXiv:2608.27115 | preprint |
| — | Characterizing Linear Alignment Across LMs | 2026 | arXiv:2603.18908 | preprint |
| — | The Invisible Editorial Layer (Prop. 1) | 2026 | arXiv:2608.24662 | preprint |
| — | Transmuting prompts into weights | 2025 | arXiv:2510.08734 | preprint |
| — | VISOR++: universal visual input steering | 2025 | arXiv:2509.25533 | preprint |
| — | A Negative Result on Cross-Model Activation Transfer | 2026 | arXiv:2606.03280 | preprint |
| — | Cross-Model Transfer of Task Vectors via Few-Shot Orthogonal Alignment | 2025 | arXiv:2505.12021 | preprint |
| — | Inverse Scaling in Activation Steering | 2026 | sohailmo.ai/research/activation-steering/ | blog, not peer-reviewed |
| — | You can't steer what you can't open | 2026 | worldagrees.com | blog |

---

# Live API constraints (2026-11-11)

This section **supersedes the API-side assumptions** in F2/F5 and in §7's option table.
Nothing here changes the **local** analysis: T1/T2 still run on local weights, where we have
logits and activations. What changes is that the *closed-model leg* now has a measured
interface, and that interface is narrower than the brief assumed in one place and richer in
another.

## 0. What I verified myself — and one correction to the brief

Key handling, for the record: sourced from `.env` (`mode 600`, present in `.gitignore`);
never printed, never written outside `.env`, no shell history containing it. The directory
is **not a git repository** (`git rev-parse` → exit 128), so there is no commit surface at
all.

Measured on 2 sessions, **≈17 generate/embed calls total, zero quota errors**. I did not
stress the quota and will not; the numbers below are sized so the reader can scale. **The
`~15 RPM` used in the wall-clock column is an assumption, not a measurement** — measure the
real rate with 20 timed calls before committing to any plan that depends on it.

| check | observed | source |
|---|---|---|
| models reachable | **62** listed for this key | `GET /v1beta/models` |
| `gemini-2.5-flash` exists | **yes** — `generateContent` routed, returned `400 INVALID_ARGUMENT: "Logprobs is not enabled for models/gemini-2.5-flash"` | live |
| logprobs on 3-flash | `responseLogprobs:true` → error (owner measured; consistent) | live |
| embeddings | `gemini-embedding-2` → **3072 dims, L2 norm = 1.0000**, `taskType` accepted, `batchEmbedContents` batches 6 texts in **one** call | live |
| model list includes | `gemini-embedding-001`, `-2`, `-2-preview`, `gemini-3.8-flash`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it` | live |

**Correction to the brief (does not change the conclusion).** "The 2.5 family is 404" is not
what I see: `gemini-2.5-flash` is listed and routes. It returns a **400 for logprobs
specifically**, not a 404 for the model. So the family is reachable and simply has logprobs
disabled, like everything else. The practical conclusion is unchanged — **there is no
logprob channel on any reachable model** — but the orchestrator should not exclude
2.5-flash from behavioural sampling on the belief that it is gone.

**A trap that will silently poison any sampling loop.** `gemini-3-flash-preview` is a
*thinking* model. My first 6-sample run returned **0 usable samples**: every call came back
`finishReason: MAX_TOKENS` with `thoughtsTokenCount: 36` and an **empty** `content`, because
the thinking budget consumed `maxOutputTokens=40` before a single visible token was emitted.
Fix: `generationConfig.thinkingConfig.thinkingBudget = 0` (verified working), or
`maxOutputTokens ≥ 256`. Anyone who writes a sampling loop without this gets a distribution
of empty strings and concludes the model is broken. `[OBS]`

**Measured sampling noise floor** (temperature 1.0, 6 samples/prompt, embedded with
`gemini-embedding-2`), because every arithmetic below depends on it:

| prompt | distinct / 6 | 1 − cos(eᵢ, mean embedding) | pairwise 1 − cos |
|---|---|---|---|
| `"Write one line of Python that prints hello."` | **5 of 6 identical** | ≈ 0.026 (pairwise cos 0.9742 ± 0.0364) | — |
| `"Suggest a name for a coffee shop…"` | **6 of 6 distinct** | **0.2255 ± 0.0684** | mean 0.4802, max 0.6839 |

`[OBS]` Read this as: a low-entropy prompt collapses to a point mass, so **all** text-only
statistics lose power; a high-entropy prompt gives a usable continuous signal with
σ ≈ 0.068 per sample. Use the high-entropy number for design, never the low one.

---

## 1. What is the strongest remaining transfer primitive with text-only access?

Ranked by information per query. All three are *behavioural*; none is mechanistic.

**P1 — Two-sample distribution test (MMD with a Hamming string kernel).**
`[OBS]` Gao, Liang, Guestrin, *Model Equality Testing: Which Model Is This API Serving?*,
arXiv:2410.20247, **ICLR 2025**: a Hamming-kernel MMD test achieves a **median 77.4% power**
against quantisation / watermarking / fine-tuning / model-swap distortions, using **10
samples per prompt over 25 prompts** (250 completions total), with a permutation test for
the p-value. Extended in arXiv:2504.04715 (L=50, T=1, 25 prompts, 1000 permutations).
This is the **cheapest and most defensible** text-only primitive we have.
**But note what it returns: a p-value for "these two distributions differ". It has no
direction and no sign.** It can tell you the API is not your local model. It cannot tell you
*which way* it moved.

**P2 — Project the hypothesis onto one scalar.** Take the direction you care about
("emits a comment line"), define a binary judge, and estimate the rate. This is a
one-dimensional functional of the output distribution — the only thing free-tier quota can
afford. See §2 for the sample arithmetic.

**P3 — Embedding-mean shift.** Embed every completion with `gemini-embedding-2` (3072-d,
unit norm, batched) and test the mean against a reference mean. This turns P2's Bernoulli
into a continuous statistic with much better power per sample. See §3.

**P4 — Prompt-level "direction search".** Optimise a system instruction to move the P2/P3
statistic. This *is* the owner's backward loop once the API is text-only: hypothesise a
direction locally → compile it to words → measure. It works, in the sense that prompt
optimisation works. It is also indistinguishable from prompting (see §4).

**The structural limit, stated once.** Text-only access gives samples from
`P(y | x)` — a marginal over the entire computation. A direction is a property of the
*conditional computation* that produces `P`. Sampling lets you test hypotheses about the
marginal. It does not let you read or write the conditional. This is the same result as F5,
now with a live instance: `[OBS]` arXiv:2608.24662 Prop. 1 — for any target Q with
supp(Q) ⊆ supp(P_θ) there is a logit policy I with P_{θ,I} = Q, so "base model + steering"
and "different model" are observationally identical. **Text-only access removes even the
logit policy from the observable set, which makes attribution strictly harder, not easier.**

---

## 2. Is sample-based distribution matching a viable substitute for logprobs?

It depends which of three questions you are asking. Conflating them is the main error in
this area.

### (a) "Do these two distributions differ?" — **YES, and cheaply.**
MMD-Hamming at 10 samples/prompt (P1). 250 completions for a 77.4%-power test.
This is the one place where sampling genuinely substitutes for logprobs.

### (b) "What is the rate of behaviour B?" — **YES, at a quantified price.**

Binomial CI, worst case p = 0.5: `n = z²·p(1−p)/ε² = 3.8416 × 0.25 / ε² = 0.9604/ε²`.

| target precision ε | samples per condition | at ~15 RPM |
|---|---|---|
| ±0.10 | **97** | 6.5 min |
| ±0.05 | **385** | 26 min |
| ±0.02 | **2,401** | 2.7 h |
| ±0.01 | **9,604** | 10.7 h |

Two-proportion comparison, α=0.05 two-sided, power 0.8:
`n/group = (1.96+0.84)²·[p₁q₁+p₂q₂]/δ² = 7.849·[p₁q₁+p₂q₂]/δ²`.

- 0.50 vs 0.40 (δ=0.10): `7.849 × (0.25+0.24) / 0.01 =` **385 per group**
- 0.50 vs 0.30 (δ=0.20): `7.849 × (0.25+0.21) / 0.04 =` **91 per group**
- 0.50 vs 0.45 (δ=0.05): `7.849 × (0.4975) / 0.0025 =` **1,562 per group**

`[INF]` **Design consequence:** free-tier quota can resolve a **≥10 percentage-point**
behavioural shift on one prompt in under an hour. It cannot resolve a 5 pp shift at all.
Since a steering-scale effect is typically *smaller* than a prompt's effect (AxBench:
diff-in-means 0.409 vs prompt 0.905), the affordable window is exactly the window in which
steering would already have lost to prompting.

### (c) "What is the token-level distribution?" — **NO, at any sane cost.**

The vocabulary is `V = 151,936` for Qwen, `50,257` for GPT-2. To see each bin once at
15 RPM: `151,936 / 15 = 10,129 min =` **169 hours**, for **one token position of one
prompt**. With add-1 smoothing you need `n ≫ V` for a non-degenerate estimate. Even the
top-1 bin alone, at p≈0.3 and ε=0.05, needs `0.9604 × 0.21 / 0.0025 =` 81 samples *once you
already know which bin it is* — and you don't.

Three further losses, all one-way:
1. **Top-p truncation.** Default sampling truncates the tail, so low-probability tokens are
   structurally unobservable. You never recover the native `P`, only a filtered version.
2. **Sampler unknown.** Temperature/top-p are part of the served policy, not the model.
   A change in defaults mimics a change in the model.
3. **Greedy decoding collapses the distribution to a point mass.** Measured above: 5 of 6
   completions identical on a low-entropy prompt.

**Verbalised probabilities are not a substitute, but they are not worthless either.**
`[OBS]` Tian et al., *Just Ask for Calibration*, arXiv:2305.14975: for RLHF models,
confidence emitted as output tokens is **better calibrated than the model's conditional
probabilities** on TriviaQA/SciQ/TruthfulQA, "often reducing expected calibration error by a
relative" large margin. `[INF]` So asking for "probability this is a comment line" is a
legitimate *cheap scalar*, but it is a self-report about an event, not `P(token)`, and it
carries no distributional information at all.

**Verdict on §2:** sampling substitutes for logprobs for **detection (a)** and **scalar
rates (b)**; it cannot substitute for **token-level distributions (c)**. Any design that
needs (c) must move to a local model.

---

## 3. Does the embedding endpoint give any leverage?

**What it is, measured:** a *separate* embedding model (`gemini-embedding-2`), 3072-d,
L2-normalised, batchable, with `taskType` support. It is not the generator's residual stream
and there is no interface that exposes one.

**What it genuinely buys us — three things.**

1. **A model-agnostic semantic ruler with a measured noise floor.** With σ ≈ 0.068 per
   sample (high-entropy prompt), detecting a mean-embedding shift `d` costs
   `n = (1.96σ/d)² = (0.1341/d)²`:

   | shift d (1−cos) | samples per condition |
   |---|---|
   | 0.050 | **8** |
   | 0.020 | **45** |
   | 0.010 | **180** |
   | 0.005 | **719** |

   Compare with (b) above: an embedding shift is far cheaper per unit of resolution than a
   Bernoulli, because it uses the whole 3072-d vector instead of one bit. (The two are not
   commensurate units — 45 samples buys a 0.02 cosine shift; a ±0.02 Bernoulli rate costs
   2,401 — so read the table as a cost curve, not a conversion.)

   *Caveat on the σ itself:* 0.0684 is the sd of six samples, so it is itself uncertain by
   roughly ±30%. Re-measure with n ≥ 20 before using these numbers to pre-register a
   criterion.

2. **Anchor-based prompt geometry.** `[OBS]` Moschella et al., *Relative representations
   enable zero-shot latent space communication*, arXiv:2209.15430: representing a point by
   its similarities to a fixed set of anchors is invariant to latent isometries and
   rescalings, "effectively enabling latent space communication: from zero-shot model
   stitching to latent space comparison". `[INF]` This is the correct way to compare *our*
   prompt set against the API's *outputs* without fitting anything: embed both, compare
   anchor coordinates. It is a comparison tool, not a steering tool.

3. **Concept-anchor prompt search.** Find the prompt whose embedding is closest to a concept
   anchor, then test it. Cheap, and strictly better than guessing. This is P4 with a
   principled search space.

**What it does not buy — and this is the important part.**

- **No map into the generator.** Every published embedding→steering link requires the
  *generator's own output embedding matrix*. `[OBS]` *Word Embeddings Are Steers for
  Language Models* (arXiv:2305.12798, ACL 2024): linear transformations of output word
  embeddings are equivalent to style steering ("LM-Steer", 0.2% of model parameters), and
  an LM-Steer "is transferrable between different language models by an explicit form
  calculation" — but that calculation is written in terms of **both models' embedding
  matrices**. `[INF]` A closed model has no accessible `E`, so the formula has no inputs.
- **Cross-tokeniser embedding transfer is a known, solved-looking problem — for a different
  task.** `[OBS]` WECHSEL (arXiv:2112.06598, NAACL 2022) maps subword embedding spaces
  across **different tokenisers** using overlapping vocabulary and a linear transform, and
  it works — as an *initialisation* for continued training. It needs both embedding
  matrices and it produces a trainable model, not a direction we can inject anywhere.
- **Embeddings are invertible.** `[OBS]` Morris et al., *Text Embeddings Reveal (Almost) As
  Much As Text*, arXiv:2310.06816. So the embedding channel leaks **text**, which is an
  attack on a retrieval store, not a window into a model.

**Verdict on §3:** the embedding endpoint upgrades our *measurement* (P3 is cheaper per unit
of resolution than a Bernoulli, and needs no reference distribution to be pre-specified) and
gives us a defensible way to compare prompt sets. It gives **zero** new control over the
generator. Do not let "we have embeddings" be mistaken for "we have activations".

---

## 4. Revised cheapest falsification, text-only

The old §7 T1/T2 used a **logit-level readout** — sound locally, impossible on the API.
Do not replace them: T1/T2 remain the mechanistic tests and unchanged. What is new is a
**third test that fits the actual interface**, and it is designed to fail informatively.

### T3 — Can the text channel carry a direction at all? (~360 calls, ~25 min)

Three conditions, one behaviour (comment-line emission, our existing L8 target), three
high-entropy code prompts, n = 30 samples per cell at temperature 1.0 with
`thinkingBudget = 0`:

| arm | system instruction | what it controls for |
|---|---|---|
| A | none | baseline rate |
| B | *"Always include at least one explanatory comment."* | **positive control** — the ceiling any text channel can reach |
| C | a prose encoding of our L8 comment direction (generated offline from the local model's top logit-lens tokens) | the hypothesis |
| D | a nonsense instruction of matched length and register | **specificity control** (this is L1 again) |

- **Primary readout:** rate of completions containing a comment line, with a permutation
  test on the pooled 4×90 samples at α = 0.05.
- **Secondary readout:** mean-embedding shift vs arm A. Note the §3 table is a *one-sample
  vs known reference* cost; a two-arm comparison doubles it, so **n = 30 per arm resolves
  d ≈ 0.035** (1.96 × 0.0684 × √(2/30)).
- **Pre-registered pass criterion:** `rate(C) − rate(D) ≥` the T3a noise floor, and
  `rate(C)` within 50% of `rate(B)`.
- **Pre-registered prediction (stated before the run):** C will fail against D, *or* it will
  pass only by being read as an instruction — in which case C ≈ B and we have measured
  prompting. Either outcome kills the mechanistic reading.

**Why not just do P1 (MMD) instead?** Because P1 answers "is the API my local model", which
nobody is asking. T3 answers the owner's actual question, and its failure mode is
interpretable.

### Does the belief survive?

- As *"a text channel can move a closed model's behaviour"* — **yes, trivially.** That is
  prompting, and T3 will confirm it in arm B.
- As *"a direction can cross the boundary"* — **no, and not because of a measurement
  failure.** There is no experiment on a text-only interface that distinguishes "the prompt
  encoded the direction" from "the prompt said words that mean the thing". That is exactly
  the observational non-identifiability result, and no amount of sampling fixes it. The
  belief is not refuted; it is **unfalsifiable** on this interface, which by `VALIDATION.md`
  rule 6 means it cannot enter `LEDGER.md`.

---

## 5. Revised three-way split (API-reachable world)

### POSSIBLE — measured or arithmetically costed

- **Detect** that an API's output distribution differs from a reference: MMD-Hamming,
  10 samples/prompt over 25 prompts, 77.4% median power (Gao et al., ICLR 2025).
- **Estimate a behavioural rate** to ±10 pp with 97 samples, ±5 pp with 385, ±2 pp with
  2,401 (binomial CI, arithmetic shown in §2).
- **Detect a mean-embedding shift** of 0.05 / 0.02 / 0.01 (1−cos) with 8 / 45 / 180 samples
  per condition (measured σ = 0.068).
- **Compare prompt sets across models** via anchor/relative representations (arXiv:2209.15430)
  without fitting anything.
- **Elicit a calibrated-ish scalar** by asking for a verbalised probability (arXiv:2305.14975).
- **Build a local surrogate** by distilling API outputs, then do real mechanistic work on the
  surrogate. This is the only route from the API back to a direction, and it ends at the
  surrogate.
- **Everything in the original POSSIBLE list**, unchanged, for models we can load.

### IMPOSSIBLE — proof, definition, or measured interface limit

- **Any token-level logprob.** Measured: `"Logprobs is not enabled"` on `gemini-2.5-flash`
  and `gemini-3-flash-preview`; the owner measured the same on `gemma-4-31b-it`,
  `gemma-4-26b-a4b-it`, `gemini-flash-latest`. There is no model on this key with logprobs.
- **Recovering a residual-stream direction from text samples.** Sampling observes a marginal
  over the computation; the direction lives in the conditional (arXiv:2608.24662 Prop. 1).
- **Mapping an external embedding into the generator's hidden state.** Requires the
  generator's `E` or its activations (arXiv:2305.12798's transfer formula is written in `E`).
- **Attributing a behavioural change to a "direction"** rather than to the semantics of the
  words used to encode it.
- **Detecting effects below ~5 pp** within free-tier quota.
- **Estimating a token-level distribution**: 169 h per token position just to visit each bin
  once at 15 RPM.

### UNKNOWN — cheap to test, not yet tested

- **Can constrained decoding manufacture a logprob substitute?** Force the model to emit one
  of a small label set (say 20 options, one token each) and use the **frequency over n
  samples** as a 20-way categorical. At n = 385 that is ±5 pp per bin — the closest thing to
  a distribution the interface allows. **No one in this project has tried it**, and it is the
  single highest-value API experiment available. It also sidesteps the token-level problem
  entirely by collapsing the vocabulary to a designed alphabet.
- **Is the embedding-mean statistic sensitive enough to see a steering-scale effect?** The
  measured noise floor is known; the effect size is not.
- **Does a prose encoding of a local direction beat a length-matched nonsense control (T3)?**
- **Is `gemini-2.5-flash` materially cheaper in quota than `3-flash-preview`**, which would
  raise the affordable effect size? The 400-vs-404 correction above suggests it is at least
  reachable.
- **Whether `thinkingBudget=0` changes the served distribution** in ways that matter for
  sampling. It changes latency and token accounting; its effect on output diversity is
  unmeasured.

---

## 6. What this changes in the document above

| section | change |
|---|---|
| TL;DR 7 / §7 decision rule | **unchanged for local models.** T1/T2 still run first; they are still the cheapest real test. |
| F5(a) black-box logprob regime | **narrowed to zero**: for this key there is no logprob regime, only a text regime. The identifiability argument gets *stronger*, because the observable set shrank. |
| §7 option 5 ("prompt/logit-bias emulation") | logit bias is **not available** either; the only input channel is text. Option 5 is now strictly "prompting", and T3 is its falsifier. |
| §7 option 3 ("distil API → local surrogate") | **promoted**: it is now the only route from the API to anything mechanistic, and it terminates at the surrogate by construction. |
| New option | **T3 + constrained-label decoding** (§5 UNKNOWN) — the only untried experiment that could make the text channel behave like a distribution channel. |

---

## Quota + memory constraints

**Hard stop recorded.** At 2026-11-11 I attempted the rescoped T3 pilot (2 arms × 2 prompts ×
8 samples = 32 generate calls, + 1 batched embed). **Call 1 returned `429 RESOURCE_EXHAUSTED`
("You exceeded your current quota")**. Per the standing rule I did **not** retry, did not
back off-and-resend, and ran **0 of 32** cells. The pilot is **not run**, not "failed".
`[OBS]`

**Memory rule adopted.** No local model was loaded at any point in this round; every
measurement in this appendix came from the API. T1/T2 in §7 (which need
`Qwen2.5-Coder-0.5B` + `1.5B` resident) are **deferred** and must not be started on the
owner's laptop until the zram + 2 QEMU VM situation (~4.3 GB) is resolved. This does not
change their design; it changes when they can run.

**T3, rescoped and re-specified.**

| property | value |
|---|---|
| design | 2 arms: `C` = prose encoding of the L8 comment direction, `D` = length- and register-matched nonsense paragraph (gardens, not code) |
| cells | 2 prompts × 2 arms × 8 samples = **32** generate calls + 1 `batchEmbedContents` |
| readout | (i) our existing 6-label enum channel — the one the orchestrator validated (8/10 "Paris") — not free-text + regex, which is degenerate at n=8; (ii) mean-embedding shift, σ ≈ 0.068 |
| setting | `temperature 1.0`, `thinkingBudget 0`, `maxOutputTokens 120` |
| cost | 33 calls, ~2.2 min at 15 RPM |

**Pre-registered power statement — read this before interpreting any result.** With n = 8
per arm the binomial 95% half-width is `1.96·√(0.25/8) = 0.347`. Fisher's exact on
8 vs 8 reaches α = 0.05 only at a **near-total separation: ≈5/8 vs 0/8 (two-sided p = 0.026)**;
2/8 vs 6/8 gives p = 0.47 and is indistinguishable from noise. Therefore:

- **A null pilot is uninformative about the hypothesis.** It must be logged as
  "no evidence at this n", **never** as "the direction does not transfer".
- **A separated pilot is the only outcome that licenses scaling up** — and it would license
  *more sampling*, not a mechanistic claim, because a prose instruction that moves behaviour
  is still prompting until shown otherwise.
- Consequence: at this quota the owner's belief remains **unfalsifiable**, exactly as §4
  argued. The pilot's real value is pipeline validation (does the enum channel classify
  reliably at 6 labels; does the embedding statistic move at all) — not hypothesis testing.

**Build on the orchestrator's result, don't repeat it.** Constrained decoding to a 6-label
enum is already shown to work as a coarse categorical channel (8/10 modal label, 2 quota
errors). T3 therefore uses that channel as its primary readout and adds **no** new
constrained-decoding calibration calls. The open question is unchanged: a 6-way categorical
at n = 8 gives ±0.35 per bin, so it is a *pipeline* probe until quota allows n ≥ 100 per
cell, at which point it becomes a genuine distributional instrument.

**Resume condition.** T3 can start when a single call succeeds without 429. When it does,
run the 33 calls in one pass, stop on the first quota error, and log the partial cell count —
partial cells are still usable for pipeline validation as long as arm and prompt coverage is
balanced.


