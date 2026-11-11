# A6 — Limits, Illusions, and Validation (the skeptic)

**Role:** what is impossible here, how we avoid fooling ourselves, how we score whether we
actually understand or control the model.
**Stance:** every claim in this project is guilty until an intervention proves it. My job is to
make self-deception *mechanically detectable*, not to argue that the project is impossible.
**Bottom line:** full neuron-level transparency is impossible in principle. *Operational causal
control* is achievable, cheap, and is the only claim worth shipping.

**Citation hygiene note — four titles in the brief are slightly wrong** (corrected here, and the
correct titles are used throughout):

| brief says | actual paper |
|---|---|
| "Interpretability Illusions in the Wild" (2021) | **"An Interpretability Illusion for BERT"**, Bolukbasi et al., 2021, [arXiv:2104.07143](https://arxiv.org/abs/2104.07143) |
| "Sparse Autoencoders Do Not Find Canonical Units of Attribution" | **"…Canonical Units of *Analysis*"**, Leask et al., 2025, [arXiv:2502.04878](https://arxiv.org/abs/2502.04878) |
| "Not All Language Model Features Are Linear" | **"…Are *One-Dimensionally* Linear"**, Engels et al., 2024, [arXiv:2405.14860](https://arxiv.org/abs/2405.14860) |
| "Activation Addition: Steering LMs Without Optimization" | published as **"Steering Language Models With Activation Engineering"**, Turner et al., 2023, [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) (ActAdd is the *method*) |

Every citation below was resolved against arXiv/OpenAlex/transformer-circuits and the abstract
text was read before use. No paper is cited from memory alone.

---

## TL;DR (5 bullets)

- **"One neuron = one concept" is not merely hard, it is *wrong as a model of what the network
  is doing*.** Superposition is the loss-optimal solution when features are sparse, so
  polysemanticity is a *predicted consequence of the objective*, not a training bug. Full
  neuron-level transparency is therefore unattainable at any scale, including 1.5B. (F1)
- **Decoding is not causation, and even causation is not attribution.** A probe that reads X, a
  logit-lens peak, an attention map, or an SAE feature are all *your decoder's* artifacts until an
  interchange intervention confirms them — and subspace patching can produce a *causal* effect
  through a dormant parallel pathway, i.e. a true intervention with a false mechanism story.
  (F5, F6)
- **We can make the microscope self-auditing.** 12 sanity checks + a *golden model* (a 2-layer
  induction circuit we train ourselves, whose answer we already know) + a per-claim JSON
  "certificate" that the UI is forbidden to render without. (F9, and §"Concrete implementation")
- **Concrete pass/fail thresholds** (pre-registered): patch recovery ≥ 0.20 with 95% CI excluding
  0 and consistent sign in ≥ 70% of pairs; dose–response Spearman ρ ≥ 0.80; test/train effect
  ratio ≥ 0.50; every negative control ≤ 0.10× the real effect; ΔPPL ≤ 10%; n = 100 pairs, which
  detects only d_z ≥ 0.42 after Bonferroni over 56 layer×direction tests. Smaller effects are
  **not claimable** at this budget. (F10)
- **The tool is a jailbreak primitive.** Refusal is mediated by a single direction in 13 open
  chat models, and erasing it works. Ship local-only, no precomputed harmful vectors, an
  append-only audit log, and a base-model allowlist. Note that Qwen2.5-Coder-1.5B-**Base** has no
  refusal behaviour, so any "we disabled safety" demo on *this* model is theatre. (F12)

---

## Findings

### F1. "One neuron = one concept" is structurally false; superposition is the *optimal* solution, not an accident
**[source: "Toy Models of Superposition", Elhage et al., 2022, [transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html) · [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)]**

- **What it is.** A minimal model (m hidden dims, n > m sparse features) trained on
  reconstruction. When features are sparse, the loss-optimal solution stores *more features than
  there are dimensions*, as near-orthogonal **directions** at ~72° spacing rather than as axes.
  The paper shows a phase change as sparsity varies, a link to the geometry of uniform polytopes,
  and evidence tying superposition to adversarial examples.
- **Why it matters.** This kills the project's most tempting demo: colouring neurons by "their
  concept". A neuron is a *scalar projection*; a concept is a *direction*. There is no canonical
  basis, so "neuron 4211 = 'snake_case'" is a statement about a coordinate system we chose, not
  about the model. Any UI that implies otherwise is a lie told with a heatmap.
- **Quantitative intuition for our model.** Hidden = 1536, vocab = 151936. The residual stream is
  read by a 151936×1536 unembedding (Qwen2.5-1.5B may tie `lm_head` to `embed_tokens` — verify
  with `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` at load; the argument holds
  either way). 151936 token directions cannot be mutually orthogonal in 1536 dims — a 99×
  overcompleteness. The *output layer alone* forces heavy superposition; layers 0–27 are worse.
- **Confidence:** very high. Replicated, mechanistically derived, and consistent with every
  downstream SAE result.
- **Cost:** zero to accept; *high* if ignored (we'd build a neuron-colouring UI that is wrong by
  construction).

### F2. Polysemanticity is a *capacity-allocation* consequence of feature importance and sparsity
**[source: "Polysemanticity and Capacity in Neural Networks", Scherlis, Sachan, Jermyn, Benton et al., 2022, [arXiv:2210.01892](https://arxiv.org/abs/2210.01892)]**

- **What it is.** Introduces *feature capacity* — the fractional dimension a feature consumes. In
  a toy model, optimal allocation is: **most important features → monosemantic (own dimension);
  less important → polysemantic, in proportion to loss impact; least important → dropped
  entirely.** Polysemanticity rises with feature sparsity (k).
- **Why it matters.** It predicts *which* neurons will be uninterpretable: the ones representing
  low-importance, sparse features. So a "this neuron is a mess" finding is not a mystery — it is
  the objective function working as designed. It also predicts that rare code tokens/idioms
  (low importance, high sparsity) will be the most polysemantic, which is exactly where a
  code-model demo will look worst and be most tempting to over-interpret.
- **Confidence:** high (toy model, but the prediction matches real-model SAE behaviour).
- **Cost:** zero; it sharpens the claim we *can* make ("important features tend to be cleaner").

### F3. Memorization is also superposition — so "feature" vs "memorized example" is not visible from one activation
**[source: "Superposition, Memorization, and Double Descent", Henighan et al., 2023, [transformer-circuits.pub/2023/toy-double-descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html)]**

- **What it is.** Extends the toy model to a training set. Memorized datapoints are stored as
  *additional, extremely sparse features*. When the number of features (general + memorized)
  crosses the dimension threshold, test loss shows **double descent**. Memorization and
  generalization are the same mechanism at different sparsity.
- **Why it matters.** A 1.5B code model has seen a lot of GitHub. Some "features" we recover will
  be *specific training strings*, not concepts. There is no activation-level test that separates
  them. The only separator is behavioural: does the direction generalise to held-out,
  differently-templated prompts? (This is why the held-out split in F10 §4f is not optional.)
- **Confidence:** high for the toy result; medium for the extrapolation to 1.5B.
- **Cost:** forces a held-out-template design in every eval. Cheap, and it is the single most
  useful control we have.

### F4. In real models the dictionary itself is non-canonical, incomplete, and not atomic
**[sources: "Towards Monosemanticity", Bricken et al., 2023, [link](https://transformer-circuits.pub/2023/monosemantic-features/index.html) · "Scaling Monosemanticity", Templeton et al., 2024, [link](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) · "Sparse Autoencoders Do Not Find Canonical Units of Analysis", Leask et al., 2025, [arXiv:2502.04878](https://arxiv.org/abs/2502.04878) · "A is for Absorption", Chanin et al., 2025, [arXiv:2409.14507](https://arxiv.org/abs/2409.14507) · "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers", 2025, [arXiv:2501.17727](https://arxiv.org/abs/2501.17727) · "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing", Gupta et al., 2025, [arXiv:2502.16681](https://arxiv.org/abs/2502.16681)]**

- **What it is.**
  - Anthropic found ~4000 features in a **512-neuron** layer (8× expansion) and tens of millions
    in Claude 3 Sonnet — but also **feature splitting**: a feature at width 4k becomes a family at
    width 64k. The *count of features is a function of dictionary width*, not a property of the model.
  - **Leask et al. 2025**: SAEs are *incomplete* (novel latents from a bigger SAE add information a
    smaller one lacks) and *not atomic* (meta-SAEs show latents are composed of others). No
    canonical unit set exists.
  - **Chanin et al. 2025**: **feature absorption** — a general feature silently fails to fire
    because its children absorb it. Caused by optimizing sparsity over a hierarchy; *not* fixed by
    changing width or sparsity. Measured on hundreds of LLM SAEs.
  - **2501.17727 (2025)**: SAEs trained on **randomly initialized** transformers score
    *similarly* on auto-interpretability and reconstruction metrics to SAEs on trained models.
  - **Gupta et al. 2025**: SAE latents do **not** consistently beat simple baselines on downstream
    probing (data scarcity, class imbalance, label noise, covariate shift).
- **Why it matters.** This is the strongest evidence that "name the feature" is a *dictionary
  artifact*. It also means: **if we ship an SAE panel, we must ship the random-model baseline
  next to it.** Without that number, the panel is unfalsifiable decoration.
- **Confidence:** very high (multiple independent groups, 2024–2025).
- **Cost:** SAE training on 8 CPU cores is the single biggest budget risk in this project. See
  [Trap 4](#what-not-to-do-traps).

### F5. Every correlational method we plan to ship has a documented failure mode
**[sources: "An Interpretability Illusion for BERT", Bolukbasi et al., 2021, [arXiv:2104.07143](https://arxiv.org/abs/2104.07143) · "Attention is not Explanation", Jain & Wallace, 2019, [arXiv:1902.10186](https://arxiv.org/abs/1902.10186) · "Attention is not not Explanation", Wiegreffe & Pinter, 2019, [arXiv:1908.04626](https://arxiv.org/abs/1908.04626) · "Is Attention Interpretable?", Serrano & Smith, 2019, [arXiv:1906.03731](https://arxiv.org/abs/1906.03731) · "Sanity Checks for Saliency Maps", Adebayo et al., 2018, [arXiv:1810.03292](https://arxiv.org/abs/1810.03292) · "A Benchmark for Interpretability Methods in DNNs" (ROAR), Hooker et al., 2019, [arXiv:1806.10758](https://arxiv.org/abs/1806.10758) · "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead", Rudin, 2019, [arXiv:1811.10154](https://arxiv.org/abs/1811.10154) · "The Road to Explainability is Paved with Bias", Balagopalan et al., 2022, [arXiv:2205.03295](https://arxiv.org/abs/2205.03295)]**

- **What it is, per method:**
  - **Neuron/linear-combination "concept" stories** — Bolukbasi et al. show individual BERT neurons
    can *spuriously* look monosemantic. Cause: geometry of the embedding space + corpora that
    represent only a narrow slice of English. Fix: test on multiple datasets. (This is exactly the
    trap for a code model evaluated on 20 Python snippets.)
  - **Attention weights as explanations** — Jain & Wallace: attention distributions can be
    permuted/adversarially perturbed with little change in prediction, so they do not explain.
    Wiegreffe & Pinter rebut that attention *can* be an explanation under proper testing, but the
    default claim is unsupported. Serrano & Smith: attention often disagrees with gradient-based
    importance.
  - **Saliency / attribution maps** — Adebayo et al.: some methods are *independent of the model
    and the data* (they pass model-parameter-randomization and data-randomization tests unchanged).
    ROAR: under remove-and-retrain, many popular attribution methods are **no better than random**
    feature importance.
  - **Post-hoc explanation as a category** — Rudin argues that for high-stakes decisions, post-hoc
    explanations of black boxes are not trustworthy and one should use an inherently interpretable
    model instead. The tell she highlights: two models with near-identical accuracy can produce
    *different* explanations, so the explanation is a property of the model-plus-method pair, not
    of the computation. **This applies to us completely**: a 1.5B transformer is not inherently
    interpretable, so *every* explanation this tool emits is post-hoc and therefore an argument,
    not a proof. Balagopalan et al. add the sharpest version: explanation *quality* itself varies
    across subgroups, so "the explanation is faithful" is not a model-level property — see F8.
- **Why it matters.** Three of the five things in `CONTEXT.md`'s "CAN" list (logit lens, attention
  patterns, attribution) are correlational. Bolukbasi's "test on multiple datasets" and Adebayo's
  "randomize the model and see if the explanation survives" are the two cheapest, highest-value
  controls we can adopt *today*.
- **Confidence:** very high.
- **Cost:** ~1 day to implement the randomization controls; saves the project from its worst
  failure mode.

### F6. Even a *successful intervention* can misattribute: the subspace-patching illusion
**[source: "Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching", Makelov, Lange, Nanda, 2023, [arXiv:2311.17030](https://arxiv.org/abs/2311.17030)]**

- **What it is.** Subspace activation patching is used to *simultaneously* (a) change behaviour and
  (b) attribute that behaviour to a feature. The paper shows these aims **diverge**: a subspace
  intervention can make the output behave *as if* the feature changed, while the actual effect
  flows through a **dormant parallel pathway in a causally disconnected subspace**. Demonstrated in
  a distilled math example, IOI, and factual recall; evidence it is common in practice.
- **Why it matters.** This is the deepest trap in the whole project. It means "we patched layer 14
  and the answer changed, therefore layer 14 holds the answer" is **not valid**, even though the
  intervention was real. Causality of the *effect* does not establish the *mechanism*. Our
  patching protocol must therefore be paired with (i) a *necessity* test (ablate, don't just add)
  and (ii) a check that the patched subspace is actually read downstream (e.g. patch → does the
  logit-lens at later layers move in the predicted direction?).
- **Confidence:** high (three domains, replicated).
- **Cost:** one extra ablation arm per experiment. Cheap relative to the false claim it prevents.

### F7. Probes measure your probe, not the model — and probe accuracy does not even separate trained from random
**[sources: "Amnesic Probing", Elazar et al., 2021, [arXiv:2006.00995](https://arxiv.org/abs/2006.00995) · "Probing Classifiers: Promises, Shortcomings, and Advances", Belinkov, 2021, [arXiv:2102.12452](https://arxiv.org/abs/2102.12452) · "Information-Theoretic Probing with MDL", Voita & Titov, 2020, [arXiv:2003.12298](https://arxiv.org/abs/2003.12298)]**

- **What it is.**
  - **Elazar et al.**: you cannot infer *behavioural* conclusions from probing results. Their fix,
    *amnesic probing*, removes the property from the representation (INLP/LEACE) and measures
    whether behaviour changes — i.e. it converts "is X encoded?" into "is X used?".
  - **Voita & Titov**: probe *accuracy* fails to distinguish pretrained from **randomly
    initialized** representations, and genuine labels from random synthetic ones, unless probe
    complexity is controlled (MDL).
  - **Belinkov**: catalogue of probe selectivity/complexity confounds.
- **Why it matters.** "We trained a probe on layer 12 and it decodes 'this is a test file' with 94%
  accuracy" is *not* a finding about Qwen. It is a finding about a logistic regression on 1536
  dimensions with 200 examples. The random-init control (same architecture, same probe, same data)
  is the cheapest possible disambiguator and almost nobody runs it.
- **Confidence:** very high.
- **Cost:** one extra model instantiation (`AutoConfig` + `AutoModelForCausalLM(config)`), ~2 min.

### F8. Low-dimensional "brain view" projections are unfaithful out-of-distribution
**[sources: "Interpretability Illusions in the Generalization of Simplified Models", Friedman et al., 2023, [arXiv:2312.03656](https://arxiv.org/abs/2312.03656) · "Interpreting Neural Networks through the Polytope Lens", Black et al., 2022, [arXiv:2211.12312](https://arxiv.org/abs/2211.12312) · "The Road to Explainability is Paved with Bias", Balagopalan et al., 2022, [arXiv:2205.03295](https://arxiv.org/abs/2205.03295)]**

- **What it is.**
  - **Friedman et al.**: a simplified representation (SVD/PCA projection) can accurately
    approximate the full model **in-distribution** and still fail to capture behaviour **OOD**.
    Simplification is not faithfulness.
  - **Polytope lens**: neurons and their linear combinations are *not* the right fundamental
    units, because directions cannot describe how nonlinearities structure the space; the natural
    unit is a **polytope region**. Polysemanticity is a symptom of describing a polytope tiling
    with directions.
  - **Balagopalan et al.**: explanation *quality differs across protected subgroups* in finance,
    healthcare, admissions and criminal justice. Explanations are not uniformly faithful — they
    are better for some inputs than others.
- **Why it matters.** Two direct consequences. (1) A5's PCA/UMAP "brain view" must be labelled
  *descriptive*, never *causal*, and must carry an in-distribution/OOD caveat. (2) Any feature
  story we tell must be checked **per prompt subgroup** (language, identifier style, comment vs
  code, short vs long) or we will report an explanation that only holds for our sample.
- **Confidence:** high.
- **Cost:** subgroup reporting is ~20 lines of pandas; it will probably produce an uncomfortable
  finding, which is the point.

### F9. Decode ≠ cause ≠ attribute ≠ explain: four claims, two of which we can earn
**[sources: "Causal Abstractions of Neural Networks", Geiger et al., 2021, [arXiv:2106.02997](https://arxiv.org/abs/2106.02997) · "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (DAS), Geiger et al., 2023, [arXiv:2303.02536](https://arxiv.org/abs/2303.02536) · "The Linear Representation Hypothesis and the Geometry of LLMs", Park et al., 2023, [arXiv:2311.03658](https://arxiv.org/abs/2311.03658) · "Open Problems in Mechanistic Interpretability", Sharkey et al., 2025, [arXiv:2501.16496](https://arxiv.org/abs/2501.16496)]**

- **What it is.** The rigorous version of "correlation isn't causation" for interp:
  - A **causal abstraction** claim is: there exists a high-level causal model *M* and an alignment
    *π* such that for **all** input pairs, an *interchange intervention* on the high-level variable
    has the same effect as the corresponding intervention on the low-level representation.
    **DAS** replaces brute-force alignment search with gradient descent, and drops the assumption
    that high-level variables map to disjoint neuron sets (they map to subspaces — which is what
    superposition demands).
  - **Park et al.** prove the duality: "linear representation" in *output* space ⇔ **probing**;
    in *input* space ⇔ **steering**. Decoding and steering are two faces of the same hypothesis —
    which is precisely why you need *both* to triangulate, and why one alone is weak evidence.
  - **Sharkey et al. 2025** list the field's open problems, including that we still lack agreed
    criteria for what counts as a satisfying explanation and how to evaluate one.
- **Why it matters.** It gives us a vocabulary that is *honest by construction*:
  **decodes** (a function of the activation predicts X) → **necessary** (removing it changes
  behaviour) → **sufficient** (adding it reproduces behaviour) → **specific** (nothing else
  changes) → **mechanistic** (a validated abstraction over all inputs). Most published "we found
  the X neuron/direction" claims only establish *sufficiency on the authors' prompts*.
- **Confidence:** very high for the framework; high that DAS-style claims are out of CPU reach for
  anything nontrivial at 1.5B.
- **Cost:** adopting the vocabulary is free. Making a *mechanistic* claim costs a full
  interchange-intervention sweep — see F11.

### F10. The validation protocol (exact metrics, exact thresholds, runnable on 8 CPU cores)
**[sources: "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods", Zhang & Nanda, 2023, [arXiv:2309.16042](https://arxiv.org/abs/2309.16042) · "Eliciting Latent Predictions from Transformers with the Tuned Lens", Belrose et al., 2023, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112) · "Analyzing the Generalization and Reliability of Steering Vectors", Tan et al., 2024, [arXiv:2407.12404](https://arxiv.org/abs/2407.12404) · "Locating and Editing Factual Associations in GPT" (ROME), Meng et al., 2022, [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)]**

- **What it is.** Zhang & Nanda show that activation-patching conclusions are **highly sensitive**
  to the metric (logit-diff vs KL) and to the corruption method; they recommend logit-difference
  (more sensitive) and reporting **both** noising and denoising. Tan et al. show steering vectors
  have "substantial limitations both in- and out-of-distribution": steerability varies per input,
  spurious biases drive apparent effectiveness, and vectors are brittle to reasonable prompt
  changes. ROME gives the clean factual-recall template with a logit-difference metric.
- **Why it matters.** These three papers define the failure modes our harness must close. Below is
  the protocol with numbers. All thresholds are **policy choices, pre-registered before running**;
  the *science* is the reported distribution + CI, not the threshold.

#### 4a. Tasks (pick 3, freeze before running)

| id | task | prompt construction | metric |
|---|---|---|---|
| **T1** | induction / copy | `[a,b,c,d, ..., a,b]` → predict `c` | logit-diff(c vs random id) |
| **T2** | factual recall (ROME) | "The capital of France is" → ` Paris` vs ` London` | logit-diff |
| **T3** | steering: type hints | build vector from `def f(x) -> int:` vs `def f(x):` pairs | `->`-presence rate in 20-token completion |

T1 is the **positive control**: induction heads are the best-verified circuit in small LMs
([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html))
and *must* be recoverable. If T1 fails, the harness is broken, not the model.

#### 4b. Activation patching (clean/corrupt, logit-difference)

```
LD(prompt)        = logit(target) − logit(distractor)          # at final position
recovery(L, p)    = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt)
```

Corrupt prompt must differ in **exactly one causal variable** with matched length/position (swap
`Paris`→`London`, swap `c`→`z`). Report **both** directions:

- **noising**: run corrupt prompt, splice in clean activation at (L, p) → recovery should rise.
- **denoising**: run clean prompt, splice in corrupt activation at (L, p) → recovery should fall.

**PASS (component (L,p) matters):**
1. `recovery ≥ 0.20` (mean over pairs), **and**
2. 95% paired-bootstrap CI excludes 0, **and**
3. sign consistent in `≥ 70%` of pairs, **and**
4. holds in *both* noising and denoising arms (|difference| between arms ≤ 0.3), **and**
5. survives all controls in §4d.

Report the **full 28-layer × position heatmap**, never the argmax layer alone.

#### 4c. Dose–response for steering

Sweep `c ∈ {−4,−3,−2,−1,−0.5, 0, 0.5, 1, 2, 3, 4}` with a **unit-normalised** vector, `positions="last"`.

**PASS:**
1. `c = 0` is **bit-identical** to no steering (max |Δlogit| < 1e-5), **and**
2. Spearman ρ(c, behaviour score) `|ρ| ≥ 0.80` (n=11 doses), **and**
3. monotone in `≥ 90%` of adjacent pairs, **and**
4. sign symmetry: `effect(+c)` and `effect(−c)` have opposite signs, **and**
5. at `|c|=4` the fluency guard in §4e still passes.

Record and publish the **saturation knee** (the coefficient where the effect stops growing) — a
linear-looking effect across the whole range usually means you are adding noise, not a feature.

#### 4d. Negative controls (all must pass, or the claim dies)

| control | construction | PASS if |
|---|---|---|
| **C1 random vector** | 10 unit vectors, same layer/norm/positions | mean \|effect\| ≤ **0.10×** real |
| **C2 shuffled labels** | build vector from randomly permuted clean/corrupt pairs | \|effect\| ≤ **0.10×** real |
| **C3 wrong layer** | same vector at L±6 | effect ≤ **0.30×** peak |
| **C4 wrong position** | same vector at a non-decision position | effect ≤ **0.30×** peak |
| **C5 random model** | full pipeline on random-init Qwen2 config | \|effect\| ≤ **0.10×** real, CI includes 0 |
| **C6 self-patch** | patch layer L with its **own** clean activation | max \|Δlogit\| < **1e-4** |
| **C7 zero coef** | `steer(coef=0)` | bit-identical to baseline |
| **C8 label-permutation null** | permute the *evaluation* labels, rerun | empirical p from null distribution |

C2 is the control that kills the most common false positive in the wild: a "steering vector" that
is really just the mean difference between two prompt templates. C5 is Adebayo's sanity check
adapted to LLMs, and 2501.17727 shows it is not a formality — pipelines routinely produce
interpretable-looking features on **randomly initialized** models.

One nuance the literature glosses over: **C5 is decisive for decoder claims and weak for causal
claims.** A random model has no structure to steer, so a steering effect vanishes trivially — the
control passes for the wrong reason. For *causal* claims the load-bearing controls are C1 (random
direction in the same space, same norm) and C2 (shuffled pairing), not C5. Use C5 where it has
teeth: SAE features, probe accuracy, neuron naming.

#### 4e. Fluency / damage

- `ΔPPL = PPL_steered / PPL_base − 1` on a held-out code corpus (50 KB Python, not in any prompt).
- **≤ 10%** → "clean"; **10–50%** → "usable but damaging" (must be stated in the claim);
  **> 50%** → "broken" (claim rejected).
- Also report **top-1 agreement** with the unsteered model on neutral prompts; ≥ 80% expected.
- **Flip rate**: for a *behavioural* claim, require the behaviour to actually change in
  `≥ 30%` of prompts. A mean logit shift with a 3% flip rate is an outlier story.

#### 4f. Generalisation

Split prompts by **template and vocabulary**, not randomly: build the vector on templates
`{A,B}`, evaluate on `{C,D}` with disjoint identifiers.

**PASS:** `effect_test / effect_train ≥ 0.50` **and** test CI excludes 0.
If it's < 0.30, label the finding **"template-specific"** and do not call it a feature.

#### 4g. Statistics

- **n ≥ 100 prompt pairs** (paired design ⇒ paired tests).
- Paired bootstrap (10 000 resamples) for CIs; **Wilcoxon signed-rank** for the null; report
  **Cohen's d_z**, not just p.
- **Holm–Bonferroni** across the 28 layers × 2 directions = **56 tests** (α = 0.05/56 ≈ 8.9e-4).
- **Honest power statement:** with n = 100, α = 8.9e-4, power 0.80 ⇒ minimum detectable
  **d_z ≈ 0.42**. Anything smaller is *not detectable at this budget* and must not be claimed.
- Pre-register the task, the layer band, and the threshold **before** looking (blindfold discipline).

- **Confidence:** high that this protocol is sound; medium on whether T2 (factual recall) is strong
  enough in a *code* model — T1 is the safer positive control.
- **Cost:** measured in forwards. A forward at 1.5B / 20 tokens / fp32 / 8 cores ≈ 1.2 s.
  200 pairs × 5 conditions ≈ 1 000 forwards ≈ 20 min. Full layer sweep 28 × 200 × 2 = 11 200
  forwards ≈ **3.7 h** — an overnight run, which is the right granularity.

### F11. What "understanding" can be claimed at each level — and why (d) is out of reach (and that is fine)

**[sources: as F1–F10; plus "Does Circuit Analysis Interpretability Scale?", Conmy et al., 2023, [arXiv:2307.09458](https://arxiv.org/abs/2307.09458) · "Interpretability at Scale: Identifying Causal Mechanisms in Alpaca", Wu et al., 2023, [arXiv:2305.08809](https://arxiv.org/abs/2305.08809) · "Transformer Feed-Forward Layers Are Key-Value Memories", Geva et al., 2021, [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)]**

| level | claim shape | what it actually asserts | how it can be falsified | reachable here? |
|---|---|---|---|---|
| **L0 descriptive** | "residual norm grows with depth; attn entropy falls after L20" | a statistic about tensors you measured | re-run; it's a measurement | **yes**, trivially |
| **L1 decoding** | "logit lens at L24 predicts the final token 71% of the time" | *your decoder* maps a layer to the output distribution | swap decoder (tuned vs logit lens) — do they agree? | **yes** |
| **L2 naming** | "SAE feature #1234 = 'snake_case identifiers'" | a statement about *your dictionary*, judged by humans | random-model baseline; held-out examples; absorption test | **partly** (1 layer, small width) |
| **L3 causal control** | "adding v at L14 raises type-hint rate by 41pp at +8% PPL" | what *you can do* to the model | full control suite §4d + generalisation §4f | **yes** — this is the target |
| **L4 mechanism** | "the model computes X via circuit C" | a validated causal abstraction over *all* inputs | interchange interventions; causal scrubbing; necessity+specificity | **no** (see below) |

**Why (d) is out of reach — four independent reasons, each sufficient:**

1. **No canonical units** (F4). Leask et al.: SAEs are incomplete and non-atomic; Chanin et al.:
   absorption is not fixed by width or sparsity. If the units aren't canonical, "the circuit" is
   one description among many.
2. **Some features aren't even 1-D** (Engels et al., 2024): circular features for days/months are
   *irreducibly 2-D* and are the true computational unit for modular arithmetic. Any
   direction-based circuit story is wrong there.
3. **The computation is piecewise-linear, not symbolic** (Polytope lens, F8). A "circuit" is a
   *local* linearisation; there are exponentially many regions. FFN layers are key→value memories
   ([Geva et al., 2021](https://arxiv.org/abs/2012.14913)), i.e. pattern→distribution lookup, not
   concept→concept edges.
4. **Cost — the arithmetic, done honestly.** At 1.5B there are ~400 components (28 layers ×
   [12 heads + 1 MLP]). ACDC-style discovery is O(n²) forward passes, so **one** task on **one**
   prompt pair costs ~1.6×10⁵ forwards ≈ 53 CPU-hours ≈ 2 days — borderline, and that is for a
   single pair. A *mechanistic* claim is a causal abstraction validated **over a distribution**
   (Geiger et al.), so 10³ pairs × 1.6×10⁵ forwards ≈ 1.6×10⁸ forwards ≈ **6 CPU-years**. That is
   the wall. Conmy et al. reached Chinchilla 70B only with a narrow task and a large GPU budget;
   on 8 cores the same claim is not "slow", it is *unavailable*.

**The honest reframe:** replace "explain the model" with **"predict the effect of an intervention,
then verify it."** That is the only claim that can be *wrong*, and therefore the only one worth
making. It is also the claim a user can actually use.

- **Confidence:** very high on 1–3; arithmetic on 4.
- **Cost:** accepting L4 as out of scope costs us the flashiest slide and buys us every defensible one.

### F12. Dual-use: the steering API *is* a jailbreak API
**[sources: "Refusal in Language Models Is Mediated by a Single Direction", Arditi et al., 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · "The Geometry of Refusal in LLMs: Concept Cones and Representational Independence", Wollschläger et al., 2025, [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) · "Improving Alignment and Robustness with Circuit Breakers", Zou et al., 2024, [arXiv:2406.04313](https://arxiv.org/abs/2406.04313) · "Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks", Andriushchenko et al., 2024, [arXiv:2404.02151](https://arxiv.org/abs/2404.02151) · "Extending Activation Steering to Broad Skills and Multiple Behaviours", Stolfo et al., 2024, [arXiv:2403.05767](https://arxiv.org/abs/2403.05767)]**

- **What it is.** Arditi et al. show, across **13 open chat models up to 72B**, a single direction
  such that **erasing it prevents refusal** on harmful instructions while adding it elicits refusal
  on harmless ones — and they present it explicitly as a *white-box jailbreak*. Wollschläger et al.
  refine this: refusal is a **cone** of multiple independent directions, so single-direction
  ablation is incomplete but still works. Zou et al. show the same representation-level primitives
  can be used *defensively* (circuit breakers). Andriushchenko et al. show adaptive attacks break
  safety models outright.
- **Why it matters.** The API in `CONTEXT.md` —
  `steer(layer, vector=v, coef=2.0, positions="all")` — is *exactly* the published jailbreak
  primitive. The same code with a different vector is a safety tool. This is not avoidable; it is
  the nature of the instrument. What *is* avoidable is shipping it as a turnkey product.
- **Important nuance for this repo:** `Qwen2.5-Coder-1.5B-Base` has **no refusal training**.
  A demo that "ablates refusal" on it proves nothing about safety and should not be shown as one.
  The risk is that the tool is model-agnostic and someone points it at an Instruct model next week.
- **Confidence:** very high (13 models, replicated, published as a jailbreak).
- **Cost:** ~1 day of plumbing for the guardrails below; near-zero runtime cost.

**Recommended guardrails (concrete, implementable):**

1. **Local-only.** No server, no telemetry, no network calls after model download. If a UI server
   exists, bind `127.0.0.1` and require a token.
2. **Model allowlist with an explicit assert.** Refuse to load a model whose config indicates
   safety tuning unless `--authorized-research` is passed *and* logged. Record
   `model_id` + weights SHA-256 in every trace.
3. **No precomputed harmful vectors in the repo.** Vectors are computed at runtime from the
   user's own prompt pairs, or fetched with a hash + provenance record. Never commit a
   `refusal_direction.npy`.
4. **Two-tier API.** `steer()` is generic. A `harmful_directions` module does not exist. If a
   refusal vector is computed, it requires an explicit opt-in flag and writes an audit entry.
5. **Append-only audit log** (`audit.jsonl`): timestamp, model hash, layer, coefficient, vector
   hash, prompt hash, positions. Reproducibility *and* traceability are the same feature.
6. **`RISKS.md` in the repo root**, stating plainly: "this is a jailbreak primitive; refusal
   ablation is published and works on 13 models; do not point this at a model you are not
   authorised to test."
7. **No "disable safety" toggle in the UI.** Keep it an instrument, not a chat app. A5 must not
   render a "jailbreak" button.
8. **Publish methods and controls, not vectors.** Publish the harness and the negative results;
   never publish "here is the working refusal vector for model X".

### F13. Red-teaming our own plan: the 5+1 ways this project fails
**[synthesis of F1–F12; no new external source]**

| # | failure mode | why it is likely here | early warning signal | mitigation |
|---|---|---|---|---|
| **1** | **The demo that proves nothing.** Beautiful per-layer logit-lens curves, attention heatmaps, PCA "brain view", 8 tok/s streaming — and zero falsifiable claims. | All of these are *descriptive*. They are cheap, they look like science, and they will fill the UI before any causal test exists. | You cannot state a claim in the form "if I do X, Y changes by Z". | **Certificate gate**: the UI may only render a "causal" badge when a `certificate.json` with all controls passing exists. Otherwise it says `descriptive only`. |
| **2** | **The steering vector is a prompt artifact.** Contrastive mean-difference between two prompt sets captures surface form, not a concept. | ActAdd works from a *single pair* — it is almost too easy, which is the tell. | Shuffled-label control (C2) was never run. Test/train effect ratio < 0.3. | C2 + C1 mandatory; report `effect_test / effect_train`; require ≥ 0.5. |
| **3** | **The effect is below the noise floor.** bf16 CPU forward on a 1.5B model has logit noise ~1e-2; a 0.01-nat "causal effect" gets reported. | `CONTEXT.md` targets the bf16 release; nobody measures the noise floor before measuring effects. | Effects of the same magnitude as run-to-run variance; "significant" results that don't replicate. | Measure `noise_floor = max\|Δlogit\|` for (a) two identical runs, (b) fp32 vs bf16. Require every reported effect ≥ **5×** noise floor. |
| **4** | **The SAE rabbit hole eats the budget.** 28 layers × 15k features × 10⁸ tokens is not a CPU workload. | Q4 in the LEDGER ("can an SAE be trained on 1 layer in <24h?") is a trap: even if yes, one layer at low width yields a non-canonical dictionary (F4) and no causal result. | Week 3 and the only artefact is a partially trained SAE. | **Cap it**: 1 layer, ≤ 4 096 features, ≤ 10 M tokens, hard timebox. Drop unless it beats a linear probe on a downstream task (Gupta et al. 2025 says it probably won't). |
| **5** | **p-hacking the layer/task sweep.** 28 layers × 3 positions × 5 tasks × 2 directions = 840 tests; report the winner. | The search space is small enough to sweep and large enough to guarantee a false positive. | The finding lives at one (layer, position) and nowhere near it. | Pre-register one task + one layer band; Holm–Bonferroni; publish the **full sweep heatmap**; require replication on a second disjoint task. |
| **6** | **The model is too small/base to contain the phenomenon** — and a negative result gets misread as a harness bug. | 1.5B base code model; "docstring feature" may not exist as a clean direction. | Endless debugging of a null result. | Pre-define **"the model doesn't have it"** as a valid publishable outcome; always run T1 (induction) as a positive control so a null on T3 is interpretable. |

---

## Concrete implementation (code sketches, real APIs)

### Design principle

**A claim is a JSON object that has passed a gate.** The UI reads certificates, not tensors.
If the certificate is missing or fails, the claim is labelled `descriptive` and may not be
rendered as causal. This makes self-deception a *build error*, not a judgement call.

### `code/microscope.py` — minimal hook layer (A4 owns the production runtime; this is the test contract)

```python
# code/microscope.py  —  reference implementation for the A6 validation contract.
# Deliberately dependency-light: torch + transformers only. No TransformerLens, no nnsight,
# so the *test* harness cannot inherit a framework's bugs.
import hashlib, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

def _hs(out):                      # Qwen2DecoderLayer returns tensor or tuple by version
    return out[0] if isinstance(out, tuple) else out

def _with_hs(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class Microscope:
    def __init__(self, path, dtype=torch.float32, threads=8):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).eval()
        self.cfg = self.m.config
        self.L, self.d = self.cfg.num_hidden_layers, self.cfg.hidden_size
        self.layers = self.m.model.layers
        self.tied = (self.m.lm_head.weight.data_ptr()
                     == self.m.model.embed_tokens.weight.data_ptr())
        self._n_fired = 0

    # ---------- primitives ----------
    def ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids

    def logits(self, ids, **hooks):
        """hooks: {layer:int -> callable(out)->out}; returns final-position logits."""
        hs = [blk.register_forward_hook(
                  (lambda i: (lambda mod, inp, out: hooks[i](out)))(i))
              for i in hooks]
        try:
            return self.m(ids).logits[0, -1]
        finally:
            for h in hs: h.remove()

    def logit_diff(self, ids, target_id, distractor_id, **hooks):
        z = self.logits(ids, **hooks)
        return (z[target_id] - z[distractor_id]).item()

    def capture(self, ids, layer):
        store = {}
        def fn(out):
            store["h"] = _hs(out).detach().clone(); return out
        self.logits(ids, **{layer: fn})
        return store["h"]                       # (1, T, d)

    # ---------- intervention: steer ----------
    def steer(self, layer, vec, coef, positions="last", schedule="constant"):
        v = (vec / vec.norm()).to(self.m.dtype)
        state = {"step": 0}
        def fn(out):
            h = _hs(out).clone()                # NEVER mutate in place
            c = coef * (0.5 ** state["step"] if schedule == "decay" else 1.0)
            if positions == "all":
                h = h + c * v
            elif positions == "last":
                h[:, -1, :] = h[:, -1, :] + c * v
            else:                                # "gen": every position, current token
                h = h + c * v
            state["step"] += 1
            return _with_hs(out, h)
        return {layer: fn}

    # ---------- intervention: patch ----------
    def patch(self, layer, clean_h, positions="last"):
        def fn(out):
            h = _hs(out).clone()
            if positions == "last":
                h[:, -1, :] = clean_h[:, -1, :]
            else:
                h[:, : clean_h.shape[1], :] = clean_h
            return _with_hs(out, h)
        return {layer: fn}

    # ---------- decoder: logit lens ----------
    def logit_lens(self, ids, layer):
        h = self.capture(ids, layer)
        return self.m.lm_head(self.m.model.norm(h))[0, -1]   # apply FINAL norm
```

**Two details that will silently produce wrong science if missed:**
- `logit_lens` **must** apply `model.model.norm` before `lm_head`. Skipping it is a common
  copy-paste bug and makes the last-layer check (below) fail by a large margin.
- Steering must be **out-of-place** (`h.clone()`). In-place edits mutate the tensor the block
  returned, which with `use_cache=True` is also what gets reused/appended for subsequent positions
  and generation steps — so the edit leaks forward and `coef=0` stops being identity (control C7).
  Aliasing is silent; the no-op test is the only thing that catches it.

### `code/golden.py` — the ground-truth test (the single most important check)

Build a model where **we already know the answer**, then require the harness to recover it.
If the pipeline cannot find a circuit in a model we built ourselves, it cannot be trusted on Qwen.

```python
# Train a 2-layer, 2-head, d=64 attention-only transformer on a synthetic copy task.
# Vocab 64, seq 32, 20k steps => ~15 min on 8 CPU cores.
# Expected ground truth: layer-1 head-0 is the induction head.
# PASS: (a) ablating layer-1 head-0 drops copy accuracy from >95% to <10%;
#       (b) full-layer patching peaks at layer 1, not layer 0;
#       (c) logit lens at layer 1 already ranks the copied token first.
```

This is the *only* check with an external oracle. Everything else is self-consistency.

### `code/certify.py` — the gate

```python
def certify(claim_id, kind, statement, model_hash, task, metric, controls, cost):
    ok = (controls["zero_coef_identity"] and controls["self_patch_noop"]
          and controls["random_vector"]["ratio"] <= 0.10
          and controls["shuffled_labels"]["ratio"] <= 0.10
          and controls["wrong_layer"]["ratio"] <= 0.30
          and controls["random_model"]["ratio"] <= 0.10
          and metric["value"] >= 5 * controls["noise_floor"]
          and metric["ci95"][0] > 0)
    return {"claim_id": claim_id, "kind": kind, "statement": statement,
            "model": {"sha256": model_hash, "dtype": "float32"},
            "task": task, "metric": metric, "controls": controls,
            "cost": cost, "verdict": "PASS" if ok else "FAIL"}

# UI contract: if verdict != "PASS", render as "descriptive only" and grey out any
# causal wording. A failed certificate is a *result*, not an error.
```

Example certificate:

```json
{"claim_id": "C-003", "kind": "causal",
 "statement": "unit direction v_typehint at layer 14, coef +2, raises type-hint rate",
 "metric": {"name": "typehint_rate_delta", "value": 0.41, "ci95": [0.31, 0.50], "d_z": 0.88},
 "controls": {"zero_coef_identity": true, "self_patch_noop": true,
              "random_vector": {"ratio": 0.05}, "shuffled_labels": {"ratio": 0.03},
              "wrong_layer": {"ratio": 0.11}, "random_model": {"ratio": 0.00},
              "noise_floor": 0.008, "delta_ppl": 0.08, "flip_rate": 0.47},
 "cost": {"forwards": 1440, "wall_s": 1830}, "verdict": "PASS"}
```

### `code/checks.py` — the 12-point "is the microscope lying?" suite

Run on every build. Any failure blocks release.

| # | check | expected |
|---|---|---|
| 0 | **golden model** — train 2-layer induction circuit | recovers layer-1 head-0; ablation → accuracy < 10% |
| 1 | determinism — same input twice | max \|Δlogit\| < 1e-5 (or documented noise floor) |
| 2 | **logit lens at last layer == real logits** | max \|Δ\| < 1e-3 (fp32) |
| 3 | logit-lens top-1 agreement vs depth | non-decreasing in expectation; no cliff at L27 |
| 4 | tuned lens ≥ logit lens at every layer ≥ 5 | if not, the lens training is broken |
| 5 | attention rows sum to 1 | \|Σ − 1\| < 1e-4; entropy ∈ [0, log T] |
| 6 | hook coverage | exactly `num_hidden_layers` fires; shape `(B,T,1536)`; no in-place mutation |
| 7 | **GQA check** | 12 Q heads / 2 KV heads ⇒ the **K and V tensors are bitwise identical within each group of 6** (Q differs per head, so attention *patterns* legitimately differ) |
| 8 | causality of patching | patching position i must not change logits at positions < i |
| 9 | noise floor (fp32 vs bf16, 2 runs) | reported; every claim ≥ 5× this |
| 10 | cross-backend (safetensors vs dequant GGUF) | top-1 agreement ≥ 99% on 100 prompts (A1 contract) |
| 11 | random-model pipeline | all "findings" vanish; ratios ≤ 0.10 |

Check **7** is a genuinely useful free test, and it is easy to get backwards: GQA shares **K and V**
across a group, not attention patterns. Each query head still has its own Q, so 6 heads legitimately
produce 6 different attention maps — but their K/V projections must be bitwise identical. If they
aren't, you are hooked into the wrong tensor (or `repeat_kv` is being applied twice).

Check **10** is the A1↔A6 interface contract: if the Q4_K_M dequant disagrees with bf16 by more than
1% top-1, the dequant is wrong and every downstream number is suspect.

### Compute budget (8 cores, fp32, 20-token prompts)

| artefact | forwards | wall clock |
|---|---|---|
| noise floor + 12 checks | ~600 | 15 min |
| golden model training | — | ~15 min |
| T1 patching sweep (28 layers × 100 pairs × 2 arms) | 5 600 | ~2 h |
| T3 steering dose–response (11 doses × 120 prompts × 4 arms) | 5 280 | ~1.8 h |
| all controls | ~2 000 | 40 min |
| **full round-1 validation** | **~14 000** | **~5 h (one overnight run)** |

Note: **run analysis in fp32**, not bf16. 1.5B × 4 B = 6 GB, fits in 31 GB. The stored weights are
bf16, so casting to fp32 preserves every weight value and only improves matmul precision; report
the bf16↔fp32 delta as the noise floor.

### Tuned lens is *not* affordable — plan around it

A full tuned lens trains an affine map to the 151 936-token vocabulary at every layer:
`1536 × 151936 ≈ 233 M` params per layer × 28 = **6.5 B extra parameters**, plus the training
corpus. That is not a CPU workload. Options, in order of preference:

1. **Logit lens + per-layer scalar calibration** (temperature + bias, 2 params) — captures most of
   the bias correction for ~0 cost.
2. **Rank-64 affine** map into the vocab (`1536→64→151936`), ≈ 10 M params/layer — still heavy.
3. **Restricted vocabulary** (top 4 096 tokens) tuned lens — cheap, but only valid for tasks whose
   answers are in that set. Fine for T1/T2.
4. Skip it. Logit lens + the last-layer exactness check (check 2) is enough for L1 claims.

---

## What NOT to do (traps)

1. **Do not ship a neuron-colouring UI.** There is no canonical basis (F1). Colour by *measured
   effect* (a coefficient you verified), never by "this neuron means X".
2. **Do not call attention weights an explanation.** Jain & Wallace 2019; Serrano & Smith 2019.
   Label the panel "attention distribution — descriptive".
3. **Do not report probe accuracy as a finding.** Voita & Titov 2020: probe accuracy doesn't even
   separate trained from random representations. Always run the random-init model.
4. **Do not report an SAE feature name without the random-model baseline.** 2501.17727: SAEs on
   random transformers score the same. The baseline is the finding.
5. **Do not report a steering result without ΔPPL.** "It worked" and "it broke the model" are the
   same measurement until you add perplexity.
6. **Do not report a patching result without a necessity arm.** Sufficiency alone is compatible with
   the subspace illusion (F6) and with a dormant parallel pathway.
7. **Do not sweep layers and report the best one.** 840 tests, no correction, guaranteed false
   positive. Pre-register; publish the heatmap.
8. **Do not use `positions="all"` for a claim about "this token's decision".** It changes the
   computation everywhere and destroys the localization claim.
9. **Do not narrate the model's reasoning from its chain-of-thought.** Turpin et al. 2023
   ([arXiv:2305.04388](https://arxiv.org/abs/2305.04388)) and Lanham et al. 2023
   ([arXiv:2307.13702](https://arxiv.org/abs/2307.13702)): CoT is often unfaithful. The microscope
   must not become a confabulation engine with extra steps.
10. **Do not compare across dtypes without a stated noise floor.** bf16 CPU logits are noisy at the
    1e-2 level; that is larger than many published "effects".
11. **Do not commit a refusal vector.** See F12.
12. **Do not let the UI be the deliverable.** A demo that streams pretty traces at 8 tok/s and
    proves nothing is the most likely failure mode (F13 #1).

---

## Open questions for other agents

- **A1.** What is the exact top-1 agreement between your Q4_K_M dequant and the bf16 release on
  100 prompts? I need a number to set check 10's threshold (proposed: ≥ 99%). Also: does the GGUF
  preserve `tie_word_embeddings`, or does it materialise a separate `lm_head`?
- **A2.** (a) Does your logit lens apply the **final RMSNorm** before `lm_head`? If not, our layer
  indices won't compose and check 2 will fail. (b) Do you index layers by *block output* (my
  convention: hook `model.model.layers[L]`, edit its output) or by *block input*? We must pick one
  or patching and steering will disagree by one layer. (c) Given the 6.5B-parameter cost of a full
  tuned lens, do you agree to drop it in favour of logit lens + scalar calibration?
- **A3.** (a) Is the vector **unit-normalised** before `coef` multiplication? My dose–response
  thresholds assume yes. (b) Is `steer()` out-of-place? In-place edits make `coef=0` non-identity
  and will fail control C7. (c) What is your `schedule="decay"` half-life — I need it fixed to
  make the sweep reproducible.
- **A4.** (a) Can you guarantee fp32 forward at ≥ 5 tok/s on 8 cores? If not, the 14 000-forward
  validation budget doubles. (b) Where does the hook sit relative to the KV cache — will steering
  at generation step *n* contaminate the cache for step *n+1*? (c) Determinism: which ops are
  nondeterministic on CPU, and what is the measured run-to-run Δlogit?
- **A5.** Will you accept a **claim-kind badge** (`descriptive | decoding | causal`) on every
  panel, driven by the certificate? A PCA/UMAP "brain view" must be labelled *descriptive* and
  carry the Friedman et al. OOD caveat. No neuron-colouring.
- **All.** Do we agree that **L4 (full mechanism) is out of scope**, and that the project's success
  criterion is "a verified causal claim with a passing certificate", not "we explained the model"?

---

## SCHEMA CHANGE (if any)

Additive only — all keys optional, existing consumers unaffected.

```json
{"step": 0, "token_id": 123, "token_str": "def",
 "layers": [{"layer": 0, "resid_norm": 12.3,
             "logit_lens_topk": [[id, tok, prob]],
             "attn_entropy": 1.7, "mlp_out_norm": 4.1}],
 "logits_topk": [[id, tok, prob]], "chosen": [id, tok, prob],

 "meta":   {"model_id": "Qwen/Qwen2.5-Coder-1.5B-Base",
            "weights_sha256": "…", "dtype": "float32",
            "noise_floor_logit": 0.008, "harness": "a6-v1",
            "layer_indexing": "block_output"},

 "intervention": {"kind": "none|steer|patch", "layer": 14, "coef": 2.0,
                  "vector_sha256": "…", "positions": "last",
                  "schedule": "constant", "arm": "noising|denoising|null"},

 "claim":  {"kind": "descriptive|decoding|causal|mechanistic",
            "certificate": "C-003", "verdict": "PASS|FAIL"}}
```

Rationale: without `meta.noise_floor_logit` a trace cannot be interpreted (F13 #3). Without
`intervention.arm` a patching result is ambiguous (Zhang & Nanda 2023). Without `claim.kind` the UI
cannot enforce honesty (F13 #1). `layer_indexing` prevents the off-by-one between A2 and A3.

---

## Build next (ordered, with effort estimate)

| # | task | effort | why first |
|---|---|---|---|
| 1 | **Noise floor + identity checks** (checks 1, 2, 6, 9) | 0.5 d | Cheapest possible; if the harness can't reproduce the model's own logits, nothing else matters. |
| 2 | **Golden model** (train 2-layer induction circuit, checks 0) | 0.5 d | The only external oracle we will ever have. Do it before touching Qwen. |
| 3 | **Patching harness on T1** (induction, 28-layer sweep, both arms) | 1.5 d | Positive control; if it can't find induction heads, the harness is broken. |
| 4 | **Full negative-control suite** (C1–C8) + certificate gate | 1.0 d | Turns honesty into a build error. |
| 5 | **Steering dose–response on T3** + ΔPPL + flip rate | 1.0 d | The actual deliverable claim; needs 4 to be meaningful. |
| 6 | **Random-model baseline** (C5) end-to-end | 0.5 d | Adebayo/2501.17727 sanity check; publish the number. |
| 7 | **Held-out template split** (F10 §4f) | 0.5 d | Separates feature from template artifact. |
| 8 | **Statistics module** (paired bootstrap, Wilcoxon, Holm, d_z) | 0.5 d | Without it, n=100 becomes n=1 with extra steps. |
| 9 | **Subgroup faithfulness report** (F8) | 0.5 d | Likely to produce the most uncomfortable, most useful result. |
| 10 | **RISKS.md + audit log + allowlist** (F12) | 0.5 d | Ship-day blocker, not a nice-to-have. |
| 11 | *(optional, timeboxed)* **1-layer SAE**, 4k features, 10M tokens, **with random-model baseline** | 3–5 d | Do last. Expect it to fail the downstream-probe test; that is a result. |

**Total to a defensible round-1 result: ~7 days**, of which ~5 h is unattended compute.
Everything after #6 is refinement. If only three things ship, ship **1, 2, 4**.

---

*Written by A6 (skeptic). Every empirical claim above carries a resolvable link; the four
protocol thresholds marked "policy" are proposals for the orchestrator to ratify or overrule,
not scientific laws.*
