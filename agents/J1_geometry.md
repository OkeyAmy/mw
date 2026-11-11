# J1 — What an LLM's internal representation actually looks like, geometrically

**Question owned:** the owner wants a 3D "brain" visualization. What shape is the thing,
*according to measurement*? This file answers only that. It does not design the UI (A5 owns
that) and it does not re-litigate steering (A3).

**Verdict up front:** there is **no published 3D shape of "the representation space"**, and
the honest answer is that the object is not 3D at all. What *is* measured is:

1. a **1536-dimensional additive bus with no privileged basis** (Elhage et al. 2021);
2. on which the *data* at any layer occupies a manifold of **intrinsic dimension ≈ 10–35**
   (Ansuini 2019; Valeriani 2023; Cheng 2025) — i.e. ~1–2% of the ambient width;
3. whose directions are **near-orthogonal feature directions, not axes** (Elhage 2022),
   arranged in the toy model as **vertices of a polytope on a sphere** and in real models as
   **lobed, power-law-tapered clouds** (Li et al. 2025);
4. with **1–3 outlier dimensions that dominate every raw distance** (Timkey & van Schijndel
   2021; Sun et al. 2024);
5. and a handful of features that are **irreducibly 2-D** (circles ⇒ cones), which no
   arrow-per-concept picture can draw (Engels et al. 2024).

Any 3D picture is therefore a projection of a ~20-dimensional object, and the projection is
the claim. This file gives the numbers, then proposes three metaphors and says what each
one lies about.

---

## Method note (so the reader can discount appropriately)

Every citation below was fetched during this session and the quoted line was read on the
fetched page (arXiv HTML, `transformer-circuits.pub`, or the published PDF). Where I only
had an abstract I say so. Quotes are marked `[OBS]` = read verbatim on the source;
`[INF]` = my inference; `[SPEC]` = speculation.

**Nothing here was measured on our model.** `Qwen2.5-Coder-1.5B` is 28 layers, hidden 1536,
12 Q heads / 2 KV heads, intermediate 8960. Every number attached to *our* model in this
file is arithmetic from `CONTEXT.md`, not measurement, and is marked
`UNVERIFIED — needs runtime` where it matters.

Two corrections to claims already circulating in this repo:

- **A6 F1 says features sit at "~72° spacing".** That number is only the regular pentagon
  (5 features in 2 dims). The paper's actual claim is that feature directions are vertices
  of *whatever uniform polytope* solves the packing problem for that (n, m, sparsity)
  (`[OBS]` "features seem to organize themselves into geometric structures such as
  pentagons and tetrahedrons"). 72° generalised to 1536 dims is wrong; the general
  condition is **near-orthogonality**, i.e. mutual cosine ≈ 0, not a fixed angle.
- **The brief says Gurnee & Tegmark found "grid-like" codes.** They found **linear**
  lat/long and time probes. Grid cells are cited as *inspiration*, and the mesh structure
  is explicitly a conjecture (`[OBS]` "We conjecture, but do not show…"). See F7.

---

## TL;DR

- **Concepts are directions, but not in the Euclidean sense you would draw.** Park et al.
  prove that the inner product which makes causally independent concepts orthogonal is
  **not** the Euclidean one; the read (probe) direction and the write (steering) direction
  for the same concept are *different vectors*, related by the unembedding covariance.
  A single arrow per concept therefore conflates two things the theory separates.
- **Superposition is the only place the literature produces an actual shape**, and it is a
  *polytope inscribed on a sphere in the hidden space* — digons, triangles, tetrahedra,
  pentagons, square antiprisms — with feature dimensionalities clustered at
  1, ¾, ⅔, ½, ⅖, ⅜, 0. The authors themselves say it may be "partly idiosyncratic to the
  toy model".
- **Intrinsic dimension is the single most important number for the visualization.**
  Measured ID is **O(10)** (5–35 depending on model/layer/estimator) against ambient widths
  of 512–4096. A 3-D view shows at most 3 of ~20 effective degrees of freedom; 3/20 = 15%.
- **In real models the shape is not a clean polytope.** Measured on Gemma-2-2B SAE features:
  a power-law eigenvalue spectrum ("fractal cucumber", slope −0.47 at layer 12 vs −0.24 /
  −0.25 at the ends) plus **functional "lobes"** (code/math vs English vs dialogue) that are
  spatially localised at 954σ and 74σ above two nulls.
- **A few dimensions own the metric, and they are not the interesting ones.** 1–3 rogue
  dimensions carry up to 99.6% of expected cosine similarity; massive activations are up to
  ~10⁵× larger than the rest and act as constant bias terms. **Plot raw activations and you
  will render the bias, not the model.**

---

## Findings

### F1. The linear representation hypothesis is about *directions under a non-Euclidean inner product*, and it separates reading from writing
**[source: Park, Choe, Veitch, "The Linear Representation Hypothesis and the Geometry of
Large Language Models", ICML 2024, [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)]**

**What it is.** The paper formalises "concept = direction" twice: in the **unembedding
(output) space** and in the **embedding (input) space**, and proves they connect to
**probing** and **steering** respectively. Its central technical move is that the inner
product which makes semantics work is *learned by the model*, not chosen by us:

> `[OBS]` "there is no obvious reason to expect that algebraic manipulations based on, e.g.,
> the Euclidean inner product, should be semantically meaningful." (§3)

> `[OBS]` "The intuition we follow here is that causally separable concepts should be
> represented as orthogonal vectors." (§3.1)

The causal inner product has an explicit form, `⟨γ̄, γ̄′⟩_C = γ̄ᵀ M γ̄′`, where `M` is
estimated from the **covariance of the unembeddings**; with `A = M^{1/2}` the embedding and
unembedding representations of a concept become the *same* vector. Consequences they
demonstrate on LLaMA-2:

- most pairs of causally separable concepts are **approximately orthogonal** under `⟨·,·⟩_C`,
  with a visible block-diagonal residual structure grouping semantically related concepts
  (`[OBS]` §4, "most pairs of concepts are nearly orthogonal … there is also a clear block
  diagonal structure");
- the same estimate works on Gemma (Appendix D.2);
- they can *construct* probes and steering vectors from counterfactual pairs.

**What is NOT linear (in this paper's own scope).** Three boundaries, all load-bearing:

1. **Euclidean geometry is the wrong geometry.** Cosine similarity in the raw residual
   stream is a statement about the model's *unembedding covariance*, not about meaning.
2. **Read ≠ write.** The probe direction (unembedding representation) and the steering
   direction (embedding representation) are related by `M`, not identical. A UI that draws
   one arrow labelled "the concept" is drawing a *choice of representation*, and the paper
   proves the two choices differ.
3. **Scope: output space only.** `[OBS]` "we do not address interpretability of either model
   parameters, nor the activations of intermediate layers." For this project — which works
   at L8–L14 of a 28-layer model — that is a real limitation, not a footnote. The causal
   inner product is defined via the unembedding; there is no published version of it for
   `resid_post[l]` for `l ≪ L`.

**Why it matters for us.** It kills the simplest 3D design (one arrow per concept in raw
space) *and* it supplies the correct alternative: if you want an arrow, you must say whether
it is the read arrow or the write arrow, and you must build the metric from `W_U`.

**VALIDATION**
```
HYPOTHESIS: a "linear representation" in an LLM is not a direction under the Euclidean inner
            product, and the read and write directions for the same concept differ.
TEST:       fetched arxiv.org/html/2311.03658 and read §3, §3.1, §3.2, Theorem 3.2, §4.
RESULT:     "there is no obvious reason to expect that algebraic manipulations based on,
            e.g., the Euclidean inner product, should be semantically meaningful";
            Theorem 3.2 maps γ̄_W ↦ λ̄_W via the Riesz isomorphism of ⟨·,·⟩_C, i.e. read and
            write are different spaces identified by M, not by the transpose.
VERDICT:    CONFIRMED (for the paper's scope: embedding/unembedding space)
FALSIFIER:  finding a passage where the authors use raw cosine in residual space as a
            semantic measure without the M transform.
```

**Confidence:** very high for the theory. **Cost of ignoring:** we draw one arrow per concept
and silently average two different objects.

---

### F2. Categorical concepts are polytopes and hierarchies are orthogonal direct sums — the closest thing to a "shape" that has been *proved and measured in a real LLM*
**[source: Park, Choe, Veitch, "The Geometry of Categorical and Hierarchical Concepts in
Large Language Models", 2024, [arXiv:2406.01506](https://arxiv.org/abs/2406.01506)]**

**What it is.** Extends F1 from binary contrasts to features (`is_animal`) and categories
(`{mammal, bird, reptile, fish}`). A feature is a **vector**, not a direction; a categorical
concept is the **convex hull of its elements' vectors**, and:

> `[OBS]` "hierarchically related concepts (such as plant ⇒ animal and mammal ⇒ bird) live in
> orthogonal subspaces, while categorical concepts are represented as polytopes." (abstract)

> `[OBS]` "we show how to extend the formalization … to represent features … as vectors. This
> allows us to immediately formalize the representation of categorical concepts as polytopes
> in the representation space." (abstract)

Measured on **Gemma and LLaMA-3**, with **593 noun + 364 verb** WordNet synsets (synsets
filtered to ≥50 words in the Gemma vocabulary). Held-out check: projections of **test** words
onto the estimated concept vector are near 1, **random** words near 0, and the effect
disappears with **shuffled unembeddings** — a null that our own `VALIDATION.md` would demand.

**The shape, stated precisely.** A concept that is a *natural category* is a **simplex** (they
prove this in Appendix C); a semantic **hierarchy is a direct sum of polytopes in mutually
orthogonal subspaces**. So the concept space is not a list of arrows and not a single
polytope: it is a **tree of polytopes, each factor living in its own orthogonal subspace**,
with the branching encoded as orthogonality.

**Boundary.** Same scope limit as F1: `[OBS]` "we do not address the 'internal' structure of
the LLMs… the results rely on the duality between the embedding and unembedding spaces."
Again: this is the *lexicon*, not layer 12.

**Why it matters.** This is the best-supported "shape" claim in the literature with a null
control, and it is *renderable*: a 3-D scene of small simplices arranged along orthogonal
axes, where moving along an axis means "more specific". It is also the reason a flat list of
feature arrows is wrong even at the top of the network.

**VALIDATION**
```
HYPOTHESIS: categorical concepts are convex polytopes (simplices) and semantic hierarchy is
            orthogonality between subspaces.
TEST:       fetched arxiv.org/html/2406.01506, read abstract, §3, §4.2 Theorem 8, §5.2.
RESULT:     Theorem 8 (Hierarchical Orthogonality); "the polytope representation … is the
            convex hull of the vector representations of the elements"; Fig. 3 shows test
            words ≈1, random words ≈0, structure absent under shuffled unembeddings.
VERDICT:    CONFIRMED (in embedding/unembedding space, Gemma + LLaMA-3, WordNet)
FALSIFIER:  a category whose element vectors were not affinely independent, or hierarchy
            structure surviving the shuffled-unembedding null.
```

---

### F3. Superposition produces the only literal geometric shape in the literature — and it is a polytope on a sphere
**[source: Elhage, Hume, Olsson, Schiefer, et al., "Toy Models of Superposition", 2022,
[transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html)
· [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)]**

**Setup, exactly.** A model `ReLU(WᵀWx − b)` with `n` input features and `m < n` hidden
dimensions, trained on reconstruction. Standard experiment: **n = 400 features, m = 30
dimensions**, `I_i = 1`, sparsity `S` swept; "the number of input features n doesn't matter
as long as it's much larger than the number of hidden dimensions" (`[OBS]` §"Uniform
Superposition").

**The mechanism.** Superposition is loss-optimal because sparse features rarely co-occur, so
buying extra directions at the cost of small interference is worth it. Features are stored as
**near-orthogonal directions** (`[OBS]` "features are represented as almost-orthogonal
directions in the vector space of neuron outputs"), and the arrangement is a Thomson-type
packing problem:

> `[OBS]` "When our model chooses to represent a feature, the feature is embedded as a point
> on an m-dimensional sphere."

**The observed shapes, by sparsity — this is the precise answer to "what shape".** Defining
per-feature dimensionality `D_i ≈ ‖W_i‖² / Σ_j (Ŵ_i·W_j)²`, the measured dimensionalities
cluster on discrete lines:

| `D_i` | configuration | features / dims |
|---|---|---|
| 1 | dedicated axis (not separable in the dense regime) | 1/1 |
| **¾** | **tetrahedron** | 4/3 |
| **⅔** | **triangle** (the "Mercedes-Benz frame") | 3/2 |
| **½** | **antipodal pair (digon)** | 2/1 |
| **⅖** | **pentagon** | 5/2 |
| **⅜** | **square antiprism** | 8/3 |
| 0 | feature not learned | — |

`[OBS]` "there are distinct lines corresponding to dimensionality of: ¾ (tetrahedron),
⅔ (triangle), ½ (antipodal pair), ⅖ (pentagon), ⅜ (square antiprism), and 0 (feature not
learned)."

Also observed in the learning-dynamics section: **octahedra** (6 features, antipodal pairs of
correlated sets). Non-uniform importance/sparsity **deforms** the polytope smoothly until it
"snaps" to another one — a genuine first-order phase change (they confirm it by computing the
two loss curves crossing).

**The three facts that matter for a visualization.**

1. **The shape is a set of directions, not a cloud of points.** It is `n` *feature* points on
   the unit sphere in `m` dims, chosen to minimise coherence. (A *token's state* is a point
   inside the ball — it is the sum of the active features — but the features themselves are on
   the surface.) A rendering that puts *features* in the interior is showing something else.
2. **It is discrete and sparse in configuration space.** Between the sticky values there is
   nothing — the model jumps. A smooth "feature landscape" is wrong.
3. **72° is not a constant.** For `n=5, m=2, density 0.05` the solution is a regular pentagon
   ⇒ 72° between neighbours. Change `n`, `m`, or `S` and the angle changes; in high `m` the
   angles go to ≈90° (near-orthogonality).

**The authors' own hedge, which must travel with the claim.**
`[OBS]` "In some ways, the structure described in this section seems 'too elegant to be true'
and we think there's a good chance it's at least partly idiosyncratic to the toy model we're
investigating."

**VALIDATION**
```
HYPOTHESIS: at fixed sparsity, features organise into vertices of a specific uniform
            polytope, with quantised per-feature dimensionality.
TEST:       fetched transformer-circuits.pub/2022/toy_model (full text, 165k chars) and read
            §"Superposition as a Phase Change", §"Feature Dimensionality", §"Non-Uniform
            Superposition"; grep for polytope/angle terms returned all quoted lines.
RESULT:     quantised dimensionalities {1, ¾, ⅔, ½, ⅖, ⅜, 0}; first-order phase change
            confirmed analytically and empirically; n=400, m=30; pentagon at density 0.05.
VERDICT:    CONFIRMED (toy model only)
FALSIFIER:  a sparsity sweep in which per-feature dimensionality varies continuously with no
            clustering, or in which polytope identity does not predict the loss ordering.
```

**Confidence:** very high for the toy result; **medium-to-low** for extrapolation to a
1536-dim 28-layer model. Nothing in this paper was measured on a transformer.

---

### F4. What survives of that shape in a *real* model: overcompleteness, extreme sparsity, and an anisotropic "fractal cucumber" with functional lobes
**[sources: Templeton et al., "Scaling Monosemanticity", 2024,
[link](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) ·
Li, Michaud, Baek, Engels, Sun, Tegmark, "The Geometry of Concepts: Sparse Autoencoder Feature
Structure", *Entropy* 27(4):344, 2025, [arXiv:2410.19750](https://arxiv.org/abs/2410.19750)]**

**How overcomplete, and how sparse, in a frontier model.** Claude 3 Sonnet, middle-layer
residual stream:

- `[OBS]` SAEs of "1,048,576 (~1M), 4,194,304 (~4M), and 33,554,432 (~34M)" features.
- `[OBS]` "the average number of features active … was fewer than 300" per token.
- `[OBS]` ~65% dead at 34M ⇒ ~12M alive.
- `[OBS]` "for 82% of our features, the most-correlated neuron has a correlation of 0.3 or
  smaller."

Arithmetic: **300 / 33,554,432 = 8.9 × 10⁻⁶** active density. The residual stream is not a
dense code; it is an extremely sparse sum of a huge dictionary.

**The measured macro-shape (this is the "brain" object that actually exists).** Li et al.
analyse **Gemma-2-2B, layer-12 residual SAE, 16k features, mean L0 = 41**, at three scales
they name atom / brain / galaxy:

- **Atom scale:** difference vectors for the same semantic transformation form
  **parallelograms / trapezoids** — "Parallelogram or trapezoid structures suggest that there
  is a unique direction in the activation space that represents each semantic transformation"
  `[OBS]`. Distractor directions (e.g. word length = PC1 of layer 0) must be projected out
  first, or the structure is invisible.
- **Brain scale ("lobes"):** features that co-fire within 256-token blocks are **also**
  spatially clustered. `[OBS]` "math and code features form a 'lobe' akin to functional lobes
  seen in neural fMRI images." Two-lobe partition ≈ {code/math} vs {English}; three-lobe splits
  English into {short messages/dialogue} vs {scientific papers}. Significance against two
  separate nulls: **954σ** (permuted cluster labels) and **74σ** (random Gaussian decoder
  vectors). This is *not* a t-SNE artefact claim — the statistics are computed in the full
  space; the picture is only for humans.
- **Galaxy scale:** the covariance eigenspectrum is **not** Wishart/isotropic. It is a
  **power law**, steeper for SAE features than for raw activations. They name the shape a
  **"fractal cucumber"**: "whose width in successive dimensions falls off like a power law."
  Slope of the top-100 eigenvalues: **−0.47 at layer 12**, **−0.24 at layer 0**, **−0.25 at
  layer 24**.

**Why this matters more than the toy polytopes.** It is a real model, with null controls, and
it reports *the opposite* of clean orthogonality: real feature space has **local neighbourhood
structure** ("distance in decoder space maps roughly onto relatedness in concept space" —
Templeton et al.) and **global anisotropy**. The toy model's polytope is the idealised
limiting case (uniform importance, uniform sparsity, independent features). Real models are
non-uniform on all three axes, so the polytope is **deformed and incomplete**.

**Caveats that must be stated.**
- The lobes live in **SAE decoder-vector space**, a *derived* basis. It is not the residual
  stream and not "the model's own coordinates".
- The "fractal cucumber" is a fit to the top 100 eigenvalues of a covariance matrix, not a
  manifold reconstruction. Slope −0.47 is a number, not a picture.
- Feature *count* is a function of dictionary width (feature splitting: "San Francisco" →
  2 features at 4M → 11 at 34M). So "how many lobes" is also width-dependent.

**VALIDATION**
```
HYPOTHESIS: real-LLM feature space is (a) massively overcomplete and sparse, (b) locally
            modular (lobes), (c) globally anisotropic with power-law spectral decay.
TEST:       fetched both sources in full; grep for feature counts, active counts, "lobe",
            "power law", "cucumber", significance values.
RESULT:     34M features / <300 active / 82% no neuron corr > 0.3; 954σ and 74σ; slopes
            −0.47 (L12) vs −0.24 (L0), −0.25 (L24); 16k SAE, L0=41, Gemma-2-2B L12.
VERDICT:    CONFIRMED as reported; UNVERIFIED on Qwen2.5-Coder-1.5B (needs runtime)
FALSIFIER:  a layer whose eigenvalue spectrum follows Marchenko–Pastur, or lobe structure
            that survives the random-Gaussian-decoder null equally well.
```

---

### F5. Intrinsic dimension: ~10–35, against ambient widths of 512–4096. This is the number that decides what a 3D view can honestly claim
**[sources: Ansuini, Laio, Macke, Zoccolan, "Intrinsic dimension of data representations in
deep neural networks", NeurIPS 2019, [arXiv:1905.12784](https://arxiv.org/abs/1905.12784) ·
Valeriani, Doimo, Cuturello, Laio, Ansuini, Cazzaniga, "The geometry of hidden representations of large transformer
models", NeurIPS 2023, [arXiv:2302.00294](https://arxiv.org/abs/2302.00294) · Cheng, Doimo,
Kervadec, Macocco, Yu, Laio, Baroni, "Emergence of a High-Dimensional Abstraction Phase in
Language Transformers", ICLR 2025, [proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/57568e093cbe0a222de0334b36e83cf5-Paper-Conference.pdf)]**

**The original CNN result (Ansuini 2019).** TwoNN estimator; ID vs layer gives a
**"hunchback"**: rise, peak, monotone fall.

- `[OBS]` VGG-16-R pool4: **ID ≈ 19** with **ED ≈ 10⁵**, ratio **2×10⁻⁴**.
- `[OBS]` Across 14 CNNs (AlexNet/VGG/ResNet), peak ID **between 100 and 120** at relative
  depth **0.2–0.4**.
- `[OBS]` Last hidden layer: ID **≈12 (ResNet152)** to **≈25 (AlexNet)**, against ED 512–4096.
- `[OBS]` ID of the last hidden layer predicts top-5 accuracy, **r = 0.94** (r = 0.99 within
  ResNets).
- `[OBS]` **Linear (PCA) dimensionality is 1–2 orders of magnitude too big**: PC-ID ≈ 200 vs
  TwoNN ID ≈ 18 in VGG-16's last hidden layer.
- `[OBS]` "The absence of a gap in the [PCA] spectrum … is, by itself, an indication that the
  data manifolds are not linear."

**Transformers (Valeriani 2023).** Same estimator, ESM-2 protein LMs and iGPT:

- `[OBS]` Peak ID ≈ **20 / 25 / 32** for ESM-2 35M / 650M / 3B.
- `[OBS]` Plateau after the peak: ID **between 5 and 7**, "independent from the embedding
  dimension that varies from 480 … to 2560".
- `[OBS]` iGPT: max ID **28 (M)**, **32 (L)**; local minimum **22** at relative depth 0.3–0.4.
  **iGPT-L has embedding dimension 1536 — the same hidden width as our Qwen.** Its measured
  manifold dimension is 22–32.
- `[OBS]` Llama-2-70B, SST: first local minimum at **0.3 relative depth** (Appendix E).

**Language models (Cheng et al., ICLR 2025).** Five LMs (llama, olmo, opt, mistral, pythia),
three corpora:

- `[OBS]` "the ID for all models is O(10), which lies orders of magnitude lower than the
  models' hidden dimension at 4096 ~ O(10³)."
- `[OBS]` the high-ID phase "tends to take place relatively early (starting approximately at
  layer 6 or 7, and mostly being over by layer 20), except for OPT, where it approximately
  lasts from layer 17 to layer 23."
- `[OBS]` the peak is **functional**, not incidental: the ID-peak layers are where surface-form
  information is dropped and abstract/semantic tasks improve; the peak marks a transition in
  layer function, and cross-model neighbourhood alignment is concentrated there.

**Honest limits on the number itself.**
- `[OBS]` (Ansuini) "For a finite number of data points, the estimated values remain very close
  to the ground truth ID, when this is smaller than ~20." Above ~20, treat with care.
- **The estimate depends on which point cloud you measure.** Cheng et al. use the **last-token**
  vector of each sequence (`[OBS]` "we choose to represent input sequences with their last token
  representation at each layer"); the token-level study of Llama-3-8B / Mistral-7B / Pythia-6.9B
  ([arXiv:2501.10573](https://arxiv.org/abs/2501.10573)) measures token clouds and finds that
  **shuffling raises the peak at prompt level but lowers it at dataset level**:
  `[OBS]` "at the dataset level, we see that the unshuffled ID has a more prominent peak than
  the shuffled ID, whereas it is the other way around at the prompt level." That is not a
  contradiction; it is two different manifolds.
- **No published ID measurement exists for Qwen2.5-Coder-1.5B.** `UNVERIFIED — needs runtime`
  for our model. The defensible prior is **~10–35**.

**Arithmetic for our box.** `20/1536 = 1.3%`; a 3-D view shows `3/20 = 15%` of the effective
degrees of freedom and `3/1536 = 0.2%` of the ambient ones.

**VALIDATION**
```
HYPOTHESIS: the intrinsic dimension of hidden-state manifolds in transformers is O(10), not
            O(hidden_size), and it peaks in early-to-middle layers.
TEST:       fetched arxiv.org/html/1905.12784, arxiv.org/html/2302.00294, and the ICLR 2025
            PDF; extracted the quoted sentences and Table-free numeric claims.
RESULT:     19 vs 10^5 (VGG pool4); 12–25 (last hidden); 20/25/32 (ESM-2); 5–7 (plateau);
            22–32 (iGPT, d=1536); "O(10)" vs 4096 (5 LMs).
VERDICT:    CONFIRMED across three independent groups and two modalities.
FALSIFIER:  a trained transformer whose layer-wise TwoNN ID is within 10x of its hidden width
            across the middle third of depth.
```

---

### F6. Some features are irreducibly 2-D (circles ⇒ cones). They are rare in the counts reported, and causally load-bearing where found
**[source: Engels, Michaud, Liao, Gurnee, Tegmark, "Not All Language Model Features Are
One-Dimensionally Linear", ICLR 2025, [arXiv:2405.14860](https://arxiv.org/abs/2405.14860)]**

**What they found.** Clustering GPT-2-small layer-7 SAE dictionary elements
(~25k features → **1000 clusters**, ~500 manually inspected) recovers **three** interpretable
circular features — **days of the week, months of the year, years of the 20th century**. On
Mistral-7B layer 8 (2¹⁶ dictionary elements → **~2700 clusters**, ~2000 inspected) they recover
**two** (days, months) and state plainly:

> `[OBS]` "we did not find other obviously interesting and clearly irreducible features."

**Quantitative status of "what fraction".** They do **not** publish a prevalence estimate, and
their own limitations paragraph says why:

> `[OBS]` "It is unclear why we did not find more interpretable multi-dimensional features. We
> are unsure if we are failing to interpret some of the high-scoring multi-dimensional
> features, if most multi-dimensional features lie in dimensions higher than two, if our
> clustering technique is not powerful enough to find some features, or if there are truly not
> that many."

`[INF]` The defensible reading is therefore: **multi-dimensional features exist, are causally
used, and are a small minority of what a current SAE surfaces** — 3/≈500 clusters in GPT-2 and
2/≈2000 in Mistral — with an unknown false-negative rate. Anyone who says "X% of features are
non-linear" is inventing a number.

**The geometry, which is the part I need.** Two precise facts:

1. **The circle is a heptagon / dodecagon, not a smooth ring.** `[OBS]` "the days of the week
   and months of the year … are clustered at the vertices of a heptagon and dodecagon, and there
   is nothing 'between' adjacent weekdays or months along the circle."
2. **The honest 3-D object is a cone.** `[OBS]` "these best dimensions are usually the second
   and third because the first PCA dim is an 'intensity' direction that manifests as the radius
   of the circle … thus the overall structure for these multi-d features is perhaps best thought
   of as a **cone**."

So: a circular feature is (angle = position in the cycle) × (radius = intensity). That is a
genuine, measured, renderable 3-D object — and it is emphatically *not* a direction.

**Causality.** They intervene on the circular subspace (k = 5 PCs, circular probe, average-ablate
the rest) and show it is causally implicated: `[OBS]` "early layer interventions on the circular
subspace have almost the same intervention effect as patching the entire layer." An
off-distribution sweep over a grid of `(r, θ)` inside the circle gives: `[OBS]` "They imply that
Mistral treats the circle as a multi-dimensional representation with α encoded in the **angle**."
(One nuance: the layer-8 SAE-derived probe transfers to neighbouring layers far better than a
layer-8 PCA probe — −2.32 vs 0.029 average logit difference at layer 6.)

**Implication for a 3-D view.** A renderer that represents each feature as **one direction**
cannot represent a heptagon: any 1-D readout of a circle is at best `cos θ`, which identifies `θ`
and `θ + π`, so the circle collapses to a diameter and half the cycle is lost. Either 2-D
subspaces are first-class objects (render the plane and put the state's `(r, θ)` on it), or
circular features will be silently mis-drawn.

**VALIDATION**
```
HYPOTHESIS: some LM features are irreducible ≥2-D, and among those found, the shape is a
            discrete polygon with a radial intensity axis (a cone).
TEST:       fetched arxiv.org/html/2405.14860v3; grep for "cone", "heptagon", "dodecagon",
            "we did not find", "Limitations".
RESULT:     quotes above; counts 1000 clusters / ~500 inspected (GPT-2), ~2700 / ~2000
            (Mistral); 3 and 2 features respectively; causal interventions ≈ whole-layer.
VERDICT:    CONFIRMED (existence + causality); prevalence UNVERIFIED (authors say so)
FALSIFIER:  a circular subspace intervention that fails to beat a matched random 5-D subspace.
```

---

### F7. Space and time *are* linearly coded — as 2-D and 1-D coordinates, with high R². "Grid-like" is a conjecture, not a result
**[sources: Gurnee & Tegmark, "Language Models Represent Space and Time", 2023/2024,
[arXiv:2310.02207](https://arxiv.org/abs/2310.02207) · Marks & Tegmark, "The Geometry of
Truth", 2023, [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)]**

**Space/time (Gurnee & Tegmark).** Six datasets (world / USA / NYC places; historical figures;
entertainment; headlines), Llama-2 7B/13B/70B and Pythia 160M–6.9B, ridge probes on the last
entity token per layer. Out-of-sample R² at **60% depth**:

| model | World | USA | NYC | Historical | Entertainment | Headlines |
|---|---|---|---|---|---|---|
| Llama-2-7B linear | **0.881** | 0.799 | 0.219 | 0.785 | 0.788 | 0.564 |
| Llama-2-70B linear | **0.911** | 0.864 | 0.359 | 0.835 | 0.885 | 0.746 |
| Llama-2-70B MLP(256) | 0.926 | 0.869 | **0.312** | 0.839 | 0.884 | 0.739 |

`[OBS]` "using nonlinear probes results in minimal improvement to R² for any dataset or model."
The MLP is sometimes *worse* (NYC 0.312 vs 0.359). That is the strongest available evidence that
a **continuous** quantity is linearly decodable — which matters, because most LRH evidence is
binary/categorical.

**The shape is 2-D (lat, long) and 1-D (time) — i.e. a plane and a line, not a grid.** The
grid-cell language is aspiration:

> `[OBS]` "We conjecture, but do not show, these basic primitives underlie a more comprehensive
> causal world model" (§6)
>
> `[OBS]` "We conjecture that the most canonical form of this structure is a discretized
> hierarchical mesh…" (§7)

**Truth (Marks & Tegmark).** `[OBS]` "clear linear structure … true statements separating from
false ones in the top two principal components" at LLaMA-2-13B layer 15 over the
clause-ending punctuation token. But the same paper supplies the best warning about drawing
that picture:

- `[OBS]` the separation axes "align often, but not always";
- `[OBS]` "in early layers, cities and neg_cities separate **antipodally**, before **rotating to
  lie orthogonally** … and finally **aligning** in later layers" (App. C).

**So: the direction for one concept rotates with depth**, passes through antipodal and
orthogonal configurations, and only converges late. Any 3-D picture that fixes a concept's
direction once is wrong for 2/3 of the layers.

**Also from the same paper, a methodological fact we should steal.** The logistic-regression
probe direction is **not** the feature direction when another correlated feature is present;
their `mass-mean` / LDA correction (`Σ⁻¹θ_mm`) recovers a direction that is *more causally
implicated* while being equally accurate. `[OBS]` "LR … treats the small projection of θ_f onto
θ_t as significant, and adjusts the probe direction to have less 'interference'." ⇒ a rendered
"concept arrow" that came from sklearn's `LogisticRegression` is a decision boundary, not a
concept.

**VALIDATION**
```
HYPOTHESIS: continuous spatial/temporal quantities are linearly decodable from mid-depth
            residual states, with nonlinear probes adding little; the code is 2-D/1-D, not a
            grid.
TEST:       fetched arxiv.org/html/2310.02207 (Table 2 + §3.2, §6, §7) and
            arxiv.org/html/2310.06824 (§3, §4, App. C, §5 mass-mean).
RESULT:     R² table above; "minimal improvement" quote; "We conjecture, but do not show";
            rotation path quote; LR-vs-mass-mean quote.
VERDICT:    CONFIRMED for linear decodability; REFUTED for "grid-like" as a measured finding.
FALSIFIER:  a dataset where a 1-layer MLP probe beats linear by >0.1 R², or a locality result
            that survives only under the linear probe.
```

---

### F8. Anisotropy is not a global property of the space — it is 1–3 dimensions, and in modern LLMs those dimensions are ~10⁵× larger
**[sources: Ethayarajh, "How Contextual are Contextualized Word Representations?", EMNLP 2019,
[arXiv:1909.00512](https://arxiv.org/abs/1909.00512) · Timkey & van Schijndel, "All Bark and No
Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality", EMNLP
2021, [arXiv:2109.04404](https://arxiv.org/abs/2109.04404) · Sun, Chen, Kolter, Liu, "Massive
Activations in Large Language Models", COLM 2024, [arXiv:2402.17762](https://arxiv.org/abs/2402.17762)]**

**The cone (Ethayarajh).** `[OBS]` representations "are not uniformly distributed with respect
to direction. Instead, they are anisotropic, occupying a narrow cone in the vector space."
For GPT-2, average cosine between two random words is **≈0.6 in layers 2–8**, then "increases
exponentially from layers 8 through 12", until "any two words have on average an almost perfect
cosine similarity". Arithmetic: cos 0.885 ⇒ **27.7° cone half-angle**; cos 0.99 ⇒ **8.1°**.

**The cause is not the space, it is a few axes (Timkey & van Schijndel).**

- `[OBS]` "a small number of rogue dimensions, often just **1-3**, dominate these measures."
  (the body reports **1–5** across the model set; the count is layer- and model-dependent).
- `[OBS]` "XLNet … a single dimension contributes more than **99%** of the expected cosine
  similarity" (Table 1: layer 11 top-dim share **0.996**, anisotropy **0.981**).
- GPT-2 layer 12: top dim share **0.763**, anisotropy **0.885**; layer 11: 0.275 / 0.640.
  BERT layer 11: top dim **0.884**, anisotropy 0.506. Word2Vec 0.130, GloVe 0.104.
- `[OBS]` that dimension has **E[x] = 180.0** while all others average **−0.084 (σ = 0.77)**.
- `[OBS]` the consequence: the metrics can "rely on less than 1% of the embedding space";
  and these dimensions "show a strong correlation with absolute position and punctuation".
- `[OBS]` therefore "the cone lies along a single axis or within a low dimensional subspace,
  rather than being a global property across all dimensions."
- Fix: **standardization** (and mean removal) restores sensible similarity.

**Modern LLMs: massive activations (Sun et al.).** `[OBS]` "very few activations exhibit
significantly larger values than others (e.g., **100,000 times larger**)" — these are
input-independent, "function as indispensable bias terms", and they concentrate attention on
their own tokens.

**Consequences for any visualization (this is the single most practical finding in this file).**
1. A PCA/UMAP of raw residual states will be a 1–3 dimensional picture of **a bias term**, not
   of the model.
2. Cosine similarity between points in that plot is 0.9+ for *everything*; the plot looks like
   a hairball by construction.
3. The fix is cheap and mandatory: **mean-centre, then standardize per dimension** (or whiten),
   and **report how much of the top-1 singular value you removed**. If PC1 explains >80% of
   variance, the picture is about PC1.
4. Standardization also changes which 3 dims you are showing — so it is a *choice*, and the UI
   must say which.

**VALIDATION**
```
HYPOTHESIS: anisotropy in transformer representations is driven by 1-3 outlier dimensions,
            not by a globally narrow cone, and removing them restores meaningful geometry.
TEST:       fetched arxiv.org/html/1909.00512, arxiv.org/html/2109.04404 (Table 1), and the
            arXiv abstract page for 2402.17762.
RESULT:     Table 1 numbers above; "E[x]=180.0" vs "-0.084 (σ=0.77)"; "less than 1% of the
            embedding space"; "100,000 times larger".
VERDICT:    CONFIRMED for GPT-2/BERT/RoBERTa/XLNet; CONFIRMED as a phenomenon for LLMs
            (massive activations), but the exact dim count for Qwen is UNVERIFIED.
FALSIFIER:  a layer whose cosine-similarity decomposition is flat across dimensions (equal
            per-dimension contribution) while global anisotropy is high.
```

---

### F9. Layer-wise dynamics: the state follows a curved, accelerating path; ID is a hunchback; the *readout* converges late
**[sources: as F5; plus Fernando & Guitchounts, "Transformer Dynamics: A Neuroscientific
Approach to Interpretability of LLMs", CCN 2025 abstract,
[PDF](https://2025.ccneuro.org/abstract_pdf/Fernando_2025_Transformer_Dynamics_A_Neuroscientific_Approach_Interpretability.pdf)
· Turner, "Residual Stream Norms Grow Exponentially over the Forward Pass", blog,
[turntrout.com](https://turntrout.com/residual-stream-norms-grow-exponentially-over-the-forward-pass)
· "Do Language Models Use Their Depth Efficiently?", NeurIPS 2025,
[PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/eae3af0f5868f0a2eceb74208966d55b-Paper-Conference.pdf)]**

**The path is not straight.** On Llama-3.1-8B, last-token residual stream, PCA-reduced:

- `[OBS]` "the RS follows a **curved trajectory** with **attractor-like dynamics in the lower
  layers**."
- `[OBS]` velocity: "relatively constant velocities in early layers followed by a slight
  increase in velocity, with the **steepest acceleration occurring in the final third**."
- `[OBS]` cosine similarity between successive layer states **increases with depth**, and
  within-layer (attn→MLP) transitions are *more* similar than cross-layer (MLP→attn) ones —
  i.e. attention and MLP move the state differently, measurably.
- `[OBS]` early layers "distribute variance across more dimensions than later layers, with
  later layers needing fewer principal components to explain the same amount of variance."

**Norm generally grows, and fast.** `[OBS]` (blog) "the norm of each residual stream grows
exponentially over the forward pass, with average per-layer growth rate of about **1.045** in
gpt-2-xl" — i.e. ~4.5%/layer, **3.43× over 28 layers** (`1.045^28 = 3.4297`, my arithmetic).
(The same post reports exceptions: some EleutherAI models *decrease* in norm at some layers, so
treat monotonicity as a tendency, not a law.) A
peer-reviewed confirmation of the mechanism: `[OBS]` "Since all interaction with the residual
stream in pre-layernorm Transformers is additive, it is expected that the norm of the residual,
‖h_l‖₂, will grow in later layers" (NeurIPS 2025 depth paper), which also measures the
**relative contribution** of each sublayer to the residual: roughly constant through the first
half, then "a significant drop around the middle", sharpest for attention, rising again in the
last few layers.

**What that means geometrically.** `[INF]` Because norms grow ~4.5%/layer *and* successive-layer
directions stay highly correlated (`[OBS]` cosine similarity increases with depth), a Euclidean
3-D plot of `h_l` for `l = 0…27` is dominated by the norm, not by the direction: it will read as a
slowly-turning **outward spiral** largely regardless of what the model computes. **Normalize per
layer before plotting, or plot direction only** (e.g. cosine to the final state, or project the
unit vector).

**The readout converges, the content does not.** Tuned-lens/logit-lens work shows that
intermediate layers can be decoded into the vocabulary, and that the *mapping* (not just the
content) changes with depth: a single affine map per layer is needed, and it beats the naive
logit lens
([Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens",
2023, arXiv:2303.08112](https://arxiv.org/abs/2303.08112)). `[OBS]` logit lens "yielded useful
insights but is often brittle"; the tuned lens is "more predictive, reliable and unbiased".
Combined with F7's rotation result, the honest picture is: **one object (the state) moving
through a space whose *axes themselves* appear to rotate relative to the readout.**

**VALIDATION**
```
HYPOTHESIS: the residual stream traces a curved, accelerating trajectory whose norm grows
            ~exponentially with depth, and whose effective dimensionality peaks early/mid.
TEST:       fetched the CCN 2025 abstract PDF, the turntrout post, and the NeurIPS 2025 depth
            PDF; quoted the sentences above.
RESULT:     "curved trajectory … attractor-like"; "steepest acceleration … final third";
            "~1.045 per layer" (gpt2-xl); "significant drop around the middle".
VERDICT:    CONFIRMED as reported. The CCN item is a 2-page abstract, not a full paper —
            weight accordingly. The norm-growth number is a blog measurement.
FALSIFIER:  a model where ‖h_l‖ is flat across depth, or where successive-layer cosine
            similarity decreases with depth.
```

---

### F10. Attention and MLP are geometrically *different objects* and must not be drawn the same way
**[sources: Elhage, Nanda, Olsson, et al., "A Mathematical Framework for Transformer Circuits",
2021, [link](https://transformer-circuits.pub/2021/framework/index.html) · Geva, Schuster,
Berant, Levy, "Transformer Feed-Forward Layers Are Key-Value Memories", EMNLP 2021,
[arXiv:2012.14913](https://arxiv.org/abs/2012.14913)]**

**The residual stream is a bus, not a pipeline.**

> `[OBS]` "the residual stream doesn't have a 'privileged basis'; we could rotate it by rotating
> all the matrices interacting with it, without changing model behavior."
>
> `[OBS]` "dimensions of the residual stream become something like 'memory' or 'bandwidth'."
>
> `[OBS]` "every individual head operates on comparatively small subspaces (often **64 or 128
> dimensions**), and can very easily write to completely disjoint subspaces and not interact."

For our model: `head_dim = 1536/12 = 128`; each head's QK map is rank ≤ 128 and each OV map
writes a 128-d vector into 1536 dims. The MLP has `8960/1536 = 5.83×` more neurons than the bus
has dimensions — `[OBS]` the paper's own framing calls these "**bottleneck activations**" and
expects them to be "unusually challenging to interpret."

**Two writes, two geometries.**

| | attention head | MLP layer |
|---|---|---|
| what it computes | **two nearly independent low-rank maps**: `W_QᵀW_K` (where to look) and `W_O W_V` (what to move) | **key → value lookup**: `W_in` rows are keys, `W_out` columns are values |
| geometric object | a **subspace read/write pair**, rank ≤ 128, plus a permutation-like routing of information *across token positions* | a **set of (key direction, output-distribution) pairs**; the output is a *distribution over the vocabulary*, not a direction in the residual stream |
| what it does to the state | adds a weighted sum of value vectors sourced from *other positions* | adds a composition of memories, `[OBS]` "the output of a feed-forward layer is a composition of its memories" |
| how it should be drawn | beams crossing the layer axis (source position → destination position), each beam confined to a 128-d subspace | a fan of small "concept → next-token" glyphs, not an arrow between concepts |

`[OBS]` (Geva) "feed-forward layers … operate as key-value memories, where each key correlates
with textual patterns in the training examples, and each value induces a distribution over the
output vocabulary"; "lower layers tend to capture shallow patterns, while upper layers learn
more semantic ones."

**Why the difference matters for a 3-D picture.** Attention's geometry is *relational across
positions* (a matrix of movements); MLP's geometry is *local to a position* (a lookup). A single
"node = layer" rendering makes them look like the same kind of object and implies a
concept→concept edge that the MLP does not implement. `[OBS]` the framework also warns that
heads are only "**largely** independent" and that composition (Q-, K-, V-) is what makes depth
matter — so drawing 12 independent beams per layer overstates independence.

**VALIDATION**
```
HYPOTHESIS: attention and MLP write to the residual stream with different geometries
            (low-rank subspace routing vs key→value lookup), and neither is a
            concept→concept edge.
TEST:       fetched transformer-circuits.pub/2021/framework in full and the arXiv abstract for
            2012.14913; quoted the lines above.
RESULT:     "no privileged basis"; "64 or 128 dimensions"; "bottleneck activations";
            "key-value memories … value induces a distribution over the output vocabulary".
VERDICT:    CONFIRMED (framework is analytical, validated on 1–2 layer attention-only models;
            the MLP claim is empirical on GPT-2-class models).
FALSIFIER:  an MLP whose output is better described as a single direction than as a
            distribution over the vocabulary.
```

---

### F11. What an HONEST 3-D visualization can show: three candidate metaphors, and what each one lies about

This section is the deliverable for the owner's question. Constraints that any candidate must
satisfy, from the findings above:

- **(C1)** the state is ~1536-d with ~10–35 effective dims ⇒ **at least 85% of the geometry is
  not on screen**;
- **(C2)** there is no privileged basis ⇒ **the choice of 3 axes is a modelling decision**, and
  must be displayed as such;
- **(C3)** 1–3 dimensions dominate raw distances ⇒ **standardize or you render the bias**;
- **(C4)** concept directions rotate with depth (F7) ⇒ **cannot be a fixed frame**;
- **(C5)** some features are 2-D cones (F6) ⇒ **arrows are not universal**;
- **(C6)** the state's norm grows ~4.5%/layer (F9) ⇒ **plot direction, not magnitude**, or the
  plot is a spiral about norm;
- **(C7)** an SAE basis is a derived basis (F4) ⇒ **label which basis you are in**.

#### Metaphor A — "The layer stack / bus" (structural frame)

A vertical spine = 28 layers. At each layer, a translucent disc = the residual stream at that
position. Inside the disc, a small constellation of beams: 12 head-beams entering from the
previous disc (confined to 128-d sub-discs) and one broad MLP fan. The token's state is a point
on each disc; a thin line joins the points.

- **Gets right:** additivity, layer order, subspace bandwidth, the read/write distinction, the
  fact that attention moves information between positions and MLP does not. It is the correct
  *skeleton* for a microscope whose product is patching and steering: "patch at (L, pos)" maps
  directly onto a disc.
- **Lies about:** (i) **dimension** — a disc reads as 2-D; (ii) **independence** — the 12 beams
  look parallel and uncoupled, but composition is the whole story; (iii) **position** — a single
  spine per layer hides that the residual stream is `[T × 1536]` and that attention's entire job
  is cross-position; (iv) it implies flow direction, whereas the stream is a sum with no
  direction of travel.

#### Metaphor B — "The feature sky / polytope" (superposition)

A sphere (or a 3-D projection of one). Each feature is a point on the sphere; near-orthogonal
features are spread; a few are grouped into the measured polytopes (triangle, pentagon,
tetrahedron, square antiprism). The token's state is the point *inside* the ball, and each
feature's activation is its dot product with that feature's direction (draw as a spike or a
radius).

- **Gets right:** concepts are directions; features are many and near-orthogonal; the state is a
  *sum* of features; interference is visible (features are not at 90°); the polytope family is a
  real published result with quantised dimensionalities.
- **Lies about:** (i) **which sphere** — the polytope result is from a 30-dim toy model with
  uniform importance; the real case is non-uniform, so the polytope is *deformed* and the
  quantised dimensionalities are not exact; (ii) **angles** — in 1536-d, showing a pentagon at
  72° is a lie of scale, because the true configuration is near-orthogonal and the "shape" is
  only a 2-or-3-plane shadow; (iii) **multi-D features** — circles/cones cannot be vertices;
  (iv) **non-linearity** — a direction story is a *local* linearisation (polytope lens: the
  space is piecewise linear, regions are separated by boundaries); (v) **prevalence** — the toy
  model says every feature is a vertex; real SAEs suggest most are messy and some are cones.

**B′ — the variant I would actually defend for "brain": the measured point cloud.**
Render the **SAE decoder vectors** of one layer (e.g. 16k points in 1536-d) projected to 3-D,
coloured by measured functional lobe (code/math, English long-form, dialogue). This is not a
metaphor: Li et al. measured that co-firing features are spatially co-located at **954σ** above
a permuted-label null, and that the cloud is a power-law-tapered "fractal cucumber", not a ball.
It is the only object in this file that is simultaneously (a) a real model, (b) null-controlled,
and (c) literally a shape.

- **Lies about:** the projection (t-SNE/PCA distances are not interpretable), the basis (SAE
  decoder space ≠ residual stream), and stability (feature splitting means the point set changes
  when you change dictionary width).

#### Metaphor C — "The trajectory ribbon" (the state over depth)

For one token position, the sequence `h_0 … h_27`, mean-centred and standardized, projected to
3-D by PCA (or by an anchored projection). Draw as a **ribbon**, not a line: width = local
intrinsic dimension estimate (TwoNN on the k-NN ball around that state), colour = logit-lens
entropy (or the top-1 token identity). Small glyphs on the ribbon where a head or MLP writes.
Steering/patching is drawn as a *branch* off the ribbon at the layer where it was applied, with
the measured post-intervention continuation as the branch's actual path.

- **Gets right:** every pixel is a measurement of a real object; curvature, acceleration, the
  early attractor-like phase, and late convergence are all documented (F9); it makes the
  intervention visible in the same frame as the state, which is the product; and it can carry
  its own uncertainty (ID width, variance explained) instead of hiding it.
- **Lies about:** (i) **faithfulness** — a 3-D projection can be accurate in-distribution and
  wrong OOD (Friedman et al. 2023), so a "this prompt went left" story is a story about the
  projection; (ii) **distance** — after standardization, 3-D Euclidean distance is a chosen
  metric, not the causal inner product of F1; (iii) **singularity** — the ribbon suggests one
  particle, but the object is a `[T × 1536]` field and the token position shown is one of T;
  (iv) **[INF] endpoint convergence** — the last layers' apparent convergence is plausibly the
  unembedding/outlier direction dominating (F8) rather than a decision, which can look like "the
  model made up its mind" when it is a readout artefact. That last one is testable: compare the
  trajectory shape in the standardized space against the raw space.

#### Recommendation

**Primary: C. Secondary inset: B′. Frame: A.** Concretely: a layer axis (A) carries a ribbon
(C) for the selected token position; a side panel shows the layer's feature cloud (B′) with the
current state's dot products highlighted. Each panel must display: variance explained by the
shown components, the ID estimate, whether centring/standardization was applied, and whether
massive-activation dimensions were excluded.

If the owner insists on **one** 3-D object, defend **C** — because it is the only candidate whose
every coordinate is a direct measurement of a tensor we actually have, and whose error can be
quantified and printed on screen. A polytope (B) is a claim about a toy model; a lobe cloud (B′)
is a claim about an SAE basis; a ribbon is a claim about one vector per layer, which is true by
construction.

**What no metaphor can show, and should not pretend to:** the mechanism. A trajectory is a
*description* of a state sequence, not a circuit. Per `VALIDATION.md` L4 and A6 F11, "this
layer's direction changed" is not "this layer computes X".

---

## Concrete implementation — the five measurements that make the picture honest

All are cheap on 8 CPU cores; all are per-layer statistics over a prompt set, not per-token
generation.

1. **Standardize before projecting.** `X -= X.mean(0); X /= X.std(0)` (or whiten via the
   covariance). Report `λ_1/Σλ` before and after. If `λ_1/Σλ > 0.8` pre-standardization, the
   raw plot is one dimension. **Cost: seconds.** (Sources: F8.)
2. **TwoNN intrinsic dimension per layer.** `d̂ = (N−1) / Σ log(r₂ᵢ/r₁ᵢ)`, over the k-NN graph
   of the layer's states (N prompts × positions). ~30 lines, no model. Report ID vs layer and
   overlay it on the ribbon width. **Cost: seconds per layer.** (Sources: F5.)
3. **Variance-explained + reconstruction error of the 3-D projection**, reported per layer, plus
   the same number on a **random-init model** of the same architecture (A6 check C5/11). If the
   projection looks the same on a random model, it is a projection artefact. **Cost: one extra
   forward pass set.**
4. **Directional stability across depth.** `cos(h_l, h_final)`, and `cos(v_concept,l,
   v_concept,l')` for a fixed concept vector (e.g. a diff-in-means at L8) across `l`. This
   quantifies F7's rotation and prevents a fixed-frame rendering. **Cost: seconds.**
5. **Logit-lens entropy per layer** (already in the canonical trace schema as `attn_entropy`;
   add `lens_entropy`) as the ribbon's colour channel, plus `resid_norm` so the grow-rate
   (F9) can be shown as a separate axis rather than smuggled into position.

**Do not** attempt a manifold reconstruction (UMAP/t-SNE) as the primary view: it is
non-parametric, non-invertible, and its distances are not the model's. Use PCA (linear,
invertible, with a meaningful residual) for the geometry, and UMAP only as a labelled "neighbour
map" inset — which is exactly the compromise Li et al. make (picture for humans, statistics in
the full space).

---

## What NOT to do (traps)

1. **Do not render raw residual states.** 1–3 dimensions own the metric; you will be drawing
   the bias term (F8). Centre + standardize, and prove it by showing `λ_1/Σλ`.
2. **Do not plot Euclidean position without normalizing norm.** Norm grows ~4.5%/layer; the
   plot becomes an outward spiral that encodes depth, not computation (F9).
3. **Do not draw one arrow per concept.** Read and write directions differ (F1); the direction
   rotates with depth (F7); some concepts are 2-D cones (F6).
4. **Do not put features at fixed angles (72°, 90°) "because superposition".** 72° is the
   pentagon case only. The correct statement is near-orthogonality with a sparsity-dependent
   polytope (F3).
5. **Do not present the toy-model polytope as the shape of Qwen.** It was measured on a
   `ReLU(WᵀWx−b)` toy model; the authors say it may be idiosyncratic (F3).
6. **Do not draw attention heads and MLPs as the same glyph.** One routes subspaces between
   positions; the other is a key→value lookup that writes a distribution (F10).
7. **Do not imply "one node = one concept" or "one neuron = one feature".** 82% of Sonnet's SAE
   features have no neuron correlated above 0.3; the residual stream has no privileged basis
   (F4, F10).
8. **Do not label a 3-D plot "causal".** A projection is descriptive. The causal badge requires
   a certificate (A6 `certify()`), and no projection can produce one.
9. **Do not compute ID on a few hundred points and call it a dimension.** Ansuini's TwoNN runs
   use O(10⁴) points and the estimator is only validated "very close to the ground truth … when
   this is smaller than ~20"; feed it ≥10³–10⁴ states per layer.
10. **Do not compare cosine similarities across layers without standardizing per layer** — the
    anisotropy changes exponentially with depth (F8), so a "distance" change is confounded with
    a metric change.

---

## Open questions for other agents

- **A5 (UI):** given that A5 currently rejects 3-D "because occlusion, no added information" —
  does a *ribbon + inset* design change that verdict, now that we have measured quantities to
  put on the third axis (ID, lens entropy) and a null (random-init model) to validate against?
  I am not asking A5 to accept 3-D; I am asking that the rejection be re-argued against these
  numbers rather than as a general taste claim.
- **A2 (methods):** can we compute TwoNN ID per layer inside the trace schema cheaply enough to
  ship it as a first-class field? It is the one geometry number that is both cheap and
  informative.
- **A3 (steering):** the read/write distinction (F1) predicts that our diff-in-means vector is a
  *read* direction unless corrected by `Σ⁻¹` (mass-mean / LDA, F7). Does the L8 comment vector
  get stronger under `Σ⁻¹`, and does that change the dose–response?
- **A6 (limits):** please audit F5's "iGPT-L d=1536 ⇒ ID 22–32" as the prior for our model. It
  is the only same-width datapoint and it is an *image* model.
- **A9 (neuron atlas):** the "lobes" result (F4) is a *functional* clustering. Is our per-layer
  activation atlas reproducing lobe structure, or just frequency structure? The null is
  permuted-lobe labels.

---

## SCHEMA CHANGE

Add three optional per-layer fields to the canonical trace schema, all cheap and all needed for
a geometrically honest rendering:

```json
{"layer": 14,
 "resid_norm": 12.3,
 "resid_norm_pctl": 0.41,          // percentile vs a reference prompt set — makes F9 visible
 "id_twonn": 18.7,                // intrinsic dimension of this layer's state set (F5)
 "pc_var_explained": [0.31, 0.12, 0.07],  // top-3, AFTER centring+standardization (F8)
 "lens_entropy": 2.14,            // logit-lens entropy; colour channel for the ribbon
 "attn_entropy": 1.7, "mlp_out_norm": 4.1}
```

Plus a top-level `"geometry": {"standardized": true, "massive_dims_excluded": [n1, n2],
"reference_set": "sha256..."}` so a rendered picture can be audited later.

---

## Build next (ordered, effort)

1. **`code/geometry.py`** — load a cached activation matrix `[N, 1536]` per layer; emit
   `id_twonn`, `pc_var_explained`, eigenvalue spectrum, `λ_1/Σλ`, per-dim mean/std. **~120
   lines, 2 h.** No model needed (test on synthetic Gaussians with known ID and on a
   power-law spectrum).
2. **`code/project.py`** — centre/standardize, PCA to 3, reconstruction error per layer, and a
   `random_init=True` mode that reruns the identical pipeline on `AutoModelForCausalLM(config)`
   (A6 control C5). **~100 lines, 2 h.** The random-init panel is the finding.
3. **`code/trajectory.py`** — per-token `h_0…h_L`, normalized, projected, plus
   `cos(h_l, h_final)` and cross-layer concept-vector cosine. Emits the ribbon geometry as
   JSON, not an image. **~120 lines, 2 h.**
4. **Run on Qwen2.5-Coder-1.5B**: 500 code prompts, one hooked forward each (all 28 layers
   captured in a single pass — do **not** loop the model per layer), last-token + punctuation-token
   states. At A6's measured ~1.2 s/forward that is **500 forwards ≈ 10 min**; repeat on a
   random-init model (+10 min). Sampling ~30 positions per prompt yields ~15,000 points per layer,
   which is the right order for TwoNN. This produces the first *measured* geometry numbers for our
   model and settles F5/F8 for us.
5. **Hand the four JSON artefacts to A5** with the caveat block from F11 attached, and require
   the UI to print variance-explained and the random-init delta on the frame.

**Total: ~6 h engineering + ~30 min compute** (plus model load). No GPU, no new dependencies
beyond numpy/torch.

---

## Geometry cheat-sheet

| property | measured value | source | implication for visualization |
|---|---|---|---|
| ambient width (ours) | 1536 (= 12 heads × 128) | `CONTEXT.md`; `1536/12 = 128` | the "space" has 1536 axes; 3 axes = 0.2% of it |
| effective dimension, CNN | ID ≈ 19 at pool4 vs ED ≈ 10⁵ (2×10⁻⁴); last layer 12–25 | Ansuini 2019, arXiv:1905.12784 | a picture of a 20-d object is a shadow; say so |
| effective dimension, transformer | peak 20 / 25 / 32 (ESM-2 35M/650M/3B); plateau **5–7** | Valeriani 2023, arXiv:2302.00294 | ID is *layer-dependent*; one number per layer |
| effective dimension, d = 1536 | peak **32**, local min **22** (iGPT-L) | Valeriani 2023, Fig. 1 | best available prior for our width; **image** model — `UNVERIFIED` for Qwen |
| effective dimension, LLMs | "O(10)" vs hidden 4096; peak layers ≈6–20 | Cheng et al., ICLR 2025 | 3/20 = 15% of the DOF on screen |
| linear (PCA) vs true dimension | PC-ID ≈ 200 vs TwoNN ≈ 18 (VGG-16) | Ansuini 2019 | PCA/UMAP *systematically* overstate dimension; never equate "PCs shown" with "structure" |
| PCA spectrum shape | no eigenvalue gap ⇒ manifold non-linear | Ansuini 2019 | a smooth ellipsoid is the wrong mental model |
| feature organisation (toy) | vertices of a polytope on the m-sphere; `D_i ∈ {1, ¾, ⅔, ½, ⅖, ⅜, 0}` | Elhage 2022, toy_model | near-orthogonal **directions**, discrete configurations, phase changes |
| specific polytopes seen | digon, triangle, tetrahedron, pentagon, square antiprism, octahedron | Elhage 2022 | these, not a "cloud" or a "brain", are the literal toy shape |
| sparsity needed | superposition appears as density `1−S` falls; pentagon at density 0.05 | Elhage 2022 | the shape is a function of sparsity, not a constant |
| overcompleteness, real LLM | 34M features, <300 active/token, ~12M alive | Templeton 2024 | 8.9×10⁻⁶ density; the code is sparse, not dense |
| feature-neuron relation | 82% of features have no neuron with r > 0.3 | Templeton 2024 | "one neuron = one feature" is false; don't colour by neuron |
| macro-shape, real LLM | power-law eigenspectrum, "fractal cucumber"; slope −0.47 (L12) vs −0.24 (L0), −0.25 (L24) | Li et al. 2025, arXiv:2410.19750 | elongated, tapered — not a ball; middle layers most anisotropic |
| meso-structure, real LLM | functional "lobes": code/math vs English vs dialogue; **954σ / 74σ** above nulls | Li et al. 2025 | the one defensible "brain-like" object; label it as SAE-basis |
| micro-structure, real LLM | parallelograms / trapezoids for one semantic transform | Li et al. 2025 | linear structure is *local* and needs distractors projected out |
| non-1-D features | circles = heptagon/dodecagon (discrete), structured as a **cone** (angle × intensity) | Engels 2024, arXiv:2405.14860 | arrows cannot represent them; 2-D subspaces must be first-class |
| prevalence of non-1-D | 3 of ~500 inspected GPT-2 clusters; 2 of ~2000 Mistral; no published rate | Engels 2024 | rare-but-real; do not claim a percentage |
| read vs write direction | differ by `M = Cov(unembedding)`; Euclidean metric is *not* the causal one | Park 2024, arXiv:2311.03658 | one arrow per concept is a category error |
| concept shape (categorical) | simplex; hierarchy = orthogonal direct sums | Park 2024, arXiv:2406.01506 | render as a tree of simplices on orthogonal axes |
| space / time code | 2-D (lat, long) + 1-D time; R² up to **0.911**; MLP adds ~0 | Gurnee & Tegmark 2023, arXiv:2310.02207 | a plane and a line, **not** a grid; "grid-like" is a conjecture |
| direction stability with depth | rotates: antipodal → orthogonal → aligned | Marks & Tegmark 2023, arXiv:2310.06824 | a fixed 3-D frame is wrong for most layers |
| probe ≠ concept | LR direction distorted by correlated features; `Σ⁻¹θ_mm` fixes it | Marks & Tegmark 2023 | don't label a classifier's normal vector "the concept" |
| anisotropy (GPT-2) | cos(random pair) ≈ 0.6 (L2–8) → ≈ 0.885 at L12 ⇒ **27.7° cone** | Ethayarajh 2019, arXiv:1909.00512; 0.885 from Timkey 2021 Table 1 | raw cosine plots collapse into a jet |
| cause of anisotropy | **1–3** rogue dims; XLNet L11: one dim = **99.6%** of cosine, `E[x]=180` vs −0.084 | Timkey & van Schijndel 2021, arXiv:2109.04404 | standardize; report `λ_1/Σλ` |
| outlier activations | up to **10⁵×** larger, input-independent, act as bias terms | Sun et al. 2024, arXiv:2402.17762 | exclude or they *are* your 3-D plot |
| residual norm | ≈ **1.045×/layer** (gpt2-xl) ⇒ 3.4× over 28 layers | Turner (blog); NeurIPS 2025 depth paper | plot unit direction, not position |
| trajectory shape | curved; attractor-like early; acceleration peaks in final third | Fernando & Guitchounts, CCN 2025 (abstract) | a smooth arc, not a straight line; velocity is data |
| sublayer contribution | roughly constant first half, drop at middle, attention drops most | NeurIPS 2025 depth paper | weight the layer axis by measured contribution, not index |
| attention geometry | QK (where) and OV (what) are two **largely independent** rank-≤128 maps; 64/128-d subspaces | Elhage 2021, framework | draw as subspace beams across positions, not as nodes |
| MLP geometry | key → value; value is a **distribution over the vocabulary** | Geva 2021, arXiv:2012.14913 | draw as concept→token fans, not concept→concept edges |
| basis status | residual stream has **no privileged basis** (rotation-invariant) | Elhage 2021 | any axis choice is a modelling decision; must be labelled |
