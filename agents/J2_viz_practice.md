# J2 — How LLM internals have actually been visualised, and whether 3D earns its pixels

**Question:** the owner rejected a 2-D heatmap and wants a 3-D "brain". What is honest, what is
decorative, and what has already failed?

**Stance:** I am not anti-3D. I am against *three projected axes pretending to be structure*.
There is exactly one 3-D design that survives the evidence, and it is specified at the end.

**Citation hygiene.** Every quote below was fetched in this session (arXiv abs pages, Distill,
transformer-circuits, PLOS, GitHub raw, OpenAlex, publisher PDFs) except where marked
`[secondary]`. Two brief items did not resolve:
- "Crisan et al. 2022, *Interactive tools for ML*" — no such paper could be resolved. The closest
  verified work is **Crisan, Drouhard, Vig, Rajani, "Interactive Model Cards", FAccT 2022**
  ([arXiv:2205.02894](https://arxiv.org/abs/2205.02894), doi:10.1145/3531146.3533108). Cited instead.
- "LIDA" — Microsoft's `lida` is LLM→chart generation, not interpretability. The tool meant is
  **LIT**, [arXiv:2008.05122](https://arxiv.org/abs/2008.05122). A5 flagged the same thing.

---

## TL;DR (5 bullets)

- **The serious tools are all 2-D, and not by accident.** In a 2-D scatter exactly two axes are
  projected, so the lie has a fixed size and a visible label. In 3-D, three axes are projected and
  the third one is where "depth" silently becomes "meaning" in the viewer's head.
- **Every 3-D LLM visualisation I could verify draws *architecture* or *embeddings*, never a
  mechanism claim.** The best-known one (llm-viz) renders a tiny A/B/C-sorting toy network; the
  peer-reviewed assessment of it says it "may overwhelm beginners"
  ([Cho et al., CHI '26](https://arxiv.org/abs/2408.04619)).
- **Projection distortion is measured, not hypothetical.** 2-D UMAP/t-SNE loses **>70 %** of
  nearest neighbours and gives **≤0.4** rank correlation for global structure; distance ratios are
  inflated **4–200×** ([Chari & Pachter 2023](https://doi.org/10.1371/journal.pcbi.1011288)).
  t-SNE cluster *sizes* and inter-cluster *distances* mean nothing
  ([Wattenberg et al. 2016](https://distill.pub/2016/misread-tsne/)). 3-D does not fix this;
  it removes the cue that would let a viewer notice.
- **HCI says 3-D on a flat screen helps *shape* and hurts *relative position***
  ([St John et al. 2001](https://doi.org/10.1518/001872001775992534)). Our questions are relative
  position ("did L14 move toward `int` before L20?"). The 3× benefit in the classic 3-D graph study
  required **head-coupled stereo**, not a laptop ([Ware & Franck 1996](https://doi.org/10.1145/234972.234975)).
- **3-D is defensible in exactly one configuration:** layer index on the depth axis (a *true*
  ordered axis, not a projection), one **shared 2-D basis** in the x–y plane, a mandatory 2-D
  unroll beside it, numeric hover, no travelling particles, no bloom. That buys one real thing —
  **20–60 token trajectories × 28 layers in a single view**, which 2-D needs small multiples for.
  Measured on this box: 500 instanced nodes + 2 000 edges + 4 000 particles = **0.11 ms/frame**;
  the killers are per-node draw calls (**30.2 ms**) and additive overdraw (**25.7 ms**).

---

## Findings

### F1. The landscape: what exists, at what dimensionality, with what projection
[OBS — tool docs/repos fetched 2026-11-11; licences as reported by the sources]

| Tool | Source | Dim | Projection | What it actually shows | Honest reading |
|---|---|---|---|---|---|
| **OpenAI Microscope** (released **2020-04-14**; the brief's "2019" is wrong — and it is *not* an activation atlas) | [openai.com/index/microscope](https://openai.com/index/microscope/) | 2-D **grid** (not a scatter) | none | "a collection of visualizations of every significant layer and neuron of eight vision model organisms"; every neuron is linkable and shareable | a *browse index* of neuron cards. No geometry is claimed, so nothing is distorted. |
| **Activation Atlas** | [Carter et al., Distill 2019](https://distill.pub/2019/activation-atlas/) | 2-D | UMAP (or t-SNE) → grid → feature-inversion icons | what *directions* in activation space look like, aggregated over 10⁶ images | still the best model for "make the projection visible as icons". Section "doesn't surface compositionality" is its own confession (see F3). |
| **Embedding Projector** | [Smilkov et al., arXiv:1611.05469](https://arxiv.org/abs/1611.05469) | 2-D / 3-D | PCA, t-SNE, UMAP | "a tool for interactive visualization and interpretation of embeddings" | projection browser, no semantics attached. The 3-D mode is `UNVERIFIED` here (UI text not fetchable; docs page no longer mentions it). |
| **Neuronpedia** | [README](https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/README.md) | dashboards + embedding explorer | UMAP (`umap` is a listed feature) | SAE feature cards, cos-sim, dashboards, attribution graphs | the embedding explorer is a *navigation* surface for features; feature identity comes from the card, not the position. |
| **circuit-tracer / attribution graphs** | [Ameisen et al., Transformer Circuits 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | 2-D node–link, layered by depth | **none** — layout is layer columns + dagre, not a projection | per-prompt causal graph of features/supernodes with edge weights | the only "brain-like" artefact with a causal claim behind it — and its layout is deliberately *not* a projection. |
| **Transformer Explainer** | [Cho et al., arXiv:2408.04619](https://arxiv.org/abs/2408.04619) | 2-D | none | GPT-2 running live in the browser; data-flow diagram + on-demand maths | 90-participant study: "significant advantages in improving user understanding and engagement". Evidence that **interactive 2-D + animation works for process**, not that 3-D works for data. |
| **BertViz** | [Vig, arXiv:1904.02679](https://arxiv.org/abs/1904.02679) | 2-D | none | "attention at three levels of granularity: the attention-head level, the model level, and the neuron level" | the head/neuron views are the origin of the spaghetti problem (F6). |
| **AttentionViz** | [Yeh et al., arXiv:2305.03210](https://arxiv.org/abs/2305.03210) | 2-D (3-D optional) | joint query/key embedding (UMAP/PCA) | global attention patterns across many inputs | docs: "Scatterplots can also be viewed in 2D or 3D." Default is 2-D. Published 3-D exists; nobody's headline figure is the 3-D one. |
| **llm-viz** | [github.com/bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) | **3-D** (WebGPU) | none — 3-D is the *layout*, not a projection | "3D interactive model of a GPT-style LLM network running inference"; the demo model "sorts a small list of the letters A, B, and C"; GPT-2 "weights aren't downloaded" | architecture walkthrough. Beautiful, pedagogically effective, and **zero activation data**. |
| **LIT** | [Tenney et al., arXiv:2008.05122](https://arxiv.org/abs/2008.05122) | 2-D panels + projector | PCA/UMAP for the embedding panel | salience, counterfactuals, datapoint explorer | the "many small 2-D panels" pattern; no 3-D. *(Link verified via search index only; abstract not fetched — `UNVERIFIED` on details, `[INF]` on the panel claim.)* |

**What the table says.** The field converged on 2-D for *evidence* and on 3-D only for
*architecture* (llm-viz) or *optional embedding rotation* (AttentionViz). The one place with a
causal claim (attribution graphs) uses a **layered 2-D node–link layout with no projection at all** —
position means layer and rank, nothing is embedded. That is the design pattern to copy if the goal
is mechanism.

---

### F2. What actually communicates mechanism: four patterns that survived
[OBS from the tools above; INF in the "why"]

1. **Ordered axes that are real.** Attribution graphs put layer on one axis and let layout do the
   rest; no claim that x–y distance means anything. BertViz's model view does the same.
2. **A small, fixed vocabulary repeated everywhere.** Tokens as text, one scalar per pixel, one
   channel at a time. Every successful tool does this; none of them encode four variables in one
   glyph.
3. **Numbers on demand.** The card/table is where interpretation happens; the picture is where
   *navigation* happens. Neuronpedia, Microscope and LIT all follow this.
4. **Process animation, not data animation.** Transformer Explainer animates *data flow through a
   known architecture* — a truthful animation because the diagram is the ground truth. Travelling
   particles over an inferred graph are the opposite: the animation asserts a path that was never
   measured.

[INF] Corollary for us: animation is legitimate when it depicts **the sequence of forward passes**
(one token = one pass), and illegitimate when it depicts **information moving between features**.

---

### F3. Projection fidelity: the numbers a viewer must be told
[OBS — quotes fetched; the two source papers are the strongest available]

**Chari & Pachter, "The specious art of single-cell genomics", PLOS Comput Biol 19(8):e1011288, 2023**
([doi](https://doi.org/10.1371/journal.pcbi.1011288)) — the most quantitative critique available:

- "extreme dimension reduction, from hundreds or thousands of dimensions to 2, inevitably induces
  significant distortion of high-dimensional datasets."
- Neighbour retention, 2-D UMAP/t-SNE vs ambient space: "Jaccard distances ... with an average
  consistently above **0.7 (70%)**" — i.e. **less than 30 % of a point's 30 nearest neighbours
  survive**.
- Global structure: "correlation of cell type neighbor rankings to that of the ambient space were
  low (**≤ 0.4**)".
- Distances: "**4- to 200-fold increases**" in max/min distance ratios. "quantitatively distinct
  groups ... displayed the same dispersion patterns, violating distance preservation".
- The killer demonstration: they fit data to an **elephant-shaped** embedding ("Picasso") whose
  inter/intra-cluster correlations are "comparable performance to t-SNE and UMAP". The shape of the
  picture is a free parameter.
- Their recommendation: "such 2D spaces should be treated more as **cartoon diagrams** to be
  displayed post-analysis."

**Wattenberg, Viégas, Johnson, "How to Use t-SNE Effectively", Distill 2016**
([doi:10.23915/distill.00002](https://distill.pub/2016/misread-tsne/)):

- "you cannot see relative sizes of clusters in a t-SNE plot."
- "distances between well-separated clusters in a t-SNE plot may mean nothing."
- "Random noise doesn't always look random" — at low perplexity, 500 points drawn from a unit
  Gaussian in 100-D "seems to show dramatic clusters".

**UMAP's own paper** hedges: it "arguably preserves more of the global structure"
([McInnes et al., arXiv:1802.03426](https://arxiv.org/abs/1802.03426)) — *arguably* is doing a lot
of work there. The practical rule is the same as t-SNE's: **local neighbourhood is a weak claim,
global geometry is not a claim at all.**

**Activation Atlas confesses the same limits in its own "Technical Notes":**
- "they ... have to **collapse local structure down to two dimensions**";
- "**No alignment between layers.** There's no way to link two views together. For instance, when
  looking at two layers side-by-side, similar features appear in random locations.";
- "these paths are **constructed after the fact** in the low-dimensional projection ... we don't
  necessarily know how the paths operate in the original higher-dimensional activation space";
- "the final output can be **very sensitive to the hyperparameters**".

**And for us specifically:** Friedman et al., "Interpretability Illusions in the Generalization of
Simplified Models" ([arXiv:2312.03656](https://arxiv.org/abs/2312.03656)) show the exact failure we
would ship: "even if the simplified representations can accurately approximate the full model on
the training set, they may fail to accurately capture the model's behavior out of distribution."
A UMAP map fitted on Python prompts is an in-distribution object; the moment the user types Rust or
a stack trace, the map is decoration.

**Minimum disclosure a viewer must get** (non-negotiable, on screen, not in a tooltip):
1. which algorithm and how many components (`PCA, 16 comps`) and **variance explained**;
2. that **x–y distances are not semantic** and clusters are not categories;
3. the **sample** the projection was fitted on, and that new inputs are OOD;
4. that changing hyperparameters changes the picture;
5. for UMAP/t-SNE: cluster size and inter-cluster distance are meaningless.

---

### F4. 3-D specifically: what exists, what it showed, and whether it helped
[OBS + INF; this is the section the owner should read]

**What published 3-D transformer visualisations actually show.**

| Artefact | 3-D used for | Activation data? | Evidence it helped? |
|---|---|---|---|
| llm-viz ([repo](https://github.com/bbycroft/llm-viz)) | spatial layout of the *architecture* (matrices as blocks, one forward pass) | no | none published; 5.6k stars, and CHI '26 says it "may overwhelm beginners" |
| AttentionViz 3-D mode | rotating a joint query/key embedding | yes (Q/K vectors) | docs confirm "Scatterplots can also be viewed in 2D or 3D"; the paper's user study evaluates the tool as a whole — I found no 3-D-vs-2-D comparison |
| Embedding Projector 3-D | rotating PCA/t-SNE/UMAP of embeddings | yes | no controlled study found; the known misreadings (Wattenberg) apply unchanged |
| Patch Explorer (diffusion) — cited in Cho et al. as "an interactive 3D heatmap" | 3-D heatmap of internal representations | yes | one system paper; no 3-D-vs-2-D comparison found (`[secondary]` — I did not read it) |
| hobby "3-D LLM" sites | glow, depth, wow | usually none | — |

**I could not find a peer-reviewed 3-D visualisation of transformer *circuit/mechanism* structure.**
Searches run: "3D transformer attention visualization paper", "3D UMAP transformer layers neurons",
"immersive VR interpretability LLM", plus the reference lists of Cho et al. 2026 and Yeh et al. 2023.
**Falsifier:** a paper showing a 3-D view of an attribution graph or circuit that produced a
mechanistic claim. If one exists, this paragraph is wrong. Until then the honest statement is:
*the field's causal artefacts are 2-D, and the field's 3-D artefacts are educational.*

**The HCI evidence, which is more useful than the tool survey.**

- **St John, Cowen, Smallman & Oonk 2001**, *Human Factors* 43(1):79–98
  ([doi](https://doi.org/10.1518/001872001775992534)): "The distortions inherent in 3D displays
  hamper judging relative positions, whereas the integration of dimensions in 3D displays
  facilitates shape understanding." Six experiments, block shapes then natural terrain.
  → **Our task is relative position, not shape.** "Is the L14 state closer to `int` than the L20
  state?" is exactly the task 3-D is documented to hurt.
- **Ware & Franck 1996**, *ACM TOG* 15(2):121–140 ([doi](https://doi.org/10.1145/234972.234975)):
  "head-coupled stereo viewing can increase the size of an abstract graph that can be understood by
  a factor of **three**; using stereo alone provided an increase by a factor of **1.6** and head
  coupling alone produced an increase by a factor of **2.2**." And: "This asymptote is lower for 2D
  viewing, presumably, because the graph appears as a **hopeless jumble** sooner than it does for
  the 3D viewing modes."
  → This is the strongest pro-3-D result in the literature, and **it required head-tracked stereo**.
  Read the conditions carefully, because they matter: their "2D" baseline was *not* a 2-D layout —
  it was "No stereo, no rotation; the 3D graph was projected onto a 2D plane." In other words their
  worst condition **is** the thing a laptop shows: the same 3-D layout, flattened, with no depth
  cues. Every improvement came from adding stereo and/or motion to that view. A laptop browser has
  *motion* (we can rotate) but not stereo or head coupling, so we inherit at most the motion
  component — and Ware's motion condition was *hand-guided or automatic rotation*, i.e. the graph
  must actually move. **A static 3-D view with no stereo and no motion is the measured-losing
  condition, not a neutral one.**
- **Transformer Explainer** ([arXiv:2408.04619](https://arxiv.org/abs/2408.04619)): a
  "90-participant user study showed that our tool offered significant advantages in improving user
  understanding and engagement" — with a **2-D** data-flow diagram and animation. Interactive 2-D is
  not the weak option people assume.

**Verdict on 3-D as a category:** defensible only if (a) at least one axis is *real and ordered*,
(b) the thing moves, and (c) there is a 2-D view of the same data one click away. A 3-D scatter of
1536-D activations satisfies none of the three.

---

### F5. What users actually misunderstand (interp-tool user studies)
[OBS — all four studies fetched]

- **Kaur et al., CHI 2020**, "Interpreting Interpretability"
  ([PDF](https://harmanpk.github.io/Papers/CHI2020_Interpretability.pdf)): contextual inquiry
  (N=11) + survey (N=197) of data scientists using GAMs and SHAP. Result: "data scientists
  **over-trust and misuse** interpretability tools. Furthermore, **few of our participants were
  able to accurately describe the visualizations** output by these tools."
  → The failure is not "the chart was wrong"; it is "the viewer formed a confident wrong model of
  what the chart encoded". A 3-D view maximises the space in which that can happen.
- **Poursabzi-Sangdeh et al., CHI 2021** ([arXiv:1802.07810](https://arxiv.org/abs/1802.07810)),
  N=3,800 pre-registered: "showing participants a clear model meant that they were **less able to
  detect and correct for the model's sizable mistakes, seemingly due to information overload**."
  → More visible internals *reduced* error detection. This is the direct empirical argument against
  "show everything, in 3-D, at once."
- **Hohman et al., IEEE TVCG 2019** ([arXiv:1801.06889](https://arxiv.org/abs/1801.06889)): the
  field survey; its organising frame is the Five W's — i.e. the questions are *who is looking, at
  what, to do what*. There is no "which dimension" answer independent of the task.
- **Crisan et al., FAccT 2022** ([arXiv:2205.02894](https://arxiv.org/abs/2205.02894)): 30
  participants, think-aloud, on interactive vs static model documentation; identifies
  "understandability & interpretability" as a design dimension, not a property of a chart.
- **Bolukbasi et al. 2021** ([arXiv:2104.07143](https://arxiv.org/abs/2104.07143)): "Activations of
  individual neurons ... may **spuriously appear to encode a single, simple concept**, when in fact
  they are encoding something far more complex." The illusion is generated by the *viewer's*
  geometry + a narrow corpus. A prettier rendering does not remove it; it strengthens it.

[INF] The through-line: **visual polish increases trust faster than it increases understanding.**
That is the exact failure mode a "3-D brain" optimises for.

---

### F6. The attention-weight trap: why "structure" drawn from attention is misleading
[OBS — three independent results]

1. **Jain & Wallace 2019** ([arXiv:1902.10186](https://arxiv.org/abs/1902.10186)): "learned attention
   weights are frequently uncorrelated with gradient-based measures of feature importance, and one
   can identify **very different attention distributions that nonetheless yield equivalent
   predictions**." → an arc diagram of attention is compatible with many different mechanisms.
2. **Abnar & Zuidema 2020** ([arXiv:2005.00928](https://arxiv.org/abs/2005.00928)): "across layers
   of the Transformer, information originating from different tokens gets increasingly mixed. This
   makes **attention weights unreliable as explanations probes**." They propose attention rollout /
   flow as the *minimum* correction. → drawing raw single-layer attention as "flow" is wrong even
   descriptively, because residual connections bypass it.
3. **Voita et al. 2019** ([arXiv:1905.09418](https://arxiv.org/abs/1905.09418)): "pruning **38 out of
   48** encoder heads results in a drop of only **0.15 BLEU**." And **Clark et al. 2019**
   ([arXiv:1906.04341](https://arxiv.org/abs/1906.04341)): heads mostly attend to "delimiter tokens,
   specific positional offsets, or broadly attending over the whole sentence, with **heads in the
   same layer often exhibiting similar behaviors**."
   → a 28×12 = 336-node head grid overstates the number of *distinct things happening* by roughly
   an order of magnitude. It is a navigation widget, not a map of computation.

**For our model specifically:** Qwen2.5-1.5B has 12 query heads but **2 KV heads**; heads in a KV
group share K and V, so head-level stories are even weaker than usual (A5 F6). Any 3-D edge
rendering of attention must (a) be a single head, (b) be labelled `[OBSERVED: attention weights,
correlational]`, (c) never be the default layer of the scene.

---

### F7. What 3-D can honestly add — exactly one thing
[INF, with the reasoning exposed]

The residual stream is `1536`-D. The layer index is `0..27`. Those are different kinds of object:
one is a high-dimensional vector, the other is **a genuine 1-D ordered coordinate**. Putting layer
on the depth axis is therefore **not a projection** — it is an axis we actually have. That leaves
only x–y projected, which is the *same* epistemic situation as a 2-D scatter, not a worse one.

What that buys: **many token trajectories in one view.** At 20 generated tokens the scene holds
`20 × 28 = 560` points. In 2-D you can show one token's layer-trajectory honestly, or 20 of them as
small multiples. In 3-D you can show all 20 and let the viewer see which trajectories converge at
L18 and which fork at L24. That is a real, checkable visual question, and it is the only one 3-D
answers better.

What it does **not** buy: any statement about geometry *within* a layer. That is still a projection
with ≥70 % neighbour loss and meaningless distances (F3), now harder to notice because the third
axis makes the whole thing look like a measured space.

**Hard rule:** the moment a design puts *three projected axes* on screen (i.e. UMAP-3D of the
activation space), it has thrown away the only honest axis it had. Do not ship that.

---

### F8. Performance: measured on this machine, not estimated
[OBS — `google-chrome --headless=new`, WebGL2, 1280×720, `gl.finish()` per frame, CPU pinned at
**799 999 kHz** (`scaling_cur_freq`), `no_turbo=1`; scripts in `/tmp/opencode/j2/bench.html`,
`bench2.html`]

Two backends, because "no GPU acceleration guarantee" is the actual requirement:
- **SwiftShader** (software rasteriser — this is what a machine without GPU accel gets, and what
  headless Chrome picks by default): `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))`.
- **Intel UHD 620** via ANGLE/GL: `ANGLE (Intel, Mesa Intel(R) UHD Graphics 620 (WHL GT2), OpenGL ES 3.2)`.
  (`glxinfo`: "Accelerated: yes", OpenGL 4.6 — the iGPU is real, but the CPU/package is pinned to
  800 MHz, so both sides are power-starved.)

Scene: 500 instanced cubes (36 tris each) + 2 000 line segments (one draw) + 4 000 points (one draw).

| Test | SwiftShader | Intel UHD 620 |
|---|---|---|
| **instanced**: 500 nodes + 2 000 edges + 4 000 points, one draw call each | **0.11 ms** avg / 0.20 ms p95 (≈9 000 fps) | **0.11 ms** avg / 0.30 ms p95 |
| **bigpoints**: 4 000 additive 64 px sprites (≈16.4 M fragments, **17.8×** screen overdraw) | **13.66 ms** / 30.3 p95 (73 fps) | **25.71 ms** / 44.9 p95 (**39 fps**) |
| **naive**: 500 separate draw calls, buffer recreated per node | **15.28 ms** / 30.2 p95 (65 fps) | **30.20 ms** / 43.6 p95 (**33 fps**) |

Main-thread costs (same box, headless Chrome):

| Test | Result |
|---|---|
| 500 DOM `textContent` updates (one frame's worth of a label grid) | **3.06 ms** avg / 6.00 ms p95 |
| 500×8 `Float32Array` writes | 1.00 ms |
| `JSON.parse` of a 3 KB 28-layer token payload | 0.18 ms |

**Arithmetic, shown:** `4 000 sprites × 64 px × 64 px = 16.38 M fragments`; `1280 × 720 = 0.92 M px`;
`16.38 / 0.92 = 17.8×` overdraw. That, not node count, is what costs 25 ms.

**Conclusions.**
1. **Geometry is free.** 500 nodes is three orders of magnitude below the budget; a 560-point
   trajectory scene at 60 fps is limited by JS/DOM (~3 ms) and compositing, not by the GPU.
   (Caveat: `performance.now()` is coarsened to ~0.1 ms in Chrome, so the 0.11 ms figure really
   means "at or below timer resolution" — the scene is free, not 9 000 fps.)
2. **Draw calls are not free.** 500 separate draws *with per-draw buffer churn* cost 30.20 ms on
   the iGPU (that is an upper bound: a real per-node mesh would keep its geometry). Even without the
   churn, per-object overhead is the item that scales with node count, so use instancing or merge:
   in react-three-fiber that means one `<instancedMesh>` + one `<lineSegments>` + one `<points>`,
   not 500 `<mesh>` components. (R3F reconciliation is CPU-side, and this CPU is at 800 MHz.)
3. **Additive glow is the real budget item.** `17.8×` overdraw with blending costs 26 ms; a bloom
   pass at 1080p would cost more. **No bloom, no DOF, no volumetric fog.**
4. **Software fallback is viable for the base scene** (0.11 ms) — so ship a "2-D/low" mode triggered
   by `chrome://gpu` reporting SwiftShader, and keep the same data.
5. Caveat: headless + `gl.finish()` measures GPU completion without browser compositing; treat these
   as relative costs, not absolute frame times. `[OBS]` for the numbers, `[INF]` for the 60 fps claim.
6. Caveat on the conditions: these ran with **no model loaded**. With `Qwen2.5-Coder-1.5B` resident,
   memory-bandwidth and cache contention will slow the browser thread; re-run the benchmark inside
   the real app before trusting the margins. `[OBS]` for the isolation numbers, `[INF]` for the
   co-resident case.

---

### F9. "Brain" is the actual problem, not the axis count
[INF]

The request is for a *brain*. A brain implies: regions with functions, pathways between them, and
something homuncular doing the reading. None of the three exists in the data:
- **no regions** — superposition means a "region" is a direction we chose (A6 F1, and
  [Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html));
- **no pathways** — the forward pass is a fixed DAG of matrix multiplies; arrows between *features*
  are an inference, and even the field's best version reports "cosine-faithfulness ... around
  **60–80 %**" one layer after an intervention, with "compounding errors [that] can have a
  **catastrophically detrimental effect** on the magnitude of perturbations"
  ([Ameisen et al. 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html));
- **no reader** — the tool is the reader, and F5 says the reader over-trusts.

Also from that same paper, about its own graphs: pruned graphs shrink by "~10×" in nodes and
"~500×" in edges, the pruning thresholds are "**arbitrary** and ... chosen manually", and related
features "represent subtly different concepts that are **hard to discern from our visualization
tools**." That is the state of the art saying its own picture is lossy.

**Therefore:** a "3-D brain" can be shipped as a *navigation metaphor* (a column of layers you can
orbit) but never as the *evidence panel*. The label on the tab matters more than the renderer.

---

## Validation blocks

```
HYPOTHESIS: A 3-D view whose three axes are all projections (UMAP-3D of activations) misleads more
            than a labelled 2-D projection, because viewers read depth as structure.
TEST:       Literature check — Chari & Pachter 2023 (PLOS Comp Biol) for the distortion magnitudes;
            Wattenberg et al. 2016 for t-SNE misreadings; St John et al. 2001 for 3-D vs 2-D task
            performance. Fetched and quoted above.
RESULT:     2-D already loses >70% of nearest neighbours and ≤0.4 global rank correlation; 3-D
            displays "hamper judging relative positions"; t-SNE cluster size/distance are meaningless.
VERDICT:    CONFIRMED for the components; UNVERIFIED for "3-D misleads *more* than 2-D" as a direct
            head-to-head on interp viewers (no such study found).
FALSIFIER:  A user study in which viewers of a 3-D activation projection make fewer false mechanistic
            claims than viewers of the same data in 2-D.
```

```
HYPOTHESIS: No peer-reviewed 3-D visualisation of transformer circuit/mechanism structure exists.
TEST:       Searches (3-D transformer attention visualization; 3-D UMAP transformer; immersive VR
            interpretability), plus reference lists of Cho et al. 2026 and Yeh et al. 2023.
RESULT:     3-D artefacts found are architecture walkthroughs (llm-viz), optional embedding rotation
            (AttentionViz), and a diffusion 3-D heatmap (Patch Explorer). No circuit/attribution
            graph in 3-D.
VERDICT:    UNVERIFIED (absence of evidence, searched not exhaustive).
FALSIFIER:  Any paper with a 3-D circuit/attribution visual that supports a mechanistic claim.
```

```
HYPOTHESIS: 500 nodes + 2 000 edges + animated particles can hold 60 fps on this laptop.
TEST:       WebGL2 microbenchmark, 1280x720, instanced vs per-object; SwiftShader and Intel UHD 620.
RESULT:     0.11 ms/frame instanced on both backends; 30.20 ms for 500 separate draws; 25.71 ms for
            4 000 additive 64px sprites (17.8x overdraw).
VERDICT:    CONFIRMED for instanced geometry; REFUTED for per-object meshes and additive glow.
FALSIFIER:  A sustained 60 fps capture of 500 independent meshes with glow on this machine.
```

```
HYPOTHESIS: Attention weights are a misleading basis for drawing "structure".
TEST:       Jain & Wallace 2019, Abnar & Zuidema 2020, Voita et al. 2019, Clark et al. 2019.
RESULT:     Different attention distributions yield equivalent predictions; residual mixing makes raw
            attention unreliable; 38/48 heads prunable for 0.15 BLEU; same-layer heads behave similarly.
VERDICT:    CONFIRMED.
FALSIFIER:  A task where permuting attention weights changes the prediction, or where head-level
            structure predicts ablation effect better than chance.
```

---

## Concrete implementation (the only projection we should compute live)

**Why PCA and not UMAP, for the live view.** UMAP/t-SNE learn a *dataset-specific* embedding; there
is no principled way to place a single new point from a single generated token into a UMAP fitted
elsewhere (the transform is approximate and unstable for out-of-sample points). PCA is a fixed linear
basis: `x = h @ B[:, 0]`, `y = h @ B[:, 1]`, exact, order-independent, and cheap.

```python
# Fit ONCE per session (a corpus of prompts), not per token.
# H: (N, 1536) residual states at the chosen layer, float32
mu = H.mean(0)
B = torch.pca_lowrank(H - mu, q=16).V[:, :2]        # (1536, 2)

# Per token, per layer: 28 x 1536 @ 1536 x 2 = 86k MACs -> free
xy = ((h - mu) @ B)                                  # (28, 2)
```

**Diagnostics that must be computed and shown** (all cheap):
- `var_explained = (S[:2]**2).sum() / (S**2).sum()` — display as a percentage on the axis label.
- `corr(pc1, layer_index)` over the pooled `(layer, token)` states. **If `|r| > 0.7`, the projection
  is mostly re-drawing the depth axis** — refuse to render it and say so. (This is the single most
  likely failure of the design, and it is a 5-line check.)
- `trustworthiness`/`continuity` on a held-out sample if anyone asks for a number; otherwise the
  neighbour-retention caveat is enough.

**Two basis choices, both defensible, pick one per session:**
1. **Final-layer basis** — fit on layer-27 states, project every layer into it. Cross-layer
   comparability is meaningful *relative to the output basis*; early layers will look collapsed.
2. **Pooled basis** — fit on all `(layer, token)` states. Better spread; but check the
   `corr(pc1, layer)` diagnostic above.

**Never:** UMAP-3D; per-layer independently fitted projections (Activation Atlas: "No alignment
between layers ... similar features appear in random locations"); force-directed layout of anything.

---

## What NOT to do (traps)

1. **Don't render a 3-D UMAP/PCA of the activation space as the hero panel.** Three projected axes,
   zero honest axes, and the third one is read as meaning.
2. **Don't draw flowing particles along edges.** Fill-rate cost measured above (26 ms) and it
   asserts information flow that was never measured.
3. **Don't use "brain" in the UI copy, tab title, or README.** It sets the interpretive frame before
   a single number is read (F9). "Layer column" or "trajectory view" are accurate and still look good.
4. **Don't make attention the edges of the 3-D scene.** F6: single head, on demand, in a 2-D strip.
5. **Don't animate the camera while data is streaming.** The viewer cannot tell whether the scene
   moved or the state moved. Camera motion is user-initiated only.
6. **Don't auto-rotate when idle.** Motion implies activity; an idle spin reads as "the model is
   computing".
7. **Don't map "importance" to size, colour, or glow.** There is no defensible importance scalar
   (A5 F7); use measured quantities with names (`norm`, `lens top-1 prob`, `⟨h,v⟩`).
8. **Don't put >5 000 points on screen without LOD** — window to the last N tokens instead.
9. **Don't ship without the 2-D unroll.** The 3-D view is an alternate projection of the same data;
   if the 2-D version is not good enough to stand alone, the 3-D version is hiding something.
10. **Don't let 3-D be the only path to any number.** Every value visible in 3-D must be readable as
    text (hover card + table view). This is also the accessibility path (WCAG 1.4.1).

---

## Open questions for other agents

- **A5 / H3:** does the existing `app/BRAIN_SPEC.md` (mode 1: 7×4 serpentine layer grid + particles)
  become the *2-D unroll* of the trajectory view, or is it replaced? Its particle animation violates
  trap 2 and its "neuron sparks" are per-neuron claims we cannot support (A6 F1, A5 F3). My
  recommendation: keep the serpentine as the 2-D unroll, drop the sparks, drop the travelling
  particles, and add layer-trajectory polylines.
- **A2:** what does PCA-2 of the residual stream actually look like at L27 on a code corpus — what
  fraction of variance is token identity? Needed to decide final-layer vs pooled basis, and whether
  `|corr(pc1, layer)| > 0.7` kills the pooled option outright. `UNVERIFIED — needs runtime`.
- **A4:** can the backend compute and ship 28×2 float32 projection coords per token (224 B) plus a
  `basis_id`, without touching the 40 KB payload cap? Arithmetic says yes.
- **A6:** should the 3-D panel carry a permanent `[DESCRIPTIVE — projection]` chip, and is a passing
  certificate required before any "this is how the model computes" wording anywhere near it?
- **Orchestrator:** is the goal *comprehension* (then Transformer Explainer-style 2-D process
  animation is the evidence-backed choice) or *impact in a demo* (then 3-D wins and we should say so
  honestly and keep it out of the evidence panel)?

---

## SCHEMA CHANGE

Additive; v1 consumers ignore unknown keys.

```json
{"step": 37,
 "projection": {
   "basis_id": "pca27-code-v1",          // identifies the fitted basis
   "algo": "pca", "comps": 2, "var_explained": 0.41,
   "pc1_layer_corr": 0.12,              // diagnostic; >0.7 means "this is just depth"
   "fit_corpus": "python-prompts-v1", "n_fit": 2048
 },
 "layers": [{"layer": 14, "resid_norm": 12.3,
             "xy": [-0.83, 1.42],        // 2 floats, in the session's basis
             "logit_lens_topk": [[1437, "int", 0.71]],
             "attn_entropy": 1.71}]}
```

Cost: 2 floats × 28 layers = 224 B/token; the projection itself is 86 k MACs/token.
Rationale: the UI must never fit a projection client-side (it would need the 1536-D vector, 172 KB
per token) and must never silently fit a *different* basis on a different token — otherwise two
tokens are not in the same space and the trajectory is a lie.

---

## Build next (ordered)

| # | step | why first | effort |
|---|---|---|---|
| 1 | **PCA basis + diagnostics script** (`var_explained`, `pc1_layer_corr`) on 2 048 prompt residuals | kills or blesses the whole design in one number | 0.5 d |
| 2 | **2-D unroll first**: layer on y, PC1 on x, one polyline per token | if this is unreadable, 3-D will not save it | 1 d |
| 3 | **3-D column**: instanced points + one `lineSegments`, orbit/zoom, polar clamp, layer axis ticks | the honest version of the request | 1.5 d |
| 4 | **Hover card + table view** (all numbers as text) | the only place interpretation is allowed | 0.5 d |
| 5 | **Perf gate**: FPS counter + `gl.finish()` timer in CI-ish harness; auto-fallback to 2-D on SwiftShader | prevents the 39 fps surprise in the demo | 0.5 d |
| 6 | **Honesty chrome**: `[DESCRIPTIVE]` chip on the 3-D panel, projection disclosure block, banned-words lint | retrofits cost 3× | 0.5 d |
| 7 | *(optional)* layer-band isolation + cross-section plane | readability at 60 tokens | 1 d |

Total ≈ 5.5 agent-days, of which 1.5 are the renderer. **Nothing here needs WebGL2 features beyond
instancing.**

---

## 3D viewer spec recommendations

**One sentence:** use 3-D to show *layer order* and *many token trajectories at once*, never to show
activation geometry.

### Coordinate system
- **z = layer index, 0 at bottom → 27 at top.** True axis, ticks every layer, label every 4th.
  This axis is not a projection and must be visually distinct (solid axis line, not implied by position).
- **x–y = one shared PCA basis for the whole session**, fitted once, same for all tokens and layers.
  2 components, `var_explained` printed on the axis label.
- **Forbidden:** 3-D UMAP; independently fitted per-layer projections; any "free" third dimension.

### Glyphs (one channel per visual variable)
| channel | encodes | notes |
|---|---|---|
| position z | layer | exact |
| position x–y | 2-D PCA of the residual state | projection; distances not meaningful |
| radius | `resid_norm` (normalised per session) | hover prints the number |
| colour | one selected channel: `lens_prob(chosen)` \| `entropy` \| `⟨h,v⟩` \| `mlp_norm` | one at a time, cividis, legend always visible |
| polyline | that token's 28 states, in layer order | the actual object of interest |

### Interactions, ranked by value
1. **Orbit (drag) + zoom (wheel)**, polar angle clamped to ±70°, no roll, no free-fly, `R` resets.
   Motion is the only depth cue a laptop has (F4); make it available but user-initiated.
2. **Hover a point → isolate that token's trajectory** (others ghost to 15 % opacity) + numeric card:
   `L14 · norm 12.3 · top-1 "int" 0.42 · entropy 1.71 · ⟨h,v⟩ 1.83`.
3. **Layer band isolation** (drag two handles): render `[a,b]`, ghost the rest. Solves occlusion.
4. **Click a point → 2-D cross-section** of that layer with the lens top-k table for that token.
   This is where reading happens; the 3-D view is for finding *where* to read.
5. **2-D unroll toggle** — same data, layer on y, PC1 on x, side by side or replacing. Mandatory.
6. **Freeze basis** toggle (so camera motion is never confused with data motion).
7. **Keyboard:** `←/→` step layer, `Space` pause/step, `Esc` deselect, `F` flip to 2-D.

### Pacing and animation
- Update on **token commit**, not per rAF. A new token adds 28 points + one polyline; fade in ≤200 ms.
- **No travelling particles. No idle rotation. No breathing.**
- If the model is generating, say so with a status line, not with motion in the scene.

### Budget (from the measurements in F8)
- ≤3 draw calls (instanced points, one line-segments object, optional one points object for particles
  — but see the ban below). Target CPU < 4 ms/frame; measured 500-node scene = 0.11 ms GPU.
- **DPR cap 1.5**; no bloom, no DOF, no fog, no shadows.
- **≤60 visible tokens** (1 680 points); beyond that, window to the last 60 or LOD.
- Auto-switch to the 2-D unroll when `chrome://gpu` reports SwiftShader **and** the scene exceeds
  ~2 000 points (the base scene is fine in software; the heavy one is not).

### Disclosure chrome (always visible, not hover)
```
3-D trajectory view · DESCRIPTIVE
z = layer (exact) · x/y = PCA of residual stream, 41% variance · fitted on python-prompts-v1 (n=2048)
Distances and clusters in x/y are not meaningful. New inputs may be out of distribution.
[2-D view] [evidence chip: DESCRIPTIVE]
```
Plus: missing layer → **hollow** point + `no data`; intervention active → `STEERING ACTIVE · causal`
on the panel that owns the causal claim, never on the 3-D canvas.

### Ship/kill criteria
Ship only if all four hold:
1. a reader can answer "did this token's prediction move toward `int` before L20?" in the 3-D view
   **and** in the 2-D view, with the same answer;
2. `pc1_layer_corr` < 0.7;
3. the perf gate holds ≥55 fps at 60 tokens on the iGPU and ≥30 fps in software;
4. every number in the scene is also available as text.

---

## What will mislead users

Ordered by how likely it is to produce a *confident wrong belief*, with the mechanism and the fix.

1. **Reading x–y distance or cluster shape as semantics.** Distortion is not small: >70 % of nearest
   neighbours lost, ≤0.4 global rank correlation, 4–200× distance inflation, and an elephant-shaped
   embedding scores as well as UMAP (Chari & Pachter 2023). *Fix:* print `var_explained`, print the
   caveat, and never colour by cluster.
2. **Reading the 3-D column as "the model's geometry".** Depth is a real axis; the viewer will
   generalise that to the other two. *Fix:* make the z axis visually different (solid, ticked) and
   the x–y plane visually "flat" (no grid perspective, no floor plane).
3. **Reading edges as information flow.** Attention weights permit "very different attention
   distributions that nonetheless yield equivalent predictions" (Jain & Wallace 2019) and are
   "unreliable as explanations probes" once residual mixing is accounted for (Abnar & Zuidema 2020).
   *Fix:* no edges except one selected head in a 2-D strip, labelled correlational.
4. **Reading a smooth path as a computation trajectory.** Activation Atlas: paths are "constructed
   after the fact in the low-dimensional projection"; we "don't necessarily know how the paths
   operate in the original higher-dimensional activation space". *Fix:* draw trajectories as
   28 discrete points + straight segments, not splines. Curves imply continuity that was not measured.
5. **Reading depth as time.** Layer ≠ time; every token is a complete forward pass (CONTEXT D5).
   *Fix:* the word "step" only for tokens, "layer" only for depth; never "progress".
6. **Reading node size as importance.** `norm` is a magnitude, not a contribution; it can rise while
   the prediction does not change. *Fix:* the legend says "residual norm" with units, and the hover
   card gives the lens probability next to it.
7. **Believing the animation.** Particles/rotation imply activity and causation that the data does
   not contain. *Fix:* decouple status (text) from motion (user-initiated camera only).
8. **Believing one projection is *the* structure.** Hyperparameters change the picture; t-SNE makes
   "dramatic clusters" out of 500 Gaussian points. *Fix:* offer the 2-D unroll and, if cheap, a second
   basis (e.g. PCA vs a random projection) with a "same data, different projection" toggle.
9. **Reading an empty layer as zero.** Missing computation looks like no computation. *Fix:* hollow
   glyph + `no data` (already in `BRAIN_SPEC.md` §Honesty; keep it).
10. **Over-trusting the whole instrument because it looks scientific.** Data scientists "over-trust
    and misuse interpretability tools" and "few ... were able to accurately describe the
    visualizations" (Kaur et al. 2020); more visible detail made users *worse* at catching model
    errors (Poursabzi-Sangdeh et al. 2021, N=3,800). *Fix:* evidence chips, intervention log, and a
    one-click "what this view cannot tell you" panel — the copy is part of the renderer.

---

*Written by J2. Measurements reproducible from `/tmp/opencode/j2/bench.html` and `bench2.html`
(headless Chrome, WebGL2, 1280×720). Numbers are `[OBS]` on this machine; generalisation to other
laptops is `[INF]`.*
