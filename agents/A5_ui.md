# A5 — Visualization and the "brain view": what to render, what to reuse, what to refuse

## TL;DR (5 bullets)

- **Almost nothing off-the-shelf is usable as-is for a live, streaming, 1.5B-on-CPU instrument.** The field's UIs are either (a) notebook widgets (BertViz, circuitsvis-in-Python), (b) heavy single-model apps (Neuronpedia, LLM-TT, Transformer Explainer), or (c) require artefacts we do not have (circuit-tracer needs transcoders that do not exist for Qwen2.5 — confirmed).
- **Only `circuitsvis` (MIT) is a genuine drop-in**: it already exports `TopkTokens`, `TokenLogProbs`, `ColoredTokens`, `AttentionPattern`, `AttentionHeads` — which is ~80% of the panel vocabulary we need. Reuse its *component set and visual language*; hand-roll the streaming host.
- **The logit lens is the most expensive pixel on screen, not the cheapest.** 28 layers × (1536×151936) unembedding ≈ **13.1 GFLOP and ~467 MB of weight reads per token** vs ~2.6 GFLOP for the model forward. Unbatched it costs ~0.7–1.3 s/token on this box; batched as one `(28×1536) @ (1536×151936)` matmul it costs one weight pass (~30–60 ms). This single implementation choice decides whether the UI can stream at all.
- **Recommended default layout** (argued in F8): token film (DOM) + one scalar layer×token heatmap (canvas) + a layer-by-layer top-k lens table for the selected token + a small attention strip + a steering dial. Nothing else on the first screen. Progressive disclosure for everything else.
- **Pause-before-emit is a two-phase commit, not a pause**: sample → *preview* (uncommitted) → accept / force-alternative / adjust coefficient & re-run → commit. Failure modes are latency (each preview = 1 forward), flicker (reordering top-k), overload, and — the serious one — **silently turning observation into rejection sampling**. Log every intervention and say so on screen.

---

## Findings

### F1. The tool landscape splits cleanly into *reusable components* and *reusable ideas*; only one is a component
[source: repos verified via GitHub API 2026-11-10; licence + last-push dates as of that date]

| Tool | What it shows | Licence | Embeddable? | Works on local Qwen2.5-Coder-1.5B? | Verdict |
|---|---|---|---|---|---|
| **circuitsvis** [CircuitsVis, Cooney & Nanda, 2023, https://github.com/TransformerLensOrg/CircuitsVis] | Attention heads grid, attention patterns, attention neurons, coloured tokens, token log-probs, top-k tokens, neuron activations. React components + Python (Jupyter) wrappers, rendered from CDN | **MIT** | **Yes** — `yarn add circuitsvis`, model-agnostic (takes tensors, not models) | **Yes** (pure renderer) | **REUSE** — highest value/effort ratio |
| **Neuronpedia** [https://github.com/hijohnnylin/neuronpedia] | SAE feature dashboards (activations, autointerp labels, UMAP, cos-sim), steering, attribution-graph viewer, `/embed/sae-explorable` iframe route (verified 200) | **Apache-2.0** | Partly (iframe embed for SAE explorer; whole app = Next.js + Postgres/pgvector monorepo) | **Data-wise no** for 1.5B (see F3) | **REUSE (stage 2)** for SAE features; steal the "feature card" UX |
| **circuit-tracer** [Ameisen et al. 2025, https://transformer-circuits.pub/2025/attribution-graphs/methods.html ; code https://github.com/decoderesearch/circuit-tracer] | Node-link attribution graph by layer column, supernodes, feature hover cards, intervention ("Steer") on pinned nodes | **MIT** | Technically yes — the frontend is *plain static D3 + dagre + pako* served by `circuit_tracer/frontend/local_server.py` (verified in `assets/index.html`), not an npm package | **No** — requires cross-layer transcoders (F2) | **CANNOT USE** for our model; keep as the visual grammar for "causal graph" if we ever get transcoders |
| **LLM Transparency Tool (LLM-TT)** [Tufanov, Hambardzumyan, Ferrando, Voita 2024, arXiv:2404.07004, https://github.com/facebookresearch/llm-transparency-tool] | Contribution graph between tokens (threshold slider), representation of any token after any block, its projection to vocab, **promoted/suppressed tokens vs previous block**, clickable heads/FFN/neurons | **CC BY-NC 4.0** (verified in README) — **non-commercial** | No — Streamlit app; model list is a config file, needs a `TransparentLlm` impl per architecture | Probably yes with a patch (Qwen1.5 is in its config; Qwen2.5 is not), but see licence | **STUDY, DO NOT VENDOR.** The "promoted/suppressed" panel is the best idea in the whole field |
| **BertViz** [Vig 2019, arXiv:1904.02679, https://github.com/jessevig/bertviz] | Head view (one/many heads, bipartite), Model view (bird's-eye all layers/heads), Neuron view (q/k neuron contributions) | **Apache-2.0** | No — Jupyter/Colab widget only | Yes but needs eager attention; memory = 28·12·T²·2 B (T=512 → **176 MB** per capture) | **STEAL THE NEURON VIEW**, don't ship it |
| **exBERT** [Hoover, Strobelt, Gehrmann 2020, arXiv:2010.11994, https://github.com/bhoov/exBERT] | Attention + hidden-state explorer with a sentence-similarity "hidden state search" panel; BERT-era | Apache-2.0 | No (web app, unmaintained since 2024-02) | BERT-only, 2020-era | **SKIP** |
| **Transformer Explainer** [Cho et al., arXiv:2408.04619, CHI 2026, https://github.com/poloclub/transformer-explainer] | Live GPT-2 in the browser (transformers.js + `onnxruntime-web`), per-layer flow, sampling animation | **MIT** | No (SvelteKit app; ~1.5 GB int8 weights for 1.5B, no hooks) | **No** for 1.5B; no hooks/steering at all | **UX REFERENCE ONLY** — best-in-class "watch a token form" animation |
| **Dodrio** [Wang et al., ACL 2021 demo, arXiv:2103.14625, https://github.com/poloclub/dodrio] | Attention + hidden-state probing with linguistic knowledge; multi-model comparison | MIT | No (Svelte, last push 2023) | BERT-family focus | UX reference only |
| **llm-viz** [https://github.com/bbycroft/llm-viz] | 3D WebGPU walkthrough of GPT-2 inference | MIT | No | educational, static | Inspiration only |
| **LIT** [https://github.com/PAIR-code/lit] | General interpretability: salience, counterfactuals, sequence salience, datapoint explorer | Apache-2.0 | Partly (server + web components) | Possible, but has **no logit lens, no steering, no residual-stream view** | **NOT A FIT** — extending it costs more than building the 5 panels we need |
| **LIDA** | *Ambiguous in this context.* `microsoft/lida` (MIT) is LLM→chart generation, irrelevant. No interpretability UI named LIDA exists that I could verify — `UNVERIFIED`. The intended tool is almost certainly **LIT** (row above) | — | — | — | **ASK** the orchestrator which was meant |
| **tuned-lens** [Belrose et al. 2023, arXiv:2303.08112, https://github.com/AlignmentResearch/tuned-lens] | Trains an affine probe per block → debiased layer-wise vocab distributions | MIT | Library, not UI | Yes, but needs probe training (extra forward passes over a corpus) | **STAGE 2 upgrade** to the lens panel |

**Confidence:** high (licences and repo contents fetched directly). **Cost:** zero to evaluate; the cost is in *not* evaluating and rebuilding BertViz badly.

---

### F2. Attribution graphs are structurally unavailable for our model — this constrains the UI, not just the backend
[source: circuit-tracer README "Available Transcoders" section, fetched 2026-11-10]

The available transcoder sets are: Gemma-2-2B (PLT + CLT 426K/2.5M), Llama-3.2-1B, **Qwen-3 0.6B/1.7B/4B/8B/14B**, GPT-OSS-20B, Gemma-3 (270M–27B), Llama-3.1-8B-Instruct. **Qwen2.5 is absent.**

Why it matters for the UI: the most impressive "brain view" artefact in the field (a causal attribution graph) is a **function of which model you chose**, not of your UI. Our 1.5B Qwen2.5 model can have: logit lens, attention, residual norms, activation patching, steering. It cannot have: transcoder attribution graphs, unless we train CLTs (GPU-scale job, out of scope for CPU-only).

**Actionable fork** (needs an orchestrator decision, not an A5 decision): if attribution graphs are a project goal, **switch the science model to Qwen3-1.7B** — same parameter class, and PLTs + the Neuronpedia graph viewer + circuit-tracer all exist for it today. If staying on Qwen2.5-Coder-1.5B (better for code, and TL-verified — see F4a), then the UI must be honest that it shows *correlational* structure plus *locally causal* steering, not circuits.

**Confidence:** high. **Cost of the fork:** re-running A1/A2/A3/A4 model-specific work.

---

### F3. Neuronpedia has a `qwen2.5-1.5b-it` entry with **zero** SAE source sets — SAE panels are stage 2
[source: `POST https://www.neuronpedia.org/api/model/list`, 2026-11-10]

Verified: `qwen2.5-1.5b-it` exists (28 layers, `visibility: UNLISTED`, `sourceSets: []`). `qwen2.5-7b-it` has 3 source sets. `qwen3-1.7b` has 3. So there are **no public SAEs for the 1.5B Qwen2.5 family**, and none for the *Base* variant we target anyway (SAEs are typically trained on base/instruct separately).

Consequence for UI design: **do not put a "features" panel in v1.** It would be an empty state at best, a lie at worst. The feature panel arrives only if we train SAEs ourselves (CPU training on 1.5B is possible but slow — A2's problem, not mine) or upload activations to a local Neuronpedia instance (their monorepo supports self-hosting with Postgres + pgvector, and an admin import path).

**Confidence:** high. **Cost:** ~0 now; ~2 days to stand up a local Neuronpedia if we ever have SAEs.

---

### F4. Two infrastructure facts that make the UI cheap
[source: TransformerLens `supported_models.json` model registry, fetched 2026-11-10]

**(a) TransformerLens fully verifies our exact checkpoint.** `Qwen/Qwen2.5-Coder-1.5B` → `Qwen2ForCausalLM`, status 1 ("Full verification completed", 2026-02-24). `Qwen/Qwen2.5-Coder-1.5B-Instruct` likewise. Quantized variants are explicitly **unsupported** by TL ("TransformerLens does not support quantized models at this time") — which independently corroborates D2. **UI consequence:** `HookedTransformer` gives us named hook points (`blocks.{L}.hook_resid_post`, `blocks.{L}.attn.hook_pattern`, …) and `.to_tokens()`/cache, so the trace producer is a ~50-line hook loop, and the UI can be developed entirely against a JSONL file before the runtime exists.

**(b) `circuitsvis` renders from a CDN in Python**, so even a non-React prototype can borrow its components by emitting their HTML — but that path is fragile for 10 Hz streaming. Use the React components only if we adopt React; otherwise copy the visual grammar (~150 lines of canvas each).

**Confidence:** high. **Cost:** 0.

---

### F5. The logit lens is the performance budget item — batch it across layers or the UI cannot stream
[derived from CONTEXT model facts; arithmetic shown so A4 can check it]

- Unembedding matrix: 151936 × 1536 × 2 B (bf16) = **466.7 MB**.
- Lens cost per layer: 2·1536·151936 = **0.467 GFLOP**; × 28 layers = **13.1 GFLOP**.
- Model forward: ≈ 2 × 1.31 B non-embedding params ≈ **2.6 GFLOP**.
- → **Lens ≈ 5× the model.** And it is **bandwidth**-bound, not FLOP-bound: naive per-layer lensing re-reads 467 MB 28 times = **13 GB/token**. At a realistic 10–20 GB/s effective streaming read, that is **0.65–1.3 s per token** — i.e. it would dominate and destroy interactivity.
- Batched: stack the 28 hidden vectors into `(28, 1536)` and do **one** matmul against `(1536, 151936)` → one weight pass, ~467 MB → **~25–60 ms**, plus 13 GFLOP of compute which the same matmul absorbs. (Caveat: `nn.Linear` with M=28 is still memory-bound on CPU; measure with `torch.set_num_threads(8)`.)

Practical ladder, in order of preference:
1. **Batch all lensed layers into one matmul per token** (default).
2. **Lens a subset**: layers `[0,4,8,12,16,20,24,27]` (8 of 28) → ~1/3.5 the compute, ~1 weight pass regardless (same 467 MB read; the FLOPs drop but the read does not — so the win is smaller than it looks: **the read is the cost**).
3. **Lens on pause only**: streaming shows norms + entropy + steering projection (all O(hidden)); the lens table populates when the user pauses or selects a token. This is the honest "instant mode".
4. **Quantized lens probe** (int8 copy of W_U, ~117 MB) for the live view, exact bf16 for the paused view. Label the difference in the UI. (`UNVERIFIED` numerically — A2/A6 should measure top-k agreement; expect small but nonzero disagreement.)

**Confidence:** high on arithmetic, medium on the wall-clock (CPU matmul efficiency varies 2×). **Cost:** the batching change is ~10 lines; the ladder is a config flag.

---

### F6. Attention is cheap to compute and weak as evidence — render it small and label it correlational
[source: Jain & Wallace, "Attention is not Explanation", NAACL 2019, arXiv:1902.10186; Serrano & Smith, "Is Attention Interpretable?", ACL 2019, arXiv:1906.03731]

Jain & Wallace: attention weights are frequently uncorrelated with gradient-based importance, and **different attention distributions yield equivalent predictions** — so a "look where the model looks" panel is not an explanation. Serrano & Smith: attention *noisily* predicts importance; gradient rankings predict it better.

Consequences for the UI:
- Attention gets a **small strip** (one selected head's last-row pattern), never a hero panel.
- The head grid (28×12 = 336 cells) is a *navigation* device coloured by **entropy**, not by "importance". Clicking a cell opens the pattern. Entropy is a fact about the distribution; "importance" is an inference.
- Never draw an attention arc diagram as the main visual (BertViz's head view) at 512+ tokens: it becomes spaghetti, and the literature says the spaghetti doesn't mean what viewers think it means.

Also note GQA: 12 query heads / 2 KV heads means **heads within a KV group share keys/values** — so "head 7 attends to X" is about the query projection, and heads in the same KV group are not independent. Show the KV grouping in the head grid (2 groups of 6) so nobody over-reads head-level stories.

**Confidence:** high. **Cost:** entropy per head is ~free once attention probs exist.

---

### F7. Ranked by information-per-pixel: what actually belongs on screen

| Signal | Cost/token | What it tells you | Causal? | Keep? |
|---|---|---|---|---|
| **Layer-wise top-k lens** (top-5 × 28 layers) | high (F5) | how the prediction crystallises; the single best "token forming" view | no (probe) | **YES — core** |
| **Promoted / suppressed vs previous layer** [LLM-TT's best idea] | free once lens exists | *what changed here*, per layer — the causal-looking signal | no | **YES — core** |
| **Residual-stream norm per layer** | ~free | where the model is "loud"; step changes = computation happening | no | YES (sparkline + heatmap channel) |
| **Output entropy / top-5 mass** | ~free | uncertainty; when the token was a coin-flip | no | YES (per-token dot on the film) |
| **Steering vector projection** ⟨h_L, v⟩ per layer | ~free (1536 mults) | whether the dial is actually engaged, and where | **yes, if you intervene** | YES (the only honest "dial readout") |
| **Attention entropy per head** | cheap | which heads are focused/diffuse; navigation | no | YES (head grid colour) |
| **Attention pattern, one head, last row** | cheap | which context tokens feed this step | no | YES (small strip, on demand) |
| **Full attention matrices** | 28·12·T²·2 B (T=512 → 176 MB) | nothing the last row doesn't | no | **NO** — never ship |
| **MLP out norm** | ~free | layer type activity | no | optional channel |
| **Per-neuron activations (8960/layer)** | medium | polysemantic soup without SAEs | no | **NO in v1** |
| **SAE features** | n/a | interpretable units | no | stage 2 (F3) |
| **Activation-patching attribution** | expensive (per pair of (layer,position)) | genuinely causal "this token caused that" | **yes** | stage 2, on-demand, not per-token |

The design rule: **one scalar per pixel, many pixels; never many scalars per pixel.** A heatmap answers "where in (layer × time) did something happen"; the table answers "what"; the strip answers "from where".

---

### F8. Recommended default layout (the specific argument)

Single screen, four regions, no tabs on first load:

```
┌──────────────────────────────────────────────────────────────┐
│  A. TOKEN FILM (DOM text, monospace)                          │
│  def  fibonacci( n : ▁int ) -> ▁int :                        │
│  ●●●○●●●●●●●●●●●●●●●  ← per-token entropy dots under glyphs  │
├───────────────┬──────────────────────────────────────────────┤
│ B. LAYER×TOKEN│  C. LENS TABLE for selected token (t=37)      │
│    HEATMAP    │  L27  ▁int .92  ▁: .03  ▁) .01  ...          │
│  (canvas,     │  L24  ▁int .71  ▁int .11 ▁: .04  ...          │
│   28 rows ×   │  L20  ▁int .33  ▁n .22   ▁int .09 ...        │
│   N token     │  ...                                          │
│   cols,       │  L00  ▁the .01 ▁a .01    ▁def .01 ...         │
│   channel ▼)  │  ── promoted ▲ / suppressed ▼ at L24 ──       │
│               │  ▲ ▁int  ▲ ▁)   ▼ ▁the  ▼ ▁,                 │
├───────────────┴──────────────────────────────────────────────┤
│ D. ATTENTION STRIP  L14·H3  ▁def ▁fib ▁( ▁n ... ████████▁     │
│  E. STEER DIAL  layer 14  [────●────] coef +2.0  ⟨h,v⟩ 1.83   │
└──────────────────────────────────────────────────────────────┘
```

Why this and not the alternatives:
- **Token film is DOM, not canvas** — selectable, screen-readable, copy-pasteable; the model's actual output must never be a picture of text.
- **One heatmap, one channel at a time** (dropdown: lens-prob-of-chosen / resid-norm / entropy / steer-projection). Four simultaneous heatmaps is the classic overload failure; switching channels is one click and keeps a single mental model of "rows = layers, cols = time".
- **The lens table is the answer to the user's actual question** ("watch a token form"). Rows are layers *descending* (top = final) because reading "L27 ▁int .92" then "L00 ▁the" matches how people read a trajectory downward-to-origin.
- **Promoted/suppressed sits under the table, attached to the selected layer** — it is the diff between two rows the user is already looking at, so it needs no new mental model.
- **Attention strip is 24 px tall.** It is navigation, not evidence (F6).
- **The steering dial shows ⟨h_L, v⟩ live** — the projection is the only readout that tells you the intervention is actually doing something, and it costs nothing.

Rejected: force-directed graph (meaningless for a feed-forward pass), 3D (occlusion, no added information), per-neuron grids (F3), "neural network diagram with glowing paths" (implies a routing story the data doesn't support).

---

### F9. The core interaction: pause → inspect → adjust → resume, as a **two-phase commit**

The naive framing ("pause the model mid-thought") is wrong mechanically and misleading psychologically. What actually happens: each token is a complete forward pass; there is no "mid-thought" to pause. So the interaction must be designed as **sample → preview → commit**, with rollback of exactly one step.

State machine (per step):

```
IDLE ──run──▶ FORWARD (hook fires, logits computed)
                 │
                 ▼
            PREVIEW  ──accept──────▶ COMMIT (append token, KV cache advances) ──▶ IDLE
                 │  ▲
        force alt│  │ adjust coef (debounced 120 ms)
                 ▼  │
            RE-FORWARD (restore KV to pre-step, re-run with new hook)
```

- **Modes:** `stream` (no pause, default), `step` (pause every token; Space = accept), `breakpoint` (pause when a predicate fires: `entropy > τ`, `lens_prob(chosen) at L27 < τ`, `⟨h,v⟩ > τ`, regex match, `step == n`). Breakpoints are the mode that actually gets used — pausing on every token is unusable at 10 Hz.
- **Preview affordances:** the pending token renders as a **ghost glyph** with a dashed underline, plus a top-5 row of alternatives with probabilities. Nothing is appended to the output until accept.
- **Force-alternative** is a legitimate feature *and* a scientific hazard: it is sampling from a truncated distribution. Mark the token with a distinct glyph (e.g. `⟂`) and count it.
- **Coefficient adjust** re-runs the step. Because the forward is 60–200 ms, debounce the slider at ~120 ms and show a spinner on the ghost glyph, not a full-panel skeleton (avoids flicker).

**Cheap preview trick, with its exact limit:** for a residual edit `h ← h + c·v` at layer L, the logit change is `W_U · Δh` only when L = last layer *and* you ignore the final RMSNorm (Qwen2 applies RMSNorm before unembedding, so the true map is not linear). So:
- precompute `u_v = W_U · v` once (~467 MB read, ~50 ms) → **slider moves become free** elementwise previews, *labelled* "linearised estimate";
- on slider release, run the exact forward and replace the estimate. Never let an estimate be committed.

**Failure modes and mitigations:**

| Failure | Mechanism | Mitigation |
|---|---|---|
| **Latency** | each preview = 1 forward (60–200 ms) + lens (30–60 ms); user reaction 250–500 ms makes auto-pause modes feel broken | debounce; render preview at 60 fps from cached tensors; never block the WS read loop on the UI |
| **Flicker** | top-k lists reorder every token; heatmap column shifts; React re-renders whole tree | fixed-rank slots with stable order ("sticky ranks"); append-only heatmap column; `rAF`-coalesced draws; CSS `will-change` only on the canvas |
| **Information overload** | 28×T×12 signals available, human attention ~1 object | one channel at a time (F8); everything else one click away |
| **Illusion of observation** (the serious one) | retrying until you like the token = rejection sampling; the trace stops being a sample from the model | count and display interventions; offer a "clean re-run from this point" that discards retries; log every accept/force/retry to the trace file |
| **Illusion of control** | a coefficient that "works" on one prompt may be a null effect | the dial shows ⟨h,v⟩ and the *measured* Δlogits, not a promise; A3/A6 must supply a null-effect check |

**Confidence:** high on mechanism (it's just loop control), medium on feel (needs a prototype to tune).

---

### F10. Rendering: canvas + `ImageData`, D3 only for layout, WebGL not needed
[engineering estimate, with the arithmetic; benchmark harness in "Build next" step 0]

Sizes: head grid 28×12 = 336 cells; layer×token heatmap 28 × T; at T=512 → 14 336 cells; a 2 000-token history → 56 000 cells. Per-token column = 28 cells. Budget: 16.7 ms/frame, ~100 ms/token → 6 frames of slack.

- **SVG/D3 per cell: no.** One DOM node per cell means 14k–56k nodes; every token triggers layout+style recalculation on the whole tree. This is the single most common reason interp dashboards stutter.
- **D3's role: axes, ticks, labels, colour scales, layout maths** — static, ~30 nodes. Perfect fit. D3 for *data binding* of heatmaps is the anti-pattern.
- **Canvas 2D with `ImageData`: yes.** Write RGBA bytes for a `(T, 28)` offscreen buffer (T=512 → 512×28×4 = 57 KB), `putImageData` once, then `drawImage` scaled with `imageSmoothingEnabled=false` to get crisp cells at any zoom. Cost: microseconds. At 10 tok/s you touch one column.
- **WebGL (`regl`/`deck.gl`): only if** you need continuous zoom/pan over >10⁶ cells, or a force-directed graph with 10⁴ nodes. Neither is v1. Adding it now buys nothing and costs a shader debugging week.
- **Sparklines:** canvas, hand-rolled (~40 lines) or `uPlot` (~40 KB, canvas-based, designed for high-rate streaming). Never an SVG chart lib per token.
- **Text:** DOM always. Tokens must be selectable and readable by assistive tech.
- **60 fps recipe:** (1) WS messages land in a ring buffer, not in React state; (2) one `requestAnimationFrame` loop drains the buffer and issues ≤3 canvas draw calls; (3) the lens table updates via direct `textContent` writes to pre-created cells, keyed by (layer, rank), not by token id — stable keys are what prevents flicker; (4) no `getBoundingClientRect()` inside the loop.
- **Decoding:** ship tensors as base64 of `Float16Array`/`Uint8Array`, not JSON arrays of floats — a 28×T attention row at T=512 is 12 KB binary vs ~90 KB JSON.

**Confidence:** high on the technique, medium on exact frame times (CPU/browser dependent). **Cost:** ~1 day to build the renderer, ~2 h to benchmark.

---

### F11. Honesty in the UI: three labels, always visible, never optional

The project's own rule (CONTEXT: "every finding must be causal") must be visible in the interface, because the interface is where the over-claiming happens.

1. **Evidence class per panel**, as a persistent chip:
   - `[OBSERVED]` — attention weights, residual norms, entropy. Facts about tensors.
   - `[PROBE]` — logit lens, tuned lens. A *readout* of a state, biased and approximate. Belrose et al. (arXiv:2303.08112) show the logit lens is "often brittle" and that the tuned lens is "more predictive, reliable and unbiased" — so the chip should read `[PROBE: logit lens, biased]`, with tuned-lens as the upgrade path.
   - `[CAUSAL]` — only after an intervention changed the output in the predicted direction, and only for the specific (layer, vector, coefficient) tested. Anything else is `[CORRELATIONAL]`.
2. **Uncertainty shown as distribution, not as a verdict.** Always render top-5 with probabilities (they sum to <1 — show the residual as "other"), plus entropy. A single "the model is thinking about `int`" sentence is the failure mode; `L27 ▁int .92` is the honest form. Do not add semantic-entropy-style labels in v1 — [Farquhar, Kossen, Kuhn, Gal, *Detecting hallucinations in large language models using semantic entropy*, Nature 2024, doi:10.1038/s41586-024-07421-0] requires multiple sampled answers, i.e. 5–10× the compute.
3. **Never anthropomorphise in copy.** Banned strings in the UI: "thinking", "thought", "decided", "wants", "believes", "understands", "the model is trying to". Permitted: "layer 14 output", "prediction at layer 14", "attention from token 37 to token 12", "steering applied: layer 14, coef +2.0". A `docs/copy-lint.md` with a regex list, run in CI, is the cheap enforcement.
4. **Show the intervention log.** If the user steered/forced/retried, the output header should read e.g. `12 tokens · 3 interventions (2 steering, 1 forced)`. Without it, screenshots of the tool become false evidence.

**Confidence:** high (normative, but grounded in the cited negative results). **Cost:** ~2 h of copy + a chip component.

---

### F12. Accessibility specifics that actually bite for a 10 Hz heatmap

[source: WCAG 2.2, https://www.w3.org/TR/WCAG22/]

- **1.4.1 Use of Color** — the heatmap cannot be the only encoding. Add a numeric readout on hover/focus and a text alternative ("layer 24, token 37: prob 0.71").
- **1.4.3 Contrast (Minimum)** — 4.5:1 for text over the film; put tokens on a solid background, not over the heatmap.
- **2.3.3 Animation from Interactions** — respect `prefers-reduced-motion`: disable the token-appear tween and the column slide; snap instead.
- **2.2.2 Pause, Stop, Hide** — a streaming view *must* have a visible pause control (we get this for free from the step mode).
- **4.1.3 Status Messages** — the token film should be an `aria-live="polite"` region, **throttled** (announce every ~2 s or on pause, not every token) or screen readers will be flooded at 10 Hz. Provide an "announce output" toggle, default off during streaming, on in step mode.
- **2.1.1 Keyboard** — `Space` accept/step, `←/→` layer, `↑/↓` head, `[`/`]` coefficient ±0.5, `Esc` cancel preview. Every mouse affordance needs a key path; the coefficient slider especially (native `<input type=range>` gives arrow-key support for free — use it instead of a custom drag handle).
- **Colour vision:** use **cividis** or **viridis**, never rainbow/jet — [Nuñez, Anderton, Renslow, "Optimizing colormaps with consideration for color vision deficiency…", PLOS ONE 2018]. Diverging channel (promoted/suppressed) needs a CVD-safe diverging map (e.g. `coolwarm` is borderline; `BuRd` from ColorBrewer is the safe pick).
- **Zoom/text size:** at 200% zoom the 28-row heatmap must still be usable → allow the heatmap to become a scrollable list with numeric values (a "table view" toggle). This is also the screen-reader path: a `<table>` of the lens top-k is the accessible equivalent of the whole visual.

**Confidence:** high. **Cost:** ~0.5 day if designed in from the start, ~3 days retrofitted.

---

## Concrete implementation

### 1. Trace transport: JSONL on disk + WebSocket live, same schema

```python
# emit per generated step; UI can replay a file with zero model running
rec = {
  "step": 37, "token_id": 1437, "token_str": "int",
  "layers": [
     {"layer": 14, "resid_norm": 12.34, "mlp_out_norm": 4.1,
      "attn_entropy": 1.71,
      "logit_lens_topk": [[1437, "int", 0.71], [338, "n", 0.11], ...],
      "steer_proj": 1.83},          # <h_L, v> — see SCHEMA CHANGE
     ...                            # only lensed layers present (F5 ladder)
  ],
  "logits_topk": [[1437,"int",0.92], ...],
  "chosen": [1437, "int", 0.92],
  "flags": {"lens_quantized": False, "intervened": False},
  "timing": {"forward_ms": 78, "lens_ms": 41}
}
ws.send_text(json.dumps(rec))        # ~2–5 KB/token
```

### 2. Batched logit lens (the F5 fix)

```python
# one weight pass per token instead of 28
lens_idx = [0, 4, 8, 12, 16, 20, 24, 27]        # configurable
H = torch.stack([resid[l][0, -1, :] for l in lens_idx])   # (k, 1536) bf16
logits = H @ model.W_U                                        # (k, vocab) — 467 MB read once
topk = logits.topk(5, dim=-1)
```

### 3. Pause / preview / retry loop

```python
class Session:
    def step(self, coef=0.0, force_token=None):
        # KV cache is at pre-step state; one forward = one token
        with steer_hook(self.model, layer=self.layer, vector=self.v, coef=coef):
            out = self.model(self.next_input, past_key_values=self.kv, use_cache=True)
        self.pending = {"logits": out.logits[0, -1], "kv": out.past_key_values}
        return self.preview(coef)          # NOT committed

    def commit(self, token_id):
        self.kv = self.pending["kv"]; self.emitted.append(token_id)
        self.next_input = torch.tensor([[token_id]]); self.pending = None

    def preview(self, coef):
        # linearised estimate: exact only for last-layer steering, ignoring final RMSNorm
        est = self.pending["logits"] + coef * self.u_v     # u_v = W_U @ v, precomputed once
        return {"estimate": est.topk(5), "exact": None}    # exact filled on slider release
```

### 4. Canvas heatmap (append-only column, no re-layout)

```ts
// buf: OffscreenCanvas(T_MAX, 28); one column per token
function pushColumn(t: number, vals: Float32Array /* len 28 */, cmap: Uint8ClampedArray) {
  const img = ctx.createImageData(1, 28);
  for (let l = 0; l < 28; l++) {
    const c = cmapIndex(vals[l]);            // cividis LUT
    img.data.set([c[0], c[1], c[2], 255], l * 4);
  }
  bufCtx.putImageData(img, t, 0);
  needsDraw = true;                           // rAF drains this flag
}
function draw() {
  if (needsDraw) { ctx.imageSmoothingEnabled = false;
                   ctx.drawImage(buf, 0, 0, W, H); needsDraw = false; }
  requestAnimationFrame(draw);
}
```

### 5. Server

```python
# FastAPI: one WS per session, trace JSONL mirrored to disk
@app.websocket("/ws/{sid}")
async def ws(sock: WebSocket, sid: str):
    await sock.accept()
    while True:
        cmd = await sock.receive_json()      # {"op":"run"|"step"|"accept"|"force"|"steer", ...}
        rec = await run_in_threadpool(session.handle, cmd)   # never block the loop
        await sock.send_text(rec)
```

### 6. Reuse map, concretely

- `yarn add circuitsvis` → `TopkTokens`, `AttentionPattern`, `ColoredTokens` if we go React. If vanilla: copy their layout constants (they're the de-facto vocabulary of this field).
- Do **not** vendor `facebookresearch/llm-transparency-tool` (CC BY-NC 4.0) or `poloclub/transformer-explainer` (no hooks). Read their code, reimplement the two ideas we want: promoted/suppressed lists, and the token-form animation.
- Neuronpedia: only if/when we have SAEs — `iframe src="https://www.neuronpedia.org/embed/sae-explorable?..."` (route verified to exist).

---

## What NOT to do (traps)

1. **Don't ship full attention matrices.** 176 MB at T=512. Ship the last row only. (A2/A4: this is a schema decision, see below.)
2. **Don't lens all 28 layers naively.** 13 GB of weight reads per token. Batch it or subsample it.
3. **Don't build the UI against the live model.** Build against JSONL replay first; otherwise every CSS tweak costs a model load.
4. **Don't use D3 to bind 14k cells.** Canvas.
5. **Don't animate reordering top-k lists.** Sticky ranks or the eye can't track anything.
6. **Don't call attention "what the model focused on"** in copy or in panel titles (F6/F11).
7. **Don't add a features/SAE panel in v1** — there are no SAEs for this model (F3), and an empty panel trains users to distrust the tool.
8. **Don't imply pause = "interrupting a thought".** The model has no state between tokens other than the KV cache; say "step", not "thought".
9. **Don't hide retries.** Unlogged rejection sampling makes the tool a generator of false evidence.
10. **Don't put a 3D graph or force-directed layout in v1.** It is the most expensive thing to build and the least informative.
11. **Don't use rainbow colormaps** (CVD-hostile, non-monotonic lightness).

---

## Open questions for other agents

- **A2:** which layers are worth lensing for a *code* model (is the trajectory flat until L20?)? And: does the int8 lens probe preserve top-5? If yes, the live view gets 4× cheaper.
- **A2/A4:** is there a cheap "attention last row" hook that survives SDPA/flash attention? `output_attentions=True` forces eager attention, which costs memory and speed — this determines whether F6's strip is live or on-pause.
- **A3:** what is the cheapest *exact* preview of a steering coefficient? If the hook can be re-run without re-running layers < L (it can't, unless we cache), the preview latency is bounded by the full forward.
- **A3/A6:** we need a **null-effect control** for the steering dial (random unit vector, same norm) so the UI can display "effect vs chance". What does that look like numerically?
- **A4:** WS backpressure policy — if the browser falls behind at 10 tok/s, drop lens fields (biggest) or drop frames? I propose: never drop tokens, always droppable = lens, attention.
- **A6:** what counts as a "verified causal" badge in the UI? I propose: predicted direction + measured direction agree on ≥N independent prompts, with the null control separated.
- **Orchestrator:** was "LIDA" meant as LIT? And is switching the science model to Qwen3-1.7B (F2) on the table, because it unlocks attribution graphs?

---

## SCHEMA CHANGE

Proposed additions to the canonical trace (all optional; v1 consumers must ignore unknown keys):

```json
{"step": 37,
 "lensed_layers": [0,4,8,12,16,20,24,27],        // which layers have logit_lens_topk
 "layers": [{"layer": 14, "resid_norm": 12.3, "mlp_out_norm": 4.1,
             "attn_entropy": 1.71,               // mean over heads
             "attn_entropy_per_head": [1.2, ...],   // 12 floats, ~free
             "logit_lens_topk": [[1437,"int",0.71], ...],
             "steer_proj": 1.83,                 // <h_L, v> for the active vector
             "attn_last_row": "b64:f16",         // OPTIONAL, 12×T, opt-in only
             "attn_kv_group": 0}],
 "steer": {"layer": 14, "coef": 2.0, "vector_id": "v_code_01",
           "positions": "gen", "schedule": "constant", "until_step": null},
 "flags": {"lens_quantized": false, "intervened": false, "forced": false, "retries": 0},
 "timing": {"forward_ms": 78, "lens_ms": 41, "emit_ms": 96}}
```

Rationale: (a) `lensed_layers` lets the UI be honest about what it *didn't* compute (a missing row is not a zero); (b) `attn_last_row` is the only attention payload we can afford; (c) `steer` + `flags` make interventions auditable, which F11 requires; (d) `timing` lets the UI display its own latency instead of pretending to be real-time.

**Explicitly rejected:** full `attn_matrix` (176 MB/token at T=512), per-neuron activations (8960×28 floats/token = 500 KB), and any derived "importance" score (no defensible definition; invites over-claiming).

---

## Build next (ordered, with effort estimate)

| # | Step | Why first | Effort |
|---|---|---|---|
| 0 | **Benchmark harness**: synth trace generator + `requestAnimationFrame` FPS counter over 2 000 tokens × 28 rows. Settles F10 empirically | kills the perf argument before it costs a week | 0.5 d |
| 1 | **Replay viewer**: read `trace.jsonl`, render film + heatmap + lens table. No model | decouples UI from runtime; A4 can feed it later | 1.5 d |
| 2 | **Live WS host**: FastAPI + WS + ring buffer + rAF drain; `stream` mode only | proves the 10 Hz path | 1 d |
| 3 | **Step / breakpoint modes** + keyboard map + ghost-token preview | the interaction the user actually asked for | 1 d |
| 4 | **Steering dial**: precomputed `u_v`, linearised estimate, exact re-run on release, ⟨h,v⟩ readout, intervention log | closes the loop; needs A3's vector | 1.5 d |
| 5 | **Attention strip + head grid** (entropy colour, KV groups) | cheap, and needed for "why this token" | 1 d |
| 6 | **Honesty + a11y pass**: evidence chips, copy lint, reduced-motion, aria-live throttle, table view, cividis LUT | retrofits cost 3× more | 1 d |
| 7 | *(optional)* **circuitsvis components** if we adopt React; **tuned lens** probes; **local Neuronpedia + our SAEs** | stage 2, only after v1 earns it | 3–5 d each |

**Total to a defensible v1: ~7 agent-days**, of which 2.5 are pure rendering and 1 is the honesty/a11y pass. Build vs reuse verdict: **build the shell (canvas + WS + DOM), reuse the vocabulary (circuitsvis) and the two best ideas (LLM-TT's promoted/suppressed, Transformer Explainer's token-form animation); reuse no application wholesale** — none of them runs a 1.5B CPU model with live hooks, and the two that come closest (LLM-TT, Transformer Explainer) are licence- or architecture-blocked.
