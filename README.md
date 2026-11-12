# mw

A local-first microscope for open-weight language models: read the internal state
behind every generated token, intervene mid-generation, and **certify** what
changed when a model is quantised or fine-tuned.

Everything here is measured. Where a claim failed, the failure is in this file.

```
prompt ──► transformer ──► next token
             ▲  ▲  ▲
             │  │  └─ steer: add coef·v to a layer, or clamp a neuron
             │  └──── lens: what does layer L want to say next?
             └─────── trace: residual magnitude, PCA position, top neurons
```

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  web/  Next.js 16 (port 3000)                                        │
│  · chat rail (300px, collapsible)                                    │
│  · brain viewport: 40 layer columns + per-token trajectories         │
│  · right panel: Trace | Neurons | Steer                              │
└───────────────▲──────────────────────────────────────────────────────┘
                │ WebSocket /ws  +  REST /api/*
┌───────────────┴──────────────────────────────────────────────────────┐
│  app/server.py  FastAPI (port 8000)                                  │
│  · one model, one generation lock, 180s watchdog                     │
│  · per-connection steering state                                     │
│  · mid-generation context injection (KV-cache extension)             │
└───────────────▲──────────────────────────────────────────────────────┘
                │ ModelAdapter API
┌───────────────┴──────────────────────────────────────────────────────┐
│  code/mw/  the instrument (no web dependencies)                       │
│                                                                      │
│  registry.py   per-architecture descriptors (tensor map, scaling)    │
│  convert.py    GGUF → HF safetensors, refuses what it cannot do      │
│  guard.py      memory safety; refuses loads that would freeze        │
│  adapter.py    load, hooks, trace, lens, steer, neurons, stream      │
│  certify.py    quantisation / fine-tune decision-flip certificate    │
└──────────────────────────────────────────────────────────────────────┘
```

### Why it is split this way

`code/mw` never imports the web app. It is the only place that touches weights,
so it can be tested, audited and reused from a script. `app/` is a thin protocol
layer. `web/` is a renderer. If the UI is wrong, the instrument is still correct;
if the instrument is wrong, no UI can save it.

### Module by module

| file | what it does | what it refuses |
|---|---|---|
| `registry.py` | one `ArchSpec` per architecture: GGUF tensor-name map, HF class, scaling factors | unknown arch, fused QKV (Phi-3), VL models |
| `convert.py` | GGUF → HF safetensors; carries `embedding/residual/attention/logit` scales into `config.json` | missing scaling metadata (would be 8× wrong), non-finite weights, fp16 overflow, tie mismatch |
| `guard.py` | reads `MemAvailable` via psutil, estimates model footprint from the safetensors header | loads that would exceed RAM; BF16→FP16 (overflows to NaN) |
| `adapter.py` | `ModelAdapter`: hooks on every layer, `trace()`, `logit_lens()`, `set_steer()`, `set_neuron_steer()`, `stream()` | out-of-range layers, missing output embedding, missing `logits_scaling` on Granite |
| `certify.py` | reference vs candidate decision comparison with margins, KL, Wilson CI, null control | stale cache (fingerprint mismatch), tokenizer mismatch |

---

## 2. Data flow, one token at a time

1. The user sends a message. The client sends the **whole conversation**
   (`messages[]`), not a single prompt — otherwise every turn is a cold start.
2. The server renders it through the model's chat template and prefills
   (~9 s for 60 tokens at 2B on this CPU).
3. For each generated token the server captures, per layer:
   - `norm` — magnitude of that layer's residual output at the last position
   - `xy` — its coordinate in a **shared 2-D PCA** of the standardised residual
     trajectory (fit over a rolling buffer of 32 tokens, re-fit every 8)
   - `lens` — top-3 next-token predictions from that layer (logit lens)
   - `neurons` — top-3 MLP intermediate indices and activation values
4. The payload also carries provenance: `steered`, `coef`, `steer_layer`,
   `neuron_steer`, and `pca.var_explained`.
5. The UI draws one trajectory per token. Tokens are keyed by `layer:neuron`,
   so a neuron that persists moves, one that drops fades out, a new one fades in.

Payload per token is ~4 KB. Full schema: `app/PROTOCOL.md`.

### What each field means

| field | meaning | how to read it |
|---|---|---|
| `norm` | residual magnitude at that layer | bigger = that layer is "loud" at this token |
| `xy` | position in shared PCA space | shape of the trajectory, **not** semantics |
| `pca.var_explained` | how much of the trajectory the 2 PCs carry | below ~0.4 the 2-D view is a poor summary |
| `lens` | what layer L would predict if it were the last layer | early layers are approximations |
| `neurons` | top-3 of 8960 MLP units by activation | the wire sends 3; the panels use up to 32 |
| `steered` | whether an intervention was active for this token | the only field that makes a token causal |
| `coef` | coefficient in **units of one residual norm** | calibrated per layer; 0 is a true no-op |

---

## 3. The UI, control by control

### Left rail (chat)

- **Composer** — Enter sends, Shift+Enter newline, Esc cancels.
- **Example chips** — click runs the prompt immediately.
- **Inject** — appears while streaming. Sends `inject`; the server closes the
  current assistant turn, appends **one user turn** (18 tokens, template diff),
  and continues. *What might happen:* the 2B model may follow it, may finish the
  old sentence first, or may ignore it. It always appears as a chip at the step
  it landed, and the next token's trace changes.
- **Cancel** — stops after the current token. A 180 s watchdog also cancels.
- **Steering readout** — `steer L8 · coef 0.0 (off)`: the layer and coefficient
  that will apply to the next token.

### Brain viewport (centre)

- **40 layer columns**, left to right = layer order. The footer says
  `top-k of 8960 per layer` because we only capture the top few.
- **Nodes** — one per (layer, neuron). Radius from `norm`; brightness from the
  neuron's value. Click a column to open that layer.
- **Trajectory** — the path of one token through the layers. Previous 3–4 tokens
  remain as 15% ghosts.
- **HUD** — tokens, active neurons, current layer, tok/s.
- **Geometry** — switches between layer-column view and the PCA `xy` view.
  *What might happen:* the PCA basis re-fits every 8 tokens, so older ribbons are
  dropped and labelled `basis vN`. That is deliberate; mixing bases would lie.
- **Export** — downloads the session trace as JSON.

### Right panel (Trace | Neurons | Steer)

**Trace** — the selected token's frame: step, token, `steered`, `coef`,
`steer_layer`, layer count. If `steered` is true, the badge reads `causal`;
otherwise `correlation, not meaning`.

**Neurons** — for the selected layer:
- top-32 MLP units with indices and activation values
- lens top-3 for that layer
- *what it means:* a neuron is an index in the MLP intermediate
  (`down_proj` input). It is **not** a named concept; the same unit fires for
  unrelated things.

**Steer** — two independent interventions:

1. **Global steering** (layer + coefficient)
   - `coef = 1` is one residual norm at that layer; the slider is clamped.
   - Applies `h += coef · v` where `v` is a diff-in-means direction built from
     contrast pairs (comment-style vs code-style prompts).
   - *What might happen:* output changes character; large coefficients produce
     repetition, empty lines, or derailment. `coef = 0` is bit-identical to off.
   - *Honest status:* it changes output, but C4 shows the direction does not
     carry the intended semantics at a usable scale (§5).

2. **Steer neuron** (index + value → `apply`, or `ablate`)
   - Writes an absolute value into that MLP unit at the last position, for
     subsequent tokens.
   - Natural activations here are ~4–6; `50` is roughly 10×.
   - `ablate` sets it to `0` — a targeted deletion, not a semantic control.
   - *What might happen:* a single unit rarely flips a token by itself; extreme
     values can destabilise the whole layer. Per-connection: another client is
     never affected.

---

## 4. Run it

```bash
# backend (one model, CPU)
PYTHONPATH=code .venv/bin/python app/server.py --model .hf/granite-hf --dtype float32
# frontend
cd web && npm install && npm run build && npm start
```

Open **http://localhost:3000**. `ui/index.html` is a redirect stub, not the app.

Other models: `--model .hf/bf16-hf` (Qwen2.5-Coder-1.5B), or convert a GGUF:

```bash
PYTHONPATH=code python -m mw.convert --gguf path/to.gguf --out .hf/model \
  --hf-repo ibm-granite/granite-3.3-2b-instruct
```

Supported: Qwen2/2.5, Llama/Mistral, Granite 3.x, Gemma (partial). Unknown
architectures are **refused**, never guessed.

---

## 5. What is actually proven (and what is not)

| link | result |
|---|---|
| C1 trace | PASS — matches HF hidden states, max diff `0.000e+00` |
| C2 logit lens | PASS — `0.000e+00` at the final layer |
| C3 causal patching | PASS — L22–27 recover ≈1.0, L0–20 ≈0 |
| C4 steering specificity | **FAIL** — calibrated, 20 controls |
| C5 mid-generation | PASS — divergence begins after the switch |
| C6 force vs steer | PASS for forcing; semantic arm provisional |
| C7 neuron atlas | PASS at threshold 1e-6 (coverage, not meaning) |
| C8 sanity gate | PASS — 10/10 |

**C4, calibrated run** (L8, coef +1 = one residual norm, 20 random controls):

```
T1 target      real -0.38    random p95 +1.12    FAIL
T2 collateral  real  2.87    random median 2.28  FAIL
T4 sign        real  0.10    random p95 1.00     FAIL
```

At a realistic perturbation the vector moves the target slightly the wrong way,
and only 1 of 10 held-out prompts responds consistently. At 130× it produced
+4.4 — that was saturation, not steering. **The slider changes output; it is a
perturbation knob, not semantic control.**

Also proven the hard way: quantisation hid a failure (Q4 passed where bf16 did
not), a BF16→FP16 cast silently produces NaN, and a half-installed `next`
package can look present while being unusable.

---

## 6. Certificate

```bash
PYTHONPATH=code python -m mw.certify \
  --reference .hf/bf16-hf --candidate .hf/q4km-hf \
  --builtin 41 --null-control --out proof/cert_qwen_q4
```

Compares **next-token decisions**, not benchmark averages, and prints a null
control (same model, other dtype) so the result has a floor.

First run, 41 unique prompts:

| metric | value |
|---|---|
| decisions flipped | 2/41 (4.9%) |
| **meaningful decisions** (no whitespace/punctuation) | **2/18 (11.1%)** [3.1%, 32.8%] |
| trivial decisions excluded | 23 |
| 95% CI (all) | [1.4%, 16.1%] |
| confident flips | 0 |
| mean / max KL | 0.042 / 0.218 |
| null control (fp32 vs bf16) | 0.0% [0%, 8.6%] |

The two flips: `Two plus two equals` → **` three`**, and `The author of Hamlet
was` → ` born`. Both were near-ties (< 0.25 nat).

**The by-class table found a flaw in our own suite.** 22 of 41 decisions were
whitespace — after a code prompt the model's top choice is often `\n` or a space
— so the headline 4.9% was diluted. Excluding whitespace and punctuation, the
rate is **11.1%** on 18 real decisions. A certificate is only as meaningful as
the decisions it declares; the suite, not the model, was the weak part.

### v2 suite, 96 unique prompts

```
decisions flipped        21 / 96   (21.9%)   CI [14.8%, 31.1%]
meaningful               19 / 76   (25.0%)   CI [16.6%, 35.8%]
null control (fp32/bf16)  6.2%     [2.9%, 13.0%]
confident flips           2
mean / max KL             0.154 / 1.319
```

The flip rate is now **well above the 6.2% dtype floor**, so the changes are
attributable to quantisation rather than precision. By class, identifiers are
the worst hit:

| class | decisions | flips | rate |
|---|---|---|---|
| `identifier` | 34 | 11 | **32.4%** |
| `prose` | 26 | 6 | 23.1% |
| `string` | 10 | 1 | 10.0% |
| `number` | 3 | 1 | 33.3% |

**A concrete artifact:** in 3 of 21 flips the Q4 build emitted a raw FIM control
token (`<|fim_suffix|>`) where bf16 produced text — e.g. `except Excepti` →
`<|fim_suffix|>`, `{"user_na` → `<|fim_suffix|>`. That is not a style change; it
is a quantised model leaking training-time control tokens into output.

And code identifiers: `def calculate_tot` → `ale` (bf16) vs `_price` (Q4),
`import nump` → `ypy` vs ` as`. A quarter of identifier decisions change, which
is exactly the kind of regression a benchmark average hides.

**It does not certify at 5%**: with 41 unique prompts the upper bound is 16.1%,
and the null control's bound (8.6%) is wider than the observed rate, so the
flips cannot be attributed to quantisation rather than precision noise. The tool
says so in the report instead of showing a pass badge.

Next size: 96 items is the first that can honestly certify a 5% gate (3.8%
upper bound on zero flips); 384 narrows it to ~1%.

---

## 7. Limits, stated plainly

- **No neuron-level concepts.** Superposition and polysemanticity are real. A
  neuron is an index, not a word.
- **The 3-D view is a projection.** `z` is layer order (real); `x/y` are PCA.
  Variance explained is printed because it matters.
- **Steering is not solved.** See C4 above.
- **Long context is not tested.** 32k is configured; nothing here validates it.

---

## 8. Layout

```
code/mw/         instrument (registry, convert, guard, adapter, certify)
code/            proof chain, experiments, converters
app/             FastAPI server + PROTOCOL.md + BRAIN_3D_SPEC + DESIGN
web/             Next.js UI
tests/           116 tests (fast: no model; slow: real weights)
proof/           artifacts (certificates, proof chain, sanity)
```

## Where the research lives

Validation, the proof chain and the research ledger are on the **`research`**
branch (`git checkout research`), not on `main`. `main` ships the instrument,
the app and the artefacts; the branch carries the evidence trail, the failed
hypotheses and the audit reports.

## License

MIT. Model weights keep their own licences.
