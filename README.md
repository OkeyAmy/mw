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
   - *Honest status (updated 2026-11-19):* the estimator that built `v` had two
     defects — position-0 attention-sink contamination, and the comment placed
     above the function header where the held-out prompts never test it. With
     both fixed the direction moves the intended target +2.4…+3.2 nats at
     coef 1 with 10/10 held-out sign consistency (screened against 5 matched
     random controls; the full 20-control rerun is pending). §5 records both
     runs.

2. **Steer neuron** (index + value → `apply`, or `ablate`)
   - Writes an absolute value into that MLP unit at the last position, for
     subsequent tokens.
   - Natural activations here are signed (SwiGLU is signed): the panel ranks
     workers by strength and shows the sign, e.g. `n1007: -30.1` at L38.
   - `ablate` sets it to `0` — a targeted deletion, not a semantic control.
   - *What might happen:* at mid layers a single unit is usually a no-op
     (20/20 ablations at L14/L30 changed nothing), but a late-layer unit with a
     large signed write can flip the next word when reversed (L38 `n1007`
     −30 → +60 changed the sentence opening). Extreme values saturate the
     layer. Per-connection: another client is never affected.

---

### Every parameter, in plain words

No ML background needed for this part. The picture: the model is a **factory
with 40 workstations in a row**. A word enters at station 1 and is passed
along; at every station, **8,192 small workers** each add one bit to the
product; what comes out of station 40 is the next word. A **layer** is a
workstation. A **neuron** is one worker. Everything below is a knob on that
factory.

#### Chat rail

| control | plain meaning | limits |
|---|---|---|
| prompt / messages | the question. The whole conversation is sent every time, so the model remembers earlier turns. | ≤ 4000 chars per message, ≤ 16 000 per conversation |
| max | how many words it may write before stopping. It also stops on its own when the answer is finished. | 1–256 (the composer starts at 48) |
| Send / Esc | start writing / stop. Stop takes effect after the word currently being written. | — |
| inject | type a new question **while it is still answering**. The unfinished sentence is closed, your text becomes a new user turn, and the model answers that instead. Works best as a concrete question — "What is the capital of Japan?" → it answers Japan. Vague instructions ("answer in French") get ignored by a 2B model; that is measured, not a bug. | ≤ 2000 chars |
| edit_prompt | throw the half-written answer away and restart on a new question. Cleaner than inject when you do not care about the interrupted sentence. | ≤ 4000 chars |

Two people can use the model at once: every knob is per connection, so your
steering never leaks into anyone else's answer (tested).

#### Steer (the layer slider)

One slider, one number. It leans a whole workstation in one direction — every
worker at that station at once.

| part | plain meaning |
|---|---|
| layer | which workstation (0–39). Early: the lean flows through 30 more stations and gets absorbed — harder to predict. Late: almost straight to the next word. |
| coef | how hard the lean is. **0 is exactly off** — the model cannot tell it apart from untouched (tested bit for bit). **1.0 is one full unit** of the signal at that station, and on the fixed model that is already enough to change the opening word. |
| sign | one end pushes toward "explain in words", the other toward "just write the code". On `def fib(n):` at layer 14: coef −1 makes it write `def fib(n):` and `if n > 1:`; coef +1 keeps it talking about the function. |

The UI slider spans −4…+4 (the protocol accepts ±20). Measured honestly: below
about ±1 at mid layers nothing visibly moved, and large values produce newlines
and loops — damage, not steering. The number means the same thing on every
layer by construction; the *effect* still differs by layer.

#### Steer a neuron

| part | plain meaning |
|---|---|
| neuron | one worker out of 8,192 at that station. An index — `n3766` is slot 3766, not a concept. |
| value | an **absolute setting** written into that worker, like a dimmer with a sign — not a multiplier. |

| you type | what happens |
|---|---|
| its natural value | the worker keeps doing its job — no change |
| 0 | send it on break |
| the opposite sign | make it push backwards — usually the only change you can see |
| bigger, same sign | usually nothing: it was already pushing that way |

The panel shows the worker's natural setting next to the input (e.g.
`n3766: 43.79`), which tells you the useful scale. The wire accepts
|value| ≤ 10 000; the panel input clamps to ±1000. Two measured facts: at mid
layers (14, 30) ablating the ten strongest workers changed nothing at all, while
at a late layer (38) one strongly negative worker flipped the sentence opening
when reversed. `clear` releases one station or all of them; afterwards the
output matches the untouched model again (tested).

#### The numbers in the right panel

| label | plain meaning |
|---|---|
| norm | how loud that workstation is for this word |
| lens top-3 | what that station alone would say next if it were the last one. Early stations guess nonsense; late stations agree with the final answer. |
| neurons top-32 | the strongest workers at that station for this word, ranked by strength with their sign |
| topk | the model's next-word shortlist with probabilities |
| steered / causal | whether THIS word was actually written while a knob was active |
| xy | a 2-D sketch of the word's path — a projection, and the printed % says how much of it the sketch captures |

#### Which knob for which job

| you want | use |
|---|---|
| it to answer something else mid-answer | inject (a concrete question) |
| a clean restart | edit_prompt |
| a different kind of answer (code vs prose, tone) | steer |
| to see what one worker is doing | steer a neuron |
| it to stop | Esc |

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
| C4 steering specificity | **FAIL pre-fix** — calibrated, 20 controls · **passes post-fix** (screened, 5 controls; see below) |
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

**C4 post-fix screening (2026-11-19).** Three defects were found and fixed after
that run:

1. `build_steer_vectors()` pooled **all** token positions, so at layers 9–39 the
   contrast difference was ~99.9 % collinear with the *negative* of the
   position-0 attention sink (‖h[0]‖ ≈ 4 800 vs 5–18 elsewhere; the 15-vs-9
   token-count difference turned it into a length artefact, `cos ≈ −1.000`).
   Position 0 is now excluded from pooling, and the default pairs share a
   prefix and suffix around the function body so only the comment differs.
2. The inserted comment sat **above** `def f(x):` while the held-out prompts
   decide inside the body after `def …:\n` — the vector was teaching the wrong
   location.
3. `.hf/granite-hf` itself was wrong: the GGUF→HF conversion had not un-permuted
   the Q/K rotary rows, so every attention head in all 40 layers used scrambled
   positional angles (baseline output was degraded: "Paris Paris, France.",
   "Cats are do not"). Fixed in `registry.py`/`convert.py`, and the artefact was
   corrected in place.

With the corrected vector at coef 1: **T1 +2.37 (L8) / +2.95 (L12) / +3.24
(L14)** against a matched random max of +0.96, and **10/10 held-out sign
consistency** (5 random controls — the full 20-control rerun is pending).
Generated completions at coef 1 write context-appropriate comments
(`# initialize the result` for `def merge(a, b):`,
`# Use aiohttp to get the response` for `async def fetch(url):`) where the same
prompts wrote plain code at coef 0.

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
- **Steering: estimator fixed 2026-11-19, re-certification pending.** The
  direction now carries the intended semantics at coef 1 (§5); the 20-control
  rerun that would retire the old FAIL row has not been run yet.
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
