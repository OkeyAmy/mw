# web/ — the mod_wei microscope

Next.js (App Router, TypeScript). **The brain is the product; chat is a 300 px
rail.** Contract: [`../app/PROTOCOL.md`](../app/PROTOCOL.md). Visual system:
[`../app/DESIGN.md`](../app/DESIGN.md). Backend: `app/server.py` on port 8000.

## Run

```bash
# terminal 1 — backend (port 8000), from the repo root
.venv/bin/python app/server.py

# terminal 2 — UI (port 3000)
cd web
npm install
npm run dev            # http://127.0.0.1:3000
```

Production check: `npm run build && npm run typecheck && npm start`.

## Layout

```
┌────────────┬──────────────────────────────────┬──────────────┐
│ chat rail  │      brain canvas (flex: 1)      │ layer detail │
│ 300 px     │  layer 0 → L-1, left to right    │ 320 px       │
│ collapsible│                                  │ 3 tabs       │
└────────────┴──────────────────────────────────┴──────────────┘
```

At 1440 px the canvas is 820 px (~57 %). The rail collapses to 44 px. Below
1000 px the three regions stack.

**Every column** has `min-height: 0; overflow: hidden`; every scroll area has
`min-height: 0; overflow-y: auto`. The right panel is split into
**Trace | Neurons | Steer** tabs with a single scrolling body, so it can never
be cut short.

## The brain — reference grammar, real data

`components/BrainGraph.tsx` is a canvas renderer. Nothing on it is synthetic:

| element | binds to |
|---|---|
| column x | layer index 0 … L−1 (real order, straight from `/api/info.layers`) |
| nodes per column | `layers[l].neurons` (top-3 in the stream; up to 12 drawn if present) |
| node y | the neuron's measured activation value, log-scaled against the token peak |
| node radius | `layers[l].norm` scaled within the current token |
| fill / edge alpha | activation relative to the token's peak (mint, no glow) |
| edges | layer *l* → *l+1* adjacency, plus faint *l* → *l+2*; structural, not weights |
| particles | one per layer transition per generated token; duration derived from that token's measured `ms` |
| HUD | tokens · active neurons (≥25 % of peak) · selected layer · tok/s |
| caption | `measured activations · correlation, not meaning` → `STEERING ACTIVE · causal` iff the frame says `steered: true` |

A layer with no payload is drawn as a hollow circle. Hover gives numbers
(layer · norm · neuron id · value). Clicking anywhere in a column opens that
layer in the right panel.

**Geometry toggle** (`geometry`) uses the real `layers[l].xy` shared-PCA
coordinates when the server sends them; layers stay in real order and points
are labelled. There is no invented axis and no proxy: without `xy` the view says
`no xy on this frame — geometry unavailable`. The default is `columns`.

## Right panel

- **Trace** — token frame (`steered`, `coef`, `steer_layer`), lens top-3,
  collapsible all-layer trace; a frame that claims `steered` without
  `coef`/`steer_layer` gets a red warning and no causal claim.
- **Neurons** — top-32 from `GET /api/detail/{session}/{step}` (stream carries
  top-3; the panel says which), neuron selection, value input,
  `neuron_steer` / `neuron_clear`, ack.
- **Steer** — layer coefficient slider (−4…+4, `0 = off`) and global layer/coef
  sliders; `coef` is in units of the layer's mean residual norm.

## Conversation

Every chat sends the **full transcript**:

```json
{"type":"chat","messages":[{"role":"user","content":"…"},
 {"role":"assistant","content":"…"},{"role":"user","content":"…"}],
 "max_new":48,"layer":8,"coef":0}
```

Sending only `prompt` made every turn a cold single-shot prefill — the "it
reloads" feeling. The assistant text is assembled from the token stream and
becomes history at `done`. **"New chat" is the only action that clears the
transcript and traces**; sending keeps the previous turn on screen until the new
turn's first token arrives.

## Mid-generation injection

While streaming, the composer grows an inject field → `{"type":"inject","text"}`.
On `injected` the text appears inline as a labelled chip
`injected @ step N`, positioned by the token step (orphans appended at the end).
`restarted` closes the turn and opens a new one.

## States (PROTOCOL.md)

| state | trigger | UI |
|---|---|---|
| `connecting` | socket opening | amber dot, composer locked |
| `ready` | `/api/info.ready` + socket open | composer enabled, autofocused |
| `streaming` | first `token` | cursor, send disabled, Cancel + inject |
| `done` | `done` | cursor removed, send re-enabled, trace kept |
| `error` | `error` frame or socket close | banner + retry, composer re-enabled, auto-reconnect |
| `busy` | second `chat` while streaming | prevented client-side |

Reconnect: 0.5 s → 8 s backoff; steering is re-asserted on reconnect.

## Design system (app/DESIGN.md)

Black base `#050505` / panels `#0a0a0a` / 1 px `#1a1a1a` hairlines. Ink
`#e8e8e8`, muted `#8a8a8a`. One weight (400); hierarchy from size and tracking.
Space Grotesk UI, JetBrains Mono for every number. Radius 0 on rectangles; the
only circle is the live dot. No shadows, gradients, glass, glow or purple.
Colour is semantic only: mint = activity, amber = steering, red = error.
Buttons are text links with `→`; fields are a single underline; tabs are text
with a 1 px underline.

## Environment

| variable | default | meaning |
|---|---|---|
| `NEXT_PUBLIC_WS_URL` | `ws://<page-host>:8000/ws` | WebSocket endpoint |
| `NEXT_PUBLIC_BACKEND_PORT` | `8000` | port used to derive the WS URL |
| `BACKEND_ORIGIN` | `http://127.0.0.1:8000` | rewrite target for `/api/*` |

Next rejects a `ws://` rewrite destination and does not reliably proxy upgrades,
so the client connects to `:8000` directly. Behind a proxy that upgrades `/ws`,
set `NEXT_PUBLIC_WS_URL=wss://<host>/ws`.

## Files

```
app/layout.tsx             html/body, fonts, metadata "mod_wei"
app/page.tsx               shell, brain wiring, export, Esc, onboarding
app/globals.css            black instrument system, layout, panel tabs
components/BrainGraph.tsx  canvas layer columns, particles, HUD, hover/pick
components/LayerDetail.tsx Trace | Neurons | Steer tabs
components/LayerTrace.tsx  all-layer table (inside the Trace fold)
components/MessageList.tsx transcript + inline injected chips
components/Composer.tsx    prompt, inject, cancel, max-new
components/Onboarding.tsx  3-step first-visit tour (localStorage, ? to reopen)
lib/useGeneration.ts       socket state machine, conversation, reconnect
lib/types.ts               wire types (incl. xy, pca, steered/coef/steer_layer)
lib/format.ts              display helpers
```

Runtime dependencies: `next`, `react`, `react-dom`. Nothing else.
