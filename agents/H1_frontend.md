# H1 — frontend (`web/`) — black-instrument redesign

Scope: `web/` only. Contracts: `app/PROTOCOL.md` (transport/data),
`app/DESIGN.md` (visual system), reference grammar from `del/ai-neurons.html`
(imitated, never its fake data). Nothing in `app/`, `code/` or `tests/` edited.

## What it is

A brain-first instrument: 300 px collapsible chat rail · flex canvas · 320 px
tabbed detail panel. The canvas is a 2D layer-column network (the reference's
grammar) driven exclusively by the payload: `norm`, `neurons`, `lens`, `ms`,
`xy`/`pca`, `steered`/`coef`/`steer_layer`.

```
web/
  package.json            next 15.5.29, react 19, typescript 5 — no UI kit
  next.config.mjs         /api/* rewrite to :8000; /ws HTTP-only (see README)
  app/layout.tsx          Space Grotesk + JetBrains Mono via Google Fonts
  app/page.tsx            shell, selection, modes, export, onboarding
  app/globals.css         DESIGN.md tokens: black base, hairlines, 0 radius
  components/BrainGraph.tsx   canvas columns, nodes, edges, particles, HUD
  components/LayerDetail.tsx  Trace | Neurons | Steer tabs
  components/LayerTrace.tsx   all-layer readout inside the Trace fold
  components/MessageList.tsx  transcript + inline injected chips
  components/Composer.tsx     prompt, mid-stream inject, cancel, max-new
  components/Onboarding.tsx   3-step first-visit tour
  lib/useGeneration.ts    socket + REST state machine, conversation, reconnect
  lib/types.ts            wire types (xy, pca, steered/coef/steer_layer, injected)
  lib/format.ts           display helpers
```

Dependencies: `next`, `react`, `react-dom` only. `three` was removed with the
3D scene.

## Layout bug (fixed)

`.shell > * { min-width:0; min-height:0; overflow:hidden }`; `.messages`,
`.detail-body` and the rail are `flex:1; min-height:0; overflow-y:auto`. The
right column is a flex column of header + tabs + scrolling body, so content
cannot be cut short. Tabs (Trace / Neurons / Steer) keep each concern short.

## Brain binding (nothing invented)

| element | data |
|---|---|
| x | layer index 0…L−1, L from `/api/info` |
| nodes | `layers[l].neurons` (top-3 stream; ≤12 if present) |
| y | activation value, log-scaled against the token peak |
| radius | `layers[l].norm` within the token |
| hue/alpha | mint only; alpha from activation relative to peak |
| edges | l→l+1 adjacency (+ faint l→l+2); structure, labelled as such |
| particles | one per layer transition per token, duration from that token's `ms` |
| HUD | tokens · active neurons (≥25 % peak) · layer · tok/s |
| tooltip | `L14 · norm 36.1 · n6376 · 41.11` |

No idle animation, no glow (DESIGN bans it), no invented edges or neurons. A
missing layer is a hollow circle. Clicking a column opens its layer; hover gives
numbers.

`geometry` toggle: real `layers[l].xy` shared PCA, layer order preserved; if a
frame has no `xy` it says `no xy on this frame — geometry unavailable` instead
of drawing a proxy. Default view is columns.

Caption: `measured activations · correlation, not meaning`, flipping to
`STEERING ACTIVE · causal` iff the frame on screen has `steered: true`.
`coef`/`steer_layer` missing on a steered frame → visible red warning.

## Conversation / persistence

- Chat sends `messages: [{role, content}…]` (full transcript), not `prompt`.
  Assistant text becomes history at `done`; "New chat" is the only clear.
- Sending does not switch `activeId`; the previous turn stays on screen until
  the new turn's first token.
- The graph redraws from state each frame; no scene is disposed per token. There
  is no 3D scene any more.

## Design system applied

Black `#050505` / panels `#0a0a0a` / 1 px `#1a1a1a` hairlines. Ink `#e8e8e8`,
muted `#8a8a8a`, faint `#5a5a5a`. Single weight 400; Space Grotesk UI, JetBrains
Mono numbers (tabular). Radius 0 on every rectangle, 50 % only on the live dot.
No shadows, gradients, glass, glow, purple. Mint = activity, amber = steering,
red = error. Buttons are text links with `→`; inputs are one underline; tabs are
text with a 1 px underline; HUD uses 11 px uppercase labels over mono values.

## Build — real output

```
$ cd web && npm install
up to date

$ cd web && npm run build
   ▲ Next.js 15.5.29
 ✓ Compiled successfully
   Linting and checking validity of types ...
 ✓ Generating static pages (4/4)

Route (app)                                 Size  First Load JS
┌ ○ /                                     14 kB          117 kB
└ ○ /_not-found                            993 B         104 kB
+ First Load JS shared by all             103 kB

○  (Static)  prerendered as static content
```

`npx tsc --noEmit` → exit 0.

## Verification

Headless Chrome at 1600×1000, backend live (Granite 40L, d2048):

- empty state + layer axis + HUD: `/tmp/opencode/shots/ui8_empty.png`
- streaming a prompt, 18 tokens: `/tmp/opencode/shots/ui11.png`
- clicked L20: drill-down shows 32 neurons from `/api/detail` with the column
  highlighted: `/tmp/opencode/shots/ui12.png`

Observed: 40 columns L0…L39, node radius growing with depth, HUD correct, tabs
scroll, no panel clipping, injected row present while streaming.

## Known caveats

- Streamed frames carry top-3 neurons, so most columns draw 3 nodes
  (`min(12, len(neurons))` is the rule; the wire decides). The drill-down serves
  top-32 for one layer.
- Frame-time budget is not benchmarked; the canvas does one pass per token plus
  particle frames, no per-node DOM.
- neuron apply/ablate acks are shown but the causal claim is the server's; the
  UI only reports `steered`.
- Geometry mode depends on `layers[l].xy`; the currently running backend sends
  it, but a frame without it is labelled, not proxied.
