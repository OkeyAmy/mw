# BRAIN CANVAS — visual specification

The right panel must show the model as a **network**, not a table. Two modes.
Canvas 2D (`requestAnimationFrame`), devicePixelRatio-aware, no WebGL dependency.

## Data available per token (from PROTOCOL.md)

```
layers[l] = { l, norm, lens[[token,prob] x3], neurons[[index,value] x3] }
```

Everything on screen derives from this. No extra backend fields are required.

## Mode 1 — OVERVIEW (default)

```
   token strip (bottom, scrolls)
        │
 [L0]→[L1]→[L2]→ … →[L27]        ← 28 layer nodes, wrapped 7 per row
   ▲     ▲     ▲            ▲
  attn  attn  attn        attn   ← 12 head dots per node group
  mlp   mlp   mlp         mlp    ← top-3 neuron sparks
```

Layout
- 7 columns × 4 rows grid, serpentine order (row0 L→R, row1 R→L, …) so edges
  never cross the whole panel.
- Each layer = one **group** of 3 elements:
  - **spine**: a filled circle, radius `4 + 6 * norm/maxNorm`
  - **attention arc**: 12 small dots on an arc above the spine, radius scaled by
    the layer's lens top-1 probability
  - **neuron sparks**: 3 short radial lines leaving the spine, length ∝ neuron
    value / maxNeuronValue, angle spread ±40°
- Edges: quadratic curves between consecutive spine circles. Stroke width
  `1 + 3 * norm/maxNorm`, opacity `0.15 + 0.5 * norm/maxNorm`.

Animation
- Particles travel along each edge, one per token, speed `≈ 220px/s`, life = edge
  length. Spawn on new token; particle colour = accent.
- A soft pulse ring expands from the spine whose `norm` increased most.
- Idle: gentle breathing (±2% radius, 2.2s period) so the graph feels alive
  without implying computation.

Token strip
- One vertical slice per generated token, width 10px, scrolls right→left.
- Colour = top-1 lens probability at the last layer (blue low → green high).
- Clicking a slice selects that token and drives the layer trace panel.

## Mode 2 — LAYER ZOOM (click a layer node)

- Centre: the selected layer's spine.
- 12 attention heads as a ring of arcs; arc opacity = head's contribution proxy
  (until the backend exposes per-head scores, use lens top-1 probability and
  label it "proxy").
- Top-3 neurons as labelled rays: `n1234 · 4.5`.
- Lens top-3 tokens as three bubbles above the centre with their probabilities.
- Back button returns to overview.

## Interaction contract

| action | result |
|---|---|
| hover spine | tooltip: `L14 · norm 36.1 · top-1 "def" 0.42 · n1234:4.5` |
| click spine | enter Mode 2 |
| click token slice | select token, layer trace panel updates, canvas highlights slice |
| drag horizontally | pan the token strip only |
| `Esc` | leave Mode 2 |

## Honesty rules (non-negotiable)

- A persistent caption on the canvas:
  `activation magnitude — correlation, not meaning`
  and, when `coef != 0`, `STEERING ACTIVE · causal`.
- Do not draw a "thought" that isn't in the data. No invented edges, no fake
  neurons, no animation that implies processing when idle.
- If a layer is missing from the payload, draw it hollow and label `no data`.

## Performance budget

- 28 groups × (1 spine + 12 dots + 3 sparks) = ~450 primitives. Fine at 60fps.
- Redraw only when: new token, animation tick, hover/selection change.
- Cap particles at 64; drop oldest.
- Canvas must not exceed 8 MB of backing store (scale DPR ≤ 2).

## Acceptance criteria

1. With no tokens, the canvas shows the 28 layer nodes in the serpentine grid.
2. After one token, every layer node has a radius derived from its `norm`, and
   three neuron sparks are visible.
3. New tokens animate; the strip scrolls; selecting a slice updates the panel.
4. Clicking a layer opens Mode 2 and `Esc` returns.
5. The caption is always visible and flips to `STEERING ACTIVE` when coef ≠ 0.
6. `npm run build` passes with the component included.
