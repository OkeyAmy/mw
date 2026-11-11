# BRAIN 3D — final spec (v1, from J1 geometry + J2 viz practice)

The brain is the product. Chat is an input box in the margin.

## The object: a trajectory ribbon

An LLM has no 3-D shape. It is a ~1536-D additive bus with no privileged basis;
per-layer data lies on a manifold of intrinsic dimension ~10–35. What we *can*
honestly draw is one token position's path through the layers.

```
for each token:  h0, h1, ... h27   (resid_post per layer, d=1536)
1. mean-centre across layers
2. standardise per dimension            (raw distances are dominated by 1-3 dims
                                         and activations up to 1e5x larger)
3. PCA to 2 components  -> x, y
4. z = layer index                      (a REAL axis, not a projection)
5. ribbon through the 28 points, width = TwoNN intrinsic-dimension estimate
6. colour = logit-lens entropy at that layer
```

Every coordinate is measured. Print the quality of the projection on the frame:
variance explained by PC1+PC2, the ID estimate, and the number of tokens shown.

## Why this and not a polytope or a lobe cloud

| metaphor | claim it makes | why rejected |
|---|---|---|
| superposition polytope | features live on a specific polytope | a toy-model result, not measured here |
| SAE lobe cloud | features are these blobs | an SAE-basis claim we cannot verify |
| **trajectory ribbon** | **only the tensors we actually have** | **chosen** |

## Layout (brain-first)

```
┌───────────┬────────────────────────────────────┬──────────────┐
│ chat rail │           3D BRAIN                 │ layer detail │
│ 300px     │           flex: 1                  │ 320px        │
│ collapsi- │                                    │              │
│ ble       │  z = layer 0..27                   │ top-32       │
│           │  x,y = PCA of standardised resid   │ neurons      │
│ messages  │  ribbons = tokens                  │ steer inputs │
│ composer  │                                    │              │
└───────────┴────────────────────────────────────┴──────────────┘
```

Brain ≥ 55% of viewport at 1440px. Chat collapses to an icon.

## Layer drill-down (the core interaction)

Click a layer on a ribbon → the right panel opens that layer:

1. **Top-32 neurons** for that layer and selected token, as a radial or bar
   cluster: `n6376  41.11`, sorted by activation.
2. **Lens top-3** for that layer with probabilities.
3. **Steer this layer**: coefficient slider (−4…+4) applying the layer's
   contrast vector.
4. **Steer this neuron**: numeric input writing a value into that MLP unit
   (server: `neuron_steer`). "Set n6376 to 0" is the ablation; "set to 80" is
   amplification.

This is the whole point: see the layer move, open it, see what is inside, change
it.

## Scene budgets (measured by J2 on this machine)

- instanced 500 nodes + 2k edges + 4k particles = **0.11 ms/frame**
- 500 separate draw calls = 30 ms/frame → forbidden
- additive glow = 26 ms/frame → forbidden
- one `InstancedMesh`, one `LineSegments`, one `Points`; DPR ≤ 2
- dependency: `three` only (no react-three-fiber), OrbitControls from examples

## Honesty requirements (blocking)

1. Persistent caption: `z = layer · x/y = PCA of standardised residual · correlation`
   → `STEERING ACTIVE · causal` when any intervention is on.
2. Hover always yields numbers, never just colour.
3. A layer with no data is hollow and labelled `no data`.
4. Print projection quality: `PC1+PC2 xx.x% · ID ~NN · N tokens`.
5. 2D unroll toggle (heatmap + trajectory plot) reachable in one click.
6. No idle animation that implies computation.

## Visual language v2 — black base, semantic accent

Owner's rule (2026-11-11): black UI is fine; **purple is banned**.

Base surfaces
```
canvas        #050505
panel         #0a0a0a
border        1px #1a1a1a
text          #e8e8e8
muted         #8a8a8a
radius        6px      shadow: none      gradients: none
type          Space Grotesk (UI) + mono for every number
```

Semantic accents — one colour of ink on black paper, never a palette
```
mint   #9fe8c8   normal activity, live tokens
amber  #e8c89f   steering active / causal state
red    #e89f9f   error, timeout, disconnected
```
No violet, magenta, or neon anywhere.

## Grammar borrowed from the reference, data bound

`del/ai-neurons.html` is the visual reference the owner expects. Its *grammar*
is good; its data is fake (`rnd()` activations, random wiring, token timer).
Adopt the grammar, replace every number with a measurement:

| reference element | our source |
|---|---|
| layer columns | 40 real layers, x = layer index |
| node cluster per layer | `min(12, len(layers[l].neurons))` nodes |
| node radius / glow | `norm` / neuron activation value |
| particles along edges | one per generated token, speed from `ms` |
| HUD counters | tokens, active neurons, current layer, tok/s |

The horizontal PCA `xy` view remains as an optional geometry toggle; the default
is the layer-column brain above, because column order is a real axis.

## Right-panel layout bug (must fix)

The panel is cut short. Flex/grid children default to `min-height:auto`, so an
inner scroll area never activates. Every column needs `min-height:0; overflow:hidden`
and the panel body `min-height:0; overflow-y:auto`. Split the panel into tabs
(Trace | Neurons | Steer) so it can never overflow.

## Acceptance criteria

1. 28 layer planes; one ribbon per generated token.
2. Ribbon width tracks the ID estimate; colour tracks lens entropy.
3. Hover any node → `L14 · "def" · norm 36.1 · top-1 "def" 0.42 · n6376 41.1`.
4. Click layer → drill-down shows top-32 neurons and both steer controls.
5. Setting a neuron to 0 changes the next token (verified via the trace).
6. `npm run build` passes; frame time < 4 ms with 20 ribbons.
