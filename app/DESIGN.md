# DESIGN.md — mod_wei instrument UI

Adapted from the Polestar system (single-weight type, 0px geometry, hairline
borders, no shadows). **This is not a marketing page** — hero bands, product
cards and footer columns are deliberately excluded. What transfers is the
restraint, applied to a dark instrument surface.

## Colour

| token | value | use |
|---|---|---|
| `canvas` | `#050505` | app background |
| `panel` | `#0a0a0a` | chat rail, brain panel, right panel |
| `stage` | `#0f0f0f` | canvas viewport inside the brain panel |
| `hairline` | `#1a1a1a` | all 1px borders and dividers |
| `hairline-strong` | `#262626` | focused input, active tab underline |
| `ink` | `#e8e8e8` | primary text |
| `ink-muted` | `#8a8a8a` | labels, metadata, captions |
| `ink-faint` | `#5a5a5a` | placeholders, disabled |
| `mint` | `#9fe8c8` | normal activity, live tokens |
| `amber` | `#e8c89f` | steering active / causal |
| `red` | `#e89f9f` | error, timeout, disconnected |

Rules: no purple, no gradients, no glow, no glassmorphism, no shadows.
Colour is one ink on black paper, used only for meaning.

## Typography

One weight. 400 everywhere — hierarchy comes from size and tracking, not weight.

| token | size | tracking | line-height | use |
|---|---|---|---|---|
| `display` | 30px | -0.6px | 1.07 | panel titles, empty-state headline |
| `title` | 20px | -0.4px | 1.2 | section headers |
| `body` | 16px | -0.2px | 1.4 | message text, prose |
| `ui` | 14px | -0.1px | 1.3 | buttons, tabs, labels |
| `caption` | 12px | 0 | 1.2 | metadata, captions |
| `micro` | 11px | 0.04em (uppercase) | 1.2 | HUD labels, status |

Family: `Space Grotesk` for UI, `JetBrains Mono` / `SF Mono` for **every number**
(token ids, norms, probabilities, layer counts, tok/s). Numbers are tabular.
Sentence case. No uppercase headlines — only `micro` labels are uppercase.

## Geometry

- radius: **0px** on every rectangle. Only the live-status dot is `50%`.
- borders: `1px solid hairline`. No border on flat bands.
- no drop shadows anywhere.
- spacing scale: 2 / 4 / 8 / 16 / 24 / 32 / 48 / 64.

## Components

- **tabs** — text only, `ui`, muted at rest, `ink` + 1px `hairline-strong`
  underline when active. No pills, no fills.
- **input** — transparent background, no box. 1px bottom border `hairline`,
  turns `hairline-strong` on focus. Placeholder `ink-faint`.
- **button** — text link with a trailing `→`, `ink`, no background, no border.
  Primary action only. `:active { transform: scale(0.98) }`.
- **panel header** — `micro` uppercase label left, mono value right.
- **chip** (injected text, errors) — 1px border, transparent fill, `caption`;
  colour by meaning (mint/amber/red), never a filled block.
- **slider** — 1px track `hairline`, 1px-high filled portion in the active
  accent, square 8px thumb, no rounding.
- **HUD** — bottom centre, `micro` labels above mono values, separated by 24px;
  exactly the reference's rhythm, driven by real counts.

## Layout rules

- Every column: `min-height: 0; overflow: hidden`. Inner scroll areas:
  `min-height: 0; overflow-y: auto`. This is what fixes the cut-short panel.
- Right panel splits into tabs (Trace / Neurons / Steer) so it never overflows.
- Brain viewport is the largest surface; chat is a 300px rail, collapsible.

## What NOT to take from Polestar

- no hero stage, no product grid, no footer inversion block, no 110px numerals
- no photography treatment (our "photography" is the live canvas)
- no 112px editorial gutters (this is a tool, not a magazine)
