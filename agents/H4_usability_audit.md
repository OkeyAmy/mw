# H4 — usability / comprehension audit (canvas, copy, first-run, "loads afresh")

**Audited at** 2026-11-11 · screenshot `/tmp/opencode/shots/ui.png` (1680×1050,
Granite 40-layer loaded, `ready`, 0 tokens) · `web/app/page.tsx`,
`web/components/{Brain3D,Composer,LayerDetail,LayerTrace,MessageList,Unroll2D}.tsx`,
`web/lib/{useGeneration,types,format}.ts`, `web/app/globals.css`,
`app/PROTOCOL.md`, `app/BRAIN_3D_SPEC.md`, `app/server.py`,
`code/mw/adapter.py`.

**Method.** Read the live screenshot at 1:1 and at 2× crops (header, rail,
canvas centre, legend, footer, detail panel). Traced every string and every
state transition in source. **Nothing was executed in a browser** — no
frame-time measurement, no screen-reader run. Where a claim depends on runtime
behaviour I say so. Contrast ratios below are computed (WCAG 2.x formula) from
the literal CSS colours, not eyeballed.

**Verdict in one line.** The product's own thesis — "the brain is the product,
chat is an input box in the margin" (`BRAIN_3D_SPEC.md:3`) — is inverted in the
shipped screen: 60% of the viewport is a 1.18:1 wireframe that a first-time user
cannot name, decode, or act on, and the one thing the product promises
(continuity of a trajectory) is destroyed on every send, every 8 tokens, and
every scene rebuild.

Severity uses the project's S1–S4 scale (S1 = ship-stopper).

---

## A. CANVAS — does it read as an LLM brain?

**No.** At rest the canvas is forty identical, near-invisible rectangles stacked
in z. There is no label, no axis, no origin, no direction, no scale, and no
object that changes between L0 and L39. It reads as a wireframe floor seen at a
shallow angle, not as a model. The screenshot makes this concrete: between
x≈305–1330, y≈40–1010 the only high-contrast pixels are two lines of 11px
centred text; the geometry itself is barely above the background.

Measured contrast (WCAG 2.x): plane stroke `0x22303f` at `opacity 0.55`
over the stage gradient `#0b0f14` composites to `#18212c` → **1.18:1**. WCAG
1.4.11 requires **3:1** for a graphical object that carries meaning. Every one
of the 40 planes fails. (Ribbons are not drawn at all until a token exists, so
the only visible object at rest is the failing one.)

### A-1 … A-10

| # | sev | evidence | what is wrong | fix |
|---|---|---|---|---|
| **A-1** | **S1** | `Brain3D.tsx:381-406`; screenshot x305–1330/y40–1010 | **40 unlabelled planes at 1.18:1.** No text anywhere maps a plane to a layer number, so the header claim `layer 0 → 39` cannot be located in the image. | Brighten to ≥3:1 (`#2c3f52` @ 0.9), ramp opacity with depth, label every 5th plane at the near edge (`L0 L5 … L35 L39`) with 10–11px mono, and draw a corner spine + ticks. |
| **A-2** | **S1** | `Brain3D.tsx:624-632`, `:649-662`; screenshot centre (827,488)-(900,558) | **No affordance and no instruction.** The only canvas copy is a 2-line paragraph at 11px/3.54:1 saying "ask the model something". Nothing says drag = orbit, scroll = zoom, click = open, or that the planes are layers. `cursor: grab` is the sole hint. | Put a 3-step card in the canvas: ① ask below ② drag to orbit ③ click a node to open a layer. Add a one-line interaction strip (`drag orbit · scroll zoom · click node`). Dismiss on first token. |
| **A-3** | **S2** | `Brain3D.tsx:628-629` vs screenshot header | Overlay says **"28 layer planes … layer 0 → 27"** while 40 planes are drawn and the header says 0→39. The one sentence that explains the object contains the wrong number. | Drive from `layerCount`: `{layerCount} layer planes … layer 0 → {layerCount-1}`. |
| **A-4** | **S2** | `Brain3D.tsx:141-143`, `651`, `672`; screenshot footer x307/y1007 and legend x305/y980 | **The canvas claims `proxy` on a model that has real `xy`.** `buildView()` sets `mode:"proxy"` when `traces` is empty (no newest frame), so the pre-generation legend reads `x/y = proxy` and `summary space (proxy)`. First token flips it to `shared PCA`. The user's first impression is a disclaimer that is false. | Tri-state label: no data → `x/y = shared PCA (waiting for first token)`; data → `PCA 63.0%`. Never say "proxy" unless the frame actually lacks `xy`. |
| **A-5** | **S2** | `Brain3D.tsx:520-531`; `page.tsx:182` | **The current steer layer is invisible.** The rail says `steer L8`, the right panel says `layer 8`, and no plane is marked. Only a selected layer gets a 1px white frame, and only after a click. | Always draw the active steer layer plane in `--accent` + its label; flash it once when the value changes. |
| **A-6** | **S2** | `Brain3D.tsx:92-95`, `:217-223`, `:660` | **Legend contradicts the code and the spec.** Legend: `colour = 1 − H(top-3)`; code colours by `confidence = 0.65·p₁ + 0.35·(1−H)`. Legend: `width/radius = norm`; spec criterion 2 says width must track the TwoNN ID estimate. So the frame is honest about a quantity it doesn't plot. | Either implement ID (not in payload) or relabel to what is plotted: `width = residual norm` and `colour = lens confidence`. One of the two, not neither. |
| **A-7** | **S2** | `Brain3D.tsx:593-600` (click only via `hoverKeyRef`), `:602` | **Layer planes are not clickable.** Spec: "Click a layer on a ribbon". Only instanced spheres are raycast, so with 0 tokens the entire canvas is inert — you cannot open a layer from the brain at all. | Raycast a second object (40 invisible plane quads, or `z`→layer from the hit point) and open `LayerDetail` on plane click. |
| **A-8** | **S2** | `Brain3D.tsx:308-318`; screenshot | **No z-direction cue.** The stack is viewed almost edge-on from (9.5,−12,14) toward the mid-plane, so 40 planes collapse into moiré. You cannot tell which end is the input or the output. | Raise/offset the default camera to a 3/4 view (e.g. `(11,−11,19)` target `z=mid`), and put `L0`/`L39` end caps with arrows. |
| **A-9** | **S3** | `Brain3D.tsx:479-490` (sphere r=0.055, ×1.7 selected) | **Node hit targets are ~5–9 px.** Hovering a node is the only path to the drill-down; missing by 3 px silently does nothing. No hover growth, no hit padding. | Raycaster threshold / larger invisible hit sphere (r≈0.14), scale-on-hover, and a "nearest node within 20 px" fallback. |
| **A-10** | **S3** | `Brain3D.tsx:287-367` deps `[layerCount]`; `page.tsx:70` | **Renderer/camera remount when `/api/info` lands.** `layers` defaults to 28, so a 28-plane scene is built, then torn down and rebuilt at 40 when info arrives (camera, controls, renderer all recreated). | Don't create the scene until `layers > 0` known, or update plane geometry in place and keep camera state. |

### BRAIN_3D_SPEC acceptance criteria — pass/fail

| # | criterion | result | evidence |
|---|---|---|---|
| 1 | 28 layer planes; one ribbon per token | **PARTIAL** — planes drawn, correct count from `info`; ribbons only for the *active message*, capped at 20, and truncated to the current PCA epoch (≤8) | `Brain3D.tsx:32, 132-133` |
| 2 | ribbon width tracks ID estimate; colour tracks lens entropy | **FAIL** — width = `0.03+0.08·norm/maxNorm`; colour = composite confidence | `Brain3D.tsx:221-223`, `92-95` |
| 3 | hover any node → `L14 · "def" · norm · top-1 · n####` | **PASS** (mouse only; string includes extra `H`) | `Brain3D.tsx:566-579` |
| 4 | click layer → top-32 neurons + both steer controls | **PARTIAL** — node click only, plane not clickable; both controls exist | A-7, `LayerDetail.tsx:310-425` |
| 5 | neuron 0 changes next token, verified via trace | **UNVERIFIABLE in UI** — ack text only, no before/after diff | `LayerDetail.tsx:359-363` |
| 6 | build passes; <4 ms/frame @ 20 ribbons | **NOT MEASURED** — but geometry is fully rebuilt per token, which is the opposite of the budget the spec was written to protect | A-10, D-3 |

Honesty requirements (`BRAIN_3D_SPEC.md:73-81`): #1 caption present ✓,
#2 hover numbers ✓, #3 *"a layer with no data is hollow and labelled `no data`"*
**FAIL** (no per-layer labels at all), #4 projection quality **PARTIAL** (prints
`PCA xx.x%` but no ID estimate — it is not in the payload), #5 2D unroll ✓,
#6 no idle animation ✓.

---

## B. COPY ACCURACY — every stale `28` / `0→27` / `proxy`

Model is **Granite, 40 layers, 40 vectors**, `/api/info` reports `layers: 40`,
the header correctly prints `layer 0 → 39`, and the footer correctly prints
`40 layers`. These strings disagree with that same screen:

| # | file:line | literal | comment |
|---|---|---|---|
| **B-1** | `components/MessageList.tsx:49` | ``The brain draws one ribbon per token through 28 real layer planes.`` | The first paragraph a new user reads. Says 28 while the rail says `40L` six centimetres above. |
| **B-2** | `components/Brain3D.tsx:628-629` | ``28 layer planes are drawn from the model shape. Ask the model something and each generated token adds one ribbon through layer 0 → 27.`` | Canvas empty-state copy; also wrong on both numbers. |
| **B-3** | `components/Brain3D.tsx:4` | ``* The brain: one trajectory ribbon per generated token through 28 real layer planes.`` | Module doc, same error. |
| **B-4** | `components/Brain3D.tsx:381` | ``// 1) layer planes — one LineSegments, 28 rectangles.`` | Comment; drifts silently. |
| **B-5** | `components/LayerTrace.tsx:121` | ``rows top → bottom = L{rows[0]?.l ?? 27} → L0`` | Fallback hardcodes 27; would print L27→L0 if a trace had no layers. |
| **B-6** | `lib/types.ts:133` | ``export const DEFAULT_LAYERS = 28;`` | Dead constant (never imported) that will be picked up by the next author. Delete. |
| **B-7** | `app/page.tsx:70` | ``info?.layers && info.layers > 0 ? info.layers : 28`` | Fallback paints a 28-plane brain before info arrives — a wrong object, not a neutral one. |
| **B-8** | `components/{Brain3D:264,LayerDetail:50,Unroll2D:53}` | ``Math.max(1, layers || 28)`` | Same fallback, three copies. |
| **B-9** | `app/PROTOCOL.md:22,27,83` | ``"layers": 28``, ``"steer_vectors": 28``, ``"n_layers": 28`` | Protocol example still teaches 28. Harmless as an example, damaging as documentation for this model. |
| **B-10** | `components/Brain3D.tsx:651,672` + screenshot legend/projection | ``x/y = proxy``, ``summary space (proxy) · 0 tokens`` | Stale *and* state-wrong: shown on a 40L model with real `xy`, only because no token has arrived. See A-4. |
| **B-11** | `components/Unroll2D.tsx:238-239` | ``x/y in the 3D view are summary space (not PCA); this unroll shows layers and tokens directly.`` | **Flatly false now** and directly contradicts the 3D caption `x/y = shared PCA of residual`. Two panels on the same screen assert opposite things. |
| **B-12** | `app/page.tsx:168-170` | ``{info?.steer_vectors} vectors`` / rail | "40 vectors" reads as 40 directions; it is one contrast vector per layer. Say `40 layer vectors (L0–L39)`. |
| **B-13** | `app/page.tsx:176` | ``GET /api/info · not ready · prompts blocked`` | Developer log line in the user-facing rail. |

**Jargon inventory with no in-product definition** (each one is a comprehension
tax on the first run): `correlational`, `causal`, `coef`, `norm`, `lens top-3`,
`top-32`, `H`, `token frame`, `steer_layer`, `trace sum`, `ablate`, `amplify`,
`contrast vector`, `coef × layer mean residual norm`, `summary space`, `PCA`,
`epoch/stale basis` (implicit), `inject`. There is no glossary, no `title` on
the slider labels, no `?` affordance anywhere.

---

## C. FIRST-RUN FLOW — landing → seeing a neuron

Counted against the screenshot. **Minimum 4 user actions + 1 network wait; 0 of
them are discoverable from the canvas.**

1. Land. Screen: 300px rail (empty chat), 60% invisible grid, 320px panel that
   says `no layer`. Nothing is focused, nothing is highlighted.
2. Find the composer: bottom of the *left* rail, 12.5px placeholder, dimmed
   Send. It is visually weaker than the dead centre of the canvas.
3. Type + Send. Nothing tells you that the model has no memory, or that the
   brain will wipe. Wait with no ETA (no prefill indicator, no first-token
   countdown, no token/s until the second token).
4. Wait for ribbons. They appear as a thin strip inside the tunnel; no legend
   explains what a line is.
5. Hover a ~5px sphere (no keyboard path). Tooltip appears — the first readable
   evidence that this is a model.
6. Click it → right panel opens. If the top-3 from the stream is already in
   hand the panel still blocks on `GET /api/detail`; `loading top-32…`.
7. Read: `layer`, `norm`, `Lens top-3`, `Neurons`, `Steer this layer`,
   `coef × layer mean residual norm`, `0 = off`. Two of these have no
   explanation of *when* they take effect.

**Ambiguity register**

| question a first-timer actually asks | current answer on screen |
|---|---|
| What is a layer? | Never stated. 40 identical planes with no labels. |
| Why 40 rows / planes? | Never stated. (Answer: one per transformer block, L0→L39.) |
| What is `coef`? | `coef × layer mean residual norm` — a formula, not a meaning. |
| What does `inject` do? | Only appears *during* streaming; no pre-generation hint except 11px chat text. |
| What changed when I moved the layer slider? | Nothing visible; the value changes in two places. |
| What is `correlational`? | Explained only in the 9px footer, in different words. |
| Does it remember my last message? | **No** (see D-1), but the transcript implies yes. |
| What is a "token"? | Never defined; the unit of the whole product. |
| How do I get from a ribbon to a neuron? | One hint line in the right panel, 9.5px, below the fold of the slider. |

**Onboarding is split across three places that don't reference each other**:
rail `empty-chat` (`MessageList.tsx:42-57`), canvas overlay
(`Brain3D.tsx:624-632`), right-panel hint (`LayerDetail.tsx:255-258`). None of
them is anchored to the thing it describes.

---

## D. "LOADS AFRESH" — the exact mechanisms

The owner's report is correct and there are **seven** independent causes. Ranked
by how much they produce the perception.

| # | sev | mechanism | file:line |
|---|---|---|---|
| **D-1** | **S1** | **No conversation history exists.** On send the client transmits `{type:"chat", prompt, max_new, layer, coef}` — only the new text. The server formats a template containing exactly one user turn and starts with `past = None`. So every message is a cold prefill of a fresh single-turn context; the assistant's previous replies are decoration in the transcript, never context. Ask "my name is X", then "what is my name?" — the model has never seen turn 1. This is the literal reading of "every message it loads afresh". | `lib/useGeneration.ts:522-529`; `app/server.py:429-439`, `:415-433`; `code/mw/adapter.py:460-472` |
| **D-2** | **S1** | **The canvas is wiped on every send.** `send()` appends a new *empty* assistant message and points `activeId` at it; `activeMessage` → `traces = []`; `Brain3D` sees no tokens and renders the `no trajectory yet` overlay. Every ribbon from the previous turn disappears instantly rather than staying as history. Old turns are reachable only by clicking `inspect` on a message. | `useGeneration.ts:500-519`, `634-653`; `page.tsx:72`, `292-298`; `Brain3D.tsx:602, 624` |
| **D-3** | **S2** | **The whole scene is disposed and rebuilt on every token.** The geometry effect depends on `view`, which is `useMemo` over `traces` — a new array identity per token. It calls `disposeGroup()` (disposing every geometry/material → shader churn), rebuilds 40 planes + ribbon mesh + `InstancedMesh`, and calls `setHover(null)`. During streaming the tooltip therefore dies on every token, and orbit/hover state churns under the cursor. | `Brain3D.tsx:265, 370-378, 534` |
| **D-4** | **S2** | **The picture truncates every 8 tokens.** The backend re-fits the shared PCA basis every 8th token (`_pca_n % 8 == 0`) and `var_explained` changes; `basisRunStart()` then hides every frame from the previous epoch and prints `N older ribbons hidden (stale basis)`. In a 48-token answer the visible trajectory is only ever the current ≤8-token epoch — the product's core claim, a path through the layers, is repeatedly deleted. | `code/mw/adapter.py:421`; `Brain3D.tsx:101-133, 675-677` |
| **D-5** | **S2** | **Renderer remounts when model info arrives** (28→40 planes) and **whenever you toggle 2D/3D** (Brain3D unmounted, scene destroyed). The camera you set is lost. | `Brain3D.tsx:287-367`; `page.tsx:290-307` |
| **D-6** | **S3** | **The drill-down is reset on send.** `setSelectedIndex(-1)` → `selectedTrace` null → panel reverts to `no token selected`; `useEffect` clears the chosen neuron and its value. Whatever you were inspecting closes when you ask the next question. | `useGeneration.ts:516`; `LayerDetail.tsx:58-63, 98-101` |
| **D-7** | **S3** | **Collapsing the chat rail deletes an unsent draft.** The rail is conditionally rendered, so `Composer` (which owns `text`, `maxNew`, `injectText` in local state) unmounts. Type, click `‹`, click `›` → the prompt is gone. Same for the message-list scroll position. | `page.tsx:158-227`; `Composer.tsx:25-29` |
| **D-8** | **S4** | `/api/info` + `/api/health` are polled every 4 s forever, re-rendering the page and flipping `modelReady` if the backend hiccups mid-session (composer disables with no explanation of why it came back). | `useGeneration.ts:596-631`; `page.tsx:121, 200-204` |

Non-mechanism, for the record: `send()` does **not** remount `Brain3D` or
`LayerDetail` (no `key` props), the WebSocket is created once and is not
reconnected on send, and `messages` is append-only. The "reload" is D-2 + D-3 +
D-4, on top of the real semantic reset in D-1.

---

## E. ERROR / STATE VISIBILITY

| # | sev | finding | evidence | fix |
|---|---|---|---|---|
| **E-1** | **S2** | **A dead backend is indistinguishable from a loading model.** `/api/info` failure → `catch { setInfo({ready:false}) }`; the chip renders `model loading`, the rail renders `model loading…`, the banner says prompts are blocked. Forever. No timeout, no "cannot reach the backend", no retry button (the only `retry` is bound to `lastError`, which a failed fetch never sets). | `useGeneration.ts:596-631`; `page.tsx:121, 173-178, 200-204` | Track `infoState: loading | ready | unreachable`; after 2 failed polls set `lastError = "backend not responding at :8000"` with the existing retry. |
| **E-2** | **S2** | **Model-loading is shown but never progresses.** Three simultaneous indicators (dot, chip, banner) and zero information: no elapsed, no "this takes ~40 s", no weights path, no log tail. A 3-minute load looks hung. | `page.tsx:174-177, 200-204, 243-250` | One banner: spinner + elapsed seconds + `GET /api/info` state, poll 4 s → 1 s while loading. |
| **E-3** | **S3** | **`health.busy` is fetched and never rendered.** A second tab generates nothing and sees nothing until it sends and the server answers `server busy`. | `useGeneration.ts:171, 615`; `page.tsx` never destructures `health` | Show `another session is generating` chip when `health.busy`, disable Send with that reason. |
| **E-4** | **S3** | **Two banners for one error** (rail + stage). On the stage, `dismiss` clears `notice`, not `lastError` (H3-11, still open) — the button appears to do nothing. | `page.tsx:186-204, 271-288` | One banner region; `dismissError` for both fields. |
| **E-5** | **S3** | **No send watchdog.** If the socket accepts `chat` and then never streams, `streamingRef` stays true, the composer is locked, Cancel reports "nothing to cancel". (H3-17, still open.) | `useGeneration.ts:519-530, 536-541` | 10 s timer on first token → error state + retry. |
| **E-6** | **S4** | Errors are strings from the backend (`TypeError: …`, `HTTP 404`) rendered verbatim; `ws.onerror` prints the raw `ws://host:8000/ws` URL. | `useGeneration.ts:431-433`; `LayerDetail.tsx:73-76` | Map to human text, keep the raw string behind a `details` fold. |
| **E-7** | **PASS** | Streaming, cancel, per-message error, `/api/detail` failure, unparseable frame, and the incoherent-steering warning all have visible text. The app is honest about failure — it is only silent about *unreachability*. | — | — |

---

## F. ACCESSIBILITY

| # | sev | finding | measurement / evidence |
|---|---|---|---|
| **F-1** | **S2** | Canvas geometry is below the 3:1 graphical minimum. | planes **1.18:1**; legend `#5f6b7a` @9px = **3.54:1**; projection `#56667a` @9px = **3.27:1**. Text needs 4.5:1 below 18.66px. (`globals.css:657-760`) |
| **F-2** | **S2** | **No visible focus anywhere.** No `:focus-visible` rule exists; the only focus style is `.composer textarea:focus` (border colour). Tabbing through Send / 2D unroll / export / sliders / neuron rows is invisible. | `globals.css` (grep: no `focus-visible`), `:497` |
| **F-3** | **S1** | **The core interaction has no keyboard path.** The 3D view is `<div role="application">` with no `tabIndex` and no key handler; tokens are picked by raycast on pointer events. The 2D unroll's cells are SVG `<rect onClick>` — not focusable. So a keyboard user cannot select a token or a layer at all, and therefore cannot reach the neuron drill-down. | `Brain3D.tsx:604-616, 593-600`; `Unroll2D.tsx:113-124` |
| **F-4** | **S3** | **Screen-reader flood while streaming.** `role="log" aria-live="polite"` on the transcript, with the assistant bubble re-rendered on every token; the cursor and per-token spans are additions, so each token is announced. | `MessageList.tsx:30-41` |
| **F-5** | **S3** | Sliders are named only `layer` and `coef` via `<label htmlFor>`. No `aria-valuetext` (screen reader says "3" not "layer 3 of 39"), no `aria-describedby` for the `coef × mean residual norm` meaning, no keyboard hint for the ±4 range. | `LayerDetail.tsx:212-243, 397-409` |
| **F-6** | **S3** | Most secondary text is 9–10px (rail grid 9.5, stage sub 9.5, legend 9, projection 9, footer 9, chip 10, control hints 9.5). The product's entire honesty apparatus is set at footnote size in the dimmest colour. | `globals.css:248, 597, 628, 714, 753` |
| **F-7** | **S4** | Canvas `aria-label` is set on the `<canvas>` element, which is not focusable and not in the a11y tree in a useful way; the container's `role="application"` promises a keyboard contract the widget does not implement. | `Brain3D.tsx:304, 614-615` |
| **F-8** | **S4** | No text alternative for the trajectory. A screen-reader user gets the layer table (good, behind a fold) but nothing that describes the shape. | `LayerTrace.tsx` behind `<details>` |
| **F-9** | **PASS** | `prefers-reduced-motion` is handled; `lang="en"`; dark `color-scheme`; buttons are real `<button>`; composer has `aria-label`. | `globals.css:1215-1223`; `layout.tsx` |

---

## G. TOP 10 FIXES — ranked by (comprehension impact) / (effort)

| rank | fix | why it wins | one-line implementation note |
|---|---|---|---|
| 1 | **Give the model a conversation** | Turns a stateless sampler into the chat the UI already pretends to be; removes the real "loads afresh". | Send `messages[]` (or a session id) with `chat`; server re-prefills transcript once and reuses `past_key_values` per session. Protocol change → note in `agents/H3_ui_audit.md` per `PROTOCOL.md:5`. |
| 2 | **Make the canvas legible: 3:1 planes + `L0/L39` labels + steer-layer highlight** | Without it the centre 60% of the screen is blank; one CSS colour + one `TextGeometry`/CSS2D label loop. | `planeMat` → `#2c3f52` opacity 0.9, depth-ramped; add 9 sprite labels (`L0,L5,…,L39`) at `(-halfX,-halfY,z)`; draw `steering.layer` plane in `--accent`. |
| 3 | **Kill every false string: 28 → `layerCount`, delete "proxy" from the empty state, fix the 2D caption** | Trust is the product; this is copy-only, zero risk. | Replace B-1…B-12 with `layerCount`; in `buildView`, when `traces.length===0` return `mode:"pca"` with `varExplained:null` and label `waiting for first token`. |
| 4 | **Stop wiping the brain on send** | Keeps the trajectory continuous across turns, which is the whole thesis. | Render `messages.flatMap(m => m.traces)` (dim inactive turns) instead of `activeMessage.traces`; keep `selectedIndex` pointing at the last token of the previous turn. |
| 5 | **Onboarding card in the canvas with a real CTA** | Converts "what am I looking at" into a first action in one second. | Absolute-positioned card in `.brain-stage-inner`: 3 numbered lines + a button that calls `areaRef.focus()` (lift a `focusComposer` callback); hide on `traces.length>0`. |
| 6 | **Explain steering where it lives** | The only editable control is currently a formula; one paragraph removes the biggest jargon wall. | Above the sliders: `coef −4…+4 × mean residual norm at L{n}; 0 = off; takes effect on the next token.` Add `aria-valuetext` and rename the rail to `steer L8 · off`. |
| 7 | **Make planes + tokens keyboard/AT reachable** | Unlocks the core interaction for keyboard and screen-reader users; also fixes touch taps. | Add `tabIndex=0` to the container, arrow keys move `selectedLayer`, `[`/`]` move token, Enter opens; mirror a focusable list of tokens + layers in the right panel (no canvas dependency). |
| 8 | **Don't rebuild the scene per token** | Removes flicker, keeps the tooltip alive mid-stream, cuts GPU churn. | Split effects: planes on `[layerCount]`, ribbons on `traces.length` (append only, or rebuild at ≤10 Hz), nodes on selection; stop calling `setHover(null)` outside pointerleave. |
| 9 | **Distinguish "backend unreachable" from "model loading"** | Removes the indefinite dead-end state; small hook change. | `infoState` tri-state; after 2 failed polls set `lastError` + show elapsed seconds while loading. |
| 10 | **A11y polish pass: focus rings, 11px floor, `aria-live` off during streaming** | Cheap, broad, and makes the honesty text actually readable. | Global `:focus-visible { outline: 2px solid var(--accent2); outline-offset: 2px }`; bump 9px→11px; set `aria-live={phase==="streaming"?"off":"polite"}`. |

**Runners-up (worth doing, lower ratio):** clickable layer planes (A-7); bigger
node hit radius (A-9); keep the camera across 2D/3D toggles (D-5); persist the
composer draft outside the rail (D-7); one banner instead of two (E-4).

---

## Appendix — exact replacement strings

```
MessageList.tsx:49   "The brain draws one ribbon per token through {n} layer planes."
Brain3D.tsx:628-629  "{n} layer planes are drawn from the model shape. Ask the model
                      something and each generated token adds one ribbon through
                      layer 0 → {n-1}."
Brain3D.tsx empty    mode: traces.length === 0 ? "pca" : …
Brain3D.tsx:651      "x/y = shared PCA"            (never "proxy" on this build)
Brain3D.tsx:672      "PCA {pct}% · {n} tokens"     or "waiting for first token"
Unroll2D.tsx:238     "x/y in the 3D view are a shared 2-D PCA of the standardised
                      residual trajectory; this unroll shows layers and tokens directly."
page.tsx:168         "{n} layer vectors"
LayerTrace.tsx:121   "{rows[0]?.l ?? layerCount-1} → L0"
types.ts:133         delete DEFAULT_LAYERS
```
