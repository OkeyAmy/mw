# H3 — UI / protocol / truthfulness audit

**Audited at** 2026-11-11 14:06 · `app/server.py` md5 `effcb37c…` (705 lines) ·
`tests/test_server.py` md5 `4ebffaef…` · `web/` as of 13:52 · `ui/index.html` 13:32.
The backend was still being edited during this audit (four revisions between 13:34
and 14:01). Every line number below was re-checked against the final hash; the
symbol names are given so a later edit cannot make a finding untraceable.

**Method.** Nothing was taken on trust and the real 1.5B model was never loaded.
I built a **tiny random 28-layer Qwen2** (hidden 64, intermediate 128, vocab
151936 — production layer count and production tokenizer, random weights, 43 MB)
and ran the audited `Engine` and a live `uvicorn` instance against it, plus a
Next.js build/typecheck of `web/`. Where a claim could not be measured I say so
and why.

Artifacts: `/tmp/opencode/{mk_tiny28.py,h3_engine.py,h3_site.py,h3_ws3.py}`.

---

## 1. Defects

Severity uses the project's S1–S4 scale (S1 = ship-stopper, S4 = polish).

| # | sev | file:line | what is wrong | evidence | fix |
|---|---|---|---|---|---|
| **H3-1** | **S2** | `app/server.py:379-393` (payload), `:255`, `:323-329` (global `_steer`), `:605` (`steer` writes it) | **A token frame carries no steering provenance.** There is no `coef`, `steer_layer` or `steered` field, so no client can tell a steered token from an observed one. Steering is also **process-global**: any socket can rewrite it mid-stream. The UI therefore cannot label honestly — which is exactly what PROTOCOL.md:108 and BRAIN_SPEC.md:73 demand. | Live probe: token keys are `['layers','ms','session','step','token','token_id','topk','type']` — no steering field. Client A sent `chat` with `coef=0`; client B sent `steer L8 coef=2.0`; A's next three tokens were generated under the new vector and still carried no flag. (`h3_ws3.py` §1/§4) | Snapshot `{"coef", "steer_layer", "steered"}` into each token payload at yield time; move `_steer` onto the session record so a stream owns its vector. Protocol addition — see §5. |
| **H3-2** | **S2** | `app/server.py:464-470` (`_cancel_current`), `:407` (`CURRENT`), `:619` | **Any client can cancel any other client's generation.** `CURRENT["cancel"]` is a single global slot and `_cancel_current()` sets it whenever *some* generation holds the lock. | Two sockets: A was streaming 256 tokens; B sent `{"type":"cancel"}`; A's stream ended with `{"type":"done","reason":"cancelled","n_tokens":5}`. B never owned that generation. | Store the owning socket id (or the `threading.Event`) with the session; `cancel` may only set its own event; shutdown may set the current one. |
| **H3-3** | **S2** | `app/server.py:270` (`self._pre[i] = h[:, -1, :]`) → `:297-330` (`_build_vectors`) | **The steering vector is built from the last token only**, so at layer 0 it is literally a token-identity direction, not a "comment vs code" direction. All 8 POS prompts end with token **198 (`"\n"`)**; no NEG prompt does (they end with 982 `"):\n"`, 510 `":\n"`, 340 `")\n"`). At L0 `resid_pre == embed(last token)`, so the vector is *by construction* `mean(emb(198)) − mean(emb(982/510/340))`. | Real tokenizer: POS last ids `[198]*8`; NEG last ids `[982,510,510,510,510,340,982,510]`. Measured on the tiny model: `cos(vector[0], emb(198) − mean(emb(neg_last))) = 1.00000`. Norm unit check: `|vector[8]| = 0.5033` vs POS residual norm `0.5033`. (`h3_site.py`, `mk_tiny28.py` + tokenizer probe) | Average the residual over **all** non-padding positions (or all but the final token) when building the vector, and add a regression test that the vector is *not* collinear with the last-token embedding delta. The site itself is correct — see §3 C — this is the estimator, not the hook. |
| **H3-4** | **S2** | `tests/test_server.py:36` (`pytestmark = pytest.mark.slow`) | **The only server-contract tests are excluded from every gate.** `make fast`, `make check` and CI (`pytest tests -m "not slow"`) never run them, so the backend contract has zero automated protection on push/PR. This is the G1-D3 failure mode ("a NameError in load() shipped while 17 fast tests passed") reproduced one layer up. | `make fast` → `32 passed`, none from `test_server.py`. `pytest tests -m "not slow"` → 32. CI workflow runs `pytest tests -m "not slow" -q`. | Split: keep the tiny-model HTTP/WS tests in the default gate (they take ~40 s, no real weights) and mark only real-weight tests `slow`. At minimum add `make contract` = `pytest tests/test_server.py` to `check`. |
| **H3-5** | **S2** | `Makefile:29`; `.github/workflows/ci.yml` (last step); `app/` | **The lint gate is red and does not cover the file under audit.** `make lint` = `ruff check code tests` → **96 errors, exit 2**. `app/server.py` is in neither the Makefile nor CI lint target. CI still ends with `ruff check code tests \|\| true`. | `make lint` → `Found 96 errors` / `make: *** [Makefile:29: lint] Error 1` (17 in `code/proof_chain.py`, 11 each in `steering_test.py` and `microscope.py`, 2 in `code/mw/adapter.py`, …). `ruff check app/server.py` → passes (after the 14:01 edit). | `ruff check code app tests` in **both** Makefile and workflow; delete `\|\| true`; either fix the 96 or narrow rules in `pyproject.toml` with a written rationale. |
| **H3-6** | S3 | `app/server.py:379-399` (yield before the eos check) and `:489-491` | **The EOS token is streamed and rendered.** `stream()` yields, *then* returns on eos; `_worker` sends the payload and increments `n_tokens` before breaking. The client gets a token whose text is `<\|endoftext\|>`. | `tok.decode([151643])` → `'<|endoftext|>'` (real tokenizer). Code order verified: `yield` at :379, `if nxt == eos: return` at :397, `send(...)` then `n_tokens += 1` then `break` at :487-491. | Check eos before yielding (or emit the step with `token: ""` and do not count it). |
| **H3-7** | S3 | `app/server.py:41-43`, `:419-423` | **The legacy `ui/index.html` is still served at `/`.** That page has no `connecting` state, no "0 = off" label, no causal/correlational label and no session handling — a human who opens `:8000` gets a UI that violates PROTOCOL.md. | `curl http://127.0.0.1:8012/` returns the old page (`<title>mod_wei — live microscope</title>`, mtime 13:32); `UI = ROOT/"ui"/"index.html"` still exists. | Delete the `@app.get("/")` handler or 302 to `http://localhost:3000`. |
| **H3-8** | S3 | `web/components/BrainPanel.tsx:31,38-39`; `web/components/BrainCanvas.tsx:26,40-43` | **The "causal vs correlational" caption is driven by the live slider, not by the selected token.** Select an old unsteered token while the slider sits at 3.0 and the panel claims `STEERING ACTIVE · causal` over a purely correlational trace; the reverse also silently mislabels a steered token. This directly breaks PROTOCOL.md:108-109 and BRAIN_SPEC.md:71-73. | `const causal = steering.coef !== 0`. No reference to `selectedTrace` provenance anywhere in either component. | Drive the caption from the *selected trace's* steering snapshot (needs H3-1). Until then, gate the label on "the selected token belongs to a message whose send-time coef was non-zero" at minimum. |
| **H3-9** | S3 | `web/lib/useGeneration.ts:392,405` (+ `:167-186`,`:413`) | **A message's steering label is the send-time value only.** The slider stays live during streaming and emits `steer` frames, so one message can be generated partly at coef 0 and partly at coef 3 while `MessageList` prints one number. | `steerRef.current = { layer, coef }` and `steering: { layer, coef }` are snapshotted in `send()`; `setSteering` keeps firing `steer` frames while `phase === "streaming"`. | Same as H3-1: render per-token provenance (and segment the label when it changes), or lock the slider while streaming. |
| **H3-10** | S3 | `web/lib/useGeneration.ts:420`; `web/components/Composer.tsx` (`max=512`); `app/server.py:47` (`MAX_NEW_LIMIT = 256`) | **UI allows 257–512 new tokens; the server rejects them.** A send looks accepted, then fails with `invalid request: max_new: Input should be less than or equal to 256`. | Server probe: `{"max_new":512}` → that exact error. UI clamps to `Math.min(512, …)`. | Put the bound in one place (a constant exported from `types.ts`, or add `max_new_limit` to `/api/info`). |
| **H3-11** | S4 | `web/app/page.tsx:231-242`; `web/lib/useGeneration.ts:554` | **The error banner cannot be dismissed.** It renders `lastError` but its "dismiss" button calls `dismissNotice`, which only clears `notice`. | `dismissNotice: () => setNotice(null)`; no `setLastError(null)` path except a successful reconnect. | Add `dismissError` (or clear both). |
| **H3-12** | S4 | `app/server.py:41` | **`torch.set_num_threads(4)` at import time** — the exact D25 item AUDIT_FIXES lists as fixed. It also fires when `tests/test_server.py` imports the module. | Line 41, module scope. | Move into `main()` / `Engine.__init__`. |
| **H3-13** | S4 | `app/server.py:249`, `:331-349`, `:297-330` | **`Engine` bypasses the audited `mw.adapter` completely** and re-implements hooks, `final_norm`, `head`, `scaling` and diff-in-means. Today the two agree (`site="pre"` on both sides) but the D6/D10/D11/D12/D13/D16 fixes exist only in the copy that is *not* used. | `self.layers = self.model.model.layers`; `lens_topk` / `_build_vectors` / the three hook builders are all local. | Build `Engine` on `ModelAdapter` (or import its helpers) so there is one implementation. |
| **H3-14** | S4 | `app/server.py:377` vs `:356` | **`topk` probabilities ignore `logits_scaling`** while the logit lens divides by it. With a Granite-style model the next-token distribution would be wrong; invisible on Qwen2 (`scaling == 1.0`). | `logits.softmax(-1)` at :377; `... / self.scaling` at :356. | Divide before softmax in both places. |
| **H3-15** | S4 | `app/server.py:76-115` (`payload_bytes`/`fit_payload`) | **No finiteness guard on emitted numbers.** `round(float(x), n)` passes NaN/Inf straight to `json.dumps`, which emits `NaN`/`Infinity` — invalid JSON. The browser rejects the whole frame and silently drops the token. This is not hypothetical: `AUDIT_FIXES.md` records bf16→fp16 NaN as a real observed failure. | Code path: `norm`/`neurons`/lens probabilities → `round(float(...))` → `json.dumps(allow_nan=True)`; the UI's `JSON.parse` catch sets "unparseable message from the backend" (`useGeneration.ts:326-330`). | Sanitise non-finite values in `fit_payload` (drop the entry or clamp) and assert finiteness. |
| **H3-16** | S4 | `web/lib/useGeneration.ts:36-47` | **The browser bypasses the documented same-origin proxy.** PROTOCOL.md:11 says the Next server proxies `/ws`; the client connects to `ws://<hostname>:8000/ws` directly. Also breaks under `https:` (mixed content) or any non-local host. | `backendWsUrl()` returns `${proto}//${host}:8000/ws`. | Use the rewrite, or make `NEXT_PUBLIC_WS_URL` mandatory outside localhost and document it in PROTOCOL. |
| **H3-17** | S4 | `web/lib/useGeneration.ts:413-425` | **No watchdog for a send that never streams.** `streamingRef` and `phase` are set to `streaming` synchronously before any token; if the server never answers, the composer stays locked and Cancel replies "nothing to cancel". | `streamingRef.current = true; applyPhase("streaming")` precedes `ws.send(...)`; no timer anywhere in the hook. | Start a timeout on send; on expiry go to `error` with a message. |
| **H3-18** | S4 | `web/components/BrainCanvas.tsx:46-54` | **The brain view is a text placeholder.** BRAIN_SPEC acceptance 1–5 (28 nodes, neuron sparks, token strip, zoom, caption) are unmet; the file points at `app/BRAIN_3D_SPEC.md`, which does not exist. | Renders `<div className="brain-placeholder">3D brain canvas — pending spec</div>` plus a `<select>`; no `<canvas>`. | Track as unfinished work, not as a bug — but do not call the panel "built". |

---

## 2. Validation blocks (VALIDATION.md format)

```
HYPOTHESIS: coef=0 is bit-identical to no steering.
TEST:       Engine(tiny28).forward logits for four states — no steer, coef=1.5,
            set_steer(8, 0.0), set_steer(None, 0.0), and set_steer(8, 1e-9).
RESULT:     torch.equal(base, zero)=True, (base, reset)=True, (base, tinycoef)=True;
            (base, steered)=False with max|d|=0.615. End-to-end: token ids and
            topk identical for coef=0 vs no layer/coef.
VERDICT:    CONFIRMED
FALSIFIER:  any torch.equal(...) == False for a coef=0 path.
```

```
HYPOTHESIS: diff-in-means is built at the same site set_steer writes to (no
            one-block mismatch).
TEST:       cos(server.vectors[8], adapter.diff_in_means(pairs, layer=8,
            site="pre")) and a capture with _collect_pre=True showing which
            resid_post layers change when steering L8.
RESULT:     cos = 1.000000 (site="pre"). resid_post changed for [8..27], NOT
            for [0..7]; resid_pre(9..27) changed, resid_pre(8) unchanged only
            because the capture runs before the injection.
VERDICT:    CONFIRMED — write site is resid_pre(8), the input to block 8.
FALSIFIER:  cos < 0.999 vs site="pre", or resid_post(7) changing.
```

```
HYPOTHESIS: a token frame stays under 40 KB for a 28-layer model.
TEST:       Stream 12 tokens from a 28-layer Qwen2 with the production
            tokenizer; json.dumps each {"type","session",…} and measure bytes.
RESULT:     min 4282, median 4312, max 4344 bytes = 4.18-4.24 KB.
            payload_bytes-style (compact) = 3765 B. Synthetic worst case with
            every lens string 32 chars: 6902 B. Budget: 40960 B.
VERDICT:    CONFIRMED (10x headroom). Hidden size does not affect frame size;
            layer count, entry counts and token strings do — all production-shaped.
FALSIFIER:  any frame > 40960 B.
```

```
HYPOTHESIS: the steering vector is a "comment vs code" direction.
TEST:       Tokenise all 16 contrast prompts with the production tokenizer;
            inspect the last token ids; compare vector[0] to
            emb(mean pos-last) - emb(mean neg-last) on the tiny model.
RESULT:     POS last ids = [198]*8; NEG last ids = [982,510,510,510,510,340,982,510].
            cos(vector[0], identity_direction) = 1.00000.
VERDICT:    REFUTED for L0 (and structurally for every layer): the vector is the
            difference of the prompts' FINAL TOKENS, not of their style.
FALSIFIER:  cos substantially below 1 at L0, or POS/NEG last tokens not separable.
```

```
HYPOTHESIS: the server isolates clients for steering and cancel.
TEST:       Two sockets; A streams at coef=0, B sends steer L8 2.0 then cancel.
RESULT:     A's later tokens carry no steering flag; B's cancel ended A's stream
            with reason="cancelled".
VERDICT:    REFUTED (H3-1, H3-2).
FALSIFIER:  A's stream unaffected by B.
```

---

## 3. What passed

**A — protocol conformance (largely clean now).**
`GET /api/info` returns exactly the nine documented fields; `GET /api/health`
returns exactly `{ok,busy,uptime_s,generations}`; `GET /api/export/{id}` returns
200 with `{session, prompt, …, tokens[]}` and 404 for an unknown id. Every `token`
frame has `type, session, step, token, token_id, ms, topk, layers`; `layers` is
one entry per layer ordered `0…27` with `l, norm, lens, neurons`; `lens` and
`neurons` are top-3; `done` carries `session, reason, n_tokens` with
`reason ∈ {max_new, eos, cancelled}`; `steer_ack` echoes `layer`/`coef` exactly.
Malformed input now returns an `error` frame instead of killing the socket — I
verified seven bad payloads (bad layer, `max_new=0`, bad coef type, coef without
layer, `coef=1e9`, unknown type) and the socket survived all of them.
`cancel` is implemented and honoured.

**C — steering correctness.**
`coef=0` is bit-identical (`torch.equal`, four paths). Steering is not a no-op
(logits differ by 0.615 at coef 1.5), so the vectors are genuinely non-zero. The
build site and the write site agree at **cos 1.000000** against
`adapter.diff_in_means(..., site="pre")`, and the write lands in
`resid_pre(layer)`, proven by which `resid_post` layers move. The `coef` unit is
right: `|vector[8]| = 0.5033` vs the layer's mean residual norm `0.5046` (the code
uses the POS mean, 0.5033 — a 0.26 % nit, not a defect).

**D — payload size.** 4.2 KB per token measured; `fit_payload` trims
neurons→lens→strings in a defined order, never drops a layer, and asserts the
budget. `payload_bytes` uses the same compact separators Starlette uses, and
`ensure_ascii=True` makes it conservative.

**E — concurrency (apart from H3-1/H3-2).** A second chat gets one `error` and
no spurious `done`; the first stream is unaffected. `cancel` stops within one
token. **Disconnecting mid-stream now frees the lock in 0.2 s** — the orphaned
generation that blocked the old server for its full `max_new` is gone.
`/api/health.busy` tracked correctly across all of it.

**F — memory safety.** `guard.require` and `guard.unsafe_downcast` are both
called before `from_pretrained`; exactly one model is loaded; `low_cpu_mem_usage`
is set; dtype is always explicit from the CLI; `max_new ≤ 256`, prompt ≤ 4000
chars, sessions bounded at 20 × 256 frames. Verified `guard.source_dtype` = BF16
and `unsafe_downcast(float16)=True` on `.hf/bf16-hf`, so the NaN path is now
refused at startup rather than silently producing NaN.

**G — tests.** 12 tests, all pass (`PYTHONPATH=code pytest tests/test_server.py
-q` → 12 dots, exit 0, re-run at 14:06 against the final hash). None is
tautological: they drive a real subprocess server; the
`coef=0` test includes a **vacuity guard** (`assert not torch.equal(base, steered)`)
so a zero-vector regression cannot pass; the busy test uses two real sockets and
asserts `health.busy` both ways; the SIGINT test asserts `rc == 0`. The gap is
coverage and gating (H3-4), not dishonesty.

**H — frontend.** The four documented states exist; `canSend` requires
socket-open + `info.ready` + not streaming; the textarea is `disabled` while
streaming and the Send button becomes Cancel; errors surface in a
`role="alert"` banner; socket close triggers exponential-backoff reconnect
(500 ms → 8 s) with a manual "retry now"; the composer placeholder explains each
blocked state; `/api/info` 503 is handled as "model loading" and sending is
blocked. `SteeringControls` says **"0 = off"** and calls the trace
**"correlational"** at coef 0. `npm run build` passes (Next 15.5.29, 8.18 kB
route) and `tsc --noEmit` exits 0.

---

## 4. Not tested, and why

- **The real 1.5B model was never loaded** (explicit rule). Every behavioural
  measurement is on random weights, so "does coef=1.5 make the model write
  comments" is *not* answerable here — that is the open C-1/C4 question in
  STATE_OF_PROJECT.md, not an audit finding.
- **Payload size** was measured on a 28-layer stand-in with hidden size 64. Frame
  size depends on layer count, lid/neuron entry counts and decoded token
  strings — all production-shaped — not on `d_model`. The number is
  representative; it was not measured on the real checkpoint.
- **No browser E2E.** The frontend was reviewed statically, typechecked and
  production-built; click-through behaviour (canvas interaction, copy button,
  export download) was not exercised in a browser.
- **`/api/info` 503 is unreachable in normal operation** because `main()` loads
  the model before `uvicorn.run()`. The "model loading" UI branch can only be
  hit if the REST proxy is up before the backend, which it is not. Not a defect;
  just note that the documented 503 path is effectively dead code.
- **`done` is never sent after an `error` frame.** `_worker` returns from the
  except branch. The UI handles a bare `error`, and PROTOCOL does not require a
  `done` in that case, so this is consistent — but a client that counts
  `done` frames will under-count.

---

## 5. Protocol notes (PROTOCOL.md says changes are recorded here)

1. **Add per-token steering provenance** (required by H3-1/H3-8/H3-9):
   `token.steered` (bool), `token.coef` (float), `token.steer_layer` (int|null).
   Without these the "correlational except when steering is active" invariant is
   unenforceable per token.
2. **State that steering and cancel are per-connection**, not global. Today the
   protocol is silent and the server is global; H3-1/H3-2 are the consequence.
3. **State the `max_new` bound** (1–256) in the protocol; the UI currently
   advertises 512 (H3-10).
4. **Clarify `norm`/`lens` are taken at `resid_post(l)`**, while the steering
   vector for layer `l` lives at `resid_pre(l)`. Both are defensible; the
   current text ("that layer's residual") lets a reader assume they are the
   same space.

---

## 6. Verdict

**Ready for a human to use as a *microscope*, not yet as an *honest steering
instrument*.**

The backend contract is now essentially conformant and the four defects that
would have made a demo embarrassing — no session ids, no `/api/health` or
`/api/export`, `cancel` ignored, bad input killing the socket, and bf16→fp16 NaN
— are fixed and verified. Concurrency, memory safety, payload budget and the
`coef=0` identity all hold under test.

Two things block "show it to someone and let them believe the labels":

1. **H3-1/H3-2** — steering is process-global and unlabelled per token, so the
   panel physically cannot say which tokens were steered, and one client can
   steer or cancel another's stream.
2. **H3-3** — the vector is built from the last token only; at L0 it is exactly
   a token-identity direction (cos 1.00000). The slider is honest about
   *magnitude* and *site*, but not about *what it steers*, and the UI's
   "STEERING ACTIVE · causal" caption inherits that (H3-8/H3-9).

Fix H3-1, H3-2, H3-3 and H3-8 before a human is asked to trust the panel. Fix
H3-4 and H3-5 before anyone trusts the gate that says these things are fine.
H3-6 through H3-18 are polish plus one unfinished panel (H3-18).
