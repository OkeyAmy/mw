# app — live server

`server.py` streams generated tokens **and the internal trace behind each one**
(per-layer logit lens + top MLP activations) over a websocket. The frozen contract
is [`PROTOCOL.md`](PROTOCOL.md); this file documents how to run it and the parts
the contract leaves to the implementation.

## Run

```bash
PYTHONPATH=code .venv/bin/python app/server.py --model .hf/bf16-hf --dtype float32
# http://127.0.0.1:8000   ws://127.0.0.1:8000/ws
```

Requires `fastapi`, `uvicorn` (and `websockets` only for the tests). The Next.js
dev server on port 3000 proxies `/api/*` and `/ws` to 8000, so the browser always
talks to one origin. `GET /` redirects to `http://localhost:3000`.

## Built on the audited adapter

`Engine` is a thin wrapper around `mw.adapter.ModelAdapter` — there is no second
copy of the model code. `Engine.adapter` is a `LiveAdapter(ModelAdapter)` which
adds only the live loop. Everything that the G1 audit fixed stays in the adapter:

| concern | owner |
|---|---|
| memory guard, BF16→fp16 refusal, dtype | `ModelAdapter.load()` |
| hooks (resid_pre / resid_post / MLP), idempotent install | `ModelAdapter` |
| logit lens incl. `logits_scaling` and tie handling | `ModelAdapter.logit_lens*` |
| steering vectors (`build_steer_vectors()`, all positions, same suffix) | `ModelAdapter` |
| activation / neuron intervention | `set_steer`, `set_neuron_steer` |
| token payload + `steered`/`coef`/`steer_layer` provenance | `ModelAdapter.stream` shape |

`LiveAdapter.stream` reimplements only the token loop that the base cannot expose
(it owns `past_key_values`), because mid-generation injection needs to extend the
cache and the lens schedule changes per step. It calls the adapter's hooks, lens,
scaling and steering, and adds `neuron_steer` provenance to each token.

## REST

| endpoint | response |
|---|---|
| `GET /api/info` | `{ready, model, arch, layers, d_model, vocab, scaling, dtype, steer_vectors}`; `503 {"ready": false}` while loading |
| `GET /api/health` | `{ok, busy, busy_s, uptime_s, generations, generations_started, generations_done, generations_timed_out, timeout_s}` — always 200 |
| `GET /api/export/{session_id}` | full stored trace; `404` for an unknown/evicted session |
| `GET /api/detail/{session_id}/{step}` | top-32 neurons per layer for one token; `404` if unknown |

`/api/export/{id}` returns the session metadata at the top level plus the token
frames:

```json
{
  "session": "s_7f3a", "prompt": "...", "max_new": 40, "layer": 8, "coef": 1.5,
  "started_at": 1762..., "finished_at": 1762..., "reason": "max_new",
  "n_tokens": 40, "tokens": [ /* exactly the `token` frames that were sent */ ]
}
```

`/api/detail/{id}/{step}`:

```json
{"session": "s_7f3a", "step": 3, "token": "def",
 "layers": [{"l": 0, "norm": 14.2, "lens": [["def", 0.2]],
             "neurons": [[6376, 41.11], "... 32 entries"]}]}
```

Sessions are held in a bounded in-memory store (last **20**) and never written to
disk. Eviction drops the oldest session first. Details are stored for every token
the session keeps (256 cap), not only for the lensed layers.

## WebSocket

Client messages and server messages are exactly `PROTOCOL.md`. Bounds enforced
with pydantic before anything is queued:

| field | rule |
|---|---|
| `messages` | non-empty transcript of `{role, content}`, roles `user|assistant|system`; whole transcript ≤ 16 000 chars, each message ≤ 4000 |
| `prompt` | shorthand for one user turn; non-empty after `.strip()`, ≤ 4000 chars |
| `max_new` | integer 1..256 (default 40) |
| `layer` | `>= 0` and `< layers`; required when `coef != 0` |
| `coef` | finite, `abs(coef) <= 20`; `coef = 0` disables steering bit-identically |
| `inject.text` | non-empty, at most 2000 chars |
| `neuron_steer.neuron` | `0 <= neuron < intermediate_size` |
| `neuron_steer.value` | finite, `abs(value) <= 1e4` |

Anything invalid produces `{"type": "error", "message": "..."}` and does not
start a generation. A second `chat` while one is streaming returns
`{"type": "error", "message": "server busy"}` (the client should prevent it too).

### Full conversation prefill

`chat` takes the **whole transcript** in `messages`, rendered through the model's
chat template (`apply_chat_template(messages, add_generation_prompt=True)`) as a
single prefill. Sending only the newest `prompt` still works but is a cold
single-turn request — that is the "it reloads every message" failure. The export
metadata stores both the last user `prompt` and the full `messages` array.

### Per-connection steering

`steer` and `neuron_steer` mutate **only the calling socket's** `SteerState`. The
server applies those settings immediately before each token's forward pass, so
client A's stream is never changed by client B's slider. Each token frame carries
the provenance that was actually in force (`steered`, `coef`, `steer_layer`,
`neuron_steer`).

### Mid-generation context injection

- `{"type":"inject","text":...}` is queued. After the current token, the text is
  rendered as a **new user turn** through the chat template (plain appended text
  measurably does nothing on an instruct model), one forward pass runs against
  the existing `past_key_values` (cache extended, not rebuilt), and generation
  continues from the logits of that forward. The last injected token is never
  re-fed, so no cached token is duplicated.
  Emits `{"type":"injected","at_step":N,"tokens":T,"text":...}` where `N` is the
  step index of the next token. If generation already finished, the inject
  starts a new turn appended to the same conversation.
- `{"type":"edit_prompt","prompt":...}` discards the cache at the next token
  boundary, restarts from a fresh one-user-turn transcript, and emits
  `{"type":"restarted","prompt":...}`. Step numbers keep counting within the
  session so exported traces have no duplicate steps.

### Neuron intervention

`neuron_steer` clamps one MLP intermediate unit at the last position (0 = ablation,
large = amplify); `neuron_clear` removes one layer's clamps or all of them
(`"layer": null`). Both are acknowledged with `steer_ack`.

### Payload budget

Every `token` frame is measured after serialization with the same encoder
Starlette uses and kept under **40 KiB**. NaN/Infinity are sanitised first (they
are not valid JSON), then `neurons`/`lens` are trimmed to top-1 / empty and
decoded strings clipped. It asserts the result fits and raises rather than
sending an oversized frame. Layer entries are never dropped — the protocol
requires one per layer.

Lens cost: the first token lenses **all** layers; later tokens lens every second
layer, and skipped layers carry `"lens": []`. Detail drill-down still has all 32
neurons per layer.

### Cancel

Only the connection that started a generation may cancel it; other sockets get
`{"type": "error", "message": "not your generation"}`. An owned cancel stops the
stream **after the token currently being produced** and ends with
`{"type": "done", "reason": "cancelled"}`. Cancelling with nothing running returns
`nothing to cancel`. A disconnected client cancels its own generation.

### Watchdog

A generation that runs longer than `--gen-timeout` seconds (default 180, also
`MOD_WEI_GEN_TIMEOUT`) is cancelled at the next token boundary and ends with
`done.reason="cancelled"` / `timed_out=true`. If the thread is stuck inside a
single forward past the hard grace period, the watchdog force-releases the
generation lock, so `busy: true` can never persist forever. The worker releases
the lock and updates `generations_started`/`generations_done` in a top-level
`finally` even when the adapter raises.

### Shutdown

`SIGINT`/`SIGTERM` cancel any in-flight generation, drain the server, wait up to
10 s for the generation lock, and exit with status 0.

## Tests

Two layers:

- `tests/test_server_fast.py` — no model, no subprocess; FastAPI `TestClient`
  checks `/api/info` 503, `/api/health`, 404s, the redirect, every pydantic bound,
  the 40 KiB budget and NaN/Infinity sanitisation. **Runs in the default gate.**
- `tests/test_server.py` — marked `slow`: builds a tiny random Qwen2 (2 layers,
  hidden 32, MLP gate/up zeroed so interventions are causal), starts the real
  server in a subprocess on a free port and drives it with a websocket client.
  Covers token/provenance shape, lens schedule, injection, edit_prompt, cancel
  ownership, per-connection steering isolation, neuron steering, `/api/detail`,
  `coef=0` bit-identity and graceful SIGINT shutdown.

```bash
PYTHONPATH=code .venv/bin/python -m pytest tests -m "not slow" -q   # gate
PYTHONPATH=code .venv/bin/python -m pytest tests/test_server.py -q  # slow
```

The 1.5B model is never loaded by either layer.
