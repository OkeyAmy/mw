# PROTOCOL — the contract between backend and UI

Frozen interface. Backend (`app/server.py`) and frontend (`web/`) implement this
independently. Neither edits the other's directory. Changes require a note in
`agents/H3_ui_audit.md`.

## Transport

- REST: `http://127.0.0.1:8000`
- WebSocket: `ws://127.0.0.1:8000/ws`
- The Next.js dev server (port 3000) proxies `/api/*` and `/ws` to 8000.
  Browser therefore always talks to the same origin.

## REST

### `GET /api/info` → 200
```json
{
  "ready": true,
  "model": ".hf/bf16-hf",
  "arch": "Qwen2ForCausalLM",
  "layers": 28,
  "d_model": 1536,
  "vocab": 151936,
  "scaling": 1.0,
  "dtype": "torch.float32",
  "steer_vectors": 28
}
```
503 `{"ready": false}` while the model is loading.

### `GET /api/health` → 200
```json
{"ok": true, "busy": false, "busy_s": 0.0, "uptime_s": 123.4,
 "generations": 7, "generations_started": 7, "generations_done": 7,
 "generations_timed_out": 0, "timeout_s": 180}
```
`busy_s` is how long the current generation has held the lock. A generation
exceeding `timeout_s` is cancelled at the next token; if its thread is stuck, the
lock is force-released after a hard grace period, so `busy: true` can never
persist indefinitely.

### `GET /api/export/{session_id}` → 200
Full trace of a finished session (array of `token` payloads + metadata).

## WebSocket

### client → server

```json
{"type": "chat",
 "messages": [
   {"role": "user", "content": "def add(a,b):"},
   {"role": "assistant", "content": "def add(a, b):\n    return a + b"}
 ],
 "max_new": 40, "layer": 8, "coef": 0.0}
```

**`messages` is the conversation and it is REQUIRED for a chat UI.** The client
must send the full transcript (all prior user and assistant turns). The server
renders it through the model's chat template when one exists, joins it with
role markers otherwise, and prefills from that.

Sending only `prompt` remains supported for single-shot use, but it makes the
model forget every previous turn. That is why the app felt like it "reloaded" on
every message: it was a cold single-turn prefill each time, not a conversation.

Roles are `user`, `assistant` and `system`; each message is capped at 4000
characters and the whole transcript at 16 000. An `inject` continues the same
conversation as a new user turn.
```json
{"type": "steer", "layer": 8, "coef": 1.5}
```
```json
{"type": "cancel"}
```
`coef` is in units of the layer's mean residual norm. `coef = 0` MUST be
bit-identical to no steering. Bounds: `max_new` 1–256, prompt ≤ 4000 chars,
`|coef| ≤ 20`.

**Steering and cancel are per connection.** A `steer` or `neuron_steer` from one
socket never changes another socket's stream; the server applies the owning
connection's settings immediately before each token's forward pass. Only the
connection that started a generation may cancel it — others receive
`{"type":"error","message":"not your generation"}`.

### server → client

`token` — exactly one per generated token:
```json
{
  "type": "token",
  "session": "s_7f3a",
  "step": 0,
  "token": "def",
  "token_id": 589,
  "ms": 1204,
  "steered": true,
  "coef": 1.5,
  "steer_layer": 8,
  "neuron_steer": {"14": {"6376": 0.0}},
  "topk": [["def", 0.42], ["return", 0.11]],
  "steered": false,
  "coef": 0.0,
  "steer_layer": null,
  "pca": {"var_explained": 0.63, "n_layers": 28},
  "layers": [
    {"l": 0, "norm": 14.2, "xy": [-0.31, 0.08],
     "lens": [["def", 0.2]], "neurons": [[1234, 4.5]]}
  ]
}
```
- `layers` is ordered by layer index, one entry per layer.
- `xy` is the REAL coordinate for the 3D view: a shared 2-D PCA of the
  mean-centred, standardised residual trajectory across layers. Not a proxy.
  `pca.var_explained` is the fraction of variance those two components carry.
- `lens` is top-3 from that layer's residual through the final norm + unembedding.
  The first token lenses **all** layers; later tokens lens every 2nd layer to
  keep the unembedding read off the critical path — skipped layers carry
  `"lens": []` and are rendered as "not sampled".
- `neurons` is top-3 MLP intermediate indices and their activation values.
  The drill-down endpoint serves top-32.
- payload MUST stay under ~40 KB per token.
- **Provenance is per token, not per message.** `steered` is true when any
  intervention was active for this forward; `coef`/`steer_layer` are the
  activation-steering settings at that moment; `neuron_steer` is `null` or a
  `{"layer": {"neuron": value}}` map of clamped MLP units. The UI must label the
  trace **correlational** iff `steered` is false.

`done`:
```json
{"type": "done", "session": "s_7f3a", "reason": "max_new|eos|cancelled",
 "n_tokens": 12, "injected_at": 3, "timed_out": false}
```
- `injected_at` is the step index at which the **last** mid-generation
  injection landed (`null` when none landed).
- `timed_out` is true when the watchdog aborted the generation because it ran
  past the configured limit. `reason` stays `cancelled` so the UI state machine
  is unchanged; `timed_out` is the distinguishing flag.

`error`:
```json
{"type": "error", "message": "server busy"}
```

`steer_ack`:
```json
{"type": "steer_ack", "layer": 8, "coef": 1.5}
```

### mid-generation context injection (the primary control)

This is different from activation steering. It adds real tokens to the sequence,
so the model conditions on new information from that point on.

```json
{"type": "inject", "text": "Use a while loop instead of recursion."}
{"type": "edit_prompt", "prompt": "def fib(n):  # must be iterative\n"}
```

Server behaviour:
- `inject`: queued. After the current token completes, tokenize `text`, run ONE
  forward pass with the existing `past_key_values` to extend the cache, then
  continue generation. Reply:
  `{"type":"injected","at_step":7,"tokens":11,"text":"..."}`.
  If generation already finished, the inject becomes the start of a new turn.
- `edit_prompt`: discard the cache and restart generation from the edited prompt.
  Reply `{"type":"restarted","prompt":"..."}`.
- Injection MUST NOT corrupt the cache: the injected tokens are appended after
  the most recent generated token, each exactly once (the server keeps the
  post-injection logits instead of re-feeding the last injected token). The UI
  must show them as a system note inside the transcript.
- On an instruct model, plain text appended mid-sequence has **no measurable
  effect**; the insertion is therefore rendered as a new user turn through the
  model's chat template (`format_injection`). `--no-chat-template` falls back to
  raw text for base models. `tokens` in the `injected` frame counts the
  formatted tokens, including the newline/role markers.

UI requirements:
- While streaming, an "inject" input is available next to the cancel button.
- Injected text appears inline as a labelled system chip, at the step it landed.
- The trace for the next token must visibly change (that is the proof it worked).

### neuron-level intervention

```json
{"type": "neuron_steer", "layer": 14, "neuron": 6376, "value": 0.0}
{"type": "neuron_clear", "layer": 14}
{"type": "neuron_clear", "layer": null}
```
Writes into the MLP intermediate at the last position before the down
projection. `value = 0` is an ablation; a large value is amplification. Server
acknowledges with `{"type":"steer_ack","layer":14,"neuron":6376,"value":0.0}`.

### layer drill-down

`GET /api/detail/{session}/{step}` → 200
```json
{
  "session": "s_7f3a", "step": 3, "token": "def",
  "layers": [
    {"l": 0, "norm": 14.2, "lens": [["def", 0.2]],
     "neurons": [[6376, 41.11], [5805, 35.79]]}
  ]
}
```
`neurons` is the top-32 captured by the adapter (the stream payload sends only
top-3 to keep frames small).

## Required UI states

| state | trigger | UI behaviour |
|---|---|---|
| `connecting` | socket opening | status dot amber, composer disabled |
| `ready` | `/api/info` ok + socket open | composer enabled |
| `streaming` | first `token` received | show cursor, disable send, live stats |
| `done` | `done` | remove cursor, enable send, keep trace |
| `error` | `error` or socket close | show message, re-enable send, auto-reconnect |
| `busy` | second `chat` while streaming | client must prevent, server returns `error` |

## Invariants the UI must display honestly

- `coef = 0` must not change output. The UI should label the slider "0 = off".
- Per-token trace is **correlational** except when steering is active; the panel
  must say so.
- If `/api/info.ready` is false, the UI shows "model loading" and does not send.
