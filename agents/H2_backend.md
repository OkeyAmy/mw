# H2 — backend server (app/server.py)

Date: 2026-11-11. Scope: `app/server.py`, `app/README.md`, `app/PROTOCOL.md`
(token example + per-connection note), `tests/test_server.py`,
`tests/test_server_fast.py`, `Makefile` (lint covers `app/`).

## Revision 2 — adapter refactor, injection, H3 fixes

### 1. Engine now wraps `mw.adapter` (no duplicate model code)

`Engine` holds `LiveAdapter(ModelAdapter)`; it owns **no** loading, hooks, lens,
scaling, tie handling or vector building:

- `LiveAdapter(path).load(dtype=...)` → `guard.require` + BF16→fp16 refusal.
- `adapter.build_steer_vectors()` with **no arguments** (default contrast pairs,
  identical suffix, mean over all positions — H3-3); `Engine.vectors[layer]`.
- `adapter.set_steer(layer, vec, coef, site="pre")`,
  `set_neuron_steer`, `clear_neuron_steer`, `top_neurons(l, 32)`.
- Token payload keeps the adapter's provenance fields (`steered`, `coef`,
  `steer_layer`) and adds `neuron_steer` (`{layer: {neuron: value}}` or `null`).
- Lens is converted to protocol pairs `[token, p]` from the adapter's
  `(id, token, p)` triples.

`LiveAdapter.stream` is the only new loop: the base `stream()` cannot expose
`past_key_values`, so it cannot inject mid-generation or vary the lens subset per
step. It calls the adapter's hooks/lens/steering and reproduces the base payload.

### 2. Mid-generation context injection

- `inject`: queued per generation; after the current token the text is tokenized
  as `"\n" + text`, one forward pass runs against the existing `past_key_values`
  (cache extended, not rebuilt), then generation continues.
  Emits `{"type":"injected","session","at_step","tokens","text"}`.
  If the stream already finished, the inject starts a new turn (re-prefill).
- `edit_prompt`: sets a restart request; at the next token boundary the cache is
  dropped and the stream restarts from the new prompt, emitting
  `{"type":"restarted","session","prompt"}`. Step numbers stay monotonic in the
  session, so export/detail never collide.
- Lens schedule: first token of each run lenses **all** layers; later tokens
  every 2nd (`lens: []` for skipped layers). `/api/detail` still stores top-32
  neurons for every layer of every token.

### 3. H3 addendum fixes

1. **Per-connection steering.** `SteerState` (layer/coef/neuron clamps) lives on
   the websocket connection. `apply_steering` runs immediately before every
   forward, inside the single generation lock, so `steer`/`neuron_steer` from
   client B cannot touch A. Tested with two sockets: B steers hard mid-stream and
   A's token ids equal the unsteered baseline.
2. **Provenance per token.** `steered`, `coef`, `steer_layer`, `neuron_steer` on
   every token frame; PROTOCOL example updated. `coef` is snapshot at yield time.
3. **Correct estimator.** No local diff-in-means. `build_steer_vectors()` default
   pairs. (`test_engine_wraps_mw_adapter` asserts the vectors came from the
   adapter and are non-zero per layer.)
4. **Cancel ownership.** Only the owning socket may cancel; others receive
   `{"type":"error","message":"not your generation"}`. Disconnect/shutdown cancel
   their own/current generation.
5. **Gates.** New `tests/test_server_fast.py` (8 tests, no model, TestClient):
   `/api/info` 503, `/api/health`, 404s, redirect, every pydantic bound, payload
   budget, NaN/Infinity sanitisation. Subprocess tests stay `slow`. `Makefile`
   lint now covers `app/`; `ruff check app/` passes. Extra: non-finite numbers are
   sanitised in `fit_payload` (H3-15), `topk` divides by `logits_scaling`
   (H3-14), no import-time `set_num_threads` (H3-12), `/` redirects to :3000
   (H3-7), eos tokens are not streamed (the loop returns before yielding the
   predicted eos on the next step; `done.reason="eos"` still reported).

### 4. Protocol conformance

| requirement | status |
|---|---|
| info / health / export / **detail** | conform |
| token keys incl. `steered`, `coef`, `steer_layer`, `neuron_steer` | conform |
| `layers` one per layer, ordered; lens top-3 pairs | conform (odd layers `[]` after step 0) |
| 40 KiB budget | enforced + asserted |
| `done` `max_new` / `eos` / `cancelled` | conform |
| `inject` → `injected`; `edit_prompt` → `restarted` | conform |
| `neuron_steer` / `neuron_clear` → `steer_ack` | conform |
| steering + cancel per connection | conform |
| `coef=0` bit-identical | tested (`torch.equal` + end-to-end) |

Deviations: (a) a generation failure sends `error` and no `done` (PROTOCOL has no
error reason); (b) `at_step` is the index of the next token, so it can be 4 when
the client injected after token 3 if the 4th token was already on the wire;
(c) detail lenses for odd layers at steps ≥1 are empty by design (lens schedule).

## Test output (real)

Fast gate (no model):

```
$ PYTHONPATH=code .venv/bin/python -m pytest tests -m "not slow" -q
40 passed, 25 deselected, 1 warning in 19.62s

$ .venv/bin/python -m ruff check app/
All checks passed!
```

Slow suite (tiny random Qwen2, real subprocess server, SIGINT teardown):

```
$ PYTHONPATH=code .venv/bin/python -m pytest tests/test_server.py -v -o addopts=""
tests/test_server.py::test_api_info_schema PASSED
tests/test_server.py::test_api_health PASSED
tests/test_server.py::test_root_redirects_to_the_ui PASSED
tests/test_server.py::test_cors_allows_next_dev_origin PASSED
tests/test_server.py::test_export_roundtrip_and_unknown_session PASSED
tests/test_server.py::test_detail_endpoint_returns_top32 PASSED
tests/test_server.py::test_token_payload_shape_and_budget PASSED
tests/test_server.py::test_injection_changes_next_token PASSED
tests/test_server.py::test_edit_prompt_restarts_generation PASSED
tests/test_server.py::test_cancel_stops_after_current_token PASSED
tests/test_server.py::test_cancel_is_owner_only PASSED
tests/test_server.py::test_cancel_when_idle_returns_error PASSED
tests/test_server.py::test_second_chat_while_streaming_is_busy PASSED
tests/test_server.py::test_steering_is_per_connection PASSED
tests/test_server.py::test_neuron_steer_ack_provenance_and_causality PASSED
tests/test_server.py::test_invalid_inputs_return_error PASSED
tests/test_server.py::test_coef_zero_matches_no_steering_end_to_end PASSED
tests/test_server.py::test_engine_wraps_mw_adapter PASSED
tests/test_server.py::test_coef_zero_logits_bit_identical PASSED
tests/test_server.py::test_fit_payload_truncates_oversized_trace PASSED
tests/test_server.py::test_nonfinite_numbers_are_sanitised PASSED
21 passed in 43.55s
```

Injection proof: baseline run of 40 tokens is `[7, 7, 7, …]` (the tiny model's
MLP gate/up are zeroed, so greedy decoding follows the last token). After the
client sends `{"type":"inject","text":"Use a while loop"}` the server emits
`injected{at_step: N}` and the token at step `N` changes from `7` to the injected
suffix's token (`0`), while tokens stay gun-barrel contiguous.

`make lint` (`ruff check code app tests`) still reports pre-existing findings in
`code/` (not owned here); `app/` is clean.

## Revision 3 — live defects (Granite run) + full-conversation prefill

### Watchdog and lock safety

- `GEN_LOCK` is now a `GenLock` (lock + `started_at` + idempotent `release`).
- `_watchdog_tick` (called by a daemon thread every `max(0.01, min(1, timeout/4))`):
  past `GEN_TIMEOUT_S` (CLI `--gen-timeout`, env `MOD_WEI_GEN_TIMEOUT`, default
  180 s) it flags `gen.timed_out` and cancels at the next token; past
  `GEN_TIMEOUT_S + GEN_HARD_GRACE_S` (10 s) it force-releases a lock whose thread
  is stuck in one forward. The worker's `finally` then no-ops that release.
- `/api/health` exposes `busy_s`, `generations`, `generations_started`,
  `generations_done`, `generations_timed_out`, `timeout_s`. Counters increment on
  acquire and in the top-level `finally`, including the adapter-raises path.
- `done` now carries `timed_out`; reason stays `cancelled` so the UI states are
  unchanged.

### Injection formatting (measured defect)

`LiveAdapter.format_injection()` renders the insertion as a new user turn through
`apply_chat_template(..., add_generation_prompt=True)` (`--no-chat-template` uses
`User: ...\nAssistant:`). Raw appended text is not used. `format_messages()`
renders the full transcript; `format_prompt()` remains a one-user-turn alias.

### Duplicate-token fix

After an injection forward the loop keeps those logits in `pending` and never
re-feeds the last injected token (`ids = ext[:, -1:]` removed). Regression test
`test_injection_feeds_every_token_exactly_once` spies on every forward and asserts
the fed stream equals `prompt + generated[:-1]` with the injected turn spliced in
exactly once.

### `chat.messages` (full conversation)

`ChatIn` accepts `messages: [{role, content}]` (roles user/assistant/system,
≤4000 chars each, ≤16000 total) OR the `prompt` shorthand. The server renders the
whole transcript as one prefill, so a multi-turn chat is continuous. Export
metadata stores `messages` plus the last user `prompt`. `inject` appends to the
same transcript; `edit_prompt` restarts from a fresh user turn.

### Test output (real)

```
$ PYTHONPATH=code .venv/bin/python -m pytest tests -m "not slow" -q
44 passed, 30 deselected, 1 warning in 20.50s

$ .venv/bin/python -m ruff check app/ tests
All checks passed!
```

```
$ PYTHONPATH=code .venv/bin/python -m pytest tests/test_server.py -v -o addopts=""
tests/test_server.py::test_api_info_schema PASSED
tests/test_server.py::test_api_health PASSED
tests/test_server.py::test_root_redirects_to_the_ui PASSED
tests/test_server.py::test_cors_allows_next_dev_origin PASSED
tests/test_server.py::test_export_roundtrip_and_unknown_session PASSED
tests/test_server.py::test_detail_endpoint_returns_top32 PASSED
tests/test_server.py::test_token_payload_shape_and_budget PASSED
tests/test_server.py::test_injection_changes_continuation PASSED
tests/test_server.py::test_messages_carry_the_whole_conversation PASSED
tests/test_server.py::test_prompt_only_path_is_one_user_turn PASSED
tests/test_server.py::test_chat_template_renders_injection_as_a_user_turn PASSED
tests/test_server.py::test_injection_feeds_every_token_exactly_once PASSED
tests/test_server.py::test_edit_prompt_restarts_generation PASSED
tests/test_server.py::test_cancel_stops_after_current_token PASSED
tests/test_server.py::test_cancel_is_owner_only PASSED
tests/test_server.py::test_cancel_when_idle_returns_error PASSED
tests/test_server.py::test_second_chat_while_streaming_is_busy PASSED
tests/test_server.py::test_steering_is_per_connection PASSED
tests/test_server.py::test_neuron_steer_ack_provenance_and_causality PASSED
tests/test_server.py::test_invalid_inputs_return_error PASSED
tests/test_server.py::test_coef_zero_matches_no_steering_end_to_end PASSED
tests/test_server.py::test_engine_wraps_mw_adapter PASSED
tests/test_server.py::test_coef_zero_logits_bit_identical PASSED
tests/test_server.py::test_fit_payload_truncates_oversized_trace PASSED
tests/test_server.py::test_nonfinite_numbers_are_sanitised PASSED
tests/test_server.py::test_watchdog_aborts_a_runaway_generation PASSED
26 passed in 70.59s
```

The watchdog test runs a second subprocess with `--gen-timeout 0.05`: a 256-token
run ends with `done{reason:"cancelled",timed_out:true}`, `/api/health` reports
`busy:false` and `generations_timed_out>=1`, and a new chat succeeds immediately.
