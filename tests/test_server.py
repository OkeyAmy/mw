"""End-to-end tests for app/server.py against PROTOCOL.md.

Slow on purpose: they build a TINY random Qwen2 (2 layers, hidden 32, ~1 MB) in a
tmp dir and run the real server against it in a subprocess. The real 1.5B model is
never touched — mw.guard.require would (correctly) be the only gate anyway.

The tiny model has `mlp.gate_proj` and `mlp.up_proj` zeroed, so its MLP
intermediate is normally zero and the residual is driven by the token embeddings
plus random attention. That makes causal tests deterministic:
  * greedy next-token predictions follow the last token,
  * injecting real context changes the next token,
  * writing one MLP unit (down_proj is left random) changes the next token.
It also makes `coef=0` bit-identity measurable rather than vacuous.

Run:
  PYTHONPATH=code .venv/bin/python -m pytest tests/test_server.py -q
CI skips these: pytest -m "not slow"
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import torch
import websockets
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from mw.adapter import ModelAdapter

from app.server import (
    MAX_TOKEN_BYTES,
    Engine,
    GenControl,
    LiveAdapter,
    SteerState,
    fit_payload,
    payload_bytes,
)

pytestmark = pytest.mark.slow

VOCAB = ["[UNK]", "def", "add", "return", "a", "+", "b", ":", "\n", "x", "=", "1"]
PROMPT = "def add a b :"
BUDGET = MAX_TOKEN_BYTES            # 40 KiB per token frame


# -- tiny model --------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tiny-server")
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[UNK]",
        additional_special_tokens=["[EOS]"])
    # eos lives outside the model vocab (id 12, logits are 0..11), so greedy
    # decoding never stops early: the cancel test gets a long stream.
    tok.eos_token = "[EOS]"
    # A real chat template, so prompt/injection formatting is exercised on the
    # same code path as an instruct model (fallback covered by a unit test).
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    cfg = Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, tie_word_embeddings=False)
    # Seed 1 (untied head) makes the tiny model's argmax genuinely
    # context-dependent: a multi-turn prefill and an injection both change the
    # continuation, which is what these tests need to measure.
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            for name in ("gate_proj", "up_proj"):
                lin = getattr(layer.mlp, name)
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live(tiny_model, tmp_path_factory):
    port = _free_port()
    log_path = tmp_path_factory.mktemp("server-log") / "server.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "code") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w+", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "app" / "server.py"),
             "--model", str(tiny_model), "--dtype", "float32",
             "--host", "127.0.0.1", "--port", str(port),
             "--gen-timeout", "120"],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 180
            while True:
                if proc.poll() is not None:
                    log.flush()
                    log.seek(0)
                    raise RuntimeError(f"server exited during startup:\n{log.read()}")
                try:
                    if httpx.get(base + "/api/info", timeout=1.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("server did not become ready in 180 s")
                time.sleep(0.1)
            yield base
        finally:
            rc = None
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)      # graceful-shutdown contract
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
            log.flush()
            log.seek(0)
            text = log.read()
            assert rc == 0, f"server did not stop cleanly on SIGINT (rc={rc})\n{text}"
            assert "steering vectors ready" in text, text


def _ws_url(base: str) -> str:
    return base.replace("http://", "ws://") + "/ws"


def _chat(prompt: str = PROMPT, max_new: int = 3, layer: int | None = 1,
          coef: float = 0.0) -> dict:
    msg = {"type": "chat", "prompt": prompt, "max_new": max_new, "coef": coef}
    if layer is not None:
        msg["layer"] = layer
    return msg


def _collect(base: str, message: dict, stop=("done", "error"), timeout: float = 120.0):
    """One request, frames until a terminal message."""

    async def run():
        async with websockets.connect(_ws_url(base), open_timeout=30) as sock:
            await sock.send(json.dumps(message))
            raws, msgs = [], []
            while True:
                raw = await asyncio.wait_for(sock.recv(), timeout=timeout)
                msg = json.loads(raw)
                raws.append(raw)
                msgs.append(msg)
                if msg["type"] in stop:
                    return raws, msgs

    return asyncio.run(run())


def _send_many(base: str, messages: list, timeout: float = 60.0):
    """Send a sequence of raw payloads on one socket, read one reply each."""

    async def run():
        async with websockets.connect(_ws_url(base), open_timeout=30) as sock:
            out = []
            for m in messages:
                await sock.send(m if isinstance(m, str) else json.dumps(m))
                raw = await asyncio.wait_for(sock.recv(), timeout=timeout)
                out.append(json.loads(raw))
            return out

    return asyncio.run(run())


def _tokens(msgs) -> list[dict]:
    return [m for m in msgs if m["type"] == "token"]


async def _next_token(sock, timeout=120) -> dict:
    """Skip non-token frames (`prefill`) — the first frame is not the first token."""
    while True:
        msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=timeout))
        if msg["type"] == "token":
            return msg


def _token_ids(msgs) -> list[int]:
    return [m["token_id"] for m in _tokens(msgs)]


# -- REST --------------------------------------------------------------------
def test_api_info_schema(live):
    r = httpx.get(live + "/api/info", timeout=10)
    assert r.status_code == 200
    body = r.json()
    for key in ("ready", "model", "arch", "layers", "d_model", "vocab",
                "scaling", "dtype", "steer_vectors"):
        assert key in body, f"/api/info missing {key}"
    assert body["ready"] is True
    assert body["arch"] == "Qwen2ForCausalLM"
    assert body["layers"] == 2
    assert body["d_model"] == 32
    assert body["vocab"] == len(VOCAB)
    assert body["dtype"] == "torch.float32"
    assert body["steer_vectors"] == 2


def test_api_health(live):
    r = httpx.get(live + "/api/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "ok", "busy", "busy_s", "uptime_s", "generations",
        "generations_started", "generations_done", "generations_timed_out",
        "timeout_s",
    }
    assert body["ok"] is True
    assert body["busy"] is False
    assert body["busy_s"] == 0.0
    assert body["uptime_s"] >= 0
    assert body["generations"] >= 0
    assert body["generations_started"] >= body["generations_done"]
    assert body["generations_timed_out"] >= 0
    assert body["timeout_s"] == 120.0


def test_root_redirects_to_the_ui(live):
    r = httpx.get(live + "/", timeout=10, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "http://localhost:3000"


def test_cors_allows_next_dev_origin(live):
    r = httpx.get(live + "/api/info", timeout=10,
                  headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    r2 = httpx.get(live + "/api/info", timeout=10,
                   headers={"Origin": "http://127.0.0.1:3000"})
    assert r2.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"
    evil = httpx.get(live + "/api/info", timeout=10,
                     headers={"Origin": "http://evil.example"})
    assert evil.headers.get("access-control-allow-origin") in (None, "null")


def test_export_roundtrip_and_unknown_session(live):
    _, msgs = _collect(live, _chat(max_new=2))
    tokens = _tokens(msgs)
    done = msgs[-1]
    assert done["type"] == "done"
    sid = done["session"]
    r = httpx.get(live + f"/api/export/{sid}", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["session"] == sid
    assert body["prompt"] == PROMPT
    assert body["reason"] == done["reason"]
    assert body["n_tokens"] == done["n_tokens"] == len(tokens)
    assert len(body["tokens"]) == len(tokens)
    assert [t["token_id"] for t in body["tokens"]] == [t["token_id"] for t in tokens]
    assert httpx.get(live + "/api/export/s_doesnotexist", timeout=10).status_code == 404


def test_detail_endpoint_returns_top32(live):
    _, msgs = _collect(live, _chat(max_new=2))
    sid = msgs[-1]["session"]

    r = httpx.get(live + f"/api/detail/{sid}/0", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["session"] == sid
    assert body["step"] == 0
    assert body["token"] == _tokens(msgs)[0]["token"]
    assert len(body["layers"]) == 2
    for layer in body["layers"]:
        assert set(layer) == {"l", "norm", "lens", "neurons"}
        assert 1 <= len(layer["neurons"]) <= 32
        assert len(layer["lens"]) == 3, "step 0 lenses every layer"
    # step 1: the lens schedule samples only every 2nd layer...
    r1 = httpx.get(live + f"/api/detail/{sid}/1", timeout=10)
    step1 = r1.json()
    by_layer = {layer["l"]: layer for layer in step1["layers"]}
    assert by_layer[0]["lens"], "even layers are still lensed"
    assert by_layer[1]["lens"] == [], "odd layers are marked as not sampled"
    assert len(by_layer[1]["neurons"]) == 32
    assert httpx.get(live + f"/api/detail/{sid}/99", timeout=10).status_code == 404


# -- websocket ---------------------------------------------------------------
def test_token_payload_shape_and_budget(live):
    raws, msgs = _collect(live, _chat(max_new=3, layer=1, coef=0.0))
    tokens = _tokens(msgs)
    done = [m for m in msgs if m["type"] == "done"]
    assert tokens, "no token frames before done"
    assert done, "stream ended without done"
    done = done[0]
    sid = done["session"]
    assert sid.startswith("s_")

    token_raws = [r for r, m in zip(raws, msgs) if m["type"] == "token"]
    for raw, msg in zip(token_raws, tokens):
        for key in ("type", "session", "step", "token", "token_id", "ms",
                    "topk", "layers", "steered", "coef", "steer_layer",
                    "neuron_steer"):
            assert key in msg, f"token frame missing {key}"
        assert msg["session"] == sid
        assert isinstance(msg["token"], str)
        assert isinstance(msg["token_id"], int)
        assert msg["steered"] is False and msg["coef"] == 0.0
        assert msg["steer_layer"] is None and msg["neuron_steer"] is None
        assert len(msg["layers"]) == 2, "one layer entry per layer"
        assert [layer["l"] for layer in msg["layers"]] == [0, 1]
        for entry in msg["layers"]:
            assert {"l", "norm", "lens", "neurons"} <= set(entry)
            assert isinstance(entry["norm"], (int, float))
            assert len(entry["lens"]) <= 3
            assert len(entry["neurons"]) <= 3
            for tok, prob in entry["lens"]:
                assert isinstance(tok, str) and 0.0 <= prob <= 1.0
        assert len(raw.encode("utf-8")) < BUDGET, (
            f"token frame is {len(raw.encode())} bytes, budget is {BUDGET}")

    # first token lenses every layer; later tokens every 2nd layer
    first = tokens[0]
    assert all(layer["lens"] for layer in first["layers"])
    if len(tokens) > 1:
        later = {layer["l"]: layer["lens"] for layer in tokens[1]["layers"]}
        assert later[0] and later[1] == []
    assert done["reason"] in {"max_new", "eos"}
    assert done["n_tokens"] == len(tokens)
    assert [m["step"] for m in tokens] == list(range(len(tokens)))


def test_injection_changes_continuation(live):
    _, base_msgs = _collect(live, _chat(max_new=40))
    baseline = _token_ids(base_msgs)
    assert len(baseline) == 40

    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps(_chat(max_new=40)))
            msgs, token_count = [], 0
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=120))
                msgs.append(msg)
                if msg["type"] == "token":
                    token_count += 1
                    if token_count == 3:
                        await sock.send(json.dumps(
                            {"type": "inject", "text": "Use a while loop"}))
                if msg["type"] in ("done", "error"):
                    return msgs

    msgs = asyncio.run(run())
    injected = [m for m in msgs if m["type"] == "injected"]
    assert injected, "server did not acknowledge the injection"
    at_step = injected[0]["at_step"]
    assert 3 <= at_step < 20, f"injection landed at an unusable step: {at_step}"
    assert injected[0]["tokens"] >= 1
    assert injected[0]["text"] == "Use a while loop"

    ids = _token_ids(msgs)
    assert len(ids) == 40
    # The continuation (argmax sequence) must differ from the un-injected run.
    assert ids != baseline, "injection did not change the argmax sequence"
    assert any(ids[i] != baseline[i] for i in range(at_step, len(ids))), (
        "tokens before the injection matched, but nothing after it changed")
    done = msgs[-1]
    assert done["type"] == "done"
    assert done["reason"] in {"max_new", "eos"}
    assert done["injected_at"] == at_step
    assert done["timed_out"] is False


def test_messages_carry_the_whole_conversation(live):
    transcript = [
        {"role": "system", "content": "You write code."},
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": "def add a b : return a + b"},
        {"role": "user", "content": "def add a b :"},
    ]
    _, ctx = _collect(live, {"type": "chat", "messages": transcript,
                             "max_new": 8, "layer": None, "coef": 0.0})
    _, solo = _collect(live, _chat(PROMPT, max_new=8, layer=None, coef=0.0))
    ctx_ids = _token_ids(ctx)
    solo_ids = _token_ids(solo)
    assert len(ctx_ids) == len(solo_ids) == 8
    assert ctx_ids != solo_ids, (
        "a multi-turn transcript produced the same argmax sequence as the "
        "last user turn alone — the conversation was not prefilled")
    sid = ctx[-1]["session"]
    body = httpx.get(live + f"/api/export/{sid}", timeout=10).json()
    assert body["prompt"] == PROMPT                   # last user turn
    assert [m["role"] for m in body["messages"]] == [
        "system", "user", "assistant", "user"]
    assert len(body["messages"]) == 4


def test_prompt_only_path_is_one_user_turn(tiny_model):
    eng = Engine(str(tiny_model), torch.float32)
    a = eng.adapter
    prompt_ids = a.format_prompt(PROMPT).tolist()
    message_ids = a.format_messages([{"role": "user", "content": PROMPT}]).tolist()
    assert prompt_ids == message_ids


def test_chat_template_renders_injection_as_a_user_turn(tiny_model):
    eng = Engine(str(tiny_model), torch.float32)
    a = eng.adapter
    assert a._has_template()

    text = "Use a while loop"
    template_ids = a.tok(
        a.tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=False),
        return_tensors="pt", add_special_tokens=False).input_ids
    assert a.format_injection(text).tolist() == template_ids.tolist()

    # fallback: no template -> explicit User:/Assistant: markers, not raw text
    a.use_chat_template = False
    fallback = a.format_injection(text).tolist()
    assert fallback != template_ids.tolist()
    raw = a.tok(text, return_tensors="pt").input_ids.tolist()
    assert fallback != raw


def _find_subseq(haystack: list[int], needle: list[int]) -> int | None:
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return None


def test_injection_feeds_every_token_exactly_once(tiny_model):
    """Regression: the old loop re-fed the last injected token from the cache."""
    eng = Engine(str(tiny_model), torch.float32)
    a = eng.adapter
    control = GenControl(SteerState())

    calls: list[list[int]] = []
    real_forward = a.model.forward

    def spy(*args, **kwargs):
        ids = kwargs.get("input_ids")
        calls.append(ids[0].tolist())
        return real_forward(*args, **kwargs)

    a.model.forward = spy
    yielded: list[int] = []
    try:
        for tr in a.stream([{"role": "user", "content": PROMPT}], 6,
                           control=control):
            yielded.append(tr["token_id"])
            if len(yielded) == 2:
                control.injections.put("Use a while loop")
    finally:
        del a.model.forward

    ext = a.format_injection("Use a while loop")[0].tolist()
    prompt_ids = a.format_messages([{"role": "user", "content": PROMPT}])[0].tolist()
    flat = [tok for call in calls for tok in call]
    assert flat[:len(prompt_ids)] == prompt_ids

    tail = flat[len(prompt_ids):]
    at = _find_subseq(tail, ext)
    assert at is not None, "the injected turn was never fed to the model"
    assert _find_subseq(tail[at + len(ext):], ext) is None, (
        "the injected turn was fed twice")
    without_ext = tail[:at] + tail[at + len(ext):]
    # Every generated token is fed exactly once, except the last (never needed).
    assert without_ext == yielded[:-1], (
        f"tokens fed to the model do not match the generated sequence: "
        f"{without_ext} != {yielded[:-1]}")


def test_edit_prompt_restarts_generation(live):
    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps(_chat(max_new=8)))
            msgs, token_count = [], 0
            send_after = 2
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=120))
                msgs.append(msg)
                if msg["type"] == "token":
                    token_count += 1
                    if token_count == send_after:
                        await sock.send(json.dumps(
                            {"type": "edit_prompt", "prompt": "return a + b"}))
                if msg["type"] in ("done", "error"):
                    return msgs

    msgs = asyncio.run(run())
    restarted = [m for m in msgs if m["type"] == "restarted"]
    assert restarted, "edit_prompt did not restart the stream"
    assert restarted[0]["prompt"] == "return a + b"
    ids = _token_ids(msgs)
    assert len(ids) >= 3, "tokens did not resume after the restart"
    assert msgs[-1]["type"] == "done"
    assert msgs[-1]["injected_at"] is None
    assert msgs[-1]["timed_out"] is False


def test_cancel_stops_after_current_token(live):
    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps(_chat(max_new=256, layer=0, coef=0.0)))
            msgs = []
            while True:                     # skip the `prefill` frame
                first = json.loads(await asyncio.wait_for(sock.recv(), timeout=120))
                msgs.append(first)
                if first["type"] in ("token", "done", "error"):
                    break
            if first["type"] != "token":
                return msgs
            await sock.send(json.dumps({"type": "cancel"}))
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=120))
                msgs.append(msg)
                if msg["type"] in ("done", "error"):
                    return msgs

    msgs = asyncio.run(run())
    tokens = _tokens(msgs)
    done = [m for m in msgs if m["type"] == "done"]
    assert done, f"cancel produced no done: {msgs[-1]}"
    done = done[0]
    assert done["reason"] == "cancelled"
    assert done["n_tokens"] == len(tokens)
    assert 1 <= len(tokens) < 32, (
        f"cancel did not stop early: {len(tokens)} tokens of max_new=256")


def test_cancel_is_owner_only(live):
    async def run():
        async with (websockets.connect(_ws_url(live), open_timeout=30) as a,
                    websockets.connect(_ws_url(live), open_timeout=30) as b):
            await a.send(json.dumps(_chat(max_new=256, layer=0, coef=0.0)))
            await _next_token(a)

            await b.send(json.dumps({"type": "cancel"}))
            denied = json.loads(await asyncio.wait_for(b.recv(), timeout=30))
            assert denied["type"] == "error"
            assert denied["message"] == "not your generation"

            # A keeps streaming: B's cancel must not have touched it.
            await _next_token(a)

            await a.send(json.dumps({"type": "cancel"}))
            while True:
                last = json.loads(await asyncio.wait_for(a.recv(), timeout=120))
                if last["type"] in ("done", "error"):
                    return denied, last

    denied, last = asyncio.run(run())
    assert denied["message"] == "not your generation"
    assert last["type"] == "done" and last["reason"] == "cancelled"


def test_cancel_when_idle_returns_error(live):
    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps({"type": "cancel"}))
            return json.loads(await asyncio.wait_for(sock.recv(), timeout=30))

    msg = asyncio.run(run())
    assert msg["type"] == "error"
    assert "nothing to cancel" in msg["message"]


def test_second_chat_while_streaming_is_busy(live):
    async def run():
        async with (websockets.connect(_ws_url(live), open_timeout=30) as a,
                    websockets.connect(_ws_url(live), open_timeout=30) as b):
            await a.send(json.dumps(_chat(max_new=256, layer=0, coef=0.0)))
            await _next_token(a)            # skip the `prefill` frame

            res = await asyncio.to_thread(httpx.get, live + "/api/health", timeout=10)
            assert res.json()["busy"] is True

            await b.send(json.dumps(_chat(max_new=2)))
            busy = json.loads(await asyncio.wait_for(b.recv(), timeout=30))
            assert busy["type"] == "error"
            assert "busy" in busy["message"].lower()

            await a.send(json.dumps({"type": "cancel"}))
            while True:
                last = json.loads(await asyncio.wait_for(a.recv(), timeout=120))
                if last["type"] in ("done", "error"):
                    return busy, last

    _, last = asyncio.run(run())
    assert last["type"] == "done" and last["reason"] == "cancelled"
    assert httpx.get(live + "/api/health", timeout=10).json()["busy"] is False


def test_steering_is_per_connection(live):
    _, base_msgs = _collect(live, _chat(max_new=8, layer=None, coef=0.0))
    baseline = _token_ids(base_msgs)

    async def run():
        async with (websockets.connect(_ws_url(live), open_timeout=30) as a,
                    websockets.connect(_ws_url(live), open_timeout=30) as b):
            await a.send(json.dumps(_chat(max_new=8, layer=None, coef=0.0)))
            seen = [json.loads(await asyncio.wait_for(a.recv(), timeout=120))]
            # ...wait until A has generated 3 tokens, then B steers hard.
            while len([m for m in seen if m["type"] == "token"]) < 3:
                seen.append(json.loads(await asyncio.wait_for(a.recv(), timeout=120)))
            await b.send(json.dumps({"type": "steer", "layer": 0, "coef": 3.0}))
            ack = json.loads(await asyncio.wait_for(b.recv(), timeout=30))
            while True:
                msg = json.loads(await asyncio.wait_for(a.recv(), timeout=120))
                seen.append(msg)
                if msg["type"] in ("done", "error"):
                    return seen, ack

    msgs, ack = asyncio.run(run())
    assert ack["type"] == "steer_ack" and ack["coef"] == 3.0
    tokens = _tokens(msgs)
    assert len(tokens) == 8
    assert all(t["steered"] is False and t["coef"] == 0.0 for t in tokens), (
        "another connection's steering leaked into A's stream")
    assert [t["token_id"] for t in tokens] == baseline


def test_neuron_steer_ack_provenance_and_causality(live):
    _, base_msgs = _collect(live, _chat(max_new=3, layer=None, coef=0.0))
    baseline = _token_ids(base_msgs)

    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps(
                {"type": "neuron_steer", "layer": 0, "neuron": 47, "value": 80.0}))
            ack = json.loads(await asyncio.wait_for(sock.recv(), timeout=30))
            await sock.send(json.dumps(_chat(max_new=3, layer=None, coef=0.0)))
            msgs = []
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=120))
                msgs.append(msg)
                if msg["type"] in ("done", "error"):
                    break
            await sock.send(json.dumps({"type": "neuron_clear", "layer": None}))
            clear = json.loads(await asyncio.wait_for(sock.recv(), timeout=30))
            return ack, msgs, clear

    ack, msgs, clear = asyncio.run(run())
    assert ack == {"type": "steer_ack", "layer": 0, "neuron": 47, "value": 80.0}
    tokens = _tokens(msgs)
    assert tokens
    assert all(t["steered"] is True for t in tokens)
    assert tokens[0]["neuron_steer"] == {"0": {"47": 80.0}}
    assert clear == {"type": "steer_ack", "layer": None}
    assert _token_ids(msgs)[0] != baseline[0], (
        "clamping one MLP unit did not change the next token")

    # clearing brings the next run back to the baseline
    _, after = _collect(live, _chat(max_new=3, layer=None, coef=0.0))
    assert _token_ids(after) == baseline


def test_invalid_inputs_return_error(live):
    bad = [
        {"type": "chat", "prompt": "", "max_new": 4},
        {"type": "chat", "prompt": "   ", "max_new": 4},
        {"type": "chat", "prompt": "hi", "max_new": 0},
        {"type": "chat", "prompt": "hi", "max_new": 257},
        {"type": "chat", "prompt": "hi", "max_new": 4, "layer": 99, "coef": 0.0},
        {"type": "chat", "prompt": "hi", "max_new": 4, "layer": 0, "coef": 1e9},
        {"type": "chat", "prompt": "hi", "max_new": 4, "coef": 1.0},   # no layer
        {"type": "chat", "prompt": "x" * 5000, "max_new": 4},
        {"type": "steer", "layer": 99, "coef": 0.5},
        {"type": "steer", "coef": 0.5},
        {"type": "inject", "text": "   "},
        {"type": "edit_prompt", "prompt": ""},
        {"type": "neuron_steer", "layer": 0, "neuron": 9999, "value": 1.0},
        {"type": "neuron_steer", "layer": 99, "neuron": 0, "value": 1.0},
        {"type": "neuron_clear", "layer": 99},
        {"type": "bogus"},
        '{"type": "chat", "prompt": "hi", "max_new": 2, "layer": 0, "coef": Infinity}',
        '{"type": "chat", "prompt": "hi", "max_new": 2, "layer": 0, "coef": NaN}',
    ]
    replies = _send_many(live, bad)
    assert len(replies) == len(bad)
    for request, reply in zip(bad, replies):
        assert reply["type"] == "error", f"{request!r} -> {reply!r}"
        assert reply["message"].strip()


def test_coef_zero_matches_no_steering_end_to_end(live):
    _, plain = _collect(live, _chat(max_new=4, layer=None, coef=0.0))
    _, zero = _collect(live, _chat(max_new=4, layer=1, coef=0.0))
    a = [(m["token_id"], m["topk"], m["steered"]) for m in plain if m["type"] == "token"]
    b = [(m["token_id"], m["topk"], m["steered"]) for m in zero if m["type"] == "token"]
    assert a and a == b, "coef=0 changed the output; it must be off"


# -- unit-level checks (same contract, no server) ----------------------------
def test_engine_wraps_mw_adapter(tiny_model):
    eng = Engine(str(tiny_model), torch.float32)
    assert isinstance(eng.adapter, ModelAdapter)
    assert isinstance(eng.adapter, LiveAdapter)
    assert eng.n_layers == 2
    assert set(eng.vectors) == {0, 1}
    for vec in eng.vectors.values():
        assert vec.shape[-1] == 32
        assert float(vec.norm()) > 0

    state = SteerState()
    state.set_base(1, 2.0)
    eng.adapter.apply_steering(state, eng.vectors)
    assert eng.adapter._steer["layer"] == 1
    assert eng.adapter._steer["coef"] == 2.0
    assert eng.adapter._steer["vec"] is eng.vectors[1]

    state.set_base(1, 0.0)
    eng.adapter.apply_steering(state, eng.vectors)
    assert eng.adapter._steer["vec"] is None


def test_coef_zero_logits_bit_identical(tiny_model):
    eng = Engine(str(tiny_model), torch.float32)
    a = eng.adapter
    ids = a.tok(PROMPT, return_tensors="pt").input_ids

    off = SteerState()
    a.apply_steering(off, eng.vectors)             # no steering
    with torch.no_grad():
        base = a.model(input_ids=ids).logits.clone()

    on = SteerState()
    on.set_base(0, 5.0)
    a.apply_steering(on, eng.vectors)              # steering is live
    with torch.no_grad():
        steered = a.model(input_ids=ids).logits.clone()

    zero = SteerState()
    zero.set_base(1, 0.0)                          # coef == 0 -> off
    a.apply_steering(zero, eng.vectors)
    with torch.no_grad():
        same = a.model(input_ids=ids).logits.clone()

    assert not torch.equal(base, steered), "steering had no effect; test is vacuous"
    assert torch.equal(base, same), "coef=0 must be bit-identical to no steering"


def test_fit_payload_truncates_oversized_trace():
    long = "z" * 4000
    trace = {
        "type": "token", "session": "s_test", "step": 0, "token": long,
        "token_id": 1, "ms": 12,
        "steered": True, "coef": 1.0, "steer_layer": 0, "neuron_steer": None,
        "topk": [[long, 0.5] for _ in range(5)],
        "layers": [
            {"l": i, "norm": 1.0,
             "lens": [[long, 0.2] for _ in range(3)],
             "neurons": [[i * 100 + j, 1.5] for j in range(3)]}
            for i in range(28)
        ],
    }
    assert payload_bytes(trace) > BUDGET
    fitted = fit_payload(trace)
    assert payload_bytes(fitted) <= BUDGET
    assert len(fitted["layers"]) == 28
    for layer in fitted["layers"]:
        assert len(layer["lens"]) <= 3 and len(layer["neurons"]) <= 3


def test_nonfinite_numbers_are_sanitised():
    trace = {
        "type": "token", "session": "s_x", "step": 0, "token": "x", "token_id": 1,
        "ms": float("inf"), "steered": False, "coef": 0.0, "steer_layer": None,
        "neuron_steer": None,
        "topk": [["x", float("nan")]],
        "layers": [{"l": 0, "norm": float("nan"),
                    "lens": [["x", float("inf")]],
                    "neurons": [[1, float("nan")]]}],
    }
    fitted = fit_payload(trace)
    encoded = json.dumps(fitted, allow_nan=False)   # raises if any NaN/Inf leaked
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_watchdog_aborts_a_runaway_generation(tiny_model, tmp_path_factory):
    """A real subprocess with a 50 ms limit: the lock must never stay stuck."""
    port = _free_port()
    log_path = tmp_path_factory.mktemp("watchdog-log") / "server.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "code") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w+", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "app" / "server.py"),
             "--model", str(tiny_model), "--dtype", "float32",
             "--host", "127.0.0.1", "--port", str(port),
             "--gen-timeout", "0.05"],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 180
            while True:
                if proc.poll() is not None:
                    log.flush()
                    log.seek(0)
                    raise RuntimeError(f"server exited during startup:\n{log.read()}")
                try:
                    if httpx.get(base + "/api/info", timeout=1.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("watchdog test server did not start")
                time.sleep(0.1)

            health = httpx.get(base + "/api/health", timeout=10).json()
            assert health["timeout_s"] == 0.05

            async def run():
                async with websockets.connect(_ws_url(base), open_timeout=30) as sock:
                    await sock.send(json.dumps(_chat(max_new=256)))
                    msgs = []
                    while True:
                        msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=60))
                        msgs.append(msg)
                        if msg["type"] in ("done", "error"):
                            return msgs

            msgs = asyncio.run(run())
            done = msgs[-1]
            assert done["type"] == "done"
            assert done["reason"] == "cancelled"
            assert done["timed_out"] is True
            assert done["n_tokens"] < 256

            after = httpx.get(base + "/api/health", timeout=10).json()
            assert after["busy"] is False, "watchdog left the lock held"
            assert after["generations_timed_out"] >= 1
            assert after["generations_done"] >= 1

            # the lock is usable again
            _, again = _collect(base, _chat(max_new=2))
            assert again[-1]["type"] == "done"
        finally:
            rc = None
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
            log.flush()
            log.seek(0)
            text = log.read()
            assert rc == 0, f"watchdog server did not stop cleanly (rc={rc})\n{text}"
