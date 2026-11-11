"""Injection tests: turn format, cache equivalence, queue semantics, steering.

Why a separate file
-------------------
`tests/test_server.py` proves an injection *changes* the argmax sequence. That is
too weak: the old implementation changed it by inserting a second system
preamble mid-conversation and degraded the output
(`'The capital city of France'` -> `'The capitalThe capital of'`). These tests
pin the *shape* of an injection and its behavioural equivalence:

1. The injected token sequence is a turn tail — no duplicated system preamble,
   no second "You are Granite" block, and it starts with the template's own
   turn terminator.
2. The incremental KV-cache path (generate -> inject -> generate) produces the
   same greedy continuation as a clean prefill of the equivalent transcript
   (prompt + assistant prefix + new user turn). This is the real regression
   test: the old format fails it, the fix passes it.
3. Queue semantics: mid-token injections apply at the next boundary, multiple
   injections apply in order, and an injection racing `done` is either taken by
   the closing turn or starts a new turn — never silently dropped.

Honesty about the 2B model
--------------------------
The tiny random Qwen2 used here has no semantics: it cannot "answer in French".
So the *meaning* claim (injection steers the content) is checked against the
live Granite server in `test_live_granite_injection_steers`, which is opt-in
(MOD_WEI_LIVE_URL) because it needs the real weights. The tiny-model tests
check the mechanism, not the meaning.

Run:
  PYTHONPATH=code .venv/bin/python -m pytest tests/test_injection.py -q
  MOD_WEI_LIVE_URL=http://127.0.0.1:8000 ... -k live_granite
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

import pytest
import torch
import websockets
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from app.server import GenControl, LiveAdapter, SteerState

pytestmark = pytest.mark.slow

VOCAB = ["[UNK]", "def", "add", "return", "a", "+", "b", ":", "\n", "x", "=", "1"]
PROMPT = "def add a b :"
TEXT = "Use a while loop"

# The real Granite 3.x template, trimmed to the branches that matter here: it
# synthesises a system preamble when messages[0] is not a system message, and
# reopens the assistant role when add_generation_prompt is true. This is the
# template whose standalone single-user render produced the 67-token injection.
GRANITE_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content'] %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
    "{%- set system_message = \"Knowledge Cutoff Date: April 2024.\\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant.\" %}"
    "{%- set loop_messages = messages %}"
    "{%- endif %}"
    "{{- '<|start_of_role|>system<|end_of_role|>' + system_message "
    "+ '<|end_of_text|>\\n' }}"
    "{%- for message in loop_messages %}"
    "{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' "
    "+ message['content'] + '<|end_of_text|>\\n' }}"
    "{%- if loop.last and add_generation_prompt %}"
    "{{- '<|start_of_role|>assistant<|end_of_role|>' }}"
    "{%- endif %}"
    "{%- endfor %}"
)

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] "
    "+ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


# -- tiny model --------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tiny-inject")
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[UNK]",
        additional_special_tokens=["[EOS]"])
    tok.eos_token = "[EOS]"
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


@pytest.fixture(scope="module")
def adapter(tiny_model) -> LiveAdapter:
    return LiveAdapter(str(tiny_model)).load(dtype=torch.float32)


def _greedy(a: LiveAdapter, ids: torch.Tensor, n: int) -> list[int]:
    past, out_ids = None, []
    with torch.no_grad():
        for _ in range(n):
            out = a.model(input_ids=ids if past is None else ids[:, -1:],
                          past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[:, -1, :].float().argmax(-1))
            out_ids.append(nxt)
            ids = torch.tensor([[nxt]])
    return out_ids


# -- format: no duplicated preamble ------------------------------------------
def _adapter_with_template(tmp_path, template: str) -> LiveAdapter:
    """Tokenizer-only adapter: no weights, just the real template logic."""
    d = Path(tmp_path)
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[UNK]",
        additional_special_tokens=["[EOS]"])
    tok.chat_template = template
    a = LiveAdapter(str(d))
    a.tok = tok
    return a


def test_injection_delta_is_only_a_turn_tail(tmp_path):
    """Root cause: a standalone single-turn render re-emits the preamble."""
    a = _adapter_with_template(tmp_path, GRANITE_TEMPLATE)
    text = "The password is ZEBRA."

    full_single_turn = a.tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=False)
    assert "<|start_of_role|>system<|end_of_role|>" in full_single_turn

    delta = a._injection_delta(text)
    assert delta is not None
    print("\nold single-turn render:", repr(full_single_turn))
    print("new injection delta  :", repr(delta))

    # the actual injected tokens
    assert text in delta
    assert "<|start_of_role|>system<|end_of_role|>" not in delta, (
        "the injection duplicates the system preamble")
    assert "Knowledge Cutoff" not in delta
    assert "You are Granite" not in delta
    # it closes the in-progress assistant message and opens a new turn
    assert delta.startswith("<|end_of_text|>"), (
        "the in-progress assistant turn is not terminated")
    assert "<|start_of_role|>user<|end_of_role|>" in delta
    assert delta.rstrip().endswith("<|start_of_role|>assistant<|end_of_role|>")
    assert len(delta) < len(full_single_turn)


def test_injection_delta_is_generic_for_chatml(tmp_path):
    a = _adapter_with_template(tmp_path, CHATML_TEMPLATE)
    delta = a._injection_delta("be brief")
    assert delta == "<|im_end|>\n<|im_start|>user\nbe brief<|im_end|>\n<|im_start|>assistant\n"


def test_granite_token_count_drops_from_67_to_18():
    """Instrumentation against the real Granite tokenizer (no weights loaded)."""
    path = ROOT / ".hf" / "granite-hf"
    if not path.exists():
        pytest.skip("Granite tokenizer not present; covered by the replica above")
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from transformers import AutoTokenizer
    from app.server import LiveAdapter as LA

    tok = AutoTokenizer.from_pretrained(str(path))
    a = LA(str(path))
    a.tok = tok
    text = "The password is ZEBRA."
    ids = a.format_injection(text)
    decoded = tok.decode(ids[0])
    print("\nformatted injection:", ids.shape[1], "tokens")
    print("decoded            :", repr(decoded))
    assert ids.shape[1] < 30, (
        f"injection is {ids.shape[1]} tokens; the single-turn render was 67")
    assert "<|start_of_role|>system" not in decoded
    assert "Knowledge Cutoff" not in decoded
    assert decoded.startswith("<|end_of_text|>")
    assert decoded.count(text) == 1


def test_no_template_fallback_is_still_a_delimited_turn(tmp_path):
    a = _adapter_with_template(tmp_path, GRANITE_TEMPLATE)
    a.use_chat_template = False

    class _Spy:
        def __init__(self, tok):
            self.tok, self.seen = tok, []

        def __call__(self, text, **kwargs):
            self.seen.append(text)
            return self.tok(text, **kwargs)

    spy = _Spy(a.tok)
    a.tok = spy
    a.format_injection("hi")
    assert spy.seen == ["\nUser: hi\nAssistant:"]


# -- behavioural equivalence --------------------------------------------------
def test_injection_equals_a_clean_new_turn(adapter):
    """The incremental path must equal a fresh prefill of the same transcript.

    Old format fails: it feeds <system>Knowledge Cutoff...</system><user>text
    where a clean transcript has <user>text, so the continuation diverges.
    """
    a = adapter
    msgs = [{"role": "user", "content": PROMPT}]
    control = GenControl(SteerState())
    at = 3
    yielded: list[int] = []
    for tr in a.stream(msgs, 8, control=control):
        yielded.append(tr["token_id"])
        if len(yielded) == at:
            assert control.inject(TEXT) is True
    assert len(yielded) == 8

    prefix_text = a.tok.decode(yielded[:at])
    clean_msgs = msgs + [
        {"role": "assistant", "content": prefix_text},
        {"role": "user", "content": TEXT},
    ]
    clean_ids = a.format_messages(clean_msgs)
    expected = _greedy(a, clean_ids, 8 - at)
    assert yielded[at:] == expected, (
        "the injected turn is not equivalent to appending a user turn to the "
        f"transcript: {yielded[at:]} != {expected}")

    # and the injection did not rewrite history
    control2 = GenControl(SteerState())
    baseline = [tr["token_id"] for tr in a.stream(msgs, at, control=control2)]
    assert yielded[:at] == baseline


def test_injection_is_fed_exactly_once_in_order(adapter):
    """Two injections at one boundary: ext1 then ext2, each exactly once."""
    a = adapter
    control = GenControl(SteerState())
    seen: list[str] = []
    calls: list[list[int]] = []
    real_forward = a.model.forward

    def spy(*args, **kwargs):
        calls.append(kwargs["input_ids"][0].tolist())
        return real_forward(*args, **kwargs)

    a.model.forward = spy
    yielded: list[int] = []
    first, second = "return", "="
    try:
        for tr in a.stream([{"role": "user", "content": PROMPT}], 5,
                           control=control,
                           on_injected=lambda _at, _n, text: seen.append(text)):
            yielded.append(tr["token_id"])
            if len(yielded) == 2:
                assert control.inject(first)
                assert control.inject(second)
    finally:
        del a.model.forward

    assert seen == [first, second], "injections were reordered"

    e1 = a.format_injection(first)[0].tolist()
    e2 = a.format_injection(second)[0].tolist()
    assert e1 != e2
    # multi-token forwards are: prefill, then the two injection forwards
    multi = [c for c in calls if len(c) > 1]
    assert len(multi) == 3, f"expected 2 injection forwards, got {len(multi) - 1}"
    assert multi[0] == a.format_messages([{"role": "user", "content": PROMPT}])[0].tolist()
    assert multi[1] == [yielded[1]] + e1, (
        "the first injection is not fed right after the pre-boundary token")
    assert multi[2] == e2, "the second injection is not fed after the first"


# -- queue handshake ----------------------------------------------------------
def test_closed_turn_refuses_late_injection():
    control = GenControl(SteerState())
    assert control.inject("a") is True
    assert control.inject("b") is True
    assert control.pop_injection() == "a"
    assert control.pop_injection() == "b"
    assert control.pop_injection() is None

    control.close()
    assert control.inject("late") is False, (
        "an injection queued after the turn closed would be lost")
    control.reopen()
    assert control.inject("next turn") is True
    assert control.pop_injection() == "next turn"


# -- live tiny server ---------------------------------------------------------
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live(tiny_model, tmp_path_factory):
    import httpx

    port = _free_port()
    log_path = tmp_path_factory.mktemp("inject-log") / "server.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "code") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w+", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "app" / "server.py"),
             "--model", str(tiny_model), "--dtype", "float32",
             "--host", "127.0.0.1", "--port", str(port),
             "--gen-timeout", "60"],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 180
            while True:
                if proc.poll() is not None:
                    log.flush(); log.seek(0)
                    raise RuntimeError(f"server exited during startup:\n{log.read()}")
                try:
                    if httpx.get(base + "/api/info", timeout=1.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("injection test server did not start")
                time.sleep(0.1)
            yield base
        finally:
            rc = None
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
            assert rc == 0


def _ws_url(base: str) -> str:
    return base.replace("http://", "ws://") + "/ws"


def test_inject_after_done_starts_a_new_turn(live):
    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps({"type": "chat", "prompt": PROMPT,
                                        "max_new": 3}))
            first = []
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=60))
                first.append(msg)
                if msg["type"] in ("done", "error"):
                    break
            assert first[-1]["type"] == "done"
            sid1 = first[-1]["session"]

            await sock.send(json.dumps({"type": "inject", "text": "second turn"}))
            ack = json.loads(await asyncio.wait_for(sock.recv(), timeout=60))
            assert ack == {"type": "injected", "session": sid1, "at_step": 0,
                           "tokens": ack["tokens"], "text": "second turn"}
            assert ack["tokens"] >= 1
            tokens, done = 0, None
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=60))
                if msg["type"] == "token":
                    tokens += 1
                if msg["type"] in ("done", "error"):
                    done = msg
                    break
            return sid1, tokens, done

    sid1, tokens, done = asyncio.run(run())
    assert done["type"] == "done", done
    assert done["session"] != sid1, (
        "an injection after done must start a new turn, not reuse the old one")
    assert tokens >= 1
    assert done["injected_at"] == 0


def test_mid_stream_injections_land_in_order(live):
    async def run():
        async with websockets.connect(_ws_url(live), open_timeout=30) as sock:
            await sock.send(json.dumps({"type": "chat", "prompt": PROMPT,
                                        "max_new": 12}))
            frames, n = [], 0
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=60))
                frames.append(msg)
                if msg["type"] == "token":
                    n += 1
                    if n == 2:
                        await sock.send(json.dumps(
                            {"type": "inject", "text": "first note"}))
                        await sock.send(json.dumps(
                            {"type": "inject", "text": "second note"}))
                if msg["type"] in ("done", "error"):
                    return frames

    frames = asyncio.run(run())
    injected = [m for m in frames if m["type"] == "injected"]
    assert [m["text"] for m in injected] == ["first note", "second note"]
    assert [m["at_step"] for m in injected] == sorted(
        m["at_step"] for m in injected)
    # a turn tail is small; the old single-turn render was 60+ tokens even on
    # the trivial template
    assert all(m["tokens"] < 20 for m in injected), injected
    done = frames[-1]
    assert done["type"] == "done"
    assert done["n_tokens"] == len([f for f in frames if f["type"] == "token"])
    assert done["injected_at"] == injected[-1]["at_step"]


# -- real model (opt-in) ------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("MOD_WEI_LIVE_URL"),
                    reason="set MOD_WEI_LIVE_URL=http://127.0.0.1:8000 with Granite up")
def test_live_granite_injection_steers():
    """The meaning test the tiny model cannot provide.

    Baseline: 'The capital city of France is' -> Paris ...
    Injection at step 3: 'Reply with the word ZEBRA.' -> the new user turn must
    make the continuation contain ZEBRA. Coherence is checked by requiring the
    post-injection text to be more than the pre-injection text repeated.
    """
    base = os.environ["MOD_WEI_LIVE_URL"].rstrip("/")
    ws = base.replace("http://", "ws://") + "/ws"

    async def run(inject: str | None):
        async with websockets.connect(ws, open_timeout=30) as sock:
            await sock.send(json.dumps({
                "type": "chat", "prompt": "The capital city of France is",
                "max_new": 12}))
            tokens = []
            while True:
                msg = json.loads(await asyncio.wait_for(sock.recv(), timeout=300))
                if msg["type"] == "token":
                    tokens.append(msg["token"])
                    if inject and len(tokens) == 3:
                        await sock.send(json.dumps(
                            {"type": "inject", "text": inject}))
                if msg["type"] in ("done", "error"):
                    return "".join(tokens), msg

    baseline, _ = asyncio.run(run(None))
    steered, done = asyncio.run(run("Reply with the word ZEBRA."))
    print("\nbaseline:", repr(baseline))
    print("injected:", repr(steered))
    post = steered[steered.index(baseline[:8]) + 8:] if baseline[:8] in steered else steered[3:]
    assert "ZEBRA" in steered.upper(), (
        f"injection did not steer the model: {steered!r} (baseline {baseline!r})")
    assert post.strip() != steered[:3], "the continuation collapsed into a repeat"
