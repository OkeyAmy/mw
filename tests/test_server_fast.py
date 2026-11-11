"""Fast protocol-contract tests — no model, no subprocess, safe for CI.

H3-4: the subprocess suite is `slow`, so it never ran in the default gate. These
tests exercise the app object itself (FastAPI TestClient) and the input/payload
contracts, so a broken route or validator fails `pytest -m "not slow"`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

import app.server as srv
from app.server import (
    MAX_TOKEN_BYTES,
    ChatIn,
    EditPromptIn,
    GenControl,
    Generation,
    InjectIn,
    NeuronSteerIn,
    SteerIn,
    SteerState,
    app,
    fit_payload,
    payload_bytes,
)

client = TestClient(app)


def test_info_is_503_while_loading():
    # ENGINE is None in this process: no model was loaded.
    r = client.get("/api/info")
    assert r.status_code == 503
    assert r.json() == {"ready": False}


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "ok", "busy", "busy_s", "uptime_s", "generations",
        "generations_started", "generations_done", "generations_timed_out",
        "timeout_s",
    }
    assert body["ok"] is False
    assert isinstance(body["busy"], bool)
    assert body["busy_s"] >= 0
    assert body["uptime_s"] >= 0
    assert body["generations"] == body["generations_done"]
    assert body["timeout_s"] > 0


def test_unknown_sessions_are_404():
    assert client.get("/api/export/s_nope").status_code == 404
    assert client.get("/api/detail/s_nope/0").status_code == 404


def test_root_redirects_to_the_next_ui():
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "http://localhost:3000"


def test_chat_bounds_and_defaults():
    chat = ChatIn.model_validate({"type": "chat", "prompt": "def f():"})
    assert chat.max_new == 40
    assert chat.layer is None and chat.coef == 0.0
    for bad in (
        {"type": "chat", "prompt": ""},
        {"type": "chat", "prompt": "  "},
        {"type": "chat", "prompt": "x" * 4001},
        {"type": "chat", "prompt": "hi", "max_new": 0},
        {"type": "chat", "prompt": "hi", "max_new": 257},
        {"type": "chat", "prompt": "hi", "coef": 21.0},
        {"type": "chat", "prompt": "hi", "coef": float("inf")},
    ):
        with pytest.raises(ValidationError):
            ChatIn.model_validate(bad)


def test_steer_and_intervention_bounds():
    SteerIn.model_validate({"type": "steer", "layer": 0})
    for bad in (
        {"type": "steer", "coef": 1.0},
        {"type": "steer", "layer": -1},
        {"type": "steer", "layer": 0, "coef": 1e9},
        {"type": "steer", "layer": 0, "coef": float("nan")},
    ):
        with pytest.raises(ValidationError):
            SteerIn.model_validate(bad)

    NeuronSteerIn.model_validate({"type": "neuron_steer", "layer": 0,
                                  "neuron": 1, "value": 0.0})
    for bad in (
        {"type": "neuron_steer", "layer": 0, "neuron": -1, "value": 0.0},
        {"type": "neuron_steer", "layer": 0, "neuron": 1, "value": 1e9},
    ):
        with pytest.raises(ValidationError):
            NeuronSteerIn.model_validate(bad)

    with pytest.raises(ValidationError):
        InjectIn.model_validate({"type": "inject", "text": "   "})
    with pytest.raises(ValidationError):
        EditPromptIn.model_validate({"type": "edit_prompt", "prompt": ""})


def test_payload_budget_holds_for_a_production_shape():
    long = "z" * 4000
    trace = {
        "type": "token", "session": "s_x", "step": 0, "token": long,
        "token_id": 1, "ms": 12, "steered": False, "coef": 0.0,
        "steer_layer": None, "neuron_steer": None,
        "topk": [[long, 0.5] for _ in range(5)],
        "layers": [{"l": i, "norm": 1.0,
                    "lens": [[long, 0.2] for _ in range(3)],
                    "neurons": [[i, 1.5] for _ in range(3)]} for i in range(28)],
    }
    fitted = fit_payload(trace)
    assert payload_bytes(fitted) <= MAX_TOKEN_BYTES
    assert len(fitted["layers"]) == 28


def test_chat_accepts_messages_and_validates_transcripts():
    chat = ChatIn.model_validate({
        "type": "chat", "max_new": 2,
        "messages": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello"}]})
    assert chat.prompt is None and len(chat.messages) == 2

    for bad in (
        {"type": "chat", "max_new": 2},                         # neither field
        {"type": "chat", "messages": [], "max_new": 2},         # empty list
        {"type": "chat", "messages": [{"role": "tool", "content": "x"}]},
        {"type": "chat", "messages": [{"role": "user", "content": "x" * 4001}]},
        {"type": "chat", "messages": [{"role": "user", "content": "x" * 4000}] * 5},
        {"type": "chat", "prompt": "", "max_new": 2},
    ):
        with pytest.raises(ValidationError):
            ChatIn.model_validate(bad)


class _FakeTok:
    eos_token_id = -1


class _FakeAdapter:
    @staticmethod
    def format_injection(text):
        return [[0, 1]]


class _FakeEngine:
    """Just enough surface for _worker: stream() and detail()."""

    def __init__(self, fail: bool = False):
        self.tok = _FakeTok()
        self.adapter = _FakeAdapter()
        self.fail = fail

    def stream(self, messages, max_new, control, on_injected=None):
        if self.fail:
            raise RuntimeError("adapter exploded")
        for i in range(2):
            yield {"step": i, "token": "x", "token_id": i, "ms": 1,
                   "topk": [], "layers": []}

    def detail(self):
        return []


def _sink(bucket: list):
    """`send` must report delivery (the worker treats a falsy send as a hang-up)."""
    def send(payload: dict) -> bool:
        bucket.append(payload)
        return True
    return send


def _new_gen() -> Generation:
    return Generation([{"role": "user", "content": "hi"}], 2,
                      GenControl(SteerState()))


def test_worker_releases_lock_when_adapter_raises(monkeypatch):
    started = srv.GENERATIONS_STARTED
    done = srv.GENERATIONS_DONE
    monkeypatch.setattr(srv, "ENGINE", _FakeEngine(fail=True))
    sent = []
    srv._worker(_new_gen(), _sink(sent))

    assert not srv.GEN_LOCK.locked(), "a raising adapter must not leak the lock"
    assert srv.GENERATIONS_STARTED == started + 1
    assert srv.GENERATIONS_DONE == done + 1
    assert sent and sent[-1]["type"] == "error"
    assert "adapter exploded" in sent[-1]["message"]


def test_worker_sends_done_with_provenance_and_counts(monkeypatch):
    started = srv.GENERATIONS_STARTED
    done = srv.GENERATIONS_DONE
    monkeypatch.setattr(srv, "ENGINE", _FakeEngine())
    sent = []
    srv._worker(_new_gen(), _sink(sent))

    assert not srv.GEN_LOCK.locked()
    assert srv.GENERATIONS_STARTED == started + 1
    assert srv.GENERATIONS_DONE == done + 1
    frame = sent[-1]
    assert frame["type"] == "done"
    assert frame["reason"] == "max_new"
    assert frame["n_tokens"] == 2
    assert frame["injected_at"] is None
    assert frame["timed_out"] is False


def test_watchdog_cancels_then_force_releases(monkeypatch):
    monkeypatch.setattr(srv, "GEN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(srv, "GEN_HARD_GRACE_S", 0.5)
    timed_out = srv.GENERATIONS_TIMED_OUT

    gen = _new_gen()
    monkeypatch.setitem(srv.CURRENT, "gen", gen)
    assert srv.GEN_LOCK.acquire(blocking=False)
    t0 = srv.GEN_LOCK.started_at

    srv._watchdog_tick(now=t0 + 2.0)
    assert gen.timed_out is True
    assert gen.control.cancel.is_set(), "watchdog did not cancel at the limit"
    assert srv.GEN_LOCK.locked()
    assert srv.GENERATIONS_TIMED_OUT == timed_out + 1

    # a generation stuck inside a forward never sees the cancel: force release
    srv._watchdog_tick(now=t0 + 2.0)
    assert not srv.GEN_LOCK.locked(), "watchdog did not force-release a stuck lock"
    srv.GEN_LOCK.release()          # the worker's finally must be idempotent
    assert not srv.GEN_LOCK.locked()


def test_nan_and_infinity_never_reach_json():
    trace = {
        "type": "token", "session": "s_x", "step": 0, "token": "x", "token_id": 1,
        "ms": float("inf"), "steered": False, "coef": 0.0, "steer_layer": None,
        "neuron_steer": None, "topk": [["x", float("nan")]],
        "layers": [{"l": 0, "norm": float("nan"),
                    "lens": [["x", float("inf")]],
                    "neurons": [[1, float("nan")]]}],
    }
    encoded = json.dumps(fit_payload(trace), allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
