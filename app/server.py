#!/usr/bin/env python3
"""
mw live server — stream tokens AND the internal trace behind each one.

Implements app/PROTOCOL.md on top of the audited `mw.adapter.ModelAdapter`.
Backend responsibilities:
  * load exactly one model, under the memory guard (inside adapter.load)
  * serve one generation at a time, with a lock released in a finally
  * per-connection steering / neuron intervention (no client affects another)
  * mid-generation context injection and prompt edits (real tokens, real cache)
  * stream a bounded trace payload per token (websocket, < 40 KB)
  * expose health/info/export/detail so the UI can fail loudly

Run:
  PYTHONPATH=code .venv/bin/python app/server.py --model .hf/bf16-hf --dtype float32
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from mw.adapter import ModelAdapter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

ROOT = Path(__file__).resolve().parent.parent

# -- protocol bounds (PROTOCOL.md) -------------------------------------------
MAX_TOKEN_BYTES = 40 * 1024      # "payload MUST stay under ~40 KB per token"
MAX_NEW_LIMIT = 256              # max_new: 1..256
MAX_PROMPT_CHARS = 4000
MAX_TRANSCRIPT_CHARS = 16000     # whole `messages` transcript cap
MAX_INJECT_CHARS = 2000
MAX_COEF = 20.0                  # coef is bounded: units of mean residual norm
MAX_NEURON_VALUE = 1e4           # an MLP activation clamp, not a weight edit
MAX_SESSIONS = 20                # bounded in-memory export store
MAX_STORED_TOKENS = MAX_NEW_LIMIT
LENS_EVERY = 2                   # first token: all layers; later: every 2nd
SEND_TIMEOUT_S = 30.0
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


# -- payload budget ----------------------------------------------------------
def payload_bytes(payload: dict) -> int:
    """Bytes on the wire. Starlette's send_json uses ensure_ascii=True, so do we."""
    return len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode())


def _clean(value, default=0.0):
    """NaN/Infinity are not valid JSON: replace them instead of breaking the frame."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return default


def fit_payload(payload: dict) -> dict:
    """Sanitise then shrink trace detail until the token payload fits.

    Never drops a layer entry (PROTOCOL: one per layer, ordered); trims the
    optional lens/neuron lists first, then clips decoded strings. The final
    assert is the contract: if it cannot fit, we raise instead of shipping an
    oversized frame.
    """
    p = dict(payload)
    p["layers"] = [dict(layer) for layer in payload.get("layers", [])]
    p["ms"] = _clean(p.get("ms", 0))
    p["topk"] = [[str(s), _clean(v)] for s, v in (p.get("topk") or [])]
    for layer in p["layers"]:
        layer["norm"] = _clean(layer.get("norm", 0.0))
        layer["lens"] = [[str(s), _clean(v)] for s, v in (layer.get("lens") or [])]
        layer["neurons"] = [[int(i), _clean(v)] for i, v in (layer.get("neurons") or [])]

    if payload_bytes(p) <= MAX_TOKEN_BYTES:
        return p

    for key in ("neurons", "lens"):
        for keep in (1, 0):
            for layer in p["layers"]:
                entries = layer.get(key) or []
                if len(entries) > keep:
                    layer[key] = entries[:keep]
            if payload_bytes(p) <= MAX_TOKEN_BYTES:
                return p

    for limit in (32, 16, 8):
        p["token"] = str(p.get("token", ""))[:limit]
        p["topk"] = [[str(s)[:limit], v] for s, v in p["topk"]]
        for layer in p["layers"]:
            layer["lens"] = [[str(s)[:limit], v] for s, v in layer["lens"]]
        if payload_bytes(p) <= MAX_TOKEN_BYTES:
            return p

    n = payload_bytes(p)
    assert n <= MAX_TOKEN_BYTES, (
        f"token payload is {n} bytes, protocol budget is {MAX_TOKEN_BYTES}; "
        f"refusing to send an oversized frame")
    return p


# -- input validation --------------------------------------------------------
def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("value must be a finite number")
    return v


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def _content(cls, v: str) -> str:
        if len(v) > MAX_PROMPT_CHARS:
            raise ValueError(f"message too long (max {MAX_PROMPT_CHARS} chars)")
        return v


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["chat"]
    prompt: str | None = None
    messages: list[Message] | None = None
    max_new: int = Field(default=40, ge=1, le=MAX_NEW_LIMIT)
    layer: int | None = Field(default=None, ge=0)
    coef: float = Field(default=0.0, ge=-MAX_COEF, le=MAX_COEF)

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.strip():
            raise ValueError("prompt must be non-empty")
        if len(v) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt too long (max {MAX_PROMPT_CHARS} chars)")
        return v

    @model_validator(mode="after")
    def _one_of_prompt_or_messages(self):
        if self.messages:
            total = sum(len(m.content) for m in self.messages)
            if total > MAX_TRANSCRIPT_CHARS:
                raise ValueError(
                    f"conversation too long (max {MAX_TRANSCRIPT_CHARS} chars total)")
            return self
        if self.messages is not None:
            raise ValueError("messages must be non-empty when provided")
        if self.prompt is None:
            raise ValueError("chat needs either messages or prompt")
        return self

    @field_validator("coef")
    @classmethod
    def _coef(cls, v: float) -> float:
        return _finite(v)


class SteerIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["steer"]
    layer: int = Field(ge=0)
    coef: float = Field(default=0.0, ge=-MAX_COEF, le=MAX_COEF)

    @field_validator("coef")
    @classmethod
    def _coef(cls, v: float) -> float:
        return _finite(v)


class CancelIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["cancel"]


class InjectIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["inject"]
    text: str

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("injected text must be non-empty")
        if len(v) > MAX_INJECT_CHARS:
            raise ValueError(f"injected text too long (max {MAX_INJECT_CHARS} chars)")
        return v


class EditPromptIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["edit_prompt"]
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must be non-empty")
        if len(v) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt too long (max {MAX_PROMPT_CHARS} chars)")
        return v


class NeuronSteerIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["neuron_steer"]
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)
    value: float = Field(ge=-MAX_NEURON_VALUE, le=MAX_NEURON_VALUE)

    @field_validator("value")
    @classmethod
    def _value(cls, v: float) -> float:
        return _finite(v)


class NeuronClearIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["neuron_clear"]
    layer: int | None = Field(default=None, ge=0)


def describe_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(x) for x in first.get("loc", ()))
    msg = first.get("msg", "invalid message")
    return f"invalid request: {where}: {msg}" if where else f"invalid request: {msg}"


# -- per-connection state ----------------------------------------------------
class SteerState:
    """One connection's steering. Never shared: no client steers another."""

    __slots__ = ("coef", "layer", "neurons")

    def __init__(self):
        self.layer: int | None = None
        self.coef: float = 0.0
        self.neurons: dict[tuple[int, int], float] = {}

    def set_base(self, layer: int | None, coef: float) -> None:
        self.layer = layer
        self.coef = coef

    def steered(self) -> bool:
        return bool(self.neurons) or (self.layer is not None and self.coef != 0.0)

    def neuron_payload(self) -> dict | None:
        if not self.neurons:
            return None
        out: dict[str, dict[str, float]] = {}
        for (layer, neuron), value in self.neurons.items():
            out.setdefault(str(layer), {})[str(neuron)] = round(float(value), 3)
        return out


class GenControl:
    """Cross-thread control for one generation run.

    The injection queue is guarded by `_inj_lock` together with a `_closed`
    flag. A turn is closed before the worker decides to stop, so an `inject`
    that races the last token either lands in this turn (it was queued before
    the close) or is reported as refused and starts a new turn in the socket
    handler. Without that handshake an injection arriving between the worker's
    final `pop_injection()` and `done` was silently lost.
    """

    def __init__(self, steer: SteerState):
        self.steer = steer
        self.cancel = threading.Event()
        self.injections: queue.Queue[str] = queue.Queue()
        self.restart: list[dict] | None = None   # replacement transcript
        self.step = 0          # monotonic across restarts / injected turns
        self.eos = False       # set by the stream loop when it hits eos
        self._inj_lock = threading.Lock()
        self._closed = False

    def inject(self, text: str) -> bool:
        """Queue an injection for this turn. False = this turn has closed."""
        with self._inj_lock:
            if self._closed:
                return False
            self.injections.put(text)
            return True

    def close(self) -> None:
        with self._inj_lock:
            self._closed = True

    def reopen(self) -> None:
        with self._inj_lock:
            self._closed = False

    def pop_injection(self) -> str | None:
        with self._inj_lock:
            try:
                return self.injections.get_nowait()
            except queue.Empty:
                return None


class Generation:
    def __init__(self, messages: list[dict], max_new: int, control: GenControl):
        self.messages = messages                   # full transcript
        self.max_new = max_new
        self.control = control
        self.session: str | None = None
        self.text = ""
        self.injected_at: int | None = None   # last step an injection landed at
        self.timed_out: bool = False          # set by the watchdog

    @property
    def prompt(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def transcript(self) -> str:
        """Rendered conversation for the export metadata."""
        return "".join(
            f"{m.get('role', 'user').title()}: {m.get('content', '')}\n"
            for m in self.messages)


class GenLock:
    """A lock with an owner timer, so a stuck generation can be timed out.

    `release()` is idempotent after a watchdog `force_release()`: the worker's
    top-level finally must never raise because the watchdog got there first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = threading.Lock()
        self.started_at: float | None = None
        self._forced = False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if timeout is not None and timeout >= 0:
            ok = self._lock.acquire(blocking=True, timeout=timeout)
        else:
            ok = self._lock.acquire(blocking=blocking)
        if ok:
            with self._state:
                self.started_at = time.monotonic()
                self._forced = False
        return ok

    def release(self) -> None:
        with self._state:
            if self._forced:
                self._forced = False       # watchdog already released it
                return
            self.started_at = None
        with contextlib.suppress(RuntimeError):
            self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def busy_s(self) -> float:
        with self._state:
            if self.started_at is None or not self._lock.locked():
                return 0.0
            return round(time.monotonic() - self.started_at, 1)

    def force_release(self) -> bool:
        with self._state:
            if not self._lock.locked():
                return False
            self._forced = True
            self.started_at = None
        with contextlib.suppress(RuntimeError):
            self._lock.release()
        return True


# -- session store -----------------------------------------------------------
class SessionStore:
    """Bounded (last 20) record of streaming/finished sessions."""

    def __init__(self, maxlen: int = MAX_SESSIONS):
        self._maxlen = maxlen
        self._d: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, *, prompt: str, max_new: int, layer: int | None, coef: float,
               messages: list[dict] | None = None) -> str:
        with self._lock:
            while True:
                sid = "s_" + uuid.uuid4().hex[:4]
                if sid not in self._d:
                    break
            self._d[sid] = {
                "session": sid,
                "prompt": prompt,
                "messages": list(messages or []),
                "max_new": max_new,
                "layer": layer,
                "coef": coef,
                "started_at": time.time(),
                "finished_at": None,
                "reason": None,
                "n_tokens": 0,
                "tokens": [],
                "details": {},      # step -> full top-32 layer payload
            }
            while len(self._d) > self._maxlen:
                self._d.popitem(last=False)
        return sid

    def add_token(self, sid: str, payload: dict, detail: list | None = None) -> None:
        with self._lock:
            rec = self._d.get(sid)
            if rec is None or len(rec["tokens"]) >= MAX_STORED_TOKENS:
                return
            rec["tokens"].append(payload)
            rec["n_tokens"] = len(rec["tokens"])
            if detail is not None:
                rec["details"][int(payload["step"])] = {
                    "session": sid,
                    "step": int(payload["step"]),
                    "token": payload.get("token", ""),
                    "layers": detail,
                }

    def finish(self, sid: str, reason: str, n_tokens: int) -> None:
        with self._lock:
            rec = self._d.get(sid)
            if rec is None:
                return
            rec["reason"] = reason
            rec["n_tokens"] = n_tokens
            rec["finished_at"] = time.time()

    def get(self, sid: str) -> dict | None:
        with self._lock:
            rec = self._d.get(sid)
            if rec is None:
                return None
            return json.loads(json.dumps(rec))   # copy, no shared mutation

    def get_detail(self, sid: str, step: int) -> dict | None:
        with self._lock:
            rec = self._d.get(sid)
            if rec is None:
                return None
            detail = rec["details"].get(step)
            return json.loads(json.dumps(detail)) if detail is not None else None


# -- engine ------------------------------------------------------------------
class LiveAdapter(ModelAdapter):
    """The audited adapter plus the control the live server needs.

    Model loading, guard, hooks, logit lens, scaling, steering and neuron
    intervention all stay in `mw.adapter.ModelAdapter`. This subclass only adds
    the live loop: cancel, mid-generation cache extension, per-step lens
    schedule, and per-connection steering applied before each forward.

    Injection formatting (measured, 2026-11-11): appending PLAIN text to the
    context has NO effect on an instruct model (probe: ' Paris.' with and
    without an inserted sentence). Formatting the insertion as a user turn DOES
    change the continuation. So we render it through the model's own chat
    template when one exists, and fall back to a delimited turn otherwise.

    The turn must be rendered as a DELTA against the conversation that is
    already in the KV cache. Rendering `[{"role": "user", ...}]` on its own
    re-emits the whole preamble: on Granite each injection inserted a second
    '<|start_of_role|>system<|end_of_role|>Knowledge Cutoff ... You are
    Granite' block mid-stream (67 tokens instead of 18) and left the assistant
    turn unterminated. That is what corrupted the continuation.
    """

    use_chat_template: bool = True

    def _has_template(self) -> bool:
        return bool(self.use_chat_template and getattr(self.tok, "chat_template", None))

    def _injection_delta(self, text: str) -> str | None:
        """The template's own 'close turn + user turn + assistant prompt' text.

        Render the same conversation twice: once up to the open assistant turn
        (`head`), once with an *empty* assistant message and the injected user
        turn appended (`full`). `head` is a prefix of `full` for every sane
        chat template, so the suffix is exactly what may be appended to a
        conversation already in the cache:

            <turn terminator><user role>text<turn terminator><assistant role>

        No system preamble, because the preamble is inside the common prefix.
        Works for Granite, ChatML/Qwen and Llama-3 style templates; returns
        None when the template does not have that shape.
        """
        probe = [{"role": "user", "content": "x"}]
        try:
            head = self.tok.apply_chat_template(
                probe, add_generation_prompt=True, tokenize=False)
            full = self.tok.apply_chat_template(
                probe + [{"role": "assistant", "content": ""},
                         {"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        except Exception:  # noqa: BLE001 - a template that refuses this shape
            return None
        if not isinstance(head, str) or not isinstance(full, str):
            return None
        if full.startswith(head) and len(full) > len(head):
            return full[len(head):]
        # Whitespace at the seam may differ; fall back to the common prefix but
        # only accept a delta that actually carries the injected text.
        n = 0
        for a, b in zip(head, full):
            if a != b:
                break
            n += 1
        delta = full[n:]
        return delta if delta and text in delta else None

    def format_messages(self, messages: list[dict]):
        """Render the FULL transcript before generation (no cold single turn)."""
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        if self._has_template():
            text = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        else:
            text = "".join(f"{m['role'].title()}: {m['content']}\n" for m in msgs)
            text += "Assistant:"
            ids = self.tok(text, return_tensors="pt").input_ids
        if ids.numel() == 0:
            raise ValueError("conversation tokenises to nothing")
        return ids

    def format_prompt(self, prompt: str):
        """Compatibility path: a lone prompt is one user message."""
        return self.format_messages([{"role": "user", "content": prompt}])

    def format_injection(self, text: str):
        """Render the insertion as the TAIL of a new user turn, never as bare text.

        Appending this to the live context closes the in-progress assistant
        message, adds exactly one user turn and reopens the assistant role.
        """
        if not self._has_template():
            return self.tok("\nUser: " + text + "\nAssistant:",
                            return_tensors="pt").input_ids
        delta = self._injection_delta(text)
        if delta:
            return self.tok(delta, return_tensors="pt",
                            add_special_tokens=False).input_ids
        # The template did not expose a clean delta. A full single-turn render
        # is still a real user turn (it just repeats the preamble) and is
        # measurably better than appending bare text.
        turn = self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        return self.tok(turn, return_tensors="pt", add_special_tokens=False).input_ids

    def apply_steering(self, state: SteerState, vectors: dict) -> None:
        if state.layer is None or abs(state.coef) < 1e-6 or state.layer not in vectors:
            self.set_steer(None, None, 0.0)
        else:
            self.set_steer(state.layer, vectors[state.layer], state.coef, site="pre")
        # Neuron clamps are per connection too: rebuild them every token so a
        # different connection can never inherit a stale clamp.
        self.clear_neuron_steer(None)
        for (layer, neuron), value in state.neurons.items():
            self.set_neuron_steer(layer, neuron, value)

    @torch.no_grad()
    def stream(self, messages: list[dict] | str, max_new: int = 48,
               lens_layers: list[int] | None = None,
               *, control: GenControl, apply=None, on_injected=None,
               on_prefill=None):
        """Yield one protocol-shaped trace per generated token.

        Same payload as `ModelAdapter.stream` (including `steered`, `coef`,
        `steer_layer`) plus `neuron_steer`, with two live additions:
        a per-step lens schedule and mid-generation context injection.
        """
        if isinstance(messages, str):       # backwards-compatible prompt path
            messages = [{"role": "user", "content": messages}]
        ids = self.format_messages(messages)
        # The first forward over the whole prompt is the slow part on CPU
        # (measured 8 s for 60 tokens at 2B/fp32). Tell the UI before we block.
        if on_prefill is not None:
            on_prefill(int(ids.shape[1]))
        past = None
        pending = None      # logits already produced by an injection forward
        t0 = time.time()
        for local in range(max_new):
            if apply is not None:
                apply()                     # this connection's steering, right now
            if pending is not None:
                logits = pending
                pending = None
            else:
                self._capturing = True
                self._capture_full = False
                try:
                    out = self.model(
                        input_ids=ids if past is None else ids[:, -1:],
                        past_key_values=past, use_cache=True, logits_to_keep=1,
                    )
                finally:
                    self._capturing = False
                past = out.past_key_values
                # HF's forward ALREADY divides by config.logits_scaling (Granite).
                # Dividing here would double-apply it and change the argmax.
                logits = out.logits[:, -1, :].float()
            nxt = int(logits.argmax(-1))
            if nxt == self.tok.eos_token_id:
                # H3-6: never stream the end-of-text token; report reason=eos.
                control.eos = True
                return
            vals, idxs = logits.softmax(-1).topk(5, -1)
            if lens_layers is not None:
                want = list(lens_layers)
            elif local == 0:
                want = list(range(len(self.layers)))          # first token: all
            else:
                want = [l for l in range(len(self.layers)) if l % LENS_EVERY == 0]
            lens_map = self.logit_lens_layers("post", 3, only=want)
            self._last_lens = lens_map
            # real coordinates for the 3-D view (shared PCA, J1/J2 spec)
            ls = sorted(self._post)
            xy, var_expl = self.project_trajectory(ls)
            st = control.steer
            step = control.step
            control.step += 1
            yield {
                "step": step,
                "token": self.tok.decode([nxt]),
                "token_id": nxt,
                "ms": round((time.time() - t0) * 1000),
                "steered": st.steered(),
                "coef": round(float(st.coef), 4),
                "steer_layer": st.layer if st.steered() else None,
                "neuron_steer": st.neuron_payload(),
                "topk": [[self.tok.decode([int(i)]).replace("\n", "\\n"),
                          round(float(p), 4)]
                         for p, i in zip(vals[0].tolist(), idxs[0].tolist())],
                "pca": {"var_explained": var_expl, "n_layers": len(ls)},
                "layers": [
                    {"l": l,
                     "norm": round(float(self._post[l].norm()), 2),
                     "xy": xy[n],
                     "lens": [[tok, round(float(p), 4)]
                              for _id, tok, p in lens_map.get(l, [])],
                     "neurons": self._mlp.get(l, [])[:3]}
                    for n, l in enumerate(ls)
                ],
            }
            if control.cancel.is_set():
                return
            if control.restart is not None:
                return
            ids = torch.tensor([[nxt]])
            # Mid-generation context injection.
            #
            # The generated token has NOT been fed to the cache yet, so the first
            # injection forward consumes [nxt] + injected_tokens. Its logits are
            # the logits for the token AFTER the injection, so we keep them in
            # `pending` and use them next iteration.
            #
            # The previous implementation then set ids = ext[:, -1:], which fed
            # the last injected token a SECOND time (it was already in `past`),
            # duplicating a token and corrupting every subsequent step.
            text = control.pop_injection()
            first = True
            while text is not None:
                ext = self.format_injection(text)
                if ext.numel() == 0:
                    text = control.pop_injection()
                    continue
                feed = (torch.cat([torch.tensor([[nxt]]), ext], dim=1)
                        if first else ext)
                if apply is not None:
                    apply()
                self._capturing = True
                try:
                    out2 = self.model(input_ids=feed, past_key_values=past,
                                      use_cache=True)
                finally:
                    self._capturing = False
                past = out2.past_key_values
                pending = out2.logits[:, -1, :].float()
                ids = None
                first = False
                if on_injected is not None:
                    on_injected(control.step, int(ext.shape[1]), text)
                text = control.pop_injection()

    def detail_layers(self) -> list[dict]:
        """Full top-32 neuron payload for the token most recently yielded."""
        lens_map = getattr(self, "_last_lens", {})
        return [
            {"l": l,
             "norm": round(float(self._post[l].norm()), 2),
             "lens": [[tok, round(float(p), 4)]
                      for _id, tok, p in lens_map.get(l, [])],
             "neurons": [list(p) for p in self.top_neurons(l, 32)]}
            for l in sorted(self._post)
        ]


class Engine:
    """Thin wrapper: the model lives in ModelAdapter, never here."""

    def __init__(self, path: str, dtype: torch.dtype):
        # load() calls guard.require() and refuses BF16 -> float16.
        self.adapter = LiveAdapter(path).load(dtype=dtype)
        # H3-3: default contrast pairs, all token positions, same suffix.
        self.vectors = self.adapter.build_steer_vectors()

    @property
    def n_layers(self) -> int:
        return len(self.adapter.layers)

    @property
    def tok(self):
        return self.adapter.tok

    @property
    def model(self):
        return self.adapter.model

    @property
    def cfg(self):
        return self.adapter.cfg

    @property
    def intermediate_size(self) -> int:
        return int(getattr(self.adapter.cfg, "intermediate_size", 0))

    def stream(self, messages, max_new: int, control: GenControl,
               on_injected=None, on_prefill=None):
        return self.adapter.stream(
            messages, max_new, control=control,
            apply=lambda: self.adapter.apply_steering(control.steer, self.vectors),
            on_injected=on_injected, on_prefill=on_prefill,
        )

    def detail(self) -> list[dict]:
        return self.adapter.detail_layers()


# -- process state -----------------------------------------------------------
ENGINE: Engine | None = None
GEN_LOCK = GenLock()
GEN_TIMEOUT_S = 180.0            # overridden by --gen-timeout / MOD_WEI_GEN_TIMEOUT
GEN_HARD_GRACE_S = 10.0          # after timeout+cancel, force-release a stuck lock
GENERATIONS_STARTED = 0
GENERATIONS_DONE = 0
GENERATIONS_TIMED_OUT = 0
WATCHDOG_STOP = threading.Event()
SESSIONS = SessionStore()
CURRENT: dict = {"gen": None}
START_TIME = time.monotonic()


def _watchdog_tick(now: float | None = None) -> None:
    """One watchdog pass. Separated from the loop so tests can call it directly.

    * past GEN_TIMEOUT_S: flag the generation, cancel it at the next token
      boundary (so the lock is normally released by the worker's finally);
    * past timeout + GEN_HARD_GRACE_S: the thread is stuck inside a forward, so
      force-release the lock. The worker's release is idempotent afterwards.
    """
    global GENERATIONS_TIMED_OUT
    now = time.monotonic() if now is None else now
    started = GEN_LOCK.started_at
    if not GEN_LOCK.locked() or started is None:
        return
    elapsed = now - started
    if elapsed <= GEN_TIMEOUT_S:
        return

    gen = CURRENT.get("gen")
    if isinstance(gen, Generation) and not gen.timed_out:
        gen.timed_out = True
        gen.control.cancel.set()
        GENERATIONS_TIMED_OUT += 1
        print(f"[mw] WATCHDOG: generation ran {elapsed:.1f}s "
              f"(limit {GEN_TIMEOUT_S:.0f}s) — cancelling at the next token",
              flush=True)
        return

    if elapsed > GEN_TIMEOUT_S + GEN_HARD_GRACE_S and GEN_LOCK.force_release():
        sys.stderr.write(
            f"[mw] WATCHDOG: generation still stuck after {elapsed:.1f}s; "
            f"force-releasing the generation lock\n")
        sys.stderr.flush()
        if CURRENT.get("gen") is gen:
            CURRENT["gen"] = None


def _watchdog_loop() -> None:
    interval = max(0.01, min(1.0, GEN_TIMEOUT_S / 4))
    while not WATCHDOG_STOP.wait(interval):
        try:
            _watchdog_tick()
        except Exception as exc:  # noqa: BLE001 - a watchdog must never die
            sys.stderr.write(f"[mw] watchdog error: {type(exc).__name__}: {exc}\n")
            sys.stderr.flush()

app = FastAPI(title="mw live")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    # The legacy ui/index.html has none of the required states (H3-7).
    return RedirectResponse("http://localhost:3000", status_code=302)


@app.get("/api/info")
def info():
    e = ENGINE
    if e is None:
        return JSONResponse({"ready": False}, status_code=503)
    return JSONResponse({
        "ready": True,
        "model": e.adapter.path,
        "arch": type(e.model).__name__,
        "layers": e.n_layers,
        "d_model": e.cfg.hidden_size,
        "vocab": e.cfg.vocab_size,
        "scaling": e.adapter.logits_scaling,
        "dtype": e.adapter.dtype_actual,
        "steer_vectors": len(e.vectors),
    })


@app.get("/api/health")
def health():
    return JSONResponse({
        "ok": ENGINE is not None,
        "busy": GEN_LOCK.locked(),
        "busy_s": GEN_LOCK.busy_s(),
        "uptime_s": round(time.monotonic() - START_TIME, 1),
        # `generations` is kept as the completed count for older clients.
        "generations": GENERATIONS_DONE,
        "generations_started": GENERATIONS_STARTED,
        "generations_done": GENERATIONS_DONE,
        "generations_timed_out": GENERATIONS_TIMED_OUT,
        "timeout_s": GEN_TIMEOUT_S,
    })


@app.get("/api/export/{session_id}")
def export(session_id: str):
    rec = SESSIONS.get(session_id)
    if rec is None:
        return JSONResponse(
            {"error": f"unknown session {session_id}"}, status_code=404)
    return JSONResponse(rec)


@app.get("/api/detail/{session_id}/{step}")
def detail(session_id: str, step: int):
    rec = SESSIONS.get_detail(session_id, step)
    if rec is None:
        return JSONResponse(
            {"error": f"unknown session/step {session_id}/{step}"}, status_code=404)
    return JSONResponse(rec)


# -- websocket ---------------------------------------------------------------
def _cancel_current() -> bool:
    gen = CURRENT.get("gen")
    if isinstance(gen, Generation):
        gen.control.cancel.set()
        return True
    return False


def _worker(gen: Generation, send) -> None:
    """Runs in a thread. Never dies silently; always releases the lock."""
    global GENERATIONS_STARTED, GENERATIONS_DONE
    acquired = False
    reason = "max_new"
    n_tokens = 0
    sid = ""
    failed = False
    try:
        acquired = GEN_LOCK.acquire(blocking=False)
        if not acquired:
            send({"type": "error", "message": "server busy"})
            return
        GENERATIONS_STARTED += 1
        sid = SESSIONS.create(prompt=gen.prompt, messages=gen.messages,
                              max_new=gen.max_new,
                              layer=gen.control.steer.layer,
                              coef=gen.control.steer.coef)
        gen.session = sid
        CURRENT["gen"] = gen

        def note_injected(at: int, nt: int, text: str) -> None:
            gen.injected_at = at
            send({"type": "injected", "session": sid, "at_step": at,
                  "tokens": nt, "text": text})

        while True:
            gen.text = ""
            gen.control.eos = False
            for tr in ENGINE.stream(gen.messages, gen.max_new, gen.control,
                                    on_injected=note_injected,
                                    on_prefill=lambda n: send(
                                        {"type": "prefill", "tokens": n})):
                payload = fit_payload({"type": "token", "session": sid, **tr})
                detail = ENGINE.detail()
                if not send(payload):
                    gen.control.cancel.set()
                    reason = "cancelled"
                    break
                n_tokens += 1
                gen.text += payload["token"]
                SESSIONS.add_token(sid, payload, detail)

            if gen.control.cancel.is_set():
                reason = "cancelled"
                break
            if gen.control.restart is not None:
                gen.messages = gen.control.restart
                gen.control.restart = None
                send({"type": "restarted", "session": sid, "prompt": gen.prompt})
                continue
            reason = "eos" if gen.control.eos else "max_new"
            # Seal this turn BEFORE checking for a late injection. An inject
            # that loses this race is refused by GenControl.inject() and starts
            # a fresh turn in the socket handler instead of being silently
            # dropped between the final pop and `done`.
            gen.control.close()
            text = gen.control.pop_injection()
            if text is None:
                break
            # An injection that arrived as generation ended starts a new turn.
            gen.control.reopen()
            nt = int(ENGINE.adapter.format_injection(text).shape[1])
            note_injected(gen.control.step, nt, text)
            turn = list(gen.messages)
            if gen.text:
                turn.append({"role": "assistant", "content": gen.text})
            turn.append({"role": "user", "content": text})
            gen.messages = turn
            continue
    except Exception as exc:  # noqa: BLE001 - report, never die silently
        failed = True
        sys.stderr.write(f"[mw] generation failed: {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        reason = "error"
    finally:
        # Top-level finally: counters and lock are correct even if the adapter
        # raised or the watchdog force-released the lock underneath us.
        if sid:
            SESSIONS.finish(sid, reason, n_tokens)
        if acquired:
            GEN_LOCK.release()
            GENERATIONS_DONE += 1
        if CURRENT.get("gen") is gen:
            CURRENT["gen"] = None
        if sid and not failed:
            send({"type": "done", "session": sid, "reason": reason,
                  "n_tokens": n_tokens, "injected_at": gen.injected_at,
                  "timed_out": gen.timed_out})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    if ENGINE is None:
        await sock.send_json({"type": "error", "message": "model not ready"})
        await sock.close()
        return

    async def send_async(payload: dict) -> bool:
        try:
            await sock.send_json(payload)
            return True
        except Exception:  # noqa: BLE001 - client may have gone away
            return False

    def send(payload: dict) -> bool:
        """Thread -> loop send. Never raises, never blocks forever."""
        try:
            fut = asyncio.run_coroutine_threadsafe(send_async(payload), loop)
            return bool(fut.result(timeout=SEND_TIMEOUT_S))
        except Exception:  # noqa: BLE001 - socket closed or loop gone
            return False

    steer = SteerState()
    mine: Generation | None = None
    last_gen: Generation | None = None

    def start(messages: list[dict], max_new: int) -> Generation:
        gen = Generation(messages, max_new, GenControl(steer))
        threading.Thread(target=_worker, args=(gen, send), name="modwei-gen",
                         daemon=True).start()
        return gen

    try:
        while True:
            try:
                raw = await sock.receive_json()
            except (ValueError, json.JSONDecodeError):
                await send_async({"type": "error", "message": "invalid JSON message"})
                continue
            if not isinstance(raw, dict):
                await send_async({"type": "error", "message": "message must be a JSON object"})
                continue

            kind = raw.get("type")

            if kind == "chat":
                try:
                    chat = ChatIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                if chat.coef != 0.0 and chat.layer is None:
                    await send_async({"type": "error",
                                      "message": "invalid request: layer required when coef != 0"})
                    continue
                if chat.layer is not None:
                    if chat.layer >= ENGINE.n_layers:
                        await send_async({"type": "error",
                                          "message": f"invalid request: layer {chat.layer} out of "
                                                     f"range 0..{ENGINE.n_layers - 1}"})
                        continue
                    if chat.coef != 0.0 and chat.layer not in ENGINE.vectors:
                        await send_async({"type": "error",
                                          "message": f"invalid request: no steering vector for "
                                                     f"layer {chat.layer}"})
                        continue
                if GEN_LOCK.locked():
                    await send_async({"type": "error", "message": "server busy"})
                    continue
                transcript = (
                    [m.model_dump() for m in chat.messages]
                    if chat.messages
                    else [{"role": "user", "content": chat.prompt}])
                steer.set_base(chat.layer, chat.coef)
                mine = start(transcript, chat.max_new)
                last_gen = mine

            elif kind == "steer":
                try:
                    st = SteerIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                if st.layer >= ENGINE.n_layers:
                    await send_async({"type": "error",
                                      "message": f"invalid request: layer {st.layer} out of range "
                                                 f"0..{ENGINE.n_layers - 1}"})
                    continue
                if st.coef != 0.0 and st.layer not in ENGINE.vectors:
                    await send_async({"type": "error",
                                      "message": f"invalid request: no steering vector for "
                                                 f"layer {st.layer}"})
                    continue
                # Per connection only: applied before this connection's next
                # forward pass, never to another client's stream.
                steer.set_base(st.layer, st.coef)
                await send_async({"type": "steer_ack", "layer": st.layer, "coef": st.coef})

            elif kind == "neuron_steer":
                try:
                    ns = NeuronSteerIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                if ns.layer >= ENGINE.n_layers:
                    await send_async({"type": "error",
                                      "message": f"invalid request: layer {ns.layer} out of range "
                                                 f"0..{ENGINE.n_layers - 1}"})
                    continue
                if ns.neuron >= ENGINE.intermediate_size:
                    await send_async({"type": "error",
                                      "message": f"invalid request: neuron {ns.neuron} out of range "
                                                 f"0..{ENGINE.intermediate_size - 1}"})
                    continue
                steer.neurons[(ns.layer, ns.neuron)] = ns.value
                await send_async({"type": "steer_ack", "layer": ns.layer,
                                  "neuron": ns.neuron, "value": ns.value})

            elif kind == "neuron_clear":
                try:
                    nc = NeuronClearIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                if nc.layer is None:
                    steer.neurons.clear()
                else:
                    if nc.layer >= ENGINE.n_layers:
                        await send_async({"type": "error",
                                          "message": f"invalid request: layer {nc.layer} out of "
                                                     f"range 0..{ENGINE.n_layers - 1}"})
                        continue
                    for key in [k for k in steer.neurons if k[0] == nc.layer]:
                        del steer.neurons[key]
                await send_async({"type": "steer_ack", "layer": nc.layer})

            elif kind == "cancel":
                try:
                    CancelIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                active = CURRENT.get("gen")
                if active is None:
                    await send_async({"type": "error", "message": "nothing to cancel"})
                elif active is mine:
                    active.control.cancel.set()
                else:
                    await send_async({"type": "error", "message": "not your generation"})

            elif kind == "inject":
                try:
                    inj = InjectIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                active = CURRENT.get("gen")
                if (isinstance(active, Generation) and active is mine
                        and mine.control.inject(inj.text)):
                    # Queued; the stream loop applies it at the next token
                    # boundary. inject() is False once the turn has closed.
                    continue
                # The turn has ended (or is ending): the inject starts a new
                # turn inside the same conversation. A worker finishing its
                # last token may still hold the lock for a few milliseconds.
                for _ in range(100):
                    if not GEN_LOCK.locked():
                        break
                    await asyncio.sleep(0.01)
                if GEN_LOCK.locked():
                    await send_async({"type": "error", "message": "server busy"})
                    continue
                nt = int(ENGINE.adapter.format_injection(inj.text).shape[1])
                prior = list(last_gen.messages) if last_gen else []
                if last_gen is not None and last_gen.text:
                    prior.append({"role": "assistant", "content": last_gen.text})
                prior.append({"role": "user", "content": inj.text})
                await send_async({"type": "injected", "session":
                                  last_gen.session if last_gen else None,
                                  "at_step": 0, "tokens": nt, "text": inj.text})
                mine = start(prior, 40)
                mine.injected_at = 0        # this turn was seeded by an injection
                last_gen = mine

            elif kind == "edit_prompt":
                try:
                    ep = EditPromptIn.model_validate(raw)
                except ValidationError as exc:
                    await send_async({"type": "error",
                                      "message": describe_validation_error(exc)})
                    continue
                restart = [{"role": "user", "content": ep.prompt}]
                active = CURRENT.get("gen")
                if isinstance(active, Generation) and active is mine:
                    mine.control.restart = restart          # worker replies restarted
                else:
                    await send_async({"type": "restarted", "prompt": ep.prompt})
                    mine = start(restart, 40)
                    last_gen = mine

            else:
                await send_async({"type": "error",
                                  "message": f"unknown message type: {kind!r}"})

    except WebSocketDisconnect:
        pass
    finally:
        if mine is not None and CURRENT.get("gen") is mine:
            mine.control.cancel.set()   # client left mid-stream: stop, free the lock


# -- entry point -------------------------------------------------------------
class _Server(uvicorn.Server):
    """uvicorn server that cancels generation before it stops."""

    def handle_exit(self, sig, frame):  # type: ignore[override]
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = str(sig)
        print(f"[mw] {name}: cancelling generation, shutting down", flush=True)
        _cancel_current()
        super().handle_exit(sig, frame)


def main() -> int:
    global ENGINE, START_TIME, GEN_TIMEOUT_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / ".hf" / "bf16-hf"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gen-timeout", type=float,
                    default=float(os.environ.get("MOD_WEI_GEN_TIMEOUT", "180")),
                    help="abort any generation running longer than this many "
                         "seconds (default 180, env MOD_WEI_GEN_TIMEOUT)")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="send raw prompts/injections instead of rendering them "
                         "through the model's chat template")
    args = ap.parse_args()
    GEN_TIMEOUT_S = max(0.05, float(args.gen_timeout))

    torch.set_num_threads(4)      # H3-12: not at import time
    START_TIME = time.monotonic()
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[mw] loading {args.model} ({args.dtype})", flush=True)
    try:
        ENGINE = Engine(args.model, dt)
        ENGINE.adapter.use_chat_template = not args.no_chat_template
        print(f"[mw] chat template: {ENGINE.adapter._has_template()}", flush=True)
    except MemoryError:
        sys.stderr.write(
            "[mw] load refused by the memory guard — nothing was loaded, exiting.\n"
            "          free memory, pick a smaller dtype, or set MOD_WEI_ALLOW_OOM=1.\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"[mw] refusing to load: {exc}\n")
        return 2
    print(f"[mw] {len(ENGINE.vectors)} steering vectors ready", flush=True)
    print(f"[mw] generation watchdog: {GEN_TIMEOUT_S:.0f}s "
          f"(hard grace {GEN_HARD_GRACE_S:.0f}s)", flush=True)
    print(f"[mw] http://{args.host}:{args.port}", flush=True)

    WATCHDOG_STOP.clear()
    threading.Thread(target=_watchdog_loop, name="modwei-watchdog",
                     daemon=True).start()

    server = _Server(uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning",
        timeout_graceful_shutdown=10,
    ))

    # uvicorn's capture_signals() restores these handlers and re-raises the
    # captured signal *after* the graceful drain. Without our own handler that
    # re-raise would kill the process with the signal, so install a no-op exit
    # handler: shutdown already happened, we just want a clean rc=0.
    def _after_drain(signum, _frame):
        print(f"[mw] {signal.Signals(signum).name} drained", flush=True)
        _cancel_current()
        server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _after_drain)

    try:
        server.run()
    finally:
        WATCHDOG_STOP.set()
        # A cancelled generation stops after its current token; give it a moment.
        if GEN_LOCK.locked():
            with contextlib.suppress(Exception):
                if GEN_LOCK.acquire(timeout=10):
                    GEN_LOCK.release()
    print("[mw] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
