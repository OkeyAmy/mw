"""Steering, neuron-clamp and per-connection isolation — tiny random model.

Fast: builds a ~0.2 MB `Qwen2ForCausalLM` from config with a byte-level stub
tokenizer. No download, no real weights, no network.

Every assertion is about *measured* numbers: a test that only checks "a dict has
a key" is theatre, so the non-zero-steering tests carry a vacuity guard
(`assert |vector| > 0`) — a zero-vector regression must fail, not pass silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from app.server import GenControl, LiveAdapter, SteerState  # noqa: E402
from mw.adapter import ModelAdapter  # noqa: E402

VOCAB = 512
EOS = 0


class _Ids:
    def __init__(self, ids: torch.Tensor):
        self.input_ids = ids


class _StubTok:
    """Deterministic byte tokenizer: same text -> same ids, every run."""

    chat_template = None
    eos_token_id = EOS
    eos_token = "<eos>"

    def __call__(self, text: str, return_tensors=None, add_special_tokens=True):
        raw = text.encode() or b" "
        ids = [1 + (b % (VOCAB - 1)) for b in raw][:48] or [1]
        return _Ids(torch.tensor([ids], dtype=torch.long))

    def decode(self, ids):
        return "".join(chr(0x20 + (int(i) % 90)) for i in ids)


def _tiny() -> ModelAdapter:
    """Assemble a ModelAdapter on a random tiny model (no disk, no guard)."""
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=True,
    )
    ad = LiveAdapter(path="<tiny-random-qwen2>")
    ad.tok = _StubTok()
    ad.model = Qwen2ForCausalLM(cfg).eval()
    ad.cfg = ad.model.config
    ad.dtype_actual = "torch.float32"
    ad._install_hooks()
    return ad


@torch.no_grad()
def _logits(ad: ModelAdapter, ids: torch.Tensor) -> torch.Tensor:
    ad._capturing = False
    return ad.model(input_ids=ids, use_cache=False).logits[:, -1, :].float()


@pytest.fixture(scope="module")
def ad() -> ModelAdapter:
    return _tiny()


@pytest.fixture(scope="module")
def ids(ad) -> torch.Tensor:
    return ad.tok("def add(a, b):\n    return a + b\n\n# sum:\n",
                  return_tensors="pt").input_ids


@pytest.fixture(scope="module")
def vecs(ad) -> dict:
    return ad.build_steer_vectors()


# -- 1. the vectors themselves ----------------------------------------------
def test_vectors_are_nonzero_every_layer(vecs):
    assert vecs, "build_steer_vectors returned nothing"
    wrong = {l: round(float(v.norm()), 6) for l, v in vecs.items()
             if float(v.norm()) <= 0.0}
    assert not wrong, f"zero-norm steering vector(s): {wrong}"


def test_vector_is_not_a_last_token_embedding_delta(ad, vecs):
    """H3-3 regression: the estimator averages all positions, so at layer 0 the
    vector must NOT be collinear with the last-token embedding difference."""
    emb = ad.model.model.embed_tokens.weight.detach().float()
    from mw.adapter import DEFAULT_NEG, DEFAULT_POS
    pos = torch.stack([ad.tok(t).input_ids[0, -1] for t in DEFAULT_POS])
    neg = torch.stack([ad.tok(t).input_ids[0, -1] for t in DEFAULT_NEG])
    ident = emb[pos].mean(0) - emb[neg].mean(0)
    cos = float(torch.nn.functional.cosine_similarity(vecs[0], ident, dim=0))
    assert abs(cos) < 0.99, f"vector[0] is a token-identity direction (cos={cos:.4f})"


# -- 2. coef = 0 is a true no-op --------------------------------------------
def test_coef_zero_is_bit_identical(ad, ids, vecs):
    base = _logits(ad, ids)
    for layer, coef in ((3, 0.0), (0, 0.0), (None, 0.0)):
        ad.set_steer(layer, vecs.get(layer), coef)
        assert torch.equal(base, _logits(ad, ids)), (
            f"coef=0 changed the logits (layer={layer})")
    ad.set_steer(None, None, 0.0)


# -- 3. coef != 0 must change the next-token distribution -------------------
@pytest.mark.parametrize("coef", (1.0, 2.0, -1.0))
def test_nonzero_coef_changes_logits(ad, ids, vecs, coef):
    assert float(vecs[2].norm()) > 0.0, "vacuous: zero vector cannot steer"
    ad.set_steer(None, None, 0.0)
    base = _logits(ad, ids)
    ad.set_steer(2, vecs[2], coef)
    steered = _logits(ad, ids)
    ad.set_steer(None, None, 0.0)
    assert not torch.equal(base, steered), f"coef={coef} was a silent no-op"
    assert float((steered - base).abs().max()) > 1e-6


# -- 4. neuron clamp -------------------------------------------------------
def test_neuron_clamp_changes_logits(ad, ids):
    ad._capturing = True
    ad._capture_full = False
    ad.model(input_ids=ids, use_cache=False)
    ad._capturing = False
    top = ad.top_neurons(2, 1)
    assert top, "no top-neuron capture; the MLP hook is not recording"
    neuron, natural = top[0]
    base = _logits(ad, ids)
    for value in (0.0, 50.0):
        ad.clear_neuron_steer(None)
        ad.set_neuron_steer(2, neuron, value)
        clamped = _logits(ad, ids)
        ad.clear_neuron_steer(None)
        assert not torch.equal(base, clamped), (
            f"clamping n{neuron} (natural {natural}) to {value} did nothing")
    ad.clear_neuron_steer(None)
    assert torch.equal(base, _logits(ad, ids)), "clear_neuron_steer left a residue"


# -- 5. provenance ---------------------------------------------------------
def _stream(ad, state, n=3):
    control = GenControl(state)
    vectors = ad.build_steer_vectors()
    return list(ad.stream([{"role": "user", "content": "def f(x):"}], n,
                          control=control,
                          apply=lambda: ad.apply_steering(control.steer, vectors)))


def test_provenance_fields_are_per_token(ad):
    off = _stream(ad, SteerState())
    assert len(off) >= 2
    for frame in off:
        assert frame["steered"] is False
        assert frame["coef"] == 0.0
        assert frame["steer_layer"] is None
        assert frame["neuron_steer"] is None

    state = SteerState()
    state.set_base(2, 1.5)
    on = _stream(ad, state)
    for frame in on:
        assert frame["steered"] is True
        assert frame["coef"] == pytest.approx(1.5)
        assert frame["steer_layer"] == 2

    nstate = SteerState()
    nstate.neurons[(2, 7)] = 0.0
    nframes = _stream(ad, nstate)
    assert nframes[0]["steered"] is True
    assert nframes[0]["neuron_steer"] == {"2": {"7": 0.0}}
    assert nframes[0]["steer_layer"] is None


def test_steering_changes_the_next_token_distribution(ad):
    """End-to-end through the live stream path: identical prompt, coef=0 vs
    coef=2 must change the per-step next-token distribution.

    A random untrained model can settle on the same argmax at every step, so the
    oracle is the distribution the frame carries (`topk`), not the sampled id —
    on real weights the argmax usually moves too (measured live: see
    agents/S1_steering.md).
    """
    off = _stream(ad, SteerState(), n=4)
    state = SteerState()
    state.set_base(2, 2.0)
    on = _stream(ad, state, n=4)
    assert len(off) == len(on) >= 1
    diff = [abs(a["topk"][0][1] - b["topk"][0][1]) for a, b in zip(off, on)]
    assert max(diff) > 1e-4, f"steering left the distribution untouched: {off[0]['topk']}"


# -- 6. per-connection isolation -------------------------------------------
def test_two_connections_do_not_share_steering(ad):
    a, b = SteerState(), SteerState()
    a.set_base(2, 2.0)
    b.set_base(0, 0.0)                       # connection B is explicitly off
    vectors = ad.build_steer_vectors()

    ad.apply_steering(a, vectors)
    assert ad._steer["layer"] == 2 and ad._steer["coef"] == 2.0
    assert a.steered() and not b.steered()

    # B's forward must not inherit A's vector.
    ad.apply_steering(b, vectors)
    assert ad._steer["layer"] is None and ad._steer["coef"] == 0.0
    assert ad._steer["vec"] is None

    # Neuron clamps are rebuilt per connection, never inherited either.
    a.neurons[(2, 7)] = 0.0
    ad.apply_steering(a, vectors)
    assert ad._neuron_steer == {2: {7: 0.0}}
    ad.apply_steering(b, vectors)
    assert ad._neuron_steer == {}, "connection B inherited A's neuron clamp"


def test_streams_with_different_states_diverge(ad):
    """The owning state is applied before every forward: two states cannot both
    be the one in effect for one token."""
    a, b = SteerState(), SteerState()
    a.set_base(2, 2.0)
    fa = _stream(ad, a, n=3)
    fb = _stream(ad, b, n=3)
    assert {f["steered"] for f in fa} == {True}
    assert {f["steered"] for f in fb} == {False}
    assert any(x["topk"] != y["topk"] for x, y in zip(fa, fb)), (
        "two connections produced identical distributions — the state leaked")


# -- 7. the server's apply_steering guard ----------------------------------
def test_apply_steering_treats_negligible_coef_as_off(ad, ids, vecs):
    base = _logits(ad, ids)
    state = SteerState()
    state.set_base(2, 1e-9)                  # below apply_steering's 1e-6 guard
    ad.apply_steering(state, vecs)
    assert torch.equal(base, _logits(ad, ids))
    assert state.steered() is True           # provenance still reports intent
