"""Tests for `mw.certify` — the quantisation decision-flip certificate.

Written by audit agent C2 (2026-11-12). Three kinds of test, deliberately kept
apart so the suite is honest about what it proves:

1. **Guards that pass today** — properties the tool must never lose.
2. **`xfail` tests that state the property the tool SHOULD have.** They fail now
   because the defect exists; each one names the defect it is waiting on. When
   the orchestrator fixes `certify.py` they turn into XPASS. They are non-strict
   so the current suite stays green while the report documents the gaps.
3. **Mechanism proofs** — pure-numpy demonstrations that the failure mode is
   real, independent of whether today's weights happen to trigger it.

Everything here is fast: a 96-token, 2-layer Qwen2 built on the fly, no download.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from mw import certify
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

ROOT = Path(__file__).resolve().parent.parent

# A char-level vocabulary: every prompt tokenises to >= 1 token, and the vocab
# is > certify.log_probs()'s hard-coded topk=20. The repo's 12-token smoke
# fixture crashes certify with "selected index k out of range" (C2 defect D8).
CHARS = ["[UNK]"] + [chr(c) for c in range(32, 127)]


def _tokenizer(order: list[str]) -> PreTrainedTokenizerFast:
    tk = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(order)}, unk_token="[UNK]")
    )
    tk.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[UNK]", eos_token="[UNK]"
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> dict[str, Path]:
    """a == b (identical weights), mut (noised lm_head), tok (same weights, remapped ids)."""
    root = tmp_path_factory.mktemp("cert_tiny")
    tok = _tokenizer(CHARS)
    cfg = Qwen2Config(
        vocab_size=len(CHARS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).eval()
    sd = {k: v.clone() for k, v in model.state_dict().items()}

    def save(name: str, state: dict, tokenizer=tok) -> Path:
        d = root / name
        d.mkdir()
        m = Qwen2ForCausalLM(cfg).eval()
        m.load_state_dict(state)
        m.save_pretrained(d)
        tokenizer.save_pretrained(d)
        return d

    paths = {"a": save("a", sd), "b": save("b", sd)}
    g = torch.Generator().manual_seed(7)
    noisy = {
        k: (v + 0.6 * torch.randn(v.shape, generator=g)) if "lm_head" in k else v.clone()
        for k, v in sd.items()
    }
    paths["mut"] = save("mut", noisy)
    paths["tok"] = save("tok", sd, _tokenizer(CHARS[::-1]))
    return paths


def _main(monkeypatch, ref, cand, out, prompts=None, builtin=None, threshold=0.05) -> int:
    argv = ["certify", "--reference", str(ref), "--candidate", str(cand),
            "--out", str(out), "--threshold", str(threshold)]
    if prompts is not None:
        argv += ["--prompts", str(prompts)]
    if builtin is not None:
        argv += ["--builtin", str(builtin)]
    monkeypatch.setattr(sys, "argv", argv)
    return certify.main()


# --------------------------------------------------------------------------
# 1. guards that pass today
# --------------------------------------------------------------------------
def test_identical_models_zero_flips_and_exit_zero(tiny, tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "p.txt"
    prompts.write_text("\n".join(certify.build_prompts(8)) + "\n")
    code = _main(monkeypatch, tiny["a"], tiny["b"], tmp_path / "o", prompts)
    capsys.readouterr()
    assert code == 0
    res = json.loads((tmp_path / "o" / "report.json").read_text())
    assert res["flips"] == 0 and res["flip_rate"] == 0.0
    assert res["mean_kl"] == 0.0 and res["max_kl"] == 0.0


def test_mutated_weights_detected_and_report_written_before_failure(
    tiny, tmp_path, monkeypatch, capsys
):
    """A wrong/noisy lm_head must flip decisions, and the report must exist even
    when the tool exits 1 (a CI gate that fails without an artefact is useless)."""
    prompts = tmp_path / "p.txt"
    prompts.write_text("\n".join(certify.build_prompts(8)) + "\n")
    code = _main(monkeypatch, tiny["a"], tiny["mut"], tmp_path / "o", prompts, threshold=0.0)
    capsys.readouterr()
    assert code == 1
    res = json.loads((tmp_path / "o" / "report.json").read_text())
    assert res["flips"] > 0
    assert (tmp_path / "o" / "report.md").exists()


def test_exit_code_is_wired_to_the_process(tiny, tmp_path):
    """`main()` returns 0/1; `python -m mw.certify` must propagate it."""
    prompts = tmp_path / "p.txt"
    prompts.write_text("\n".join(certify.build_prompts(4)) + "\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "code"))
    out = subprocess.run(
        [sys.executable, "-m", "mw.certify", "--reference", str(tiny["a"]),
         "--candidate", str(tiny["b"]), "--prompts", str(prompts),
         "--out", str(tmp_path / "o")],
        check=False, capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_measure_is_deterministic(tiny, tmp_path):
    """No seed, no sampling — two passes over the same weights must be bit-identical.
    `certify` never checks this itself (C2 defect D7); this test is that check."""
    prompts = certify.build_prompts(6)
    a1, _ = certify.measure(str(tiny["a"]), prompts, "float32", tmp_path / "c1.npz")
    a2, _ = certify.measure(str(tiny["a"]), prompts, "float32", tmp_path / "c2.npz")
    assert np.array_equal(a1, a2)


def test_kl_is_non_negative_and_matches_the_definition():
    torch.manual_seed(0)
    for _ in range(20):
        la = torch.log_softmax(torch.randn(5000), -1)
        lb = torch.log_softmax(torch.randn(5000), -1)
        kl = float((la.exp() * (la - lb)).sum())
        p, q = la.exp(), lb.exp()
        reference = float((p * (p.log() - q.log())).sum())
        assert kl >= 0.0
        assert kl == pytest.approx(reference, rel=1e-6, abs=1e-8)
        assert float((la.exp() * (la - la)).sum()) == pytest.approx(0.0, abs=1e-7)


def test_kl_survives_the_fp16_cache():
    """`exp(logp)` happens after a float32 cast, so fp16 log-probs of -20 do not
    underflow to 0 in the KL sum (checked, not a defect)."""
    torch.manual_seed(1)
    la = torch.log_softmax(torch.randn(20000) * 3, -1)
    lb = torch.log_softmax(torch.randn(20000) * 3, -1)
    k32 = float((la.exp() * (la - lb)).sum())
    k16 = float((la.half().float().exp() * (la.half().float() - lb.half().float())).sum())
    assert abs(k32 - k16) < 1e-3
    assert np.isfinite(np.exp(np.float32(np.float16(-20.0))))
    assert float(np.exp(np.float16(-20.0))) == 0.0  # would underflow if exp ran in fp16


def test_empty_prompt_file_never_produces_a_pass_report(tiny, tmp_path, monkeypatch, capsys):
    """An empty prompt list must not be certified. Today it dies with a numpy
    traceback and exit 1 — loud, but indistinguishable from 'threshold failed'."""
    prompts = tmp_path / "empty.txt"
    prompts.write_text("\n\n   \n")
    with pytest.raises((ValueError, SystemExit, RuntimeError)):
        _main(monkeypatch, tiny["a"], tiny["b"], tmp_path / "o", prompts)
    capsys.readouterr()
    assert not (tmp_path / "o" / "report.json").exists()


def test_build_prompts_repeats_beyond_the_base_set():
    """Documents the independence defect: the default --builtin 120 is 41
    distinct prompts measured ~3x, not 120 independent samples."""
    assert len(certify.build_prompts(41)) == 41
    p = certify.build_prompts(120)
    assert len(p) == 120 and len(set(p)) == 41
    assert max(p.count(x) for x in set(p)) == 3


# --------------------------------------------------------------------------
# 2. mechanism proofs (pure numerics, independent of today's weights)
# --------------------------------------------------------------------------
def test_float16_rounding_collapses_a_near_tie_to_the_lower_token_id():
    """float16 rounding is monotone, so it can never *reorder* two values — but it
    can make them equal, and then np.argmax silently picks the lower token id.
    Anyone reading the certificate sees a decision the model never made."""
    vocab = 151936
    logits = torch.full((vocab,), -40.0)
    logits[120000] = 0.0            # the model's real choice
    logits[5] = -1e-5               # 1e-5 nats behind, lower id
    lp = logits.log_softmax(-1)
    assert int(lp.argmax()) == 120000
    assert int(lp.half().float().argmax()) == 5
    ulp = float(np.spacing(np.float16(0.693)))
    assert float(lp[120000] - lp[5]) < ulp


def test_float16_cache_band_is_wide_enough_to_matter():
    """The band where the cache can lie is not academic: a 1e-3 nat margin at
    |logp| ~ 1 is inside one fp16 ULP."""
    for mag, ulp in ((0.5, 4.8828e-4), (1.0, 9.7656e-4), (2.0, 1.9531e-3), (4.0, 3.9063e-3)):
        assert abs(float(np.spacing(np.float16(mag))) - ulp) < 2e-6


# --------------------------------------------------------------------------
# 3. xfail: properties the tool should have (defects tracked in
#    agents/C2_certificate_code.md)
# --------------------------------------------------------------------------
@pytest.mark.xfail(reason="C2 D3: npz cache has no fingerprint of prompts/model/dtype/code",
                   strict=False)
def test_cache_is_invalidated_by_a_prompt_change(tiny, tmp_path, monkeypatch, capsys):
    out = tmp_path / "o"
    p1 = tmp_path / "p1.txt"
    p1.write_text("abc\n" * 1 + "xyz\n")
    _main(monkeypatch, tiny["a"], tiny["b"], out, p1)
    capsys.readouterr()
    p2 = tmp_path / "p2.txt"
    p2.write_text("THE OTHER PROMPT\nAND ANOTHER\n")
    _main(monkeypatch, tiny["a"], tiny["b"], out, p2)
    capsys.readouterr()
    res = json.loads((out / "report.json").read_text())
    assert [r["prompt"] for r in res["rows"]] == ["THE OTHER PROMPT", "AND ANOTHER"]


@pytest.mark.xfail(reason="C2 D3: a different model pair reuses the previous npz cache",
                   strict=False)
def test_cache_is_invalidated_by_a_model_change(tiny, tmp_path, monkeypatch, capsys):
    out = tmp_path / "o"
    prompts = tmp_path / "p.txt"
    prompts.write_text("\n".join(certify.build_prompts(4)))
    _main(monkeypatch, tiny["a"], tiny["b"], out, prompts)
    capsys.readouterr()
    first = (out / "ref.npz").stat().st_mtime_ns
    _main(monkeypatch, tiny["a"], tiny["mut"], out, prompts)
    capsys.readouterr()
    res = json.loads((out / "report.json").read_text())
    assert (out / "ref.npz").stat().st_mtime_ns != first
    assert res["flips"] > 0


@pytest.mark.xfail(reason="C2 D4: compare() zip()-truncates mismatched caches silently",
                   strict=False)
def test_compare_rejects_mismatched_row_counts():
    V = 64
    ref = np.full((4, V), -12.0, np.float16)
    cand = np.full((2, V), -12.0, np.float16)
    meta = [{"prompt": f"p{i}", "argmax_tok": "x", "margin": 2.0} for i in range(4)]
    with pytest.raises((ValueError, AssertionError)):
        certify.compare(ref, meta, cand, meta[:2])


@pytest.mark.xfail(reason="C2 D5: no tokenizer equality check between ref and cand",
                   strict=False)
def test_tokenizer_mismatch_is_refused(tiny, tmp_path):
    """Same weights, different id->string mapping, currently reported as 100%
    flips — a certificate that invents a regression."""
    prompts = certify.build_prompts(6)
    lp_a, meta_a = certify.measure(str(tiny["a"]), prompts, "float32", tmp_path / "a.npz")
    lp_t, meta_t = certify.measure(str(tiny["tok"]), prompts, "float32", tmp_path / "t.npz")
    with pytest.raises((ValueError, AssertionError)):
        certify.compare(lp_a, meta_a, lp_t, meta_t)


@pytest.mark.xfail(reason="C2 D1: argmax is taken from the float16 cache; on a near "
                          "tie it must equal the float32 argmax", strict=False)
def test_near_tie_argmax_uses_the_true_float32_decision(tmp_path):
    V = 151936
    for i in range(3):
        a = np.full((1, V), -30.0, np.float32)
        a[0, 120000] = -0.6931
        a[0, 5] = -0.69311                      # 1e-5 nats behind
        b = a.copy()
        cached_a = torch.from_numpy(a).half().numpy()
        cached_b = torch.from_numpy(b).half().numpy()
        meta = [{"prompt": "p", "argmax": 120000, "argmax_tok": "T", "margin": 1e-5}]
        res = certify.compare(cached_a, meta, cached_b, meta)
        assert res["rows"][0]["ref"] == 120000


@pytest.mark.xfail(reason="C2 D6: dtype argument is checked by guard.require but the "
                          "load is hard-coded to float32", strict=False)
def test_measure_loads_the_dtype_it_guarded(tiny, tmp_path, monkeypatch):
    seen = {}
    real = certify.AutoModelForCausalLM.from_pretrained

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(certify.AutoModelForCausalLM, "from_pretrained", spy)
    certify.measure(str(tiny["a"]), certify.build_prompts(2), "float16", tmp_path / "c.npz")
    assert seen.get("dtype") == torch.float16


@pytest.mark.xfail(reason="C2 D9: no uncertainty is reported; 0/41 flips cannot "
                          "support a 5% pass (rule of three: 3/41 = 7.3%)", strict=False)
def test_certificate_reports_the_upper_bound_not_just_the_point_estimate(tiny, tmp_path, monkeypatch,
                                                                        capsys):
    out = tmp_path / "o"
    prompts = tmp_path / "p.txt"
    prompts.write_text("\n".join(certify.build_prompts(41)))
    code = _main(monkeypatch, tiny["a"], tiny["b"], out, prompts, threshold=0.05)
    capsys.readouterr()
    res = json.loads((out / "report.json").read_text())
    assert "ci95_high" in res and res["ci95_high"] <= 0.05
    assert code == 0
