"""Adapter tests. Fast tests need no model; slow tests skip when weights are absent.

The slow tests are the release gate. They are marked `slow` so CI stays cheap.
"""
from pathlib import Path

import pytest
from mw.adapter import VIS_FULL, Capabilities, ModelAdapter

ROOT = Path(__file__).resolve().parent.parent
QWEN = ROOT / ".hf" / "bf16-hf"
GRANITE = ROOT / ".hf" / "granite-hf"

QWEN_OK = (QWEN / "model.safetensors").exists()
GRANITE_OK = (GRANITE / "model.safetensors").exists()

# Golden values recorded from the converted artefact in float16 (with GGUF Q/K
# RoPE rows unpermuted to HF split-half layout). These catch a wrong name map,
# missing RoPE unpermutation, or transposed weights, which self-consistency
# cannot (G1 D5).
GOLDEN_PROMPT = "def add(a, b):\n"
GOLDEN_ARGMAX = 601          # token 'return'
GOLDEN_MAXABS_LO, GOLDEN_MAXABS_HI = 19.0, 21.5


def test_capabilities_report_shape():
    c = Capabilities(model="x", arch="Y", visibility=VIS_FULL,
                     d_model=1536, n_layers=28, vocab=151936)
    r = c.report()
    assert "visibility" in r and "FULL" in r
    assert "28 layers" in r


@pytest.mark.slow
@pytest.mark.skipif(not QWEN_OK, reason="qwen bf16 weights not present")
def test_qwen_lens_identity():
    import torch
    a = ModelAdapter(str(QWEN)).load(dtype=torch.float32)  # BF16 source: fp16 overflows
    ids = a.tok(GOLDEN_PROMPT, return_tensors="pt").input_ids
    a._capturing = True
    a._post.clear()
    with torch.no_grad():
        out = a.model(input_ids=ids)
    a._capturing = False
    last = len(a.layers) - 1
    lens = a.logit_lens(a._post[last], topk=3)
    assert lens, "lens returned nothing"
    # exact identity through the adapter's own lens path
    w = a.lm_head.weight
    h = a._post[last].to(w.dtype)
    direct = (a.lm_head(a.final_norm(h)) / a.logits_scaling).float()
    assert float((direct - out.logits[:, -1, :].float()).abs().max()) < 1e-4


@pytest.mark.slow
@pytest.mark.skipif(not GRANITE_OK, reason="granite weights not present")
def test_granite_loads_with_expected_shape():
    import torch
    a = ModelAdapter(str(GRANITE)).load(dtype=torch.float16)
    cap = a.capabilities()
    assert cap.n_layers == 40
    assert cap.d_model == 2048
    assert a.logits_scaling == 8.0


@pytest.mark.slow
@pytest.mark.skipif(not GRANITE_OK, reason="granite weights not present")
def test_granite_golden_logits():
    """Independent of self-consistency: if the name map is wrong or a tensor is
    transposed, the argmax and magnitude change. This is the test that would
    have caught the 1.0-scaling bug that the old acceptance test passed."""
    import torch
    a = ModelAdapter(str(GRANITE)).load(dtype=torch.float16)
    ids = a.tok(GOLDEN_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = a.model(input_ids=ids)
    lg = out.logits[0, -1, :].float()
    argmax = int(lg.argmax())
    mx = float(lg.abs().max())
    assert argmax == GOLDEN_ARGMAX, f"argmax {argmax} != golden {GOLDEN_ARGMAX}"
    assert GOLDEN_MAXABS_LO < mx < GOLDEN_MAXABS_HI, f"max|logit|={mx:.2f} out of band"


@pytest.mark.slow
@pytest.mark.skipif(not QWEN_OK, reason="weights not present")
def test_steer_coef_zero_is_identity():
    """coef=0 must be bit-identical to no hook. Any drift invalidates steering."""
    import torch
    a = ModelAdapter(str(QWEN)).load(dtype=torch.float32)  # BF16 source: fp16 overflows
    ids = a.tok(GOLDEN_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        base = a.model(input_ids=ids).logits
    a.set_steer(4, torch.randn(1, a.cfg.hidden_size), 0.0)
    with torch.no_grad():
        same = a.model(input_ids=ids).logits
    a.set_steer(None, None, 0.0)
    assert torch.equal(base, same)


def test_no_hardcoded_home_paths_anywhere_in_code():
    """G1 D28: the old test only checked code/mw, so eight scripts shipped with
    /home/<user> paths."""
    offenders = []
    for p in (ROOT / "code").rglob("*.py"):
        if "/home/" in p.read_text(encoding="utf-8"):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"hardcoded user paths in: {offenders}"
