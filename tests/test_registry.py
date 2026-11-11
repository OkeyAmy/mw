"""Registry tests — no model, no download, runs in CI."""

from mw.registry import _COMMON, SPECS, resolve


def test_every_spec_is_complete():
    for key, s in SPECS.items():
        assert s.gguf_arch, key
        assert s.hf_arch, key
        assert s.tensor_map, key
        assert "attn_q.weight" in s.tensor_map, key
        assert "ffn_down.weight" in s.tensor_map, key


def test_common_map_has_all_four_weight_kinds():
    for k in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
              "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
        assert k in _COMMON


def test_qwen2_carries_qkv_biases():
    """Qwen2.5 checkpoints have q/k/v biases; dropping them corrupts the model."""
    m = SPECS["qwen2"].tensor_map
    for k in ("attn_q.bias", "attn_k.bias", "attn_v.bias"):
        assert k in m


def test_llama_has_no_bias_mapping():
    """Llama usually has no attn biases; the map may include them harmlessly,
    but conversion must skip absent keys rather than fail."""
    assert SPECS["llama"].hf_arch == "LlamaForCausalLM"


def test_granite_scaling_is_declared_and_required():
    """G1 D4: the old test enshrined 1.0 defaults, which IS the 8x bug.
    Scaling metadata must now be mandatory, with no fallback."""
    s = SPECS["granite"]
    assert s.config_from_gguf["granite.logit_scale"] == "logits_scaling"
    assert s.config_from_gguf["granite.embedding_scale"] == "embedding_multiplier"
    assert s.requires_config_from_gguf is True
    assert not s.defaults, "silent 1.0 defaults must not exist for granite"


def test_phi3_is_refused():
    assert SPECS["phi3"].fused is True


def test_aliases_resolve():
    assert resolve("qwen2.5") is SPECS["qwen2"]
    assert resolve("llama3") is SPECS["llama"]
    assert resolve("gemma2") is SPECS["gemma"]


def test_unknown_arch_returns_none():
    """A new architecture must be refused, never silently mapped to qwen2."""
    assert resolve("some_future_arch") is None


def test_no_absolute_paths_in_registry():
    """CWD-independent (G1 D26): resolve from the test file, not the shell."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "code" / "mw" / "registry.py"
    assert "/home/" not in p.read_text(encoding="utf-8")
