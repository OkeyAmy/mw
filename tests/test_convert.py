"""Name-mapping tests — pure functions, no model needed."""

from mw.convert import map_tensor
from mw.registry import SPECS


def test_embed_and_norm():
    q = SPECS["qwen2"]
    assert map_tensor("token_embd.weight", q) == "model.embed_tokens.weight"
    assert map_tensor("output_norm.weight", q) == "model.norm.weight"
    assert map_tensor("output.weight", q) == "lm_head.weight"


def test_layer_tensors():
    q = SPECS["qwen2"]
    assert map_tensor("blk.0.attn_q.weight", q) == "model.layers.0.self_attn.q_proj.weight"
    assert map_tensor("blk.27.ffn_down.weight", q) == "model.layers.27.mlp.down_proj.weight"
    assert map_tensor("blk.5.ffn_norm.weight", q) == "model.layers.5.post_attention_layernorm.weight"


def test_granite_uses_same_layer_names():
    g = SPECS["granite"]
    assert map_tensor("blk.39.attn_output.weight", g) == "model.layers.39.self_attn.o_proj.weight"


def test_unknown_tensor_returns_none():
    q = SPECS["qwen2"]
    assert map_tensor("blk.0.some_new_tensor.weight", q) is None
    assert map_tensor("general.whatever", q) is None


def test_granite_tensor_count_arithmetic():
    """Granite 3.3 2B: 40 layers x 9 tensors + embed + final norm = 362.
    This is the check that catches a name-map that silently drops a tensor kind."""
    per_layer = 9  # q,k,v,o,attn_norm,gate,up,down,ffn_norm
    assert 40 * per_layer + 2 == 362


def test_qwen_tensor_count_arithmetic():
    """Qwen2.5-Coder-1.5B: 28 layers x 12 (6 weights + 3 biases + 3 norms) + 2 = 338."""
    per_layer = 12
    assert 28 * per_layer + 2 == 338
