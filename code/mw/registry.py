"""
mw.registry — per-architecture descriptors.

The whole point of this layer: every architecture has quirks that will silently
produce wrong numbers if ignored. We encode them explicitly instead of assuming
Qwen2 everywhere.

Fields
------
gguf_arch     : string in GGUF metadata `general.architecture`
hf_arch       : transformers class
tensor_map    : GGUF suffix -> HF suffix (2-D weights; 1-D names are identical)
config_from_gguf : gguf metadata key -> HF config key (scaling factors etc.)
fused         : True if the GGUF fuses tensors (qkv/gate_up) and needs splitting.
                We REFUSE these rather than guess.

Anything not in this table is reported as UNSUPPORTED, never silently mapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchSpec:
    gguf_arch: str
    hf_arch: str
    tensor_map: dict[str, str]
    config_from_gguf: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, object] = field(default_factory=dict)
    # If True, conversion MUST find every config_from_gguf key in the GGUF.
    # Falling back to 1.0 is the exact failure this field exists to prevent.
    requires_config_from_gguf: bool = False
    fused: bool = False
    notes: str = ""


_COMMON = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

SPECS: dict[str, ArchSpec] = {
    "qwen2": ArchSpec(
        gguf_arch="qwen2", hf_arch="Qwen2ForCausalLM",
        tensor_map=dict(_COMMON),
        notes="Qwen2/Qwen2.5 family. Has q/k/v biases. No scaling factors.",
    ),
    "llama": ArchSpec(
        gguf_arch="llama", hf_arch="LlamaForCausalLM",
        tensor_map=dict(_COMMON),
        notes="Llama family. Usually no attention biases; absent keys are skipped.",
    ),
    # NOTE: Mistral GGUFs report arch "llama", so a separate registry entry can
    # never be selected by `resolve()`. The HF class is decided by the config
    # fetched via --hf-repo, not by this table. Deliberately omitted (G1 D22).
    "granite": ArchSpec(
        gguf_arch="granite", hf_arch="GraniteForCausalLM",
        tensor_map=dict(_COMMON),
        config_from_gguf={
            "granite.embedding_scale": "embedding_multiplier",
            "granite.residual_scale": "residual_multiplier",
            "granite.attention.scale": "attention_multiplier",
            "granite.logit_scale": "logits_scaling",
        },
        # G1 D4: NO 1.0 defaults. All four factors are load-bearing (measured:
        # each alone moves logits 9.6-10.2; logits_scaling 122). A missing key
        # must abort the conversion, not silently reproduce the 8x bug.
        requires_config_from_gguf=True,
        notes=(
            "IBM Granite. Carries embedding/residual/attention/logit scaling. "
            "transformers defaults all four to 1.0, so a naive conversion loads "
            "fine and is numerically wrong. Conversion REFUSES if any is absent."
        ),
    ),
    "gemma": ArchSpec(
        gguf_arch="gemma", hf_arch="GemmaForCausalLM",
        tensor_map=dict(_COMMON),
        defaults={"hidden_activation": "gelu_pytorch_tanh"},
        notes=(
            "Gemma 2/3 embed scaling is handled inside HF from config; GGUF arch "
            "strings differ (gemma/gemma2/gemma3). Verify per version."
        ),
    ),
    "phi3": ArchSpec(
        gguf_arch="phi3", hf_arch="Phi3ForCausalLM",
        tensor_map=dict(_COMMON), fused=True,
        notes="Phi-3 fuses qkv into blk.N.attn_qkv in many GGUFs. REFUSED.",
    ),
}

# Aliases: ONLY names that denote the same graph as the target.
# qwen2_vl is deliberately absent: it has vision tensors and must be refused,
# not silently accepted as qwen2 (G1 D24).
ALIASES = {
    "qwen2.5": "qwen2",
    "llama3": "llama",
    "gemma2": "gemma", "gemma3": "gemma",
}

# HF architectures we can hook generically even if GGUF conversion is unsupported
HF_SUPPORTED = {
    "Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM",
    "GraniteForCausalLM", "GemmaForCausalLM", "Gemma2ForCausalLM",
    "Gemma3ForCausalLM", "PhiForCausalLM", "GPT2LMHeadModel",
}


def resolve(gguf_arch: str) -> ArchSpec | None:
    key = ALIASES.get(gguf_arch, gguf_arch)
    return SPECS.get(key)
