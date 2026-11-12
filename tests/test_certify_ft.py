"""Fine-tune certificate path: a fresh LoRA is mathematically identical to its
base (B is zero-initialised), so the certificate must report zero flips.

This is the strongest cheap test of the adapter loader: if merging is wrong
(dtype, which modules, base resolution), it cannot come out at exactly zero.
"""
import json

import pytest
import torch
from mw import certify
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

VOCAB = {w: i for i, w in enumerate(
    ["[UNK]", "def", "add", "return", "a", "+", "b", ":", "\n", "x", "=", "1"])}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("ftbase")
    tk = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]",
                                  pad_token="[UNK]", eos_token="[UNK]")
    cfg = Qwen2Config(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def adapter(base, tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    d = tmp_path_factory.mktemp("lora")
    m = Qwen2ForCausalLM.from_pretrained(base)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                     task_type="CAUSAL_LM")
    pm = get_peft_model(m, cfg)
    pm.save_pretrained(d)
    c = json.loads((d / "adapter_config.json").read_text())
    c["base_model_name_or_path"] = str(base)
    (d / "adapter_config.json").write_text(json.dumps(c))
    return d


def test_resolve_model_finds_base(base, adapter):
    b, a = certify.resolve_model(str(adapter))
    assert str(b) == str(base)
    assert a == str(adapter)
    _b2, a2 = certify.resolve_model(str(base))
    assert a2 is None


def test_fresh_lora_is_identity(base, adapter):
    """The whole point: B=0 means merged weights equal the base, so zero flips."""
    prompts = certify.build_prompts(6)
    lp_b, meta_b = certify.measure(str(base), prompts, "float32", base / "b.npz")
    lp_a, meta_a = certify.measure(str(adapter), prompts, "float32", adapter / "a.npz")
    res = certify.compare(lp_b, meta_b, lp_a, meta_a)
    assert res["flips"] == 0
    assert res["mean_kl"] == pytest.approx(0.0, abs=1e-6)


def test_adapter_cache_is_not_the_base_cache(base, adapter):
    """Two different artefacts must not share a cache fingerprint."""
    prompts = certify.build_prompts(2)
    assert (certify.cache_fingerprint(str(base), prompts, "float32")
            != certify.cache_fingerprint(str(adapter), prompts, "float32"))
