"""Smoke test — builds a tiny random Qwen2 and exercises the full public API.

This is the test G1 D3 said was missing: the previous fast suite never called
load(), so a NameError in load() shipped while 17 tests passed.

No download, no weights, <20 MB.
"""
import pytest
import torch
from mw.adapter import CapabilityError, ModelAdapter
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    vocab = {w: i for i, w in enumerate(
        ["[UNK]", "def", "add", "return", "a", "+", "b", ":", "\n", "x", "=", "1"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]",
                                  pad_token="[UNK]", eos_token="[UNK]")
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=64, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg).eval()
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_load_succeeds(tiny):
    """The exact path that shipped broken (G1 D1)."""
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    assert a.model is not None
    assert a.cfg is a.model.config          # D10: one config object
    assert a.dtype_actual == "torch.float32"


def test_capabilities_reports_shape(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    c = a.capabilities()
    assert c.n_layers == 2 and c.d_model == 32
    assert c.visibility in ("FULL", "PARTIAL")


def test_trace_has_provenance_and_exact_lens(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    tr = a.trace("def add a b :")
    assert "model" in tr and tr["model"]["dtype"]
    assert tr["model"]["site"] == "resid_post"
    # lens identity must be exact even on a random model
    ids = a.tok("def add a b :", return_tensors="pt").input_ids
    a._capturing = True
    a._post.clear()
    with torch.no_grad():
        out = a.model(input_ids=ids)
    a._capturing = False
    lens = a.lm_head(a.final_norm(a._post[len(a.layers) - 1])) / a.logits_scaling
    assert float((lens - out.logits[:, -1, :]).abs().max()) < 1e-4


def test_steer_coef_zero_is_bit_identical(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    ids = a.tok("def add a b :", return_tensors="pt").input_ids
    with torch.no_grad():
        base = a.model(input_ids=ids).logits
    a.set_steer(1, torch.randn(1, a.cfg.hidden_size), 0.0)
    with torch.no_grad():
        same = a.model(input_ids=ids).logits
    assert torch.equal(base, same)


def test_diff_in_means_and_set_steer_share_a_site(tiny):
    """D6: the vector is built at the same site steering writes to."""
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    # use only tokens the tiny vocab contains, or the input is empty
    v = a.diff_in_means([("return a + b", "def add a b :")], layer=1)
    assert v.shape[-1] == a.cfg.hidden_size
    assert a._steer["site"] == "pre"          # default site matches diff_in_means
    a.set_steer(1, v, 0.0)
    assert a._steer["layer"] == 1


def test_out_of_range_layer_raises(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    with pytest.raises(ValueError):
        a.set_steer(99, torch.zeros(1, a.cfg.hidden_size), 1.0)


def test_generate_runs(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    toks = a.generate("def add a b :", max_new_tokens=3)
    assert 1 <= len(toks) <= 3
    assert "token_str" in toks[0]


def test_hooks_are_idempotent(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    n = len(a._hooks)
    a._install_hooks()
    assert len(a._hooks) == n, "double install would apply steering twice"


def test_missing_output_embedding_raises(tiny):
    a = ModelAdapter(str(tiny)).load(dtype=torch.float32)
    class NoHead:
        def get_output_embeddings(self):
            return None
    a.model = NoHead()
    with pytest.raises(CapabilityError):
        _ = a.lm_head
