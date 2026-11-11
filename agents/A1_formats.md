# A1 — Model weights, formats, and extraction

*Scope: how to obtain exact, analysable weights from any LLM format — especially a
Q4_K_M GGUF in Ollama — and normalise them into one `ModelAdapter` loader interface.*

Everything below marked **[verified]** was executed on this machine (Fedora, Intel
i7-8665U — 4 physical cores / 8 threads, AVX2 only, no bf16, ~17 GB available RAM,
per LEDGER [X1]; python 3.14 + numpy 2.5.2, `gguf` 0.19.0 in `.venv-a1`). Everything
else is cited or marked **UNVERIFIED**.

---

## TL;DR (5 bullets)

1. **`Q4_K_M` is a recipe, not a tensor type.** A real Q4_K_M file mixes `Q4_K`,
   `Q6_K`, `Q8_0`, `Q5_0` and `F32` per tensor by a name-based heuristic. Measured on
   a real file: 57 tensors → Q5_0×35, F32×13, Q4_K×4, Q8_0×3, Q6_K×2. Any loader that
   assumes "Q4_K_M ⇒ Q4_K" is broken on the first tensor it touches. **[verified]**
2. **Dequantisation is exact, cheap, and *does not* recover the original weights.**
   The rule is `w = d·sc·q − dmin·m` per 32-weight sub-block. I reimplemented it from
   the ggml struct and it is **bit-identical (max abs deviation 0.0)** to
   `gguf.quants.dequantize` on real llama.cpp output. But against the f16 original of
   the same model, Q4_K tensors carry **7.17 % relative Frobenius error / 22.9 dB SNR**
   and only **1.86 %** of individual weights are reproduced exactly. **[verified]**
3. **`Qwen/Qwen2.5-Coder-1.5B-Base` does not exist** — HF returns HTTP 401 for it. The
   correct repo is **`Qwen/Qwen2.5-Coder-1.5B`**, which *is* the pretrained base
   (`Training Stage: Pretraining`). Read via HTTP range request: **338 tensors, 100 %
   BF16, 1,543,714,304 params, 3,087,467,144 bytes, no `lm_head` (tied embeddings)**.
   CONTEXT.md D2 is right about the *science*, wrong about the *repo name*. **[verified]**
4. **The empty `~/.ollama/models/blobs` is explained and is not a bug.** The server runs
   with `OLLAMA_MODELS=/run/media/okeyamy/Linux/ai/ollama-models` on a **removable exfat
   USB drive**. The drive was mounted when I started and **vanished mid-session**;
   `ollama list` then failed with `mkdir /run/media/okeyamy/Linux: permission denied`.
   Do not build a pipeline on that path — copy the blob out first. **[verified]**
5. **Format heterogeneity is real but shallow.** safetensors / GGUF / GPTQ / AWQ / EXL2 /
   MLX all collapse to `dict[str, Tensor]` in HF naming plus a config. ONNX and TFLite
   do not — they are graph formats and need a conversion step. One `ModelAdapter` with
   per-format dequantisers plus an explicit **fidelity tag** covers everything honest.

---

## Findings

### F1. GGUF v3 is a self-describing container: 24-byte header, typed KV metadata, tensor table, aligned data blob
[source: ggml-org/ggml, *GGUF file format specification* (`docs/gguf.md`), 2023–2026,
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md]

**What it is.** Exactly four regions, in order:

| region | contents |
|---|---|
| header (24 B) | `magic=0x46554747 ("GGUF")`, `version u32` (=3), `tensor_count u64`, `metadata_kv_count u64` |
| metadata | `kv_count` pairs: length-prefixed key string, `u32` value-type tag, typed value |
| tensor table | per tensor: name, `n_dims u32`, `dims u64[]`, `ggml_type u32`, `offset u64` |
| tensor data | contiguous; each tensor's offset is **relative to `data_start`** and a multiple of `general.alignment` (default 32) |

`data_start = align_up(header + metadata + tensor_table, alignment)`.

**Why it matters.** You can read the entire tensor map without touching a weight byte, and
mmap the data region. That is what makes "open a 1 GB GGUF and tell me every tensor type"
a 2 ms operation. Metadata keys are namespaced by architecture (`qwen2.block_count`,
`qwen2.embedding_length`, …), so arch facts come from the file, not from a side-channel.

**Confidence: high** (spec + reproduced on a real file). **Cost: trivial** (~120 lines).

**[verified]** on `llama2.c-stories15M.Q4_K_M.gguf`: version 3, alignment 32, 57 tensors,
`data_start = 727,872` = **3.47 %** of the file. My reader's `data_offset` and full
57/57 tensor table (name, dims, type) agree exactly with `gguf.GGUFReader`.

---

### F2. Q4_K is a 256-weight super-block with *two-level* scaling; dequantisation is 6 lines of arithmetic
[source: ggml-org/llama.cpp, `ggml/src/ggml-common.h` + `ggml/src/ggml-quants.c`
(`dequantize_row_q4_K`, `get_scale_min_k4`), 2024–2026,
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h]

**What it is.** `block_q4_K` is **144 bytes per 256 weights = 4.5 bpw**:

```c
typedef struct {
    ggml_half d;                 //  2 B  super-block scale, applied to the scales
    ggml_half dmin;              //  2 B  super-block scale, applied to the minimums
    uint8_t   scales[12];        // 12 B  8 × 6-bit scales + 8 × 6-bit mins, bit-interleaved
    uint8_t   qs[128];           // 128 B 256 × 4-bit codes, two per byte
} block_q4_K;
```

The 256 weights are **8 sub-blocks of 32**. Each sub-block `j` has a 6-bit scale `sc_j`
and a 6-bit minimum `m_j`, both themselves quantised by the super-block `d`/`dmin`:

```
w = d · sc_j · q − dmin · m_j        q ∈ [0,15]
```

This is *asymmetric* (it has a zero-point via `dmin·m_j`), unlike Q4_0. Two facts bite
implementers:

* The 12 scale bytes are **bit-interleaved**, not eight contiguous bytes. `get_scale_min_k4`:
  sub-blocks 0–3 read `sc[j] & 63` / `sc[j+4] & 63`; sub-blocks 4–7 splice low nibbles of
  `sc[8+j]` with the **top 2 bits of `sc[j]`/`sc[4+j]`**.
* The 128 `qs` bytes are consumed in **four 64-weight spans**: for each span the 32 low
  nibbles are weights 0–31 and the 32 high nibbles are weights 32–63.

**Why it matters.** This is the whole of "what Q4_K_M stores". Everything else
(Q5_K = Q4_K + 1 high bit; Q6_K = 16×16 sub-blocks, 8-bit scales, symmetric) is a
variation. Getting the interleave wrong yields plausible-looking but wrong weights —
exactly the failure mode that silently poisons an interpretability result.

**Confidence: high** — **[verified]** by two independent tests: (a) a synthetic
round-trip where max error sat exactly at the theoretical half-step bound
(0.003691 ≤ 0.003795), and (b) bit-exact agreement with `gguf.quants.dequantize`
on real llama.cpp output. **Cost: ~60 lines of numpy.**

---

### F3. A "Q4_K_M" file is a *mixture*; per-tensor dispatch is mandatory
[source: llama.cpp `llama-quantize.cpp` / `quantize` name-heuristic; measured directly,
2026]

**What it is.** `_M` = "medium", a heuristic that decides *which tensors get upgraded*
based on tensor name and row count. Measured composition of a real Q4_K_M file
**[verified]**:

| type | bpw | tensors | what they were |
|---|---|---|---|
| `Q5_0` | 5.50 | 35 | attn q/k/o (+ most v), ffn gate/up, token_embd |
| `F32` | 32.00 | 13 | all RMSNorm weights |
| `Q4_K` | 4.50 | 4 | `ffn_down` |
| `Q8_0` | 8.50 | 3 | `attn_v` (layers 2,5), `output` |
| `Q6_K` | 6.56 | 2 | `ffn_down` (layers 2,5) |
| **effective** | **6.64** | 57 | 24.4 M weights |

Note the file is labelled Q4_K_M yet is **mostly Q5_0** — on a 24 M-param model the
heuristic upgrades nearly everything. On the real 1.5 B model the Ollama blob is
986,048,512 B for 1,543,714,304 params ≈ **5.11 bpw**, consistent with a 4.5/6.56/8.5 mix.

**Why it matters.** The loader must be `for each tensor: dispatch on ggml_type`, and any
"compression ratio" or "fidelity" claim must be computed per tensor, not assumed from the
label. Also: **RMSNorm weights are F32 even in 4-bit files** — norms are never quantised.

**Confidence: high** (direct measurement). **Cost: trivial** (a dict).

---

### F4. Dequantisation cannot recover the original weights — measured, not assumed
[source: own experiment on `llama2.c-stories15M` Q4_K_M vs its `.f16` sibling
(mradermacher/llama2.c-stories15M-GGUF), 2026; both files downloaded and compared tensor
by tensor]

**What it is.** Dequantisation is *lossless with respect to the stored codes* and
*irreversibly lossy with respect to the trained weights*. Measured **[verified]**:

| type | tensors | mean rel. Frobenius error | mean SNR | min cosine |
|---|---|---|---|---|
| `Q4_K` | 4 | **7.17 %** | 22.89 dB | 0.99741 |
| `Q5_0` | 35 | 4.41 % | 27.16 dB | 0.99723 |
| `Q6_K` | 2 | 1.80 % | 34.89 dB | 0.99983 |
| `Q8_0` | 3 | 0.66 % | 43.88 dB | 0.99991 |
| **whole model** | 57 | **4.09 %** | **27.77 dB** | — |

Additional measured facts:
* Only **1.86 %** of individual weights are reproduced exactly (`|err| < 1e-7`).
* **1.41 %** of weights flip sign — and in a SwiGLU `gate`/`up` projection a sign flip
  moves the value to the other branch of `silu`, so this is not cosmetic.
* Cosine similarity stays ≥ 0.997 everywhere: **direction survives, magnitude is
  perturbed.** That is the honest summary of 4-bit for interpretability.

**Why it matters.** This is the quantitative backing for D2. It also tells you *which*
conclusions are safe on a dequantised model: anything that depends on weight *direction*
(logit lens, cosine-similarity probes, steering-vector geometry) is roughly preserved;
anything that depends on fine weight magnitude or on exact sign agreement at neuron
granularity is not. Note also that SNR 27.8 dB ≈ 4.6 effective bits, i.e. the 4.5 bpw
figure is honest.

**Confidence: high** for this model family; **medium** for extrapolating the exact
percentages to a 1.5 B model (smaller models quantise worse — see F9).
**Cost: 70 MB download + 5 s of compute.**

---

### F5. The tooling that dequantises correctly, and the commands that actually work
[source: `gguf` PyPI package 0.19.0 (2026-05-06), https://pypi.org/project/gguf/ ;
llama.cpp, https://github.com/ggml-org/llama.cpp]

**What it is.** Three layers, use them in this order:

1. **`gguf` python package** (pure python + numpy, no compilation, CPU-friendly — the
   right choice here). **[verified]** `gguf 0.19.0` installs into a py3.12 venv and
   exposes `GGUFReader`, `GGUFWriter`, `gguf.quants.dequantize`, and console scripts
   `gguf-dump`, `gguf-set-metadata`, `gguf-new-metadata`, `gguf-convert-endian`,
   `gguf-editor-gui`.

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python gguf numpy
   .venv/bin/gguf-dump --no-tensors model.gguf      # metadata + tensor table
   .venv/bin/gguf-dump model.gguf                   # + tensor type/size table
   ```

   ```python
   import gguf, numpy as np
   r = gguf.GGUFReader("model.gguf")
   t = next(x for x in r.tensors if x.name == "blk.0.ffn_down.weight")
   qt = gguf.GGMLQuantizationType(int(t.tensor_type))     # e.g. Q4_K
   w  = gguf.quants.dequantize(t.data, qt)                # (out, in) float32
   assert w.shape == tuple(int(x) for x in t.shape[::-1])
   ```

2. **llama.cpp C++** — fastest, and the only path that produces a *new GGUF*:
   ```bash
   # GGUF(Q4_K_M) -> GGUF(F16):  dequantise to a lossless-container copy
   ./llama-quantize model-Q4_K_M.gguf model-f16.gguf F16
   # HF safetensors -> GGUF (optionally dequantising / re-quantising)
   python convert_hf_to_gguf.py ./Qwen2.5-Coder-1.5B --outtype bf16 --outfile qwen.bf16.gguf
   ```
   **UNVERIFIED on this machine** — no llama.cpp binary is installed and building it is
   out of scope for A1. `llama-quantize <in> <out> F16` is the documented dequantise
   path; treat the exact CLI as needing a smoke test.

3. **`transformers` / `safetensors`** — for the science target (no dequantisation at all).

**Cheapest correct path from a Q4_K_M GGUF to usable weights** (LEDGER open question Q2):

```
GGUF blob  →  gguf.GGUFReader  →  gguf.quants.dequantize (per tensor, per ggml_type)
           →  reverse dims (out,in)  →  HF state_dict  →  AutoModelForCausalLM.load_state_dict
```

No llama.cpp build, no GPU, no C++ — `gguf` + numpy + transformers on CPU. For the
986 MB Qwen blob expect ~1–2 min of numpy and ~6.2 GB of fp32 RAM. **The result is
"usable fp16 weights" in the only sense that exists: the dequantised values, carrying
4.09 % relative error (F4). There is no path back to the original bf16 weights — that
information is gone.** If a task needs exact weights, download the safetensors (F7).

**Why it matters.** `gguf.quants` already implements every type including all i-quants
(`IQ2_XXS` … `IQ4_XS`), `TQ1_0/TQ2_0`, `MXFP4`, `NVFP4`. **Do not hand-roll a
dequantiser for exotic types** — only hand-roll Q4_K/Q6_K if you want to avoid the dep,
and then verify against `gguf.quants` as I did.

**Confidence: high.** **Cost: one `uv pip install`.**

---

### F6. Ollama's store is an OCI-style content-addressed blob dir, relocated here by `OLLAMA_MODELS` onto a removable drive
[source: Ollama, `manifest/paths.go` + *Manifest and blobs* wiki, 2024–2026,
https://github.com/ollama/ollama/blob/main/manifest/paths.go ; measured on this host]

**What it is.** `$OLLAMA_MODELS` (default `~/.ollama/models`, or
`/usr/share/ollama/.ollama/models` when installed as a systemd service) contains:

```
manifests/<host>/<namespace>/<model>/<tag>   # JSON: layers[] with mediaType + sha256 digest + size
blobs/sha256-<64 hex>                        # the actual bytes, named by digest
```

The weights layer is the one with `mediaType: application/vnd.ollama.image.model`; the
others are `…image.template`, `…image.license`, `…image.params`.

**Why the local `blobs/` looked empty — diagnosed, not guessed. [verified]**

| evidence | value |
|---|---|
| `~/.ollama` total size | **48 K** (manifests and blobs genuinely empty) |
| `systemctl cat ollama` drop-in | `Environment="OLLAMA_MODELS=/run/media/okeyamy/Linux/ai/ollama-models"`, `OLLAMA_HOST=0.0.0.0:11434`, `User=okeyamy` |
| `/proc/<pid>/environ` of `ollama serve` | confirmed `OLLAMA_MODELS=/run/media/...` — so it *is* the server's env, not the shell's |
| that path | exfat, `/dev/sdb1`, **15 G**, `findmnt -T` shows it mounted under `/run/media/okeyamy/Linux` |
| the model's blob | `sha256-6a77366395772462c84f0c4d226ac404674327cbe78c01e4391cc7e0c698851e`, **986,048,512 B**, magic `GGUF` |
| `ollama show` | `architecture qwen2 · 1.5B · ctx 32768 · embedding 1536 · quant Q4_K_M` |

So: **not a different user, not an overlay, not a stale cache — a relocated store on an
external volume.** `~/.ollama/models/{blobs,manifests}` are decoys created by the CLI.

**The failure mode is worse than "empty".** Mid-session the USB drive disappeared
(`sdb1` gone from `lsblk`), and then:

```
$ ollama list
Error: mkdir /run/media/okeyamy/Linux: permission denied: ensure path elements are traversable
```

That string is verbatim `manifest/paths.go`'s `BlobsPath` → `os.MkdirAll` wrapper
(`fmt.Errorf("%w: ensure path elements are traversable", err)`), which is why the error
mentions `mkdir` when you only asked to *list*. The `RequiresMountsFor=` directive in the
drop-in does not help once the device is physically gone.

**Confidence: high** (direct `/proc` + systemd inspection). **Cost: 5 minutes.**

---

### F7. The correct source of truth is `Qwen/Qwen2.5-Coder-1.5B` (the `-Base` name is wrong), 3.09 GB of pure BF16, tied embeddings
[source: HF model card + safetensors header read by HTTP range request, 2026,
https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B ; Qwen2.5-Coder report, arXiv:2409.12186,
2024]

**What it is. [verified]** Exact facts, obtained without downloading the 3 GB:

| fact | value |
|---|---|
| `model.safetensors` | **3,087,467,144 B** (header 38,528 B + 8 B length prefix + 3,087,428,608 B tensors) |
| tensors | **338**, dtype **BF16 for all 338** |
| params | **1,543,714,304** |
| `lm_head` | **absent** — `tie_word_embeddings: true`, logits = `embed_tokens.weight @ h` |
| norms | 57 × `[1536]` BF16 (`input_layernorm`, `post_attention_layernorm` ×28, `model.norm`) |
| `q_proj` / `k_proj` / `v_proj` | `[1536,1536]` **+ bias `[1536]`**, `[256,1536]` **+ bias `[256]`**, `[256,1536]` **+ bias `[256]`** |
| tensors per layer | **12**: q(w+b), k(w+b), v(w+b), o(w), gate, up, down, input_norm, post_norm |
| `mlp.down_proj` / `gate_proj` | `[1536,8960]` / `[8960,1536]` |
| `embed_tokens` | `[151936,1536]` |

Also: `torch_dtype: bfloat16`, `rope_theta: 1e6`, `rms_norm_eps: 1e-6`,
`max_position_embeddings: 32768`, `num_key_value_heads: 2`, `sliding_window: 32768` but
`use_sliding_window: false`.

**Two traps in the naming.**
1. `Qwen/Qwen2.5-Coder-1.5B-Base` → **HTTP 401**, and it does not appear in HF search at
   all. It does not exist. `Qwen/Qwen2.5-Coder-7B-Base` likewise 401.
2. The repo that *does* exist, `Qwen/Qwen2.5-Coder-1.5B`, **is the pretrained base** — its
   card says `Training Stage: Pretraining` and "We do not recommend using base language
   models for conversations". The instruct model is `…-1.5B-Instruct`.

**Do we need an fp32 copy?** No — but on *this* box you want fp32 **in RAM**.
Two separate questions:

* *On disk / as the source of truth:* **bf16 is the training dtype and is exact.** Keep
  the 3.09 GB bf16 checkpoint as the artifact of record. Downloading a separate fp32
  release would be 6.2 GB and would *invent* precision that was never there.
* *For the forward pass:* the LEDGER's **[X1]** audit established this CPU is an
  **i7-8665U with AVX2 only — no AVX-512, no bf16**. So bf16 matmul is emulated and slow;
  cast the model to **fp32 once at load** (`model.to(torch.float32)`), which costs 6.2 GB
  of RAM (fits the ~17 GB available) and removes per-op upcast overhead. bf16 weights →
  fp32 activations is *lossless* (every bf16 value is exactly representable in fp32), so
  this costs nothing in fidelity and is the only sane default here.

Also note **Qwen2 has QKV bias** — `q_proj.bias [1536]`, `k_proj.bias [256]` **and
`v_proj.bias [256]`** (three per layer; `o_proj` has none). A hand-written forward pass
that omits them is silently wrong, and the two 256-element biases are easy to miss
because `k_proj`/`v_proj` weights are only `[256,1536]` under GQA.

**Confidence: high.** **Cost: 3.1 GB download, ~2 min on this link.**

---

### F8. "Any format" is really four classes; only two of them are honestly convertible to a hookable PyTorch module
[source: GPTQ — Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers*, ICLR 2023, https://arxiv.org/abs/2210.17323 ; AWQ — Lin et al.,
*AWQ*, MLSys 2024 Best Paper, https://arxiv.org/abs/2306.00978 ; EXL2 — exllamav2,
https://github.com/turboderp-org/exllamav2 ; safetensors — HF, https://huggingface.co/docs/safetensors]

| format | what it physically is | lossless → hookable `nn.Module`? | cost / caveat |
|---|---|---|---|
| **safetensors** (bf16/fp16/fp32) | raw tensors + JSON header, no pickle | **Yes, exact** | none. This is the science target. |
| **PyTorch `.bin`** | `torch.save` pickle | **Yes, exact** | arbitrary-code-execution risk on load; needs `weights_only=True` |
| **GGUF K-quants** (Q4_K_M…) | block-quantised super-blocks + metadata | **Yes, but lossy-dequantised** — reproduces the *quantised* weights exactly, never the originals (F4) | ~7 % rel. error for Q4_K; must dispatch per tensor (F3) |
| **GGUF i-quants** (IQ1_S…IQ4_XS) | codebook/grid-indexed, importance-matrix-derived | **Yes, lossy-dequantised** | lower bpw (1.5–4.25), *worse* error; dequantiser is a lookup table — **use `gguf.quants`, never hand-roll** |
| **GPTQ** (int3/4, group 128) | int codes + per-group scales/zeros, act-order permutation | **Yes, lossy-dequantised** | dequant is deterministic; **must undo the `g_idx` act-order permutation** or weights are scrambled |
| **AWQ** | int4 codes + per-channel scales folded in | **Yes, lossy-dequantised** | must divide out the per-channel scale into the *preceding* op; the scaling is an equivalent transformation, so get it wrong and the model is subtly off |
| **EXL2** | GPTQ-style, **mixed bit-width per row** + per-row bit-allocation table | **Yes, lossy-dequantised** | per-row widths means the dequantiser is data-dependent; needs exllamav2's own tables |
| **MLX** (4/8-bit) | safetensors + group scales/biases (32- or 64-group) | **Yes, lossy-dequantised** | Apple-specific layout; weights are plain safetensors so the *container* is easy, the *dequant* is MLX's |
| **ONNX** | **graph** (op DAG) + initializers, often int8 QDQ | **No, not directly** | needs `onnx2torch` or a re-export; QDQ nodes must be folded. Hooks attach to the *converted* torch module, not to ONNX. |
| **TFLite** | flatbuffer graph, often int8 per-axis | **No** | no credible PyTorch round-trip. **Declare out of scope.** |

**The key structural point.** Every weight-only quantisation scheme is
`W ≈ dequant(q, scale, zero)` — an *affine decode of stored integers*. So all of them
collapse to the same thing: **dequantise to a `state_dict`, load into the architecture,
then hook normally.** Quantisation is a *storage* problem, not a *runtime* problem, once
you refuse to use the vendor's kernel.

**Why it matters for us.** This is what makes D4 achievable: the analysis layer never sees
a format. It sees `nn.Module` + a fidelity tag.

**Confidence: high** for the taxonomy and for GGUF (measured); **medium** for the exact
GPTQ/AWQ/EXL2 dequant details, which I have not executed here — mark those
**UNVERIFIED-by-execution** until someone runs them.

---

### F9. 4-bit does not destroy interpretability, but it *does* perturb it in a measurable, direction-preserving way — and 2-bit does destroy it
[source: Singh & Sajjad, *Interpreting the Effects of Quantization on LLMs*, AACL 2025,
https://arxiv.org/abs/2508.16785 ; *From Signal Degradation to Computation Collapse:
Uncovering the Two Failure Modes of LLM Quantization*, Findings of ACL 2026,
https://aclanthology.org/2026.findings-acl.1162/ ; *Through a Compressed Lens:
Investigating The Impact of Quantization on Factual Knowledge Recall*, TrustNLP 2026,
https://aclanthology.org/2026.trustnlp-main.2/]

**What the evidence says.**

1. **Singh & Sajjad (AACL 2025)** apply calibration analysis, dead-neuron counts, and
   neuron-contribution analysis to several LLMs at 4-bit and 8-bit. Findings: calibration
   change is "generally minor"; **the number of dead neurons is unchanged by
   quantisation**; neuron-redundancy effects vary by model. Their conclusion: "we did not
   observe any drastic change which may discourage the use of quantization".

2. **"Signal Degradation to Computation Collapse" (ACL Findings 2026)** is the sharpest
   result and the one to design around. Using logit lens and causal tracing on quantised
   models, it separates two failure modes and reports that the **quantisation error
   subspace alignment** differs qualitatively by bit-width: at **2-bit the error is highly
   aligned with the signal subspace (similarity ≈ 0.8)** — i.e. the noise is *not* random,
   it directly corrupts the model's primary features; at **4-bit alignment is ≈ 0.3**,
   "resembling random noise that affects precision without destroying signal structure".
   At 2-bit they also measure **> 30 % sign flips** in gate activations and a **Jaccard
   index of ≈ 0.1** for the top-1 % activated neurons versus FP16 — the model "activates
   the wrong neurons".

3. **"Through a Compressed Lens" (TrustNLP 2026)** studies factual knowledge recall with
   interpretability probes across three quantisers and finds quantisation causes
   information loss that degrades FKR, and that **the effect is amplified in smaller
   models within the same family**. We are on a 1.5 B model — the worst case in that
   finding.

**So: is Q4_K_M structurally interpretable?** A defensible, evidence-backed position:

* **Yes at the level we need** — residual-stream directions, logit-lens trajectories,
  attention entropy, and steering geometry are all *direction* quantities, and my measured
  weight cosine similarity is ≥ 0.997 with 22.9 dB SNR on the quantised tensors (F4).
  Consistent with (1) and (2)'s 4-bit "random-noise-like" characterisation.
* **No at the level of "this neuron means X"** — 1.4 % sign flips and 7.2 % weight error
  mean individual-neuron stories will not transfer, and there is no evidence anyone has
  shown they do.
* **The killer argument is not "is it noisy" but "is it the same model".** Any causal
  claim (steering, patching) validated on a dequantised Q4_K_M must be *re-validated* on
  bf16, because the two are different functions. This is exactly why D2 is correct: Q4_K_M
  is a **format-compatibility target**, never the science target.

**Confidence: high** that these are the right papers and that 4-bit ≫ 2-bit in
interpretability-preservation; **medium** on transferring exact numbers to Qwen2.5-Coder-1.5B.
**Cost: zero (reading).**

---

## Concrete implementation (code sketches, real APIs)

### 1. Diagnose where Ollama actually put the models (run this before anything else)

```bash
# the authoritative answer is the *server's* environment, not your shell's
systemctl cat ollama | grep -E 'Environment|User='
tr '\0' '\n' < /proc/$(pgrep -f 'ollama serve' | head -1)/environ | grep OLLAMA

M="${OLLAMA_MODELS:-$HOME/.ollama/models}"
findmnt -T "$M"                    # is it even mounted? exfat? removable?
du -sh "$M"                        # 0 or 48K => wrong location
ls -la "$M"/blobs | head

# cross-check the two candidate defaults
du -sh ~/.ollama/models /usr/share/ollama/.ollama/models 2>/dev/null
```

### 2. Extract the raw `.gguf` blob for a model (manifest is the source of truth)

```bash
MODEL="qwen2.5-coder:1.5b-base-q4_K_M"
M="${OLLAMA_MODELS:-$HOME/.ollama/models}"
# registry.ollama.ai/library/<name>/<tag>  (a colon becomes a path separator)
MAN="$M/manifests/registry.ollama.ai/library/${MODEL%%:*}/${MODEL##*:}"
python3 - "$MAN" <<'PY'
import json, sys, os, shutil
man = json.load(open(sys.argv[1]))
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(sys.argv[1])))))          # .../models
for L in man["layers"]:
    if L["mediaType"] == "application/vnd.ollama.image.model":
        src = os.path.join(root, "blobs", L["digest"].replace(":", "-"))
        print("blob:", src, L["size"], "bytes")
        shutil.copyfile(src, "model.gguf")        # copy OFF the removable drive
        print("wrote model.gguf")
        break
else:
    sys.exit("no .image.model layer in manifest")
PY
sha256sum model.gguf          # must equal the manifest digest
head -c4 model.gguf           # must be "GGUF"
```

`ollama show "$MODEL" --modelfile` is the *human* view (it reconstructs `FROM`, `SYSTEM`,
`PARAMETER` from the stored layers) — useful for prompts/templates, but it is **not** how
you get the weights. **UNVERIFIED on this host**: the drive vanished before I could run
the `--modelfile` variant (plain `ollama show` did work).

### 3. Read a GGUF without any third-party dependency (~120 lines, verified)

The probe I actually ran (`gguf_probe.py`) — header, typed KV metadata, tensor table,
alignment, plus Q4_K/Q6_K dequantisers:

```python
def parse(path):
    f = open(path, "rb"); r = Reader(f)
    assert r.read(4) == b"GGUF"
    version  = r.unpack("<I", 4)
    n_tensor = r.unpack("<Q", 8)
    n_kv     = r.unpack("<Q", 8)
    meta = {r.string(): (lambda t: read_meta_value(r, t))(r.unpack("<I", 4))
            for _ in range(n_kv)}
    tensors = []
    for _ in range(n_tensor):
        name = r.string()
        nd   = r.unpack("<I", 4)
        dims = [r.unpack("<Q", 8) for _ in range(nd)]
        tensors.append(dict(name=name, dims=dims,
                            type=r.unpack("<I", 4),
                            offset=r.unpack("<Q", 8)))
    align = meta.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align
    return dict(f=f, version=version, meta=meta, tensors=tensors,
                data_start=data_start)
```

```python
def dequant_q4_k(block):                      # 144 bytes -> 256 float32
    d, dmin = float(f16(block[0:2])[0]), float(f16(block[2:4])[0])
    sc, qs  = block[4:16], block[16:144]
    scales, mins = [0]*8, [0]*8
    for j in range(4):
        scales[j] = sc[j] & 63
        mins[j]   = sc[4+j] & 63
    for j in range(4):                        # the bit-interleave that bites
        scales[4+j] = (sc[8+j] & 0x0F) | ((sc[j]   >> 6) << 4)
        mins[4+j]   = (sc[8+j] >> 4)   | ((sc[4+j] >> 6) << 4)
    out = np.empty(256, np.float32); is_, qi, o = 0, 0, 0
    for _ in range(4):                        # four 64-weight spans
        d1, m1 = d*scales[is_],   dmin*mins[is_]
        d2, m2 = d*scales[is_+1], dmin*mins[is_+1]
        c = np.frombuffer(qs[qi:qi+32], np.uint8)
        out[o:o+32]    = d1*(c & 0x0F) - m1   # low nibbles
        out[o+32:o+64] = d2*(c >> 4)   - m2   # high nibbles
        o += 64; qi += 32; is_ += 2
    return out
```

**Verification harness** (this is the part that matters — run it, don't trust the code):

```python
# 1) synthetic round-trip: error must sit at the half-step bound, not above
#    measured: max|err| 0.003691 <= half-step 0.003795   PASS
# 2) bit-exactness against the reference implementation on a REAL file
import gguf, numpy as np
r = gguf.GGUFReader("model.gguf")
t = next(x for x in r.tensors if x.name == "blk.0.ffn_down.weight")
ref = gguf.quants.dequantize(t.data, gguf.GGMLQuantizationType(int(t.tensor_type)))
mine = np.concatenate([dequant_q4_k(blk) for blk in blocks_of(t)])
assert np.abs(mine - ref.reshape(-1)).max() == 0.0     # measured: 0.0 exactly
```

### 4. GGUF → HF `state_dict`: the two rules that must not be guessed

```python
GGUF2HF = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",                      # absent when embeddings are tied
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",  "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

def gguf_tensor_to_hf(t):
    """(a) dequantise by type;  (b) assert the orientation is (out, in)."""
    qt = gguf.GGMLQuantizationType(int(t.tensor_type))
    # t.data is the raw block array, shape (ne1, bytes_per_row) for K-quants.
    # dequantize() already returns (ne1, ne0) == (out, in); the reshape is an assert.
    w  = gguf.quants.dequantize(t.data, qt)
    return w.reshape(tuple(int(x) for x in t.shape[::-1]))   # (out, in)
```

Measured orientation **[verified]**: `t.shape = (768, 288)` = `(ne0=in, ne1=out)`,
`t.data.shape = (288, 432)` = `(out, 432 B/row)` where `432 = 768/256 × 144` bytes,
and `dequantize(...).shape = (288, 768)` = `(out, in)` — i.e. the library already hands
you HF orientation, and `t.shape[::-1]` is the assertion that proves it.

**Rule (a): dispatch per tensor, never by file label** (F3).
**Rule (b): GGUF stores `ne0 = in`, `ne1 = out`; HF stores `[out, in]`, so reverse.**
Measured confirmation **[verified]**:

```
blk.0.ffn_down.weight   gguf.shape=(768, 288)  t.data.shape=(288, 432)
   -> reversed (288, 768) == HF down_proj [hidden=288, ffn=768]   ✓
blk.0.ffn_gate.weight   gguf.shape=(288, 768)  -> (768, 288) == gate_proj ✓
token_embd.weight       gguf.shape=(288,32000)-> (32000,288) == embed_tokens ✓
```

Then `assert` every shape against the HF config before loading; a transpose bug is
otherwise invisible (all the square matrices pass).

### 5. `ModelAdapter` — the interface (canonical internal representation)

Canonical form = **`dict[str, torch.Tensor]` in HF naming, contiguous, `[out, in]`, in a
declared dtype** + a `ModelSpec` + a `Provenance`. Everything else is a reader.

```python
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Protocol, Iterator, Literal

Fidelity = Literal["exact", "dequantized_lossy", "reimplemented"]

@dataclass(frozen=True)
class ModelSpec:
    arch: str; n_layers: int; hidden: int
    n_heads: int; n_kv_heads: int; head_dim: int
    intermediate: int; vocab: int
    rope_theta: float; rms_eps: float
    tie_embeddings: bool
    has_qkv_bias: bool                      # Qwen2: q_proj + k_proj + v_proj bias

@dataclass(frozen=True)
class Provenance:
    """Honesty record. Travels with every trace."""
    source: str                             # path / repo / ollama tag
    container: str                          # "safetensors" | "gguf" | "onnx" | ...
    quant: str | None                       # "Q4_K_M" | "gptq-4bit-g128" | None
    tensor_types: dict[str, int]            # {"Q4_K": 4, "Q6_K": 2, ...}
    fidelity: Fidelity
    effective_bpw: float | None = None
    weight_snr_db: float | None = None      # measured vs a higher-precision sibling

class WeightReader(Protocol):
    """One per container. The ONLY format-aware code in the system."""
    def spec(self) -> ModelSpec: ...
    def read(self) -> Iterator[tuple[str, torch.Tensor]]: ...   # HF names, [out,in]
    def provenance(self) -> Provenance: ...

class ModelAdapter:
    """Format-agnostic. Everything downstream (A2/A3/A4) sees only this."""
    spec: ModelSpec
    provenance: Provenance
    module: torch.nn.Module                 # hookable

    def load(self, *, device="cpu", dtype=torch.float32,
             materialize=True) -> "ModelAdapter": ...
    # dtype defaults to float32, NOT bfloat16: this CPU has no bf16 hardware
    # (LEDGER [X1]). bf16-on-disk -> fp32-in-RAM is lossless.
    def named_parameters_hf(self) -> Iterator[tuple[str, torch.Tensor]]: ...
    def forward_trace(self, input_ids, *, hooks: list | None = None): ...
    def attach_steering(self, layer: int, vector: torch.Tensor,
                        coef: float, positions="all") -> "Handle": ...

# readers: one line each to register
READERS = {
    "safetensors": "readers.safetensors_reader:SafetensorsReader",  # exact
    "pytorch_bin": "readers.torch_reader:TorchBinReader",           # exact
    "gguf":        "readers.gguf_reader:GGUFReader",                # dequantised
    "gptq":        "readers.gptq_reader:GPTQReader",                # dequantised
    "awq":         "readers.awq_reader:AWQReader",                  # dequantised
    "mlx":         "readers.mlx_reader:MLXReader",                  # dequantised
    "exl2":        "readers.exl2_reader:EXL2Reader",                # dequantised
    "onnx":        "readers.onnx_reader:ONNXReader",                # via onnx2torch
}
```

Load path, in words: `detect(path) → Reader → (spec, stream of [out,in] tensors,
provenance) → build empty nn.Module from spec (transformers `AutoModelForCausalLM
.from_config`) → load_state_dict(strict=True) → module`. Hooks, logit lens, steering all
happen on `module` and never learn where the weights came from — except that
`provenance.fidelity` is stamped into the trace so A6 can refuse to compare across
fidelities.

### 6. The one-command sanity gate for the whole pipeline

```bash
# after loading, before trusting any trace:
python -m modwei.check --model ./Qwen2.5-Coder-1.5B \
  --expect '{"n_layers":28,"hidden":1536,"n_heads":12,"n_kv_heads":2,"vocab":151936}' \
  --expect-tensors 338 --expect-no-lm-head \
  --logits-ref ./golden_logits.npy      # bf16 reference forward, 1 fixed prompt
```

---

## What NOT to do (traps)

1. **Do not trust the quantisation label.** `Q4_K_M` files contain Q5_0/Q8_0/Q6_K/F32
   tensors. Dispatch on `ggml_type` per tensor. (Measured: 5 distinct types in one file.)
2. **Do not assume `~/.ollama/models` is where models are.** Check
   `/proc/<pid>/environ` of `ollama serve`. Here the real store was on a removable exfat
   USB drive that disappeared mid-session.
3. **Do not build a pipeline that reads weights in place from `$OLLAMA_MODELS`.** Copy the
   blob to local disk and `sha256sum` it against the manifest digest first.
4. **Do not hand-roll i-quant (IQ*) dequantisers.** They are grid/codebook lookups with
   importance-matrix-derived scales. Use `gguf.quants`. Hand-rolling Q4_K/Q6_K is fine
   *only* with the bit-exact cross-check in §3.
5. **Do not forget the transpose.** GGUF is `(in, out)`, HF is `(out, in)`. Square
   matrices hide the bug; assert against `config` shapes for the non-square ones
   (`k_proj [256,1536]`, `down_proj [1536,8960]`).
6. **Do not add `lm_head` to this checkpoint.** It is tied; `lm_head` does not exist in
   the safetensors and must be aliased to `embed_tokens.weight`, or you will load 151936×1536
   of garbage.
7. **Do not forget Qwen2's QKV bias** — `q_proj.bias [1536]`, `k_proj.bias [256]`,
   **and `v_proj.bias [256]`**. That is 3 biases per layer (12 tensors/layer total), and
   `o_proj` has none. A hand-written forward that omits them is subtly wrong everywhere;
   a `strict=True` load will catch it, a `strict=False` load will not.
8. **Do not download an fp32 release "for precision".** bf16 is the training dtype; fp32
   adds bits that were never there and 3 GB of download. Do, however, cast to fp32 **in
   RAM** for compute — this CPU has no bf16 support (LEDGER [X1]) and bf16→fp32 is lossless.
9. **Do not call dequantised Q4_K_M "the model".** 7.2 % weight error, 1.4 % sign flips.
   Any causal claim must be re-validated on bf16.
10. **Do not chase TFLite/ONNX-runtime support.** Graph formats have no PyTorch hooks;
    converting them back is a separate project. Declare them out of scope in v1.
11. **Do not run `ollama` commands in a script that must not fail** while the store is on
    removable media — `ollama list` itself errors with a `mkdir` message.

---

## Open questions for other agents

* **A4 (runtime):** the adapter must hand you a `nn.Module` whose `forward` you can hook
  per layer. Do you want `output_hidden_states=True` (cheap, one flag, gives all 29
  hidden states but *not* per-head attention or MLP-out), or explicit
  `register_forward_hook` on `model.layers[i]`? The former is 1 line and enough for logit
  lens; the latter is needed for `mlp_out_norm` and `attn_entropy` in the trace schema.
  Also: at 1.5 B bf16 on CPU, is `torch.compile` worth it or does it just add latency?
* **A2 (interp):** given F4 (direction preserved, magnitude perturbed, 1.4 % sign flips),
  which of your metrics are direction-only and therefore safe on dequantised weights?
  I claim: logit lens, cosine probes, attention entropy, steering-vector geometry = safe;
  neuron-level attribution, sparse-feature dictionaries trained on quantised activations =
  not safe. Confirm or refute.
* **A3 (steering):** steering vectors are differences of mean activations, i.e.
  directions — so they should transfer bf16↔Q4_K_M better than weights do. Worth
  measuring: compute the same steering vector on both and report cosine. That is a cheap,
  publishable sanity result and it directly tests my F4 claim.
* **A6 (skeptic):** I am proposing `provenance.fidelity` in the trace (below). Please
  make it a hard gate: refuse to aggregate metrics across different `fidelity` values
  without an explicit override, and require a bf16↔dequantised agreement check before any
  causal claim is accepted.
* **A5 (UI):** do you want to *show* the quantisation (a per-tensor type badge, a
  "fidelity: dequantised Q4_K_M, SNR 22.9 dB" chip)? If yes, `provenance.tensor_types` is
  already the right shape to render.

---

## SCHEMA CHANGE

Add a `model` block at the **root** of every trace. Rationale: a trace is meaningless
without knowing whether it came from bf16 or from dequantised Q4_K_M, and A6 cannot
enforce anything it cannot see.

```json
{"model": {"id": "Qwen/Qwen2.5-Coder-1.5B",
           "container": "safetensors",
           "quant": null,
           "fidelity": "exact",
           "effective_bpw": 16.0,
           "weight_snr_db": null,
           "dtype": "bfloat16"},
 "step": 0, "token_id": 123, "token_str": "def",
 "layers": [ ... ], "logits_topk": [ ... ], "chosen": [ ... ]}
```

* `fidelity ∈ {"exact","dequantized_lossy","reimplemented"}`.
* For a GGUF run: `"quant": "Q4_K_M"`, `"fidelity": "dequantized_lossy"`,
  `"tensor_types": {"Q4_K": 4, "Q6_K": 2, "Q8_0": 3, "Q5_0": 35, "F32": 13}`,
  `"weight_snr_db": 27.8`.
* Backwards compatible: existing consumers ignore unknown root keys.

---

## Build next (ordered, with effort estimate)

| # | task | effort | why now |
|---|---|---|---|
| 1 | **Fix the repo name** everywhere: `Qwen/Qwen2.5-Coder-1.5B-Base` → `Qwen/Qwen2.5-Coder-1.5B`. Download bf16 (3.09 GB), `sha256` it, record in provenance. | 20 min | blocks everyone; the old name 401s |
| 2 | **`readers/safetensors_reader.py`** + `ModelAdapter.load()` + the shape/bias/tied-embedding asserts (§4, §6). | 3–4 h | this is the science path; everything else is optional |
| 3 | **`provenance.fidelity` into the trace** (schema change above) before any trace is written to disk. | 1 h | retrofitting provenance is painful; A6 needs it |
| 4 | **`readers/gguf_reader.py`** using `gguf.quants` + the GGUF2HF map + reverse-dims, with the bit-exact cross-check as a unit test. | 4–5 h | makes the "any format" claim true for the format we actually have |
| 5 | **Copy the Q4_K_M blob off the USB drive** (986 MB) to `./models/`, verify `sha256`, and run the same 5 fixed prompts through Ollama (behavioural baseline) *and* through the dequantised adapter. Report logit agreement. | 3 h | this is the bf16↔Q4_K_M agreement check A6 needs, and it de-risks the removable drive |
| 6 | **`readers/{gptq,awq}.py`** — only if a target model requires them. Note the act-order permutation (GPTQ) and per-channel scale folding (AWQ). | 1–2 d each | defer; no GPTQ/AWQ model in scope today |
| 7 | **`readers/onnx.py`** via `onnx2torch`, and an explicit `NotImplementedError` for TFLite/EXL2. | 1 d | honesty over coverage |
| 8 | **`modwei.check`** CLI (§6) as a pre-flight for every experiment. | 4 h | cheap insurance against silent transpose/bias bugs |

*Ordering logic:* 1 unblocks; 2–3 give a trustworthy science path; 4–5 make the format
claim real and produce the fidelity evidence; 6–7 are demand-driven; 8 hardens everything.

---

### Appendix — reproduction log (what was actually run)

| step | command | result |
|---|---|---|
| venv | `uv venv --python 3.12 .venv-a1 && uv pip install gguf numpy` | gguf 0.19.0, numpy 2.5.4 |
| locate store | `tr '\0' '\n' < /proc/<ollama-pid>/environ` | `OLLAMA_MODELS=/run/media/okeyamy/Linux/ai/ollama-models` |
| mount | `findmnt -T $OLLAMA_MODELS` | `/dev/sdb1 exfat`, 15 G |
| manifest | `cat .../manifests/registry.ollama.ai/library/qwen2.5-coder/1.5b-base-q4_K_M` | weights digest `sha256:6a7736…851e`, 986,048,512 B |
| blob magic | `head -c4 sha256-6a7736…` | `GGUF` |
| **drive loss** | `lsblk`, then `ollama list` | `sdb1` gone; `mkdir … permission denied` |
| parser test | `python3 gguf_probe.py stories15M.Q4_K_M.gguf` | 57 tensors, data_start 727,872, 5 ggml types |
| bit-exactness | `python3 crosscheck_quants.py` | max abs deviation `0.000e+00` on all 6 K-quant tensors |
| loss | `python3 measure_loss.py` (Q4_K_M vs .f16) | whole-model relFro 4.09 %, SNR 27.77 dB |
| HF facts | `curl -r 0-7` + `curl -r 8-…` on `model.safetensors` | 338 tensors, all BF16, 1,543,714,304 params, no `lm_head` |

Artifacts are copied into `code/a1_formats/` with a `README.md` — `gguf_probe.py` (reader
+ Q4_K/Q6_K dequantisers), `test_gguf_roundtrip.py` (synthetic round-trip vs the
half-step bound), `crosscheck_quants.py` (bit-exactness vs `gguf.quants`),
`measure_loss.py` (Q4_K_M vs `.f16`). They expect the `gguf` package importable (use
`.venv-a1/bin/python`) and the two `stories15M` files in the CWD (~70 MB, from
`mradermacher/llama2.c-stories15M-GGUF`). All three were re-run from their new location
and pass.
