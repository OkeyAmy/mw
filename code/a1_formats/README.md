# A1 — GGUF format verification suite

Executable evidence for `agents/A1_formats.md`. All three scripts were run and pass.

## Setup

```bash
cd /home/okey/Desktop/Projects/mod_wei
uv venv --python 3.12 .venv-a1
uv pip install --python .venv-a1/bin/python gguf numpy
```

## Test data (~70 MB, not vendored)

```bash
mkdir -p /tmp/ggufdata && cd /tmp/ggufdata
BASE=https://huggingface.co/mradermacher/llama2.c-stories15M-GGUF/resolve/main
curl -sSL -O $BASE/llama2.c-stories15M.Q4_K_M.gguf
curl -sSL -O $BASE/llama2.c-stories15M.f16.gguf
mv llama2.c-stories15M.Q4_K_M.gguf stories15M.Q4_K_M.gguf
mv llama2.c-stories15M.f16.gguf     stories15M.f16.gguf
```

## Run (from the directory holding the .gguf files)

```bash
P=/home/okey/Desktop/Projects/mod_wei/.venv-a1/bin/python
S=/home/okey/Desktop/Projects/mod_wei/code/a1_formats

$P $S/gguf_probe.py stories15M.Q4_K_M.gguf   # header, metadata, per-tensor type histogram
$P $S/test_gguf_roundtrip.py                 # synthetic GGUF round-trip vs half-step bound
$P $S/crosscheck_quants.py                   # bit-exactness vs gguf.quants on real data
$P $S/measure_loss.py                        # Q4_K_M vs f16: relFro, SNR, sign flips
```

## Expected results

| script | assertion | measured |
|---|---|---|
| `gguf_probe.py` | parses a real Q4_K_M | 57 tensors, 5 ggml types, data_start 727,872 (3.47 % of file) |
| `test_gguf_roundtrip.py` | Q4_K error ≤ half a quant step | max\|err\| 0.003691 ≤ 0.003795; F32 exact; Q6_K 3.8× tighter than Q4_K |
| `crosscheck_quants.py` | `max\|mine − gguf.quants\| == 0` | **0.000e+00** on all 6 K-quant tensors |
| `measure_loss.py` | quantisation is lossy | whole model 4.09 % relFro / 27.77 dB SNR; Q4_K 7.17 % / 22.89 dB; 1.86 % weights exact; 1.41 % sign flips |

## What each file is

- `gguf_probe.py` — dependency-light GGUF v3 reader (header, typed KV metadata, tensor
  table, alignment) plus hand-written `dequant_q4_k` / `dequant_q6_k`. This is the
  reference implementation referenced in the report.
- `test_gguf_roundtrip.py` — writes a synthetic GGUF containing F32, Q4_K and Q6_K
  tensors from known ground truth, then re-reads and dequantises it. Catches packer and
  unpacker bugs against theory rather than against another implementation.
- `crosscheck_quants.py` — the important one: compares the hand-written dequantisers
  against `gguf.quants.dequantize` on real llama.cpp output, tensor by tensor.
- `measure_loss.py` — quantifies what is permanently lost, by diffing a Q4_K_M file
  against the f16 file of the same model.

## Caveats

- `crosscheck_quants.py` and `measure_loss.py` need the `gguf` package (hence the venv).
  `gguf_probe.py` needs only numpy.
- The two hard-coded reference shapes in `measure_loss.py` are for llama2.c stories15M
  (hidden 288, ffn 768, vocab 32000). For another model only the printed numbers change.
- The Q4_K_M file used here is a 24 M-param model, so its label-to-content mapping is
  dominated by the `_M` upgrade heuristic (mostly Q5_0). A 1.5 B model will be mostly
  Q4_K. The *method* is what transfers, not the exact percentages.
