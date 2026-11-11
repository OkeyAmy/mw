# STATUS

## Agents
| agent | topic | status |
|---|---|---|
| A1 | weights & formats | done — **fully verified** |
| A2 | interpretability methods | done |
| A3 | steering & control | done |
| A4 | runtime engineering | done |
| A5 | visualization / UI | done — 2 corrections |
| A6 | limits & validation | done |
| A7 | CPU inference at scale | done |
| A8 | guided compression | done |
| A9 | neuron atlas | done — found the hardware error |
| D1 | proof harness | done — `code/proof_chain.py`, `--selftest` |
| D2 | adversarial audit | running |
| D3 | precedent hunter | running |

## Hardware truth (corrected)
Intel i7-8665U, 4 physical cores / 8 threads, 1.9 GHz, AVX2 only, no bf16,
~17 GB free RAM, **currently pinned at 800 MHz** (`powersave`, `no_turbo=1`).
Measured: ~6.4 tok/s on Q4_K_M via llama.cpp at 4 threads; ~2–3 tok/s fp32 HF.

## Verified numbers
- Qwen2.5-Coder-1.5B: 1,543,714,304 params, 3,087,467,144 B, 338 tensors, all BF16,
  tied embeddings (no `lm_head`), 84 QKV biases. **Verified from the safetensors header.**
- Logit lens: naive 2.50 s/token → batched 0.49 s/token (A4, measured).
- Q4_K_M: 7.17% relative Frobenius error, cosine ≥ 0.997 → lens/steering survive,
  neuron claims do not (A1, measured).

## Blockers
1. **CPU governor** — needs `sudo`. ~2x speedup available.
2. **Network** — ~88 KB/s per connection. Worked around with parallel range fetch
   (`code/pget.sh`, `code/fetch_model.sh`).
3. **Torch install** — pip kept timing out; now fetching the wheel with `pget.sh`
   and installing from the local file.

## Proof chain
| link | status |
|---|---|
| C1 see inside | harness written, **not run** |
| C2 lens faithful | harness written, **not run** |
| C3 causal patching | harness written, **not run** |
| C4 steering dose-response | harness written, **not run** |
| C5 mid-generation | harness written, **not run** |
| C6 force vs steer | harness written, **not run** |
| C7 atlas coverage | designed (A9), not started |
| C8 sanity gate | designed (A6), not started |

**Nothing is proven yet.** The instrument is built and self-testable; it has not
touched the real model.
