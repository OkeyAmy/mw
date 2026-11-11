# A7 — Conventional CPU inference at scale (8B / 14B / 32B / 70B on this box)

Scope: what it actually takes to run a bigger LLM on 8 "cores" + 31 GB RAM, what fits,
how fast, and what quality the quantization costs. All arithmetic below is shown so it can
be re-derived; every number that is not arithmetic is marked **[MEASURED]**, **[PUBLISHED]**
or **[ESTIMATE]**.

> **Measurement conditions (matter for every number in this file).**
> The box was measured while 5 other research agents were running (load average 13.9).
> The CPU had `intel_pstate/no_turbo = 1` and `scaling_governor = powersave`, so all 8
> threads were pinned at **~0.90 GHz** (47% of the 1.9 GHz base clock). I could not change
> either setting — `sudo` requires a password. So the "now" column of every speed table is a
> **floor**, not a fair benchmark. The "clocks fixed" column is what you get after
> `sudo cpupower frequency-set -g performance` + re-enabling turbo.

---

## TL;DR (5 bullets)

1. **This machine is 4 physical cores, not 8.** It is an `i7-8665U` (Whiskey Lake, 15 W mobile
   part): 4 cores / 8 threads, **AVX2 + FMA only — no AVX-512, no AVX-VNNI, no AMX**, 8 MB L3,
   dual-channel DDR4-2400. Everything in `CONTEXT.md` that assumes "8 CPU cores" over-estimates
   compute throughput by ~2× and under-estimates the memory-bandwidth wall by a lot. **[MEASURED]**
2. **CPU decode speed is a bandwidth equation, not a FLOP equation:**
   `tok/s ≈ effective_GB/s ÷ (weights_GB + KV_read_GB)`. Measured here: **6.37 tok/s** for
   Qwen2.5-Coder-1.5B Q4_K_M (0.99 GB) at 8 threads → **6.3 GB/s effective**; the same box's
   STREAM-triad ceiling is **12.2 GB/s at 4 threads** vs **11.4 at 8 threads** — i.e. the
   default thread count is already ~7% off the optimum because SMT halves your L1/L2. **[MEASURED]**
3. **Everything up to 32B fits in 31 GB. Almost none of it is usable.** 8B Q4_K_M = 4.9 GB →
   ~1.3 tok/s now, ~2.5 tok/s with clocks fixed. 14B Q4 → 0.7 / 1.3. 32B Q4 → 0.3 / 0.6.
   70B needs IQ2_XXS (25.5 GB) and then thrashes. Fitting is not the problem; **speed is**. **[MEASURED + arithmetic]**
4. **Quality cost per quant step (7B, wikitext ppl):** Q6_K +0.07%, Q5_K_M +0.24%,
   Q4_K_M +0.9%, Q3_K_M +4.1%, Q2_K +14.7%. Q4_K_M is the knee. **Q3 and below is where
   code breaks** — the syntax survives, identifier/API recall does not. **[PUBLISHED: llama.cpp PR #1684]**
5. **The exotic paths are dead ends here**: unstructured/2:4 sparsity buys memory but no CPU
   speed (no sparse kernels, no VNNI); BitNet needs a ternary-*trained* model that does not
   exist at 8B; low-rank (SliceGPT/SVD-LLM) works but **rotates the residual-stream basis and
   would destroy the interpretability we are building the project for**. The one real,
   free win left is **n-gram self-speculative decoding** (`--spec-type ngram-simple`), which
   is ~1.5–2× on code-rewriting tasks at zero memory cost. **[PUBLISHED + ESTIMATE]**

---

## Findings

### F1. The hardware is not what the brief says: 4 physical cores, no wide SIMD, no turbo  [MEASURED]

```
$ lscpu | head -12
Model name:  Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
CPU(s):      8      Thread(s) per core: 2      Core(s) per socket: 4
L3 cache:    8 MiB     Flags: ... avx2 fma bmi2 ... (no avx512*, no amx*, no avx_vnni)
$ cat /sys/devices/system/cpu/intel_pstate/no_turbo   -> 1
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor -> powersave
$ cat .../scaling_cur_freq                             -> 899966  (0.90 GHz, all 8 threads)
```

What this means, concretely:

| property | this box | a "real" 8-core desktop | consequence |
|---|---|---|---|
| physical cores | **4** (8 SMT threads) | 8 | SMT threads share an FP unit and an L1/L2 — they add ~0% for GEMM and +sync overhead |
| SIMD | AVX2 (256-bit) | AVX-512 (Zen4/Spr+) or AMX (SPR) | AVX-512 typically 1.3–1.6× on quantized GEMM; AMX int8 another 2–4× **[ESTIMATE, order-of-magnitude from vendor data]** |
| int8 dot | `vpmaddwd`/`vpmaddubsw` only | `vpdpbusd` (VNNI) | Q4_K/Q8_0 dequant+dot is ~1.5–2× slower per byte here |
| memory | DDR4-2400 dual-ch, ~38 GB/s peak | DDR5-5200, ~83 GB/s peak | **this is the binding constraint** |
| effective BW | **7.3 / 10.6 / 12.2 / 11.4 GB/s** at 1/2/4/8 threads **[MEASURED, triad]** | ~50–55 GB/s | 4 threads is the optimum; 8 is a regression |
| TDP | 15 W | 65–170 W | sustained all-core clock is nowhere near the 1.9 GHz base |
| disk | 2.4 GB/s O_DIRECT seq read **[MEASURED, 3 GB dd]** | 3–7 GB/s NVMe | irrelevant unless you stream weights (F10) |

**Why it matters:** the project's own budget line ("8 CPU cores, 31 GB") was written from
`nproc`. `nproc` counts SMT threads. Any plan that assumes 8 cores of throughput is off by
~2×, and any plan that assumes AVX-512/AMX throughput is off by another 1.5–4×.
**Confidence: certain.** Cost: free to fix the *description*; fixing the *clocks* needs root.

> Actionable aside: `sudo sh -c 'echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo'`
> and `sudo cpupower frequency-set -g performance` is worth roughly **2×** on this box
> (0.9 → 1.9 GHz base, higher on burst). It is the single highest-leverage change available
> and it costs nothing but a password.

### F2. Fit table — exact weight footprint per quant level  [MEASURED sizes + arithmetic]

Arithmetic used throughout:
`bytes = params × bpw_eff / 8`, where `bpw_eff` is the **effective** bits-per-weight including
the block scales, the higher-precision `output.weight`, the embeddings, and GGUF alignment.
Nominal bpw (from the block structs, Kawrakow's own numbers in llama.cpp PR #1684):
`Q2_K 2.5625, Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625, Q8_0 8.5, IQ2_XXS 2.0625`.
Effective bpw is **higher** because Qwen's 152 064-token vocabulary puts 14% of a 7B model's
parameters in `token_embd`+`output`, which are quantized less aggressively. I calibrated
`bpw_eff` against real published GGUF file sizes and it reproduces them to ~1%:

| check | predicted | real file (bartowski GGUF) |
|---|---|---|
| Qwen2.5-7B Q4_K_M | 4.6 GB | 4.68 GB |
| Qwen2.5-14B Q4_K_M | 8.9 GB | 8.99 GB |
| Qwen2.5-32B Q4_K_M | 19.9 GB | 19.85 GB |
| Qwen2.5-32B IQ2_XXS | 9.0 GB | 9.03 GB |
| Qwen2.5-32B Q8_0 | 34.8 GB | 34.82 GB |
| Qwen2.5-Coder-1.5B Q4_K_M | 0.93 GB | 0.99 GB (local `ollama list`) |

**Weight footprint (GB, decimal). Bold = fits comfortably (≤20 GB). *Italic* = fits only with
small ctx and nothing else running (20–27 GB). Plain = does not fit (>27 GB of a 31 GB box).**

| model | params | fp32 | fp16/bf16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q3_K_M | Q2_K | IQ2_XXS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B · Qwen2.5-Coder-1.5B | 1.54B | **6.2** | **3.1** | **1.6** | **1.3** | **1.1** | **1.0** | **0.8** | **0.6** | **0.4** |
| 3B · Qwen2.5-3B | 3.09B | **12.4** | **6.2** | **3.3** | **2.5** | **2.2** | **2.0** | **1.5** | **1.2** | **0.8** |
| 7B · Qwen2.5-7B | 7.62B | 30.5 | **15.2** | **8.1** | **6.2** | **5.4** | **4.7** | **3.8** | **3.0** | **2.4** |
| 8B · Llama-3.1-8B | 8.03B | 32.1 | **16.1** | **8.5** | **6.6** | **5.7** | **4.9** | **4.0** | **3.1** | **2.2** |
| 14B · Qwen2.5-14B | 14.7B | 58.8 | 29.6 | **15.7** | **12.1** | **10.5** | **9.0** | **7.3** | **5.8** | **4.9** |
| 32B · Qwen2.5-32B | 32.8B | 131 | 65.6 | 34.8 | *26.9* | *23.3* | **19.9** | **15.9** | **12.3** | **9.0** |
| 70B · Llama-3.1-70B | 70.6B | 282 | 145 | 77.2 | 59.6 | 51.6 | 47.4 | 37.7 | 29.8 | *25.5* |

Notes that matter more than the table:
- **fp16 does not fit for anything ≥14B.** 14B fp16 is 29.6 GB — that is the entire machine
  with no OS, no KV cache, no page cache. This is why D2 ("run on the original bf16
  safetensors") is only viable at 1.5B, and A1's GGUF→fp16 extraction must be done
  **tensor-by-tensor / layer-streamed**, never whole-model.
- **Q4_K_M is the only sane operating point above 3B.** Q5_K_M costs +16% memory to recover
  0.65 of the 0.91 ppl points Q4_K_M loses; Q6_K costs +35% to recover 0.83 of them. On a
  bandwidth-bound box both are pure speed losses for a difference no benchmark can see.
- **70B never fits.** Even IQ2_XXS at 25.5 GB leaves 5.5 GB for OS + KV; at 4k ctx the KV is
  1.34 GB, so 26.8 GB of 31 GB — plus page cache for the mmap, plus the other agents. It will
  page-fault continuously and fall to <0.2 tok/s. Treat 70B as **not available**.
- `unsloth`/`bartowski` dynamic quants (`Q4_K_XL`, `IQ4_XS`, `Q3_K_L`) are within ±5% of the
  canonical sizes above; they change *which* tensors get extra bits, not the budget.

### F3. KV cache — the thing that actually kills long-context CPU runs  [arithmetic, configs verified]

Formula (this is the whole thing):

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × ctx × bytes_per_element
           ^ K and V          ^ GQA: KV heads, not query heads
```

Per-token cost is the number to memorise, because it is what you multiply by context:

| model | layers | kv heads | head_dim | **KV per token (f16)** | at 32k ctx (f16) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 28 | 2 | 128 | **28 KiB** | 0.94 GB |
| Qwen2.5-7B | 28 | 4 | 128 | **56 KiB** | 1.88 GB |
| Qwen2.5-14B | 48 | 8 | 128 | **192 KiB** | 6.44 GB |
| Qwen2.5-32B | 64 | 8 | 128 | **256 KiB** | 8.59 GB |
| Qwen2.5-72B | 80 | 8 | 128 | **320 KiB** | 10.74 GB |

(Configs read from each model's `config.json`: 7B = 28L/4KV/3584h, 14B = 48L/8KV/5120h,
32B = 64L/8KV/5120h, all `head_dim = 128`.)

Absolute KV cache size, the three cache precisions llama.cpp actually supports
(`-ctk/-ctv` accept `f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
— verified in `common/arg.cpp`):

| model | ctx | f16 | q8_0 | q4_0 |
|---|---|---|---|---|
| Qwen2.5-7B | 4k | 0.23 GB | 0.12 GB | 0.07 GB |
| Qwen2.5-7B | 8k | 0.47 GB | 0.25 GB | 0.13 GB |
| Qwen2.5-7B | 32k | 1.88 GB | 1.00 GB | 0.53 GB |
| Qwen2.5-14B | 4k | 0.81 GB | 0.43 GB | 0.23 GB |
| Qwen2.5-14B | 8k | 1.61 GB | 0.86 GB | 0.45 GB |
| Qwen2.5-14B | 32k | 6.44 GB | 3.42 GB | 1.81 GB |
| Qwen2.5-32B | 4k | 1.07 GB | 0.57 GB | 0.30 GB |
| Qwen2.5-32B | 8k | 2.15 GB | 1.14 GB | 0.60 GB |
| Qwen2.5-32B | 32k | 8.59 GB | 4.56 GB | 2.42 GB |
| Qwen2.5-72B | 4k | 1.34 GB | 0.71 GB | 0.38 GB |
| Qwen2.5-72B | 32k | 10.74 GB | 5.70 GB | 3.02 GB |

Three consequences that are easy to get wrong:

1. **llama.cpp preallocates the KV cache for the full `-c` at load time.** `-c 32768` on a
   32B model reserves 8.59 GB *before you type anything*, even for a 10-token prompt. This is
   the #1 cause of "it OOMs at load but the model is only 20 GB". Set `-c` to what you need.
2. **Decode reads the whole KV cache every token.** At 32k ctx, a 32B model reads
   8.59 GB of KV *on top of* 19.9 GB of weights → **+43% bytes/token vs the same model with no
   cache, i.e. −30% tok/s**. At 4k it is +5%. Long context is not free on CPU; it is a direct
   speed tax, and it compounds with prefill being quadratic.
3. **q8_0 KV is a near-free win** (half the memory, no measurable quality cost in the
   literature for K; V is slightly more sensitive). q4_0 KV is where you start to see
   degradation — use it only to make 32k fit, never as a default.
   **Confidence: high** on the arithmetic; **medium** on "q4_0 V is where quality drops"
   (widely reported, I did not re-measure it).

### F4. llama.cpp CPU reality — what actually changes the number  [MEASURED + source-verified]

**The one real measurement I got on this box** (before the external drive holding the GGUF
was unmounted): `qwen2.5-coder:1.5b-base-q4_K_M`, 0.99 GB, `llama-server`, `-c 4096`,
**default thread count (= 8 logical CPUs)**, greedy, load average 13.9, 0.90 GHz:

```
prompt_n 1  |  predicted_n 64  |  predicted_per_token_ms 157.0  |  6.37 tok/s
```

Reproduce with `curl localhost:PORT/v1/completions -d '{"prompt":"def fibonacci(n):",
"max_tokens":64,"temperature":0}'` and read the `timings` object. That is the cleanest
CPU benchmark interface llama.cpp exposes — use it, not wall-clock around a chat client.

Cross-check against a published benchmark on real desktop silicon, same PR #1684, 7B Q4_K_S
(3.56 GB) at 4 threads on a Ryzen 7950X: **68 ms/token = 14.7 tok/s = 52 GB/s effective**.
That is DDR5-5200 bandwidth-bound, and it confirms the model: `tok/s ≈ BW ÷ bytes`.
Ours is 6.3 GB/s, i.e. **8× less memory throughput** than a desktop — that 8× is the whole
story, and only ~2× of it is recoverable by fixing clocks.

| lever | does what | verdict for this box |
|---|---|---|
| **thread count** | llama.cpp's default (`n_threads = -1`) is resolved by `postprocess_cpu_params()` to **`common_cpu_get_num_math()`**, which on non-hybrid x86-64 Linux is `sysconf(_SC_NPROCESSORS_ONLN)` — i.e. **all 8 logical CPUs** (verified in `common/common.cpp`, 2026 master). My 6.37 tok/s was therefore measured at **8 threads**, the default, not at the optimum. | **Use `-t 4`.** Measured triad: 4 threads 12.23 GB/s, 8 threads 11.41 GB/s — SMT costs ~7% and adds jitter. Never leave it on the default. |
| **AVX-512 / AMX** | 1.3–1.6× / 2–4× on quantized GEMM | **Unavailable.** Ollama ships `libggml-cpu-alderlake.so` but the ISA check will reject it on Whiskey Lake. |
| **`--mlock`** | `mlock(2)` the weights so the kernel can never evict them | **Use it for any model ≥12 GB.** Without it, a burst of other processes pushes weight pages out and your next token stalls on a disk read (2.4 GB/s → 20 ms/token extra *per evicted MB*). |
| **`--no-mmap`** | read weights into anonymous memory instead of mapping the file | Use with `--mlock` when the model fits: guarantees no page faults, costs ~2× transient RAM during load. Keep **mmap on** when the model + KV leaves <4 GB headroom (32B), so the kernel can reclaim instead of OOM-killing. |
| **`-fa on`** (flash attention) | fuses the attention matmuls, avoids materialising the score matrix | **Use it.** Biggest effect at long ctx; also lets `-ctk/-ctv` quantized caches work well. **[PUBLISHED, llama.cpp has CPU FA kernels since 2025]** |
| **quant repacking** (`Q4_0_4_4`/`4_8`/`8_8`) | interleaved weight layout so SIMD loads are contiguous | **Unverified on x86 AVX2.** The measured gains in the repo are ARM/NEON. Do not budget for it. **[UNVERIFIED]** |
| **`-b 512 -ub 512`** | prefill batch size | Keep at 512. Larger ubatch speeds prefill but spikes RSS; on a 31 GB box with a 20 GB model that spike is what OOMs you. |
| **`--n-gpu-layers 0`** | forces all layers to CPU | Correct and required (default when no GPU), but it is not an optimisation — it is the only option. |

### F5. Quality cost per quant level — where it breaks  [PUBLISHED]

Canonical table, LLaMA-7B/13B, wikitext-2 perplexity, from Kawrakow's quantization PR
(llama.cpp PR #1684, "llama.cpp: 2-6 bit quantization in super-blocks"):

| quant | 7B ppl | Δ vs fp16 | 7B file | 13B ppl | Δ vs fp16 |
|---|---|---|---|---|---|
| fp16 | 5.9066 | — | 13.0 G | 5.2543 | — |
| Q6_K | 5.9110 | **+0.07%** | 5.15 G | 5.2568 | +0.05% |
| Q5_K_M | 5.9208 | +0.24% | 4.45 G | 5.2638 | +0.18% |
| Q5_K_S | 5.9419 | +0.60% | 4.33 G | 5.2785 | +0.46% |
| Q4_K_M | 5.9601 | +0.91% | 3.80 G | 5.3002 | +0.87% |
| Q4_K_S | 6.0215 | +1.94% | 3.56 G | 5.3404 | +1.64% |
| Q3_K_M | 6.1503 | **+4.13%** | 3.06 G | 5.4498 | +3.72% |
| Q3_K_S | 6.4571 | +9.32% | 2.75 G | 5.6033 | +6.64% |
| Q2_K | 6.7764 | **+14.7%** | 2.67 G | 5.8545 | +11.4% |

Reading this honestly:
- **Q6_K is free** (+0.07% ppl) but costs +35% memory. On a bandwidth-bound box that is
  a 35% speed loss for a number nobody can measure. Only use Q6_K for a *reference* run.
- **Q4_K_M is the knee.** +0.9% ppl for 30% of fp16's size. This is the default for a reason.
- **Q3_K_M at +4% ppl is where instruction-following visibly degrades** and, in my
  experience, code generation starts emitting plausible-looking calls to APIs that do not
  exist. Perplexity understates the damage on code because code is low-entropy: the model
  still assigns high probability to *syntactically* correct continuations.
- **Q2_K (+15% ppl) is a different, dumber model.** IQ2_XXS is *better than Q2_K at the same
  size* on big models (imatrix-driven), and is the only defensible 2-bit choice — but only at
  70B, where there is enough redundancy. At 8B, 2-bit is not a compression of the model, it is
  a replacement for it.
- **Independent confirmation that low-bit hurts reasoning/code specifically** (not just ppl):
  "An empirical study of LLaMA3 quantization: from LLMs to MLLMs" (Huang et al., 2024,
  arXiv:2404.14047) reports that at 2–3 bit, HumanEval/MBPP and reasoning scores fall far
  faster than perplexity does. **Confidence: high.**

**Rule for this project:** if a downstream result depends on *which token the model chose*,
Q4_K_M is the floor and any A/B must be run at the same quant. Never compare a Q4 result to
an fp16 result and call the difference "the model's behaviour".

### F6. Unstructured vs structured sparsity — memory yes, CPU speed no  [PUBLISHED + honest]

- **SparseGPT** (Frantar & Alistarh, 2023, arXiv:2301.00774): one-shot pruning to 50–60%
  unstructured with "negligible" ppl increase on OPT/BLOOM-175B, no retraining. Generalises
  to 2:4 and 4:8 semi-structured. Crucially, their *speedups* come from a **custom
  sparse-dense CUDA kernel** (SparTA). Without such a kernel you have saved bytes and
  changed nothing about FLOPs.
- **Wanda** (Sun et al., 2023, arXiv:2306.11695): magnitude × activation pruning, no
  retraining, no weight update, beats SparseGPT at 2:4 in their tables. Same kernel caveat.
- **The general statement is in SliceGPT's abstract** (Ashkboos et al., 2024,
  arXiv:2401.15024): existing sparsification methods "need additional data structures and
  offer constrained speedup with current hardware" — that is the authors of a competing
  method saying it plainly.

On **this** CPU: ggml/llama.cpp has **no sparse kernels at all** — a pruned GGUF would be
stored dense and run at identical speed. The only x86 runtime that ever exploited 2:4 was
Neural Magic's DeepSparse, which (a) required AVX-512 VNNI, (b) was **deprecated and EOL'd
on 2025-06-02** after the Red Hat acquisition (per its own README). And our CPU has no VNNI.

**Verdict: reject.** Unstructured sparsity on CPU is a memory trick that requires a custom
kernel to become a speed trick, and we have neither the kernel nor the ISA. **Confidence: high.**

### F7. Low-rank compression — one works, and it breaks the thing we care about  [PUBLISHED]

| method | what it does | post-hoc? | real CPU speedup? | fatal flaw for us |
|---|---|---|---|---|
| **SliceGPT** (arXiv:2401.15024) | deletes rows/columns: replaces each weight with a **smaller dense** matrix, cuts embedding dim by up to 25% on Llama-2-70B | yes, but needs calibration data + computes a per-layer rotation | **yes** — dense smaller matrices are hardware-friendly, and the paper reports speedups on GPU/CPU | **it applies an orthogonal rotation to the residual stream.** Every direction we read (logit lens, steering vectors, SAE features) is in the *old* basis. Slicing silently invalidates all of A2/A3's artifacts. |
| **SVD-LLM** (arXiv:2403.07378) | truncation-aware whitened SVD + iterative weight update | yes, needs calibration set | yes (dense) | same basis problem for the decomposed matrices; quality at 20–30% compression is modest |
| **ASVD** (arXiv:2312.05821) | activation-aware SVD, **training-free**, handles activation outliers by transforming them away | yes, training-free | yes (dense) | still changes each weight matrix's basis; per-layer rank choice is blind to what the layer *does* |
| **Palu** (arXiv:2407.21118) | low-rank **KV-cache** projection, not weights | yes, needs calibration | no — needs custom kernels (GPU-oriented) | solves a problem we do not have (our KV is 0.06–0.5 GB); no CPU kernels |

**The honest read:** low-rank is the only post-hoc family that gives a *real* CPU speedup,
because it produces dense smaller matrices rather than sparse ones. But every member of it
works by changing the coordinate system of the weights, which is exactly the coordinate
system the entire mod_wei project is trying to read. Adopting SliceGPT would mean re-deriving
every steering vector and every SAE after compression — which is precisely the experiment A8
should run deliberately, not a shortcut to run 32B.

**Confidence: high** on the mechanism; **medium** on "25% for 70B transfers to 8B" — smaller
models are much less redundant, and SliceGPT's own numbers degrade sharply below 10B.

### F8. Ternary / 1.58-bit — real, but it is a *training* method, not a quantization  [PUBLISHED]

- **BitNet b1.58** (Ma et al., 2024, arXiv:2402.17764): every weight ∈ {−1, 0, +1}. It
  "matches the full-precision Transformer LLM **with the same model size and training
  tokens**" — i.e. the comparison is *trained-ternary vs trained-fp16*. It is a new
  training recipe and a new scaling law, not a post-hoc conversion.
- **bitnet.cpp** (Microsoft, arXiv:2410.16144 and arXiv:2502.11880): the real CPU kernels.
  Verified from the repo README: **2.37×–6.17× speedup on x86** vs full precision,
  **71.9%–82.2% energy reduction**, and "a 100B BitNet b1.58 model on a single CPU at
  5–7 tokens/s". That is a genuinely large, real CPU win.
- **The catch, stated plainly:** the speedup is *lossless* only because the model was
  trained ternary. There is exactly one official model — `microsoft/BitNet-b1.58-2B-4T`
  (2.4B params, 4T tokens). There is **no ternary 8B/14B/32B**, and no way to make one on
  8 CPU cores. Post-hoc ternary quantization of an fp16 model collapses (it is a 1.58-bit
  round-to-nearest with no learned scales — the error is ~5–10× Q2_K's).

**Verdict: the most impressive CPU result in the literature, and completely inapplicable
here.** The only thing to take from it is the *direction*: if you ever want 70B-class
behaviour on a CPU, it will come from a model trained for it, not from compressing a big
fp16 one. **Confidence: high.** Cost to adopt: a training run we cannot do.

### F9. Speculative decoding on CPU — yes, but only self-speculation is free  [PUBLISHED + ESTIMATE]

The algorithm (Leviathan, Kalman & Matias, 2022, arXiv:2211.17192) is exact — it does not
change the output distribution. The economics on a CPU are different from a GPU because
**both models compete for the same bandwidth**. Model it:

```
speedup = E[accepted tokens per target pass] / (1 + k · c)
   c = draft_bytes / target_bytes        (draft cost as a fraction of one target pass)
   k = draft length;  α = per-token acceptance
   E[accepted] = (1 − α^(k+1)) / (1 − α)
```

With an 8B target (4.9 GB) and a 0.5B draft (0.35 GB), `c = 0.07`, `k = 4` draft tokens
(cost = 1 + 4×0.07 = **1.28 target-passes per step**):

| α (per-token acceptance) | E[accepted] | speedup | when you get this |
|---|---|---|---|
| 0.9 | 4.10 | **3.2×** | re-emitting code it was just shown (near-copy) |
| 0.7 | 2.77 | **2.2×** | tight code continuation, same file |
| 0.4 | 1.65 | **1.29×** | open-ended prose / novel reasoning |

**[ESTIMATE — the arithmetic is exact, the acceptance rates are priors, and 3.2× is an upper
bound: it ignores the draft model's per-layer overhead and the fact that the verify pass is a
*batch* pass, which on a bandwidth-bound CPU costs slightly more than one decode step. Real
measured CPU gains in the literature are 1.3–2×, not 3×.]** The rule that follows: **the
draft must be ≥10× smaller than the target**, otherwise drafting eats the win. A 1.5B draft
for a 14B target (`c = 0.11`) is still fine; a 7B draft for a 14B target is pointless.

**The free version, which is the actually-useful one here:** llama.cpp's **n-gram self-
speculation** needs *no second model at all* — it drafts from the token history
(`docs/speculative.md`, implementations `ngram-simple`, `ngram-map-k`, `ngram-cache`).
Its own docs name the use case: "the rewriting of source code by an LLM". For a code model
echoing identifiers, docstrings, or re-emitting a file it was shown, acceptance is very high
and the draft cost is ~0. This is **the single best speed lever available on this box** for
the kind of workload we have (patch/reformat/explain-existing-code), and it costs zero bytes
of RAM. It does nothing for open-ended generation — be honest about that.
**Confidence: high** that it exists and is free; **medium** on the magnitude (task-dependent,
0–3×).

### F10. Weight streaming from SSD — the throughput math says no  [MEASURED disk + arithmetic]

Measured sequential read on this box: **2.4 GB/s** (3 GB O_DIRECT `dd`, cold).

```
tok/s_streaming ≈ SSD_GB/s ÷ bytes_read_per_token
32B Q4_K_M : 19.85 GB → 8.3 s/token  (0.12 tok/s)
70B IQ2_XXS: 25.5  GB → 10.6 s/token (0.09 tok/s)
```

FlexGen (Sheng et al., 2023, arXiv:2303.06865) is the reference result and it does *not*
rescue this: it ran OPT-175B on a single 24 GB GPU by aggregating GPU+CPU+disk, but it is a
**throughput** system — it targets batched, latency-insensitive jobs, and its headline is
~1 tok/s for 175B in the *best* configuration, achieved with 4-bit weights, a large GPU-side
KV cache, and batching. It is not interactive single-stream decoding.

Two more reasons it fails here specifically:
1. llama.cpp has **no disk-offload tier**. `--n-gpu-layers` moves layers to a *GPU*; with no
   GPU there is nowhere to offload *to*. What you get instead is mmap + page-cache thrash,
   which is the pathological version of streaming (unpredictable 20 ms stalls, no prefetch).
2. The failure mode is not "slow", it is "**stalls mid-sentence**". A token that needs a
   layer that was evicted blocks on 4 KB random reads. Interactive use is not viable.

**Verdict: no.** Only relevant as an offline batch job, and even then FlexGen-class
engineering (prefetch scheduling, LP-solved tensor placement) would be needed, which does not
exist in any CPU-only tool. **Confidence: high.**

### F11. The practical answer for this box

Speed model (all rows assume 4 threads, Q4-class weights, `-c 4096`, q8_0 KV, and
`tok/s = effective_BW ÷ (weights + KV_read)`. The MEASURED row was taken at the default
8 threads, so it is ~7% pessimistic even before the clock issue):

| config | weights | +KV@4k | **now (0.9 GHz, contended)** | **clocks fixed (1.9 GHz)** | interactive? |
|---|---|---|---|---|---|
| 1.5B Q4_K_M | 1.0 GB | 0.06 | **6.4 tok/s** [MEASURED] | ~12 tok/s | yes, comfortably |
| 7B Q4_K_M | 4.7 GB | 0.12 | **1.35 tok/s** | ~2.5 tok/s | marginal (like reading slowly) |
| 8B Q4_K_M | 4.9 GB | 0.12 | **1.3 tok/s** | ~2.5 tok/s | marginal |
| 14B Q4_K_M | 9.0 GB | 0.43 | **0.70 tok/s** | ~1.3 tok/s | no — batch only |
| 32B Q3_K_M | 15.9 GB | 0.57 | **0.40 tok/s** | ~0.75 tok/s | no |
| 32B Q4_K_M | 19.9 GB | 0.57 | **0.32 tok/s** | ~0.60 tok/s | no — 100 tokens = 5 min |
| 70B IQ2_XXS | 25.5 GB | 0.71 | **0.25 tok/s** + thrash | ~0.5 tok/s | no |

Reference point for "what 8 cores normally means": a Ryzen 7950X (16C, DDR5) does 7B Q4 at
**14.7 tok/s** on 4 threads. We are 8× off that, and only 2× of the gap is the clock.

**Recommendation for THIS box:**
- **8B:** yes, `Qwen2.5-Coder-7B-Instruct Q4_K_M` (4.68 GB). It fits with 25 GB to spare, so
  you can run it at `-c 8192` with f16 KV and still keep the page cache warm. Expect
  **1.3 tok/s now, ~2.5 tok/s after the clock fix.** Acceptable for "generate one function
  and read it", not for a chat loop.
- **14B:** only as an offline batch tool. 0.7 tok/s means a 500-token answer takes 12 minutes.
- **32B:** technically fits at Q3_K_M/Q4_K_M, practically unusable (0.3–0.4 tok/s). If you
  must, use Q3_K_M at `-c 2048` and treat it as a "run overnight" resource.
- **70B:** do not.
- **For the project itself (interpretability):** stay at 1.5B. The value of mod_wei is
  hooks + logit lens + steering, and every one of those needs *many* forward passes per
  experiment (a patching sweep is 28 layers × N positions forwards). At 1.5B you get ~6
  tok/s and a forward pass in ~150 ms; at 8B a single forward is ~800 ms and a 28-layer
  patching sweep becomes 20+ minutes. **Bigger models make the science slower, not better.**

---

## Concrete implementation

### 1. llama.cpp server, tuned for this box

```bash
# 8B — the realistic upper bound for interactive use
llama-server -m Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  -c 8192 -t 4 -fa on \
  -ctk q8_0 -ctv q8_0 \
  --mlock --no-mmap \
  -b 512 -ub 512 \
  --host 127.0.0.1 --port 8080

# 32B — batch only. Note -c 4096: KV is preallocated, 32k would reserve 8.6 GB.
llama-server -m Qwen2.5-32B-Instruct-Q3_K_M.gguf \
  -c 4096 -t 4 -fa on -ctk q8_0 -ctv q8_0 \
  -b 256 -ub 256 --host 127.0.0.1 --port 8081
#   (no --no-mmap here: leave the kernel able to reclaim pages instead of OOM-killing)

# free speed on code-rewrite tasks: self-speculation from the token history
llama-server -m ... --spec-type ngram-simple --spec-draft-n-max 64
```

Benchmark correctly — use the server's own timings, not wall clock:

```bash
curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"def quicksort(a):","max_tokens":128,"temperature":0}' \
| python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"];
print(f"prefill {t[\"prompt_per_second\"]:.1f} tok/s  decode {t[\"predicted_per_second\"]:.2f} tok/s")'
```

### 2. Ollama equivalents (it is a behavioural baseline only, per D1)

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1      # never run 2 models at once on 31 GB
export OLLAMA_KEEP_ALIVE=10m
ollama run qwen2.5-coder:7b-instruct-q4_K_M
```

Per-request overrides go in `options` (the `num_ctx` default is **4096** — verify with
`ollama ps`, which prints the CONTEXT column):

```bash
curl -s localhost:11434/api/generate -d '{
  "model":"qwen2.5-coder:7b-instruct-q4_K_M",
  "prompt":"def fib(n):", "stream":false,
  "options":{"num_thread":4,"num_ctx":8192,"num_batch":512,"num_predict":128}
}' | python3 -c 'import json,sys;d=json.load(sys.stdin);
print(d["eval_count"]/(d["eval_duration"]/1e9),"tok/s")'
```

### 3. The calculator that should live in `code/scale.py`

Every number in this document is reproducible from these 30 lines. Run it before you
download anything.

```python
# code/scale.py — footprint, KV, and speed for a model on THIS box.
BPW_EFF = {"fp32":32.0,"fp16":16.0,"Q8_0":8.5,"Q6_K":6.55,"Q5_K_M":5.70,
           "Q4_K_M":4.85,"Q3_K_M":3.95,"Q2_K":3.05,"IQ2_XXS":2.20}   # effective, calibrated
TOTAL_RAM_GB   = 31.0
OS_RESERVE_GB  = 3.0
EFF_BW_GB_S    = 6.3    # MEASURED on this box (0.9 GHz, contended)
EFF_BW_FIXED   = 12.0   # triad ceiling at 4 threads; the wall after fixing clocks

def weights_gb(params_b, quant):
    return params_b * 1e9 * BPW_EFF[quant] / 8 / 1e9

def kv_gb(n_layers, n_kv_heads, head_dim, ctx, bytes_per_elem):
    return 2 * n_layers * n_kv_heads * head_dim * ctx * bytes_per_elem / 1e9

def report(name, params_b, L, H, D, quant, ctx=4096, kv_bytes=1.0):
    w = weights_gb(params_b, quant)
    kv = kv_gb(L, H, D, ctx, kv_bytes)
    total = w + kv
    budget = TOTAL_RAM_GB - OS_RESERVE_GB
    fits = "FITS" if total <= budget else "NO FIT"
    tps_now   = EFF_BW_GB_S  / total
    tps_fixed = EFF_BW_FIXED / total
    print(f"{name:34s} {quant:8s} w={w:5.1f} kv={kv:5.2f} "
          f"tot={total:5.1f}/{budget:.0f}GB {fits:6s} "
          f"{tps_now:5.2f} tok/s now / {tps_fixed:5.2f} fixed")

#             name                              params  L   H   D   quant
report("Qwen2.5-Coder-1.5B",                  1.54, 28,  2, 128, "Q4_K_M")
report("Qwen2.5-7B",                          7.62, 28,  4, 128, "Q4_K_M", ctx=8192)
report("Llama-3.1-8B",                        8.03, 32,  8, 128, "Q4_K_M", ctx=8192)
report("Qwen2.5-14B",                        14.70, 48,  8, 128, "Q4_K_M")
report("Qwen2.5-32B",                        32.80, 64,  8, 128, "Q3_K_M", ctx=4096, kv_bytes=1.0625)
report("Qwen2.5-32B",                        32.80, 64,  8, 128, "Q4_K_M", ctx=4096, kv_bytes=1.0625)
# 70B/72B: formula under-predicts here — bartowski's real Qwen2.5-72B IQ2_XXS file is 25.5 GB
# (2.80 effective bpw, not 2.20). Use the measured size; it does not fit either way.
report("Qwen2.5-72B",                        72.70, 80,  8, 128, "IQ2_XXS", ctx=4096, kv_bytes=1.0625)
```

(`kv_bytes`: f16 = 2.0, q8_0 = 1.0625, q4_0 = 0.5625 — q8_0 costs 1 byte/weight + a 16-bit
block scale per 32 values.)

### 4. Ordering of the levers, by measured payoff per unit effort

| # | lever | payoff | effort | blocked on |
|---|---|---|---|---|
| 1 | `no_turbo=0` + `governor=performance` | **~2×** | 1 min | **root password** |
| 2 | `-t 4` instead of 8 | ~7% | free | — |
| 3 | `-fa on` + `-ctk/-ctv q8_0` | 5–20%, and halves KV | free | — |
| 4 | `-c` sized to the actual task | prevents OOM; large speed effect at long ctx | free | — |
| 5 | `--spec-type ngram-simple` | 0–3× on code-rewrite tasks | free | task-dependent |
| 6 | draft model (1.5B → 14B) | ~1.5–2× **[ESTIMATE]** | medium | needs a 2nd model in RAM |
| 7 | Q4_K_M instead of Q6_K | 35% | free | +0.07% ppl |
| 8 | anything from F6/F7/F8 | 0 on this box | weeks | no |

---

## What NOT to do (traps)

1. **Do not trust `nproc`.** It said 8; the machine has 4 physical cores. Any capacity plan
   built on `nproc` is wrong by 2×. `lscpu | grep "Core(s) per socket"` is the number.
2. **Do not leave the thread count on the default.** It is 8 logical CPUs (verified in
   `common.cpp`), and SMT threads on a 15 W part make quantized GEMM *slower* (measured
   12.23 → 11.41 GB/s) because they halve L1/L2 per thread and add barrier cost. Pass `-t 4`.
3. **Do not set `-c 32768` "just in case".** KV is preallocated: 8.6 GB on a 32B model before
   a single token. Size it to the task.
4. **Do not use Q2_K/Q3_K on an 8B and expect the same model.** +15%/+4% ppl understates it;
   code loses API-name recall first. Q4_K_M is the floor for any behavioural claim.
5. **Do not compare a quantized run to an fp16 run.** That difference is the quantizer, not
   the phenomenon you are studying.
6. **Do not download a 32B fp16 or a 70B anything.** 65 GB and 47–145 GB respectively; 293 GB
   of disk makes it *possible*, which is exactly the trap.
7. **Do not expect `--mlock` to save you on a 20 GB model with 31 GB of RAM and other
   processes running.** mlock on a model that does not fit converts an OOM-kill into a
   machine-wide stall. Check `free -g` first.
8. **Do not pursue unstructured sparsity for speed.** No ggml kernels, no VNNI, and the only
   x86 runtime that did it (DeepSparse) was EOL'd in June 2025.
9. **Do not "just quantize to 1.58-bit".** BitNet's numbers require a ternary-trained model.
   Post-hoc ternary of an fp16 checkpoint is a destroyed model, not a small one.
10. **Do not adopt SliceGPT/SVD-LLM casually.** They rotate the weight basis. Every steering
    vector and SAE feature A2/A3 produce would be expressed in a coordinate system that no
    longer exists after compression. This is a deliberate experiment (A8's job), not a
    deployment shortcut.
11. **Do not benchmark while other agents are running.** My 6.37 tok/s was measured at load
    average 13.9. Re-measure on an idle box before believing any absolute number here.

---

## Open questions for other agents

- **A1 (formats):** F2 says a 14B fp16 is 29.6 GB — the whole machine. Can your
  GGUF→safetensors extractor work **per-tensor / per-layer streaming** (dequantize layer *i*,
  hand it to the harness, free it) rather than materialising the model? If not, D2's "run on
  original bf16" is only ever possible at 1.5B. This is the single most important constraint
  I can hand you.
- **A4 (runtime):** what is a *forward pass* on this box, not a token? A 1.5B bf16 forward in
  PyTorch is ~3 GFLOP; at the ~20 GFLOP/s this CPU actually delivers that is ~150 ms/pass,
  and a 28-layer patching sweep is 28× that. Does your architecture batch the sweep, or does
  it need 28 sequential forwards? That number decides whether interpretability at 8B is
  feasible at all.
- **A8 (guided compression):** F7 is your opening. SliceGPT/SVD-LLM/SparseGPT all pick ranks
  or masks *blind to function* — using calibration activations only. If per-layer sensitivity
  measured on *our* harness (does the logit lens survive? does the steering dose-response
  survive?) picks better ranks than SVD's singular values do, that is a real result. Also:
  can you compress in a basis-preserving way (e.g. per-channel scaling + quant only) so A3's
  vectors stay valid? That is a stronger constraint than perplexity and nobody has published it.
- **A6 (skeptic):** please attack the `tok/s = BW ÷ bytes` model. It assumes every weight is
  read exactly once per token, which is true for batch-1 decode on a bandwidth-bound machine
  but breaks if the working set fits in L3 (8 MB here — nothing does) or if the kernel is
  compute-bound (my 1.5B measurement suggests it partly is: 6.3 GB/s achieved vs 12.2
  available).
- **Orchestrator:** `CONTEXT.md` says "8 CPU cores". It is 4 physical / 8 logical, AVX2-only,
  15 W, currently pinned at 0.9 GHz. Worth correcting, because the 5–20 tok/s estimate for
  1.5B bf16 is only reachable after the governor/turbo fix; measured today is 6.4 tok/s for a
  *quantized* 1.5B, which implies bf16-in-PyTorch will be slower, not faster.

---

## SCHEMA CHANGE

**Proposed: add a `runtime` block to the canonical trace, sibling to `layers`.**

Reason: every quantitative claim in this file is only interpretable with the machine state
that produced it. Two traces taken at 0.9 GHz and 1.9 GHz are not comparable, and a trace
taken at Q4_K_M cannot be compared to one at bf16 (D2). Without this, A6 cannot audit and A5
cannot label the UI honestly.

```json
{"step": 0, "token_id": 123, "token_str": "def",
 "runtime": {"backend": "llama.cpp|torch", "quant": "bf16|Q4_K_M|...",
             "ctx_len": 8192, "n_threads": 4, "clock_ghz": 0.90,
             "tok_per_s": 6.37, "rss_gb": 2.1, "kv_gb": 0.12},
 "layers": [...],
 "logits_topk": [...], "chosen": [id, tok, prob]}
```

Backwards-compatible (additive), and it is the only way to compare a 1.5B-bf16 run against a
7B-Q4_K_M run without lying. If rejected, at minimum log it as trace metadata in a sidecar file.

---

## Build next (ordered, with effort estimate)

| # | task | effort | why |
|---|---|---|---|
| 1 | **Fix the clocks**: `no_turbo=0`, governor `performance`; re-run the 1.5B benchmark and record before/after | 10 min (**needs sudo**) | ~2× on everything, for free. Every later number depends on which state the box is in. |
| 2 | Land `code/scale.py` (sketch above) and run it for every candidate model | 1 h | Stops anyone from downloading a model that cannot work. Cheap insurance. |
| 3 | Install llama.cpp from source (or use ollama's bundled `llama-server`) and establish the **tok/s-per-GB constant** on an idle box with 3 models (1.5B Q4, 7B Q4, 14B Q4) | 3 h | Turns the whole table from ESTIMATE into MEASURED. This is the single most valuable calibration for A8. |
| 4 | Measure `ngram-simple` on a real code-rewrite task (give the model a file, ask it to reformat) | 2 h | The only free speedup left; either it is 2× or it is nothing, and only a measurement will say. |
| 5 | 7B Q4_K_M end-to-end with the canonical trace attached (hooks not available in llama.cpp → this is A4's problem, but the *behavioural* baseline is here) | 1 day | Gives the project a legitimate "big model" reference to compare the 1.5B against. |
| 6 | Decide, with A8, whether basis-preserving compression (per-channel scale + quant, no rotation) can buy 8B→5B. If yes, that is the only route to a big model that keeps our science valid. | 1 week | The real prize. Everything in F6–F8 is a dead end; this is not obviously one. |
| 7 | (Do not do) 32B/70B interactive serving | — | 0.3 tok/s is not a product. Only as an overnight batch. |
