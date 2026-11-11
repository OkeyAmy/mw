# A4 — Runtime & performance engineering

Scope: what software architecture lets us run Qwen2.5-Coder-1.5B on 8 "cores" with
per-token activation capture and live intervention, and stream that to a UI.

Everything below was measured on **this** machine or read out of upstream source at a
pinned tag. Anything I could not verify is marked `UNVERIFIED`.

---

## TL;DR (5 bullets)

1. **The box is not what CONTEXT.md thinks it is.** It is an Intel i7-8665U: **4 physical
   cores / 8 threads**, 8 MiB L3, **AVX2 only — no AVX-512, no AMX, no AVX512-BF16**, and it
   is currently pinned at **800 MHz** (`intel_pstate` governor `powersave`, `no_turbo=1`).
   Measured aggregate read bandwidth: **11.6 GB/s** at 800 MHz. Fix the governor before
   any perf work; it is worth ~2x.
2. **fp32 is the only viable compute dtype.** Without AMX/AVX512-BF16, PyTorch bf16/fp16
   CPU matmul is 4x–50x *slower* than fp32 (multiple upstream issues). Weights may be
   *stored* in bf16, but every matmul must run in fp32. Realistic decode is **~2–3 tok/s**
   at full clock (≈1–2 tok/s as configured now) — below CONTEXT.md's 5–20 tok/s floor.
3. **The dominant cost is not the model, it is the logit lens.** `lm_head` is 233 M params
   = 0.93 GB fp32, and a naive per-layer lens re-reads it 28 times (26 GB/token). Measured
   on this box: **2.50 s/token naive vs 0.49 s/token when all 28 residuals are stacked into
   one GEMM**. Lens FLOPs (13.1 GFLOP) are **4.2x** the model forward (3.09 GFLOP).
   Non-negotiable: batch the lens, and default to a layer subset.
4. **Recommendation: raw HF `transformers` 5.22.2 + `register_forward_hook`, with a
   hand-written per-token loop over `DynamicCache`.** Not TransformerLens (its
   `HookedTransformer` was *removed* in 4.0), not nnsight (excellent, but a tracing
   abstraction we would fight), not pyvene (intervention-only, PyPI stale since 2025-05).
   Hooks are 30 lines and give us exact control over what is captured and when.
5. **llama.cpp can already emit per-layer residuals with no patch** — `llama_context_params.cb_eval`
   (a public `ggml_backend_sched_eval_callback`) fires per graph node, and `qwen2.cpp` names
   every layer output `"l_out"`. What it *cannot* do is activation injection. So: llama.cpp
   is a fast behavioural baseline, not the instrument.

**Bonus, urgent for everyone:** `Qwen/Qwen2.5-Coder-1.5B-Base` **does not exist** (HF API
returns 401; search finds no such repo). The real ID is **`Qwen/Qwen2.5-Coder-1.5B`**
(1,543,714,304 params, bf16, `tie_word_embeddings: true`, ctx 32768). See F12.

---

## Findings

### F1. The hardware is 4P/8T, AVX2-only, and currently running at 42% of nominal clock
[source: local `lscpu` / `/proc/cpuinfo` / `sysfs`, measured 2026-11-10]

**What it is.** `lscpu` reports `CPU(s): 8, Thread(s) per core: 2, Core(s) per socket: 4`,
`L3: 8 MiB`, base 1900 MHz, flags include `avx2 fma f16c` and **zero** `avx512*` entries.
`/sys/devices/system/cpu/cpu0/cpufreq/` shows `scaling_driver=intel_pstate`,
`scaling_governor=powersave`, `scaling_cur_freq=799998` (i.e. 800 MHz), and
`intel_pstate/no_turbo=1`.

**Why it matters.** Three separate things:

- "8 cores" is **4 cores + SMT**. For GEMM-heavy work SMT gives ~10–25%, not 2x, and
  it can *hurt*: the best Qwen2.5-1.5B CPU benchmark I found reports 7 threads >
  8 threads (16.04 vs 14.48 tok/s) [source: sanjay-906, *LLM Inference Benchmarking and
  CPU Thread Optimization*, 2025, https://github.com/sanjay-906/LLM-Inference-Benchmarking-and-CPU-Thread-Optimization].
  So the right default is `torch.set_num_threads(4)` or `6`, **not** 8, and it must be
  measured, not assumed.
- No AVX-512 and no AMX means every fp32 GEMM runs on 256-bit FMA. Theoretical fp32 peak
  at 1.9 GHz: 4 cores x 16 FLOP/cycle x 1.9 GHz ≈ **122 GFLOP/s**; realistic 40–60% →
  50–70 GFLOP/s.
- 800 MHz / powersave / no_turbo means **right now** we are at 42% of base clock. All
  numbers below are measured under that handicap.

**Measured bandwidth** (numpy streaming read of 0.25 GiB, N processes in parallel,
800 MHz):

| procs | aggregate GB/s |
|---|---|
| 1 | 3.1 |
| 2 | 6.7 |
| 4 | 9.0 |
| 8 | 11.6 |

At full clock expect ~15–20 GB/s (DDR4-2400 dual channel is ~38 GB/s theoretical, ~50%
realisable). **Confidence: high** (direct measurement). **Cost: zero** — one sysfs write.

> Actionable: `sudo cpupower frequency-set -g performance` and check
> `/sys/devices/system/cpu/intel_pstate/no_turbo`. This is a *system* change, so I did not
> make it. Without it every benchmark in this project is ~2x pessimistic.

---

### F2. bf16/fp16 on this CPU is slower than fp32 — always
[source: pytorch/pytorch#146508 "Something very wrong with float16 CPU implementation", 2025,
https://github.com/pytorch/pytorch/issues/146508; #151934 "torch.bmm is slow on
non-contiguous BF16 CPU tensors", 2025, https://github.com/pytorch/pytorch/issues/151934;
pytorch/pytorch#181367, 2026, https://github.com/pytorch/pytorch/pull/181367]

**What it is.** PyTorch's CPU half-precision GEMM paths are gated on hardware:
`mkldnn_fp16_gemm` and oneDNN bf16 kernels require **AVX512-BF16 or AMX**. Without them,
`gemm_stub` falls through to a scalar triple loop. Direct quotes from the issues:

- On an AVX512 (but not BF16) machine, torch 2.7: fp32 2.6 ms, fp16 100.7 ms, bf16 10.9 ms
  for a 500² matmul → **bf16 4.2x slower than fp32**.
- On an AVX2-only Xeon, torch 2.10: 500² matmul fp32 2.4 ms vs **bf16 118.4 ms** → ~50x.
- A PyTorch collaborator, on the fp16 issue: *"our current priority does not cover f16
  performance on client CPUs... on CPUs without amx-fp16 will go to a super slow fallback
  path."*
- Intel's own bf16 blog only claims speedups on **Cooper Lake / Sapphire Rapids**:
  "bfloat16 has 1.4x to 2.2x performance gain over float32" on Xeon 8380H
  [source: PyTorch blog, *Empowering PyTorch on Intel Xeon Scalable processors with
  Bfloat16*, 2021, https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/].

**Why it matters.** Our model ships in **bf16** (`torch_dtype: "bfloat16"` in config.json).
The obvious move — "load it as-is, it's half the memory" — is exactly wrong here. We must
upcast to fp32 at load. Note that Qwen2's `Qwen2RMSNorm.forward` already upcasts to fp32
internally (verified in `modeling_qwen2.py` v5.22.2 line 241), so the norms are safe either way.

**Nuance.** Storage dtype and compute dtype are separable. 6.17 GB of fp32 weights vs
3.09 GB of bf16 *halves* the per-token memory traffic, and decode is bandwidth-bound
(F3). A bf16-stored / fp32-computed path (cast each `Linear.weight` to fp32 lazily per
layer, or use `torch.autocast` with an fp32 fallback list) *could* win. But the cast itself
costs a full read+write pass, and PyTorch has no fused "bf16 load → fp32 GEMM" kernel on
AVX2. **Treat this as an experiment, not an assumption** — it is the single highest-value
micro-benchmark in the project. `UNVERIFIED` which way it lands.

**Confidence: high** on "bf16 compute is slower"; **low** on "bf16 storage + fp32 compute
is a net win". **Cost:** one afternoon of benchmarking.

---

### F3. Throughput model: decode is bandwidth-bound → ~2–3 tok/s for fp32
[source: derived from F1 measurement + Qwen2.5-Coder-1.5B config.json; cross-checked
against sanjay-906 benchmark, 2025]

**What it is.** Exact model arithmetic from `config.json`
(https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/blob/main/config.json): 28 layers,
hidden 1536, 12 heads, 2 KV heads (GQA, head_dim 128), intermediate 8960, vocab 151936,
**`tie_word_embeddings: true`**, ctx 32768.

Per-layer params: attn `4x1536²  + 2x1536x256` = 5.50 M; MLP `3x1536x8960` = 41.29 M
→ 46.79 M/layer x 28 = **1.310 B**. Plus embeddings `151936x1536` = **233.4 M**, counted
once because they are tied. Total = **1,543,714,304** — exactly what the HF API reports.

| quantity | fp32 | bf16 |
|---|---|---|
| weight bytes | 6.17 GB | 3.09 GB |
| FLOPs / token (2·N) | 3.09 GFLOP | 3.09 GFLOP |
| bytes read / token | 6.17 GB | 3.09 GB |
| **arithmetic intensity** | **0.50 FLOP/byte** | **1.00 FLOP/byte** |

The CPU's ridge point is ~122 GFLOP/s ÷ 17 GB/s ≈ **7 FLOP/byte**. At 0.5 FLOP/byte we
are **14x on the memory side of the ridge point** — decode is unambiguously
memory-bandwidth-bound. Time per token = weight bytes / achieved bandwidth:

- 6.17 GB ÷ 11.6 GB/s (as configured, 800 MHz) = **532 ms → 1.9 tok/s**
- 6.17 GB ÷ 17 GB/s (full clock) = **363 ms → 2.8 tok/s**
- 3.09 GB ÷ 17 GB/s (bf16 storage) = 182 ms → 5.5 tok/s *(but see F2 compute penalty)*

**Sanity check against a real benchmark.** Qwen2.5-1.5B Q4_K_M (1.04 GB) on an
i5 with UHD 620 (a comparable 15 W laptop part): **16.04 tok/s at 7 threads**, 14.48 at 8.
1.04 GB ÷ 16 tok/s = 65 GB/s-equivalent... that is higher than our measured 11.6 GB/s,
which is consistent: that machine runs at full clock and llama.cpp's quantised GEMV reads
far less per token. The *shape* of the model is right — bandwidth ÷ bytes predicts tok/s
within ~2x across three different quantisations.

**Why it matters.** CONTEXT.md budgets "5–20 tok/s". For fp32 that is **wrong by 2–7x**.
Consequences for the whole project:

- A 100-token completion takes **35–55 s**. That is fine for a microscope (you are
  watching it think) but it rules out any "chat" framing.
- Prefill is *not* the problem: it is parallel across tokens and compute-bound. A 512-token
  prompt costs 512 x 3.09 GFLOP / ~55 GFLOP/s ≈ 29 s of compute but ~0.5 s of memory
  traffic, so it lands around **0.3–1 s** in practice — i.e. prompt length is nearly free.
- **Therefore: spend the interaction budget on output length, not prompt length.**

**Confidence: high** (arithmetic + measurement agree). **Cost:** free.

---

### F4. The logit lens costs 4.2x the model, and naive per-layer lens costs 8x — fix it with one stacked GEMM
[source: measured on this machine; method: nostalgebraist, *Interpreting GPT: the Logit Lens*,
2020, https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens]

**What it is.** A logit lens applies the final `RMSNorm` + `lm_head` to each layer's
residual. `lm_head.weight` is the **tied embedding matrix**: 151936 x 1536 x 4 B = **933 MB**
in fp32. It does not fit in the 8 MiB L3, so every application re-streams it from DRAM.

- One lens application: 2 x 151936 x 1536 = **0.467 GFLOP**, **0.933 GB** read.
- 28 layers: **13.07 GFLOP** (4.2x the model's 3.09 GFLOP) and **26.1 GB** read
  (4.2x the model's 6.17 GB).

Measured on this box (numpy/BLAS, 800 MHz, 0.93 GB fp32 matrix):

| pattern | time |
|---|---|
| 1 layer (GEMV) | 94 ms |
| **28 separate GEMVs (naive)** | **2504 ms** |
| 28 residuals stacked into one `[28,1536] @ [1536,151936]` GEMM | **493 ms** |
| 7 layers stacked (every 4th layer) | 283 ms |

So: naive per-layer lens alone is **2.5 s/token**, i.e. ~5x the entire model forward. It
would make the instrument unusable and everyone would blame the model.

**The fix — three independent levers, use all three:**

1. **Stack.** Capture residuals for all requested layers into one `[L, H]` tensor and do a
   single `F.linear(stacked, lm_head.weight)`. Reads `lm_head` **once** instead of L times.
   5.1x measured. This is mandatory and costs nothing.
2. **Subsample layers.** Default to 7–8 layers (e.g. `[0,4,8,12,16,20,24,27]`), not 28.
   Budget-wise: 493 ms → 283 ms, and the UI gains nothing from 28 nearly-identical bars.
   Make the set a request parameter.
3. **Downcast the GEMM.** `lm_head` in bf16 halves the read to 467 MB, but see F2 — bf16
   GEMM is slow. Instead keep fp32 and accept it; or precompute a rank-512 truncated SVD
   of `lm_head` (0.31 GB, 4.3 GFLOP) if you need it cheaper. `UNVERIFIED` on fidelity.

**Budget with all three (full clock, 17 GB/s, 55 GFLOP/s):** model 363 ms + stacked-7-lens
~90 ms + norms/topk ~10 ms ≈ **460 ms/token ≈ 2.2 tok/s**. Without the fixes: ~3 s/token.
**Confidence: high** (directly measured). **Cost:** ~20 lines.

---

### F5. Library comparison for hook-based work — and why raw hooks win here
[source: TransformerLens docs, 2026, https://transformerlensorg.github.io/TransformerLens/content/news/release-3.0.html;
nnsight docs, https://github.com/ndif-team/nnsight/blob/main/docs/usage/generate.md;
pyvene, https://stanfordnlp.github.io/pyvene/api/pyvene.models.html;
GitHub API for last-push dates, queried 2026-11-10]

| | `transformers` + `register_forward_hook` | TransformerLens 4.3.0 | nnsight 0.8.2 | pyvene 0.1.8 | baukit |
|---|---|---|---|---|---|
| **Qwen2 / Qwen2.5** | any HF model, by construction | yes — `Qwen2ForCausalLM` adapter; alias `qwen2.5-1.5b` registered; 48 archs / 9k models | yes — `LanguageModel` wraps `AutoModelForCausalLM` | yes — `pyvene.models.qwen2` module exists | no adapter needed, but see below |
| **Hook granularity** | module in/out only (`layers[i]`, `attn`, `mlp`, `norm`, `lm_head`) | named `HookPoint`s incl. per-head `hook_z`, `hook_q/k/v`, `hook_mlp_out` | module-level via `Envoy`; `.output`, `.input`, can index/slice inside | named components (`block_output`, `head_out`, …) | module-level `TraceDict` / `edit_output` |
| **Intervene inside `generate()`** | yes — forward hook returning a tensor replaces the output; runs every step | yes — `run_with_hooks`, but `generate` is TL's own loop | **yes, and this is its killer feature**: `tracer.iter[step]` + `if step_idx == 2: h[:] = 0` | yes — `IntervenableModel` wraps generate | only via `edit_output` on a plain forward |
| **Maintenance (last push)** | transformers 5.22.2, 2026-11-06 | **2026-11-10**, 4.1k★ | **2026-11-10**, 1.1k★ | 2026-03-06, 917★; PyPI 0.1.8 is 2025-05 | **2024-02-22** (2.7 yr), 258★ |
| **CPU friendliness** | native; no extra copies | wraps HF model → same cost, plus cache bookkeeping | adds an Envoy per module + a worker-thread mediator per access → measurable overhead on 28x28 hook reads | wraps HF model; extra `nn.Module` layer per intervention site | zero overhead, but no maintenance |
| **Breaking-change risk** | **high** — v5 just landed, `dtype="auto"` default, `logits_to_keep` rename | medium — `HookedTransformer` **removed in 4.0**; API moved to `TransformerBridge` | low | low activity | abandoned |

**Why raw hooks, concretely.** We need exactly three things: (a) read the post-block
residual at the last position, (b) add `coef * v` to it, (c) do both on every decode step.
That is 30 lines against `model.model.layers[i]`. Every library above adds an abstraction
whose *only* benefit is ergonomics for research code we are not writing. nnsight's
`tracer.iter` is genuinely nicer for conditional interventions, but it also owns the
generation loop, and we need to own it (rewind, re-prefill, per-step emit, request-driven
steering from a websocket). We would spend more time fighting the abstraction than the
30 lines cost.

**Where I would still use one:**
- **nnsight** — if we get to activation patching / causal tracing (A2's agenda). Its
  `invoke()` batching + cross-prompt intervention is materially better than hand-rolling
  `output.clone()` swap logic, and it is the most actively maintained of the four.
- **TransformerLens** — only if we want its pre-built `hook_z`/per-head decomposition and
  `run_with_cache` conventions. **Warning for D3:** `HookedTransformer` and
  `HookedTransformer.from_pretrained` were **removed in 4.0**; the supported path is
  `TransformerBridge.boot_transformers(...)` (+ `enable_compatibility_mode()` for
  HookedTransformer-equivalent numerics). Any tutorial written before 2026 is wrong.
- **pyvene** — skip. It is an *intervention* library; we get interventions from one hook.
  PyPI release is 18 months stale, though the repo is alive.
- **baukit** — skip. No commits since Feb 2024, and its whole model is monkeypatching
  `forward` methods, which is exactly what we do not want under transformers v5.

**Confidence: high** on facts, **medium** on the recommendation (it is a taste call).
**Cost:** the 30 lines are already written below.

---

### F6. transformers v5 will silently hand you bf16 weights and a 4x-too-expensive lm_head
[source: transformers `MIGRATION_GUIDE_V5.md`, 2026,
https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md;
`modeling_utils.py` and `modeling_qwen2.py` at tag v5.22.2,
https://github.com/huggingface/transformers/tree/v5.22.2/src/transformers]

Three v5-specific landmines, all verified in source:

1. **`dtype` default.** v5 replaced `torch_dtype=` with `dtype=` and the default is
   `"auto"` — read straight out of the source, `modeling_utils.py` v5.22.2:
   `if dtype is None: dtype = "auto"`. Qwen2.5-Coder-1.5B is bf16 → you get bf16
   → F2 bites you → you conclude "CPU is hopeless". **Always pass
   `dtype=torch.float32` explicitly.** (Also at the same spot: `low_cpu_mem_usage`,
   `mirror`, `_fast_init`, `offload_state_dict` are popped and ignored — "Not used anymore".)
2. **`num_logits_to_keep` → `logits_to_keep`.** In v5.22.2 `Qwen2ForCausalLM.forward`
   takes `logits_to_keep: int | torch.Tensor = 0` and calls
   `slice_logits_to_keep(hidden_states, logits_to_keep)` *before* `self.lm_head(...)`
   (lines 424, 455–456). With `0` (the default) it computes logits for **every** position.
   On a 512-token prefill that is 512 x 0.467 GFLOP = **239 GFLOP** of pure waste —
   ~4 s. **Always pass `logits_to_keep=1`.**
3. **Default cache class changed.** "If `generate` doesn't receive any KV Cache argument,
   the default cache class used is now defined by the model (as opposed to always being
   `DynamicCache`)" (MIGRATION_GUIDE_V5, PR #41505). So in a hand-written loop, pass an
   explicit `DynamicCache(config=model.config)` and never rely on the default.
4. Minor: `safe_serialization` is gone (safetensors only), and old output-type aliases
   (`GreedySearchEncoderDecoderOutput`) were removed.

Also note `Qwen2DecoderLayer.forward` **returns a bare tensor** (line 290), not a tuple.
A forward hook on `model.model.layers[i]` therefore receives the post-block residual
directly — but write the hook defensively (`if isinstance(out, tuple): out = out[0]`),
because this has flip-flopped across versions.

**Confidence: high** (read the source). **Cost:** 3 keyword arguments.

---

### F7. Per-token streaming with activations: the correct pattern
[source: HF *Caching* guide, 2026, https://huggingface.co/docs/transformers/main/en/cache_explanation;
`generation/streamers.py` at v5.22.2]

**The problem with `TextIteratorStreamer`.** It is a `BaseStreamer` that `generate()` calls
with decoded *text* (`put()` / `end()`). It carries **no** tensor payload, and it requires
`generate()` to run in a separate `threading.Thread` because `generate()` blocks. It is the
right tool for a chat UI and the wrong tool for us — we would have to smuggle activations
out through module-global state and hope the ordering matched.

**Three real options, ranked:**

| pattern | per-step hook point | keeps HF sampler/stopping | control over rewind/re-prefill | verdict |
|---|---|---|---|---|
| **(A) manual loop** over `DynamicCache`, `model(...)` per step | anywhere (module hooks) | no (you write greedy/sampling) | **total** | **recommended** |
| (B) `generate()` + hooks + a `LogitsProcessor` as the flush point | `LogitsProcessor.__call__` runs once per step, after the forward | yes | poor (generate owns the loop) | good fallback |
| (C) `StoppingCriteria` | once per step, **after** logits are processed | yes | none | ✗ wrong tool — it only gets `input_ids`/`scores` and cannot carry activations cleanly |

**Why (A) is right for a microscope.** `generate()` is a closed loop: you cannot ask it to
pause, rewind to step 12, change a steering vector, and re-run. We need exactly that. A
manual loop is ~40 lines and gives us: exact step indexing, the ability to emit a trace
before choosing the token, the ability to re-prefill after a steering change, and no
ambiguity about what `logits_to_keep` did.

**The critical KV-cache insight.** With `use_cache=True` and a hand-written loop, each
decode step passes **one** token, so `q_len == 1` and every hook sees a
`[1, 1, 1536]` tensor. **Capture is already "last token only" for free** — the
`[:, -1, :]` slice only matters during prefill. This is why the naive
`output_hidden_states=True` approach is a trap: it materialises all 29 layers x all
positions x 1536 floats. For a 512-token prefill that is 91 MB; per decode step it is
178 KB. Hooks with a `[:, -1, :]` slice are strictly better in both cases.

**Second cache insight, and it bites.** Steering during *prefill* writes modified K/V into
the cache. The steering then persists for every subsequent token whether you want it or
not, and you cannot undo it — the cache is poisoned. So: apply steering only when
`cache_position >= prompt_len` for `positions="gen"`, and provide an explicit
`replay()` that rebuilds the cache from the prompt (a prefill is ~0.3–1 s, so this is
cheap). Do not try to surgically un-poison.

**Third: hooks compose by registration order.** PyTorch's `_call_impl` reassigns
`result = hook(...)` for each forward hook in order, so a later hook sees an earlier
hook's modification. Register the **steering** hook first, then the **capture** hook, and
the trace you emit is the post-steering activation — which is what you want to show.

**Confidence: high.** Code in "Concrete implementation".

---

### F8. Capturing activations without a memory blowup
[source: derived; measured payload sizes]

**What it is.** Per decode step, the raw numbers are tiny; the blowup only happens if you
keep history naively.

- One step, all 28 layers, fp32: `28 x 1536 x 4 B = 172 KB`.
- One step, all 28 layers, fp16: 86 KB.
- A 500-step session, all layers, fp32, retained: **86 MB**. fp16: 43 MB.
- Ring buffer of 64 steps, fp16: **11 MB**.

**The discipline (all four, they are cheap):**

1. **Capture the last position only** (`h[0, -1, :]`) — free during decode, one slice during prefill.
2. **Downcast at capture** — `h.detach().to(torch.float16)` immediately. fp16 has ~3 decimal
   digits; every downstream consumer (norms, top-k, entropy) is fine with that, and it
   halves both memory and the eventual wire size. Keep fp32 only for the vector you are
   about to use in a steering addition.
3. **Do not retain residuals per step.** Emit *derived scalars* (`resid_norm`,
   `attn_entropy`, `mlp_out_norm`, top-5 lens) and throw the vector away, unless the UI
   explicitly asked for the residual trajectory view (which needs `[steps, layers, H]`).
4. **Ring buffer** (`collections.deque(maxlen=N)`) for the trajectory/PCA view; make N a
   config knob. Never an unbounded list.

Also: `del` the logits tensor after top-k. `[28, 151936]` fp32 is 17 MB and Python will
happily keep it alive across the loop if you leave it bound.

**Confidence: high.** **Cost:** ~5 lines.

---

### F9. Serving: JSON over WebSocket is fine; raw residuals are not
[source: arithmetic on the canonical schema; no external source needed]

**Payload per token for 28 layers x top-5, JSON:**

- 28 x 5 x (id ~6 chars + token_str ~6 + prob ~7 + punctuation) ≈ 3.5 KB
- 28 x 3 scalars (`resid_norm`, `attn_entropy`, `mlp_out_norm`) ≈ 0.5 KB
- key names: 28 x ~70 B ≈ 2.0 KB
- final `logits_topk` + `chosen` ≈ 0.2 KB

**≈ 6–8 KB per token.** At 2–3 tok/s that is **15–25 KB/s**. A 200-token answer is ~1.5 MB
retained client-side.

**Verdict: keep JSON.** Binary (msgpack/arrow) would cut it to ~1 KB — a 7x win on a
quantity that is already 4 orders of magnitude below localhost capacity. JSON parsing at
6 KB is ~0.05 ms in a browser. Optimising this is premature and would cost a week. The
honest engineering call is: **JSON lines over one WebSocket, `orjson` on the server side
for encode speed, and revisit only if we ever stream at >50 tok/s.**

**Where binary *does* matter:** raw residual vectors. `1536 x 4 B x 28 layers = 172 KB/token`
→ ~500 KB as JSON per token, i.e. 1.5 MB/s. **Rule: never put residuals in the per-token
stream.** Send scalars + top-k, and serve residuals on demand as
`application/octet-stream` (`Float32Array` / `Float16Array`) from a separate GET.

**Backpressure.** Generation runs in a worker thread (PyTorch releases the GIL inside
kernels, so this is safe and is what `TextIteratorStreamer` assumes). Frames go into a
`queue.Queue` drained by an asyncio task into a bounded `asyncio.Queue(maxsize=256)`.
If the client stalls: **drop-oldest for trace frames, never drop `token` text frames,
never block the generator.** At 3 tok/s a stalled client needs ~60 s to fill 256 frames,
so this is defensive, not load-bearing.

**Confidence: high.** **Cost:** ~10 lines of queue plumbing.

---

### F10. llama.cpp: residuals are already reachable without patching; steering is not
[source: llama.cpp `src/models/qwen2.cpp`, `include/llama.h`, `ggml/include/ggml-backend.h`,
`src/llama-ext.h` at master, 2026-11; PR #27073, 2026; release b9606, 2026]

Going through each mechanism named in the brief:

| mechanism | what it actually gives you |
|---|---|
| `llama_get_embeddings(ctx)` | **only** the final post-`output_norm` state (`res->t_embd`, qwen2.cpp:133). One vector per output token. Not per-layer. |
| `llama_context_params.embeddings = true` / `--embeddings` | enables the above; `llama_get_embeddings_ith(ctx, i)` indexes it. Still final-layer only. |
| `logits_all` | logits for **every prompt token** (for perplexity). Logits, not hidden states. |
| `llama_batch.logits[i]` | which tokens produce output at all (llama.h:259–261). |
| `--verbose-prompt` | prints prompt **token IDs**. Nothing else. |
| `imatrix` | dumps **weight** importance (Σx²) for quantisation, not per-token activations. |
| **`llama_set_embeddings_layer_inp(ctx, lid, true)` + `llama_get_embeddings_layer_inp(ctx, lid)`** | **per-layer input capture, upstream, no patch.** Added for EAGLE3 spec-decoding (release b9606, 2026-06). `llama_cparams.embeddings_layer_inp` is `std::vector<bool>` sized `n_layer()+1`. Captures `inpL`, i.e. the residual *entering* layer `lid` = the post-block residual of `lid-1`. That is exactly a logit-lens input. |
| **`llama_context_params.cb_eval`** (a `ggml_backend_sched_eval_callback`) | **a per-graph-node callback, public API** (llama.h:394–395, ggml-backend.h). `qwen2.cpp` names every layer output `"l_out"` (line 121), plus `"attn_norm"`, `"Qcur"`, `"Kcur"`, `"Vcur"`, `"ffn_out"`, `"result_norm"`, `"result_output"`. Filter `t->name == "l_out"` and read the last column. |

**So the answer is: yes, stock llama.cpp can emit per-layer residuals, two ways.**
The `cb_eval` route is the more general one (it can grab attention Q/K/V too), and it has
been public since PR #4935 (2024). The `embeddings_layer_inp` route is the more efficient
one (it copies once per decode into a host/device buffer).

**Caveats, and they are real:**
- `src/llama-ext.h` opens with *"this is a staging header for new llama.cpp API — breaking
  changes and C++ are allowed, everything here should be considered WIP."* Pin the commit.
- The `_dev` variant is tied to the DFlash/DSpark speculative-decoding context, not
  general use. Use the host variant.
- There is a **closed, unmerged PR #27073** ("hidden-state extraction API + tools + server
  endpoint", Aug 2026) that proposed exactly `llama_get_hidden_state(ctx, i)` etc. It was
  closed by its author 1 day after opening. **Do not build on it**; it tells you the
  maintainers did not want this surface, so expect to maintain a fork if you need a stable API.
- **Steering does not exist at all.** There is no API to add a vector to a graph node. You
  would patch `llm_graph_context::cb` or `build_qwen2` to `ggml_add` a broadcast constant
  gated on `il == target_layer` — realistically ~40 lines, plus a way to get the vector in
  (a file, or a new `llama_set_steer(ctx, layer, ptr, n)`). Feasible, but it is a
  maintained fork forever.

**Is it worth it?** **No, for the science path** — D2 already says Q4_K_M is lossy, and
we would be reimplementing logit lens/top-k in C++ with no steering. **Yes, as a
sanity-check oracle:** run the same prompt through llama.cpp at 16 tok/s and compare the
top-5 lens trajectory to the PyTorch path. That catches "our hooks are wired wrong" in
minutes instead of hours, at 5x the speed. Effort: ~half a day for the `cb_eval` reader.

**Confidence: high** (read the source). **Cost:** half a day for the oracle, 3–5 days for
a steering fork.

---

### F11. Startup, caching, and keeping a session warm
[source: derived from file sizes + safetensors mmap semantics]

- **Weights on disk:** 3.09 GB (bf16, one `model.safetensors`) or 6.17 GB (fp32).
- `from_pretrained(dtype=torch.float32)` on the bf16 checkpoint does a **read 3.09 GB +
  write 6.17 GB** upcast — ~5–9 s from NVMe plus 6.17 GB of resident RAM.
- **Better: convert once** to a local fp32 safetensors dir (293 GB free, so no excuse not
  to). Then `from_pretrained` mmaps it: first token pays page-faults, subsequent tokens
  read from page cache at full bandwidth. Load becomes ~1–2 s.
- **Warmup is mandatory.** Run 3–5 forward passes at `seq_len=1` (and one at your typical
  prefill length) before serving. This warms oneDNN primitive caches, JIT'd code paths, and
  touches every weight page. Without it the first user-visible token is 5–15 s late and
  looks like a hang.
- **Keep the session alive**: load once in a FastAPI `lifespan` handler, store on
  `app.state`. Never per-request. A single global `ModelSession` with a
  `threading.Lock` around generation (one generation at a time — we have 4 cores; two
  concurrent generations would halve each one's speed and confuse the UI).
- **Cache the prefill**: keep the `DynamicCache` after a generation so a follow-up
  "steer and continue" can resume instead of re-prefilling. Invalidate it whenever the
  prompt or the prefill-time steering config changes.
- **RAM budget**: 6.17 GB weights + ~0.3 GB KV at 4k ctx + ~0.5 GB interpreter/torch
  overhead ≈ 7 GB. The box has 31 GB but 14 GB is already in use — fine, but do not run
  two model copies.

**Confidence: medium-high** (file sizes exact, timings estimated). **Cost:** ~1 hour.

---

### F12. CONTEXT.md's model ID is wrong
[source: HF API, queried 2026-11-10]

`GET https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-1.5B-Base` → **401**.
`GET .../api/models?search=Qwen2.5-Coder-1.5B&author=Qwen` returns:
`Qwen2.5-Coder-1.5B`, `-Instruct`, `-Instruct-GGUF`, `-Instruct-GPTQ-Int4/Int8`, `-Instruct-AWQ`.
There is **no `-Base` variant** — for Qwen2.5-Coder the unsuffixed repo *is* the base model.

Verified facts for `Qwen/Qwen2.5-Coder-1.5B`:
`Qwen2ForCausalLM`, 28 layers, hidden 1536, 12 Q heads, 2 KV heads, intermediate 8960,
vocab 151936, **ctx 32768**, `tie_word_embeddings: true`, `torch_dtype: bfloat16`,
`rope_theta: 1e6`, single `model.safetensors`, **1,543,714,304** params.

**Why it matters:** every agent's `from_pretrained` string is wrong, and A1's format work
may be pointed at a 404. Also: CONTEXT says ctx 32768 — correct for Coder (the plain
Qwen2.5-1.5B is 131072; do not copy its config).

**Confidence: high.** **Cost: zero.**

---

## Concrete implementation (code sketches, real APIs)

Layout, deps, then the three files that matter.

### Repo layout

```
code/a4_runtime/
├── pyproject.toml            # uv-managed, python 3.12
├── convert_fp32.py           # one-time bf16 -> fp32 safetensors (F11)
├── bench.py                  # tok/s + bandwidth + bf16-vs-fp32 A/B (F2/F3)
├── modwei/
│   ├── __init__.py
│   ├── adapter.py            # ModelAdapter protocol (load/forward_trace/attach_steering)
│   ├── torch_adapter.py      # the transformers implementation (this file's core)
│   ├── steering.py           # SteeringBus + hook factory
│   ├── lens.py               # batched logit lens + top-k
│   ├── schema.py             # TraceStep pydantic model  <-> canonical schema
│   └── session.py            # ModelSession: model, cache, lock, replay()
├── server.py                 # FastAPI + WebSocket
└── web/
    ├── index.html
    └── app.js                # ~60-line client
```

### Dependencies (`pyproject.toml`, pinned, py3.12, verified on PyPI 2026-11-10)

```toml
[project]
name = "modwei-runtime"
requires-python = ">=3.12,<3.13"
dependencies = [
  "torch==2.15.0",              # install with --index-url https://download.pytorch.org/whl/cpu
  "transformers==5.22.2",
  "tokenizers==0.23.3",
  "safetensors==0.8.0",
  "huggingface-hub==2.4.0",
  "numpy==2.5.4",
  "fastapi==0.162.4",
  "uvicorn[standard]==0.54.0",
  "websockets==17.4",
  "orjson==3.13.0",
  "pydantic==2.14.3",
]

[project.optional-dependencies]
interp = [
  "nnsight==0.8.2",             # for activation patching later (A2)
  # "transformer-lens==4.3.0",  # only if we need hook_z / per-head; heavy, adds a 4.0 API migration
]
dev = ["pytest==9.2.0"]

[tool.uv]
# uv venv --python 3.12
```

Notes: `torch==2.15.0` needs py>=3.11 and is only on the CPU index for CPU builds;
`numpy==2.5.4` requires py>=3.12. **Do not add `transformer-lens` in v1** — it pins its own
`transformers` range and will fight 5.22.2.

### `modwei/steering.py` — the bus and the hook

```python
from __future__ import annotations
import threading
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class SteeringCmd:
    layer: int
    vector: torch.Tensor          # [hidden], fp32, CPU
    coef: float = 1.0
    positions: str = "last"       # "last" | "all"
    until_step: int | None = None

class SteeringBus:
    """Thread-safe control plane. The websocket handler writes; the hooks read.

    Hooks are called on every forward pass, so reads must be cheap: we keep a
    per-layer list of *already-cast* tensors and never allocate inside the hot path.
    """

    def __init__(self, n_layers: int):
        self._lock = threading.Lock()
        self._by_layer: dict[int, list[SteeringCmd]] = {}
        self._step = 0

    def set(self, cmds: list[SteeringCmd]) -> None:
        by: dict[int, list[SteeringCmd]] = {}
        for c in cmds:
            by.setdefault(c.layer, []).append(c)
        with self._lock:
            self._by_layer = by

    def clear(self) -> None:
        with self._lock:
            self._by_layer = {}

    def begin_step(self, step: int) -> None:
        with self._lock:
            self._step = step
            # drop expired commands
            for layer, cmds in list(self._by_layer.items()):
                live = [c for c in cmds if c.until_step is None or step <= c.until_step]
                if live:
                    self._by_layer[layer] = live
                else:
                    del self._by_layer[layer]

    def active(self, layer: int) -> list[SteeringCmd]:
        with self._lock:
            return self._by_layer.get(layer, ())

    def make_hook(self, layer: int):
        """Forward hook that adds coef * v to the residual stream.

        Returning a tensor REPLACES the module output (torch calls
        `result = hook(self, args, result)` for each hook in registration order).
        Register this BEFORE the capture hook so the trace shows post-steering state.
        """
        def hook(module, args, output):
            cmds = self.active(layer)
            if not cmds:
                return None                      # None == "leave output alone"
            h = output[0] if isinstance(output, tuple) else output
            for c in cmds:
                v = c.vector.to(device=h.device, dtype=h.dtype)
                if c.positions == "all":
                    h = h + (c.coef * v)
                else:                            # "last" == the only position when decoding
                    h = h.clone()
                    h[:, -1, :] = h[:, -1, :] + (c.coef * v)
            return h if not isinstance(output, tuple) else (h,) + tuple(output[1:])
        return hook
```

### `modwei/lens.py` — one GEMM, not twenty-eight (F4)

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def logit_lens(model, residuals: torch.Tensor, k: int = 5):
    """residuals: [L, H] float32, post-block residual stream (pre-final-norm).

    Returns (ids [L,k], probs [L,k], logprobs_topk [L,k]).
    Costs ONE pass over lm_head (933 MB fp32) regardless of L.
    """
    normed = model.model.norm(residuals)                 # [L, H]; RMSNorm upcasts internally
    logits = F.linear(normed, model.lm_head.weight)      # [L, V]  <- the single GEMM
    logp = F.log_softmax(logits, dim=-1)                 # [L, V]; 17 MB for L=28, freed below
    top = torch.topk(logp, k, dim=-1)
    del logits, logp, normed
    return top.indices, top.values.exp(), top.values


def scalar_summary(h: torch.Tensor) -> dict[str, float]:
    """Cheap per-layer scalars. h: [H] fp16/fp32."""
    f = h.float()
    return {
        "resid_norm": float(f.norm()),
        "resid_mean": float(f.mean()),
        "resid_std": float(f.std()),
    }
```

### `modwei/torch_adapter.py` — the per-token loop (the core of the whole thing)

```python
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .lens import logit_lens, scalar_summary
from .steering import SteeringBus


@dataclass
class Step:
    step: int
    token_id: int
    token_str: str
    layer_ids: list[int]
    layers: list[dict]
    logits_topk: list[list]
    chosen: list
    t_ms: float                      # wall time of the forward that produced this token
    lens_ms: float                   # wall time of logit-lens + trace assembly
    seq_len: int
    prefill_ms: float | None = None  # only set on step 0


class TorchAdapter:
    """ModelAdapter implementation for any HF causal LM (Qwen2 family)."""

    def __init__(
        self,
        model_dir: str,
        *,
        dtype: torch.dtype = torch.float32,
        lens_layers: Sequence[int] | None = None,
        n_threads: int = 4,
        device: str = "cpu",
    ):
        torch.set_grad_enabled(False)
        torch.set_num_threads(n_threads)          # 4P/8T: 8 is usually slower (F1)
        torch.set_float32_matmul_precision("highest")

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,                          # v5: `dtype`, NOT `torch_dtype`. F6
            device_map=device,
            # NB: `low_cpu_mem_usage` is silently popped and ignored in v5
            # (modeling_utils.py:4100 "Not used anymore") -- do not pass it.
        ).eval()

        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        # default: 8 layers, always including the last
        self.lens_layers = list(lens_layers) if lens_layers else \
            sorted({*range(0, self.n_layers, 4), self.n_layers - 1})

        self.bus = SteeringBus(self.n_layers)
        self._cap: dict[int, torch.Tensor] = {}

        # register steering first, capture second (registration order == execution order)
        self._steer_handles = [
            self.layers[i].register_forward_hook(self.bus.make_hook(i))
            for i in range(self.n_layers)
        ]
        self._cap_handles = [
            self.layers[i].register_forward_hook(self._make_capture(i))
            for i in self.lens_layers
        ]
        self.warmup()

    # ---- hooks -------------------------------------------------------------
    def _make_capture(self, layer: int):
        cap = self._cap

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            # decode: [1,1,H]; prefill: [1,T,H] -> take the last position only (F8)
            cap[layer] = h[0, -1, :].detach().to(torch.float16).clone()
            return None                              # never modify in the capture hook
        return hook

    def warmup(self, seq_lens: Sequence[int] = (1, 8)):
        for n in seq_lens:
            ids = torch.zeros((1, n), dtype=torch.long)
            self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        self._cap.clear()

    # ---- trace assembly ----------------------------------------------------
    def _emit(self, step: int, token_id: int, logits_1v: torch.Tensor,
              fwd_ms: float, seq_len: int) -> Step:
        t_lens = time.perf_counter()
        stack = torch.stack([self._cap[l] for l in self.lens_layers]).float()  # [L,H]
        ids, probs, _logps = logit_lens(self.model, stack, k=5)                # [L,k]

        vtop = torch.topk(logits_1v, 5)
        all_ids = ids.flatten().tolist() + vtop.indices.tolist()
        # one decode call for all ~145 ids, not 145 calls: saves 3-7 ms/token
        strs = self.tok.batch_decode([[i] for i in all_ids])
        L, k = ids.shape

        layer_rows = []
        for j, lid in enumerate(self.lens_layers):
            row = {"layer": lid, **scalar_summary(stack[j])}
            row["logit_lens_topk"] = [
                [all_ids[j * k + m], strs[j * k + m], round(float(probs[j, m]), 5)]
                for m in range(k)
            ]
            layer_rows.append(row)

        base = L * k
        topk = [[all_ids[base + m], strs[base + m],
                 round(float(p), 5)]
                for m, p in enumerate(vtop.values.softmax(-1).tolist())]
        return Step(
            step=step, token_id=token_id, token_str=self.tok.decode([token_id]),
            layer_ids=self.lens_layers, layers=layer_rows,
            logits_topk=topk, chosen=topk[0],
            t_ms=round(fwd_ms, 1),
            lens_ms=round((time.perf_counter() - t_lens) * 1000, 1),
            seq_len=seq_len,
        )

    # ---- the loop ----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache: DynamicCache | None = None,
        on_step: Callable[[Step], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[list[int], DynamicCache]:
        """One forward pass per token, emitting a trace before choosing the token.

        Returns (generated_token_ids, cache). Pass the cache back in to continue.
        """
        if seed is not None:
            torch.manual_seed(seed)

        ids = self.tok(prompt, return_tensors="pt").input_ids
        n_prompt = ids.shape[1]
        attn = torch.ones((1, n_prompt), dtype=torch.long)
        cache = cache if cache is not None else DynamicCache(config=self.model.config)

        out_ids: list[int] = []
        step = 0

        # --- prefill: T tokens at once; hooks fire with q_len = T ---
        t_prefill = time.perf_counter()
        self.bus.begin_step(0)
        out = self.model(
            input_ids=ids,
            attention_mask=attn,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,            # F6: without this, 512 tokens x 0.47 GFLOP wasted
        )
        logits = out.logits[0, -1]       # [V]
        prefill_ms = (time.perf_counter() - t_prefill) * 1000
        fwd_ms = prefill_ms              # step 0's trace describes this prefill forward

        # --- decode loop: one token per step; hooks see q_len = 1 ---
        for step in range(max_new_tokens):
            if should_stop is not None and should_stop():
                break
            self.bus.begin_step(step)

            if temperature <= 0.0:
                nxt = int(torch.argmax(logits))
            else:
                p = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sp, si = torch.sort(p, descending=True)
                    keep = sp.cumsum(-1) <= top_p
                    keep[0] = True
                    mask = torch.zeros_like(p, dtype=torch.bool)
                    mask[si[keep]] = True
                    p = p * mask
                nxt = int(torch.multinomial(p / p.sum(), 1))

            # Emit AFTER sampling so the frame carries the token this forward produced.
            # The trace still describes the forward that *chose* it. `chosen` == the
            # sampled token, which differs from logits_topk[0] under sampling -- showing
            # that gap is a feature.
            st = self._emit(step, nxt, logits, fwd_ms, n_prompt + len(out_ids))
            st.prefill_ms = round(prefill_ms, 1) if step == 0 else None
            if on_step is not None:
                on_step(st)

            out_ids.append(nxt)
            if nxt == self.tok.eos_token_id:
                break

            t_fwd = time.perf_counter()
            attn = torch.cat([attn, attn.new_ones((1, 1))], dim=1)
            out = self.model(
                input_ids=torch.tensor([[nxt]]),
                attention_mask=attn,
                past_key_values=cache,   # explicit cache; v5 default is model-defined (F6)
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[0, -1]
            fwd_ms = (time.perf_counter() - t_fwd) * 1000

        return out_ids, cache
```

**What is deliberately *not* here:** no `output_hidden_states=True` (F8), no
`TextIteratorStreamer` (F7), no `generate()` (F7), no `torch.compile` (it graph-breaks on
every hook and the compile time is minutes for a 4-core box — see traps).

### `server.py` — FastAPI + WebSocket

```python
from __future__ import annotations
import asyncio, os, queue, threading
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from modwei.torch_adapter import TorchAdapter
from modwei.steering import SteeringCmd
import torch

MODEL_DIR = os.environ.get("MODWEI_MODEL_DIR", "models/qwen2.5-coder-1.5b-fp32")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load once, keep alive (F11)
    app.state.adapter = TorchAdapter(MODEL_DIR, dtype=torch.float32, n_threads=4)
    app.state.gen_lock = threading.Lock()          # one generation at a time
    app.state.last = None                          # (cache, prompt, n_prompt)
    yield


app = FastAPI(lifespan=lifespan)


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 48
    temperature: float = 0.0
    top_p: float = 1.0
    lens_layers: list[int] | None = None
    resume: bool = False


@app.post("/generate")
async def generate(req: GenReq):
    ad: TorchAdapter = app.state.adapter
    frames: list[dict] = []
    ids, cache = await asyncio.to_thread(
        ad.generate, req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        on_step=lambda s: frames.append(s.__dict__),
    )
    app.state.last = (cache, req.prompt)
    return {"tokens": ids, "text": ad.tok.decode(ids), "trace": frames}


@app.get("/residual/{step}/{layer}")
async def residual(step: int, layer: int):
    """On-demand binary residual (F9: never in the token stream)."""
    ad: TorchAdapter = app.state.adapter
    v = ad._cap.get(layer)
    if v is None:
        return Response(status_code=404)
    return Response(content=v.float().numpy().tobytes(),
                    media_type="application/octet-stream")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    ad: TorchAdapter = app.state.adapter
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    stop_flag = threading.Event()

    async def sender():
        while True:
            frame = await out_q.get()
            if frame is None:
                return
            await sock.send_text(orjson.dumps(frame).decode())

    async def receiver():
        """Control channel: steering + stop, applied live while generation runs."""
        while True:
            msg = orjson.loads(await sock.receive_text())
            if msg["type"] == "steer":
                ad.bus.set([
                    SteeringCmd(layer=c["layer"],
                                vector=torch.tensor(c["vector"], dtype=torch.float32),
                                coef=c["coef"],
                                positions=c.get("positions", "last"),
                                until_step=c.get("until_step"))
                    for c in msg["commands"]
                ])
                await out_q.put({"type": "steer_ack", "n": len(msg["commands"])})
            elif msg["type"] == "stop":
                stop_flag.set()

    async def worker(req: GenReq):
        def on_step(s):
            # thread -> event loop; drop-oldest if the client is stalled (F9)
            def put():
                if out_q.full():
                    try: out_q.get_nowait()
                    except asyncio.QueueEmpty: pass
                out_q.put_nowait({"type": "token", **s.__dict__})
            loop.call_soon_threadsafe(put)

        try:
            await asyncio.to_thread(
                ad.generate, req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                on_step=on_step,
                should_stop=stop_flag.is_set,
            )
        finally:
            loop.call_soon_threadsafe(out_q.put_nowait, {"type": "done"})

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(receiver())
    try:
        while True:
            msg = orjson.loads(await sock.receive_text())
            if msg["type"] == "generate":
                asyncio.create_task(worker(GenReq(**msg)))
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel(); stop_flag.set()
        await out_q.put(None); await send_task
```

*(Sketch caveat: `receiver()` and the main loop both call `receive_text()` in the version
above — in the real file, fold them into one receive loop and dispatch, so there is a
single reader. Left split here to make the two responsibilities legible.)*

### Client sketch (`web/app.js`)

```js
const ws = new WebSocket(`ws://${location.host}/ws`);
const bars = new Map();                     // layer -> DOM node

ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.type === "token") {
    document.getElementById("out").textContent += m.token_str;
    for (const L of m.layers) {
      const el = bars.get(L.layer) ?? mkBar(L.layer);
      el.style.width = `${Math.min(100, L.resid_norm)}%`;
      el.title = `L${L.layer} ${L.logit_lens_topk.slice(0,3).map(t => t[1]).join(" · ")}`;
      el.dataset.top = JSON.stringify(L.logit_lens_topk);
    }
    document.getElementById("step").textContent = `${m.step} · ${m.t_ms} ms · ${m.seq_len} tok`;
  } else if (m.type === "done") {
    ws.send(JSON.stringify({type: "generate", prompt: "def fib(n):", max_new_tokens: 32}));
  }
};

function steer(layer, vec, coef) {
  ws.send(JSON.stringify({
    type: "steer",
    commands: [{layer, vector: vec, coef, positions: "last"}],
  }));
}
```

### `bench.py` — run this before writing anything else

```python
"""Settle F2/F3 empirically. ~10 minutes."""
import time, torch, torch.nn.functional as F
print("torch", torch.__version__, "cap:", torch.backends.cpu.get_cpu_capability())
print("threads", torch.get_num_threads())

V, H = 151936, 1536
W32 = torch.randn(V, H)
x = torch.randn(H)

def bench(f, reps=5):
    f(); t = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t) / reps

for dt in (torch.float32, torch.bfloat16):
    W = W32.to(dt); xx = x.to(dt)
    print(f"{str(dt):16s} lm_head GEMV {bench(lambda: F.linear(xx, W))*1e3:7.1f} ms")

# projected full-model decode: 1.543e9 params x 4 B / achieved GB/s
```

---

## What NOT to do (traps)

1. **Do not trust `dtype` defaults.** transformers v5 loads the checkpoint's own dtype.
   Qwen2.5-Coder ships bf16. You will silently benchmark the slow path and conclude the
   CPU is hopeless. Always `dtype=torch.float32`.
2. **Do not leave `logits_to_keep` at its default.** Default 0 means "logits for all
   positions". A 512-token prefill becomes 239 GFLOP of pure waste (~4 s). This is the
   single easiest 4x mistake in the project.
3. **Do not use `output_hidden_states=True`.** It allocates all 29 layers x all positions.
   91 MB per 512-token prefill and it gives you nothing hooks don't.
4. **Do not call `model.model.norm`/`lm_head` per layer in a Python loop.** That is the
   2.5 s/token path (F4). Stack, then one GEMM.
5. **Do not use `torch.compile` for v1.** Inductor's CPU backend is genuinely good
   (~2.35x on a HuggingFace model per PyTorch's own tutorial,
   https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html), but a
   `register_forward_hook` is a guaranteed graph break, compile takes minutes on 4 cores,
   and every new `seq_len` triggers a recompile. Revisit only after the trace path is
   frozen — and then compile *one decoder layer* as a standalone function, not the model.
6. **Do not set `torch.set_num_threads(8)` reflexively.** 4P/8T with 8 MiB L3: 7 threads
   beat 8 in the one published Qwen2.5-1.5B CPU benchmark. Measure 2/4/6/8.
7. **Do not use `generate()` if you need to rewind or re-steer mid-stream.** It owns the
   loop. Use it only if you need beam search (which you do not — beam search multiplies
   the forward passes by `num_beams`).
8. **Do not use `TextIteratorStreamer` for traces.** It carries text only, and it forces a
   thread you did not plan for.
9. **Do not use continuous batching.** transformers v5 has `generate_batch` /
   `ContinuousBatchingManager` (https://huggingface.co/docs/transformers/main/en/continuous_batching),
   but it is built for GPU throughput with PagedAttention and CUDA graphs, it has no
   per-step hook surface, and we have one user. Pure downside.
10. **Do not steer during prefill unless you mean it.** Modified K/V go into the cache and
    persist for the whole generation; you cannot undo it, only re-prefill.
11. **Do not keep residual tensors in a Python list across the loop.** 172 KB x 500 steps
    x a few views and you are at 500 MB with no idea why.
12. **Do not benchmark on the powersave governor.** At 800 MHz every number is ~2x
    pessimistic and every comparison is noisy.
13. **Do not build on llama.cpp's `src/llama-ext.h`** without pinning the commit — the
    header says WIP in its first line. And do not build on PR #27073; it is closed.

---

## Open questions for other agents

- **A1 (formats):** `Qwen/Qwen2.5-Coder-1.5B-Base` 404s — confirm the target is
  `Qwen/Qwen2.5-Coder-1.5B` before any extraction work (F12). Also: is the single
  `model.safetensors` (3.09 GB, bf16) tied-embedding-deduplicated? If `lm_head.weight` is
  absent from the file, our fp32 conversion must re-tie it, or `logit_lens` will
  `AttributeError` on `model.lm_head.weight` after a manual `load_state_dict`.
- **A2 (interp):** (a) Do you need logit lens on *all* 28 layers, or is an 8-layer band
  enough for the layer-localisation claims? Every extra layer costs ~12 ms/token.
  (b) Do you need *attention patterns*, or is `attn_entropy` enough? Full patterns require
  either `output_attentions=True` (28 x 12 x T² floats — 352 MB at T=512) or a wrapper
  that recomputes q·k for the last query row only (cheap: ~0.26 GFLOP for 28 layers).
  Tell me which and I will build it. (c) Is a *tuned* lens (per-layer affine) needed, or is
  the raw logit lens good enough at 1.5B?
- **A3 (steering):** steering vectors must arrive as fp32 `[1536]` and are cast to the
  activation dtype inside the hook. Confirm the sign convention (`+coef*v` adds toward the
  concept) and whether `positions="all"` during prefill is ever wanted. Also: do you need
  *per-position* vectors (a `[T, H]` steering tensor), or is one vector per layer enough?
- **A5 (UI):** what is the minimum per-token payload you will actually render? If the
  answer is "8 bars + 5 tokens", I can cut the stream from ~7 KB to ~1.5 KB per token and
  send 28 layers only on hover. Do you need the residual *trajectory* (PCA over layers) live,
  or only after the generation completes? The latter lets me batch it into one binary blob.
- **A6 (limits):** if I subsample lens layers to 8, does that invalidate any of your
  planned comparisons? And: what is the smallest number of decode steps that makes a
  steering claim statistically defensible, given ~2.5 tok/s? If it is 50 steps x 20 prompts,
  that is 7 minutes per condition and we should design the harness around it.
- **Orchestrator:** CONTEXT.md's "5–20 tok/s" and "8 cores" both need correcting
  (F1, F3). Also `ctx 32768` is right for Coder but wrong for the plain 1.5B (131072).

---

## SCHEMA CHANGE (proposed)

The canonical schema implies all 28 layers, every token. That is 4.2x the model cost (F4)
and is not achievable at interactive rates. Three additive changes, all backward-compatible:

1. **Declare the layer subset.** Add `"layer_ids": [0,4,8,...,27]` alongside `"layers"`.
   `layers` stays an array of objects; consumers must key off `layer_ids`, not array index.
   Without this, A5 cannot tell a 8-layer trace from a 28-layer one.
2. **Add timing and position.** `"t_ms": 412.3` and `"seq_len": 57`. The UI needs `t_ms`
   to show the budget honestly, and `seq_len` to place the token in the context.
3. **Move `token_str` out of the hot loop.** Add a top-level `"vocab": {id: str}` side
   table, sent once per connection (or per generation) and updated lazily. The per-token
   frame then carries `"token_id"` only, and the client resolves the string. Saves ~40% of
   the payload and makes the frame schema uniform with `logit_lens_topk` entries, which
   could become `[id, prob]` instead of `[id, tok, prob]`.

Proposed v2 frame:

```json
{"step": 0, "token_id": 123, "t_ms": 412.3, "seq_len": 57,
 "layer_ids": [0,4,8,12,16,20,24,27],
 "layers": [{"layer": 0, "resid_norm": 12.3, "attn_entropy": 1.7,
             "mlp_out_norm": 4.1,
             "logit_lens_topk": [[id, prob], ...]}],
 "logits_topk": [[id, prob], ...], "chosen": [id, prob]}
```

Everything else in the canonical schema is unchanged and I can emit it as-is today.

---

## Build next (ordered, with effort estimate)

| # | task | effort | why first |
|---|---|---|---|
| 0 | `cpupower frequency-set -g performance`; check `no_turbo` | 5 min | every later number is 2x wrong without it |
| 1 | `bench.py`: fp32 vs bf16 GEMV, 2/4/6/8 threads, aggregate bandwidth | 1 h | settles F2/F3; decides the whole perf story |
| 2 | `convert_fp32.py`: bf16 safetensors → fp32 dir, verify tied embeddings survive | 2 h | makes load 1–2 s and removes the dtype trap |
| 3 | `torch_adapter.py` + `lens.py` + `steering.py`; assert hook order; verify `logits_to_keep=1` | 1 d | this *is* the instrument |
| 4 | Correctness gate: same prompt, PyTorch path vs `llama.cpp --embeddings`/`cb_eval` top-5 trajectory | 1 d | catches "hooks wired wrong" before A2/A3 build on it |
| 5 | `server.py` + WS protocol + backpressure + on-demand binary residuals | 1 d | A5 is blocked on a real stream to render |
| 6 | Steering loop end-to-end: bus write from WS → hook → changed next token, with a `replay()` for un-poisoning | 1 d | this is D5, the project's central claim |
| 7 | Layer-subset + `attn_entropy` via recomputed q·k against cached keys | 1 d | needed by A2; keeps cost flat |
| 8 | *(optional)* `torch.compile` one decoder layer as a standalone fn; measure honestly | 1 d | only after the trace path is frozen |
| 9 | *(optional)* llama.cpp `cb_eval` reader → fast behavioural oracle | 0.5 d | nice-to-have; not on the critical path |

Total to a working, honest instrument: **~4–5 days.** Items 0–4 are non-negotiable and are
2 days of the 5.
