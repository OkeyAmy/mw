# mod_wei — shared context (READ FIRST, DO NOT EDIT)

**Project:** build a working "model microscope": inspect, trace, and steer an LLM's
internal computation *during* generation — not prompt engineering, actual
activation-level intervention.

**Target model:** `Qwen/Qwen2.5-Coder-1.5B` (the fp16/bf16 safetensors release,
NOT the Q4_K_M GGUF — see decision D2).

## MEMORY SAFETY (hard rule — the owner's laptop freezes otherwise)

- The box uses **zram swap** (compressed RAM). Under pressure it spirals instead
  of degrading; the machine hangs and needs a hard reboot.
- **Two QEMU VMs (~4.3 GB) run alongside the desktop.** Available RAM is well
  below the 31 GB total.
- Rules:
  1. Never load two models at once.
  2. Use `dtype=torch.float16` and `low_cpu_mem_usage=True`. fp32 doubles RAM and
     duplicates the state dict during load.
  3. For architecture/key/name-map checks use a **meta device**:
     `with torch.device("meta"): AutoModelForCausalLM.from_config(cfg)` — zero RAM.
  4. Call `mw.guard.require(path)` before any real load; it refuses unsafe loads.
  5. Nothing larger than ~2B on this box, and only after `ollama stop` and with
     VMs paused.
  6. Report any memory-related finding in your agent file.

## Hardware (hard constraint) — CORRECTED 2026-11-10 by A9, verified by orchestrator
- **Intel i7-8665U @ 1.90 GHz. 4 physical cores / 8 logical threads. 8 MB L3.**
  (Earlier "8 cores" was wrong — that was `nproc`, i.e. logical threads.)
- **AVX2 only. No AVX-512. No bf16 hardware support.**
- 31 GB RAM total, **~17 GB actually available** (14 GB used by the desktop).
- No GPU. 293 GB free disk. System python 3.14 → use `.venv` (3.12).
- Measured by A9: ~54–67 GFLOPS fp32 GEMM, ~9.7 GB/s streaming read.
- **Budget: ~1.9 tok/s at batch 1; 8–12 tok/s at batch 8.** bf16 buys nothing;
  use fp32 for fidelity. Batching is a precondition, not an optimisation.
- Consequence: interactive single-token streaming is slow. Design the UI around
  *batch traces* and *pause-and-inspect*, not 60fps live generation.

## Scale question (added round 1b)
The user also wants larger models (8B+) to run on this CPU box, ideally using
knowledge of the internals to make that possible. That is a second research axis:
- A7 = conventional CPU scaling (quantization, mmap, sparsity, offload, KV cache).
- A8 = interpretability-guided compression (use measured per-layer sensitivity to
  allocate bits / prune heads / skip layers), which is the novel contribution.
Rule: any compression we adopt must be **evaluated on our own harness** (does the
behavior survive? does the interpretability survive?), not just perplexity.

## Model facts (Qwen2.5-Coder-1.5B-Base)
- arch `Qwen2ForCausalLM`, 28 layers, hidden 1536, 12 query heads / 2 KV heads (GQA),
  intermediate 8960, SwiGLU, RMSNorm, RoPE, vocab 151936, ctx 32768, Apache-2.0.

## Decisions already made (do not re-litigate, but do challenge with evidence)
- **D1.** Do not try to make Ollama the instrument. Ollama only exposes tokens.
  Interpretability needs hidden states + hooks. Ollama stays as a *behavioral
  baseline* only.
- **D2.** Interpretability runs on the original bf16 safetensors. Q4_K_M is
  block-quantized and lossy; it is a valid *format* target (we must support it),
  but not the *science* target.
- **D3.** Primary runtime = PyTorch + HF `transformers` with forward hooks, with
  TransformerLens where it fits. `nnsight` is the fallback for intervention graphs.
- **D4.** Canonical abstraction = a `ModelAdapter` interface so any format
  (safetensors / GGUF / GPTQ / AWQ / ONNX / MLX) normalizes into the same trace
  schema. Formats differ in loading, not in the analysis layer.
- **D5.** "Mid-generation steering" is real and cheap: every generated token is a
  fresh forward pass, so a hook that edits the residual stream at layer L before
  that token's logits are computed *is* live steering. No retraining needed.

## Canonical trace schema (all backends must emit this)
```json
{"step": 0, "token_id": 123, "token_str": "def",
 "layers": [{"layer": 0, "resid_norm": 12.3,
             "logit_lens_topk": [[id, tok, prob], ...],
             "attn_entropy": 1.7, "mlp_out_norm": 4.1}],
 "logits_topk": [[id, tok, prob], ...], "chosen": [id, tok, prob]}
```

## Steering API (target shape)
```
steer(layer=14, vector=v, coef=2.0, positions="all"|"last"|"gen",
      schedule="constant"|"decay", until_step=n)
```
Implemented as `h[:, pos, :] += coef * v` inside a forward hook on layer `layer`.

## What "transparency" can and cannot mean (be honest)
- CAN: per-layer logit lens, attention patterns, residual-stream trajectories,
  causal attribution (patching), sparse features (SAE), and *verified* steering.
- CANNOT: a 1:1 "this neuron = this concept" map. Superposition and
  polysemanticity are real. Any claim of full transparency is false.
- Therefore every finding must be **causal**, not correlational: show that
  intervening changes the output in the predicted direction.

## Workspace protocol
- `CONTEXT.md` — this file. Orchestrator-owned. Read-only for agents.
- `LEDGER.md` — merged findings, ID'd `[F-xx]`. Orchestrator-owned.
- `agents/<agent-id>.md` — YOUR output file. Only you write it.
- `agents/<agent-id>.status` — one line: `running|done|blocked: <note>`.
- `code/` — working prototypes.
- Cite papers with title + year + arXiv/URL. No unsourced claims.

## Rule zero
If you cannot verify it, label it `UNVERIFIED`. If a technique needs a GPU, say so.
