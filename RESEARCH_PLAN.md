# RESEARCH_PLAN — round 1

Six independent agents, run in parallel. Each owns a disjoint slice.
Read `CONTEXT.md` first. Write ONLY your own file in `agents/`.

| id | file | owns | must answer |
|----|------|------|-------------|
| A1 | `agents/A1_formats.md` | weights, formats, extraction | how do we get exact fp16 weights out of ANY format, incl. Q4_K_M GGUF and Ollama's store? |
| A2 | `agents/A2_interp.md` | what to compute | which interpretability methods are implementable on a 1.5B CPU model, in what order, with what math? |
| A3 | `agents/A3_steering.md` | how to change behavior | how do we build + validate a steering vector, and inject it live mid-generation? |
| A4 | `agents/A4_runtime.md` | engineering | what runtime/hook architecture survives CPU-only, and how does per-token streaming work? |
| A5 | `agents/A5_ui.md` | presentation | what should the "brain view" show, and what existing tools can we reuse? |
| A6 | `agents/A6_limits.md` | skeptic / validation | what is impossible, how do we avoid fooling ourselves, how do we score success? |
| A7 | `agents/A7_cpu_scale.md` | CPU inference at scale | how do we actually run 8B / 14B / 32B / 70B on 8 cores + 31 GB RAM? what fits, how fast, at what quality cost? |
| A8 | `agents/A8_guided_compression.md` | interpretability-guided compression | can seeing the activations tell us *which* weights/layers/heads to compress, so a big model fits without blind quantization? |

## Cross-agent interface (agreed now, so outputs compose)
- Everyone emits to the **canonical trace schema** in `CONTEXT.md`.
- Everyone assumes the **`ModelAdapter`** abstraction: `load()`, `forward_trace()`,
  `attach_steering()`. A1 defines loaders, A4 defines the runtime, A3 defines steering,
  A2 defines what goes inside the trace, A5 renders the trace, A6 tests the trace.
- If you need to change the schema, propose it in your file under "SCHEMA CHANGE".

## Output format for every agent file
```
# A<n> — <title>
## TL;DR (5 bullets)
## Findings
### F1. <claim>  [source: title, year, link]
...  (each with: what it is, why it matters, confidence, cost)
## Concrete implementation (code sketches, real APIs)
## What NOT to do (traps)
## Open questions for other agents
## SCHEMA CHANGE (if any)
## Build next (ordered, with effort estimate)
```
