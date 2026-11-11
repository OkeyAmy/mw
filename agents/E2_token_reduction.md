# E2 — The token-reduction landscape: what actually saves money, what only moves it

**Scope:** every real technique for reducing what an LLM app pays for tokens, with
measured savings and measured costs. Target reader: someone deciding where a product
could win, not someone shopping for a library.

**Method note.** Every number below is either (a) fetched from a primary source today
(pricing page, docs page, arXiv abstract) or (b) computed by me with the arithmetic
shown. `code/e2_token_measure.py` reproduces the token counts with the actual
Qwen2.5-Coder tokenizer. Paper *abstracts* are quoted as abstracts; I did not
reproduce their benchmark tables, and I say so where it matters.

---

## TL;DR (5 bullets)

1. **Prompt engineering is not fading — it is relocating.** The prose you can rewrite
   is nearly exhausted: correctness-gated prompt compression on a *production* agent
   shrank its whole prompt surface by only **3.5–7.6%** [2610.31972]. The remaining
   money is in structural artifacts — tool schemas, cache layout, output grammar,
   retrieved chunks — where the measured wins are **49–90%**, not 4%.
2. **Caching is the biggest free win, and it is bigger than the marketing says, but
   only if you engineer it.** Measured **41–80%** total API cost reduction on
   long-horizon agentic tasks [2601.06007]. Theoretical ceiling on a cached slice:
   **95%**. Real-world hit rate is **ρ ≈ 0.83**, not 1.0 [2607.15516], and
   mis-integration is endemic and silent (32 defects across 571 agent projects,
   18 upstream fixes [2610.34817]).
3. **The single most under-appreciated lever is the output grammar, not the input
   prose.** Measured on this box with the real tokenizer: a prose spec + 2 few-shot
   examples for JSON extraction costs **675 tokens**; the same task with a
   decoder-enforced schema costs **223** (**−67%**), and **156 → 16** (**−90%**) for
   enum classification. Quality is neutral-to-better for extraction.
4. **Prompt compression and caching fight each other.** A cached token costs 0.05–0.10×
   (≈0.30× at the *measured* ρ ≈ 0.83); a 5× compressor costs 0.20×. So compression
   needs roughly **r > 10 under an ideal cache, r ≈ 3–6 under a realistic one**, just to
   tie — and *query-aware* compressors, which rewrite the prefix on every call, get no
   cache discount at all. Measured on tau-bench: query-aware compression was **+40.1%
   more expensive than doing nothing** [2607.15516]. Compressing the *uncached tail* is
   the only version that wins by construction, because it competes against 1.0×, not 0.1×.
5. **Unclaimed ground = the volatile tail, plus verification.** Everything stable
   should be cached (free, lossless). Everything unique — fresh retrieved docs, user
   uploads, tool output — cannot be cached and is where compression still pays. No
   shipped library exposes "compress only the uncacheable suffix, never change the
   prefix hash". The one paper that does it (CAPC, Jul 2026) beats cache-only by
   **49%** and vanilla by **90%** — and is implemented nowhere.

---

## The framing that makes the rest of this document make sense

Three different things get called "saving tokens". They have different physics:

| What you cut | Price effect | Who controls it |
|---|---|---|
| **Tokens sent** (input compression, pruning, schemas) | linear, 1× per token removed | you |
| **Price per token** (caching, batch, model choice) | 0.05×–0.5× on the affected slice | provider (you choose layout) |
| **Tokens generated** (verbosity, effort, stop conditions) | **5×** per token, on every model I checked | provider knobs + your prompt |

The output row is the one nobody prices. Live 2026 list prices are uniformly **5:1
output:input** — Sonnet 5.5 ($2/$10), GPT-6.1-sol ($2/$10), Haiku 5.5 ($0.10/$0.50),
Fable 5.2 ($8/$40). One output token avoided is worth five input tokens avoided.
Every product in the "token reduction" category targets the input side.

---

## Findings

### F1. Prompt compression: real, measurable, and narrower than advertised

**What it is.** Drop tokens from the prompt before sending. Two families: *scoring*
(keep tokens with high information content) and *selection* (keep sentences/chunks).

| Method | Mechanism | Ratio claimed | Where it runs | Source |
|---|---|---|---|---|
| **LLMLingua** (2023) | budget controller + token-level iterative compression, small LM scores perplexity | "up to **20×** with little performance loss" | inference-time (small LM pass) | arXiv 2310.05736 |
| **LongLLMLingua** (2023) | question-aware + position-aware variant for long context | **2–6×**; +21.4% accuracy on NQ with ~4× fewer tokens; 94.0% cost cut on LooGLE | inference-time | arXiv 2310.06839 |
| **LLMLingua-2** (2024) | token classification with XLM-RoBERTa trained by distillation from GPT-4 | **2–5×**; 3–6× faster than prior compressors; 1.6–2.9× end-to-end latency | compressor is *trained*; target untouched | arXiv 2403.12968 |
| **Selective Context** (2023) | drop low self-information content | **50%** context cost; −36% memory; −32% time; −0.023 BERTScore, −0.038 faithfulness | inference-time | arXiv 2310.06201 |
| **RECOMP** (2023) | trained extractive/abstractive compressor for RAG docs | **as low as 6%** of original (≈16×) with "minimal loss" | trained per task | arXiv 2310.04408 |
| **Gist tokens** (2023) | fine-tune the model to compress a prompt into a few gist tokens | **up to 26×**, 40% FLOPs, 4.2% wall-time | **training-time** | arXiv 2304.08467 |

**Reading it honestly.** The 20× headline is real only on tolerant tasks. LLMLingua-2's
own abstract claims 2–5× — a 4–10× retreat from the v1 headline, by the same authors.
An independent 2026 evaluation of LLMLingua-2 at ~50% compression found summarization
"relatively robust" while "mathematical reasoning degrades substantially despite high
semantic similarity", and concluded failures are "driven by information omission rather
than semantic drift" [2605.17932]. That is the correct mental model: **compression
deletes reasoning steps, and no semantic-similarity metric catches it.**

**Confidence:** high on mechanism and on the 2–5× reliable range; medium on any
specific task's degradation (task-dependent, must be measured per app).
**Cost:** an extra model in the path (latency + ops), plus an eval harness you cannot
skip. On short prompts the compressor costs more than it saves.

---

### F2. Prompt caching: the biggest lever, and the one with the most silent failure

This is where the money is. Prices are live, fetched today.

**Anthropic** (per MTok, base input → cache read; write multipliers 1.25× for 5-min,
2× for 1-hour):

| Model | input | cache read | read multiplier |
|---|---|---|---|
| Fable 5.2 | $8 | $0.20 | **0.025×** |
| Opus 5.5 | $4 | $0.20 | 0.05× |
| Sonnet 5.5 | $2 | $0.10 | 0.05× |
| Haiku 5.5 (≤100K) | $0.10 | $0.01 | 0.1× |

**OpenAI** (GPT-5.6 and later): cache write **1.25×** for a 30-minute TTL, **2.5×** for
12 hours; cache read **0.1×**, and **0.05× on GPT-6.1 Sol/Terra**. Minimum cacheable
prefix: **1,024 visible tokens**. Earlier models: no write premium at all.

**The arithmetic (mine, shown):** 4,000-token static prefix, 100k calls/month, Sonnet 5.5:

```
no cache:  100,000 × 4,000 = 400 MTok × $2.00        = $800.00
hit 100%:  400 MTok × $0.10                          = $ 40.00   (save 95.0%)
hit  90%:  360 × $0.10 + 40 × $2.50 (writes)         = $136.00   (save 83.0%)
hit  70%:  280 × $0.10 + 120 × $2.50                 = $328.00   (save 59.0%)
hit  50%:  200 × $0.10 + 200 × $2.50                 = $520.00   (save 35.0%)
break-even: write premium 0.25×P vs 0.95×P saved/hit (read = 0.05×) = 0.26 hits
```

**So the break-even is 0.26 calls.** A prefix reused twice already wins. There is no
other technique in this document with that ratio.

**What the field actually measures.** A 2026 evaluation across OpenAI, Anthropic and
Google on DeepResearch Bench (500+ agent sessions, 10k-token system prompts) found
**41–80% API cost reduction and 13–31% TTFT improvement**, and — importantly — that
*naive full-context caching can paradoxically increase latency*; placing dynamic
content at the end and excluding dynamic tool results worked better [2601.06007].

**The part nobody quotes: ρ ≈ 0.83, not 1.0.** A July 2026 measurement on **Sonnet 4.6**
(whose cache-read multiplier is 0.1×) found the cache has a **two-tier architecture with a
sharp threshold near 3,500 tokens**; below it, hit rate plateaus at **ρ ≈ 0.83** over
30-call sessions [2607.15516]. So the "95%" number is a ceiling on the cached slice under
perfect conditions. My table above already prices this honestly: at ρ = 0.83 the effective
read cost is `0.83×0.10 + 0.17×1.25 = 0.2955×`, so you keep ~78% of the theoretical
saving. (On Sonnet 5.5's deeper 0.05× read it is `0.254×`, i.e. ~79%.)

**And it fails silently.** An empirical study of 120 prompt-caching PRs across 10
open-source agent systems found mis-integration in both request-side configuration and
read-side accounting, and that "most mis-integrations fail silently without causing
observable runtime failures". Their detector found **32 mis-integrations across 571
agent projects**, 21 developer confirmations, **18 upstream fixes including one in
Anthropic's official SDK** [2610.34817].

**Confidence:** high (primary pricing pages + two independent measurements).
**Cost:** near-zero quality cost. Effort is real: prefix layout, breakpoint placement,
the 20-block lookback window, the 1,024/512-token minimum, and the >15 req/min routing
threshold that scatters your cache across machines.

---

### F3. KV cache compression: saves memory and latency, **not billed tokens**

This is the section where a business case usually dies, so state it plainly:
**for an API user, KV compression reduces the provider's cost, not your bill.** It
matters only if you self-host (which is exactly this project's situation).

| Method | What it does | Measured effect | Source |
|---|---|---|---|
| **H2O** (2023) | keep heavy-hitter + recent KV, evict rest | 20% heavy hitters; up to **29×** throughput vs DeepSpeed/FlexGen; 1.9× lower latency | 2306.14048 |
| **StreamingLLM** (2023) | attention sinks + sliding window | stable to **4M tokens**; **22.2×** faster than sliding-window recompute | 2309.17453 |
| **SnapKV** (2024) | per-head clustered important positions from an observation window | **3.6×** generation speed, **8.2×** memory efficiency at 16K; 380K ctx on one A100-80GB | 2404.14469 |
| **KIVI** (2024) | 2-bit KV, per-channel keys / per-token values | **2.6×** less peak memory, 4× batch, 2.35–3.47× throughput, "almost the same quality" | 2402.02750 |
| **KVQuant** (2024) | sub-4-bit KV | premise: existing methods "fail to represent activations accurately in sub-4-bit" | 2401.18079 |

**Q8 vs Q4, computed for our model** (28 layers, 2 KV heads, head_dim 128, fp16):

```
elements/token = 2 (K,V) × 28 layers × 2 kv_heads × 128 head_dim = 14,336
fp16: 28,672 B/token = 28.0 KiB/token → 32k ctx = 896.0 MiB
q8  : 14,336 B/token = 14.0 KiB/token → 32k ctx = 448.0 MiB
q4  :  7,168 B/token =  7.0 KiB/token → 32k ctx = 224.0 MiB
weights (fp16) = 3,087,467,144 B = 2.88 GiB
```

At full 32k context the KV cache is **~30% of the weight footprint** — material on a
17 GB box, and the reason context length and batch size fight each other. Q8 halves it
essentially for free; Q4 halves it again and is where quality claims get contested
(KIVI reports 2-bit "almost the same" on *its* benchmarks; I did not reproduce that).

**Confidence:** high on memory arithmetic; medium on quality at 2-bit.

---

### F4. Fine-tuning / distillation to delete the prompt: correct only in one regime

**Mechanism.** Bake the system prompt + few-shot examples into weights. Variants:
gist tokens (training-time compression, 26×), prompt distillation for knowledge
injection [2412.14964], and plain LoRA on synthetic data.

**Evidence it works.** LoRA fine-tuning on **100 samples** of simulated data beat a
complex system prompt at hitting a required conversational tone, on both a 1B
open-weight model and two closed models, with no semantic-similarity loss [2507.04889].

**The economics nobody computes.** Fine-tuned inference is priced *above* base:
gpt-4.1-mini base $0.40/$1.60 per MTok, fine-tuned **$0.80/$3.20** — **2×**. So, with
removed prompt `P` and retained content `C`:

```
base:      P + C
fine-tune: 2 × C
wins iff   2C < P + C   →   C < P
```

**Rule: fine-tuning to remove a prompt pays only when the retained per-call content is
smaller than the prompt you deleted.** For a 675-token instruction + a 300-token
document, it wins (600 vs 975 token-equivalents). For a 675-token instruction + a
20,000-token RAG context, it is a 2× price increase on 97% of your tokens — it loses
badly. And on 2026 pricing pages, **OpenAI self-serve fine-tuning is closed to new
users** — the largest provider is exiting the market you'd be selling into.

**Confidence:** high on arithmetic and prices; medium on transferability of the 100-sample
result (single study, style task).
**Cost:** data generation, an eval set, retraining on every prompt change, loss of
per-tenant prompt flexibility.

---

### F5. RAG: the largest measured waste in the stack

**How much context is actually wasted.** Chroma's token-level evaluation is the
cleanest number available. With 5 retrieved chunks and `text-embedding-3-large`:

| Strategy | size / overlap | recall | **token precision** |
|---|---|---|---|
| Recursive | 200 / 0 | 88.1 | **7.0%** |
| ClusterSemantic | 200 / 0 | 87.3 | **8.0%** |
| Recursive | 400 / 0 | 89.5 | 3.6% |
| TokenText (the **OpenAI file-search default config**) | **800 / 400** | 87.9 | **1.4%** |

Source: Chroma technical report, *Evaluating Chunking Strategies for Retrieval*, Jul 2024.
Recall is roughly flat at ~88% while **precision varies 5.7× with chunking choice** —
and the default most people ship is the worst row.

**Read that carefully:** 7% token precision means ~93% of retrieved tokens are not
relevant tokens. That is the headroom. It does not mean 93% of tokens are useless (some
provide contrast/context), but it does mean the input side of RAG is mostly filler.

**Corroboration from compression work.** FIT-RAG saves "approximately half of the tokens
on average" *while* improving answering accuracy by 14.3% / 19.9% / 27.5% on
TriviaQA / NQ / PopQA with Llama2-13B-Chat [2403.14374]. LongLLMLingua gets 4× fewer
tokens with +21.4% accuracy on NQ [2310.06839].
RECOMP reaches 6% of original with minimal loss [2310.04408]. When deleting half the
context *improves* accuracy, the context was noise.

**What does not work.** Semantic chunking: "computational costs … are not justified by
consistent performance gains" [2410.13070]. Complex chunking at scale: "computationally
expensive methods rarely provide consistent gains over simpler chunking" [2608.16586].
Reranking and *extractive* pruning are where the wins are; clever segmentation is not.

**Confidence:** high on the direction and the ~50% figure; the exact precision number is
metric-specific (token-level IoU on synthetic queries) and should be read as an upper
bound on waste, not a promise.

---

### F6. Structured output / constrained decoding: the under-appreciated real saver

**What it is.** Move the output contract out of prose and into the decoder: JSON Schema
mode, Outlines FSM indexing, Guidance, llama.cpp GBNF grammars, Anthropic tool schemas.

**Measured, on this box, with the real Qwen tokenizer** (`code/e2_token_measure.py`):

| Task | Prose spec + few-shot | Schema/grammar-constrained | Saving |
|---|---|---|---|
| Invoice extraction (8 fields, nested array) | **675** tok | **223** tok | **−67.0%** |
| Support-ticket classification (10 labels, precedence rules) | **156** tok | **16** tok | **−89.7%** |

The classification case is the striking one: 140 of 156 tokens existed only to explain
*which string to emit*. An `enum` makes that explanation unnecessary — the invalid
strings are unrepresentable.

**And it compounds with caching.** The schema is static; put it in the cached prefix and
the marginal per-call cost collapses to the user text + a 5-token instruction. The 223
figure is the *uncached* number. Exact split: `5 (instruction) + 151 (schema) + 67 (user)`.

**Quality: the honest picture.** Format restriction genuinely hurts some reasoning:
Tam et al. found "a significant decline in LLMs reasoning abilities under format
restrictions" and that "stricter format constraints generally lead to greater
performance degradation" [2408.02442]. But a 2026 causal re-examination across four
model families and eleven models found "apparent performance degradation largely stems
from non-equivalent prompting and output formats rather than from decoder-level masking",
and that validity constraints *improve* reliability [2610.35328]. Two fixes exist and
both work: reason first, constrain after (CRANE, up to +10 points [2502.09061];
DCCD, +24 points, 15.2→39.0 on GSM8K-1B [2603.03305]).

**Practical rule:** constrain *extraction and classification* freely. For reasoning,
use a two-phase decode (free-form reasoning, then constrained answer) — this costs
nothing in prompt tokens and removes the failure mode.

**Confidence:** high on token counts (measured); high on the "safe for extraction"
conclusion; medium on reasoning (task-dependent).

---

### F7. Tool and function calling: the biggest absolute token sink in agent stacks

Not "tool calling saves tokens" — **tool definitions are the single largest avoidable
block in most agents.** From Anthropic's own measurements [advanced-tool-use, Nov 2025]:

```
GitHub   35 tools ≈ 26K tokens
Slack    11 tools ≈ 21K tokens
Sentry    5 tools ≈  3K tokens
Grafana   5 tools ≈  3K tokens
Splunk    2 tools ≈  2K tokens
58 tools          ≈ 55K tokens before the conversation starts
Jira alone        ≈ 17K tokens
worst case seen at Anthropic: 134K tokens
```

**Fixes, with measured results:**

| Technique | Measured saving | Quality effect |
|---|---|---|
| **Tool Search Tool** (`defer_loading: true`, search on demand) | 72K → 8.7K prefix, **−85%**, preserves 95% of context | **improves**: Opus 4 49%→74%, Opus 4.5 79.5%→88.1% |
| **Programmatic tool calling** (tools called from code) | 43,588 → 27,297 tokens, **−37%** on complex research tasks | accuracy up (GIA 46.5%→51.2%) |
| **Code execution with MCP** (tools as a file tree) | 150,000 → 2,000 tokens, **−98.7%** | — |
| **Tool Use Examples** (`input_examples`) | **adds** tokens | 72%→90% parameter accuracy |

Two things follow. First, the largest wins in the whole landscape are *structural*
(defer, don't describe). Second, "tool examples improve accuracy" is a *token-costing*
technique — it belongs on the other side of the ledger, and the two are routinely
conflated.

**Confidence:** high (vendor-measured, with accuracy numbers, and the mechanism is
inspectable). **Commoditized:** yes on Anthropic and OpenAI APIs as of late 2025 /
2026 (`defer_loading`, tool search, `allowed_tools`); **no** for self-hosted
vLLM/llama.cpp/Ollama, where nothing equivalent ships.

---

### F8. Steering vectors / activation control: where it fits (one paragraph, deferring to E1)

Steering is the only technique here that removes prompt tokens *without* removing
information from the context — it replaces an instruction with a direction in activation
space. The evidence that it works is solid: ITI moved Alpaca's TruthfulQA score from
32.5% to 65.1% by shifting activations at a few attention heads [2306.03341]; CAA
averages residual-stream differences between positive and negative example pairs and
adds the result at inference with a signed coefficient [2312.06681]; RepE generalises
this to population-level representations for monitoring and control [2310.01405]. But
the economics are the opposite of every other technique in this document: it requires
white-box access to weights and residual stream, so **it saves zero tokens for anyone
calling an API**, and it trades an auditable prompt for an unauditable vector — which is
a governance regression in any regulated setting. Its honest role is not "cheaper
prompting"; it is *runtime behaviour control on models you own*, where the alternative
is a retrain. Coordinate with E1 on mechanism and validation; do not double-count it as
a token-saving product.

---

## The honest ranking

Ranked by **(savings × ease × reliability)** for an app spending $X/month on input
tokens. Savings are of *total* input cost unless stated. Ranks 1–3 are additive and
routinely compound to 90%+.

| # | Technique | Typical measured saving | Quality cost | Effort | Commoditized? |
|---|---|---|---|---|---|
| 1 | **Prefix caching, engineered** | 41–80% of total API cost (agentic); 95% ceiling on the cached slice; break-even at 0.26 reuses | **none** | 1 day to start, 1–2 wks for ρ>0.8 | **Yes** at API level; **No** as discipline/tooling |
| 2 | **Schema/grammar-constrained output** replacing format prose + few-shot | **−67%** extraction, **−90%** classification (measured) | neutral for extraction; risky for reasoning → draft-then-constrain | 1–3 days | **Yes** (OpenAI/Outlines/GBNF) |
| 3 | **Tool-definition deferral / dynamic loading** | **−85%** of a 72K tool prefix; −98.7% in code-exec mode | **improves** accuracy (49→74%) | 1–2 wks | **Newly yes** on frontier APIs; **No** self-hosted |
| 4 | **Cache-aware query-agnostic compression (CAPC)** | **−49%** vs cache-only, −90% vs vanilla; −51.7% on a 94K schema prefix | within 0.05 of baseline | 2–4 wks | **No** — paper only (Jul 2026) |
| 5 | **RAG pruning / reranking / extractive compression** | ~**−50%** tokens *with* accuracy gains; retrieval token precision 1.5–8% | often **positive** | 1–3 wks | **Partly** (rerankers yes, pruning no) |
| 6 | **Query-aware prompt compression** (LLMLingua family) | 2–5× reliably; 20× only on tolerant tasks | real on reasoning; **negative ROI under caching (+40.1%)** | 1 wk | **Yes** as OSS library |
| 7 | **Fine-tune / distill the prompt away** | 100% of removed prompt, but **2× price** on what remains | needs eval; wins iff `C < P` | 2–6 wks + data | **Partly**; OpenAI closed self-serve FT |
| 8 | **KV cache compression** | 2–4× memory, 1.9–3.6× speed | ~0 at 8-bit, contested at 2-bit | 1–2 wks | **Yes** in vLLM/SGLang |
| 9 | **Steering vectors** replacing instructions | removes instruction tokens entirely | drift; white-box only | research | **No** |

**Read the table this way:**

- **Commoditized (do not build a company here):** 2, 3 (frontier APIs), 6, 8. These are
  features of a provider or a pip-installable library. Competing means competing with
  Microsoft Research and the model vendors on their own surface.
- **Commoditized-but-botched (services opportunity):** 1. The primitive is free; the
  *hit rate* is not. ρ ≈ 0.83 and 32 silent defects in 571 projects say the discipline
  is missing, not the feature.
- **Open (product opportunity):** 4, 5, 7-in-the-tail, and the whole output side.

---

## Where the unclaimed ground actually is

I checked whether my first three candidate ideas were already taken. Two were:

- "Compression is dominated by caching" — **taken**, and better than my version:
  2607.15516 derives the crossover (query-aware compression only wins at r ≥ 6 under
  realistic ρ) and measures it.
- "Cache mis-integration is silent and pervasive" — **taken** as a detector:
  CACHELINT, 0.71 precision / 0.78 recall, 32/571 projects [2610.34817].
- "Compaction destroys cache value" — **discussed in blogs, not measured.** Decagon's
  glossary states it plainly ("every compaction pass invalidates that prefix"); no
  paper prices it per turn. This one is still soft.

So the ground that remains is narrower and more specific than "save tokens":

### G1. The volatile tail is the only place compression still pays — and nothing targets it

This is the sharpest opening, and it follows from the arithmetic. Two regimes, and the
difference between them is the whole business case:

```
REGIME A — compress the STABLE PREFIX (what LLMLingua does)
  cached token       0.05–0.10×           (ρ=1 ideal)
  cached token       ≈0.30×               (at measured ρ≈0.83 on a 0.1×-read model:
                                            0.83×0.1 + 0.17×1.25; 0.25× on Sonnet 5.5)
  compressed token   1/r × 1.00×
  → must reach r ≈ 3.4 (realistic) to r > 10 (ideal) merely to TIE
  → and query-aware compressors change the prefix, so they get NO discount:
    they pay 1/r × 1.00× + a fresh write, and measured +40.1% vs doing nothing

REGIME B — compress the VOLATILE TAIL (what nothing ships)
  tail token         1.00×                (uncacheable by definition)
  compressed token   1/r × 1.00×
  → r=2 already halves it; r=3 saves 67%; no cache interaction, no tier risk
```

Everything *stable* should be cached and never compressed. Everything *unique* — the
fresh 50K-token document, the user's upload, the tool result, yesterday's transcript —
cannot be cached and is the only place compression has a business case by construction.
And that is exactly where the waste is largest (retrieval token precision 1.4–8.0%).

**The gap:** no shipped library exposes the split. LLMLingua takes a prompt and returns
a prompt; query-aware variants change the prefix *every call*, mechanically destroying
the cache. CAPC proved the right design in July 2026 — query-agnostic compression, an
explicit `cache_control` breakpoint, and a "tier-preserving ratio bound" so you don't
compress the prefix below the ~3,500-token hot tier — and got 49% over cache-only with
quality within 0.05. **It is implemented nowhere.** That is a paper with a product-shaped
hole behind it.

### G2. Cost is never measured per successful task

Every paper above reports tokens, dollars, or accuracy. CAPC's headline is "cheapest in
16/16 configurations … quality within 0.05". "Within 0.05" is a quality *cost*, and
nobody converts it into the only number a buyer cares about: **dollars per completed
task at fixed success rate**. That conversion is what turns a compression library into a
budget controller, and it is missing from the literature and from every product I found.

### G3. The compaction/cache trade-off has no policy, only a warning

Blogs warn that compaction invalidates the prefix. Providers warn that "summarization,
compaction, or context truncation can change the prefix and reset cache reuse" (OpenAI
prompt-caching guide, fetched). Nobody ships the decision. Here is the arithmetic that
makes it a product rather than a footnote (Sonnet 5.5 prices, 40K history compacted to 8K):

```
trim 32,000 tokens
  saved per turn, no cache:    $0.06400
  saved per turn, with cache:  $0.00320      ← 20× less
  one-off cost: summarise $0.0900 + rewrite prefix $0.0200 = $0.1100
  break-even, no cache:          1.7 turns
  break-even, with cache:       34.4 turns
```

**Caching inverts the economics of compaction by 20×.** A team that "optimises context"
on turn 5 of a 30-turn session is buying a $0.11 bill to save $0.003/turn and deleting
information — negative value on both axes. The unclaimed product is a per-turn planner
that prices *keep / compact / retrieve* against the current model's cache tiers and
holds task success fixed. Providers will not build it (it tells you to send *more*
tokens); compression vendors will not build it (it tells you their product is
counter-productive on stable prefixes).

### G4. Tool economics for self-hosted inference (directly relevant to mod_wei)

Anthropic solved 94K-token schema prefixes for *their* API with Tool Search. On
vLLM/llama.cpp/Ollama there is no equivalent — and for a small model it is not a cost
problem, it is a feasibility problem: 55K tokens of tool definitions **exceed our 32K
context window outright**, so the agent cannot be built at all. Even at a shorter prefix,
prefill is not free: `2 × 1.54e9 params × 55,000 tok = 1.7e14 FLOP`, which at A9's
measured ~60 GFLOPS fp32 is **~47 minutes of prefill per call** (and that assumes a
perfectly compute-bound GEMM; expect worse). Note this is a *prefill* estimate — the
1.9 tok/s figure in CONTEXT.md is decode, which is bandwidth-bound and much slower per
token. Dynamic tool loading is a *precondition* for local agents, not an optimisation.
Nothing ships it for local stacks.

**My call:** G1 is the best product thesis (clear mechanism, published proof, zero
implementation, measurable). G3 is the most defensible *insight* and the easiest to
demo. G4 is the one this project can actually build and dogfood on the 1.5B box.

---

## Concrete implementation

### 1. Measure before you optimise (done — reproduce it)

`code/e2_token_measure.py` counts real tokens with the Qwen2.5-Coder tokenizer. The
substance is four `len(tokenizer.encode(...).ids)` counts and a delta. Run it on *your*
prompts, not mine — the 67%/90% figures are specific to a format-heavy instruction block,
and a prompt with no format prose has nothing to save.

### 2. A prompt-surface compiler (the thing I would actually build)

Treat the prompt as a build artifact with four sections, and never let them mix:

```
[ IMMUTABLE ]  system persona + tool schemas + output schema     → cache_control here
[ STABLE    ]  per-tenant config, policy docs (changes daily)     → second breakpoint
[ VOLATILE  ]  retrieved chunks / uploads / tool results          → compress THIS only
[ TURN      ]  the user message                                   → never cache
```

Then enforce, mechanically:

```python
# cache-layout lint: run in CI, fail the build
def check(prefix_tokens, first_breakpoint, model):
    assert prefix_tokens >= model.min_cacheable      # 1024 OpenAI / 512–4096 Anthropic
    assert first_breakpoint <= prefix_tokens         # breakpoint on the last STABLE block
    assert not has_timestamp(prefix_tokens)          # a timestamp in the prefix = 0% hit
    assert tools_order_is_stable()                   # reordering tools invalidates everything
    assert blocks_since_last_write < 20              # Anthropic lookback window
```

Three concrete traps this catches, all documented: a timestamp in the prefix (pays a
write on every request, never a read), a breakpoint on the varying block (same), and a
tool list that changes per request (invalidates tools + system + messages at once).

### 3. Tier-aware tail compression

```python
def compress_tail(volatile_text, budget_ratio=0.5):
    # precondition: volatile_text lives strictly BELOW the last cache breakpoint
    # query-AGNOSTIC: same input → same output, so no prefix hash ever moves
    # extractive only: cannot hallucinate, and parallelises across chunks
    return extractive_select(volatile_text, keep=budget_ratio)
```

Extractive, not abstractive: it cannot hallucinate, and it is parallelisable. Abstractive
compression costs a generation pass and re-introduces the latency you were avoiding.

### 4. The cost model (put this in the product, not in a blog post)

```python
cost = (plain*P + cached*P*r + writes*P*w + out*P*5) / 1e6   # P = $/MTok input
```
with `r` = 0.05/0.1 and `w` = 1.25/2.5. Report **$/successful task**, not $/1k tokens.
The second metric hides every trade-off in this document.

---

## What NOT to do (traps)

1. **Do not compress a stable prefix.** A 5× compressor makes a token cost 0.20×; a cache
   hit makes it cost 0.05–0.10×. You lose 2–4× on price *and* you break the cache. This
   is the single most common wrong move in the category.
2. **Do not put anything variable before your cache breakpoint** — a timestamp, a user
   id, a request id, a reordered tool list. You will pay write pricing forever and never
   see a read.
3. **Do not quote "up to 20×" compression without the task.** The same authors retreated
   to 2–5× in LLMLingua-2. Reasoning tasks degrade in ways BERTScore cannot see.
4. **Do not put a format spec in prose and then also constrain the decoder.** You pay
   for the explanation and get none of the benefit.
5. **Do not constrain reasoning directly.** Reason free-form, then constrain the answer
   (CRANE / DCCD). DCCD recovers up to **24 points** over standard constrained decoding
   (15.2→39.0 on GSM8K-1B) purely by drafting before masking.
6. **Do not fine-tune to remove a prompt when the retained context is larger than the
   prompt.** You are paying 2× on the big number to save 1× on the small one.
7. **Do not count KV compression as a token saving for API users.** It is a provider-side
   cost. It only shows up on your P&L if you self-host.
8. **Do not trust a cache hit rate you have not measured.** ρ ≈ 0.83 is the realistic
   plateau below ~3.5K tokens on Sonnet; the 95% figure is a ceiling, not a plan.
9. **Do not ship a prompt change without a cost regression test.** 32 silent
   mis-integrations in 571 projects is the base rate for "it looked fine".

---

## Open questions for other agents

- **A7 / A8:** the KV arithmetic above says 32k context costs 896 MiB fp16 against
  2.88 GiB of weights. Does KV compression buy *context length* or *batch size* on this
  box, and which one does the 1.9 tok/s / 8–12 tok/s budget actually need? The batching
  constraint in CONTEXT.md makes this a real fork.
- **E1 (steering):** can a steering vector replace a ~200-token style/format instruction
  at equal reliability *as measured by a held-out eval*? If yes, that is the only
  technique that removes instruction tokens without removing information — but only for
  models we own. Worth one experiment, not a product.
- **A4 (runtime):** if we build the prompt-surface compiler, does it belong in the
  `ModelAdapter` layer or as a separate pre-processing stage? It is orthogonal to the
  trace schema, so I would keep it out of the adapter.
- **Orchestrator:** do we want a `cost_trace` alongside the activation trace in the proof
  harness — i.e. per-step token accounting so a steering experiment can report
  "behaviour changed *and* cost changed by X"? Cheap to add, and it makes the instrument
  answer a business question as well as a scientific one.

---

## SCHEMA CHANGE

**None.** This agent does not touch the canonical trace schema. The cost model above is
deliberately orthogonal: it consumes token counts, not activations, and can be computed
without the runtime existing.

---

## Validation blocks

Per `VALIDATION.md`. Cited-paper numbers are marked `[CITED-ABSTRACT]` — fetched today,
quoted from the abstract, **not** independently reproduced.

```
HYPOTHESIS: A decoder-enforced schema costs materially fewer prompt tokens than the
            equivalent prose format spec + few-shot examples.
TEST:       code/e2_token_measure.py, real Qwen2.5-Coder tokenizer (tokenizer.json,
            7,031,645 B, fetched from HF), counting the exact prompt strings.
RESULT:     extraction 675 → 223 tok (−67.0%); classification 156 → 16 tok (−89.7%);
            10 verbose tool defs = 1,410 tok/call.
VERDICT:    CONFIRMED
FALSIFIER:  the schema variant counting >= the prose variant, or the delta vanishing
            when the prose block contains no format description.
```

```
HYPOTHESIS: Prefix caching dominates every other input-side technique on $ saved per
            unit of effort, and breaks even after fewer than one reuse.
TEST:       arithmetic at live Sonnet 5.5 prices ($2 in / $2.50 write / $0.10 read),
            4,000-token prefix, 100k calls/mo; break-even = write premium / per-hit saving.
RESULT:     no cache $800.00; hit 100% $40.00 (−95.0%); 90% $136.00 (−83.0%);
            70% $328.00 (−59.0%); 50% $520.00 (−35.0%); break-even 0.26 hits.
VERDICT:    CONFIRMED (arithmetic); external corroboration 41–80% [CITED-ABSTRACT 2601.06007]
FALSIFIER:  a provider pricing scheme where cache reads cost >= input, or a measured
            hit rate so low that write pricing dominates. Solving
            1.25 - 1.15ρ > 1.0 puts that crossover at ρ < ~0.22 — far below the
            measured 0.83, which is why the conclusion is robust.
```

```
HYPOTHESIS: Query-aware prompt compression is dominated by caching, because it rewrites
            the prefix and therefore never earns the cache-read discount.
TEST:       cost per original token, prefix regime: cache read 0.05–0.10× (ideal) and
            ≈0.30× at measured ρ≈0.83 vs compressed 1/r; solve for the tie point.
            Tail regime: uncached 1.00× vs 1/r.
RESULT:     prefix: tie at r ≈ 3.4 (realistic) to r > 10 (ideal); query-aware
            compressors get no discount and pay 1/r × 1.00× plus a write.
            tail: r=2 halves cost, break-even at r=1 (any compression wins).
            External: crossover reported at r >= 6 under their fuller model, and
            query-aware compression measured +40.1% vs vanilla on tau-bench
            [CITED-ABSTRACT 2607.15516]; CAPC r=3 on a 94K schema prefix = −51.7%.
VERDICT:    CONFIRMED — and NOT novel; 2607.15516 published the crossover first. What
            is novel here is only the framing (prefix vs tail), which follows from it.
FALSIFIER:  a workload where compression raises cache hit rate, or where the tail is
            cacheable (chunk-level KV reuse, e.g. CacheBlend 2405.16444, does exactly
            that — but only inside a serving system, not via an API).
```

```
HYPOTHESIS: Under caching, context compaction has a much longer payback period than
            under no caching, and is often value-destroying.
TEST:       arithmetic, Sonnet 5.5, 40K history → 8K, summarisation 40K in + 1K out,
            new prefix written at 1.25×.
RESULT:     saves $0.064/turn uncached vs $0.0032/turn cached (20× less);
            one-off $0.110; break-even 1.7 turns uncached vs 34.4 turns cached.
VERDICT:    CONFIRMED (arithmetic). Mechanism is documented by OpenAI's caching guide
            ("summarization, compaction, or context truncation can change the prefix and
            reset cache reuse"); the per-turn pricing is my own.
FALSIFIER:  a session shorter than the break-even where compaction still reduces total
            spend — or a provider that preserves cache across compaction.
```

```
HYPOTHESIS: KV cache size is a material fraction of the model footprint at full context.
TEST:       elements/token = 2 × 28 layers × 2 kv_heads × 128 head_dim = 14,336;
            × 2 B (fp16) = 28,672 B/token; × 32,768 tokens.
RESULT:     896.0 MiB at 32k fp16 (448.0 MiB q8, 224.0 MiB q4) vs 2.88 GiB weights
            → KV ≈ 30% of weights at full context.
VERDICT:    CONFIRMED
FALSIFIER:  a config.json with different num_key_value_heads or head_dim than
            CONTEXT.md states (2 KV heads, hidden 1536, 12 query heads).
```

**What I could not verify and am labelling UNVERIFIED:**
- Whether OpenAI bills the JSON Schema as input tokens in every API mode (Anthropic
  documents tool definitions as cacheable/billed input; OpenAI's rendered-prefix
  behaviour is documented for tools, not explicitly for `text.format` schemas).
- KIVI's "almost the same quality" at 2-bit on tasks other than its own benchmark.
- Whether any commercial product already implements CAPC-style tier-aware compression —
  I searched and found none, but absence of evidence at this search depth is not proof.

---

## Build next (ordered, with effort estimate)

| # | Step | Effort | Why first |
|---|---|---|---|
| 1 | Run `e2_token_measure.py` against our own agent prompts (the skill catalogue in this session is a natural test corpus: its per-skill descriptions are a large, cacheable, rarely-changing prefix) | 1 h | establishes whether *our* bloat is format prose, tool defs, or history |
| 2 | Cache-layout lint: 5 assertions, run in CI | 1 day | highest $ per hour of work in this document |
| 3 | Move output contracts to schema/grammar; delete the prose | 1–2 days | measured −67% to −90% on the instruction slice |
| 4 | Measure ρ on a real workload before quoting any caching number | 2–3 days | ρ ≈ 0.83 changes every business case |
| 5 | Tail-only extractive compressor with a prefix-hash invariant | 1–2 wks | the G1 gap; novel, small, demonstrable |
| 6 | `$/successful task` instrumentation in the harness | 2–3 days | converts every finding above into a decision |
