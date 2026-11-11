# E1 — Can activation steering replace system prompts, and thereby save tokens?

**Scope:** the project's central technical thesis. The user's claim is *"prompt engineering is
fading; we need a better way to save tokens."* This file evaluates that claim and does **not**
validate it by default.

**Stance:** I tried to kill it, then tried to rescue it, then tried to kill it again. The result is
not "no" — it is a **precise, narrow, and much smaller yes**, and the "save tokens" half of the
claim is **arithmetically false** for every workload I could construct.

**Citation hygiene.** Every source below was fetched during this session (arXiv abstract pages,
arXiv HTML, PDFs via `pdftotext`, GitHub raw files, the GitHub REST API, and vendor pricing pages).
Quotes are verbatim. Where I could not fetch a page directly I say so. Numbers I computed are in
`code/e1_economics.py`, which prints its own arithmetic.

---

## TL;DR (5 bullets)

- **The economics do not work, and this is the decisive finding.** A system prompt is **0.5–6% of
  the bill** in every realistic app I could construct, and **0.05–0.6%** once the prefix is cached.
  Output tokens cost **10×** input tokens and cached input costs **0.1×** input. Deleting a
  300-token system prompt entirely — assuming a *perfect, free* vector — saves **5.9%** of an
  uncached chat bill and **0.62%** of a cached one. In the measured coding-agent workload the system
  prompt was **110 tokens**; tool schemas were **1,602**. There is nothing here to save.
- **The direct benchmark already ran, and prompting won.** AxBench (2025), 500 concepts, Gemma-2-2B
  and 9B, steering factor tuned *per concept to each method's own optimum*: **prompt 0.905** mean
  steering score vs **diff-in-means 0.409**, SAE 0.315, LAT 0.223. "For steering, we find that
  prompting outperforms all existing methods, followed by finetuning." That is a 2.2× gap against
  the exact estimator this project is building, with the vector given every advantage.
- **But the thesis is *mechanistically* true and *partially* true behaviourally.** A system prompt
  *is* approximately a direction: subliminal-learning work (2026) finds "the teacher's system prompt
  is well approximated by a steering vector". And CAA's own Table 3 shows steering moves behaviours
  that a system prompt does *not* move (Hallucination: prompt 0.52 vs no-prompt 0.54 — nothing;
  steering 0.78). So: **where the prompt works, the vector loses; where the prompt doesn't work,
  there are no tokens to save.**
- **"Reusable vector" is the weakest sub-claim.** Soft prompts are learned per task, per model, per
  initialization; below ~11B parameters they still lag fine-tuning badly (SPoT, ACL 2022), and our
  target is 1.5B. Cross-model transfer needs a *trained projector*. And Hewitt (Nov 2026) shows one
  direction steers four different tasks at four different norms — so the artifact is not even a
  self-describing specification of one behaviour.
- **Deployment is real but narrow.** `llama.cpp` mainline supports `--control-vector` and `qwen2.cpp`
  calls `build_cvec()` per layer — so our exact target model *can* be steered in a production
  server today. Mainline vLLM **cannot**: PR #7906 (control vectors) was **closed, not merged**.
  Hosted APIs cannot at all. And the one company that productised this — Goodfire — says
  "feature steering **enhances rather than replaces** prompting", and **archived its SDK**
  (`archived: true`) while deprecating the API in Feb 2026.

**Verdict: a partial truth with a dead economic premise.** Details in §Verdict.

---

## Findings

### F1. The economics: the system prompt is not where the money is, and the arithmetic is not close

**Sources (fetched):** OpenAI pricing table, `https://developers.openai.com/api/docs/pricing`
(read 2026-11-10); Anthropic pricing table,
`https://platform.claude.com/docs/en/about-claude/pricing` (read 2026-11-10).

**Verified price structure** (verbatim from the tables):

| model | input /MTok | cached input /MTok | output /MTok | output ÷ input |
|---|---|---|---|---|
| `gpt-5` | $1.25 | $0.125 | $10.00 | **8.0×** |
| `gpt-5.4-nano` | $0.20 | $0.02 | $1.25 | 6.3× |
| `gpt-6-luna` | $0.10 | $0.01 | $0.50 | 5.0× |
| `gpt-6-astra` | $10.00 | $1.00 | $50.00 | 5.0× |
| Claude Sonnet 5.5 | $2 | $0.10 | $10 | 5.0× |
| Claude Haiku 5.5 (≤100K) | $0.10 | $0.01 | $0.50 | 5.0× |
| Claude Fable 5.1 | $10 | $0.25 | $50 | 5.0× |

Anthropic states the multiplier explicitly: *"A cache hit costs 10 percent of the standard input
price"*, and *"On Claude Fable 5.2, Claude Mythos 5.2, Claude Fable 5.1, and Claude Mythos 5.1, a
cache hit costs 2.5 percent of the standard input price."*

**Two structural facts follow, and they are the whole argument:**

1. **Output ≫ input.** One output token buys 5–10 input tokens. Anything that shortens the *output*
   is worth 5–10× anything that shortens the *input*. A vector that makes the model write 5% fewer
   tokens beats a vector that deletes the entire system prompt.
2. **Cached input ≪ input.** A stable system prompt is *already* priced at 1/10 of input, i.e.
   1/50 to 1/100 of an output token. Prompt caching is not a competitor to the thesis; it is the
   thesis, executed by the vendor, for free, on every model, with no hidden-state access.

**The arithmetic** (`python3 code/e1_economics.py`, output reproduced):

```
PART 1 — what fraction of the bill IS the system prompt?
app                                         S     ctx   out  S% input  S% $ (nocache)  S% $ (cached)
chat assistant                            300     800   400    27.27%           5.88%          0.62%
RAG question answer                       500    6000   500     7.69%           4.35%          0.45%
coding agent (ACL'26 measured)            110   12000   900     0.91%           0.52%          0.05%
doc analysis / long ctx                  1000  100000  1000     0.99%           0.90%          0.09%
'heavy prompt' agent                     5000    5000   500    50.00%          33.33%          4.76%
```

Read the `S% $` columns: even when the system prompt is **27% of all input tokens** (the chat case),
it is **5.9% of the dollars**, because output is 8× the price of input. And at 0.1× cache pricing
it is **0.62%**.

```
PART 2 — what does a 100% / 90% / 50% system-prompt reduction actually save?
chat assistant  (S=300, ctx=800, out=400)
   uncached        50% cut ->  2.94% of bill |  90% cut ->  5.29% of bill | 100% cut ->  5.88% of bill
   cached prefix   50% cut ->  0.31% of bill |  90% cut ->  0.56% of bill | 100% cut ->  0.62% of bill
coding agent (ACL'26 measured)  (S=110, ctx=12000, out=900)
   uncached        50% cut ->  0.26% of bill |  90% cut ->  0.47% of bill | 100% cut ->  0.52% of bill
   cached prefix   50% cut ->  0.03% of bill |  90% cut ->  0.05% of bill | 100% cut ->  0.05% of bill
```

In dollars, at 1M requests/month on `gpt-5` list prices:

```
chat assistant          uncached  $ 5,375/mo -> delete prompt entirely: $ 5,000/mo (saves $375/mo = 6.98%)
chat assistant          cached    $ 5,038/mo -> delete prompt entirely: $ 5,000/mo (saves $ 38/mo = 0.74%)
coding agent            uncached  $24,138/mo -> delete prompt entirely: $24,000/mo (saves $138/mo = 0.57%)
doc analysis / long ctx uncached  $136,250/mo-> delete prompt entirely: $135,000/mo(saves $1,250/mo= 0.92%)
```

**$38/month.** That is the entire prize for a 1M-request/month chat product, *given a perfect vector
that costs nothing to build, validate, version and maintain*. A single engineer-hour per month
exceeds it.

**The only workload that survives** is the adversarial one I constructed to give the thesis its best
shot: a 5,000-token system prompt with only 5,000 tokens of everything else. There, deleting the
prompt saves **33%** uncached. But caching the same prompt already saves **30%** — the cache captures
**90% of the prize for free** — and once cached, deletion is worth only **5.3%**.

- **Why it matters:** this is a *hard* falsifier of the "save tokens" framing, and it does not depend
  on any claim about steering quality. Even a perfect vector is worth <1% of a realistic bill.
- **Confidence:** high on the price structure (read from vendor tables); high on the arithmetic
  (printed by a script); **medium** on the workload archetypes — they are constructed, not measured,
  except for the coding-agent row, which is measured (F2).
- **Cost:** zero.

---

### F2. What the bill *is* actually made of — a measured anatomy, and it is not the system prompt

**Source (fetched, PDF):** "Where the Tokens Go: A Measured Anatomy of LLM Agent Trajectory Costs",
Tran, 2026, REALM workshop, `https://aclanthology.org/2026.realm-1.24.pdf`

**What it measured.** 954 episodes / 4,891 requests, one `claude-haiku-4-5-20251001`, three
synthetic task families, prompt caching disabled, with a *token-counting-endpoint* decomposition
validated to 0.00% error against the billing meter on all 144 baseline episodes.

**Verbatim results:**

> "the recurring fixed prefix accounts for **76.6%** of baseline input tokens. Tool schemas
> contribute **69.1%**, echoed assistant output 6.5%, and the tool-result residual, including
> framing, 16.9%."

> "the tool-schema component … is **1602 tokens per request, versus 110 for the system prompt**."

> "Output tokens are only **4.4%** of total tokens, but at a 5× per-token price they are **18.9%**
> of baseline dollar cost."

> "family-informed tool pruning reduces mean cost by **24.1%** with a success difference of
> **−1.4 percentage points** (paired 95% CI [−3.5, 0.0])."

**Why it matters — three things, all bad for the thesis:**

1. **The prose system prompt was 110 tokens. The tool schemas were 1,602 — 14.6× larger.** In an
   agent, the fixed cost lives in machine-readable schemas, which a steering vector *cannot* replace
   at all: the model must know the exact tool name, JSON parameter names and types. You cannot
   "steer" a model into knowing that a field is called `max_results` and is an integer.
2. **The largest legitimate saving in the paper is 24.1%** — from *pruning unused tool schemas*, an
   exact, inspectable, deterministic edit. That is the real competitor to "prompt engineering", and
   it is not interpretability.
3. **Output is 18.9% of dollars at 4.4% of tokens.** Confirms F1's structure on real traffic.

- **Confidence:** high — peer-reviewed workshop paper, closed accounting identity, reproducible
  package from frozen logs. **Caveat the authors state themselves:** "These findings apply to the
  tested model, mock tools, and short horizons."
- **Cost:** zero.

**The honest counter-datapoint.** Other sources put system prompts much higher: a production
write-up reports "system prompt: 2,800 tokens, tool definitions: 3,100 tokens, conversation history
11,200, tool results 18,400, output 2,100" for a 37,600-token session
(`https://runguard.dev/blog/ai-agent-cost-engineering-production-guide.html`, blog, 2026). Even
there, the system prompt is **7.4% of tokens** and, at 10× cheaper input vs output, ~2% of dollars.
A blog claiming "System Prompt 42% of input tokens" for coding agents
(`https://locsic.com/thinking/ai-observability-dd1-token-economics/`) is **UNVERIFIED** — it states
its ratios are "based on empirical distributions from typical Coding Agent workloads" without
publishing the measurement. I do not rely on it.

---

### F3. Latency: the same answer, and caching already took it

- **Prefill is what a system prompt costs in latency.** 300 tokens at a plausible GPU prefill rate
  of 5,000–20,000 tok/s is **15–60 ms** of time-to-first-token. At a *cached* prefix it is ~0.
- **Prompt caching is measured to deliver most of the TTFT win:** "prompt caching reduces API costs
  by 45-80% and improves time to first token by 13-31% across providers", with 10,000-token system
  prompts over 500+ agent sessions ("Don't Break the Cache", 2026, `https://arxiv.org/html/2601.06007v1`).
- **Steering's latency cost is ~0**, and this is measured: 42.8 tok/s steered vs 42.7 tok/s
  unsteered on Gemma-4-31B/vLLM, 30 layers steered — "That 0.1 tok/s difference is within
  run-to-run noise" (Tetracorp Lab, 2026-05-18, `https://tetracorp.dev/lab/steering-gemma`).

So the honest latency comparison is: **vector saves ~0 ms; caching saves the whole prefill.** The
vector's real latency argument is not about the system prompt at all — it is that steering can
*shorten the output* (e.g. a "concise" direction), and shortening output is the only lever worth
5–10×. That is a different product thesis (F13).

- **On our own box** the numbers invert and become embarrassing: 300 prompt tokens at ~10 tok/s
  batched is **~30 s** of prefill. But that is a property of a 4-core laptop, not of a product, and
  a 20-token prompt would cost 2 s. The fix is a shorter prompt, not a vector.

---

### F4. Prior work I: soft prompts are learned per-task artifacts, and they are *weakest at 1.5B*

**Sources (fetched, arXiv abstracts):**
- "Prefix-Tuning: Optimizing Continuous Prompts for Generation", Li & Liang, 2021,
  `https://arxiv.org/abs/2101.00190`
- "The Power of Scale for Parameter-Efficient Prompt Tuning", Lester, Al-Rfou, Constant, EMNLP 2021,
  `https://arxiv.org/abs/2104.08691`
- "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer", Vu et al., ACL 2022,
  `https://arxiv.org/abs/2110.07904`
- "On Transferability of Prompt Tuning for Natural Language Processing", 2021,
  `https://arxiv.org/abs/2111.06719`

**What these actually are.** Prefix-tuning "optimizes a small continuous task-specific vector
(called the prefix) … allowing subsequent tokens to attend to this prefix as if it were 'virtual
tokens'." Prompt tuning learns "soft prompts … through backpropagation." These are
**parameter-efficient fine-tuning**, not prompt compression: the artifact is learned, it lives in
embedding/KV space, and it is trained against a task loss on labelled data.

**The three findings that matter for "reusable vector":**

1. **Scale dependence, verbatim from Lester et al.:** "prompt tuning becomes more competitive with
   scale: as models exceed billions of parameters, our method 'closes the gap' and matches the
   strong performance of model tuning". And SPoT restates the converse plainly: **"at smaller model
   sizes (below 11B parameters), there are still large gaps between PROMPT TUNING and MODEL
   TUNING."** Our target is **1.5B**. We are in the worst regime for learned soft prompts.
2. **Cross-task transfer works; cross-model transfer needs a trained projector.** SPoT: prompt
   transfer "matches or outperforms standard Model Tuning … while using up to 27,000× fewer
   task-specific parameters" — *within T5*. For across models, the 2021 transferability study is
   explicit: soft prompts transfer "to other PLMs **with a cross-model projector trained on similar
   tasks**". A projector is a training run. So the "write it once, ship it to any model" story is
   **false**, and the fix (a learned projector) destroys the "zero training" pitch.
3. **Soft prompts are initialization-sensitive and expensive to train.** The same transferability
   study: "PT requires much more training time than fine-tuning." SPoT's headline gain on T5-Base
   was **+10.1 points** over vanilla prompt tuning — i.e. vanilla was bad enough that a whole paper
   was needed to fix it.

**Why it matters.** The "prompt → reusable vector" idea is not new; it is the *soft prompt* lineage,
and its own literature says: per-model, per-task, needs labels, needs gradient steps, weakest at
small scale. Anyone pitching a reusable instruction vector in 2026 is re-pitching prompt tuning
with a worse training signal (diff-in-means instead of backprop) and a worse interface (a
coefficient instead of a task).

- **Confidence:** high on the quotes; high that these are different artifacts from a CAA steering
  vector (the soft prompt is *attended to*, the steering vector is *added to every position*).
- **Cost:** zero to read; the honest cost of ignoring it is a year of re-deriving 2021.

---

### F5. Prior work II: gisting is the closest real "compress the prompt into the model" — and it buys 4.2% wall clock

**Source (fetched, arXiv abstract):** "Learning to Compress Prompts with Gist Tokens", Mu, Li,
Goodman, NeurIPS 2023, `https://arxiv.org/abs/2304.08467`

**Verbatim:** "we present gisting, which trains an LM to compress prompts into smaller sets of
'gist' tokens which can be cached and reused for compute efficiency … gisting enables up to **26x
compression** of prompts, resulting in up to **40% FLOPs reductions, 4.2% wall time speedups**, and
storage savings, all with minimal loss in output quality."

**Why it matters — this is the most important negative result in the file.** Gisting is the
*strongest possible* version of the thesis: it is a learned, per-model, end-to-end-optimised
compression of the prompt into the model's own representation, trained with the actual task loss.
26× compression. And the wall-clock win is **4.2%**.

The FLOPs and the wall clock diverge by ~10× because **decoding dominates**: you save prefill, and
prefill was never the cost. Gisting is what "prompt → vector" looks like when you do it properly
with gradients and a real training set. It got 4.2%. A diff-in-means vector — no gradients, ~200
pairs, one forward pass per pair — will not beat that.

**Also note what gisting did *not* become:** it is not a product. There is no gisting API at OpenAI
or Anthropic. That is a market signal, not a proof, but it is a signal from the best-resourced
attempt.

- **Confidence:** high (verbatim numbers).
- **Cost:** zero. The lesson costs a year if ignored.

---

### F6. The direct experiment exists: Stolfo et al. show steering *can* substitute for an instruction — at ~30% success where the instruction gets 60–90%

**Source (fetched, arXiv abstract + HTML):** "Improving Instruction-Following in Language Models
through Activation Steering", Stolfo, Balachandran, Yousefi, Horvitz, Nushi (ETH / Microsoft
Research), **ICLR 2025**, `https://arxiv.org/abs/2410.12877`, code
`https://github.com/microsoft/llm-steer-instruct`

**What they did.** Computed "the difference in activations between inputs with and without
instructions" — i.e. *exactly* "distil a system prompt into a vector" — for IFEval constraints
(format, length, word inclusion), on Phi-3, Gemma-2 2B, Gemma-2 9B, Mistral 7B.

**The numbers, verbatim:**

> "In this setting [no instruction in the text], the instruction-following accuracy without steering
> hovers around **10%** … When steering is applied, we observe a consistent increase in instruction
> adherence across all models, with accuracy improving to approximately **30%**."

> "we evaluate on inputs with instructions provided in the text … the instruction-following accuracy
> without steering is higher in this setting, ranging between **60% and 90%**. Nevertheless, steering
> still results in a noticeable performance boost for **two out of four models**."

**This is the thesis, measured, and it is a partial success and a decisive failure at once:**

| condition | prompt tokens | instruction adherence |
|---|---|---|
| no instruction, no vector | 0 | ~10% |
| **vector, no instruction** | **0** | **~30%** |
| instruction in the prompt | ~30–60 | **60–90%** |
| instruction + vector | ~30–60 | small gain, **2 of 4 models only** |

The vector recovers **~30%** of a behaviour where the prompt gets **60–90%**. It is a real signal —
10%→30% is not noise — but it is **not a substitute**. And when the prompt is present, the vector
adds nothing on half the models.

**The one place they found a genuine win:** "steering vectors computed on instruction-tuned models
can transfer to improve base models", and "in the case of Gemma 2 2B, cross-model steering
outperforms same-model steering". Read carefully: the *instruction-tuned* model's vector helps the
*base* model of the *same family*. That is not "portable across models"; it is "the same
architecture, same tokenizer, same pretraining".

**Also note the authors' own limitations statement**, from the repo's responsible-AI note: "the
method faces limitations, including potential reductions in text quality and incomplete guarantee of
instruction adherence, especially in complex scenarios."

- **Confidence:** high (verbatim, ICLR 2025, code released).
- **Cost:** zero.

---

### F7. The direct *benchmark* also exists, and prompting beats every steering method — including the one we are building

**Source (fetched, arXiv abstract + HTML v2, Table 2 read directly):** "AxBench: Steering LLMs?
Even Simple Baselines Outperform Sparse Autoencoders", 2025, `https://arxiv.org/abs/2501.17148`

**Verbatim abstract:** "For steering, we find that **prompting outperforms all existing methods**,
followed by finetuning. For concept detection, representation-based methods such as
difference-in-means, perform the best. On both evaluations, SAEs are not competitive."

**Table 2 — "Mean overall steering scores for each method, after steering factor selection"**
(Gemma-2-2B L10/L20, Gemma-2-9B L20/L31; LM-judge scored; **the steering factor was tuned per
concept, per method, to that method's own best value**):

| method | 2B L10 | 2B L20 | 9B L20 | 9B L31 | **avg** |
|---|---|---|---|---|---|
| **Prompt** | 0.731 | 0.744 | 1.081 | 1.062 | **0.905** |
| LoReFT | 0.705 | 0.744 | 0.790 | 0.767 | 0.752 |
| ReFT-r1 | 0.798 | 0.723 | 0.812 | 0.607 | 0.735 |
| LoRA | 0.646 | 0.679 | 0.618 | 0.592 | 0.634 |
| SFT | 0.631 | 0.629 | — | — | 0.630 |
| **DiffMean (≈CAA, our method)** | 0.480 | 0.359 | 0.505 | 0.289 | **0.409** |
| SAE | 0.338 | 0.310 | 0.341 | 0.272 | 0.315 |
| LAT | 0.201 | 0.232 | 0.220 | 0.240 | 0.223 |
| Probe | 0.225 | 0.220 | 0.222 | 0.214 | 0.220 |
| PCA | 0.183 | 0.177 | 0.196 | 0.213 | 0.192 |
| SSV | 0.057 | 0.000 | 0.024 | 0.008 | 0.022 |

**This is the single strongest piece of evidence in this file.** Prompting scores **0.905**;
diff-in-means — the exact estimator A3 specifies as our primary extractor — scores **0.409**. A
**2.2× gap**, on 500 concepts, on models in our size class, with the vector's coefficient optimised
per concept. Note also that *finetuning* (0.630) beats diff-in-means: if you are willing to pay a
training run, LoRA is the better buy.

**The authors of AxBench are not anti-steering** — they propose ReFT-r1 precisely because they want
steering to be competitive. That makes the result harder to dismiss.

**The corroborating analysis**, from the AxBench-adjacent economics post: "representation steering
methods, although providing interpretable representation edits, **lag behind simple prompting
baselines**" (Zhengxuan Wu, "On representation steering", 2025-07-12,
`https://nlp.stanford.edu/~wuzhengx/steer/index.html`; **retrieved via search-index full text —
direct fetch of the origin returned HTTP 000 during this session**).

- **Confidence:** very high. This is a direct head-to-head, published, with released code and
  dictionaries.
- **Cost:** zero. **The cost of ignoring it:** we build the thing that scored 0.409 and demo it as
  if the 0.905 baseline did not exist.

---

### F8. What steering actually controls, with effect sizes — and it is not what a system prompt does

The behaviours split cleanly into three tiers. This is the most useful table in the file.

**Tier A — reliably steerable, and a vector can *match* a prompt (narrow, high-salience, single
behaviour):**

| behaviour | source | number |
|---|---|---|
| Refusal | Arditi et al., NeurIPS 2024, `arXiv:2406.11717` | single direction; ablation removes refusal on 13 chat models ≤72B |
| Sentiment / detoxification | Turner et al. 2023, `arXiv:2308.10248` | SOTA negative→positive sentiment shift |
| Truthfulness | Li et al., ITI, NeurIPS 2023, `arXiv:2306.03341` | Alpaca TruthfulQA 32.5% → **65.1%** |
| Format / length / word-inclusion | Stolfo et al., ICLR 2025 | 10% → ~30% (F6) |
| Language, code style, sycophancy | Hewitt 2026, `https://www.cs.columbia.edu/~johnhew/do-not-rely-linearity.html` | Qwen3-14B: French +0.83, code style +0.78, sycophancy +0.84, neg-sentiment +0.60 (efficacy), crosstalk ≤ +0.06 |

**Tier B — steerable in principle, but the effect is *not* a prompt substitute (CAA's own Table 3,
Llama-2-13B-Chat, layer 13, avg token prob on 50 MC items):**

| behaviour | no prompt, no steer | **+ system prompt** | **+ steering (+1)** | both |
|---|---|---|---|---|
| Refusal | 0.78 | **0.95** | 0.86 | 0.92 |
| Corrigibility | 0.57 | 0.79 | 0.83 | 0.93 |
| Hallucination | 0.54 | **0.52** ← prompt does nothing | **0.78** | 0.87 |
| Myopic Reward | 0.49 | 0.81 | 0.66 | 0.94 |
| Sycophancy | 0.63 | 0.67 | 0.60 | 0.63 |

Read the **Refusal** row: the positive system prompt (**0.95**) *beats* the vector (**0.86**), and
adding the vector to the prompt makes it *worse* (0.95 → 0.92). Read the **Hallucination** row: the
system prompt is inert (0.52 vs 0.54) while the vector moves it to 0.78. This is the thesis in one
table — **it is behaviour-dependent, and the two instruments are not interchangeable.**

The paper's own summary is honest: "for **most** behaviors tested, CAA can modify model behavior
beyond what is achieved through prompting alone." Note "most", and note the metric is *average token
probability on a multiple-choice answer*, not free-form instruction adherence over a distribution.

**Tier C — not steerable by a single vector, or actively broken by it:**

- **Multi-step reasoning / factual recall.** No credible result shows a vector adding knowledge.
  Goodfire states the boundary themselves: "One case where a developer definitely should fine-tune
  instead of steer features is if they want to **introduce new knowledge** into the model. Feature
  steering leverages the model's internal intelligence — it doesn't introduce external information."
- **Long-horizon coherence.** "Prompt–Activation Duality" (2026, `arXiv:2605.10664`) measures the
  failure: over 10 persisted-KV turns, residual-stream steering degrades coherence drift to **−18.6**
  while a system prompt "maintains coherence across dialogue turns". Their diagnosis is exactly A3's
  F6: "steered token states are stored and repeatedly reused, turning a local perturbation into
  cumulative coherence degradation." Their fix (GCAD) improves drift to −1.9 — still worse than a
  prompt, and it requires extracting the steering signal *from the system prompt's attention
  contributions*.
- **Complex instruction adherence.** AxBench (F7). Also note the size scaling in Hewitt's data:
  efficacy +0.29 at Qwen3-1.7B vs +0.76 at 14B — **we are at 1.5B**.

- **Confidence:** high on Tier A/B numbers (tables read from PDFs/HTML); high on Tier C direction;
  medium on the exact CAA cells' generalisation beyond Llama-2-13B-Chat.
- **Cost:** zero.

---

### F9. The reliability problem: enumerate the failure modes, because they are the product

A system prompt is inspectable, diffable, versionable, portable across models, and its failure is
debuggable ("the model ignored rule 4"). A steering vector is none of those. Concretely:

| # | failure mode | evidence |
|---|---|---|
| 1 | **Opaque.** No specification you can read. You cannot answer "what does this vector do?" without an eval suite. | AxBench scores vectors by LM judge precisely because there is no other way |
| 2 | **Model-specific.** Hidden width, layer index and residual basis all change per model. | F4 (cross-model needs a trained projector) |
| 3 | **Layer-specific and coefficient-specific.** Effect is non-monotone and sign-asymmetric; negative steering is stronger than positive. | A3 F1; Tan et al. 2024 |
| 4 | **Not even one-behaviour-per-direction.** One direction steers **four different tasks at four different norms**. | Hewitt, Nov 2026 (F10) |
| 5 | **Per-input variance.** "In-distribution, steerability is highly variable across different inputs. Depending on the concept, spurious biases can substantially contribute." | Tan et al., NeurIPS 2024, `arXiv:2407.12404` |
| 6 | **OOD brittleness.** "for several concepts they are brittle to reasonable changes in the prompt, resulting in them failing to generalise well." | same |
| 7 | **Degrades capability, and the damage is silent.** | CAA's own "minimally reduces capabilities"; A6's ΔPPL gate exists because of this |
| 8 | **Long-horizon collapse.** Coherence drift −18.6 over 10 turns. | `arXiv:2605.10664` |
| 9 | **Not composable.** Five personas "still costs one addition per selected layer" — but the *behaviour* of the sum is not the sum of behaviours. | Tetracorp (implementation); Hewitt: `v1+v2` steers an unrelated third task |
| 10 | **No rollback story.** A prompt regression is a one-line revert. A vector regression is a re-extraction, a re-eval, and a coefficient re-sweep. | — |

**The asymmetry that kills it commercially:** a system prompt is a *specification* that a
non-ML engineer can edit and a code reviewer can diff. A vector is a *binary blob* whose meaning
lives in an eval harness. In a regulated or audited setting ("why did the model say that?"), the
prompt is the audit trail and the vector is a gap in it.

- **Confidence:** high. Items 1–6, 8, 9 are each directly sourced; 7 and 10 are [INF] from the
  sources plus ordinary engineering.
- **Cost:** the reason A3's validation harness (F9 there) is 10 items long.

---

### F10. The linearity problem: the artifact is not a stable specification of even one behaviour

**Source (fetched):** "Do Not Rely on the Linear Representation Hypothesis", John Hewitt, Columbia
University, **November 2026**, `https://www.cs.columbia.edu/~johnhew/do-not-rely-linearity.html`

**Verbatim:** "you can learn a single steering vector that steers for **4 four different tasks when
applied at four different norms**. You can also learn two steering vectors that steer well for a task
each, but when summed, steer for a completely unrelated task."

Their Qwen3-1.7B result: one direction `v`, with `‖αᵢv‖` = 44.7 / 71.2 / 54.9 / 97.6, steers
negative-sentiment / French / code-style / sycophancy respectively — each at its own magnitude, each
with crosstalk ≤ +0.06. And `v₁+v₂` (French + neg-sentiment, `cos(v₁,v₂) = −0.19`) steers **code
style** at +0.45.

**Why this is worse than it first looks.** The coefficient is not an intensity knob; it is a *task
selector*. So:
- "the vector for 'be concise'" is not a thing — there is a *family* of behaviours along one
  direction, selected by magnitude.
- Tuning `coef` is therefore not tuning strength, it is **redefining the instruction**. A user
  dragging a slider is silently changing *what* is being asked, not *how much*.
- Composition is not additive, so "vector for persona + vector for format" may produce a third,
  unrelated behaviour. This directly undermines the modularity pitch that makes vectors attractive
  over prompts in the first place.

Also note the magnitudes: `‖αv‖` of 44.7–97.6 against residual norms of ~24–34 (their Table) is a
**1.8×–4× residual-norm perturbation**. These are not subtle nudges; they are large edits, which is
consistent with the coherence damage in F8 Tier C.

- **Confidence:** high on the quoted claims and the numbers (page fetched and parsed). **Medium** on
  generalisation — it is a note, not a peer-reviewed paper, and the vectors are gradient-optimised
  rather than diff-in-means. But the authors pre-empt that objection explicitly.
- **Cost:** zero to read; the cost of ignoring it is shipping a slider that lies about what it does.

---

### F11. Deployment: this only works on open weights, and mainline vLLM does not support it

**The constraint.** Steering needs the residual stream. `openai` and `anthropic` API surfaces expose
tokens, logprobs and (limited) logit_bias — not hidden states. **A steering vector cannot be applied
to GPT-5 or Claude by anyone outside the vendor.** This is not a temporary limitation; it is the
product boundary. Corroboration: the one company that *does* sell hosted steering serves only
**Llama-3.3-70B and Llama-3.1-8B** (Goodfire Ember) — open-weight models, served by them.

**What self-hosters actually run, and the support status of each (all verified this session):**

| server | steering support | evidence |
|---|---|---|
| **llama.cpp** | **YES, mainline.** `--control-vector FNAME`, `--control-vector-scaled FNAME:SCALE`, `--control-vector-layer-range START END` | `tools/cli/README.md`, fetched. Origin PR #5970: "adds the ability to use control vectors, in GGUF format, with Llama-architecture models" |
| **llama.cpp on Qwen2 specifically** | **YES.** `src/models/qwen2.cpp:120` → `cur = build_cvec(cur, il);` immediately before `cb(cur, "l_out", il)`; `src/llama-graph.cpp:1606` → `return cvec->apply_to(ctx0, cur, il);` | raw source fetched from `master` |
| **vLLM mainline** | **NO.** PR #7906 "[Core] Adding Control Vector Support" → `state: closed`, `merged: false` | GitHub REST API |
| vLLM forks/plugins | yes — `UKGovernmentBEIS/vllm-lens` (`SteeringVector`, `apply_steering_vectors`), `ZJU-REAL/EasySteer` (EMNLP 2026 demo, tracks vLLM v0.29.0), RhizoNymph/vllm fork (`enable_steering=True`) | repos fetched |
| Ollama | **no** hidden-state access at all (matches D1) | — |
| TGI | no first-class control-vector API found | **UNVERIFIED** |

**So the addressable market is: people self-hosting open-weight models on llama.cpp, plus people
willing to run a vLLM fork.** That is a real market — llama.cpp is ubiquitous — but it is the
*opposite* market from the one where token costs are a line item. Anyone self-hosting a 7B on their
own hardware has already solved the token bill by construction: their marginal token cost is
electricity. **The customer who cares about $/token is the customer who cannot use the feature.**

That is the structural contradiction at the heart of the thesis: **the people who can run a steering
vector are not the people who need to save tokens, and the people who need to save tokens cannot run
a steering vector.**

**One useful implementation detail from llama.cpp's own docs:** "the control vector works better when
apply to layers higher than 10", and the shipped example uses `--control-vector-layer-range 10 31`.
Also note `build_cvec` is applied at **`l_out`** — i.e. post-layer residual, every layer — which is
A3's `resid_post` site, not `resid_pre`. That difference matters for the cache semantics in A3 F6.

**A caveat about GGUF control vectors:** the DeepWiki summary of `jukofyork/control-vectors` notes
"a critical architectural requirement … is the mandatory use of a **de-bias vector**", i.e. the
neutral midpoint must be subtracted. This is exactly A3's point that raw coefficients are not
comparable — a vector without its debias partner is a vector with an uncontrolled offset.

- **Confidence:** high on llama.cpp and vLLM status (source + API). High on the API constraint.
  **UNVERIFIED** on TGI.
- **Cost:** if we ship a runtime, llama.cpp is the *only* mainstream server that can carry a vector
  to a user today. That is a design constraint for A4.

---

### F12. The commercial counter-case: the one company that built this says it *enhances* prompts, and archived the SDK

**Sources (fetched):** Goodfire blog "Feature Steering for Reliable and Expressive AI Engineering"
(2024-11-20); `github.com/goodfire-ai/goodfire-sdk` via GitHub REST API (`archived: true`,
`pushed_at: 2025-07-14`); Goodfire "Mapping the Latent Space of Llama 3.3 70B" page.

**Three verbatim facts:**

1. **Their own positioning rejects the thesis.** "**Feature steering enhances rather than replaces
   prompting** but is useful when prompting falls short … When steering with features, an AI
   developer **will still write a carefully worded system prompt**, but feature steering comes when
   prompt engineering falls short."
2. **The public API was deprecated.** "Update (Feb 2026): Our SAE demo interface and API have been
   deprecated." The SDK repo is `archived: true`.
3. **Pricing was per token, not per vector** — "$0.35/million tokens for smaller models like Llama
   3.1 8B, scaling to $1.90/million tokens for larger models like Llama 3.3 70B" (Contrary Research
   business breakdown). So the business model was *inference resale with a white-box differentiator*,
   not "buy our vector".

**Interpretation, carefully.** Goodfire raised a $50M Series A (Menlo Ventures, 2025-04) on
*interpretability*, and the surviving product is the platform (activation harvesting, featurizer
training, probes, monitoring, causal analysis — see `docs.goodfire.com/concepts/capabilities/`), not
the steering demo. Their one hard-nosed deployment case study is a **PII guardrail at Rakuten** —
i.e. *monitoring*, not prompt replacement.

So the market has already voted, and the vote was: **steering is a feature of an interpretability
platform, not a replacement for prompting.** That is also the honest positioning for *this* project.

- **Confidence:** high on all three facts. **Medium** on the causal story for the deprecation — I do
  not know *why* they deprecated, and I should not claim to.
- **Cost:** zero.

---

### F13. The steelman: the strongest version of the user's claim, and the version that survives

I want to state the claim at its best before rejecting the weak form.

**Weak form (rejected):** "Replace the system prompt with a vector to save input tokens."
→ Dies on F1 (arithmetic: <1% of the bill), F2 (the fixed cost is tool schemas), F5 (gisting, the
best-case version, gets 4.2% wall clock), F7 (prompting scores 2.2× higher anyway), F11 (the people
who can run it are not the people who need it).

**Strong form (partially supported):** *"For a fixed behaviour that (a) prompting cannot reliably
elicit, (b) must hold across a long generation, and (c) can be measured programmatically, a steering
vector gives a control knob that a prompt does not — and the knob is continuous, composable in
principle, and costs zero tokens."*

That version is **true in three narrow places**, each with evidence:

1. **Suppression that survives prompt injection.** A prompt saying "do not discuss X" can be
   overridden by the user. A vector cannot be talked out of the residual stream. Wu's post makes
   exactly this argument: "prompting remains vulnerable to prompt-injection and jail-breaking attacks
   that can override or dilute suppression prompts" and cites RePS as the method that resists them.
   **This is a security property, not a token saving.**
2. **Behaviours where the prompt is inert** — CAA's Hallucination row (0.52 with a positive system
   prompt vs 0.54 with none) is the clean example. But note: if the prompt was doing nothing, there
   are no tokens to save. The win is capability, not cost.
3. **Continuous dials over a fixed behaviour** — "how concise", "how many tool calls". A 2026 paper
   steers agent tool-call rate "from near 0% to over 90%" with "no prompt change", extracted
   "without any training from the model's own tool-use preference signal", and reports the movement
   is **monotonic in α** (`arXiv:2608.25198`, "Tunable Tool-Call Rates in LLM Agents via
   Representation Steering"). This is the most commercially interesting version **because it changes
   the output length / call count**, which is the only thing worth 5–10× per token (F1). But note
   what it is: a *dial on a behaviour the prompt already controls*, used to trade accuracy for cost
   at runtime. Not a replacement. And note it is a partial counterexample to F10 — for *this*
   behaviour the coefficient really is a volume knob, monotonic, not a task selector. So F10's claim
   is "linearity is not guaranteed", not "linearity is always false".

**And the mechanistically true version, which is the most interesting thing I found:**
"Subliminal Learning Is Steering Vector Distillation" (2026, OpenReview `qlYv5y0cWf`) reports that
"the teacher's system prompt **is well approximated by a steering vector**" — and immediately adds
the two caveats that matter: "System prompts that are **not** well approximated by steering vectors
are not subliminally learned", and "this also explains **why subliminal learning does not transfer
between models**."

So: **a system prompt is, to first order, a direction — and a direction is model-local.** The
thesis is right about the mechanism and wrong about the economics and wrong about portability.

**The reframe I recommend for the project:** stop selling "save tokens". Sell **"a second control
channel, for behaviours the prompt cannot hold"**, and be explicit that it costs a model, a
coefficient sweep, and an eval harness. That is a defensible product. "Save tokens" is not.

---

## VERDICT

> **Is "steering replaces prompts to save tokens" a real product thesis, a partial truth, or a dead
> end?**

**It is a partial truth with a dead economic premise — and the dead premise is the one in the
sentence.** Split the claim into its two halves and they get opposite verdicts.

**Half 1 — "steering can replace (part of) a system prompt": PARTIALLY TRUE, narrowly.**
A vector can substitute for a prompt on a *single, high-salience, programmatically measurable*
behaviour, and it can reach behaviours a prompt cannot. The evidence is real: 10% → 30% adherence
with no instruction (Stolfo, ICLR 2025); refusal 0.78 → 0.86 with no prompt (CAA Table 3);
truthfulness 32.5% → 65.1% (ITI). But the substitution is *lossy* — the prompt gets 60–90% where the
vector gets 30% — and on refusal the prompt actually wins (0.95 vs 0.86). AxBench puts the aggregate
gap at **0.905 vs 0.409**.

**Half 2 — "…and thereby save tokens": FALSE, by arithmetic, independent of everything above.**
A system prompt is 0.5–6% of a realistic bill and 0.05–0.6% of a cached one. Deleting it entirely,
with a *perfect free* vector, saves **$38/month** at 1M requests/month. Prompt caching already
captures 90% of the prize, for free, on every model, with no hidden states. Gisting — the
best-possible version of the idea, trained end-to-end with gradients, 26× compression — delivered
**4.2% wall clock**. There is no version of this where the token savings justify the engineering.

**Why the "dead end" reading is also wrong.** The thing that dies is the *pitch*, not the *tool*. The
tool has three genuine uses, all of which are about *control*, not *cost*: suppression that survives
prompt injection, behaviours where prompting is inert, and continuous dials over output length (the
only lever that touches the expensive token class). Goodfire reached the same conclusion and said so
in public: "feature steering **enhances rather than replaces** prompting."

**The one-sentence version:** *A system prompt is a direction, but a direction is a model-local,
coefficient-indexed, unversioned artifact — so it is a worse specification at a price that was
already ~1% of the bill.*

**Falsifier for this verdict.** I would reverse it if any of the following were observed:
(a) a workload where the system prompt is >30% of *cached* dollars and the behaviour is in AxBench's
"prompt wins" regime; (b) a steering method that beats prompting on AxBench's steering task; (c) a
demonstration that a vector trained on model A transfers to model B with no projector and ≥90%
effect retention; (d) evidence that hosted APIs expose hidden states. None of these currently exists
in anything I could fetch.

---

## Validation blocks (required by `VALIDATION.md`)

Ten load-bearing claims, each with the command/URL actually used and the quoted output. These are the
only claims in this file that should enter `LEDGER.md`.

```
V1  HYPOTHESIS: A system prompt is a negligible fraction of LLM app cost, so replacing it cannot
                be a token-saving product.
    TEST:       python3 code/e1_economics.py   (price ratios P_OUT=10*P_IN, P_CACHE=0.1*P_IN,
                verified against the OpenAI and Anthropic pricing tables fetched 2026-11-10)
    RESULT:     chat assistant (S=300, ctx=800, out=400): S = 27.27% of input tokens but
                5.88% of $ uncached, 0.62% cached. Coding agent (S=110, ctx=12000, out=900):
                0.52% uncached, 0.05% cached. 1M req/mo on gpt-5: deleting the prompt saves
                $38/mo (0.74%) on a cached bill. Most favourable constructed case (S=5000,
                ctx=5000): 33.3% uncached, of which caching already delivers 30.0%.
    VERDICT:    CONFIRMED (for the "save tokens" half of the thesis)
    FALSIFIER:  a common workload where the system prompt is >30% of *cached* dollars, or an
                output:input price ratio near 1:1 (it is 5:1 to 8:1 at every vendor checked).

V2  HYPOTHESIS: Prompting outperforms diff-in-means steering on the same behaviour.
    TEST:       fetched https://arxiv.org/html/2501.17148v2, parsed Table 2 ("Mean overall
                steering scores for each method, after steering factor selection")
    RESULT:     Prompt 0.731 / 0.744 / 1.081 / 1.062 (avg 0.905);
                DiffMean 0.480 / 0.359 / 0.505 / 0.289 (avg 0.409).
                Abstract: "For steering, we find that prompting outperforms all existing
                methods, followed by finetuning."
    VERDICT:    CONFIRMED
    FALSIFIER:  a published benchmark where diff-in-means ≥ prompting on the steering task.

V3  HYPOTHESIS: Compressing a prompt into the model buys negligible wall-clock time.
    TEST:       fetched https://arxiv.org/abs/2304.08467 (abstract)
    RESULT:     "gisting enables up to 26x compression of prompts, resulting in up to 40% FLOPs
                reductions, 4.2% wall time speedups"
    VERDICT:    CONFIRMED
    FALSIFIER:  a prompt-compression result with >20% wall-clock gain at equal quality.

V4  HYPOTHESIS: Our exact target model can be steered in a mainstream production server today.
    TEST:       curl raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen2.cpp
                and .../src/llama-graph.cpp ; grep build_cvec
    RESULT:     qwen2.cpp:120  "cur = build_cvec(cur, il);"  (immediately before cb(cur,"l_out",il))
                llama-graph.cpp:1606  "ggml_tensor * llm_graph_context::build_cvec(...) {
                return cvec->apply_to(ctx0, cur, il); }"
                CLI: "--control-vector FNAME", "--control-vector-scaled FNAME:SCALE",
                "--control-vector-layer-range START END"
    VERDICT:    CONFIRMED (with the caveat that the site is post-layer `l_out`, not `resid_pre`)
    FALSIFIER:  qwen2.cpp without a build_cvec call, or a CLI lacking --control-vector.

V5  HYPOTHESIS: Mainline vLLM supports control vectors.
    TEST:       curl https://api.github.com/repos/vllm-project/vllm/pulls/7906
    RESULT:     "state: closed merged: False merged_at: None"  (title "[Core] Adding Control
                Vector Support"). Steering in vLLM lives in forks/plugins: vllm-lens, EasySteer,
                RhizoNymph/vllm.
    VERDICT:    REFUTED (mainline has no first-party support)
    FALSIFIER:  a merged PR adding control vectors to vllm-project/vllm.

V6  HYPOTHESIS: The leading commercial steering product positions steering as a complement to
                prompting, not a replacement.
    TEST:       fetched goodfire.com/blog/feature-steering-for-reliable-and-expressive-ai-engineering;
                curl https://api.github.com/repos/goodfire-ai/goodfire-sdk;
                fetched goodfire.com/research/mapping-latent-spaces-llama
    RESULT:     "Feature steering enhances rather than replaces prompting ... an AI developer will
                still write a carefully worded system prompt".
                API: "archived: True", "pushed_at: 2025-07-14".
                Blog: "Update (Feb 2026): Our SAE demo interface and API have been deprecated."
    VERDICT:    CONFIRMED
    FALSIFIER:  a Goodfire (or peer) product marketed as replacing system prompts to cut tokens.

V7  HYPOTHESIS: A steering vector alone reaches materially lower instruction adherence than an
                instruction in the prompt.
    TEST:       fetched https://arxiv.org/abs/2410.12877 (abstract + HTML), ICLR 2025
    RESULT:     no instruction, no steering ~10%; with steering ~30%;
                instruction in the prompt 60–90%; steering adds a boost on "two out of four models".
    VERDICT:    CONFIRMED
    FALSIFIER:  vector-only adherence ≥ prompt adherence on IFEval-style constraints.

V8  HYPOTHESIS: For at least one well-defined behaviour, a system prompt beats the steering vector.
    TEST:       pdftotext -layout on https://arxiv.org/pdf/2312.06681, read Table 3
    RESULT:     Refusal row (Llama-2-13B-Chat, layer 13, avg token prob over 50 MC items):
                no prompt/no steer 0.78 | +system prompt 0.95 | +steering(+1) 0.86 | both 0.92.
                Hallucination row: 0.54 | 0.52 (prompt inert) | 0.78 | 0.87.
    VERDICT:    CONFIRMED (and behaviour-dependent — the Hallucination row runs the other way)
    FALSIFIER:  a behaviour where the vector ≥ prompt and the prompt is not inert.

V9  HYPOTHESIS: Learned soft prompts underperform at small model scale, so a 1.5B target is the
                worst case for "learn the prompt into a vector".
    TEST:       fetched https://arxiv.org/abs/2110.07904 (ACL 2022 PDF text)
    RESULT:     "at smaller model sizes (below 11B parameters), there are still large gaps between
                PROMPT TUNING and MODEL TUNING"
    VERDICT:    CONFIRMED
    FALSIFIER:  prompt tuning matching model tuning at ≤2B.

V10 HYPOTHESIS: A single steering direction is not a stable specification of one behaviour.
    TEST:       fetched https://www.cs.columbia.edu/~johnhew/do-not-rely-linearity.html, parsed
                the result tables
    RESULT:     one direction v steers 4 tasks at ||a_i v|| = 44.7 / 71.2 / 54.9 / 97.6
                (neg-sentiment / French / code-style / sycophancy), crosstalk <= +0.06;
                v1+v2 (cos = -0.19) steers an unrelated third task (code style) at +0.45.
    VERDICT:    CONFIRMED (single author note, Nov 2026, gradient-optimised vectors)
    FALSIFIER:  a direction that steers exactly one behaviour across a coefficient range.
```

**One citation I could not fetch directly:** Zhengxuan Wu, "On representation steering" (2025-07-12).
`curl` to `nlp.stanford.edu/~wuzhengx/steer/index.html` and to `zen-wu.social/steer/index.html`
both returned HTTP `000` in this session; the full text was obtained through the search tool's
fetcher. The quoted sentence ("A 50–400-token steering prompt adds less than 0.06 % compute or
KV-cache overhead once the total context exceeds 4k tokens … so the relative cost of prompting
vanishes as windows scale") is corroborated independently by V1's arithmetic, so the verdict does
not depend on it. Marked `UNVERIFIED-BY-DIRECT-FETCH`.

---

## The experiment that would settle it

The literature compares **vector vs 300-token prompt**. Nobody compares **vector vs 20-token prompt**.
That comparison is the whole question, because a 20-token prompt is free, portable, inspectable and
versionable. If a terse prompt matches the long one, the thesis is dead regardless of how good the
vector is. **This experiment is genuinely novel and it fits in one overnight run on 4 cores.**

### Design

**Model:** `Qwen/Qwen2.5-Coder-1.5B` (per LEDGER X5 — no `-Base` suffix), fp32, `transformers`,
forward hooks. Not GGUF (D2).

**Behaviour:** a *style contract* on Python function completion. Chosen because it is (i) in the
model's domain, (ii) verifiable by AST + regex with **zero judge model**, and (iii) a real thing
people write system prompts for.

The 300-token contract specifies five rules: (1) PEP-257 docstring on every function, (2) type hints
on every parameter and return, (3) no `#` comments, (4) no bare `print()`, (5) max line length 79.

**The 20-token contract** is what a competent engineer writes when told to be terse:
`Docstring. Type hints. No comments. No print. <=79 cols.` (13 tokens).

**Tasks:** 60 held-out function-signature prompts, split by *template* (build on templates A/B,
evaluate on C/D) per A6 §4f. Baseline headroom check first: pick tasks where the unsteered model
satisfies 20–50% of rules (A6 trap #7).

### Conditions (8 arms)

| arm | prompt | vector | question it answers |
|---|---|---|---|
| **P0** | none | none | floor |
| **P300** | 300-tok contract | none | ceiling (the thing we want to replace) |
| **P20** | 13-tok contract | none | **the real competitor** |
| **V** | none | distilled from P300 vs P0 | **the thesis** |
| **V+P20** | 13-tok contract | same vector | does it compose? |
| **V-rand** | none | matched-norm Gaussian | negative control 1 |
| **V-shuf** | none | shuffled-label extraction | negative control 2 |
| **V-layer** | none | true vector at L±6 | negative control 3 |

### Vector extraction

200 contrast pairs = (task + 300-token contract, task + no contract), identical task text.
`v = mean(h⁺) − mean(h⁻)` at the last prompt token, unit-normalised, scaled by mean residual norm
(CAA recipe, A3 F1). Layer swept over `{10, 14, 18}`, best-on-validation chosen, **held-out test
never touched during the sweep**. Applied at `resid_pre(L)`, `positions="gen"` (CAA semantics).

### Metrics (all programmatic, no LLM judge)

1. **Rule satisfaction rate** — fraction of the 5 rules satisfied, per completion. AST for docstring
   and annotations; regex for comments/print; `len()` for line length.
2. **Exact-match rate** — all 5 rules.
3. **Prompt tokens consumed** (tokenizer count, per arm).
4. **Completion tokens emitted** — because this is the 5–10× cost class (F1).
5. **Total tokens per *solved* task** = (prompt + completion) / success — the honest efficiency metric.
6. **`ΔPPL`** on 50 KB held-out Python (A6 §4e) — the damage number.
7. **Variance**: 60 tasks × bootstrap 10k for CIs; plus 20 tasks × 3 seeds at T=0.7 for sampling
   variance.

### Pre-registered decision rule (write this down before running)

Let `S(arm)` be mean rule-satisfaction on the held-out split.

- **R0 (validity gate).** `S(P300) − S(P0) ≥ 0.20` with 95% CI excluding 0. If this fails, the
  behaviour is not prompt-controllable on this model and **the experiment is void** — report the
  null, do not tune the prompt until it works.
- **R1 (the thesis survives).** `S(V) ≥ 0.90 × S(P300)` **and** `V` beats `V-rand` and `V-shuf` by
  ≥ 5× at the chosen coefficient. → steering is a real prompt substitute on this behaviour.
- **R2 (the thesis is dead).** `S(P20) ≥ 0.90 × S(P300)`. → a 13-token prompt already does the job;
  the vector has no economic niche regardless of R1.
- **R3 (the interesting outcome).** `S(P20) < 0.80 × S(P300)` **and** `S(V) ≥ 0.90 × S(P300)`. →
  **this is the only result that would make the thesis a product**, and it would be a genuine
  finding: the behaviour is not expressible in 13 tokens but *is* expressible as a vector.

Prediction, stated in advance so I can be wrong in public: **R2 fires.** The style contract is five
short rules; a 13-token prompt is a perfectly good specification of it, and F7 says prompting beats
vectors on this class of task. I expect `S(P20) ≈ S(P300)` and `S(V) ≈ 0.4–0.6 × S(P300)`.

### Runtime on 4 physical cores (i7-8665U, ~10 tok/s batched, per CONTEXT.md)

| stage | count | tokens | wall clock |
|---|---|---|---|
| headroom probe (P0 only, pick 60 tasks from 120) | 120 gens × 80 tok | 9,600 | ~16 min |
| vector extraction, 3 layers | 400 forwards × ~180 tok | 72k prefill | ~35 min |
| layer sweep (validation, 3 layers × 7 coefs × 20 tasks) | 420 gens × 80 tok | 34k | ~55 min |
| main eval: 8 arms × 60 tasks | 480 gens × 80 tok | 38k | ~65 min |
| 3-seed variance: 8 arms × 20 tasks × 3 | 480 gens × 80 tok | 38k | ~65 min |
| ΔPPL + controls | 200 forwards × 512 tok | 100k prefill | ~20 min |
| **total** | **~1,900 generations** | **~290k tokens** | **≈ 4 h 15 m** |

**Minimum decisive variant (~1 h 40 m):** drop the layer sweep (fix L=14, coef ∈ {0, 1, 2}), drop
the 3-seed arm, use 40 tasks. Keeps R0/R1/R2/R3 intact. Run this first; it is enough to settle the
question.

**Cost of being wrong:** ~4 CPU-hours. **Cost of not running it:** we ship a tool whose central
claim was already refuted in 2025 by AxBench and we never checked.

### What this experiment cannot settle

It cannot show cross-model transfer (needs ≥2 models, 2× the budget) and it cannot show long-horizon
behaviour (needs multi-turn, and F8 Tier C says that is where steering loses worst). Both are
correctly out of scope for round 1 — but both are where a *product* would have to live, so a null on
R1 here should not be read as "steering is useless", only as "steering does not replace this prompt".

---

## Concrete implementation

### 1. The economics, recomputed, not asserted

`code/e1_economics.py` — every number in F1 is printed by this file. It takes no arguments and has
no dependencies. **Rule: no cost claim in this project may be stated without running it.** If someone
wants to argue the thesis, they change the `APPS` table and re-run; the arithmetic is the argument.

```python
@dataclass
class App:
    name: str; sys_tok: int; ctx_tok: int; out_tok: int
P_IN, P_OUT, P_CACHE = 1.0, 10.0, 0.1     # verified ratios, F1
def cost(s, c, o, cached):
    return s * (P_CACHE if cached else P_IN) + c * P_IN + o * P_OUT
```

### 2. The experiment harness (extends `code/microscope.py`, does not replace it)

```python
# code/e1_experiment.py  — sketch; A4 owns the production runtime.
CONTRACT_300 = """You are a Python style enforcer. Every function you write MUST satisfy all
five rules, in priority order:
1. DOCSTRING: begin the body with a PEP-257 docstring ...
2. TYPE HINTS: annotate every parameter and the return type ...
3. NO COMMENTS: never emit a `#` comment ...
4. NO PRINT: never call print(); return values instead ...
5. WIDTH: no line may exceed 79 characters ...
... (expand to ~300 tokens with worked examples and one counter-example)"""

CONTRACT_20 = "Docstring. Type hints. No comments. No print. <=79 cols."

ARMS = {
    "P0":     dict(prompt="",           vec=None),
    "P300":   dict(prompt=CONTRACT_300, vec=None),
    "P20":    dict(prompt=CONTRACT_20,  vec=None),
    "V":      dict(prompt="",           vec="v_style"),
    "V+P20":  dict(prompt=CONTRACT_20,  vec="v_style"),
    "V-rand": dict(prompt="",           vec="v_rand"),
    "V-shuf": dict(prompt="",           vec="v_shuf"),
    "V-layer":dict(prompt="",           vec="v_style_wronglayer"),
}

def score_completion(src: str) -> dict:
    """Programmatic, no judge model. AST + regex only."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"parsed": False, "docstring": 0, "hints": 0,
                "no_comments": 0, "no_print": 0, "width": 0}
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return {
        "parsed": True,
        "docstring":    float(all(ast.get_docstring(f) for f in fns)) if fns else 0.0,
        "hints":        float(all(all(a.annotation for a in f.args.args)
                                  and f.returns for f in fns)) if fns else 0.0,
        "no_comments":  float("#" not in src),
        "no_print":     float("print(" not in src),
        "width":        float(max((len(l) for l in src.splitlines()), default=0) <= 79),
    }
```

Two details that will silently invalidate the result if missed:

- **Score the *code block*, not the whole completion.** Models wrap output in ```` ```python ````;
  a `#` inside the fence marker's language tag is not a comment. Strip fences before scoring.
- **`ast.get_docstring` returns the docstring of the *module* for `Module` nodes.** Iterate
  `FunctionDef` nodes explicitly, as above, or you will score "the file has a docstring" and call it
  success on functions that have none.

### 3. Reporting shape

One table, one figure, one null.

| arm | prompt tok | completion tok | tok/solved | rule sat. [95% CI] | exact | ΔPPL |
|---|---|---|---|---|---|---|
| P0 | 0 | | | | | |
| P300 | ~300 | | | | | |
| P20 | 13 | | | | | |
| **V** | **0** | | | | | |
| … | | | | | | |

Figure: `S(arm)` vs `tokens per solved task`, one point per arm, with bootstrap CIs. **If the `P20`
point sits on top of the `P300` point, the thesis is over** — and that is a publishable result, not
a failure.

---

## What NOT to do (traps)

1. **Do not quote a token saving without running `code/e1_economics.py`.** The intuition "a system
   prompt is most of the prompt" is true; the intuition "a system prompt is most of the *bill*" is
   false by 10×, because output costs 5–10× input. This is the single error that makes the thesis
   look viable.
2. **Do not compare a vector to a *long* prompt.** The competitor is the shortest prompt that works.
   If we benchmark 0-token-vector vs 300-token-prompt and declare victory at 80%, we have beaten a
   strawman that costs 0.6% of the bill.
3. **Do not call diff-in-means "our method" without citing AxBench's 0.409.** We are building the
   estimator that lost by 2.2× to prompting. Cite it, and design around it.
4. **Do not present a vector as "the system prompt, compressed".** Hewitt 2026: one direction steers
   four tasks at four norms. The coefficient is a *task selector*, not a volume knob. A slider that
   silently changes the instruction is a UI lie.
5. **Do not claim portability.** Cross-model soft-prompt transfer needs a trained projector
   (2111.06719). Cross-model *steering* transfer has one encouraging result within a model family
   (Stolfo, Gemma-2 2B/9B) and no general result. Say "same family, same tokenizer".
6. **Do not target a behaviour where the prompt already works.** CAA's Refusal row: prompt 0.95,
   vector 0.86. You will lose, and the loss will look like a harness bug.
7. **Do not ignore the output-token asymmetry.** If we want to sell a cost saving, the only honest
   target is *output length* — a "conciseness" vector that shortens completions by 20% is worth
   ~20× a vector that deletes a 300-token system prompt. That is a different product, and it is the
   one worth building.
8. **Do not let "it works on llama.cpp" become "it works".** `build_cvec` is applied at `l_out`
   (post-layer), once per layer, in the f32/f16 graph. That is *not* A3's `resid_pre` and it is not
   the same intervention. Verify equivalence before porting a coefficient.
9. **Do not read Goodfire's archiving as proof the technique is worthless** — they raised $50M and
   still sell steering inside a platform. Read it as proof that **steering-as-a-product-standalone**
   did not clear the bar, while **steering-as-a-platform-feature** did.
10. **Do not ship the slider without the certificate.** A6's rule stands and matters more here than
    anywhere: a vector has no specification, so the eval *is* the specification. No certificate, no
    causal badge.

---

## Open questions for other agents

- **A3:** Given AxBench (prompt 0.905 vs DiffMean 0.409), is diff-in-means still the right extractor,
  or should we switch to the paper's own recommendation — `ReFT-r1` (rank-1 representation
  fine-tuning, 0.735) — for any experiment where we intend to claim a *usable* behaviour? Note the
  cost: ReFT-r1 needs gradient steps, which A3 F2 already rules out on CPU for LoRRA. **This may be
  the project's real ceiling, and it is worth stating out loud.**
- **A3:** Does a vector extracted from a *300-token prompt* contrast differ from one extracted from a
  *20-token prompt* contrast? If they are ~identical (high cosine), the "distil the prompt" framing is
  empty — the vector only ever encoded the 13-token essence, and the other 287 tokens were noise.
  **This is a 10-minute experiment and it would be very informative.**
- **A4:** llama.cpp `build_cvec` runs at `l_out`; our hook runs at `resid_pre(L)`. Do they produce the
  same behaviour at the same effective magnitude? If not, "steering works in llama.cpp" and "steering
  works in our microscope" are two different claims and must be labelled as such.
- **A6:** Does my pre-registered R0/R1/R2/R3 decision rule satisfy your §4 gates? Specifically: is
  `S(P300) − S(P0) ≥ 0.20` a sufficient validity gate, or do you want the full dose–response on
  `P300` token count (0/13/50/150/300 tokens) as the manipulation check?
- **A5:** If the verdict is "partial truth", how does the UI say so without either overselling or
  burying it? My suggestion: the two-slider panel gets a third, non-negotiable readout —
  **"tokens this replaces: N / % of bill: 0.4%"** — computed live from `e1_economics.py`. That single
  number is the honest framing and it is more interesting than a cost claim.
- **A9 / A7:** Since output tokens are 5–10× input, is a *conciseness* vector a better first product
  than a style vector? It is measurable (completion length), it targets the expensive token class,
  and it has a clean falsifier.

---

## SCHEMA CHANGE

One additive, optional block. Rationale: the verdict of this file is *economic*, and the trace schema
currently has no place to record the economics of a steering decision. Without it, the UI can render
"steering replaced 300 prompt tokens" and a reader will assume that mattered.

```json
"economics": {
  "prompt_tokens_replaced": 300,
  "completion_tokens_delta": -18,
  "bill_share_before": 0.0588,        // system prompt as fraction of $, uncached
  "bill_share_after": 0.0000,
  "cache_would_have_saved": 0.0529,   // fraction of $, if the prompt were merely cached
  "output_is_5_to_10x_input": true,
  "verdict": "negligible",            // negligible | material | dominant
  "method": "code/e1_economics.py"
}
```

`verdict` is computed, not chosen: `negligible` if `bill_share_before < 0.10`, `material` if
`0.10 ≤ x < 0.30`, `dominant` otherwise. Every existing emitter stays valid because the block is
optional. **A5 must render `verdict` whenever `steering.vector_id` is present.**

---

## Build next (ordered, with effort estimate)

| # | task | effort | why first |
|---|---|---|---|
| 1 | Run `code/e1_economics.py` with **our own** workload numbers and paste the output into the ledger | 15 min | it is the cheapest possible falsifier and it settles 80% of the thesis |
| 2 | The **13-token-vs-300-token** prompt ablation (R0 + R2 only, no vectors, 60 tasks) | ~40 min CPU | if `S(P20) ≈ S(P300)`, the vector thesis is dead *before* we spend a night extracting anything |
| 3 | 10-minute check: cosine similarity of vectors extracted from the 300-tok and 13-tok contrasts | 10 min CPU | tells us whether "distil the prompt" is a real object or a re-description |
| 4 | Full E1 experiment (8 arms, R0–R3), minimum variant | ~1 h 40 m CPU | the actual answer to the user's question |
| 5 | Full E1 experiment, extended (layer sweep + 3 seeds) | ~4 h 15 m CPU | gives the dose–response A6 requires for any causal claim |
| 6 | Conciseness-vector pilot (output-length target, not prompt target) | ~1 h CPU | the only cost lever that touches the 5–10× token class |
| 7 | Write the honest positioning paragraph for the README, using F12's Goodfire quote | 20 min | prevents the project from making a claim the field already refuted |

**Effort to a defensible answer: ~2.5 CPU-hours** (items 1–4). Everything after that is
characterisation, not decision.
