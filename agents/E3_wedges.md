# E3 — Business wedge evaluation (the one that says no)

**Scope:** the user is an indie hacker with a 4-core laptop, no GPU, no funding, and a working
activation-level instrument. The orchestrator proposed three wedges (A fine-tune diff debugger,
B steering-as-brand-voice, C local-first explorer + audience). The user said they were "good but
not so good". This file agrees with the user, kills two of the three outright, demotes the third
to a marketing asset, and proposes five replacements with named buyers.

**Rule applied:** a wedge is only real if I can name a job title, a painful moment, a current
line item in their budget, and a channel that reaches ten of them this week. Anything I could not
source is marked `UNVERIFIED`. Every number below was fetched this session (§11).

---

## TL;DR (5 bullets)

1. **Kill A.** `science-of-finetuning/diffing-toolkit` already does base-vs-fine-tune diffing
   with an interactive UI, MIT licence, last push 2026-10-25. 88 stars — not because it is bad,
   because almost nobody has this problem badly enough to pay. The buyer is a researcher, and
   researchers get this free from the lab next door.
2. **Kill B.** "Brand voice without fine-tuning" is a $1.9B market (Writer) whose buyer is a
   marketing-ops director who purchases *outcomes and services*, not steering vectors. A vector
   is an implementation detail the buyer will never see, price, or defend in a budget review.
3. **C is not a wedge, it is a funnel.** The explorer is the demo. The course is the product.
   Ship C only as top-of-funnel for #1 or #2 below — never as the revenue line.
4. **The hardware constraint selects the market, and that is the most useful sentence in this
   file.** No GPU ⇒ you can only serve small models ⇒ the only buyers who care about small-model
   internals are (a) people shipping quantized/on-device models and (b) people teaching. Every
   "serve enterprise 70B" idea dies on your laptop, correctly.
5. **Best wedge: a quantization decision-flip certificate — a local CLI + CI gate that tells a
   team which behaviours their 4-bit model will break, before they ship it.** The science is
   published (`m' ≈ c·m + b`, median 1.0pp flip-rate error, arXiv 2608.06564), the demand is
   visible (llama.cpp 133,949 stars, Ollama 186,566), the competition is behavioural-only, and
   it runs on the customer's hardware, so your CPU is irrelevant.

---

## §1. The three proposals, interrogated

Scoring is 1–5 per criterion, 30 max. Criteria and weights are in §4.

### A. Fine-tune diff / regression debugger — **DO NOT BUILD**

| question | answer |
|---|---|
| **Buyer (title, size)** | "Research engineer" or "ML engineer, post-training" at a 5–50 person applied-AI startup or a lab. Occasionally an academic. |
| **Painful moment** | "Our LoRA made the model better at the task and worse at everything else, and we don't know which behaviours died." Real pain. Documented everywhere (catastrophic forgetting). |
| **What they pay today** | **$0.** `diffing-toolkit` (MIT) does activation analysis, crosscoders, KL, PCA, weight amplification, and ships a UI. TransformerLens (MIT, 4,059★) does the primitives. Anthropic's crosscoder paper is public. |
| **What they'd pay** | Maybe $50/mo, and only if it were a *hosted* service they didn't have to maintain. They will not pay for a wrapper on MIT code they can read. |
| **Reach 10 this week** | r/MachineLearning, the Alignment Forum, the `science-of-finetuning` Discord. You can reach them. You cannot *charge* them. |
| **Moat after 6 months** | None. The technique is a paper. The UI is 88 stars of prior art. Goodfire has 50 researchers and $200M and is building the same thing at frontier scale. |
| **Fastest falsification (<1 week, <$100)** | Post a "I'll diff your base vs LoRA and explain the regression, free, first 5 people" on the Alignment Forum + r/LocalLLaMA. Measure: how many take the free offer (expect 3–5), then how many convert to $99 when you quote it (expect 0–1). |

**Why it dies, in one line:** the artefact the buyer wants is a *report they can read once*, not a
tool they run weekly, and a one-off report is a consulting gig, not a product.

**What would change my mind:** evidence of ≥3 companies paying ≥$500/mo for *any* model-diffing
tool. I searched; I found none. `diffing-toolkit` has 13 open issues and 24 forks — that is a
research artefact's traffic pattern, not a product's.

---

### B. Steering-as-a-product for brand voice — **DO NOT BUILD**

| question | answer |
|---|---|
| **Buyer (title, size)** | Two incompatible buyers: (i) "Head of Content / Brand" at 200–5,000 person enterprise; (ii) "Founding engineer" at a 5–30 person startup. (i) has budget but will not buy a vector. (ii) will buy a vector but has no budget. |
| **Painful moment** | "Our AI drafts don't sound like us." Real, but it is a *content* problem, and the market has decided the answer is prompts + style guide + retrieval + human review, not activation editing. |
| **What they pay today** | Writer: $200M Series C at $1.9B (Nov 2024), 250+ enterprises, explicitly sells brand-voice consistency. Jasper, Copy.ai, Typeface at the low end. Free: a 400-word style guide in the system prompt. |
| **What they'd pay** | A marketing buyer pays for "on-brand output at 95% pass rate", priced per seat or per word. They will not pay for "a 1536-dim vector at layer 14". You would have to sell the outcome and hide the mechanism — at which point you are a worse Writer. |
| **Reach 10 this week** | Content-marketing LinkedIn, SEO. Slow, expensive, and you are competing with a $1.9B company's sales team. |
| **Moat after 6 months** | None. The moment OpenAI ships "brand voice" as a project setting (they ship style/persona controls continuously), the mechanism advantage evaporates. |
| **Fastest falsification (<1 week, <$100)** | Build the demo on one brand (yours). Show 20 side-by-side outputs to 5 marketing people. Ask one question: "would you pay $200/mo for this, or would you just paste a style guide into ChatGPT?" Expect 4/5 to say the latter. |

**Why it dies, in one line:** the buyer cannot evaluate the mechanism, so you are forced to compete
on output quality against companies with proprietary models and a sales team, using a 1.5B model
on a laptop.

**What would change my mind:** a *narrow* vertical where brand voice is a compliance requirement
(pharma MLR review, legal disclaimers) and where the buyer is technical. That is not "brand voice",
that is regulated content — a different, harder business.

---

### C. Local-first explorer, monetised by audience/course — **KEEP AS FUNNEL, NOT AS PRODUCT**

| question | answer |
|---|---|
| **Buyer** | Individual engineers learning interpretability; later, their employers. |
| **Painful moment** | "I want to understand what's happening inside, and every tool assumes a GPU and a research setup." Genuine. |
| **What they pay today** | Maven "AI Evals for Engineers & PMs": **$4,200**/seat, and Hamel Husain's earlier course sold to 1,500+ students at ~$1,800. BlueDot, Maven cohorts $1,500–4,200. |
| **What they'd pay** | $50–500 for a course; $0 for the explorer. Transformer Explainer is free with 490k users. Neuronpedia is free. ARENA is free with a full curriculum (1,223★, 9 cohorts). |
| **Reach 10 this week** | Ship the explorer, post on HN. You'll reach 10,000. That is the problem: reach ≠ revenue. |
| **Moat after 6 months** | Audience, not code. Real but slow, and it does not compound into a product. |
| **Fastest falsification (<1 week, <$100)** | Run one 90-minute paid workshop, $49, 20 seats, announced to whatever audience you have. If <8 seats sell, the audience is not there yet. |

**Why it survives only as a funnel:** the explorer is the *proof of competence* that makes the
course sellable. Charging for the explorer is charging for the free thing (Transformer Explainer
exists). Charging for the course is charging for *you*, which is the only defensible asset here.

**Verdict:** do not build C as the business. Build it as the demo that sells #1 or #2.

---

## §2. The constraint nobody used: your laptop is a market filter

This is the section the orchestrator's three wedges all skipped.

```
No GPU  ->  you can only run <=3B models, slowly  ->  you can only serve buyers who care about
            small-model internals  ->  those buyers are: quantized/on-device shippers, and teachers.
```

Two corollaries that decide everything:

1. **Do not sell compute. Sell software that runs on the customer's compute.** A local CLI / GitHub
   Action has no marginal cost, needs no GPU on your side, and the customer's H100s do the work.
   This single move converts "I have a 4-core laptop" from a fatal flaw into a non-issue.
2. **Do not sell to enterprises that run 70B.** You cannot reproduce their workload, so you cannot
   debug their problem. Sell to the people whose model *fits on your laptop* — because then you
   are testing on the same class of hardware they ship to.

The market that satisfies both: **teams shipping quantized open-weight models** (self-hosted
inference, desktop/on-device apps, edge). They are numerous, technical, self-serving, and their
release decision is binary.

---

## §3. Five alternative wedges

### W1. Quantization decision-flip certificate — **the recommendation**

**One line:** "Your Q4 model is not your FP16 model. Here is exactly which decisions it flips,
before you ship it."

| | |
|---|---|
| **Buyer** | Applied-AI / ML-platform engineer, or founding engineer, at a 10–150 person company that self-hosts an open model or ships one on-device. Owns the release. Also: the maintainer of a popular GGUF quant repo (reputational risk). |
| **Painful moment** | "We moved to Q4 to cut GPU spend 4×. Capability evals passed. Then a red-team prompt that FP16 refused went through on INT4, in front of a customer." Or: "our on-device model got worse at 2-bit and we can't tell which behaviours broke." |
| **Pays today** | Nothing directly — they run `lm-eval` (capability only) and hope. Adjacent spend: FutureAGI / Giskard / Promptfoo for behavioural evals; Red Hat's LLM Compressor for the compression itself (free). |
| **Would pay** | $99–$499/mo per repo/model in CI, or $1.5–5k for a one-off certification report with a signature. Price anchor: Braintrust $249/mo, LangSmith $39/seat, Goodfire's Silico $1,000/mo academic. |
| **Reach 10 this week** | (a) GitHub issues on llama.cpp / Ollama / `llm-compressor` tagged quantization-quality; (b) r/LocalLLaMA; (c) the HF model-card authors of the top-50 GGUF quants — DM them a free report on their own quant. These people are *already* arguing about this. |
| **Moat after 6 months** | The paired-activation harness + the per-decision-type calibration library + accumulated "which decisions break at which bit-width" dataset. A behavioural eval vendor cannot copy it without adding hooks; a hooks library cannot copy it without the calibration data. |
| **Fastest falsification (<1 week, <$100)** | Produce three public reports (Qwen2.5-Coder-1.5B, Llama-3.2-1B, Gemma-3-1B at Q8/Q4/Q2), post one to r/LocalLLaMA and HN. Count: ≥10 "can you do model X" replies and ≥3 people who will pre-pay $99. If neither, kill it. Cost: ~20 CPU-hours, $0. |

**Evidence of demand (all fetched, §11):**

- `ggml-org/llama.cpp` — **133,949 stars**, MIT, pushed 2026-11-10. `ollama/ollama` — **186,566
  stars**. GGUF is the default distribution format for local models.
- *Which Decisions Low-Bit Quantization Breaks, and How to Predict Them* (arXiv **2608.06564**,
  v5 Oct 2026): 16 models / 8 families at 4, 3, 2 bits. "We find an approximately linear
  relationship across decisions: `m' ≈ c m + b` … the predicted flip rate differs from the
  observed flip rate by a **median of 1.0 percentage point**." That is a *predictive* product,
  not just a test.
- "Quantization Security Risks" (FutureAGI glossary, 2026): "**most production open-source LLMs
  in 2026 ship quantized** … The safety gap appears when teams assume the quantized model inherits
  the alignment of the full-precision base. **It does not.**" FutureAGI sells *behavioural*
  regression only.
- Red Hat / Neural Magic: "We ran over half a million evaluations on quantized LLMs" — a vendor
  investing that much eval effort is evidence the question matters.
- `Q-resafe` (arXiv 2506.20251): quantization degrades safety alignment; proposes safety patching.
  Confirms the failure mode is real and known.
- **Your own prior work is the differentiator:** A1 measured Q4_K_M at 7.17% relative Frobenius
  error, cosine ≥ 0.997, and concluded "lens/steering survive, neuron claims do not." That is
  exactly the layer at which this product operates.

**Honest risk:** the paper is a preprint; the effect may be model-family-dependent. And the
buyer may decide "re-run my red-team suite on the quant, that's enough." The counter is that
behavioural suites miss margin shrinkage until it flips — which is precisely the paper's finding.

---

### W2. Paid cohort workshop / course — "Debug your model with its internals"

**One line:** a 4-week cohort that teaches engineers to diagnose fine-tune regressions and
quantization damage with hooks, logit lens, and steering — using a laptop.

| | |
|---|---|
| **Buyer** | Senior AI engineer / tech lead at a 20–500 person company; the person who owns "why is our model worse". Secondary: the L&D budget of an AI team. |
| **Painful moment** | "I can fine-tune, I cannot explain what changed." Also: "my team ships models and none of us can answer 'why' in a postmortem." |
| **Pays today** | Maven evals course **$4,200**; other Maven AI cohorts $1,500–2,250; BlueDot; corporate private cohorts at a premium. |
| **Would pay** | $400–1,500/seat for a 4-week technical cohort; $5–15k for a private team cohort. |
| **Reach 10 this week** | One free 60-min workshop ("watch a fine-tune break a model, live, on a laptop") promoted on LinkedIn + r/LocalLLaMA. Convert attendees to a paid cohort. |
| **Moat after 6 months** | None on content (ARENA is free and excellent). Moat is *your* reputation + the specific case studies you accumulate + the certificate employers start asking for. |
| **Fastest falsification (<1 week, <$100)** | One paid 90-min workshop, $49, 20 seats. <8 sales ⇒ no audience yet. |

**Evidence:** Maven's "AI Evals for Engineers & PMs" at **$4,200** is the single strongest
willingness-to-pay datapoint in this entire file; Zach Wilson reports **1,500+ students** at
~$1,800. Counter-evidence: **ARENA is free**, 1,223★, 9 cohorts, and is the default recommendation
in every "how do I learn mech interp" thread. You cannot sell "intro to interpretability". You can
sell "fix the model you shipped last week", which ARENA does not teach.

**Why it scores well for *this* person:** zero infrastructure, zero support burden beyond the
cohort, no GPU, and it compounds into the audience that sells W1 and W3.

---

### W3. Fine-tune safety regression audit (service → product)

**One line:** "You fine-tuned Llama-3.2-3B for your domain. We prove whether you also removed its
refusal behaviour, and ship a steering patch that restores it without retraining."

| | |
|---|---|
| **Buyer** | AI security lead / Head of AI at a 50–500 person company deploying a fine-tuned open model; later, their compliance owner. |
| **Painful moment** | Pre-launch security review or a customer security questionnaire: "prove your fine-tune didn't degrade safety." Nobody can. |
| **Pays today** | AI red-teaming vendors: Gray Swan ($40M Series A, May 2026, customers incl. OpenAI/Anthropic/Meta), Haize Labs (~$12.5M seed), Promptfoo (acquired by OpenAI, reported $86M). Plus Big-4 / boutique AI audit firms at "tens of thousands of dollars" per ISO 42001-style audit. |
| **Would pay** | $2–10k per audit; $1–3k/mo for continuous monitoring. |
| **Reach 10 this week** | 20 DMs to companies that published a fine-tune on HF in the last 30 days, each with a free one-page refusal-regression finding. |
| **Moat after 6 months** | The audit corpus + the "we found it and patched it" case studies. Reputation, not code. |
| **Fastest falsification (<1 week, <$100)** | Send 20 free one-page findings. Ask $1,500 for the full report on 3 of them. Two sales ⇒ validated. |

**Evidence:** EU AI Act GPAI obligations apply from **2 Aug 2025**, enforcement powers and fines
from **2 Aug 2026**; the Commission has already sent formal requests to GPAI providers asking
"whether independent external evaluators have assessed them" (securing.ai summary). Gray Swan and
Haize prove enterprises pay for adversarial assessment. **Counter-evidence:** the guardrail market
is consolidating hard — Lakera→Check Point, Protect AI→Palo Alto, Robust Intelligence→Cisco,
Aporia→Coralogix (Giskard, 2026). Security buyers want suites and logos, and an unknown solo
operator will struggle to clear procurement.

**Verdict:** best ACV, slowest trust. Run it as a *service* first; productise only if 3+ repeat
buyers appear. Note A6 F12: the instrument is a jailbreak primitive — this wedge is the defensive
side of the same capability, which is the only commercially and ethically clean framing.

---

### W4. Behaviour CI for small models (GitHub Action)

**One line:** a CI check that fails the PR when a model change flips a behaviour — runs on CPU,
on 1–3B models, in under 10 minutes.

| | |
|---|---|
| **Buyer** | Founding engineer at a dev-tool / on-device AI company (5–50 people) with a model in the repo. |
| **Painful moment** | "Someone bumped the adapter / quantization / tokenizer and the model got worse; we found out from a user." |
| **Pays today** | Promptfoo (free MIT core, commercial tier), Braintrust $249/mo, LangSmith $39/seat. |
| **Would pay** | $49–199/mo per repo. |
| **Reach 10 this week** | Ship the Action, post the diff it caught on a real model. GitHub Marketplace + HN. |
| **Moat after 6 months** | Weak. Promptfoo/Braintrust can add "activation-level" as a feature; you cannot add their distribution. |
| **Fastest falsification (<1 week, <$100)** | Publish the Action free; measure installs in 14 days. <50 installs ⇒ no. |

**Evidence:** Promptfoo — 26.9k★, MIT, 350k developers, acquired by OpenAI — proves the *category*
(CI-gated model testing) is real and valuable. That is also the problem: it proves the category is
already won by someone with distribution.

**Verdict:** good feature, bad company. Build only as the delivery mechanism for W1.

---

### W5. Interpretability for small domain-science models (downmarket Goodfire)

**One line:** "We reverse-engineer your 1–2B domain model and tell you which input features it
actually uses" — for genomics, materials, climate, agritech labs.

| | |
|---|---|
| **Buyer** | PI / staff scientist at a lab, or Head of ML at a 20–200 person biotech/agritech. |
| **Painful moment** | "The model works and we cannot publish it, because reviewers ask *what it learned*." |
| **Pays today** | Grant budget; contract research organisations; Goodfire (enterprise pricing, Silico $1,000/mo academic). |
| **Would pay** | $10–50k per project, from a grant line. Not from an opex budget. |
| **Reach 10 this week** | Cold email 10 corresponding authors of recent "foundation model for X" papers. Slow. |
| **Moat after 6 months** | Domain case studies. Real but narrow. |
| **Fastest falsification (<1 week, <$100)** | Email 10 PIs offering a free 2-week pilot. ≥2 replies with a dataset ⇒ possible. |

**Evidence:** Goodfire's Alzheimer's-biomarker result with Prima Mente, plus partnerships with
Mayo Clinic and Arc Institute, proves the *top* of this market exists. Schmidt Sciences funds
interpretability research; Transluce runs on ~$6.9M of grants (FY2024 990). **But** the same
evidence shows the money is grant-shaped and slow, and the buyers are the people most likely to
say "just publish it".

**Verdict:** real but un-sellable by a solo operator with no academic affiliation. Deprioritise.

---

## §4. Scored table

Criteria: **B** = buildable solo in 8 weeks on CPU; **S** = sellable without a sales team;
**R** = recurring revenue; **T** = buyer technical enough to self-serve; **L** = survives
OpenAI/Anthropic shipping it; **E** = strength of demand evidence. 1–5 each.

| wedge | B | S | R | T | L | E | **total /30** | verdict |
|---|---|---|---|---|---|---|---|---|
| **W1 quant decision-flip certificate** | 4 | 4 | 4 | 5 | 5 | 4 | **26** | **build** |
| **W2 paid cohort workshop** | 5 | 3 | 2 | 4 | 5 | 4 | **23** | **build (fast cash)** |
| **W4 behaviour CI for small models** | 4 | 4 | 4 | 5 | 3 | 3 | **23** | fold into W1 |
| **C explorer + audience** | 5 | 4 | 2 | 5 | 4 | 3 | **23** | funnel only |
| **W3 fine-tune safety audit** | 4 | 2 | 3 | 2 | 4 | 4 | **19** | services first |
| **A fine-tune diff debugger** | 4 | 1 | 2 | 4 | 2 | 2 | **15** | **kill** |
| **W5 domain-science interp** | 3 | 1 | 2 | 3 | 4 | 2 | **15** | deprioritise |
| **B steering brand voice** | 3 | 1 | 3 | 3 | 1 | 2 | **13** | **kill** |

Considered and rejected without scoring (see §8): LLM observability for self-hosted (Langfuse,
MIT, 8,000 self-hosted instances, acquired by ClickHouse Jan 2026 — the market consolidated
before you arrived); model due diligence for acquisitions (consulting, thin, relationship-gated);
agent debugging (crowded, and does not need activations); inference cost optimisation (needs GPUs;
Neural Magic→Red Hat, Deci→NVIDIA $300M); dataset quality (different product, compute-heavy);
model selection (Artificial Analysis / OpenRouter own it).

---

## §5. Market evidence appendix (verified this session)

| claim | source | status |
|---|---|---|
| Goodfire: $150M Series B at $1.25B, Feb 2026; >$200M total; Silico AI $1,000/mo academic, enterprise negotiated | goodfire.com/blog/our-series-b; siliconangle.com; movetheneedle.news | CONFIRMED |
| Gray Swan: $40M Series A, May 2026; $45.7M total; 15,000 researchers | grayswan.ai/news; madrona.com; pitchbook | CONFIRMED |
| Haize Labs: ~$12.5M seed; Anthropic/Scale/AI21 customers | aisecurityandsafety.org; guardion.ai | CONFIRMED (seed figure differs: $12.5M vs $13M) |
| Promptfoo: 26.9k★ MIT, 350k devs, acquired by OpenAI Mar 2026 | github.com/promptfoo/promptfoo; appsecsanta.com; dev.to | CONFIRMED (price "$86M" is reported, not official) |
| Langfuse: MIT, 35k★, >8,000 self-hosted instances, acquired by ClickHouse 16 Jan 2026 | langfuse.com/handbook; langfuse.com/blog | CONFIRMED |
| OpenPipe: $6.7M seed, acquired by CoreWeave 2025, hosted platform retired 30 Jul 2026 | geekwire.com; sacra.com; ai.engineer | CONFIRMED |
| Neural Magic acquired by Red Hat (Jan 2025); Deci acquired by NVIDIA (~$300M, 2024) | webwire.com; calcalistech.com | CONFIRMED |
| Writer: $200M Series C at $1.9B, Nov 2024, 250+ enterprises | writer.com/blog/series-c; saleshive.com | CONFIRMED |
| Maven "AI Evals for Engineers & PMs": $4,200/seat | maven.com/parlance-labs/evals | CONFIRMED |
| Maven cohorts generally $1,500–2,250; 1,500+ students at ~$1,800 | techinterview.org; teamblind.com; linkedin (Zach Wilson) | CONFIRMED (self-reported) |
| ARENA free curriculum, 1,223★ (Aug 2026), 9 cohorts | arena.education; dev.classmethod.jp; lesswrong | CONFIRMED |
| Transluce: 501(c)(3), $6.87M contributions FY2024, open core | transluce.org; causeiq.com | CONFIRMED |
| `diffing-toolkit`: MIT, 88★, pushed 2026-10-25, interactive UI | api.github.com/repos/science-of-finetuning/diffing-toolkit | CONFIRMED |
| TransformerLens 4,059★; nnsight 1,145★; SAELens 1,583★; all MIT | api.github.com | CONFIRMED |
| llama.cpp 133,949★; Ollama 186,566★ | api.github.com | CONFIRMED |
| `m' ≈ c·m + b`; median 1.0pp flip-rate prediction error; 16 models | arxiv.org/abs/2608.06564 | CONFIRMED |
| "Most production open-source LLMs in 2026 ship quantized" | futureagi.com/glossary/quantization-security-risks | vendor claim — `UNVERIFIED` but directionally corroborated by GGUF star counts |
| "60% of quantized HF models ship GGUF" | towardsai.net blog | `UNVERIFIED` (second-hand) |
| EU AI Act: GPAI obligations 2 Aug 2025; enforcement/fines 2 Aug 2026; AI Office requesting info on external evaluation | artificialintelligenceact.eu; securing.ai; digital-strategy.ec.europa.eu | CONFIRMED |
| Interpretability is *explicitly required* by the AI Act | — | `UNVERIFIED` — the Act requires documentation, robustness and evaluation; it does **not** name interpretability techniques. Do not claim this in a sales deck. |

---

## §6. The uncomfortable question

**"Is interpretability tooling simply a bad indie business because the buyers are researchers who
expect everything free and open-source?"**

### The case that it is

- **The free tier is not a degraded tier — it is the whole product.** TransformerLens (MIT, 4,059★)
  has `run_with_cache`, `accumulated_resid`, a logit lens with a built-in identity check, and
  patching utilities. nnsight (MIT, 1,145★) does per-step intervention inside `generate()`.
  SAELens (MIT, 1,583★) loads every published SAE. `diffing-toolkit` (MIT) does model diffing with
  a UI. There is no feature gap left for a paid wrapper to fill.
- **The field's norm is publish-everything.** Anthropic publishes crosscoders and Gemma Scope
  weights (CC-BY-4.0). Goodfire publishes SPD. Transluce is a 501(c)(3) with an explicit
  "open core, hosted services later" revenue model and ~$6.9M of grants. ARENA gives away a full
  curriculum. The social reward is citation, not payment.
- **The buyer's budget is a grant, not a P&L.** A PI can pay $1,000/mo (Goodfire's academic
  price) but cannot sign a $50k contract without a procurement cycle measured in quarters.
- **Everything in this space gets acquired, not scaled.** Promptfoo→OpenAI, Langfuse→ClickHouse,
  Neural Magic→Red Hat, Deci→NVIDIA, OpenPipe→CoreWeave. The exit is real; the independent
  business is not. An indie cannot outlast a vendor that gives the same tool away to sell GPUs.
- **Goodfire needed $200M and 50 researchers** to reach $1.25B. That is the capital intensity of
  selling interpretability at the top. It is not a template for one person.

### The case that it is not — but only when reframed

- **Promptfoo is the counter-example that matters.** MIT core, 26.9k★, 350k developers, $18.4M
  Series A, acquired for a reported $86M. It sold *model behaviour testing* — not
  interpretability — to engineers, as a CLI, open source, self-serve. The mechanism was free; the
  enterprise workflow, CI integration and dashboards were not.
- **Langfuse** did the same in observability: MIT, self-hosted, 8,000 instances, then a
  ClickHouse acquisition. Open-source dev tools in AI infra *do* monetise — at the workflow layer.
- **Maven's $4,200 evals course** proves practitioners in this exact space will pay four figures
  for *knowledge*, which is the cheapest possible product for a solo operator.
- **Goodfire's own pitch is the tell:** it does not sell "interpretability", it sells *reliability*
  ("reduced hallucinations by half"). Enterprises pay for a release decision, not for understanding.

**Synthesis.** The claim "interpretability tooling is a bad indie business" is **true for
researchers and false for engineers**, and the difference is not the technology — it is whether the
buyer has a *binary decision* (ship / don't ship, pass / fail, hire / don't hire) that your output
resolves. Researchers have no binary decision; they have an open-ended inquiry. Engineers have
dozens, weekly.

So: **do not sell interpretability. Sell a release gate that happens to be computed from
activations.** The word "interpretability" should appear in the docs, not on the pricing page.

---

## §7. Final recommendation and the fastest experiment for each

### #1 — Quantization decision-flip certificate (W1)

**Why first:** it is the only wedge where (a) the demand is visible in star counts and vendor
spend, (b) the science is published and predictive, (c) the competition is behavioural-only, (d)
your CPU is irrelevant because the tool runs on the customer's machine, and (e) OpenAI/Anthropic
will never care about your GGUF.

**Shape:** `qdiff` — a CLI. Input: fp16/bf16 reference + quantized artefact (GGUF/GPTQ/AWQ) +
a decision suite (tool-call yes/no, refusal yes/no, format yes/no, answer-choice). Output: a
signed JSON/Markdown report — per-decision margin before/after, predicted flip rate, observed
flip rate, and a pass/fail per decision class. Then a GitHub Action that fails the PR.

**Fastest falsification (5 days, $0):**
1. Run the paired-margin harness on 3 models × 3 bit-widths (~20 CPU-hours, overnight).
2. Publish **one** report publicly: "What Q4 breaks in Qwen2.5-Coder-1.5B — 41 of 400 decisions
   flip, and here are the 41."
3. Post to r/LocalLLaMA + HN + one llama.cpp issue thread.
4. **Pass:** ≥10 "can you run this on <model>" replies **and** ≥3 people who will pre-pay $99 for
   a report on their model. **Fail:** silence, or "just use lm-eval".
5. Kill criterion: if the flip rates you measure are <1% at Q4 on all three models, the pain is
   too small to sell. Say so and move to #2.

### #2 — Paid cohort workshop (W2)

**Why second:** fastest cash, zero infra, and it builds the audience that sells #1 and #3.

**Fastest falsification (7 days, <$50):** one 90-minute live workshop, $49, 20 seats, titled
"Watch a fine-tune break a model — live, on a laptop". **Pass:** ≥8 seats. **Fail:** <5.

### #3 — Fine-tune safety regression audit (W3)

**Why third:** highest contract value and the strongest regulatory tailwind, but it needs trust
you do not have yet.

**Fastest falsification (7 days, $0):** 20 DMs to recent HF fine-tune publishers, each with a free
one-page refusal-regression finding on their own model. Ask $1,500 for the full report.
**Pass:** 2 paid. **Fail:** 0 paid from 20 → the buyer does not exist at this price.

### Sequencing

```
Week 1     W1 falsification (report) + W2 falsification (workshop) in parallel
Week 2-4   If W1 passes: build CLI, publish 3 more reports, open-source the harness (MIT)
           If W2 passes: run cohort #1, record it, sell cohort #2
Week 5-8   W1: GitHub Action + $99 self-serve report + $499/mo CI tier
           W2: private team cohort at $5-15k (this is where the money is)
Month 3+   W3 only if inbound demand appears from W1/W2 customers
```

Open-sourcing the harness is deliberate: it is the distribution strategy (Promptfoo's playbook),
and the moat is the calibration data + the report + the CI integration, not the hooks.

---

## §8. What NOT to do (traps)

1. **Do not build a dashboard.** Dashboards are bought by people with a budget line called
   "observability", and that market consolidated (Langfuse→ClickHouse). Ship a CLI that emits a
   file a human reads once and a CI job that fails.
2. **Do not sell "interpretability".** Sell "release gate", "regression certificate", "flip-rate
   report". The word attracts researchers and repels budget holders.
3. **Do not compete with free on features.** `diffing-toolkit`, TransformerLens, nnsight, SAELens
   and ARENA are free forever. Compete on *workflow* (CI, signed reports, zero setup) and
   *calibration data*, never on "we also have logit lens".
4. **Do not build for 70B.** You cannot test it, so you cannot support it. Scope the product to
   models that fit in 16 GB and say so on the tin. That is a feature ("runs where your edge model
   runs"), not a limitation.
5. **Do not sell to marketing.** Wedge B's buyer cannot evaluate the mechanism and will benchmark
   you against ChatGPT with a style guide. You lose that comparison at any price.
6. **Do not build the explorer first.** It is 3 weeks of UI for a product that has no buyer yet.
   Build it *after* the first paying customer asks for it, as the demo that closes deals.
7. **Do not claim the EU AI Act requires interpretability.** It requires documentation, robustness
   and evaluation. Claiming more is a lie a lawyer will find. (See §5.)
8. **Do not ship a "disable refusal" feature.** A6 F12 is right: this is a published jailbreak
   primitive. The commercial framing must be defensive — "prove the fine-tune didn't break
   safety" — or you inherit the liability with none of the enterprise trust.
9. **Do not accept "we'll just re-run our red-team suite on the quant".** That is the objection
   that kills W1, and the answer is the paper: behavioural suites miss margin shrinkage until it
   flips, and the flip is predictable from the margin. Put that sentence on the landing page.
10. **Do not price below $99.** A $19/mo tool attracts exactly the buyers who will not pay for the
    $1,500 report, and it makes the CI tier look like a toy to the person who signs.

---

## §9. Open questions for other agents

- **A1:** how faithfully can we dequantize Q4_K_M back to a comparable activation space? The
  7.17% Frobenius / cosine 0.997 number is the whole product's foundation — if dequantised
  activations are not comparable to fp16, W1 must compare *behavioural margins only* (still
  sellable, weaker moat).
- **A6:** the certificate must be honest. Which of the 12 checks are required before a
  "PASS" badge can be printed on a customer's model? A signed report that overclaims is a
  liability, not a product.
- **A9:** does per-layer activation drift after quantization predict *which* decisions flip
  better than the margin model? If yes, that is the differentiated claim and the moat.
- **D2:** is the "median 1.0pp" replication claim robust, or is it an artefact of the paper's
  model selection? Re-derive on Qwen2.5-Coder-1.5B before quoting it in marketing.
- **Orchestrator:** the user's "good but not so good" is a correct instinct. The failure mode of
  all three original wedges is that they are *capabilities looking for a buyer*. The fix is to
  start from a release decision, not from the instrument.

---

## §10. SCHEMA CHANGE (proposed)

W1 needs two fields the canonical trace schema cannot express:

```json
{"step": 0, "token_id": 123, "token_str": "def",
 "layers": [...],
 "logits_topk": [...], "chosen": [id, tok, prob],
 "decision": {"kind": "tool_call", "margin": 1.83,
              "options": [[id_a, "yes"], [id_b, "no"]]},
 "quant": {"format": "Q4_K_M", "ref_format": "bf16", "margin_ref": 2.41,
           "predicted_flip_prob": 0.07}}
```

- `decision` — a margin is only meaningful relative to a named pair of first tokens. Without it,
  a flip rate cannot be audited or reproduced.
- `quant` — the certificate must carry both sides of the comparison in one record, or the report
  cannot be regenerated from the trace alone.

---

## §11. Verification log (fetched this session)

| URL / command | what came back |
|---|---|
| `goodfire.com/blog/our-series-b` | $150M Series B, $1.25B valuation, Feb 5 2026 |
| `siliconangle.com/2026/02/05/goodfire-raises-150m...` | same, + Prima Mente / hallucination-halving detail |
| `movetheneedle.news/.../can-mechanistic-interpretability...` | "Silico … $1,000 a month" academic; commercial negotiated |
| `grayswan.ai/news/gray-swan-announces-series-a` | $40M Series A, 28 May 2026, Cygnal/Shade/Arena, 15,000 researchers |
| `madrona.com/gray-swan-series-a/` | co-led Wing + Madrona; 1M+ attack trajectories |
| `pitchbook.com/profiles/company/633269-80` | $45.7M raised to date |
| `aisecurityandsafety.org/en/organizations/haize-labs` | $12.5M seed; Anthropic/Scale/AI21 customers |
| `guardion.ai/ai-security-index/alternatives/haize-labs` | "$13M Seed (2025-12)" — discrepancy noted |
| `langfuse.com/blog/announcing-our-seed-round` | $4M seed, Lightspeed/La Famiglia/YC |
| `langfuse.com/handbook/chapters/story` | "Jan 2026: Langfuse joins ClickHouse" |
| `langfuse.com/blog/2025-06-04-open-sourcing-langfuse-product` | ">8,000 monthly active self-hosted instances" |
| `langfuse.com/` | "35,000+ GitHub stars", MIT |
| `github.com/promptfoo/promptfoo` | 26.9k stars, MIT, "Promptfoo is now part of OpenAI" |
| `appsecsanta.com/promptfoo` | 22,351★ snapshot, 350k devs, $18.4M Series A Jul 2025, OpenAI acquisition Mar 2026 |
| `dev.to/thedailyagent/top-5-ai-agent-eval-tools-after-promptfoos-exit-576i` | "acquired Promptfoo for $86 million" (reported) |
| `geekwire.com/2024/seattle-startup-openpipe-raises-6-7m...` | $6.7M seed |
| `sacra.com/c/openpipe` | "Acquired by CoreWeave"; legacy platform retirement |
| `ai.engineer/orgs/openpipe` | "retirement ended model training and inference on OpenPipe.ai effective July 30, 2026" |
| `webwire.com/ViewPressRel.asp?aId=329547` | Red Hat to acquire Neural Magic |
| `calcalistech.com/ctechnews/article/bkj6phggr` | NVIDIA/Deci, ~$300M |
| `writer.com/blog/series-c-funding-writer-press-release` | $200M Series C, $1.9B |
| `saleshive.com/vendors/writer` | "over 250 leading enterprises" |
| `maven.com/parlance-labs/evals` | $4,200 USD |
| `techinterview.org/bootcamp_directory/p/maven-ai-engineering-cohorts` | $1,500–4,200 range; 1,500+ students claim |
| `arena.education` | ARENA 9.0, Oct 5–Nov 6 2026, free |
| `dev.classmethod.jp/en/articles/mech-interp` | ARENA repo 1,223★, 777 forks (Aug 2026) |
| `transluce.org/2025-fundraiser` | open-core revenue model; "12000+ downloads" |
| `causeiq.com/organizations/clarity-ai-research,993262331` | FY2024 contributions $6,870,000 |
| `api.github.com/repos/science-of-finetuning/diffing-toolkit` | MIT, 88★, pushed 2026-10-25, "UI to visualize them interactively" |
| `api.github.com/repos/TransformerLensOrg/TransformerLens` | MIT, 4,059★ |
| `api.github.com/repos/ndif-team/nnsight` | MIT, 1,145★ |
| `api.github.com/repos/decoderesearch/SAELens` | MIT, 1,583★ |
| `api.github.com/repos/ggml-org/llama.cpp` | MIT, 133,949★ |
| `api.github.com/repos/ollama/ollama` | MIT, 186,566★ |
| `arxiv.org/abs/2608.06564` | `m' ≈ c m + b`; 16 models; median 1.0pp flip-rate error |
| `futureagi.com/glossary/quantization-security-risks` | "most production open-source LLMs in 2026 ship quantized"; behavioural-only regression |
| `developers.redhat.com/articles/2024/10/17/we-ran-over-half-a-million-evaluations-quantized-llms` | 500k+ evals on quantized models |
| `alphaxiv.org/abs/2506.20251` | Q-resafe: quantization degrades safety alignment |
| `artificialintelligenceact.eu/high-level-summary` | GPAI obligations 2 Aug 2025; enforcement 2 Aug 2026 |
| `securing.ai/eu-ai-act-security-requirements` | AI Office RFIs asking about independent external evaluators |
| `giskard.ai/knowledge/ai-guardrails-comparison-2026` | Lakera→Check Point, Protect AI→Palo Alto, Robust Intelligence→Cisco, Aporia→Coralogix |
| `github.com/science-of-finetuning/diffing-toolkit` (README) | 8 diffing methods incl. crosscoder, KL, PCA, activation analysis, UI |
| `mutekigroup.com/post/technical-due-diligence-for-ai-startups` | AI DD is now central to deals; model registry / dataset chain-of-title |
| `news.ycombinator.com/item?id=45633081` | HN thread: fine-tuning / catastrophic forgetting debate |
| `v-chandra.github.io/on-device-llms` | on-device quantization failure modes; 2-bit changes representations |

`[OBS]` = read from the page above. `[INF]` = my inference or arithmetic. `[SPEC]` = my guess.
