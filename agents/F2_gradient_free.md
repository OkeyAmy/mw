# F2 — Is there a backward loop? Gradient-free optimisation against a logprob-only API

**Question (from the orchestrator):** an API gives token logprobs (sometimes) but no
gradients and no activations. Is there a *backward loop* — can you recover something
gradient-like or direction-like from query access alone, and use it to control the
model?

**Scope discipline.** This file analyses mechanism and cost. It does not contain a
working jailbreak recipe; where the literature is an attack, I extract the *optimisation
structure* and the *measured query counts*, which is what we need for our own loop.
No harmful strings appear here.

**Citation hygiene.** Every paper below was fetched during this session (arXiv abs pages,
`arxiv.org/html` full text, or `ar5iv`). Quotes are verbatim from the fetched page.
Prices were read from the OpenAI and Anthropic pricing pages on 2026-11-11. Numbers I
computed show their arithmetic. Where I am reasoning rather than measuring, the sentence
is tagged `[INF]`; speculation is `[SPEC]`.

---

## TL;DR (5 bullets)

1. **Yes, a backward loop exists — but it closes on behaviour, not on mechanism.** The
   signal you need for the *loss* is one logprob, and you get it. The signal you need for
   the *search* is a gradient over `V = 151,936` token coordinates; a top-k logprob
   response gives you `k ≤ 20` of them, and the rest are *censored*, not merely unobserved
   (OpenAI: "if it is within the top 20 most likely tokens. Otherwise, the value -9999.0
   is used"). So the loop is a **derivative-free search over a discrete space with a
   scalar oracle** — real, well-studied, and expensive in queries rather than in dollars.
2. **Logit leakage is real and gives you more than top-k suggests.** Finlayson et al.
   (2024) show the softmax bottleneck lets you reconstruct *full-vocabulary* outputs from
   an API — but their attack consumes `logit_bias`, and they discuss removing `logit_bias` as
   the defence. Carlini et al. (2024) recover the entire projection matrix for
   under **$20** on Ada/Babbage. Recovering the output layer, however, is not recovering
   an intermediate-layer direction: the map from an L8 intervention to final logits is
   many-to-one.
3. **What GCG finds is a large-norm perturbation whose dominant component is
   anti-refusal — a real internal effect, but not a clean "direction".** Arditi et al.
   (2024) §5 is titled *"Adversarial suffixes suppress the refusal-mediating direction"*;
   the same suffix also *"hijack[s] the attention of important heads"*. Our own C4 result
   (LEDGER P4) is the same phenomenon seen from the other side: a large perturbation
   produces a monotone dose-response whether or not it is specific.
4. **There is no equivalence between an optimised prompt and a steering vector, and in
   principle there cannot be.** Prompting beats every representation-based steering method
   on AxBench (0.905 vs 0.409 mean steering score), and steering vectors are formally
   **non-identifiable** from behaviour: "any direction in this null space can be added to
   a steering vector without altering any observable output" (2026).
5. **The cost is not the blocker; the identifiability is.** A black-box GCG-equivalent
   run is **2.56 × 10⁵ queries ≈ $2–$205** depending on tier (arithmetic in §F6), and
   **26 min–8.5 h** at realistic rate limits. The cheaper logprob-only methods (AutoDAN's
   genetic search, `~10⁴` queries) are already inside that budget. What no query budget
   buys is a *unique* direction — because behaviour does not determine one.

---

## Findings

### F1. What each gradient-free method actually needs — a signal ledger

The literature is usually read as "GCG = white-box, AutoDAN = black-box". That framing
hides the thing that matters for us: **which scalar the optimiser consumes, and how many
oracle calls one improvement step costs.**

| method | what it needs from the model | gradient? | oracle calls to converge (reported or derivable) | what it actually optimises |
|---|---|---|---|---|
| **GCG** (Zou et al. 2023) | full logits + one backward per step | **yes, white-box** | 500 steps; **batch 512, top-k 256, 20 optimisable tokens** | the logprob of an affirmative continuation |
| **GCG re-run through an API** | loss only (1 logprob per candidate) | no | 500 × 512 = **2.56 × 10⁵ forwards**, plus a lost gradient → worse sample efficiency | same |
| **AutoDAN** (Liu et al. 2024) | loss/logprob of a target phrase per candidate | no (genetic) | population × generations ≈ **10⁴** (est.) | a *readable* prompt with the same effect |
| **PEZ** (Wen et al. 2023) | embeddings + straight-through gradient | yes | hundreds of Adam steps | a hard prompt through a continuous relaxation |
| **ZOO** (Chen et al. 2017) | scalar loss, 2 evals per coordinate | no | `O(d)` per step, `10⁴–10⁵` (est.) | input-space perturbation |
| **SPSA / MeZO** (Spall 1992; Malladi et al. 2023) | scalar loss, **2 forwards per step** | no | `10³–10⁵` steps (est.); variance scales with `d` | full parameter/prompt vector, dimension-free cost per step |
| **BEAST** (Sadasivan et al. 2024) | logprobs, beam search | no | ~one forward per candidate; "one GPU minute" | token sequence under a beam |
| **PAIR / TAP** (Chao et al. 2023) | natural-language responses only | no | "**fewer than twenty queries**" — but needs an attacker LLM | semantic prompt |
| **GrIPS** (Prasad et al. 2023) | score of an output given a prompt | no | hundreds of edits | phrase-level edits to an instruction |
| **ZOPO / ZIP** (2024–25) | loss on a *continuous* prompt embedding | no | reduced-dimension ZO; "far less queries" | intrinsic-subspace prompt |

Sources: GCG — Zou, Wang, Carlini, Nasr, Kolter, Fredrikson, "Universal and Transferable
Adversarial Attacks on Aligned Language Models", 2023,
[2307.15043](https://arxiv.org/abs/2307.15043); config quoted from the HTML full text:
"Candidates are optimized using Adam with cosine annealing. AutoPompt and GCG share the
same configuration with a batch size of 512 and a top-k of 256. The number of optimizable
tokens is 20 for all methods, and all methods are run for 500 steps." AutoDAN — Liu, Xu,
Chen, Xiao, ICLR 2024, [2310.04451](https://arxiv.org/abs/2310.04451): "a carefully
designed hierarchical genetic algorithm … preserving semantic meaningfulness". PEZ — Wen,
Jain, Kirchenbauer, Goldblum, Geiping, Goldstein, "Hard Prompts Made Easy", 2023,
[2302.03668](https://arxiv.org/abs/2302.03668). ZOO — Chen et al., AISec 2017,
[1708.03999](https://arxiv.org/abs/1708.03999). MeZO — Malladi et al., NeurIPS 2023,
[2305.17333](https://arxiv.org/abs/2305.17333): "Zeroth-order (ZO) methods can in principle
estimate gradients using only two forward passes but are theorized to be catastrophically
slow". SPSA — Spall, *IEEE Trans. Automatic Control* 37(3), 1992 (no arXiv; DOI
10.1109/9.119632). BEAST — Sadasivan, Saha, Sriramanan, Kattakinda, Chegini, Feizi, "Fast Adversarial Attacks
on Language Models In One GPU Minute", 2024, [2402.15570](https://arxiv.org/abs/2402.15570).
PAIR — Chao et al., 2023, [2310.08419](https://arxiv.org/abs/2310.08419): "PAIR often
requires fewer than twenty queries". GrIPS — Prasad et al., EACL 2023,
[2203.07281](https://arxiv.org/abs/2203.07281). ZOPO — [2403.02993](https://arxiv.org/abs/2403.02993);
ZIP — [2504.06838](https://arxiv.org/abs/2504.06838).

**Three consequences that matter for us.**

1. **GCG's expensive part is not the gradient, it is the candidate evaluation.** The paper
   says so explicitly: "for the same batch size `B` per iteration (i.e., the same number of
   total forward evaluations, which is by far the dominant computation)". A logprob-only
   loop pays the *same* 512 forwards per step; what it loses is the *ranking* that turns
   512 candidates into 512 good candidates. The gradient over the vocabulary is the
   proposal distribution, not the objective.
2. **AutoDAN is already a logprob-only method.** Its contribution was not "attack without
   gradients" but "attack without *unreadable* tokens" — which is the constraint our own
   loop would face anyway, because an API only accepts text.
3. **Dimension-free ≠ cheap.** SPSA/MeZO need 2 forwards per step regardless of `d`, but
   the estimator's variance is `O(d/√B)`; at `d = 1536` (a prompt-length-concatenated
   embedding) or `d = 1.5×10⁹` (weights) the constant is brutal. MeZO is the honest
   reference point: "catastrophically slow" was the prior, and they only rescue it by
   in-place perturbation of *all* weights with memory savings as the goal, not query
   savings.

**Confidence: high** on the signal column (each is stated in the fetched paper), **medium**
on the query-count column: GCG's is exact from the paper, but AutoDAN/ZOO/SPSA are
order-of-magnitude estimates derived from their algorithm class, not numbers they report.

---

### F2. Can you recover a gradient from logprobs? No — and the reason is structural

**What one query returns.** At a single position, up to `k = 20` pairs
`(token_id, logprob)`. OpenAI's API reference (fetched 2026-11-11) is explicit about both
the cap and the tail:

> `top_logprobs`: "An integer between 0 and 20 specifying the maximum number of most likely
> tokens to return at each token position … minimum: 0, maximum: 20"

> `logprob`: "The log probability of this token, if it is within the top 20 most likely
> tokens. Otherwise, the value **-9999.0** is used to signify that the token is very
> unlikely."

That `-9999.0` is the whole story. The tail is **censored**: for the `V − k` unlisted
tokens you learn one inequality each (`logit_t ≤ logit_k`) and nothing about their
relative order. A gradient step in token space needs the *sign and magnitude* of
`∂L/∂e_{x_i} ∈ R^V` — i.e. a ranking of all `V` tokens at that position. You observe 20.

**How much information is that, numerically.** `[OBS]` arithmetic:

- one logprob, fp32, carries ≈ 20 bits of useful mantissa; `k = 20` → **≤ 400 bits ≈ 50 B**
  per position. For comparison the fp32 logit vector is `151,936 × 4 = 607,744 B`, so a
  query reveals **8.2 × 10⁻⁵** of the bytes (50 / 607,744).
- to *name* one token among 151,936 needs `log2(151,936) = 17.2` bits, so 20 logprobs is
  barely enough to rank 20 items — not to rank 151,936.
- but the logit vector is **not** 151,936-dimensional: by the softmax bottleneck it is
  `W_U h` with `h ∈ R^1536`, so its intrinsic dimension is ≤ 1536. At a generous 10 bits
  per coordinate that is `1536 × 10 = 15,360` bits, and `15,360 / 400 = 38.4` queries as a
  *lower bound* on identifying the subspace — before any noise, and assuming the 20
  observed coordinates span independently each time.

**Finite differences, done properly.** If you want a gradient in *embedding* space rather
than token space you can do it: `g ≈ (L(θ+εu) − L(θ−εu))/(2ε)·u` with `u` random
(SPSA), costing **2 queries per iteration regardless of dimension**. That is exactly MeZO.
Two caveats:
(i) **the API does not accept embeddings**, only text, so a recovered soft prompt is
*unsubmittable* — PEZ's whole point is to project back to hard tokens, and that projection
is where its effect is lost; (ii) the estimator is unbiased only for the *scalar* you
differentiate, and the natural scalar (loss of a target continuation) is one number, so
the estimator's variance is set by the loss landscape, not by how many logprobs you asked
for. Asking for 20 logprobs instead of 1 does not make the estimate 20× better; it gives
20 different scalars.

**Verdict on the sub-question:** you cannot recover a token-space gradient from top-k
logprobs. You can recover a *subspace* of the output map, given many queries (F3), and you
can run a scalar-oracle search (F1). Those are different objects: one is a model of the
readout, the other is a search procedure.

---

### F3. Logit leakage: what *is* recoverable, and the exact identifiability ceiling

This is the part of the literature that genuinely is a "backward channel", and it is
better than the top-k cap suggests.

**Carlini et al. (2024)** — "Stealing Part of a Production Language Model",
[2403.06634](https://arxiv.org/abs/2403.06634). Verbatim:

> "our attack recovers the embedding projection layer (up to symmetries) of a transformer
> model, given typical API access. For under **$20 USD**, our attack extracts the entire
> projection matrix of OpenAI's Ada and Babbage language models. We thereby confirm, for
> the first time, that these black-box models have a hidden dimension of 1024 and 2048,
> respectively. We also recover the exact hidden dimension size of the gpt-3.5-turbo
> model, and estimate it would cost under **$2,000** in queries to recover the entire
> projection matrix."

**Finlayson, Ren, Swayamdipta (2024)** — "Logits of API-Protected LLMs Leak Proprietary
Information", [2403.09539](https://arxiv.org/abs/2403.09539). Verbatim:

> "most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to
> a linear subspace of the full output space. We exploit this fact to unlock several
> capabilities, including (but not limited to) obtaining **cheap full-vocabulary outputs**"

and, decisively for our question, their method *requires* a lever:

> "We will assume that the API accepts a prompt on which to condition the distribution, as
> well as a list of up to `k` tokens and a **bias term β** … to add to the logits of the
> listed tokens before applying the softmax function."

One defence they consider is removing `logit_bias`, and they state the consequence:

> "This would be effective, since there are no known methods to recover full outputs from
> such an API."

So the practical answer to "can I get the full logit vector from top-k?" is:

- **With `logit_bias`:** yes, cheaply — pump a token into the top-k and read its logit, one
  query per token per position, or exploit the low-rank image to solve for many at once.
  This is a real backward channel into the *readout*.
- **Without `logit_bias`:** no published method recovers the tail from a plain top-k API.

**The ceiling — three identifiability losses, in order of severity.** `[INF]`, each
derived here:

1. **Softmax shift.** `softmax(z) = softmax(z + c·1)`. One degree of freedom is
   unobservable, always: every logit is known only up to an additive constant.
2. **RMSNorm scale.** The final logits are computed from `RMSNorm(h)`, which divides by
   `‖h‖`. Therefore **the norm of the final residual state is not in the output at all**.
   Even with a perfectly recovered `W_U` you get the *direction* of the last hidden state,
   never its magnitude. Since a steering vector is `coef · v` and `coef` is exactly a
   magnitude, this is the first place where "recover the direction" and "recover the
   intervention" come apart.
3. **Many-to-one at every earlier layer.** Even granting (1) and (2), what you recover is
   the *last* layer. An intervention at L8 is identified only through its image under
   layers 9–27. Nothing in the literature inverts a transformer, and the 2026
   non-identifiability result (below) shows why you should not expect to: the equivalence
   class of behaviourally identical interventions is large.

**The result that actually settles the question.** Venkatesh & Kurapath, "On the
Non-Identifiability of Steering Vectors in Large Language Models", 2026,
[2602.06801](https://arxiv.org/abs/2602.06801). Verbatim:

> "any direction in this null space can be added to a steering vector without altering any
> observable output, producing infinitely many behaviorally equivalent interventions …
> orthogonal perturbations to a steering vector are behaviorally indistinguishable from the
> original, that vectors extracted by entirely different methods are semantically
> equivalent despite being geometrically distinct"

`[INF]` This is the load-bearing citation for the whole file. It says: **the map from
behaviour to direction is not injective.** A logprob-only loop optimises against behaviour.
Therefore it can, at best, land anywhere in an equivalence class. It cannot return *the*
direction, and no amount of query budget fixes a null space.

`[INF]` Caveat, and it is a real one: their result is stated for white-box single-layer
access and a linearised model; the null space is estimated from the activation covariance,
not proven for the full nonlinear model. It bounds the *linear* claim, which is exactly the
claim our steering API makes (`h += coef·v`). It does not say "no direction is causal" — it
says "causality does not pick a unique vector".

---

### F4. What GCG-style optimisation actually finds

Three independent lines of evidence say the same thing, and it is not "a clean recovered
direction". (Four, counting our own harness at the end.)

**(a) The suffix suppresses a specific direction — measured.** Arditi et al., "Refusal in
Language Models Is Mediated by a Single Direction", 2024,
[2406.11717](https://arxiv.org/abs/2406.11717). §5.1 is literally titled **"Adversarial
suffixes suppress the refusal-mediating direction"**. Their method, verbatim:

> "For each instruction, we run the model three times: first with the unedited instruction,
> second with the adversarial suffix appended, and third with a freshly-sampled random
> suffix of the same length appended. By comparing the adversarial suffix to random
> suffixes, we aim to control for the effect of appending any suffix at all. … Figure 5
> shows that the expression of the refusal direction is very high for harmful instructions,
> and remains high when a random suffix is appended."

That is a *length-matched* random-suffix control — exactly the control our own C4 lacked at
first. It is the strongest published evidence that a token-level optimiser moves a
*specific* internal quantity, not just any quantity.

**(b) But it is not only a direction.** Same paper, §5.2: **"Adversarial suffixes hijack the
attention of important heads."** So the suffix also reroutes attention. A rank-1 additive
vector cannot reproduce that.

**(c) The suffix is partly a large orthogonal perturbation.** Nasrati et al., "Toward
Understanding the Transferability of Adversarial Suffixes in Large Language Models", 2025,
[2510.22014](https://arxiv.org/abs/2510.22014). Verbatim: they "identify three statistical
properties that strongly correlate with transfer success": "(1) how much a prompt without a
suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a
push away from this direction, and (3) **how large these shifts are in directions orthogonal
to refusal**". And "prompt semantic similarity only weakly correlates with transfer
success."

So the suffix's effect decomposes into *one anti-refusal component plus a big orthogonal
component*. The orthogonal part is not mechanism; it is the input-space exploit — the part
that makes the state leave the training distribution.

**(d) Optimising the direction directly is *better* than optimising the output.** Cakar et
al., "Optimizing Against Safety Representations: Activation-Guided Adversarial Suffixes and
the Geometry of Refusal", 2026, [2607.08883](https://arxiv.org/abs/2607.08883). They
"replace output-based objectives with losses that directly target a model's internal
refusal direction" and find that "suppressing refusal globally across all layers and
positions is more effective than targeting a single layer-position pair", concluding that
"safety representations are distributed across the forward pass rather than causally
localized to a single site". They also report Soft-GCG "achieves a 33× speedup over
standard GCG".

`[INF]` (d) cuts both ways for us. It proves the optimiser *can* be pointed at a mechanism
— but only because the authors already knew the mechanism (they use "the precomputed
refusal direction provided by Arditi et al.", layer 14, position −1). That is not a
black-box loop; it is a white-box loop with a better loss. It is the strongest evidence
that direction-targeted objectives beat output-targeted ones, i.e. that **the output
signature is a lossy proxy for the direction**.

**(e) Our own harness already reproduced the failure mode.** LEDGER P4/C4: at layer 14 the
dose-response was V-shaped, and "random vectors produced LARGER spans than the real
vector"; the decisive test (`proof/c4_decisive.txt`) failed T2/T3/T4 with the real vector's
target effect 5.702 barely above the random reference 5.251. VALIDATION L1: "a monotone
dose-response curve is produced by *random vectors too*". Any gradient-free loop optimising
against a logit-level objective would have maximised exactly this non-specific span and we
would have called it a success.

**Confidence: high** that a suffix moves the refusal direction (directly measured with a
matched control); **high** that it is not *only* that (attention hijack, orthogonal
component); **medium** on how the two decompose quantitatively — no paper gives a clean
variance budget.

---

### F5. Prompt-based steering vs activation steering: no equivalence, and that is informative

**The direct benchmark says prompting wins.** AxBench — Wu et al., "AxBench: Steering LLMs?
Even Simple Baselines Outperform Sparse Autoencoders", 2025,
[2501.17148](https://arxiv.org/abs/2501.17148). Verbatim:

> "For steering, we find that prompting outperforms all existing methods, followed by
> finetuning. For concept detection, representation-based methods such as difference-in-means,
> perform the best. On both evaluations, SAEs are not competitive."

`[OBS]` E1 already carries the per-number version (prompt 0.905 vs diff-in-means 0.409 vs
SAE 0.315 vs LAT 0.223 on 500 concepts, Gemma-2-2B/9B) and cites the correct id
(`2501.17148`). Flagging the id because the *wrong* one is easy to grab: my own first guess,
`2501.11120`, resolves to an unrelated paper on behavioural self-awareness, and I only caught
it by fetching both.

**The one genuine equivalence result is at the *soft*-prompt level.** He, Zhou, Ma, Taylor,
Fan — "Towards a Unified View of Parameter-Efficient Transfer Learning", ICLR 2022,
[2110.04366](https://arxiv.org/abs/2110.04366) — re-frames prefix tuning, adapters and LoRA
as "modifications to specific hidden states", and identifies prefix tuning with a *weighted
addition* to hidden states, "while the adapter one is unweighted". So:

- a **soft prompt** *is* an activation intervention — rank-`l`, per-layer, position-dependent,
  learned;
- a **hard prompt** is that intervention *filtered through* the embedding matrix and 28
  layers of attention. It reaches the residual stream only via `W_E`-shaped directions at
  the positions it occupies.

`[INF]` That asymmetry is the whole answer to Q4. A steering vector can be any direction in
`R^1536`; a prompt-induced shift is constrained to the span of token embeddings and to
whatever the network computes from them. They are not the same function class, so no
equivalence can hold in general — which is why AxBench finds a 2.2× gap rather than parity.

**And the intervention itself is not unique.** Arditi et al. Appendix I.1 compares
"activation addition" to "directional ablation" and reports them "approximately as
effective", with a revealing asymmetry: the addition "pushes harmless activations far out of
distribution" while ablation does not. `[INF]` So even two *white-box* interventions that
both "work" have different off-target geometry. A black-box loop matching outputs cannot
tell them apart.

**What this means for the thesis.** If the goal is "control the model", prompt optimisation
is the strongest known method and needs no internals. If the goal is "prove what changed
inside", neither prompt optimisation nor a black-box loop can help, because the thing to be
proved — which direction — is not identifiable from behaviour (F3).

---

### F6. Query cost reality: arithmetic

Prices `[OBS]`, read 2026-11-11: OpenAI `developers.openai.com/api/docs/pricing` —
`gpt-6-luna` **$0.10 / $0.50** per M input/output; `gpt-6.1-terra` **$1.00 / $5.00**;
`gpt-6-astra` **$10.00 / $50.00**; cached input is **0.1×** on all three tiers. Anthropic
`platform.claude.com/docs/en/about/claude/pricing` — Claude Haiku 5.5 **$0.10 / $0.50** per
MTok for prompts under 100k. `[OBS]` The Anthropic *messages API reference* page I fetched
(121 KB of text) contains **no occurrence of "logprob"** — i.e. that provider's chat API
does not return logprobs at all, so a logprob-only loop is impossible there and you are
back to a pure response-text oracle. That should be checked against the specific model you
target before budgeting; a negative grep is strong but not a contract.

**Per-run accounting for a GCG-equivalent black-box loop** (one behaviour, one model):

- candidates per step `B = 512`; steps `T = 500`; **queries = 512 × 500 = 256,000**.
- prompt per query = 60-token instruction + 20-token suffix = 80 tokens.
- input tokens = `256,000 × 80 = 20,480,000 = 2.048 × 10⁷`.
- cost: `20.48 M × $0.10/M = $2.05` (luna); `× $1.00/M = $20.48` (terra);
  `× $10.00/M = $204.80` (astra).
- with the 60-token instruction cached at 0.1×, the effective price multiplier is
  `(60×0.1 + 20×1)/80 = 0.325`, so `$0.67 / $6.66 / $66.6`.
- verification generations: 500 × 100 output tokens = 50,000 → `$0.025 / $0.25 / $2.50`.
- **totals ≈ $2.1 / $21 / $207 per behaviour per model** (uncached), i.e. **$0.7 / $6.9 /
  $69** with prefix caching.

**The full paper recipe** (25 behaviours × 4 models = 100 runs):

- queries `2.56 × 10⁷`; input tokens `2.048 × 10⁹`; **$205 / $2,048 / $20,480**.
- `[INF]` That is the price of *one* universal suffix search. At this scale the Batch API
  (50% discount, up to 24 h turnaround) is the only sane channel.

**Wall clock.** `256,000 / RPM` minutes per run:
500 RPM → **512 min = 8.5 h**; 5,000 RPM → **51 min**; 10,000 RPM → **25.6 min**.
`[INF]` Each candidate is a separate HTTP request (the API offers no way to score 512
distinct prompts in one call except the async Batch API), so the loop is *latency-bound at
low tiers and cost-bound at high tiers* — the opposite of the GPU case, where it is
compute-bound.

**The cheaper alternative, for comparison.** AutoDAN-class genetic search at `10⁴` queries:
`10⁴ × 80 = 8 × 10⁵` tokens → **$0.08 / $0.80 / $8.00**. PAIR at 20 queries: `20 × 80 =
1,600` tokens in, plus ~500 output tokens each → **$0.005** on luna, but you must host or
rent the attacker model, and the attack is semantic, not a direction.

**Conclusion for Q5.** A steering-prompt search for one behaviour on a real API is
**$1–$200 and under a day**, not a research programme. Dollars are not what stops this.
Two things do: (i) prompt optimisation is the *weak* version of steering (F5), and (ii) the
recovered object is behaviour, not direction (F3).

---

### F7. Transferability implies a shared *axis*, not a shared *direction*

GCG's transfer numbers are large and specific `[OBS]`, quoted from the paper's results:
suffixes optimised on Vicuna-7B + Vicuna-13B reach "GPT-3.5 (87.9%) and GPT-4 (53.6%),
PaLM-2 (66%), and Claude-2 (2.1%)". Also: "Running for many steps (e.g., 500) can decrease
the transferability and over-fit to the source models."

`[INF]` What mechanism could produce 88% cross-model transfer of a nonsense token string?
The 2025 analysis ([2510.22014]) answers it: transfer is predicted by (i) the prompt's
baseline activation of the refusal direction, (ii) the suffix's push away from it, and (iii)
the orthogonal shift magnitude — **not** by semantic similarity. Combined with Arditi's
13-model, up-to-72B single-direction result, the natural reading is: *safety tuning
converges on a low-dimensional, model-agnostic refusal axis*, and a suffix that pushes along
it transfers.

`[INF]` Does that license a *steering* transfer loop? Partly, with a sharp caveat:

- **For coarse axes (refusal, sentiment, "comment vs code" style)**, yes: the axis is shared,
  so a prompt found to move it in model A plausibly moves it in model B. This is the same
  claim as "steering vectors transfer between models", which is well supported for
  diff-in-means vectors on safety traits.
- **For fine concepts, no evidence.** Transfer of a *jailbreak* is measured by a binary
  success/failure, which is a very coarse statistic; a 60% transfer rate is compatible with
  wildly different internal mechanisms in the target. No paper measures
  cross-model *cosine alignment of the induced shift*; they measure ASR. `[SPEC]` That
  experiment appears to be unrun.
- **Transfer is also explained by a null result.** If many directions are behaviourally
  equivalent (F3), then a random-ish perturbation that happens to have anti-refusal
  component will "work" on the target too. High transfer is *consistent with*
  non-identifiability; it is not evidence for a canonical vector.

---

### F8. Our own box: what a forward pass costs `[OBS]`

`[OBS]` Measured this session with `torch 2.15.0+cpu`, fp32, `torch.set_num_threads(4)`, on
`Qwen2.5-Coder-1.5B` loaded from `.hf/bf16-hf` (bf16 weights cast to fp32, 338 tensors):

```
{"load_s": 30.5,
 "bs8_seq64":  {"sec_per_batch": 41.72, "sec_per_seq": 5.216, "tok_per_s": 12.3},
 "bs32_seq64": {"sec_per_batch": 104.18, "sec_per_seq": 3.256, "tok_per_s": 19.7}}
```

`[OBS]` So: a 64-token prompt forward costs **3.26 s** at batch 32 (5.22 s at batch 8), and
batched throughput saturates at **19.7 tok/s**. Cross-check: `19.7 tok/s × 3.086 GFLOP/token
= 60.8 GFLOPS`, which sits inside A9's measured 54–67 GFLOPS GEMM window — the model is
FLOP-bound, and the timing is self-consistent.

`[OBS]` Note the budget in CONTEXT.md ("~49 tok/s batched") is **not** reproducible in torch
fp32: `49 tok/s × 3.086 GFLOP/token = 151 GFLOPS`, which exceeds the measured 54–67 GFLOPS
GEMM ceiling by ~2.5× and exceeds the 60.8 GFLOPS implied by my own timing. The 49 tok/s
figure must come from the Q4_K_M llama.cpp path.
**Use the measured torch number for all torch-side loop budgets.** This is a
`[INF]` correction to the project's cost model, not a contradiction of it — the two paths
are different artefacts (see VALIDATION L2: never generalise across artefacts).

The important structural point: a **full-prompt** forward costs
`2 × 1.543×10⁹ × 64 = 1.98×10¹¹ FLOP` and re-does the whole 151,936-way readout for every
candidate. A **cached** forward costs one token (`3.08 GFLOP`) plus a readout. Any
affordable local loop must be built as *"one fixed prefix, score one candidate token at a
time against a KV cache"*. The scheme in §Concrete implementation is designed around exactly
that.

---

## Concrete implementation — the decisive experiment (Q7)

**Name:** `BLACKBOX-MECH`. **Question:** does a logprob-only search that reproduces a known
intervention's *output signature* also reproduce its *internal direction*?

**Why this is the right experiment, and its built-in limit.** The optimiser is given only
`top-k logprobs`. The target signature is produced by a white-box intervention. Therefore
the loop is *supervised by the intervention's effect* — it is distillation of an effect, not
blind discovery. That is unavoidable: you cannot ask a black-box loop to find a direction
you cannot describe, because "which direction" is not in the output (F3). The experiment
therefore tests the **narrower and still decisive** claim: *given an effect, can a
logprob-only loop recover the mechanism that produced it?*

**Apparatus.** Everything exists: `code/microscope.py` (hooks), `code/steering_test.py`
(diff-in-means vectors, 16 train pairs, 10 held-out prompts), `code/proof_chain.py`
(patching), `.hf/bf16-hf` weights. Add one module, `code/blackbox_loop.py`, with a hard
interface boundary:

```python
class LogprobOracle:            # the ONLY thing the optimiser may call
    def score(self, text: str, topk: int = 20) -> dict[int, float]:
        """Return {token_id: logprob} for the top-k next tokens at the last position.
        Internally: tokenize, one forward, softmax, topk. No hooks, no hidden states,
        no gradients, no .backward()."""
```

Enforce the boundary in code review, not by discipline: the optimiser module must not
import `torch`, and gets `score` as a callable.

**Design.**

1. **Target intervention.** `v` = diff-in-means comment vector at `L = 8`, `α = 3` (the
   top of the sweep range over which LEDGER P5 measured a logit span of 12.79 for the real
   vector vs 6.62 median for random controls). It is a *measurable, target-specific* logit
   effect even though it failed the behavioural test (T1 passed in `proof/c4_decisive.txt`,
   T2–T4 failed). That is exactly what we want: we are asking whether the optimiser can
   recover a mechanism that is measurable but *not* behaviourally specific. If it can, it is
   doing something stronger than behaviour-matching.
2. **Target signature.** For 20 held-out prompts and 10 train prompts, run the steered
   forward and record the top-20 distribution at the first position. `D_target`. Select the
   train prompts by largest `KL(steered ‖ clean)` so the target signature is non-trivial —
   otherwise `J(S)` is dominated by the unsteered distribution and the loop has nothing to
   match. Store the full 151,936 logits too — the optimiser never sees them, but the
   *analysis* needs them.
3. **Objective.** `J(S) = mean_h KL( D_target(h) ‖ softmax(logits(h + S))[top-20] )`,
   computed over the union of the two top-20 sets (≤40 coordinates, renormalised). One
   `score()` call per candidate.
4. **Search.** Coordinate descent over an 8-token suffix; candidate pool = the 20 observed
   tokens at that position ∪ 20 high-frequency tokens ∪ 4 random vocab samples (24/pool);
   batch 24 candidates; 12 sweeps. **No gradient anywhere.**
5. **Mechanism read-out** (allowed: it's local, and it happens *after* the loop closes).
   On held-out prompts, with the optimised suffix `S*`:
   - `Δ_prompt = resid_8(h+S*) − resid_8(h)` (mean over positions, matching how the vector
     is applied);
   - `cos(Δ_prompt, v)`, `cos(Δ_prompt, random direction)` × 100, and
     `‖Δ_prompt‖ / (α‖v‖)`.
6. **Pre-registered pass gates** (`[OBS]` → `CONFIRMED` only if all four). Named `G` not
   `C` to avoid collision with the project's existing C1–C8 proofs:
   - `G1` output match: median `J(S*)` drops ≥ 80% from `J(S_0)`;
   - `G2` direction: `cos(Δ_prompt, v) ≥ 0.5` **and** above the 95th percentile of 100
     random directions of matched norm;
   - `G3` causal carrier: patching `resid_8` from the `S*` run into the clean run recovers
     ≥ 0.8 of the steered distribution (rules out "the prompt matched the output at L27 and
     L8 is irrelevant");
   - `G4` control arm beats random target: a second loop optimised against the signature of
     a **random** vector `v_rand` (matched norm) must **not** produce `cos(·, v) ≥ 0.5`.
7. **The null-space discriminator (the part that makes it publishable).** Following
   [2602.06801]: find `u` in the low-variance subspace of the layer-8 activation covariance
   (from ~2,000 forward passes over code text), with `‖u‖ = 0.2‖v‖`. Re-run the entire
   experiment with target `v' = v + u`. If the recovered prompt's `cos(Δ, v)` and
   `cos(Δ, v')` are statistically indistinguishable, the loop recovered an **equivalence
   class**, not a direction — regardless of C2 passing. This is the test that separates
   "mechanism" from "behaviour that looks like mechanism".

**Falsifiers, stated before running.**

- If `G1` passes and `G2` fails → **behaviour only**. That is the headline negative result,
  and it is the most likely outcome given F4(d).
- If `G1` fails → the optimiser cannot even match a logit-level effect at this query budget;
  report the query-to-loss curve and stop.
- If `G1`–`G4` pass but the null-space discriminator cannot tell `v` from `v+u` →
  **equivalence class only**.
- If `G1`–`G4` pass *and* `v` is distinguishable from `v+u` → black-box optimisation
  recovered a mechanism, and we would have something genuinely novel. `[SPEC]` I put this
  at well under 20%.

**Cost.** `[OBS]` 8 positions × 24 candidates × 12 sweeps × 10 prompts = **23,040
evaluations**. Each evaluation recomputes from the changed suffix position (average 4 of 8
suffix tokens) plus 1 scoring position = **5 token-forwards**, so **115,200 token-forwards**
≈ 3.6×10¹⁴ FLOP. At the measured 19.7 tok/s (batch 32) that is **1.6 h**; at batch-8
throughput (12.3 tok/s) **2.6 h**. Call it **2–3 CPU-hours**, zero dollars. The
random-direction controls add 100 × 30 = 3,000 full-prompt forwards (≈3 min); the null-space
variant doubles the loop and needs ~2,000 forwards for the covariance (≈30 min).
**Total: one overnight run, ≤ 8 h wall clock, $0.** Downsizing to 6 positions × 12
candidates × 8 sweeps cuts the loop to ~35 min at the cost of a weaker search.

---

## What NOT to do (traps)

1. **Do not call a logit-span objective "recovering the direction."** VALIDATION L1 already
   burned this: random vectors produce larger spans than the real one. A loop that maximises
   a logit margin will maximise span, and span is not specificity.
2. **Do not treat top-k logprobs as "the gradient, approximately".** The tail is censored
   (`-9999.0`), not noisy. A finite-difference estimator over token coordinates needs one
   query per coordinate; you have 20.
3. **Do not assume a logprob API exists.** Anthropic's messages reference page has no
   `logprob` field at all; on such a provider the oracle is response text, which costs
   output tokens (5–10× input) and destroys the loop's economics.
4. **Do not generalise a black-box loop's success to "we recovered a mechanism".**
   Output-matching plus a high cosine is exactly what a large non-specific perturbation also
   produces (F4, F5).
5. **Do not run the loop against a served model you do not own.** Apart from the ethics and
   the ToS, rate limits turn an 8.5-hour run into a week and the provider sees 2.56 × 10⁵
   probes from one key. All numbers in F6 assume *your own* endpoint.
6. **Do not measure transfer by ASR and call it representational sharing.** ASR is binary;
   it cannot distinguish "same direction" from "any of many directions". Measure the induced
   shift's cosine across models, which nobody has published.
7. **Do not optimise over raw token IDs and expect to submit them.** GCG's suffixes are
   token sequences that may not round-trip through the API's text interface; the
   detokenise→retokenise step silently changes the suffix. AutoDAN exists because of this.

---

## Open questions for other agents

- **A2/A3:** is there any *implementable* local probe that estimates a layer-8 direction from
  final-layer behaviour? If a tuned lens can invert layers 8→27 approximately, then a
  black-box loop plus a tuned lens is a plausible (speculative) path to an
  intermediate-direction estimate. Nobody has tried this as far as I can find.
- **E1:** no action needed — I re-verified AxBench = `2501.17148` and E1 already cites it
  correctly. Recorded here only because I initially mis-resolved it to `2501.11120`.
- **A6:** the null-space discriminator in the Concrete implementation (step 7) is a
  *specificity test* that does not depend on a dose-response curve. It might be a better
  C4 than C4. Worth adopting as the standard steering control.
- **E4:** the refusal-axis literature is about *instruct* models. Our target is a **base
  coder** model with no refusal direction; every safety-adjacent citation in F4 may not
  transfer to our artefact. Flagged, not resolved.

---

## SCHEMA CHANGE

One addition to the canonical trace schema, for black-box experiments:

```json
{"step": 0, "oracle": {"call": 4127, "prompt_tokens": 80, "topk": 20,
                       "tokens": [[id, logprob], ...], "tail": "censored"}}
```

Rationale: a trace produced by a *query-limited* path is a different evidentiary object from
a trace produced by hooks, and the ledger must be able to tell them apart. `tail: "censored"`
records that the distribution is not the full one — without that field, a downstream reader
would mistake a top-20 trace for a full-logit trace and every KL in this file would be
silently wrong.

---

## Build next (ordered, effort estimate)

1. **`code/blackbox_loop.py` + `LogprobOracle` boundary** — 3 h. Includes the guard test
   that the optimiser module cannot import `torch`.
2. **`D_target` capture for the L8 comment vector** (10 + 20 prompts) — 1 h, reuses
   `steering_test.py`.
3. **Run the loop, no null space** — one overnight, ~3 CPU-h. Produces `proof/f2_loop.json`
   with `J(S)` per sweep, `cos(Δ,v)`, the 100-random-direction null, and the patch recovery.
4. **Null-space discriminator** — +4 h of compute, needs the activation covariance from
   ~2,000 forwards (~1.5 h on its own).
5. **Cross-model shift-alignment probe** (answers F7's open question): repeat the L8
   measurement on `Qwen2.5-Coder-1.5B` vs a second small model and report
   `cos(Δ_A, Π Δ_B)` — 4 h. This is the experiment the transfer literature has not run.

Total: ~2 days, $0. If step 3's `C2` fails, stop — the answer is "behaviour only" and it is
a clean, publishable negative.

---

## Verdict

**A logprob-only loop can recover behaviour. It cannot recover mechanism, and the reason is
structural rather than budgetary.**

- The **loop exists** and is cheap: `10⁴–10⁵` queries, **$1–$200**, hours to a day. GCG's
  full recipe is `2.56 × 10⁵` queries/run — **≈$2 / $21 / $207** at 2026 prices, **8.5 h at
  500 RPM**.
- What it optimises against is an **output signature**. The gradient-like object it needs —
  a ranking over `V = 151,936` token coordinates — is **not** in a top-20 response: 20
  coordinates are returned, the remaining 151,916 are censored to `-9999.0`. And 400 bits
  per position is a lower bound of ~40 queries just to pin a 1536-dimensional *readout*
  subspace, which additionally requires `logit_bias` to reach the tail at all.
- Even a *perfect* recovery of the readout stops at the final layer, and only up to
  RMSNorm's scale. Steering vectors live at L8/L14.
- Decisively: **steering directions are non-identifiable from behaviour** — "any direction in
  this null space can be added to a steering vector without altering any observable output"
  ([2602.06801], 2026). A loop that only sees behaviour therefore returns an *equivalence
  class*. There is no query budget that resolves a null space.
- Empirically the direction that GCG-like loops *do* find is a **large-norm anti-refusal
  perturbation with a substantial orthogonal component** (Arditi §5.1–5.2; [2510.22014]),
  which is the same object our own C4 found and rejected as non-specific.

**Consequence for the project.** This closes the "learn to steer from the outside" product
path. It does *not* damage the measurement path: `BLACKBOX-MECH` is worth running precisely
*because* its likely outcome is the negative result "output-matched, direction-not-recovered",
which is a defensible, falsifiable statement about what black-box access can and cannot buy —
and it is an ~8-hour, $0 experiment on hardware we already have.

**One number to remember: 2.56 × 10⁵ queries ≈ $2–$205 per behaviour, and it still returns
a behaviour, not a direction.**

---

## Validation ledger

Per `VALIDATION.md`. `[OBS]` measurements are mine; `[CIT]` are quoted from fetched pages.

```
HYPOTHESIS: GCG's reported configuration is 20 optimisable tokens, batch 512, top-k 256, 500 steps.
TEST:       fetched https://ar5iv.labs.arxiv.org/html/2307.15043 and searched "batch size"/"500 steps".
RESULT:     [CIT] "AutoPompt and GCG share the same configuration with a batch size of 512 and a
            top-k of 256. The number of optimizable tokens is 20 for all methods, and all methods
            are run for 500 steps."
VERDICT:    CONFIRMED
FALSIFIER:  a different configuration stated in the paper's method section.
```

```
HYPOTHESIS: A top-20 logprob API censors the tail rather than omitting it.
TEST:       fetched the OpenAI chat-completions API reference (2026-11-11), searched "logprob".
RESULT:     [CIT] "The log probability of this token, if it is within the top 20 most likely tokens.
            Otherwise, the value -9999.0 is used to signify that the token is very unlikely."
            and "minimum: 0, maximum: 20".
VERDICT:    CONFIRMED
FALSIFIER:  documentation returning a true distribution or a null for out-of-top-k tokens.
```

```
HYPOTHESIS: Anthropic's messages API does not return logprobs, so a logprob-only loop is
            impossible there.
TEST:       fetched https://platform.claude.com/docs/en/api/messages (121,045 chars of extracted
            text) and regex-searched case-insensitively for "logprob".
RESULT:     [OBS] zero matches.
VERDICT:    CONFIRMED for the fetched page (a negative grep is strong, not conclusive; provider
            docs change and the model-specific reference may differ).
FALSIFIER:  a "logprobs" parameter in that page or in the current OpenAPI schema.
```

```
HYPOTHESIS: The full GCG recipe costs $2-$205 per behaviour per model at 2026 prices.
TEST:       arithmetic, prices fetched from developers.openai.com/api/docs/pricing.
RESULT:     [OBS] 512 x 500 = 256,000 queries; x 80 tokens = 2.048e7 input tokens;
            2.048e7/1e6 x {0.10, 1.00, 10.00} = {$2.05, $20.48, $204.80}.
VERDICT:    CONFIRMED (arithmetic), given the fetched prices and an 80-token prompt.
FALSIFIER:  a re-read of the pricing page showing different per-MTok rates.
```

```
HYPOTHESIS: Steering vectors are not identifiable from input-output behaviour.
TEST:       fetched https://arxiv.org/html/2602.06801v4 and searched "null space".
RESULT:     [CIT] "any direction in this null space can be added to a steering vector without
            altering any observable output, producing infinitely many behaviorally equivalent
            interventions".
VERDICT:    CONFIRMED as a claim about the paper; the paper's own scope is white-box
            single-layer access under a linearisation — the extrapolation to "no black-box
            loop can recover a direction" is [INF] and is labelled as such in F3.
FALSIFIER:  a published black-box method that recovers a direction which is distinguishable
            from its null-space perturbations by output behaviour alone.
```

```
HYPOTHESIS: Adversarial suffixes act by suppressing the refusal direction (not merely by
            producing any perturbation).
TEST:       fetched https://arxiv.org/html/2406.11717v3, section 5.
RESULT:     [CIT] section 5.1 title "Adversarial suffixes suppress the refusal-mediating direction";
            method: "three runs: unedited, adversarial suffix, freshly-sampled random suffix of
            the same length"; observed: "the expression of the refusal direction is very high for
            harmful instructions, and remains high when a random suffix is appended." My page
            capture ends mid-sentence there; the section's conclusion is carried by its title and
            by the abstract ("how adversarial suffixes suppress propagation of the
            refusal-mediating direction"). Also 5.2 "Adversarial suffixes hijack the attention of
            important heads".
VERDICT:    CONFIRMED (direction of effect), with the last step of the quote not captured.
FALSIFIER:  a reproduction where a length-matched random suffix produces the same suppression.
```

```
HYPOTHESIS: A GCG-equivalent logprob-only loop costs 256,000 API calls per run.
TEST:       arithmetic on the fetched configuration; no assumption beyond B and T.
RESULT:     [OBS] 512 x 500 = 256,000. Note GCG's own text endorses this accounting:
            "for the same batch size B per iteration (i.e., the same number of total forward
            evaluations, which is by far the dominant computation)".
VERDICT:    CONFIRMED
FALSIFIER:  a batched-scoring endpoint that prices 512 prompts as one logical call.
```

```
HYPOTHESIS: torch fp32 forward throughput on this box is lower than the 49 tok/s in the
            project budget, and a cached single-token step is the only affordable loop primitive.
TEST:       /tmp/opencode/f2_time.py: load .hf/bf16-hf in fp32, batch 8/32 x seq 64, two
            inference-mode forwards after a warm-up, timed.
RESULT:     [OBS] load 30.5 s; bs8 41.72 s/batch = 12.3 tok/s; bs32 104.18 s/batch = 19.7 tok/s;
            implied 19.7 x 3.086 GFLOP = 60.8 GFLOPS (inside A9's 54-67 window). 49 tok/s would
            require 151 GFLOPS, 2.5x the measured ceiling.
VERDICT:    CONFIRMED — the 49 tok/s figure is not a torch-fp32 number for this model on this box.
FALSIFIER:  a re-run showing >= 49 tok/s sustained for fp32 torch forwards at seq 64.
```

```
HYPOTHESIS: AxBench is arXiv 2501.17148 (my first guess, 2501.11120, is a different paper).
TEST:       arXiv API `ti:"AxBench"`, then direct fetches of both ids.
RESULT:     [OBS] 2501.17148 = "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse
            Autoencoders"; 2501.11120 = a paper on behavioural self-awareness. E1 already cites
            2501.17148, so no correction to the ledger is required.
VERDICT:    CONFIRMED (and E1 is clean — this block corrects *me*, not E1).
FALSIFIER:  an arXiv page showing the AxBench title at 2501.11120.
```

---

## Live API findings (2026-11-11)

**Scope.** The owner supplied a Google AI Studio key after the main report was written.
Everything below is either an API call I made (7 in total, listed in L1), a documentation
fetch, or arithmetic. I re-ran **no** local model after the memory constraint was issued.
One local measurement completed *before* that constraint (L3) is reported because it is the
only direct evidence on the top-1 question, and it is labelled with the risk it carried.

**Verification note (required):** verification for this section was limited to live API
calls and static analysis for memory reasons — loading a 1.5B model in fp32 on this box
(zram swap, two QEMU VMs, real free RAM far below 31 GB) froze the machine once. The one
local run that did complete (L3) was launched **before** the `mw.guard.require` rule existed
and must not be taken as a template.

### L1. What this key actually returns `[OBS]`

| # | request | result | verbatim |
|---|---|---|---|
| 1 | `GET /v1beta/models` | 200, 50 models | no model advertises a logprob method; `supportedGenerationMethods` ∈ {generateContent, countTokens, createCachedContent, batchGenerateContent, embedContent, …} |
| 2 | `gemini-3.5-flash:generateContent` + `responseLogprobs:true, logprobs:5` | **400** | `"Logprobs is not enabled for this model"` (`INVALID_ARGUMENT`) |
| 3 | `gemini-3.8-flash:generateContent` + same flags | **400** | `"Logprobs is not enabled for this model"` |
| 4 | `gemini-3.8-flash:generateContent`, **no** logprob flags, `maxOutputTokens:3` | **200** | `modelVersion: gemini-3.8-flash`, `content: {}`, `finishReason: MAX_TOKENS` |
| 5 | Interactions API docs | — | no `logprobs`/`responseLogprobs` field in the Interactions generation config |
| 6 | `gemini-embedding-001:embedContent` | 200 | 3072 floats, **L2 norm 1.0000** |
| 7 | `gemini-embedding-001:batchEmbedContents` (8 texts) | 200 | 8 × 3072 |

Call 4 is the control that makes 2 and 3 interpretable: the model is reachable and answers;
only the logprob flag is refused. This independently reproduces the orchestrator's finding on
two models it had not tested.

**The "Interactions API" lead is closed, and closed by Google.** The migration page lists no
logprob capability, and on the developer forum a Google staff reply of 2026-05-14 states:
*"This is working as intended, We streamlined interactions API by removing candidates,
allowing developers to directly access the model's top output as the single best answer."*
Removing `candidates` removes the only place logprobs could be attached. The same thread's
follow-up (2026-06-29) reports that even the legacy `logprobs` field was reachable only via a
vertex/enterprise flag that is now deprecated.

**One more structural point, from call 4.** `content: {}` with `finishReason: MAX_TOKENS` at
a 3-token budget means the first tokens emitted are **thinking tokens**, not answer tokens.
So this is not a next-token endpoint that happens to hide probabilities: the token you would
want to score is not the token the response starts with. A text API is strictly weaker than
even a top-1 logprob API. `[OBS]` for the empty content; `[INF]` for the thinking-token
attribution, which follows from the observed budget exhaustion and the model's documented
thinking-token billing (pricing page: "Output price (including thinking tokens)").

### L2. Surface taxonomy — what each access regime supports, with query cost

Let `l = 151,936` (our vocab), `d = 1536` (hidden), `k` = returned top-k, `B` = logit range,
`ε` = precision. Costs are **queries per one full logit vector**, which is the unit every
gradient-free method ultimately consumes.

| surface | who does this | queries / logit vector | source |
|---|---|---|---|
| all logits | textbook | 1 | Carlini §4 |
| top-k logprobs (+ logit bias) | OpenAI, 2024 | `l/(k−1)` = 37,984 at k=5 | Finlayson Table 1: `v/(k−1)` |
| top-5 logprobs, after image recovered | — | `O(d)` = 1,536 | Finlayson Table 1 "LLM Image", 800 calls for their model |
| top-1 + **binary** logit bias | their strongest restricted attack | **3.7** per logit → **562,163** for `l` | Carlini §7.1: "recovers 18 bits of precision at just 3.7 queries per logit" |
| top-1 identity only, no bias | information-theoretic floor | ≥ `l·log₂(B/ε)/log₂(l)` = **234,573** | Carlini Lemma G.1 |
| **text only (this key)** | — | **undefined** | no per-token surface at all (L1 call 4) |

Arithmetic for the last two rows, shown:

```
l = 151,936     log2(l)        = 17.213
B/ε = 100/1e-6 = 1e8          log2(1e8) = 26.575
bound = 151,936 * 26.575 / 17.213 = 4,037,700 / 17.213 = 234,573 queries
3.7 per logit * 151,936                                 = 562,163 queries
```

The 3.7 figure is *not* achievable here: it requires a `logit_bias` knob that Gemini's
`generateContent` does not expose, and it is a per-logit binary search — i.e. it needs both
a bias channel **and** the identity of the argmax token. This key has neither.

### L3. Measured: top-1 is enough for a token *behaviour*, and blind to *magnitude*

`code/f2_top1_only.py`, three simulated API views of the local model at an **identical
16-query budget** (greedy coordinate ascent, 2-token suffix, 8 single-token candidates,
1 pass). Base prompt `def parse(x):\n`. Metric: `TE = logsumexp(logits[comment]) −
logsumexp(logits[code])` at the last position, computed white-box *after* the search.

| oracle | suffix found | TE base | TE final | gain | informative steps | argmax |
|---|---|---|---|---|---|---|
| top-20 + logprobs | `#\n` | −6.968 | **+1.000** | +7.968 | 1.00 | `#` |
| top-1 + its logprob | `#\n` | −6.968 | **+1.000** | +7.968 | 1.00 | `#` |
| top-1 identity only | `#\n` | −6.968 | **+1.000** | +7.968 | 1.00 | `#` |

**This is a null result about the experiment, and I report it as such.** All three arms are
identical because the task ("make `#` the argmax") is solvable by *inserting the token*. A
label-only oracle is fully sufficient when the behaviour is a single token and that token is
in the pool. The measurement therefore does **not** discriminate between feedback regimes; it
only establishes the floor: **16 queries** of 1-bit feedback move TE by +7.97, which is
*larger* than the L8 steering vector's +5.81 on the same prompt (`TE −6.97 → −1.15` at
coef 2.0, measured in the same run). Prompting wins on this metric, exactly as AxBench says.

What it does **not** show — and what the constraint now prevents me from measuring — is the
magnitude regime: once `#` already wins, top-1 returns the same symbol for a good and a
better suffix, so a graded objective (match the vector's +5.81 nats, or its 151,936-dim logit
delta) is unobservable. That is where the information ceiling actually bites, and it needs
the top-k arm to demonstrate a difference.

**Memory constraint (required disclosure).** This run loaded `.hf/bf16-hf` as **fp32**
(`dtype=torch.float32`, eager attention, 4 threads) with no `mw.guard.require` call — the
guard rule did not exist when it started. That is precisely the load pattern that later froze
the machine. It completed in ~30 s per arm with peak RSS in the ~6 GB range, but it must not
be repeated: any continuation must use `mw.guard.require(path)` +
`dtype=torch.float16, low_cpu_mem_usage=True`, one model at a time, or a ≤200M model.

### L4. Embeddings: the only live surface, and what it is worth

One `batchEmbedContents` call, 8 texts (4 comment-style, 4 code-style), 3072 dims, unit norm:

| measurement | value |
|---|---|
| cosine(mean_comment, mean_code) | **0.9629** |
| projection of the 4 comment texts on the diff-in-means axis | +0.111, +0.121, +0.168, +0.056 |
| projection of the 4 code texts | −0.084, −0.148, −0.176, −0.141 |
| sign separation | **8/8** |

So the *concept* is linearly decodable from the only surface this key exposes. That is a real
capability and it is worth stating plainly: an embedding-only API supports **measurement**
(probe the behaviour, score prompts, build a classifier) but not **control**, and not
mechanism. Three reasons, in increasing order of hardness:

1. The embeddings come from `gemini-embedding-001`, a *different* model from any chat model on
   the key. A direction in its output space is not a direction in any residual stream. `[OBS]`
   for the endpoint name; `[INF]` for the "different weights" claim, which Google does not
   document either way — mark it `UNVERIFIED`.
2. The vector is **pooled over tokens and L2-normalised**, so both position information and
   magnitude are destroyed. The activation-steering object is per-position and scale-bearing.
   `[OBS]` norm = 1.0000.
3. There is no injection channel. Even a perfect direction in that space cannot be added to
   anything.

Published work on this surface recovers the **input text**, not the computation: vec2text
(Morris et al., EMNLP 2023, arXiv:2310.06816) reconstructs "92% of 32-token text inputs
exactly" from embeddings. That is an inversion of the *text*, and it is the ceiling of what
embedding access buys: you learn what went in, not what the model did with it.

### L5. Revised arithmetic — dollars are trivial, requests are the wall

One GCG-shaped run, `B=512`, `T=500`, 60 tokens per candidate (Zou 2023 config, §3.1):

```
queries   = 512 * 500                        = 256,000
tokens    = 256,000 * 60                     = 15,360,000
cost      = 15.36 M * $0.75/M                = $11.52      (gemini-3.8-flash, standard)
```

At 562,163 queries (the top-1+bias route): `562,163 × 60 = 33.7 M tokens = $25.3`.
Even a 2,000-query search costs `2,000 × 60 = 120 k tokens = $0.09`.

**So the price of a gradient-free loop on a 2026 flash model is between 9 cents and $26.**
The binding constraint is not money, it is (a) request count against per-project RPM/RPD,
which Google does not publish for the free tier ("Specified rate limits are not guaranteed"),
and (b) that no per-token surface exists at all, so the loop has to run on generated *text*.

### L6. Revised verdict

The main report's verdict — *a logprob-only loop recovers behaviour, not mechanism* — is
**strengthened, not weakened**, by the live API, and the revision is sharper than "no
logprobs":

1. **On this key, the loop is not logprob-only; it is logprob-free.** Calls 2 and 3 fail on
   two models, the orchestrator's six fail on the rest, and the Interactions API removes
   `candidates` by design. There is no top-k to be information-limited by.
2. **A logprob-free surface does not even give you the argmax token**, because the first
   emitted tokens are thinking tokens (call 4). The classic label-only attacks assume
   `argmax` of the next-token distribution; here that object is not exposed.
3. **Therefore no mechanism recovery is possible on this surface** — not "expensive", not
   "lossy": the observable is a string, and the target object is a per-position vector in a
   different model. The identifiability argument in F3 of the main report already says a
   black-box behavioural success cannot be attributed to a direction; the live API removes
   even the behavioural precision that would make the attempt well-posed.
4. **What survives is measurement through embeddings** (L4): 8/8 linear separation of the
   comment/code concept. That is a legitimate product surface (probe + score), it is not a
   steering surface, and it says nothing about the chat model's internals.
5. **What the local measurement adds** (L3): for a single-token behaviour, 16 queries of
   1-bit feedback beat the L8 steering vector's logit effect (+7.97 vs +5.81 nats). If the
   goal is *control of a token behaviour*, an API prompt search is cheaper and stronger than
   a steering vector. If the goal is *mechanism*, neither the API nor the search touches it.

### L7. The cheapest experiment that would prove the rest (not run — memory constraint)

**Claim to test:** on a genuinely top-1-only surface, a gradient-free loop can match a known
steering vector's *graded* effect and do so through the same internal change.

**Protocol (≈$0.10 + minutes):**
1. Surface: the real API, text-only, or a local ≤200M model (`mw.guard.require`, fp16,
   `low_cpu_mem_usage=True`) with a wrapper that returns only the argmax token.
2. Target: the L8 comment-mode vector's logit delta on a held-out prompt (a 151,936-dim
   vector; on the small model, its own analogous vector).
3. Arms, identical budget (2,000 queries): (a) coordinate/GA search on top-1 identity;
   (b) the same search with top-k logprobs (simulated locally only); (c) 200 random suffixes.
4. Readout: cosine(best prompt's L8 residual delta, v) versus the 95th percentile of the
   random-suffix null — the L1 criterion from `VALIDATION.md`, not a dose-response curve.
5. **Pre-registered prediction:** (a) at or below the null; (b) above it only when the
   objective is the graded delta, which is exactly the signal this API withholds. If (a)
   clears the null, the owner's backward loop exists in a form I have not found.

Cost: `2,000 × 60 = 120 k tokens ≈ $0.09` on gemini-3.8-flash; the local white-box check is
minutes on a ≤200M model. This is the whole experiment; there is nothing more expensive that
would be more decisive.
