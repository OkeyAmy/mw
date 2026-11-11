# F1 — Distil, interpret, transfer? The surrogate loop for an API-only teacher

**Question owned:** an API-only model exposes no hidden states. Is there a closed loop
that still yields *mechanistic* insight — distil a local surrogate from the teacher's
outputs, interpret the surrogate, and have the findings transfer to the teacher?

**Verdict up front:** **No — it is behavioural mimicry, with one narrow exception our
target does not satisfy.** Distillation reliably transfers *behaviour*. It does not
transfer *mechanism*, and the loop cannot detect the difference, because the only
measurement that would distinguish them requires the teacher's internals, which is the
thing the API withholds. The one exception is narrow and real: when the mechanism is a
member of a **finite, pre-registered hypothesis class** and the teacher is available as
an oracle for discriminating queries, behavioural extraction can identify the algorithm
up to equivalence (Weiss et al. 2018; §F7). That is *behavioural mechanistic inference*,
not surrogate interpretation, and it does not need a surrogate at all.

**Reading conventions:** `[OBS]` = measured/quoted from a fetched source (link in
§Sources). `[INF]` = my inference. `[SPEC]` = speculation. `[preprint]` = not
peer-reviewed. Per `VALIDATION.md` rule 3, **no claim here about
Qwen2.5-Coder-1.5B's internals is verified** — we cannot introspect an API model, which
is the whole problem.

---

## TL;DR

1. **The loop is well-defined but cannot close on the teacher.** Every step after
   "distil" measures the surrogate. The surrogate's internals are facts about the
   surrogate. Nothing in the procedure returns a fact about the teacher that was not
   already implied behaviourally.
2. **Extraction is real, cheap, and tells you nothing about internals.** 265–4,450
   queries for >99% agreement on small models (Tramèr et al. 2016); 100 queries to
   task-clone a 175B model (LoRD 2024); the GPT-3.5 projection layer for <$2,000
   (Carlini et al. 2024). Fidelity is a behavioural quantity; extraction papers say so
   explicitly.
3. **The one published surrogate-vs-teacher *circuit* comparison is not black-box.**
   GPT-2 → DistilGPT-2 lost one circuit completely (IOI: 35.2% vs 94.8%, zero
   name-mover preservation) and remapped the other (GT, r = 0.945) — but DistilGPT-2 was
   trained with a **cosine-distance loss on teacher hidden states** and **initialised
   from every other teacher layer** (Sanh et al. 2019). It had the internals. `[OBS]`
4. **Under genuinely output-only distillation, mechanisms diverge, systematically.**
   Logit matching has a *simplicity bias* — students learn the simplest mechanism in the
   distillation set, which may be one the teacher does not possess (Wu et al. 2024);
   distilled models contain *unique* feature directions (Baek & Tegmark 2025);
   distillation *impairs* induction heads (Goyal et al. 2025); students
   "reorganize, compress, and discard teacher components" (Haskins & Adams 2025).
5. **The usual fidelity bridge is broken.** CKA between a teacher and a
   randomly-initialised (then distilled) student is already 0.939/0.925/0.885, and
   sharing the teacher's init moves it to 0.951/0.937/0.890 with no functional change
   (Stanton et al. 2021, Table 1). A student whose IOI circuit is dead still has mean
   CKA = 0.921 with its teacher (Faumi & Sumanathilaka 2026). **CKA cannot be the
   evidence.**
6. **Our hardware forbids a big-enough surrogate.** A 30M student is 1.9% of the 1.54B
   teacher. A 27%-size student (Qwen-0.5B-class, 0.42B) costs ~88 CPU-hours to train on
   5M tokens — and 5M tokens is ~2 orders of magnitude below what produced the published
   students. §6.
7. **The decisive test is a symbolic known-circuit teacher, not a language model.** One
   CPU-hour, no download, and it isolates "can output-only training recover a known
   circuit?" from "is our student too small?" — see *The decisive cheap experiment*.

---

## Findings

### F1. Model extraction: what is actually achieved, and why it does not answer the question
[source: Tramèr et al., *Stealing Machine Learning Models via Prediction APIs*, USENIX
Security 2016, arXiv:1609.02943]

**What it is.** Query the API, fit a substitute. The paper's claim is behavioural by
construction: `[OBS]` "we show simple, efficient attacks that extract target ML models
with near-perfect fidelity for popular model classes including logistic regression,
neural networks, and decision trees." Their Table 1: Amazon logistic regression, 650
queries, 100% agreement; BigML decision tree, 4,013 queries, 100%; an MLP on
Adult/Race reaches `[OBS]` "99.96" test agreement at 11,125 queries.

**Why it matters.** >99% output agreement is *easy*. If output agreement implied
mechanism, mechanistic interpretability of closed models would be solved. It is not,
because the extracted model is one member of an equivalence class.

**The equivalence class is a theorem, not an engineering limitation.**
[source: Jagielski et al., *High Accuracy and High Fidelity Extraction of Neural
Networks*, USENIX Security 2020, arXiv:1909.01838]

`[OBS]` "Given access to input-output behavior, the best we can do is identify the
equivalence class the oracle belongs to." And, decisively for the two objectives:
`[OBS]` "a high-fidelity extraction should replicate the errors of the victim, whereas a
high-accuracy model should instead try to make an accurate prediction." Accuracy and
fidelity *conflict*. Their functionally-equivalent attack (weight recovery) works only
for 2-layer ReLU nets queried in float64 — 100% fidelity up to 200k parameters, decaying
above (`[OBS]` 99.98% at 100k params).

**LLM-specific extraction is a mature, cheap field.** `[OBS]`

| result | access assumed | cost / scale |
|---|---|---|
| Ada/Babbage **projection matrix** recovered; GPT-3.5 hidden dim recovered (Carlini et al. 2024, arXiv:2403.06634) | full logits | `[OBS]` "under $20" / "<$2,000" |
| 175B commercial model task-cloned into an 8B local model, **100 queries** (LoRD, arXiv:2409.02718, 2024) | text responses | `[OBS]` "statistically similar … for tasks not requiring extra knowledge" |
| ChatGPT-3.5 task distillation, **73% exact-match** (Model Leeching, arXiv:2309.10544, 2023) | text responses | `[OBS]` "$50 in API cost" |
| 6-layer student of a 6-layer teacher, 10k queries, `[OBS]` "97.6% of the … teacher's hidden-state geometry" and 7.31% perplexity increase (arXiv:2509.00973, 2025) | top-k logits | 24 GPU-hours |

Two things to notice about that last row, because it is the strongest "internals transfer"
claim in the extraction literature and it is misleading if read carelessly: (a) the
teacher was **already 6 layers**, so no depth reduction was required; (b) "hidden-state
geometry" is a cosine-similarity statistic, which §F4 shows is near-ceiling for unrelated
models. `[INF]` It is not evidence that the student implements the teacher's circuits.

**Cost for us.** A 10M-token output corpus is order **$6–$150** depending on tier
(`[SPEC]`, prices volatile) — i.e. cheap in money, expensive in *permission*: most
frontier providers' terms forbid training competitors on outputs, which is a
non-technical gate this document does not resolve.

---

### F2. Distillation-for-interpretability: the surrogate literature is behavioural, and knows it
[source: Ribeiro et al., *"Why Should I Trust You?"*, KDD 2016, arXiv:1602.04938]

LIME is the canonical interpretable surrogate: `[OBS]` "learning an interpretable model
**locally around the prediction**". It is a *local linear approximation of the decision
boundary*, fitted to perturbed inputs and the model's answers. Nobody claims LIME
recovers the network's mechanism; its guarantee is local fidelity to outputs. That is the
honest template for what an output-only surrogate can be.

**The only work that interprets a distilled student *as a mechanism* and checks it against
the teacher** is the pair of 2025–2026 circuit studies. Both are worth reading in full
because they are the closest thing to the experiment the owner is proposing, and they
disagree in an informative way:

**F2a. "Distilled Circuits" — students reorganise, compress, discard.**
[source: Haskins & Adams, arXiv:2505.10822, 2025]

`[OBS]` "student models can reorganize, compress, and discard teacher components, often
resulting in a stronger reliance on fewer individual components." Concretely, in
GPT-2 → DistilGPT-2: the teacher's layer-1 head T-L1-H5 ("similar member detection") has
`[OBS]` "No corresponding behavior … observed in the student head"; the induction head is
present but the student is `[OBS]` "copying across the broad structure … with
significantly smaller activation values"; and where functions survive, reliance is
concentrated — numeral detection drops performance by 87.73% in the student vs 33.18% in
the teacher, i.e. **the student has no backup heads**. Function survives; redundancy and
implementation do not.

**F2b. "What Survives Distillation?" — a compression/collapse dichotomy.**
[source: Faumi & Sumanathilaka, arXiv:2610.37157, 2026, `[preprint]`]

The sharpest data in this literature. GPT-2 (12L) → DistilGPT-2 (6L):

- **IOI circuit: dead.** `[OBS]` "35.2% accuracy vs. 94.8%, zero name-mover preservation";
  logit difference **−0.467** (a *sign reversal*, worse than chance); `[OBS]` "no
  matched head reproduces IO-selective attention: name-mover score preservation is at
  most 0% for every match." 19 of the 24 IOI heads are structurally absent in a 6-layer
  student.
- **Greater-Than circuit: remapped.** `[OBS]` "87.7% task performance recovery through
  active remapping of the teacher's layer-9 comparison head into student layer 4 (mean
  Pearson r=0.945)." 93.6% of examples have r > 0.8.
- **CKA over the *same* pair: 0.921 ± 0.109.** `[OBS]`

`[OBS]` "Distillation preserves circuits whose critical computations fit within the
student's available depth and destroys those whose pipelines extend beyond it."

**Three caveats that make this *not* the experiment we need.**
1. `[OBS]` DistilGPT-2 was trained with "a triple loss combining language modeling,
   distillation and cosine-distance losses" (Sanh et al. 2019, arXiv:1910.01108) and
   initialised from every other teacher layer. It had teacher **hidden states** and
   teacher **weights**. A closed API gives neither. `[INF]` The GT survival result is
   therefore an *upper bound* on what a black-box loop could achieve, and the bound is
   generous.
2. `[OBS]` The authors list as a limitation: "Incomplete verification of GT circuit
   survival … would require verifying the full circuit beyond the key head match."
3. It is a 2026 preprint, n = 1 teacher-student pair, two circuits, one dataset
   (TinyStories). `VALIDATION.md`'s standing warning about 2026 preprints applies.

**Confidence:** high that the *dichotomy* is real (two independent labs, different
methods); medium on the specific numbers.

---

### F3. Do distilled models share mechanisms? The direct evidence says "partly, and less than you want"

This is the load-bearing question. Four independent lines:

**F3a. Simplicity bias: the student learns the *easiest* mechanism, not the teacher's.**
[source: Wu, Lubana, Mlodozeniec, Kirk, Krueger, *What Mechanisms Does Knowledge
Distillation Distill?*, UniReps 2024, PMLR 243:60–75]

`[OBS]` "the standard distillation pipeline of matching teacher logits suffers from a
simplicity bias, resulting in the student learning primarily the simplest mechanisms in
the distillation dataset. If the distillation dataset and the teacher's pretraining
dataset have different distributions, **the student and teacher may learn entirely
distinct prediction mechanisms**."

Crucially, the two methods they find that *do* improve mechanism sharing are **Jacobian
matching** and **contrastive representation learning** — i.e. they require the teacher's
**gradients** or **embeddings**. Neither exists behind a text API. `[INF]` Every known
lever for mechanism transfer is an internals-access lever.

**F3b. Distilled models get their own features.**
[source: Baek & Tegmark, *Towards Understanding Distilled Reasoning Models: A
Representational Approach*, arXiv:2503.03730, 2025, ICLR 2025 workshop]

Crosscoders over Qwen-series teachers and their distilled students: `[OBS]` "distilled
models contain **unique reasoning feature directions**", usable to steer the model into
over-thinking or incisive-thinking mode. If the student's features were the teacher's,
they would not be "unique". `[INF]` Distillation *creates* representational structure as
well as copying it.

**F3c. Distillation actively harms a known mechanism: induction heads.**
[source: Goyal, Lopez-Paz, Ahuja, *Distilled Pretraining*, arXiv:2509.01649, 2025]

1B students, 8B teacher, 1T tokens, with and without distillation. `[OBS]`
"distillation impairs in-context learning capabilities, particularly the one modeled via
induction heads." Their mechanism story: induction is a **low-entropy** mapping; a
perfect teacher's soft labels carry no extra information beyond the hard label, and an
imperfect teacher *injects noise* into a deterministic copy. `[OBS]` "distillation can't
help — it can only match the hard label at best, and at worst, it actively hinders
learning for such copying tasks by softening the supervision."

This matters because induction heads are the single most universal mechanism we know:
`[OBS]` the two-head circuit (previous-token head → induction head) with a phase change
"early in training for language models of every size (provided they have more than one
layer)" (Olsson et al. 2022, transformer-circuits.pub/2022/in-context-learning). If even
*that* is degraded by distillation, the prior for any *less* universal mechanism is poor.

**F3d. The positive universality results all require both models open.**
- `[OBS]` SAE feature spaces across LLMs are highly similar under rotation
  ("Analogous Feature Universality") — but features are **paired by activation
  correlation**, which requires running both models on the same inputs (Lan et al. 2024,
  arXiv:2410.06981).
- `[OBS]` "most features are similar" between Transformers and Mambas, and induction
  circuits in Mamba are "structurally analogous" — with a real difference, the
  "Off-by-One motif" (Wang et al. 2024, arXiv:2410.06672). Again: SAEs trained on both.
- `[OBS]` Only **1–5%** of neurons are universal across five GPT-2 seeds — 1,253 of
  98,304 in GPT-2-medium — and they are *found* by correlating activations over 100M
  tokens (Gurnee et al. 2024, arXiv:2401.12181).

`[INF]` The universality literature is a *ceiling on what could transfer if you had
access*, not evidence that a black-box surrogate inherits it. Every one of these
measurements is an internals measurement.

---

### F4. Why the metrics cannot be the bridge: CKA, Procrustes, stitching

**F4a. CKA is near-ceiling for unrelated models.**
[source: Stanton, Izmailov, Kirichenko, Alemi, Wilson, *Does Knowledge Distillation
Really Work?*, NeurIPS 2021, arXiv:2106.05945]

Their Table 1 (ResNet-20 self-distillation, CIFAR-100, 10 trials) is the cleanest
null-calibration in the distillation literature:

| student init | top-1 agreement | predictive KL | CKA stage 1 | stage 2 | stage 3 |
|---|---|---|---|---|---|
| random | 77.174 (0.352) | 0.836 (0.016) | **0.939** (0.017) | **0.925** (0.027) | **0.885** (0.011) |
| teacher weights | 77.098 (0.238) | 0.838 (0.020) | 0.951 (0.017) | 0.937 (0.020) | 0.890 (0.015) |

`[OBS]` "A shared initialization does make the student slightly more similar to the
teacher in activation space (measured by CKA), but **in function space the results are
indistinguishable from randomly initialized students**."

Read that table twice. CKA ≈ 0.89–0.95 for a student that shares neither weights nor
initialisation with the teacher. This is the same failure mode as F3 §F1.3 reports from arXiv:2608.16347 (untrained
models scoring *higher* than trained on CKA), reproduced in a peer-reviewed paper. Any
CKA number we report without a random-init null is noise. This is `VALIDATION.md` L1 one
level up: **representation similarity is not mechanism similarity.**

**F4b. Stitching proves compatibility, not identity.**
[source: Bansal, Nakkiran, Barak, *Revisiting Model Stitching*, arXiv:2106.07682, 2021]

`[OBS]` "good networks of the same architecture, but trained in very different ways
(e.g.: supervised vs. self-supervised learning), can be stitched to each other without
drop in performance", and stitching "reveals aspects of representations that measures
such as centered kernel alignment (CKA) cannot."

The stitch layer is *trained*. So the claim is: there exists a simple map making the two
halves compose. That is exactly the Procrustes/affine result F3 already established
(retrieval 0.82–0.93 at final layers), and it is compatible with completely different
internal algorithms. `[INF]` Stitching is a *lower* bar than circuit identity, not a
proxy for it.

**F4c. The only study measuring both finds no monotone link.**
Faumi & Sumanathilaka report CKA 0.921, perplexity +52.1%, IOI dead and GT alive **in the
same model pair**. Whatever predicts circuit survival there, it is not CKA and it is not
perplexity. Their candidate explanation is **compressibility relative to student depth** —
a structural fact about the student, not a behavioural fact about the teacher.

**Answer to "is there a measured fidelity threshold?"** **No.** I found no published
number of the form "≥ X% output agreement ⇒ mechanism shared". The nearest quantitative
statements are: >99% agreement achievable on tiny models (Tramèr), 77–86% attainable on
ResNet self-distillation (Stanton), and 100% fidelity with *zero* mechanism guarantee
(Jagielski). Anyone asserting a threshold is inventing it. `[INF]`

---

### F5. What can never cross, and what can

| object | crosses? | evidence |
|---|---|---|
| **neuron / head index** | **no** | 1–5% of neurons are universal even across seeds of the same architecture+data (Gurnee 2024); Qwen-1.5B has 12 heads/28 layers, Qwen-0.5B has 14/24 (configs fetched) — no index correspondence even in-family |
| **layer-localised circuit topology** | **no** | 19/24 IOI heads structurally absent in a 6-layer student; "zero name-mover preservation" (2610.37157) |
| **weights** | **no** | only up to permutation/scaling; "the best we can do is identify the equivalence class" (Jagielski 2020) |
| **a claim that needs a teacher-side intervention** (patching, ablation, steering) | **no** | no interface exists; F3 §F5 |
| **feature *directions*, up to rotation** | **conditionally** | analogous feature universality (2410.06981), but paired via both models' activations |
| **task strategy / algorithm, at coarse grain** | **conditionally** | GT circuit remapped with r=0.945 (but with teacher init + hidden-state loss); induction circuit analogous across architectures (2410.06672) |
| **behavioural invariances and failure modes** | **yes** | substitute-model adversarial examples transfer at `[OBS]` 84.24% to a held-out API model (Papernot et al. 2017, arXiv:1602.02697) |
| **negative claims** ("the teacher cannot be doing X") | **yes** | a mechanism entails behaviour; absent behaviour refutes the mechanism. See F7 |

`[INF]` The asymmetry in that table is the whole answer. Positives require internals;
**negatives only require the API.** A black-box study can *eliminate* mechanisms. It
cannot *localise* them.

---

### F6. The logical core: why the loop's terminal measurement is behavioural

Write the loop explicitly.

```
H  = a candidate mechanism, expressed as a behavioural prediction
S  = surrogate trained on teacher outputs only
I(S) = mechanistic interpretation of S (patching, ablation, logit lens)
P  = new predictions I(S) makes that were NOT in the distillation data
q  = query the teacher on P
```

The step everyone skips: **what does `I(S)` license about the teacher?** Three possible
answers, and only one is available.

1. **Nothing** (the naive reading). `I(S)` is a fact about `S`.
2. **Something, if S and teacher are behaviourally indistinguishable on a probe set that
   discriminates mechanisms.** Then the teacher is in the same behavioural equivalence
   class as `S` *with respect to that probe set*. This is real but conditional, and its
   strength is exactly the discriminating power of the probe set.
3. **Something, if the surrogate and teacher share weights or training trajectory.**
   Not available.

Path (2) is the only live one, and note what it costs: **you must be able to say, in
advance, which observations would distinguish the candidate mechanisms.** That is a
finite-hypothesis-class requirement. For an LLM it is not satisfiable in general, and it
has nothing to do with the surrogate — you could run the same probes without distilling
anything.

Formally, this is F3's Proposition-1 argument from arXiv:2608.24662 applied one level up:
if `supp(Q) ⊆ supp(P_θ)`, a base model plus a policy is observationally identical to a
different model. Here: *a student that matches the teacher on all probes is
observationally identical to the teacher, including on probes it fails.* `[INF]`

**The one-sided test.** Mechanism M entails behaviour B(M). Observing ¬B refutes M.
Observing B does **not** confirm M, because many mechanisms entail B. Therefore:

> **Behaviour-only access supports exclusion of mechanisms, never identification.**

That is not a counsel of despair. Exclusion is falsifiable, and `VALIDATION.md` only
requires falsifiability. It just is not what "we read the model" means.

---

### F7. The exception that is real: finite hypothesis classes *are* identifiable from behaviour

[source: Weiss, Goldberg, Yahav, *Extracting Automata from Recurrent Neural Networks Using
Queries and Counterexamples*, ICML 2018, arXiv:1711.09576]

`[OBS]` "We present a novel algorithm that uses exact learning and abstraction to extract
a deterministic finite automaton describing the state dynamics of a given trained RNN. We
do this using Angluin's L\* algorithm as a learner and **the trained RNN as an oracle**."

This is a genuine black-box mechanistic extraction, and it works. Note precisely why:

1. The hypothesis class (DFAs with ≤ n states) is **finite and enumerable**.
2. L\* has **equivalence-query guarantees** — it terminates with a hypothesis
   *provably* equivalent to the oracle on the queryable domain.
3. The target behaviour is **regular**, so it is exactly representable.

Our target satisfies none of these. But the template is the right one, and it tells us
what a defensible "closed loop" claim looks like: **not "the surrogate's circuit is the
teacher's circuit", but "the teacher's behaviour is inconsistent with all mechanisms in
this enumerated family except M".**

The same logic at toy scale: `[OBS]` a single-layer, two-head attention-only transformer
reaches "perfect IOI accuracy", with the two heads specialising into "additive and
contrastive subcircuits", and a two-layer one-head model does it by query-key composition
(Adhikari, arXiv:2510.25013, ACL SRW 2026). A finite, small circuit family *is*
behaviourally identifiable — which is exactly why this is the right testbed for E0.

---

## Concrete implementation for our constraints

Hardware per `CONTEXT.md`: 4 physical cores, AVX2, ~17 GB free, measured ~54–67 GFLOPS
fp32 GEMM. Use **40 GFLOPS effective** (≈60–75% of peak, allowing for small-matrix
inefficiency and optimiser overhead). All arithmetic below is derived, not asserted.

**FLOP model** `[INF]` arithmetic: forward ≈ 2N FLOPs/token; backward ≈ 2× forward;
so training ≈ 6·N·D for N parameters and D tokens. (Same convention as Kaplan et al.
2020, arXiv:2001.08361, which uses C ≈ 6NBS.)

### Budget table

| object | N (derived) | 10M tokens | 5M tokens |
|---|---|---|---|
| symbolic-IOI teacher (2L, d=64, V=64, attention-only) | 36.9k | 0.015 h | 0.008 h |
| symbolic-IOI student (3L, d=64, V=64, attention-only) | 53.2k | 0.022 h | 0.011 h |
| GPT-2 small **teacher, forward only** | 123.5M | 17.2 h | 8.6 h |
| student 6L d384 V=50257 | 29.9M | **12.5 h** | 6.2 h |
| student 4L d256 V=50257 | 16.0M | 6.7 h | 3.3 h |
| Qwen-0.5B-class student (config-derived) | 0.422B | 176 h | **88 h** |
| Qwen-1.5B **teacher, forward only** | 1.544B | 214 h | 107 h |

Worked examples so the reader can check:
- 6L/d384 student, 10M tokens: `6 × 2.99e7 × 1e7 = 1.795e15` FLOPs;
  `1.795e15 / 4.0e10 = 4.49e4 s = 12.5 h`.
- GPT-2 small **teacher forward**, 10M tokens: `2 × 1.235e8 × 1e7 = 2.47e15`;
  `6.18e4 s = 17.2 h`.
- Qwen-0.5B-class student, 5M tokens: `6 × 4.22e8 × 5e6 = 1.266e16`;
  `3.17e5 s = 88 h`.

Note the counter-intuitive planning fact in rows 3–4: **generating the corpus costs more
than training the student**, because the teacher is 4× larger and each teacher token is
`2N`, not `6N`/4. On this box the GPT-2 experiment is corpus-generation-bound.

### The size argument, which is the real blocker

| student | params | fraction of Qwen-1.5B (1.544B) |
|---|---|---|
| 6L d384 | 29.9M | **1.9%** |
| 4L d256 | 16.0M | 1.0% |
| Qwen-0.5B-class | 422M | **27.4%** |

`[OBS]` DistilGPT-2 is 82M/124M = 66% of its teacher and *still* lost the IOI circuit
entirely. `[INF]` A 1.9% student is not a plausible host for a 1.5B model's circuits;
a 27% student is, but costs 88 CPU-hours per 5M tokens, and 5M tokens is far below the
scale of the published students, which were trained on their teacher's pretraining corpus
`[INF]`. For calibration: LeNet-5 self-distillation reached >99% agreement only with
60k–700k samples, while ResNet-56 on CIFAR-100 plateaued at `[OBS]` "a modest 86% test
agreement" even with GAN augmentation and the best augmentation policy (Stanton 2021).
Both are at ~100% capacity ratio. We would be at ~10^7 tokens and 1.9%.

**Honest expected fidelity for a real API surrogate (30M student, 10M text-only output
tokens)** `[SPEC]`, reasoning shown: same-architecture self-distillation at 100% capacity
reaches 77–86% top-1 (ResNet-56/CIFAR). Our ratio is 1.9% and the domain is code. Predict
**perplexity ratio 1.5–2.5×, top-1 agreement 40–60% in-distribution, near-zero on held-out
languages, and no faithful mid-depth circuit**. That is a *stand-in*, not a *surrogate in
the mechanistic sense*.

**What I would actually build for our constraints.** Distillation is not the instrument.
The instrument is *discriminative behavioural probing with a pre-registered finite
mechanism family* (§F7). It costs one forward pass per probe, needs no training, and its
conclusions are falsifiable. The surrogate's only defensible role is **hypothesis
generation**: interpret the student, extract candidate algorithms, then test those
algorithms against the API teacher. That is a real pipeline. It is just not a microscope.

---

## The decisive cheap experiment

The question is: *does output-only training recover a known circuit?* The literature's
answer is confounded by DistilGPT-2's teacher-hidden-state loss and layer initialisation.
**Nobody has run the pure black-box version on a known circuit.** That is the gap, and it
is cheap.

### E0 — Symbolic known-circuit transfer (DECISIVE, ~1–2 CPU-hours, 0 GB download)

**Teacher T.** 1-layer, 2-head attention-only transformer, d=64, V=64, trained on the
symbolic IOI task of Adhikari (arXiv:2510.25013). `[OBS]` That paper reports perfect
accuracy and two heads specialising into additive/contrastive subcircuits. Reproduce,
then **verify the circuit yourself** with three probes: per-head ablation, residual-stream
decomposition, attention-pattern inspection. Record T's circuit as ground truth.

**Distillation corpus.** 5M tokens sampled from T (a mix of greedy and temperature-1
sampling so the corpus is not degenerate). Hold out 0.5M.

**Students (four conditions, all random init, no teacher weights):**
- `S-hard`: 3L, 4 heads, d=64 (53k params — **larger and deeper than the teacher**, so
  capacity is not a confound), cross-entropy on T's **sampled tokens** (text only).
- `S-soft`: same architecture, KL on T's **full next-token distribution** (logit access).
- `S-data`: same architecture, trained on **ground-truth** symbolic IOI data (control for
  "the circuit is data-driven, not teacher-transferred").
- `S-rand`: same architecture, trained on the outputs of a **randomly-initialised**
  teacher (structure-free control).

**Interpretation.** Run the same three probes on every student, **blind** (label students
A–D; an analyst who does not know which is which). Then unblind.

**Pre-registered pass criterion** (state it before running): the student's mechanism
counts as *the same* only if all three hold —
1. ablating the matched heads reduces accuracy by ≥ 50% of the teacher's drop;
2. matched-head attention patterns correlate r ≥ 0.8 with the teacher's on held-out
   data, at matched *functional* (not index) position;
3. OOD profile matches: longer sequences, more names, unseen name pairs.

**Pre-registered prediction** `[SPEC]`, stated before the run: `S-hard` will learn *a*
circuit but probably not T's — it has enough depth to implement "attend to the
non-repeated name" directly, which is the simplicity-biased solution Wu et al. describe.
`S-soft` at matched capacity has the best chance of matching. **If `S-soft` matches but
`S-hard` does not**,
then logit access is doing the work — and an API without logprobs cannot supply it.
**If neither matches**, output-only mechanism transfer is refuted even in the most
favourable possible setting, and the API loop is dead with no further work.

**Why this is decisive and not a toy.** It removes every confound at once: capacity
(student ≥ teacher), tokeniser (identical), domain (synthetic), data volume (adequate at
this scale), and ground truth (we know the answer). Whatever it shows about transfer is
attributable to **the information channel**, which is the only variable we cannot fix.

**Cost:** teacher 5 min; each student ~10 min (53k params × 5M tokens = 1.6e12 FLOPs
= 40 s of pure compute, so wall-clock is Python overhead, not FLOPs); analysis 20 min.
**Total ≈ 1.5 h, 0 GB.**

### E1 — Real-LM replication (GPT-2 small, ~30 CPU-hours, 0.5 GB)

Only if E0 is positive. GPT-2 small is the only model with two independently validated,
end-to-end documented circuits (IOI: Wang et al. 2022, arXiv:2211.00593, 26 heads / 7
classes; GT: Hanna et al. 2023).

- Teacher forward on 10M tokens: **17.2 h**.
- Student: 6L, d=384, 6 heads, 29.9M params, **random init, hard labels only** (this is
  the difference from DistilGPT-2): **12.5 h**.
- Controls: `S-data` on TinyStories ground truth; `S-rand` on a randomly-initialised
  GPT-2. **Do not skip these** — without `S-data` you cannot tell "distillation
  transferred the circuit" from "both models learned the obvious thing from data".
- **Harness-validation step first (~1 h):** run the same readouts on the public
  DistilGPT-2 and check you reproduce `[OBS]` 35.2% IOI / 87.7% GT. If you do not, your
  harness is wrong, not the literature. (This is `LEDGER.md` X6's lesson: validate the
  validator.)
- Pre-registered prediction: IOI < 50%, no name-mover; GT ≥ 70% if the student can host a
  comparison head; induction present but weaker (Goyal 2025). Falsifier: IOI ≥ 80% with a
  name-mover at matched functional depth.

### E2 — The actual API surrogate (do not run for science)

10M output tokens (~$6–$150), 30M student, 12.5 CPU-hours. It yields a stand-in with
40–60% behavioural agreement `[SPEC]`. Its internals are facts about itself. **Run it
only if the product needs a local stand-in, never as evidence about the API model.**

---

## What NOT to do (traps)

1. **Do not report CKA between a surrogate and a teacher without a random-init null.**
   Randomly-initialised (then distilled) ResNet students score 0.885–0.939 (Stanton,
   Table 1). The number is
   meaningless alone. Same for Procrustes without a shuffled-label null (F3 §F1.3).
2. **Do not call DistilGPT-2 a black-box distillation.** It used a cosine loss on teacher
   hidden states and layer initialisation (Sanh 2019). Citing its GT survival as evidence
   for an output-only loop is a category error, and it is the most likely error anyone
   reading this literature will make.
3. **Do not let "the student has an induction head" stand as "the mechanism transferred".**
   Haskins & Adams found the induction head present but with `[OBS]` "significantly
   smaller activation values" and no backup heads. Presence ≠ function ≠ equivalence.
4. **Do not skip the `S-data` control.** If a student trained on ground-truth data
   develops the same circuit, the circuit came from the data, not the teacher. This single
   control kills most "distillation transferred X" claims.
5. **Do not transfer a finding upward in scale.** A 30M student's circuit tells you about
   30M students. `[OBS]` Universal-neuron fraction is *higher* in GPT-2-small (4.16%)
   than in GPT-2-medium (1.23%) — small models are atypical.
6. **Do not use the surrogate's steering vector on the teacher and call it a result.**
   That is exactly F3's T1/T2 transfer problem, now with a *third* model in the chain.
   Distillation does not solve transfer; it relocates it and adds a hop.
7. **Do not treat a 2026 preprint as settled, and do not fuse the two circuit papers into
   one "result".** 2505.10822 measures component-level alignment; 2610.37157 does
   circuit-level patching. They are complementary, neither replicates the other, and both
   are n = 1 teacher-student pairs.

---

## Open questions for other agents

- **A2 / D1 (harness):** can our existing `forward_trace` emit per-head attention and
  residual decomposition for a 2-layer, 2-head model with no extra engineering? E0 depends
  on it, and it is the cheapest thing in the plan.
- **A6 (sceptic):** is "the teacher's behaviour is inconsistent with mechanism family M"
  an admissible finding under `VALIDATION.md`? I argue yes — it is falsifiable, has a
  named falsifier, and needs no randomness controls. But it is a *negative*, and the
  ledger currently has no category for those.
- **F3 (transfer):** E0 should run **before** T1/T2. It is the same class of test but with
  a known ground truth, so it calibrates whether T1/T2's null result means "transfer is
  hard" or "our readout is broken".
- **E3 (wedge):** the quantisation certificate is unaffected — fp16 vs Q4 within one model
  needs no cross-model anything. This document removes a distraction, not a product.
- **C-7 (model choice):** an instruct model makes the surrogate story strictly worse —
  RLHF adds a behavioural layer (refusal style, formatting) that is *more* imitable and
  *less* mechanistic than base-model next-token prediction. Any surrogate study should use
  base weights.

---

## SCHEMA CHANGE

One addition, needed so a trace cannot be silently over-interpreted:

```json
"provenance": {
  "access": "blackbox-text" | "blackbox-logits" | "weights" | "weights+acts",
  "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
  "surrogate_of": null | "teacher_id",
  "distil": {"loss": "ce" | "kl" | "cosine-hidden", "init": "random" | "teacher-layers",
             "tokens": 10000000}
}
```

Rationale: the same `logit_lens_topk` array means different things depending on whether
`access` is `weights+acts` (a measurement of the teacher) or `blackbox-text` (a
measurement of a surrogate). Without this field, `LEDGER.md` cannot distinguish a
mechanistic finding from a behavioural one, and the F1 verdict becomes invisible at the
schema level.

---

## Build next (ordered, with effort)

1. **`code/e0_symbolic_ioi.py`** — teacher + 4 students + IOI data generator. ~200 lines,
   **3 h**. No download.
2. **`code/circuit_probes.py`** — per-head ablation, residual decomposition, attention
   correlation, blind-labelled output. ~150 lines, **2 h**. Test on the teacher first
   (where ground truth is known).
3. **Run E0.** **1.5 h compute.** Pre-register the pass criterion in a config file
   *before* looking at any student.
4. **Decide.** If `S-soft` and `S-hard` both fail → write it up as a negative and stop.
   If `S-soft` passes and `S-hard` fails → the answer is "logit access is mandatory", and
   we test whether the API exposes logprobs.
5. **Only then** E1 (GPT-2, ~30 h) and the DistilGPT-2 harness check (~1 h).

Total to a defensible answer: **~6 h engineering + ~2 h compute, 0 GB for E0** (add ~31 h
and 0.5 GB if E1 runs; add ~88 h+ for a Qwen-0.5B-class surrogate). That compares with
~30–90 h for the language-model version, which answers a strictly weaker question.

---

## Sources (all fetched unless marked)

| cited as | title | year | link | status |
|---|---|---|---|---|
| Tramèr et al. | Stealing Machine Learning Models via Prediction APIs | 2016 | arXiv:1609.02943 | USENIX Security 2016 |
| Jagielski et al. | High Accuracy and High Fidelity Extraction of Neural Networks | 2020 | arXiv:1909.01838 | USENIX Security 2020 |
| Carlini et al. | Stealing Part of a Production Language Model | 2024 | arXiv:2403.06634 | ICML 2024 |
| LoRD | Alignment-Aware Model Extraction Attacks on LLMs | 2024 | arXiv:2409.02718 | preprint |
| Model Leeching | An Extraction Attack Targeting LLMs | 2023 | arXiv:2309.10544 | preprint |
| — | Clone What You Can't Steal | 2025 | arXiv:2509.00973 | preprint |
| — | Survey on Model Extraction Attacks and Defenses for LLMs | 2025 | arXiv:2506.22521 | preprint |
| Ribeiro et al. | "Why Should I Trust You?" (LIME) | 2016 | arXiv:1602.04938 | KDD 2016 |
| Stanton et al. | Does Knowledge Distillation Really Work? | 2021 | arXiv:2106.05945 | NeurIPS 2021 |
| Wu et al. | What Mechanisms Does Knowledge Distillation Distill? | 2024 | PMLR 243:60-75 | UniReps 2024 |
| Haskins & Adams | Distilled Circuits | 2025 | arXiv:2505.10822 | preprint |
| Faumi & Sumanathilaka | What Survives Distillation? | 2026 | arXiv:2610.37157 | `[preprint]`, n=1 |
| Baek & Tegmark | Towards Understanding Distilled Reasoning Models | 2025 | arXiv:2503.03730 | ICLR 2025 workshop |
| Goyal et al. | Distilled Pretraining | 2025 | arXiv:2509.01649 | preprint |
| Sanh et al. | DistilBERT | 2019 | arXiv:1910.01108 | NeurIPS 2019 workshop |
| Olsson et al. | In-context Learning and Induction Heads | 2022 | transformer-circuits.pub/2022/in-context-learning | Anthropic, not peer-reviewed |
| Gurnee et al. | Universal Neurons in GPT2 Language Models | 2024 | arXiv:2401.12181 | preprint |
| Lan et al. | Quantifying Feature Space Universality via SAEs | 2024 | arXiv:2410.06981 | preprint (v4 2025) |
| Wang et al. | Towards Universality: Mechanistic Similarity Across Architectures | 2024 | arXiv:2410.06672 | preprint |
| Bansal et al. | Revisiting Model Stitching | 2021 | arXiv:2106.07682 | NeurIPS 2021 |
| Weiss et al. | Extracting Automata from RNNs | 2018 | arXiv:1711.09576 | ICML 2018 |
| Adhikari | Minimal Circuits for IOI in Attention-Only Transformers | 2025 | arXiv:2510.25013 | ACL SRW 2026 |
| Papernot et al. | Practical Black-Box Attacks against Machine Learning | 2017 | arXiv:1602.02697 | AsiaCCS 2017 |
| Kaplan et al. | Scaling Laws for Neural Language Models | 2020 | arXiv:2001.08361 | preprint |
| Wang et al. | Interpretability in the Wild (IOI circuit) | 2022 | arXiv:2211.00593 | ICLR 2023 |
| — | DistilGPT-2 `config.json` | — | huggingface.co/distilbert/distilgpt2 | fetched: n_layer 6, n_embd 768, vocab 50257 |
| — | Qwen2.5-Coder-0.5B `config.json` | — | huggingface.co/Qwen/Qwen2.5-Coder-0.5B | fetched: 24 layers, 896, 4864, 14 heads, vocab 151936, tied |

---

## Validation blocks

**H1 — Output agreement does not imply representation agreement.**
```
HYPOTHESIS: A student can match its teacher's predictions while its activation geometry
            is indistinguishable from a randomly-initialised network.
TEST:       Fetched arXiv:2106.05945v2 HTML; read Table 1 (ResNet-20 self-distillation,
            CIFAR-100, 10 trials) and the accompanying text.
RESULT:     `[OBS]` "A shared initialization does make the student slightly more similar
            to the teacher in activation space (measured by CKA), but in function space
            the results are indistinguishable from randomly initialized students."
            Table 1: rand-init agreement 77.174, KL 0.836, CKA 0.939/0.925/0.885;
            teacher-init agreement 77.098, KL 0.838, CKA 0.951/0.937/0.890.
VERDICT:    CONFIRMED (source claim, peer-reviewed).
FALSIFIER:  A replication in which CKA differs materially between teacher-init and
            random-init students, or in which CKA predicts agreement.
```

**H2 — No published output-agreement ⇒ mechanism-transfer threshold exists.**
```
HYPOTHESIS: There is a measured fidelity threshold above which mechanisms are shared.
TEST:       Searched for and read the extraction/distillation/mechanism literature above
            (Tramèr, Jagielski, Stanton, Wu, Haskins & Adams, Faumi & Sumanathilaka,
            Baek & Tegmark, Goyal). Looked specifically for an agreement-vs-circuit curve.
RESULT:     No such curve found. The only study measuring both in one pair reports mean
            CKA 0.921 simultaneously with a dead IOI circuit (2610.37157). Jagielski
            states accuracy and fidelity are in conflict; Stanton reports 77–86%
            agreement as a ceiling for ResNet/CIFAR.
VERDICT:    REFUTED as a claim about the literature (no such threshold published).
            Any threshold asserted by us would be UNVERIFIED invention.
FALSIFIER:  A paper reporting circuit survival as a function of teacher-student agreement.
```

**H3 — Our hardware cannot host a plausible mechanistic surrogate of a 1.5B teacher.**
```
HYPOTHESIS: A student large enough to plausibly host Qwen-1.5B's circuits is not trainable
            on this box at a useful data scale.
TEST:       Derived parameter counts from fetched configs and the FLOP model
            6ND at 40 GFLOPS effective:
              Qwen2.5-Coder-0.5B: 151936*896 + 24*(4*896^2 + 2*896*4864)
                                 = 136,134,656 + 286,261,248 = 422,395,904 params
                                 = 27.4% of the 1.544e9 teacher.
              Train 5M tokens: 6 * 4.224e8 * 5e6 = 1.267e16 FLOPs
                              / 4.0e10 = 3.17e5 s = 88.0 h.
              6L/d384 student: 29.92M params = 1.94% of teacher;
                              train 10M tokens = 6*2.992e7*1e7 = 1.795e15
                              / 4.0e10 = 4.49e4 s = 12.5 h.
RESULT:     27%-capacity student: 88 CPU-h for 5M tokens. 1.9%-capacity student: 12.5 h
            for 10M tokens. The comparable published data points are at ~100% capacity
            ratio (LeNet-5 60k–700k samples for >99%; ResNet-56 ~100k samples for 86%).
VERDICT:    CONFIRMED (arithmetic). Consequence: the surrogate is either too small or
            too expensive.
FALSIFIER:  A demonstration that a ~2%-capacity, 10^7-token student reproduces a teacher
            circuit under causal tests.
```

**H4 — The published positive (GT circuit survival) was not obtained black-box.**
```
HYPOTHESIS: DistilGPT-2 had access to teacher internals, so its survival result does not
            license an output-only loop.
TEST:       Fetched arXiv:1910.01108 abstract; fetched arXiv:2610.37157 §3 (models).
RESULT:     `[OBS]` Sanh et al.: "we introduce a triple loss combining language modeling,
            distillation and cosine-distance losses." Cosine-distance loss is computed on
            teacher hidden states. `[OBS]` 2610.37157 §3: DistilGPT-2 is "initialised from
            every other teacher layer".
VERDICT:    CONFIRMED. The loop in the literature had hidden states *and* weights.
FALSIFIER:  Documentation showing DistilGPT-2 was trained on text/logits only.
```

**H5 — A black-box mechanism extraction is possible when the hypothesis class is finite.**
```
HYPOTHESIS: Behaviour-only access can recover an algorithm exactly, given a finite
            hypothesis class and discriminating queries.
TEST:       Fetched arXiv:1711.09576 abstract (ICML 2018).
RESULT:     `[OBS]` "uses exact learning and abstraction to extract a deterministic finite
            automaton ... using Angluin's L* algorithm as a learner and the trained RNN as
            an oracle."
VERDICT:    CONFIRMED (source claim). This is the existence proof for the narrow
            exception, and it does not involve a surrogate.
FALSIFIER:  A demonstration that L* extraction requires internals.
```

**H6 — Does our target satisfy the finite-class condition?**
```
HYPOTHESIS: Qwen2.5-Coder-1.5B's mechanisms are identifiable from behaviour alone.
TEST:       Not run. The open 1.5B is a control, not the target; the target is an API-only
            model we cannot load, and no API budget exists for the query side.
RESULT:     —
VERDICT:    UNVERIFIED. `[INF]` The hypothesis class of "all algorithms a 1.5B transformer
            could implement" is not finite and not enumerable, so the condition fails.
FALSIFIER:  A finite, published circuit vocabulary that provably spans the behaviours of
            interest, plus a probe set that discriminates members of it.
```

---

## Verdict

**Behavioural mimicry, with one exception our target does not satisfy.**

The distillation loop is not incoherent, and it is not worthless. It is *mis-scoped*. It
produces a local model that behaves like the teacher, and a true and useful mechanistic
account of **that local model**. It produces **no** mechanistic fact about the teacher that
was not already available from the teacher's behaviour, because the two ways to convert
surrogate-internals into teacher-facts both fail:

- **shared weights / shared training trajectory** — unavailable behind an API, and
  DistilGPT-2's one positive circuit result depended on exactly this;
- **a finite, pre-registered mechanism family with discriminating probes** — the only
  sound route (Weiss 2018), but it is behavioural inference that needs no surrogate, and
  for a general LLM the family is not finite.

The empirical record points the same way and is unusually consistent for a young field:
output-only distillation has a *simplicity bias* that selects the easiest mechanism in the
distillation set, which may be one the teacher does not have (Wu 2024); distilled students
develop their own feature directions (Baek & Tegmark 2025); distillation *degrades* the
most universal mechanism we know, induction heads (Goyal 2025); and where a student is
compared to a teacher circuit-by-circuit, one of the two documented circuits is dead
(IOI: 35.2% vs 94.8%, zero name-mover preservation) while the other survives only through
remapping in a model that was *given the teacher's layers and hidden states* (r = 0.945,
Faumi & Sumanathilaka 2026).

The decisive counter-evidence that would change this verdict is concrete and cheap to
produce: **a student with random initialisation and no hidden-state loss, trained only on
a teacher's outputs, whose matched circuit passes ablation, attention-pattern and OOD
tests against a known teacher circuit.** Our E0 (above) is that experiment, at ~1.5
CPU-hours. It has not been run. Until it is, the prior is negative and the honest label on
any surrogate-derived claim about the teacher is `UNVERIFIED — behavioural only`.

**What survives as a real capability:** an API-only study can *exclude* mechanisms, because
a mechanism entails behaviour and absent behaviour refutes it. That is falsifiable,
publishable, and genuinely more than prompt engineering — but it is elimination, not
localisation, and it should be sold as such: **"we can prove what the model cannot be
doing."** Not, as `STATE_OF_PROJECT.md` puts it, "we can prove what changes inside it" —
for a model we cannot open, we can only prove what the change *isn't*.

**One-sentence rule:** *the surrogate is a hypothesis generator, not an instrument; only
run loops whose terminal measurement is inside a model you can open, and treat every
closed-model loop as behavioural science with a mechanistic vocabulary.*
