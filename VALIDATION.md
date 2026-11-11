# VALIDATION PROTOCOL — mandatory

Every agent claim must survive this before it enters `LEDGER.md`.
A finding without a validation block is **not a finding**.

## Required block per hypothesis

```
HYPOTHESIS: <falsifiable statement>
TEST:       <what was actually done — command, URL, or computation>
RESULT:     <observed output, quoted or numeric>
VERDICT:    CONFIRMED | REFUTED | UNVERIFIED
FALSIFIER:  <what observation would have proven it wrong>
```

## Rules

1. **Citations are claims.** If you cite a paper, repo, licence, or API field,
   you must have fetched it. Quote the line. "I recall that..." is `UNVERIFIED`.
2. **Numbers are claims.** Recompute them. Show the arithmetic.
   `151936 * 1536 * 2 bytes = 466,706,432 B = 467 MB` — show it, don't assert it.
3. **Model-behaviour claims must be run on the model.** Once the runtime exists,
   "layer 14 carries style" is not a finding until you patched layer 14 and
   measured the output change. Until then it is `UNVERIFIED — needs runtime`.
4. **Correlation is not causation.** A probe that decodes X does not show the
   model uses X. Only intervention (patching, steering, ablation) does.
5. **Negative controls are required.** For every positive result, run:
   - `coef = 0` must be bit-identical to no intervention
   - a random vector of matched norm must produce no systematic effect
   - a shuffled-label control for any probe
6. **State what would falsify you.** If you cannot name a falsifier, the claim
   is unfalsifiable and gets deleted.
7. **Separate observation from inference.** Mark each sentence:
   `[OBS]` measured, `[INF]` inferred, `[SPEC]` speculation.

## Hard-won lessons

### L1. Monotonicity is NOT specificity. (found 2026-11-10, on bf16)
For any fixed direction `v`, adding `α·v` to a residual stream moves the final
logits approximately linearly in `α` (first-order Taylor). Therefore a monotone
dose-response curve is produced by *random vectors too*. On bf16, 2 of 10 random
directions achieved |spearman| = 0.86 and 0.89 with spans of 12.6 and 12.3 —
both LARGER than the real steering vector's 10.8.

**Consequence:** `spearman(coef, effect) >= 0.8` is a test of linearity, not of
a meaningful direction. It must never be used alone. A steering claim additionally
requires at least one of:
- **selectivity**: the vector moves the *target* contrast much more than an
  unrelated control contrast, relative to random vectors of matched norm;
- **sign consistency across held-out prompts**: the same direction of effect
  on prompts never used to build the vector;
- **behaviour**: the generated output actually changes, measured by an external
  checker (e.g. does the model emit more comment lines?).

### L2. A quantised model can hide a non-specificity failure.
The same L8 vector passed the span criterion on Q4_K_M (real 11.8 vs random max
10.9) and failed on bf16 (real 10.8 vs random max 12.6). Random-vector spans moved
by ~15% between artefacts while the real effect stayed flat. **Never validate a
steering claim on a quantised artefact and generalise it.**

### L5. A failed test is evidence, not a proof of impossibility. (2026-11-11)
"Steering does not work" was written in the README on the strength of one failed
test (C4), whose pass criterion had been changed twice during the same session
and which had not been audited. That is `[INF]` stated as `[OBS]`.

Rule: a negative result may be published only with (a) a stable, pre-registered
criterion, (b) a negative control that passes, and (c) a second independent
implementation or an external audit. Otherwise it is `PROVISIONAL`.

Corollary: "we cannot see inside an API model" is also too strong. Carlini et al.
2024 (arXiv:2403.06634) recovered a production model's embedding projection layer
from logits alone. The accurate statement is: *no hidden states are exposed;
partial parameter recovery from outputs is possible in restricted settings.*

### L4. Similarity is not function. (F1, 2026-11)
In the GPT-2 -> DistilGPT-2 circuit audit (arXiv:2610.37157), the IOI circuit was
destroyed (94.8% -> 35.2% recovery, sign reversal, 0% name-mover preservation)
while the standard similarity metrics looked healthy: **CKA 0.21-0.52 and
attention correlation up to 0.78 on heads with ZERO functional preservation**.
Greater-Than survived only by *remapping* L9H1 -> L4H1 (r = 0.945), not by
preserving the index.

**Consequence:** never use CKA, RSA, cosine, or attention correlation as evidence
that two models share a mechanism. The only admissible evidence is a functional
test: does the same intervention produce the same behavioural change?

### L3. Replacing an activation is not the same control as adding to it.
NC3-style "random replacement" destroys the signal and should be expected to
produce near-zero or negative recovery. Asserting |recovery| ~ 0 conflates
"destruction control" with "no-op control".

## Audit trail

The orchestrator audits a random sample of every agent's claims by re-running
the test independently. Discrepancies get logged in `LEDGER.md` under
`## Audit corrections`.
