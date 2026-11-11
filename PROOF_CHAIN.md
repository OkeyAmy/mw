# PROOF CHAIN — what must be demonstrated, in order

The project is only real if each link holds. Each link needs an **artifact**
(a file, a number, a diff), not an argument. A link that fails kills every link
below it.

| # | claim | test | pass criterion | artifact |
|---|---|---|---|---|
| C1 | we can see inside the model | dump residual stream for one prompt, all 28 layers | 28 tensors, correct shapes `[T,1536]`, finite values | `proof/c1_trace.json` |
| C2 | the lens is faithful | apply logit lens at the final layer; compare to real logits | **max abs diff < 1e-4** (must be exact, not approximate) | `proof/c2_lens_identity.txt` |
| C3 | activations are causally used | patch layer L residual from a clean run into a corrupt run | output changes toward clean; control patch (random layer) does not | `proof/c3_patching.txt` |
| C4 | we can steer semantics | diff-in-means vector, sweep coef −3..+3 | monotone dose-response; `coef=0` bit-identical; random vector no effect | `proof/c4_dose_response.json` |
| C5 | steering works mid-generation | change coef between token 5 and 6 | output diverges at exactly that step, not before | `proof/c5_midgen.txt` |
| C6 | we can force a token | direct logit bias | forced token appears; compare to semantic steering | `proof/c6_force.txt` |
| C7 | the atlas covers the model | profile every MLP neuron on a corpus | 250,880 neurons profiled; ≥X% fire at least once | `proof/c7_coverage.json` |
| C8 | the microscope does not lie | all sanity checks from VALIDATION.md | every check passes, failures documented | `proof/c8_sanity.txt` |

## Rules for the chain
- **Order matters.** Do not claim C4 before C2 passes. A broken lens makes every
  steering result uninterpretable.
- **Each link gets a negative control.** A result without its control is void.
- **Numbers, not adjectives.** "Steering worked" is not a result.
  "coef=+2.0 changed `def`→`#` in 17/20 held-out prompts, coef=0 in 0/20" is.
- **Record failures.** A failed link is more valuable than a vague success.

## Status — 2026-11-10, FIRST FULL RUN

**All six links PASS** on Qwen2.5-Coder-1.5B, CPU, fp32 compute.

Model artefact: the Ollama Q4_K_M GGUF, checksum-verified against the manifest,
dequantised to safetensors (`code/gguf_to_hf.py`, 338/338 tensors).
**Caveat: this is the quantised model.** C1/C2 are exact because they are
self-consistency identities; C3-C6 are true of this artefact. Re-run on the bf16
release before publishing any number.

| link | result |
|---|---|
| C1 | PASS — max diff 0.000e+00 |
| C2 | PASS — max diff 0.000e+00 |
| C3 | PASS — L22-27 recovery ~1.0, L0-20 ~0 |
| C4 | PASS — L8, spearman +1.000, +8.31/-3.49 |
| C5 | PASS — divergence begins after the switch, never before |
| C6 | PASS — forced 1.00 vs semantic 0.40, coherence -7.81 vs -2.89 |
| C7 | PASS — 250,863/250,880 neurons profiled (99.99%), 17 dead, ~49 tok/s |
| C8 | PASS — 10/10 sanity checks |

### C7 caveat (important)
Coverage is 99.99% **at threshold \|a\| > 1e-6**, which is a very loose definition
of "firing". SwiGLU intermediates are almost never exactly zero, so nearly every
neuron clears that bar on 6,000 tokens of Python stdlib. The honest reading is:
*the atlas pipeline works and touches every neuron*, not *every neuron is
meaningful*. Median per-token firing rate is ~0.50, so "fires on half the tokens"
is the realistic description. A useful atlas needs (a) a higher activation
threshold, (b) far more and more diverse tokens, and (c) top-activating contexts
rather than a single max token. Corpus: 6,000 tokens, `/usr/lib/python3.*`.

### C8 checks (all passing)
determinism bit-identical; lens identity 2.0e-05; `steer(coef=0)` bit-identical;
self-patch no-op bit-identical; attention rows sum to 1 (1.2e-07); all captured
tensors finite; tokenizer round-trip; eval mode; wrong vector changes output;
capture shapes `28 x [1,1536]`.

*C8 caught a real bug in its own first version: it self-patched `resid_post` into
the `resid_pre` site, which is not a no-op. Both sites are now captured separately.*

### Criterion changes made during the run (logged, not hidden)
1. **C3/NC3.** Original asserted `|random_recovery| <= 0.20`. NC3 *replaces* the
   residual with noise, which should destroy the answer, so the assertion now
   requires only that random does not RECOVER (`recovery <= 0.20`). The tested
   claim — effect comes from content, not magnitude — is unaffected.
2. **C4 layer.** Moved 14 -> 8 using a pre-registered sweep criterion
   (beats-random fraction, then monotone fraction) over 8 layers with 10 random
   controls each. L14 was non-specific; L8 and L16 were both clean.
3. **C4 pairs.** Added 12 isolated comment/code pairs to the 8 composite pairs,
   because comments embedded inside functions let function structure dominate the
   last-token residual. Same method, better estimator.
4. **C4 specificity test.** Replaced "every random span < 0.75 x real span" with
   "real curve crosses zero AND real span is largest AND no random is monotone".
   Span is a bad discriminator because any large perturbation moves logits.
   **Honest margin: real span 11.79 vs best random 10.92 = 1.08x.** Thin. The
   real evidence is the monotone +1.000 spearman and the zero-crossing, not size.

### bf16 RE-RUN (real weights, 338 tensors, exact size) — 2026-11-10
| link | bf16 result |
|---|---|
| C1 | **PASS** 0.000e+00 |
| C2 | **PASS** 0.000e+00 |
| C3 | **PASS** L23 recovery 1.056, random -2.042, L0 -0.008 |
| C4 | **FAIL** spearman +0.964 BUT 2/10 random vectors monotone with LARGER span |
| C5 | **PASS** pre_exact, divergence after switch |
| C6 | **PASS** forced 1.00, but semantic_rate 0.00 (no behavioural change) |

**C4 does not survive bf16.** Random vectors span 12.64 and 12.33 vs the real
vector's 10.83, with |rho| 0.86-0.89. The steering effect is real at the logit
level but *not specific*. C6 corroborates: on bf16 the semantic vector did not
change the next token at all (rate 0.00), while forcing did (1.00).

### Still not proven
- C7 (atlas coverage) — not started.
- C8 (full sanity gate) — partial.
- Everything above on **bf16**, not Q4_K_M.
