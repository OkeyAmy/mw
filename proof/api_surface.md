# API surface probe — measured 2026-11-11

Purpose: determine what a real, standard Google AI Studio key actually exposes,
because the whole "backward loop" question depends on it.

Method: `generateContent` with `responseLogprobs: true, logprobs: 5`,
`maxOutputTokens: 1`, `temperature: 0`. Key stored in local `.env` (gitignored,
mode 600), never printed.

## Result

| model | reachable | logprobs |
|---|---|---|
| gemini-3-flash-preview | yes | **disabled** ("Logprobs is not enabled for this model") |
| gemma-4-31b-it | yes | **disabled** |
| gemma-4-26b-a4b-it | yes | **disabled** |
| gemini-flash-latest | yes | **disabled** |
| gemini-3.1-pro-preview | quota exceeded (free tier) | — |
| gemini-pro-latest | quota exceeded | — |
| gemini-2.5-flash / -pro / -lite | **404** "no longer available to new users" | — |
| gemini-2.0-flash-001, gemini-1.5-flash(-8b) | not in model list | — |

Model list size: 50 entries. Embedding models (`gemini-embedding-001`,
`gemini-embedding-2-preview`) are reachable but return vectors, not token
distributions.

**Corrections from F3 (2026-11-11):**
- `gemini-2.5-flash` is **reachable** (returns 400 "logprobs not enabled"), not
  404. Only `-pro` and `-lite` are retired.
- `gemini-embedding-2` returns **3072-d unit vectors** — a real, usable channel.
- `gemini-3-flash-preview` returns empty content unless `thinkingBudget: 0`.
- Sampling noise floor (temp 1.0, 6 samples): sigma(1-cos) = 0.068 on a
  high-entropy prompt, ~0.026 on a low-entropy one.
- Query cost for behaviour estimates: +/-10pp = 97 samples, +/-5pp = 385 samples,
  0.02 cosine shift = 45 samples. Token-level distributions are unreachable:
  169 h per token position just to visit each of 151,936 bins once.

## What this means

1. **On this access tier, the observable surface is text only.** No logits, no
   top-k, no per-token probabilities, no gradients.
2. Therefore every gradient-free method (GCG, SPSA, finite differences) collapses
   to **top-1 feedback**. That is still a signal, but it is the weakest possible
   one, and query cost explodes (see F2 for the arithmetic).
3. Decision-margin analysis — the cheapest useful black-box measurement — is
   **unavailable** on these models. A "certificate" built on logprobs cannot be
   sold against this provider.
4. This does **not** generalise to all providers. OpenAI exposes `top_logprobs`
   (up to 20) on its completion models; Anthropic does not expose logprobs at all.
   Logprob access is provider- and model-dependent, and it changes over time —
   the 2.5 family was retired mid-experiment.

## Consequence for the project

The open-source instrument's value is concentrated where weights are open.
For API models the tool can only:
- run behavioural tests (input → output, many trials),
- compare model versions,
- measure refusal/format/safety drift by behaviour alone.

It cannot measure margins, cannot attribute, cannot steer. That is now a
measured statement about a real key, not an assumption.

## Caveat

This is one key, one tier, one day. Re-run `tools/api_probe.py` to refresh.
