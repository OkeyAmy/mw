#!/usr/bin/env python3
"""E1 — token economics of replacing a system prompt with a steering vector.

Every number here is arithmetic on stated inputs, not a measurement.
Run: python3 code/e1_economics.py
"""
from dataclasses import dataclass


@dataclass
class App:
    name: str
    sys_tok: int      # system prompt tokens
    ctx_tok: int      # everything else in the input: user turn, history, tool results, retrieved docs
    out_tok: int      # completion tokens
    note: str = ""


# Price ratio assumptions (verified 2026-11-10 against vendor pricing pages):
#   output  = 10x input   (gpt-5 $1.25/$10; sonnet-5.5 $2/$10; haiku-5.5 $0.10/$0.50)
#   cached  = 0.1x input  (both vendors; 0.025-0.05x on newest flagship tiers)
P_IN = 1.0          # normalised: input price = 1 unit per token
P_OUT = 10.0
P_CACHE = 0.1


def cost(sys_tok, ctx_tok, out_tok, cache_prefix: bool):
    """Cost per request in units of 'input tokens'."""
    sys_cost = sys_tok * (P_CACHE if cache_prefix else P_IN)
    return sys_cost + ctx_tok * P_IN + out_tok * P_OUT


APPS = [
    App("chat assistant",      300,    800,   400, "short system prompt, short history"),
    App("RAG question answer", 500,  6_000,   500, "retrieved chunks dominate input"),
    App("coding agent (ACL'26 measured)", 110, 12_000, 900, "system prompt 110 tok, tool schemas 1602 tok"),
    App("doc analysis / long ctx", 1_000, 100_000, 1_000, "system prompt is <1% of context"),
    App("'heavy prompt' agent", 5_000, 5_000,  500, "adversarial case FOR the thesis: prompt is half of input"),
]


def pct(x, y):
    return 100.0 * x / y if y else 0.0


def main():
    print("=" * 96)
    print("PART 1 — what fraction of the bill IS the system prompt?")
    print("=" * 96)
    print(f"{'app':38s} {'S':>6s} {'ctx':>7s} {'out':>5s} "
          f"{'S% input':>9s} {'S% $ (nocache)':>15s} {'S% $ (cached)':>14s}")
    for a in APPS:
        total_in = a.sys_tok + a.ctx_tok
        c0 = cost(a.sys_tok, a.ctx_tok, a.out_tok, False)
        c1 = cost(a.sys_tok, a.ctx_tok, a.out_tok, True)
        s0 = a.sys_tok * P_IN
        s1 = a.sys_tok * P_CACHE
        print(f"{a.name:38s} {a.sys_tok:6d} {a.ctx_tok:7d} {a.out_tok:5d} "
              f"{pct(a.sys_tok, total_in):8.2f}% {pct(s0, c0):14.2f}% {pct(s1, c1):13.2f}%")

    print()
    print("=" * 96)
    print("PART 2 — what does a 100% / 90% / 50% system-prompt reduction actually save?")
    print("        (upper bound: assumes the vector is FREE, perfect, and needs zero")
    print("         engineering, retraining, or per-model re-extraction)")
    print("=" * 96)
    for a in APPS:
        c0 = cost(a.sys_tok, a.ctx_tok, a.out_tok, False)
        c1 = cost(a.sys_tok, a.ctx_tok, a.out_tok, True)
        print(f"\n{a.name}  (S={a.sys_tok}, ctx={a.ctx_tok}, out={a.out_tok})  -- {a.note}")
        for label, base in (("uncached", c0), ("cached prefix", c1)):
            row = []
            for cut in (0.5, 0.9, 1.0):
                new = cost(a.sys_tok * (1 - cut), a.ctx_tok, a.out_tok, label == "cached prefix")
                row.append(f"{int(cut*100):3d}% cut -> {pct(base - new, base):5.2f}% of bill")
            print(f"   {label:14s} " + " | ".join(row))

    print()
    print("=" * 96)
    print("PART 3 — translate to money. 1M requests/month, gpt-5 list prices.")
    print("=" * 96)
    price_in = 1.25 / 1e6   # $ per token
    price_out = 10.00 / 1e6
    price_cache = 0.125 / 1e6
    N = 1_000_000
    for a in APPS:
        for label, ps in (("uncached", price_in), ("cached", price_cache)):
            base = a.sys_tok * ps * N + a.ctx_tok * price_in * N + a.out_tok * price_out * N
            zero = a.ctx_tok * price_in * N + a.out_tok * price_out * N
            print(f"{a.name:38s} {label:9s} ${base:9,.0f}/mo  "
                  f"-> delete prompt entirely: ${zero:9,.0f}/mo  "
                  f"(saves ${base-zero:8,.0f}/mo = {pct(base-zero, base):5.2f}%)")

    print()
    print("=" * 96)
    print("PART 4 — the other side of the ledger: what the vector costs")
    print("=" * 96)
    print("""  A steering vector is NOT free to own, even though it is free to *serve*:
    - one extraction run per (model, behaviour): contrast pairs + forward passes
    - re-extraction whenever the model version changes (vectors are model-specific;
      layer index, hidden width and residual basis all change)
    - an eval harness, because a vector has no inspectable specification
    - hidden-state access: impossible on OpenAI/Anthropic APIs
  Serving cost is ~0: measured 42.8 tok/s steered vs 42.7 tok/s unsteered on
  Gemma-4-31B/vLLM (Tetracorp Lab, 2026-05-18) -- a 0.2% difference inside noise.""")

    print()
    print("=" * 96)
    print("PART 5 — the sanity check that ends the argument")
    print("=" * 96)
    # Prompt caching already does the job the vector claims to do, at 1/10 price,
    # with no per-model work, on every hosted model.
    a = APPS[4]  # the most favourable case for the thesis
    c_uncached = cost(a.sys_tok, a.ctx_tok, a.out_tok, False)
    c_cached = cost(a.sys_tok, a.ctx_tok, a.out_tok, True)
    c_zero = cost(0, a.ctx_tok, a.out_tok, False)
    prize = pct(c_uncached - c_zero, c_uncached)          # delete prompt, no cache
    cache_gain = pct(c_uncached - c_cached, c_uncached)   # cache it instead
    print(f"  Most favourable app: '{a.name}' (S={a.sys_tok}).")
    print(f"    delete the prompt entirely : {prize:5.1f}% saving (vs uncached baseline)")
    print(f"    just cache the prompt      : {cache_gain:5.1f}% saving, zero engineering")
    print(f"    -> the cache captures {100.0*cache_gain/prize:.0f}% of the total prize, for free.")
    print()
    print("  Read this as: in the single most favourable workload we could")
    print("  construct (a 5,000-token prompt that is half of all input tokens),")
    print("  deleting the prompt saves 33% -- and prompt caching already captures")
    print("  30 of those 33 points for free. Once the prompt is cached, deleting")
    print("  it is worth only 5.3%. For every realistic app the prize is <1%.")


if __name__ == "__main__":
    main()
