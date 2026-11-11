#!/usr/bin/env python3
"""F1: what does output-only distillation cost on THIS box?

All inputs are measured elsewhere in the project (do not re-derive here):
  - effective fp32 GEMM throughput, 54-67 GFLOPS   (A9, CONTEXT.md)
  - batched inference ~49 tok/s for the 1.5B model  (LEDGER, measured)
  - 4 physical cores, AVX2, no GPU

Training FLOPs use the standard approximation  C ~= 6 * N * D
(forward 2ND + backward 4ND), Kaplan et al. 2020 / Hoffmann et al. 2022.
It ignores attention-score FLOPs (O(D * L)) which are small at 2k context but
NOT negligible at 32k; it is therefore a LOWER bound on training cost.
"""
import math

N_STUDENT = 1.5e9
PEAK_GFLOPS = 60.0            # midpoint of A9's measured 54-67
EFF_FRACS = (0.15, 0.30, 0.50)  # train-step efficiency vs peak GEMM
TEACHER_TOK_S = 49.0          # measured batched decode, 1.5B

def hms(seconds):
    if seconds < 3600: return f"{seconds/60:,.1f} min"
    if seconds < 86400: return f"{seconds/3600:,.1f} h"
    return f"{seconds/86400:,.1f} days"

print("=" * 78)
print("PART 1 - student training cost (6ND), student N = 1.5e9 params")
print("=" * 78)
print(f"{'tokens':>12} {'train FLOPs':>12} | " + " | ".join(f"@ {int(f*100)}% eff" for f in EFF_FRACS))
for D in (1e5, 1e6, 1e7, 1e8):
    f = 6 * N_STUDENT * D
    cells = []
    for e in EFF_FRACS:
        cells.append(hms(f / (PEAK_GFLOPS * e * 1e9)))
    print(f"{D:>12,.0f} {f:>12.2e} | " + " | ".join(f"{c:>14}" for c in cells))

print()
print("Arithmetic shown for D = 1e6:")
f = 6 * N_STUDENT * 1e6
print(f"  6 * 1.5e9 * 1e6 = {f:.1e} FLOPs")
for e in EFF_FRACS:
    print(f"  at {int(e*100)}% of {PEAK_GFLOPS:.0f} GFLOPS = {PEAK_GFLOPS*e:.0f} GFLOPS"
          f" -> {f/(PEAK_GFLOPS*e*1e9):,.0f} s = {hms(f/(PEAK_GFLOPS*e*1e9))}")

print()
print("=" * 78)
print("PART 2 - teacher data generation")
print("=" * 78)
for D in (1e6, 1e7):
    print(f"  {D:>9,.0f} teacher tokens, local 1.5B @ {TEACHER_TOK_S:.0f} tok/s -> "
          f"{hms(D/TEACHER_TOK_S)}")
print("  (API teacher instead: at $0.10/MTok output, 1e7 tokens = $1.00. Money is not")
print("   the constraint. ELECTRICITY-TIME ON THIS BOX IS.)")

print()
print("=" * 78)
print("PART 3 - what the literature actually used")
print("=" * 78)
# Gudibande et al. 2023, "The False Promise of Imitating Proprietary LLMs"
print("  False Promise (2023) imitation data: 0.3M - 150M tokens, students 1.5B-13B")
for D, label in ((3e5, "0.3M (their minimum)"), (1.5e8, "150M (their maximum)")):
    f = 6 * N_STUDENT * D
    print(f"    {label:>22}: {hms(f/(PEAK_GFLOPS*0.3*1e9)):>12} at 30% eff, 1.5B student")

print()
print("=" * 78)
print("PART 4 - the only affordable version: behaviour-only, small data")
print("=" * 78)
print("  10k examples x 200 tokens = 2e6 tokens -> see PART 1 row for 1e6 x2")
print("  i.e. ~4-8 days at 30% efficiency, single process, no other work on the box.")

print()
print("=" * 78)
print("PART 5 - inference-side cost of the decisive experiment (GPT-2 124M)")
print("=" * 78)
N_T = 124e6
for n_prompts, L in ((1000, 20), (200, 20)):
    f = 2 * N_T * n_prompts * L
    print(f"  {n_prompts} prompts x {L} tokens, 2*N*D = {f:.2e} FLOPs -> "
          f"{hms(f/(PEAK_GFLOPS*0.3*1e9))} at 30% eff")
print("  Two models (teacher + student) -> double it. This is why the replication")
print("  experiment is cheap and the distillation experiment is not.")
