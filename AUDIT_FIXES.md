# AUDIT FIXES — G1 → code

Audit: `agents/G1_audit_mw.md` (29 defects: 2 blockers, 5 high, 11 medium, 9 low).
Status as of 2026-11-11. Every "fixed" line below is covered by a test or a
measured check.

| # | severity | defect | status | evidence |
|---|---|---|---|---|
| D1 | S1 | `load()` called `guard.require` without importing `guard` → every load raised `NameError` | **fixed** | `tests/test_smoke.py::test_load_succeeds` |
| D2 | S1 | `make check` used `ruff ... \|\| true` and ruff wasn't installed → lint could never fail | **fixed** | `make lint` is a real failing target; `ruff check` passes |
| D3 | S1 | no fast test called `load()`; two tests were arithmetic tautologies | **fixed** | `test_smoke.py` (tiny random Qwen2, 9 tests) |
| D4 | S2 | Granite scaling defaulted to 1.0 — the exact 8× bug | **fixed** | `requires_config_from_gguf`; `convert.py` refuses; `test_granite_scaling_is_declared_and_required` |
| D5 | S2 | Granite test was self-consistent only; all four factors wrong still passed | **fixed** | `test_granite_golden_logits` (argmax 589, max\|logit\| 19–21.5) |
| D6 | S2 | capture stored `resid_post(L)`, steering wrote `resid_pre(L)` | **fixed** | both sites captured; `site="pre"` shared; `test_diff_in_means_and_set_steer_share_a_site` |
| D7 | S2 | dtype chosen silently from transient free RAM | **partial** | `warnings.warn` + `dtype_actual` recorded in `trace()` provenance; auto-select still allowed |
| D8 | S2 | fp32 memory estimate dropped the ×2 for files ≥8 GB | **fixed** | header-derived params; Granite fp32 now 14.19 GB (was 7.09) |
| D9 | S2 | unknown size (repo id) bypassed the guard | **fixed** | resolves via `huggingface_hub.model_info`; otherwise refuses unless `MOD_WEI_ALLOW_OOM=1` |
| D10 | S3 | `self.cfg` and `model.config` were different objects | **fixed** | `self.cfg = self.model.config` |
| D11 | S3 | `logits_scaling` masked `None`/`0.0` | **fixed** | raises for granite if absent |
| D12 | S3 | `lm_head` masked `None` | **fixed** | raises `CapabilityError`; `test_missing_output_embedding_raises` |
| D13 | S3 | no layer validation | **fixed** | raises `ValueError`; `test_out_of_range_layer_raises` |
| D14 | S3 | hooks not idempotent (double steering) | **fixed** | `test_hooks_are_idempotent` |
| D15 | S3 | `trace()` claimed per-token but stored last only | **fixed** | `full=True` returns `[T,d]`; provenance records `positions` |
| D16 | S3 | no provenance / fidelity block | **fixed** | every trace carries `model: {path, arch, dtype, logits_scaling, site, positions}` |
| D17 | S3 | capability report always FULL; `HF_SUPPORTED` unused | **fixed** | unsupported arch → `PARTIAL` |
| D18 | S3 | header self-check only detects truncation | **partial** | added per-tensor finiteness + fp16 overflow check; per-tensor shape assertion still missing |
| D19 | S3 | `fetch_small_files` swallowed every error | **fixed** | failures collected and reported; required files enforced |
| D20 | S3 | silent bf16→fp16 downcast | **fixed** | range check before cast; default dtype follows source |
| D21 | S3 | tied config + output tensor would alias and drop a matrix | **fixed** | conversion refuses the mismatch |
| D22 | S3 | `mistral` entry unreachable (same gguf arch as llama) | **fixed** | entry removed; documented |
| D23 | S3 | gemma bias-drop contradicted "refuse unknown tensors" | **fixed** | gemma uses `_COMMON` |
| D24 | S3 | aliases silently mapped `qwen2_vl` → qwen2 | **fixed** | VL alias removed |
| D25 | S3 | `torch.set_num_threads` at import time | **fixed** | moved to `load(n_threads=...)` |
| D26 | S3 | test used a CWD-relative path | **fixed** | resolved from `__file__` |
| D27 | S3 | skip guard tested directory existence, not weights | **fixed** | checks `model.safetensors` |
| D28 | S4 | 8 scripts hardcoded `/home/okey/...` | **fixed** | 0 remaining; test scans all of `code/**/*.py` |
| D29 | S4 | C8 gate defaulted to the quantised artefact | **fixed** | defaults to `.hf/bf16-hf` |

## Additional defects found while fixing (not in G1)

| defect | how found | fix |
|---|---|---|
| fp16 `lm_head` rejected float32 captures (`RuntimeError: expected m1 and m2 to have the same dtype`) | running the fp16 Granite path | cast at the lens boundary |
| **bf16→fp16 silently produces NaN** (bf16 exponents exceed fp16's 65504) | Qwen slow test returned `nan` | `guard.unsafe_downcast`; adapter refuses; tests use fp32 for bf16 sources |
| `source_dtype` read the first tensor (a fp32 norm), so a 2.5B fp16 model looked like fp32 | memory estimate off by 2× | param-weighted dominant dtype |

## Verification

```
pytest -m "not slow"     32 passed
pytest -m slow -k granite 2 passed   (includes golden logits)
pytest -m slow -k qwen    2 passed
ruff check code/mw tests  All checks passed
grep -r "/home/" code    0 hits
```

## Still open

- **D7**: dtype still auto-selects (with a warning). For strict reproducibility,
  pass `dtype=` explicitly.
- **D18**: per-tensor shape assertions against the config are not implemented.
- **Slow suite**: Qwen and Granite are run in separate processes here to stay
  within memory. A CI machine with more RAM can run them together.
