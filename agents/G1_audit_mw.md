# G1 — adversarial audit of the `mw` package

**Role.** Independent audit of `code/mw/{__init__,registry,adapter,convert}.py` and `tests/`,
immediately before an MIT open-source release. I assumed the package was wrong and looked for
the defects the orchestrator's checks structurally cannot see.

**Headline: 2 blockers, 5 high, 11 medium, 9 low.** The package as it stands on disk
(2026-11-11 12:47) **cannot load any model** — `load()` raises `NameError` — and the two checks
that would catch that (`make check`, slow tests) both report success. The numerical core is
genuinely exact: I reproduced lens/residual identity at **0.000e+00 on both models, every layer,
every token position**. Everything that is wrong is wrong at the *edges*: memory safety, the
scaling defaults, the hook site, and the release gates.

**Timing note.** Checks A–E and G were measured on the real weights (bf16 Qwen 1.5B fp32,
Granite 3.3 2B fp32) *before* the memory constraint arrived. After the laptop freeze I switched
to meta-device, header-only and tiny-random-model methods, as instructed. Every number below is
labelled with how it was obtained. Nothing is extrapolated from a run I did not do.

---

## 1. Defects

Severity: **S1** = blocks release / wrong results; **S2** = will produce wrong science in a
plausible user's hands; **S3** = correctness risk or broken gate; **S4** = portability/hygiene.

### S1 — the package cannot load a model

| # | file:line | what is wrong | evidence | fix |
|---|---|---|---|---|
| **D1** | `code/mw/adapter.py:72` | `guard.require(...)` is called but **`guard` is never imported**. `import time` is imported (and unused) instead. Every `load()` raises `NameError` before touching a weight. | `PYTHONPATH=code .venv/bin/python -c "import mw.adapter as A; A.ModelAdapter('.hf/bf16-hf').load()"` → `load() -> NameError: name 'guard' is not defined`. Also `pytest tests -m slow -x` → `FAILED tests/test_adapter.py::test_qwen_lens_identity_exact - NameError: name 'guard' is not defined` at `adapter.py:72`. | add `from . import guard` (and drop `import time`). |
| **D2** | `Makefile:23-24` | `make check` runs `ruff check code tests \|\| true`, and **ruff is not installed**, so the lint step can never fail and in fact never runs. It would have caught D1 (`F821 undefined name 'guard'`) and `F401 'time' imported but unused`. | `make check` → `17 passed, 3 deselected` then `No module named ruff`, `MAKE_CHECK_RC=0`. | install ruff in `[dev]` (it *is* listed — the venv predates it) and remove `\|\| true`; make lint a separate failing target. |
| **D3** | `tests/test_adapter.py` | **No fast test calls `load()`.** The `-m "not slow"` gate is model-free by design, so the entire load path is untested in CI while D1 was live. Two "structural" tests are tautologies about arithmetic: `40*9+2 == 362` (`test_convert.py:33`) and `28*12+2 == 338` (`:40`) — they assert Python arithmetic, not that the converter wrote those tensors. | `make check` → 17 passed while `load()` raises `NameError`. | add a fast test that builds a tiny random `Qwen2ForCausalLM` from config and exercises `load`-adjacent paths (`_install_hooks`, `trace`, `generate`) on it — <20 MB, no download. |

### S2 — Granite scaling: the default is the bug it claims to prevent, and the test cannot see it

| # | file:line | what is wrong | evidence | fix |
|---|---|---|---|---|
| **D4** | `registry.py:75-80` | `defaults={..., "logits_scaling": 1.0}` etc. If a Granite GGUF is missing any `granite.*_scale` key, conversion silently writes **1.0** — which *is* the "numerically wrong (logits off by 8x)" failure the note above it warns about. It should refuse, or require an explicit `--allow-scaling-default`. `tests/test_registry.py:39-42` asserts these defaults exist, enshrining the bug. | All four factors are load-bearing, each set to 1.0 on the live model: `embedding_multiplier` max\|Δlogits\| **1.018e+01**, `residual_multiplier` **9.644e+00**, `attention_multiplier` **1.010e+01** (argmax flips 203→other), `logits_scaling` **1.223e+02**. | refuse on missing metadata for any arch with `config_from_gguf`; drop the `1.0` defaults for granite (keep them only for arch-with-no-scaling). |
| **D5** | `tests/test_adapter.py:62` | **The Granite acceptance test is blind to exactly that failure.** With *all four* scaling factors wrong the lens identity still reads 4.578e-05 (< 1e-4) → test passes. Self-consistency cannot detect a scaling error because both sides of the identity use the same wrong model. | `all four = 1.0 : max\|dlogits\|=8.991e+01  argmax 203 -> 30  lens-identity=4.578e-05`; `>>> SELF-CONSISTENCY IS BLIND`. | the Granite test must assert against a **golden logit vector** produced by `transformers` from the same checkpoint (or at minimum assert `next-token argmax == 203` and `max\|logits\|` within a band), not only self-consistency. |

### S2 — hook-site mismatch: capture is `resid_post`, steering is `resid_pre`

| # | file:line | what is wrong | evidence | fix |
|---|---|---|---|---|
| **D6** | `adapter.py:124` (capture) vs `:139` (steer) vs `:198` (`diff_in_means`) | `_mk_capture` stores the **output** of layer *L* (`resid_post(L)`); `_mk_steer` writes the **input** of layer *L* (`resid_pre(L)`). `diff_in_means(layer=L)` therefore returns a vector from *after* layer L, and `set_steer(layer=L)` applies it *before* layer L — a one-block basis mismatch, silently. This is precisely the bug `PROOF_CHAIN.md` C8 documents having caught in `sanity.py` ("it self-patched `resid_post` into the `resid_pre` site, which is not a no-op"). `mw` reintroduces it and offers no `resid_pre` capture at all, so the C8 self-patch no-op test cannot even be expressed through the public API. | `post[L-1] == pre[L]` → `max\|diff\| = 0.000e+00` at L=1, 5, 14, 27 (post is the site of the *next* layer). `L14: cos(pre,post)=0.9451 \|pre\|=32.92 \|post\|=36.09`. C8-style patch on the live model: `self-patch pre[L] -> pre site: bit-identical True`; `self-patch post[L] -> pre site: bit-identical False \| max\|dlogits\| = 1.184e+00`. | capture **both** sites (`self._pre[i]`, `self._post[i]`) and make `diff_in_means(layer, site="pre"\|"post")` + `set_steer(..., site=...)` share one site name; default both to `pre` so `diff_in_means(L)` → `set_steer(L)` is a no-op-composable pair. Document "layer L" as one site only. |

### S2 — dtype/scaling decided from transient free memory, silently

| # | file:line | what is wrong | evidence | fix |
|---|---|---|---|---|
| **D7** | `guard.py:77-84`, `adapter.py:68-84` | `load(dtype=None)` picks fp32 or fp16 from **current `MemAvailable`**, prints nothing, and records nothing. The same script therefore produces different numerics on a busier machine, with no trace of the change. For a measurement instrument that is a reproducibility hazard; it also contradicts CONTEXT/A1 ("fp32 for fidelity; bf16 buys nothing" and "cast to fp32 in RAM"). | `plan('.hf/granite-hf')` at 23.87 GB free → `{"ok": true, "dtype": "float32", "need_gb": 14.19}`; the same call on a 16 GB-available box returns `float16` (arithmetic: `avail > need_fp32 + 4`). No `stderr` line in either case. | require an explicit dtype at the API boundary, or emit `warnings.warn` + stamp `adapter.dtype_actual` into `trace()`. |
| **D8** | `guard.py:77` | `need_fp32 = disk_gb * (2.0 if disk_gb < 8 else 1.0) * 1.4`. The `2.0` (bf16/fp16-on-disk → fp32-in-RAM) is **dropped for every model ≥ 8 GB on disk** — i.e. the guard under-estimates fp32 memory exactly where it matters most. | granite fp32 = 2,533,539,840 params × 4 B = **10.13 GB** of weights, plus a 5.07 GB checkpoint transient → ~15.2 GB peak; the formula would say `16*1.4 = 22.4` for a 16 GB bf16 model whose real fp32 need is **32 GB**. | always `disk_gb * 2.0` for a 2-byte-per-param file; detect bytes/param from the header (`dtype` map) instead of guessing from file size. |
| **D9** | `guard.py:53-65`, `:72-74` | Size unknown ⇒ `{"ok": true, ...}`. Any HF **repo id** (i.e. the README quickstart `ModelAdapter("Qwen/Qwen2.5-Coder-1.5B").load()`) and any typo'd path bypasses the guard entirely. | `guard.plan('Qwen/Qwen2.5-Coder-1.5B')` → `{"ok": true, "dtype": "float16", "reason": "size unknown"}`; `guard.plan('/nonexistent')` → identical. | resolve repo ids via `huggingface_hub` metadata (or `model_info(..., files_metadata=True)`) before load; unknown ⇒ refuse unless `MOD_WEI_ALLOW_OOM=1`. |

### S3 — correctness risks

| # | file:line | what is wrong | evidence | fix |
|---|---|---|---|---|
| **D10** | `adapter.py:78` + `:111` | `self.cfg` and `self.model.config` are **two different objects**, and the lens reads `self.cfg` while the model reads `self.model.config`. They agree today only because both are parsed from the same file. They already differ in `dtype` (`cfg.dtype=float16` vs `model.config.dtype=float32`), proving the divergence is real. | `adapter.cfg is model.config → False`; ids differ; `cfg.logits_scaling=1.0` while `model.config.logits_scaling=8.0` → model logits bit-identical (`True`) but `adapter lens error vs model logits: 1.223e+02`. | use `self.model.config` exclusively (`self.cfg = self.model.config` after load), or delete the property and read `self.model.config.logits_scaling`. |
| **D11** | `adapter.py:111` | `float(getattr(self.cfg, "logits_scaling", 1.0) or 1.0)` masks a `0.0` (would be a div-by-zero the model also has) and any `None`; a model with a *different* scaling attribute name silently gets 1.0. | static; combined with D10 this is the whole 8× risk surface. | `getattr(..., None)`; if absent, assert the arch has no scaling (consult `registry`), else raise. |
| **D12** | `adapter.py:106` | `self.model.get_output_embeddings() or self.model.lm_head` — the `or` masks `None`, so a model with no output embedding fails later with a confusing `AttributeError`, and a model that returns `None` but has an `.lm_head` is silently accepted (nothing checks it is the tied matrix). | `no-head model -> AttributeError: 'NoHead' object has no attribute 'lm_head'`; `get_output_embeddings()==None but .lm_head exists -> silently used: True`. | `oe = get_output_embeddings()`; `if oe is None: raise CapabilityError(...)`; assert `oe.weight.data_ptr() == embed.weight.data_ptr()` when `cfg.tie_word_embeddings`. |
| **D13** | `adapter.py:185-187` | `set_steer` never validates `layer`. Out-of-range layers are a silent no-op — the most likely user error when hand-porting a layer index from another model. | tiny-model probe: `layer=99 (>= n_layers=3) same output: True  <- no error, no warning`. | `if layer is not None and not 0 <= layer < len(self.layers): raise ValueError`. |
| **D14** | `adapter.py:114-117` | `_install_hooks` is not idempotent. Calling `load()` twice (or `_install_hooks` twice) doubles every hook; PyTorch chains pre-hooks, so steering is applied **twice**. | `forward hooks on layer0 after 2x _install_hooks: 2`, `pre hooks: 2`; pre-hook chaining means the second hook receives the output of the first. | store handles and `remove()` them first, or guard with `if self._hook_handles: return`. |
| **D15** | `adapter.py:124`, `:167` | `trace()` is documented and README'd as "every layer's residual stream, **per token**", but `_mk_capture` stores `hs[:, -1, :]` only. `trace()["layers"]` is a single token's lens. There is no per-token residual anywhere in the public API. | `adapter _cap stored shape per layer: (1, 1536) -> only last position`, for both models (d=1536 and 2048). | capture `[:, -1:, :]` vs full `[T, d]` behind a flag; make `trace()` return `{layer: [T, d]}` or rename to `trace_last()`. |
| **D16** | `adapter.py:163-177` | `trace()` emits **no provenance/fidelity block** and no `step`/`token_id`/`token_str`/`attn_entropy`/`mlp_out_norm`. A1's required schema change (`fidelity ∈ {exact, dequantized_lossy, reimplemented}`) and CONTEXT's canonical trace schema are both unimplemented, and nothing stops a `.hf/q4km-hf` trace being mixed with a bf16 one. | schema of the returned dict, read from source + a live `trace()` call (`keys = prompt, n_tokens, layers, resid_norms, next_topk`). | add the `model` block from A1 §SCHEMA CHANGE before release; a trace without fidelity is unpublishable under this project's own rules. |
| **D17** | `adapter.py:233-240` | `capabilities()` always returns `visibility=VIS_FULL` and only reads `d_model/n_layers/vocab` from `self.cfg`. `VIS_PARTIAL`/`VIS_BLACKBOX` are defined, exported and **never assigned**; `HF_SUPPORTED` is exported and **never consulted**, so an unsupported architecture loads and reports FULL. | `grep -rn "HF_SUPPORTED\|VIS_PARTIAL\|VIS_BLACKBOX" code tests` → definitions and `__all__` only, no use sites. | either implement the gate (`if arch not in HF_SUPPORTED: visibility = VIS_PARTIAL`) or delete the constants; a capability report that cannot report a limitation is marketing. |
| **D18** | `convert.py:122-131` | The "header self-check" only compares `file_size == 8 + header_len + max_offset`. It cannot detect a transposed tensor, a swapped name map, or NaN/Inf; it only detects a truncated write. It is presented as `header self-check: OK`, which reads as a correctness proof. | read of `convert.py:126-137`; the check passes on any file whose offsets are self-consistent. | add (a) per-tensor `shape == expected_from_config`, (b) `np.isfinite(t).all()`, (c) a `min/max` per tensor into a sidecar. |
| **D19** | `convert.py:46-54` | `fetch_small_files` swallows **every** exception (`except Exception: pass`). A repo without `tokenizer.json` (or a transient 500) yields a model dir that loads no tokenizer; the only hard failure is a missing `config.json`. On an open-source release this looks like "conversion succeeded, model broken". | source read; the granite dir does have all files, so this is latent. | collect failures and `sys.exit` if `config.json`/`tokenizer.json`/`tokenizer_config.json` are missing. |
| **D20** | `convert.py:62`, `:96` | Default `--dtype float16` downconverts a **bf16** source to fp16 with no warning and no loss check, while forcing 1-D tensors to fp32. The result is a mixed-precision file (`F16`×281, `F32`×81) whose `config.json` claims `"torch_dtype": "float16"`. | header scan of the shipped artefact: `dtypes in file: {'F32': 81, 'F16': 281}`; 81 = 40×2+1 norms, exactly the `ndim==1` branch. | default to the source dtype (read `general.file_type`/tensor dtypes), and assert `max\|w\| < 65504` before `astype(np.float16)`. (For this model it is safe: measured `max \|weight\| = 56.5`, `non-finite tensors: none`, 1159× headroom.) |
| **D21** | `convert.py:32-43` vs config | `map_tensor` will emit `lm_head.weight` for a GGUF `output.weight`, but nothing checks `cfg.tie_word_embeddings` against whether an output tensor was written. With a tied config the two keys **alias the same `nn.Parameter`**, so `load_state_dict` silently applies last-write-wins and one of the two matrices disappears. | tiny-model probe: `loaded lm_head equals embed_tokens (distinct tensor silently dropped): True`; `lm_head still equals the 0.123 tensor: True` → the checkpoint's `embed_tokens` was replaced, silently. | after building `tensors`, assert `("lm_head.weight" in tensors) != cfg["tie_word_embeddings"]` and refuse the mismatch. |
| **D22** | `registry.py:61-65` | `SPECS["mistral"]` has `gguf_arch="llama"`, the same key as `SPECS["llama"]`. `resolve()` can therefore never return the Mistral spec from a real GGUF arch string, and the two entries are indistinguishable by their own advertised key. | `resolve('llama').hf_arch → LlamaForCausalLM`, `resolve('mistral').hf_arch → MistralForCausalLM`; a Mistral GGUF (arch `llama`) resolves to the **llama** spec. | key the dict by `gguf_arch` and put the disambiguation in the HF config; or drop the mistral entry and rely on `--hf-repo` config (which is what actually decides the class). |
| **D23** | `registry.py:89` + `:17` | Two contradictory policies: the gemma map *drops* bias entries, but `convert.py:101` treats any unmapped tensor as fatal (`UNKNOWN TENSORS (refusing to drop weights)`). So gemma "bias-dropping" is really "bias-refusal" — safe, but the note in `registry.py:92-94` and the docstring "absent keys are skipped" (llama) describe behaviour `convert` does not implement. | `gemma map_tensor('blk.0.attn_q.bias') -> None` → `sys.exit("UNKNOWN TENSORS ...")`. | decide one policy: rename the field to `tensor_map` + `skip: set[str]` and implement `skip` explicitly, or delete the gemma special case. |
| **D24** | `registry.py:104-107` | The docstring says "Anything not in this table is reported as UNSUPPORTED, **never silently mapped**", but `ALIASES` silently maps `qwen2_vl → qwen2`, `gemma3 → gemma`, `llama3 → llama`. A Qwen2-VL GGUF is thus *accepted as qwen2* and only fails later on its vision tensors. | `ALIASES` read + `resolve()` behaviour. | distinguish "rename" from "same architecture": keep only names that are literally the same arch string in llama.cpp; refuse VL models explicitly. |
| **D25** | `adapter.py:25` | `torch.set_num_threads(4)` at **import time**. Importing a library must not reconfigure the host process's thread pool; an open-source consumer embedding `mw` will silently lose throughput elsewhere. | `import mw.adapter` → `torch.get_num_threads()` becomes 4. | move into `load()` (or a `configure()` function) and make it an argument. |
| **D26** | `tests/test_registry.py:61` | `open("code/mw/registry.py")` is **CWD-relative**. The suite only works when pytest is run from the repo root, while `test_adapter.py` correctly uses `ROOT`. Also `pytest tests` vs `pytest` differ because `testpaths` is ignored when a path argument is given. | `cd /tmp/g1cwd && pytest /home/okey/.../tests -m "not slow"` → `FAILED test_no_absolute_paths_in_registry - FileNotFoundError: 'code/mw/registry.py'` (21 passed, 1 failed). | `Path(__file__).resolve().parent.parent / "code/mw/registry.py"`. |
| **D27** | `tests/test_adapter.py:27,45,66` | The skip guard is `not QWEN.exists()` — **directory** existence, not weight existence. Any partial download, or the empty `.hf/bf16-hf` the README tells users to create, turns skips into failures. | copied the test into a root with empty `.hf/bf16-hf` and `.hf/granite-hf`: `3 failed, 2 deselected` — no skips. | skip on `(QWEN / "model.safetensors").exists()` (or `any(QWEN.glob("*.safetensors"))`). |
| **D28** | repo-wide (not the package) | 8 shipped scripts hardcode `/home/okey/Desktop/Projects/mod_wei`, including the C8 gate. The only guard test checks `code/mw/*.py` only, so this passes CI. | `grep -rn "/home/okey"` → `code/sanity.py:21,28`, `atlas.py:30,58`, `steering_test.py:38,103`, `layer_sweep.py:55,164`, `f2_gradient_free.py:35,36`, `f2_top1_only.py:23,24`, `a1_formats/README.md:8,27,28`. | resolve from `__file__`/env, and extend the test to every `code/**/*.py`. |
| **D29** | `code/sanity.py:28` | The C8 release gate **defaults to `.hf/q4km-hf`**, the dequantised artefact that VALIDATION **L2** says must never be used to validate a steering claim. `proof_chain.py` correctly defaults to `Qwen/Qwen2.5-Coder-1.5B`. | source read: `ap.add_argument("--model", default=".../.hf/q4km-hf")`. | default C8 to the bf16 artefact; require `--allow-quantised` for the other. |

### S4 — small

| # | file:line | what is wrong | evidence |
|---|---|---|---|
| D30 | `adapter.py:219-220` | temperature path: `torch.multinomial((logits / temperature).softmax(-1), 1)` — divides before softmax (fine), but the RNG is unseeded and there is no `top_p`; and `generate` returns no seed. Not reproducible even in principle. | tiny-model run returned `[23, 6, 45, 4]`, different on each call. |
| D31 | `adapter.py:229` | `torch.tensor([[nxt]])` is CPU-only and `torch.cat` every step is O(n²) string→tensor work; also no `attention_mask`, so padding (if a batch were ever added) would be silently attended to. | source read. |
| D32 | `adapter.py:165-168` | `trace()` sets `self._capturing = True` with no `try/finally`. Any exception leaves capture armed, so the next unrelated forward silently overwrites `_cap`. | source read (`diff_in_means` has the same pattern at `:192-199`). |
| D33 | `adapter.py:213` | `logits_to_keep=1` is **not** bit-identical to the full forward (different GEMM shape): Qwen `max\|diff\| = 1.621e-05`, Granite `6.676e-06` (argmax agreed). Greedy trajectories are therefore shape-dependent at 1e-5. Worth stating in the docs; the 1e-4 lens criterion already absorbs it. |
| D34 | `adapter.py:18` | `import time` unused (ruff `F401`). |
| D35 | `pyproject.toml:11` | `license = { file = "LICENSE" }` is the legacy table form (setuptools warns; PEP 639 wants a string). `README` claims MIT, `LICENSE` is MIT — consistent. |
| D36 | `pyproject.toml` | `requires-python = ">=3.10,<3.14"` but `mw/adapter.py` uses `torch.dtype \| None` in a signature with `from __future__ import annotations` (fine) — however `str \| None` in `guard.py` is fine too. No issue, listed only because CI never tests 3.10. |
| D37 | `mw/__init__.py` | `guard` and `convert` are not exported; `mw.convert` is only reachable as a submodule. Minor, but the README's "add an ArchSpec" story implies `mw.registry` is the public façade. |
| D38 | `registry.py:110-114` | `HF_SUPPORTED` contains `PhiForCausalLM`, `GPT2LMHeadModel`, `Gemma2/3ForCausalLM` while `SPECS` contains only `gemma` — the two support matrices disagree and neither is enforced (D17). |

---

## 2. What passed — with measured numbers

All numbers below are from this audit, not copied from the orchestrator. Methods are marked.

### A. Key space (meta device + safetensors header; zero weight RAM — reproduced independently)

```
qwen-bf16 : meta state_dict 339 | header 338 | MISSING ['lm_head.weight'] | UNEXPECTED []
            layer0 tensors 12 -> 12*28+2 = 338 | shape mismatches: none | dtypes {'BF16': 338}
granite   : meta state_dict 363 | header 362 | MISSING ['lm_head.weight'] | UNEXPECTED []
            layer0 tensors  9 ->  9*40+2 = 362 | shape mismatches: none | dtypes {'F32': 81, 'F16': 281}
```
The only missing key is `lm_head.weight`, which is **correct** for `tie_word_embeddings=True`. No
UNEXPECTED key, no MISMATCH. **PASS** (and it independently confirms the orchestrator's 338/362).

### B. Numerical identity — every layer, every position, both models (real weights, fp32, measured pre-freeze)

```
qwen-bf16  resid_post identity layers 0..26, all 12 positions : max|diff| = 0.000e+00
           adapter _cap vs true resid_post(L), all 28 layers   : max|diff| = 0.000e+00
           lens identity (true resid_post), all 12 positions   : max|diff| = 0.000e+00
granite    resid_post identity layers 0..38, all 13 positions : max|diff| = 0.000e+00
           adapter _cap vs true resid_post(L), all 40 layers   : max|diff| = 0.000e+00
           lens identity (true resid_post), all 13 positions   : max|diff| = 0.000e+00
```
Identity holds at **every** position, not only the last, and at mid-layers (adapter capture vs
manual: `0.00e+00` at L0, L14/L20, L26/L38). Repeat forwards are bit-identical
(`repeat forward bitwise identical: True`). My number is **0.000e+00** where the orchestrator
reported `9.537e-06`; the difference is GEMM shape — `lm_head` applied to a `[1,1,D]` vector
gives 6.7e-06 vs the model's `[1,T,D]` pass, while applying it to the full `[1,T,D]` residual
gives exactly 0. Both are inside the 1e-4 gate.

**Trap found (not a `mw` bug, but a live hazard for every consumer):** in transformers **5.22.2**
`out.hidden_states[-1]` is **overwritten with the post-final-norm `last_hidden_state`**
(`transformers/utils/output_capturing.py:234`, `tie_last_hidden_states=True`). Verified:
`hs[-1]` vs the true last-layer output differs by `2.039e+02` (Qwen) / `2.289e+03` (Granite), and
a consumer using `hs[-1]` for the lens gets `1.932e+01` / `1.712e+01` error. `code/proof_chain.py:505-511`
handles this correctly; `mw`'s docstrings do not mention it, and D15/A3's "`hidden_states[L] ==
resid_pre(L)`" rule is **false at L = n_layers**.

### C. Which scaling fields are load-bearing (Granite, live weights, mutate the object the model reads)

| factor → 1.0 | max\|Δlogits\| | relative RMS | argmax moves | inference |
|---|---|---|---|---|
| `embedding_multiplier` (12.0) | 1.018e+01 | 5.46e-01 | no | **load-bearing** |
| `residual_multiplier` (0.22) | 9.644e+00 | 6.89e-01 | no | **load-bearing** |
| `attention_multiplier` (0.015625) | 1.010e+01 | 5.69e-01 | **yes** | **load-bearing, changes the answer** |
| `logits_scaling` (8.0) | 1.223e+02 | 7.00e+00 | no | **load-bearing (pure output scale)** |
| all four | 8.991e+01 | — | 203→30 | catastrophic |

So all four are load-bearing; `attention_multiplier` is the only one that flips the argmax on
this prompt, and `logits_scaling` is the only one that the *lens identity test cannot see* (D5).

### D. Hook sites (see D6 for the defect)

`post[L-1] == pre[L]` at `0.000e+00`; the pre→pre self-patch is bit-identical; the post→pre
self-patch changes logits by `1.184e+00`. The adapter's *capture* hook is faithful (`_cap[i] ==
resid_post(i)` exactly) — the problem is that steering uses the other site.

### E. Tied embeddings

```
qwen  : get_output_embeddings() -> Linear (151936,1536); lm_head.weight IS embed_tokens.weight: True
        data_ptr equal: True; bitwise equal: True; _tied_weights_keys {'lm_head.weight': ...}
granite: get_output_embeddings() -> Linear (49159,2048);  IS embed_tokens.weight: True; data_ptr equal: True
```
The weight **is** genuinely shared (object identity + `data_ptr`), so `lm_head` resolution is
safe *for these two models*. The `or` fallback is the defect (D12), not the tie itself.

### F. `generate()` / mid-gen steering (tiny random Qwen2 from config, 3 layers, <20 MB)

```
F1  logits_to_keep=1 + DynamicCache           : OK, 6 tokens
    pre-hook calls for 4 decode steps         : 12 == NL(3) * 4 forwards
F2  schedule fired for steps [0..7]; switch at 3
    base : [8,8,8,8,8,8,8,8]
    steer: [8,8,8,8,54,54,54,54]
    tokens 0..2 identical (no divergence before the switch): True
    divergence after the switch: True
F3  coef=0 injected at step 2 -> identical token ids: True
F4  force_token=11 -> [11,11,11,11]; temperature=0.7 -> OK
F7  two _install_hooks -> 2 forward + 2 pre hooks (double steering)
```
So the **C5 shape is implemented correctly**: the hook fires on every decode step, the
intervention takes effect on the next token only, and `coef=0` is a true no-op. Verified on both
real architectures earlier for the `logits_to_keep=1` kwarg (`(1,1,V)` logits, argmax equal to
the full-forward argmax). `logits_to_keep=1` is supported by **all 9** classes in `HF_SUPPORTED`
in transformers 5.22.2 (GPT2 included — my prior suspicion that GPT2 would `TypeError` is
**refuted**: `GPT2LMHeadModel(logits_to_keep=1) -> accepted`).

### G. Registry / convert

```
unsupported arch -> rc=1  "UNSUPPORTED ARCH 'some_future_arch'. Add it to mw/registry.py. Refusing to guess."
phi3             -> rc=1  "ARCH 'phi3' uses fused tensors (...). Conversion not implemented; refusing..."
```
Both refusal paths work and exit non-zero — this is the behaviour the docstring promises and it
holds. `mixtral`/unknown arch are likewise refused (`resolve()` returns `None`). `map_tensor`
is correct for the two real models: all 338/362 keys are produced by the map, and **no shape
mismatch** against the meta model means the A1 "never transpose" rule is honoured
(`down_proj` `[2048,8192]` vs `gate_proj` `[8192,2048]` would have caught it).

### H. Converted artefact integrity (streamed, one tensor at a time)

```
tensors scanned: 362 | non-finite tensors: none
max |weight| = 56.5 in model.norm.weight (fp16 max 65504) | headroom 1159x
```
The fp16 conversion of Granite did not overflow. The claim "362 tensors, 5.07 GB, header
self-check OK" is reproducible; the *header self-check* is weak (D18) but the file is fine.

### I. Tests

`make fast` → `17 passed, 3 deselected in 18.24s`. `pytest tests -m slow -x` → **`1 failed, 17
deselected`** (NameError, D1) — i.e. the orchestrator's "granite slow test passes" was true of the
code *before* the 12:47 guard patch, not of the code that would ship. All three slow tests fail
the same way (see D27 for the run in a synthetic root: `3 failed, 2 deselected`).

---

## 3. Cross-check against prior findings

| prior finding | status in `mw` |
|---|---|
| **A1 trap 5** "assert every shape against the HF config before loading; a transpose bug is invisible otherwise" | **VIOLATED.** `convert.py` asserts nothing; the "self-check" is a file-size comparison (D18). Only the meta-model comparison *I* ran proves the shipped artefact is right. |
| **A1 trap 6** "do not add `lm_head` to this checkpoint; it is tied" | **VIOLATED in the general case.** `map_tensor` maps `output.weight → lm_head.weight` with no tie check; with a tied config the load silently replaces `embed_tokens` (D21, measured). |
| **A1 "never transpose dequantised tensors"** | **HONOURED**, and the comment at `convert.py:95` says so explicitly. Verified: zero shape mismatches. |
| **A1 "dispatch per tensor, never by file label"** | **HONOURED**: `quants.dequantize(t.data, t.tensor_type)` per tensor. |
| **A1 §SCHEMA CHANGE** (provenance `fidelity` in every trace) | **NOT IMPLEMENTED** (D16). |
| **D2 "the science target is bf16 safetensors, not quantised"** | **WEAKENED.** `convert` downconverts bf16→fp16 by default (D20), and the C8 gate defaults to the Q4_K_M artefact (D29). |
| **VALIDATION L1** (monotonicity ≠ specificity) | Not violated in code; but `diff_in_means` ships with no matched-norm random control and the README's quickstart presents it as a steering vector. Acceptable if documented (it is, in README "What it does not do"). |
| **VALIDATION L2** (never validate steering on a quantised artefact) | **VIOLATED by the gate default** (D29: `sanity.py` defaults to `q4km-hf`). |
| **VALIDATION L3** (replacement ≠ addition) | **HONOURED**: `_mk_steer` only adds; `sanity.py`'s patch is a replacement and is labelled as such. |
| **PROOF_CHAIN C8 note** ("C8 caught a real bug: it self-patched resid_post into resid_pre") | **REGRESSED** in `mw` (D6): capture post / steer pre, with no pre-site capture available. |

---

## 4. Memory findings (explicit, as required)

1. **The pre-12:47 `load()` was the freeze risk.** It defaulted to `dtype=torch.float32` with no
   `low_cpu_mem_usage`, so `from_pretrained` streamed the checkpoint and held it while casting:
   Granite = 2,533,539,840 × 4 B = **10.13 GB** of weights **+ 5.07 GB** transient ≈ **15.2 GB
   peak** on a box with two QEMU VMs (`-m 6144` + `-m 1024`) and zram swap. The current code
   adds `low_cpu_mem_usage=True` and the guard — that is the right direction.
2. **`guard.plan` under-estimates exactly where it matters** (D8): the `×2.0` for fp32-in-RAM is
   applied only when `disk_gb < 8`. Any model ≥ 8 GB bf16 (i.e. every 7B+) gets
   `need_fp32 = disk × 1.4`, which is ~44% too low.
3. **The guard is bypassed for repo ids and unknown paths** (D9) — the README quickstart is
   unguarded.
4. **`load()` silently changes dtype with free memory** (D7), so two runs of the same experiment
   can differ in numerics with no record.
5. **`guard.zram_swap()` returns True on this box** (verified) and the message text is good.
6. I ran two real loads before the constraint arrived (bf16 Qwen fp32 and Granite fp32, never
   simultaneously in one process — but the *first* script loaded both sequentially and that is
   exactly the pattern the constraint forbids; I stopped after the freeze notice). Everything
   after that was meta-device, header-only, tiny-random-model, or ≤200 MB streamed tensor scans.
   **Granite was not loaded again.** Peak RSS for the post-freeze work stayed under ~1 GB.

---

## 5. What I could not test, and why

- **Mid-generation steering on the real models.** The code path is verified on a tiny random
  model (F2: divergence exactly after the switch); the real-weight version of C5 was not
  re-run because loading 3–5 GB is now forbidden. The code path is architecture-independent
  (all 9 supported classes call decoder layers positionally, verified by source scan), so the
  risk is low but the *number* is not measured.
- **The C1 identity at n_layers on real weights** — done before the freeze (0.000e+00), not
  re-run.
- **Gemma / Gemma2 / Gemma3 / Llama / Mistral conversion.** No GGUF of those architectures on
  this box. I verified the *refusal* paths and the name maps only. The gemma norm convention
  (`GemmaRMSNorm` here applies `(1 + weight)`, and llama.cpp also adds 1 at runtime, so a direct
  copy should be right) is **UNVERIFIED by execution**.
- **`granite.*` GGUF metadata key names.** Cannot re-check without the GGUF, but the produced
  `config.json` contains 12.0 / 0.22 / 0.015625 / 8.0, which are the published Granite 3.3
  constants — so the mapping worked on the real file.
- **`attn_entropy` / `mlp_out_norm`** — not implemented at all, so nothing to test (D16).

---

## 6. Verdict

**Not safe to build on as it stands — but very close, and the problems are shallow.**

What is solid: the numerical core. Residual capture (`resid_post`), the logit lens with Granite's
`logits_scaling`, tied-embedding resolution, the hook site *as a capture point*, the
`logits_to_keep`/DynamicCache decode loop, coef=0 inertness, and mid-generation scheduling are all
**exactly** right — 0.000e+00 at every layer and every token position on both architectures, with
the C5 divergence shape reproduced. The conversion artefact is correct too (no unexpected keys, no
shape mismatch, no non-finite values, 1159× fp16 headroom). The refusal paths refuse. That is a
better core than most released interpretability tooling has.

What blocks a release, in order:

1. **D1** — `guard` is not imported; `load()` raises `NameError`. One line. Nothing else can be
   measured until it is fixed, and the fact that it shipped shows **D2/D3** are the real disease:
   `make check` exits 0 with no linter installed, and no fast test ever calls `load()`.
2. **D4 + D5** — Granite scaling defaults to 1.0 and the only Granite test is self-consistent, so
   a wrongly-converted Granite passes the gate at 4.578e-05. This is the exact failure the
   registry's own note warns about. Fix by refusing on missing metadata and asserting against a
   golden logit vector.
3. **D6** — `diff_in_means` returns a `resid_post(L)` vector that `set_steer(L)` applies at
   `resid_pre(L)`. Every steering number produced through the public API is one block off from the
   site its own vector was measured at, which is the C8 lesson the project already paid for.
4. **D7–D9** — the memory guard must not pick the dtype from transient free RAM without a
   warning, must not drop the ×2 for large models, and must not wave through repo ids.

Then the medium set (D10–D29) is ordinary hardening: one config object, one validated layer index,
idempotent hooks, a real trace schema with provenance, `capabilities()` that can say "no",
shape/finiteness checks in `convert`, weight-presence-based skips, and repo-relative test paths.

**Minimum bar before `git push` to a public MIT repo:** D1, D2, D3, D4, D5, D6, D28 (hardcoded
home paths), and D16 (provenance in `trace`) — the first six because they produce wrong science,
the last two because they will embarrass the project in public exactly the way six misattributed
citations did in `LEDGER`.
