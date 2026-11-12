# Quantisation decision-flip certificate

- reference : `.hf/bf16-hf`
- candidate : `.hf/q4km-hf`
- prompts   : 96 measured, **96 unique**
- gate      : flip rate <= 5%

| metric | value |
|---|---|
| decisions flipped | **21 / 96** (21.9%) |
| meaningful decisions (no whitespace/punctuation) | **19 / 76** (25.0%) [16.6%, 35.8%] |
| trivial decisions excluded | 20 |
| 95% CI on flip rate | [14.8%, 31.1%] |
| confident flips (ref margin > 1 nat) | **2** |
| mean KL(ref‖cand) | 0.1543 |
| max KL | 1.3191 |
| **null control** (same model, other dtype) | **6.2%** flips [2.9%, 13.0%] |

> The null control is the floor: a same-model dtype conversion already flips 6.2% of decisions. Any candidate flip rate at or below this is indistinguishable from storage format.

## By decision class

| class | decisions | flips | rate |
|---|---|---|---|
| `identifier` | 34 | 11 | 32.4% |
| `prose` | 26 | 6 | 23.1% |
| `punctuation` | 13 | 1 | 7.7% |
| `number` | 3 | 1 | 33.3% |
| `whitespace` | 7 | 1 | 14.3% |
| `string` | 10 | 1 | 10.0% |
| `other` | 1 | 0 | 0.0% |
| `keyword` | 2 | 0 | 0.0% |

> Classes are heuristic, from the flipped token string. The authoritative grouping is the prompt that declared the decision.

## Gate

- flip rate **exceeds** the threshold.
- **not certifiable** at 5%: with 96 unique prompts the 95% upper bound is 31.1%. Add prompts or widen the gate.

> A low flip rate is not a statement that the model is safe. It is a statement about the decisions in this prompt set.

## Flips

| prompt | reference | candidate | ref margin | KL |
|---|---|---|---|---|
| `def calculate_tot` | `ale` | `_price` | 0.36 | 0.211 |
| `def parse_us` | `_state` | `able` | 0.31 | 0.152 |
| `import nump` | `ypy` | ` as` | 1.05 | 0.717 |
| `if not is_val` | `ide` | `uable` | 0.61 | 0.122 |
| `with self._lo` | `oper` | ` as` | 0.02 | 0.102 |
| `except Excepti` | `
` | `<|fim_suffix|>` | 0.85 | 0.311 |
| `re.compil` | `ador` | `ations` | 0.32 | 0.162 |
| `def merge_di` | `agon` | `(d` | 0.15 | 0.491 |
| `self.cache.ge` | `ocode` | `ode` | 0.66 | 0.226 |
| `class CacheEnt` | `y` | ` {
` | 0.05 | 0.204 |
| `{"user_na` | `am` | `<|fim_suffix|>` | 0.46 | 0.702 |
| `{"expires_` | `1` | `ago` | 0.06 | 0.235 |
| `{"page_si` | `":` | `<|fim_suffix|>` | 1.29 | 1.113 |
| `{"nested": {"de` | `eper` | `":` | 0.28 | 0.269 |
| `search_web(query="pyth` | `ons` | `o` | 0.96 | 0.193 |
| `run_tests(suite="unit", verb` | `="` | `os` | 0.06 | 0.036 |
| `The author of Hamlet was` | ` a` | ` born` | 0.21 | 0.081 |
| `Two plus two equals` | ` four` | ` three` | 0.19 | 0.113 |
| `# HACK: workaround for the old AP` | `IS` | `FS` | 0.03 | 0.098 |
| `# WARNING: do not call this before ini` | ` file` | `()` | 0.47 | 0.188 |
| `# returns the parsed value or No` | ` value` | `Value` | 0.22 | 0.161 |
