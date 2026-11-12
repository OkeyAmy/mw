# Quantisation decision-flip certificate

- reference : `.hf/bf16-hf`
- candidate : `.hf/q4km-hf`
- prompts   : 41 measured, **41 unique**
- gate      : flip rate <= 5%

| metric | value |
|---|---|
| decisions flipped | **2 / 41** (4.9%) |
| 95% CI on flip rate | [1.4%, 16.1%] |
| confident flips (ref margin > 1 nat) | **0** |
| mean KL(ref‖cand) | 0.0420 |
| max KL | 0.2175 |
| **null control** (same model, other dtype) | **0.0%** flips [0.0%, 8.6%] |

> The null control is the floor: a same-model dtype conversion already flips 0.0% of decisions. Any candidate flip rate at or below this is indistinguishable from storage format.

## Gate

- flip rate is **within** the threshold.
- **not certifiable** at 5%: with 41 unique prompts the 95% upper bound is 16.1%. Add prompts or widen the gate.

> A low flip rate is not a statement that the model is safe. It is a statement about the decisions in this prompt set.

## Flips

| prompt | reference | candidate | ref margin | KL |
|---|---|---|---|---|
| `The author of Hamlet was` | ` a` | ` born` | 0.21 | 0.081 |
| `Two plus two equals` | ` four` | ` three` | 0.19 | 0.113 |
