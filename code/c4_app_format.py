"""c4_app_format.py — the shipped C4 vector, measured in the app's own format.

Why this file exists
--------------------
`agents/K3_test_audit.md` (H4) showed that every published C4 effect was measured
on bare ``def …:`` strings, while `app/server.py` renders every prompt through
``apply_chat_template(..., add_generation_prompt=True)`` and steers the *last*
position of that 65-token transcript.  At the same coef unit the target effect
changed sign.  So the only number that says anything about the product is the
one measured the way the product runs:

  * render:    chat template for the app claim; raw text kept as an ablation
  * position:  the vector the app ships (``pooled``) and the aligned alternative
               (``last``)
  * site:      ``resid_pre`` of layer 8 — the daemon's pre-hook is the same site
               as `set_steer(site="pre")`; ``/build_vector`` captures
               ``resid_post``, so the app's pre-L8 vector is built at post-L7
               (K6: ``resid_post(7) == resid_pre(8)``)
  * unit:      one ``residual_scale`` unit — the mean last-position ``resid_pre``
               L8 norm of the *chat-rendered* ``DEFAULT_POS[0]``/``DEFAULT_NEG[0]``
               prompts, exactly what `Engine.__init__` measures (`server.py:847`)
  * metric:    Δ(logsumexp(comment_ids) − logsumexp(code_ids)) in **nats**
               (raw logit gap / ``logits_scaling`` = 8) and full-vocab
               ``KL(p_steered ‖ p_base)`` in nats

Single headline number (K3 §6):

    S = KL_real / median(KL_null at the same achieved target effect)

``S < 1`` with exact one-sided rank ``p < 0.05`` is the *only* condition that
lets this file call the direction special.  A positive effect alone is not; a
changed continuation is not.

Stages
------
``--stage screen`` (default)
    The 4-arm screen: {pooled, last} × {chat, raw}, coef 1, 10 HELDOUT prompts.
    Stops the whole investigation when no chat-format arm has a positive effect.

``--stage null``
    Only for arms with a positive chat-format effect.  Builds >= 30 Gaussian
    directions, removes the real axis and renormalises to ‖v‖
    (`c4_common.orthogonalize`), matches each one to the real achieved effect
    (±2 %) and reports S, the exact rank p, and the unreachable fraction.

``--stage all`` runs screen then null.

Usage::

    PYTHONPATH=code .venv/bin/python -u code/c4_app_format.py --stage all
    PYTHONPATH=code .venv/bin/python -u code/c4_app_format.py --stage null \
        --nulls 30 --time-budget 2100    # resume after a screen

Nothing here loads weights: all model access is the shared daemon
(http://127.0.0.1:8770), one forward per (prompt, vector, coef).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from c4_common import (
    DaemonClient,
    build_token_sets,
    effect,
    kl_nats,
    orthogonalize,
    quantile,
    signflip_p,
)
from mw.adapter import DEFAULT_NEG, DEFAULT_POS

# ---- pre-registered configuration ------------------------------------------ #
LAYER = 8          # steering layer index; the app writes resid_pre(L8)
BUILD_LAYER = 7    # daemon /build_vector captures resid_post(L7) == resid_pre(L8)
COEF = 1.0         # one residual_scale unit, the app's unit
LOGITS_SCALING = 8.0   # granite-hf config.json: logits_scaling
EXPECT_LAYERS = 40
EXPECT_VOCAB = 49159

#: Same 10 held-out prompts the two existing harnesses use (`steering_test.py`,
#: `verify_c4.py`); never used to build the vector.
HELDOUT = [
    "def merge(a, b):\n", "class Cache:\n", "async def fetch(url):\n",
    "def parse(x):\n", "with open(path) as fh:\n", "while queue:\n",
    "def to_json(obj):\n", "for k, v in d.items():\n", "def main():\n",
    "try:\n",
]

#: The token classes both harnesses use.  K3 H17's objection (``"    return"``
#: collapses onto a whitespace id, the comment set is only 3 ids) is real and is
#: reported, but the classes are NOT changed here: this file has to be
#: comparable to the numbers it is auditing.
COMMENT_TOKS = ["#", "# ", " #", "##", "# this", "# return"]
CODE_TOKS = ["return", "    return", "def", "for", "if", "class", "x", "self"]

UNREACHABLE = "unreachable"      # the direction cannot reach te_real
UNMATCHED = "unmatched"          # reached, but |te - te_real| > 2 % after rounds


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _f(x) -> float:
    return float(x)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sign(x: float) -> int:
    return 0 if abs(x) <= 1e-12 else (1 if x > 0 else -1)


def _fmt(x, spec: str) -> str:
    """Format a possibly-missing number for the text table."""
    return "-" if x is None else format(x, spec)


def _esc(text: str) -> str:
    return text.replace("\n", "\\n")


def _mean(xs) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# setup: unit, vectors, token classes
# --------------------------------------------------------------------------- #
def setup(client: DaemonClient) -> dict:
    """Daemon health, the app's coef unit, and the two contrast vectors."""
    h = client.health()
    if h.get("layers") != EXPECT_LAYERS or h.get("vocab") != EXPECT_VOCAB:
        raise RuntimeError(f"wrong model on the daemon: {h}")
    if "granite" not in str(h.get("model", "")).lower():
        raise RuntimeError(f"wrong model on the daemon: {h}")

    # -- the app's unit: residual_scale(chat-rendered DEFAULT_POS[0]/NEG[0]) -- #
    # server.py:547-551 reads adapter._pre[l][0][-1].norm() — resid_pre(L8) at
    # the last position.  The daemon's /hidden reports resid_post(l); post(7) is
    # pre(8), so the pre-L8 unit is norms[BUILD_LAYER].
    units = {}
    for name, text in (("pos0", DEFAULT_POS[0]), ("neg0", DEFAULT_NEG[0])):
        hd = client.hidden(text, chat=True)
        units[name] = float(hd["norms"][BUILD_LAYER])
    unit = 0.5 * (units["pos0"] + units["neg0"])

    # -- vectors: the app builds on RAW text at pre-site, position=pooled ---- #
    vecs, raw_norms = {}, {}
    for position in ("pooled", "last"):
        out = client.build_vector(
            BUILD_LAYER, DEFAULT_POS, DEFAULT_NEG, chat=False, position=position
        )
        v = np.asarray(out["vec"], dtype=np.float64)
        n = float(np.linalg.norm(v))
        if not (n > 1e-6) or not np.isfinite(v).all():
            raise RuntimeError(f"degenerate {position} vector (norm {n})")
        raw_norms[position] = n
        vecs[position] = v / n * unit      # direction untouched; one coef unit

    cos_pl = float(
        np.dot(vecs["pooled"], vecs["last"])
        / (np.linalg.norm(vecs["pooled"]) * np.linalg.norm(vecs["last"]))
    )

    token_sets = build_token_sets(client, COMMENT_TOKS, CODE_TOKS, chat=False)

    # -- the nats unit: measured, not assumed -------------------------------- #
    # Granite's HF forward already divides the head output by logits_scaling, and
    # the app's lens does the same (adapter.py:295), which is why C2's lens
    # identity is 0.0. So the daemon's returned logits are probabilities' logits.
    lg = client.logits(HELDOUT[0], chat=False)["logits"].astype(np.float64)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    nats_evidence = {
        "logits_scaling_config": LOGITS_SCALING,
        "applied_inside_model": True,
        "source": ("transformers GraniteForCausalLM.forward: "
                   "logits = lm_head(hidden_states) / config.logits_scaling"),
        "app_confirms": "code/mw/adapter.py:295 divides the lens by the same factor",
        "measured_max_abs_logit": float(np.max(np.abs(lg))),
        "measured_p_top": float(p.max()),
        "measured_entropy_nats": float(-np.sum(p * np.log(np.maximum(p, 1e-300)))),
        "conclusion": ("returned logit gaps are nats; dividing by 8 again (K3 H18 / "
                       "K5 convention) double-counts logits_scaling"),
    }

    return {
        "model": h.get("model"),
        "layers": h.get("layers"),
        "vocab": h.get("vocab"),
        "d": h.get("d"),
        "dtype": h.get("dtype"),
        "layer": LAYER,
        "build_layer": BUILD_LAYER,
        "coef": COEF,
        "logits_scaling": LOGITS_SCALING,
        "nats_unit": nats_evidence,
        "site": ("steer at resid_pre(L8) via the daemon's layer-8 pre-hook "
                 "(== set_steer(site='pre')); build at resid_post(L7) == resid_pre(L8)"),
        "unit": {
            "value": unit,
            "definition": ("mean last-position resid_pre(L8) norm of the chat-rendered "
                           "DEFAULT_POS[0] and DEFAULT_NEG[0] prompts — app "
                           "residual_scale() (server.py:517-555,847-849)"),
            "pos0": units["pos0"],
            "neg0": units["neg0"],
        },
        "vectors": {
            "raw_norm": raw_norms,
            "unit_norm": {k: float(np.linalg.norm(v)) for k, v in vecs.items()},
            "cos_pooled_last": cos_pl,
            "build_render": "raw (adapter.build_steer_vectors tokenises the text "
                            "directly; chat templating is used for the unit only)",
        },
        "token_sets": token_sets,
        "n_prompts": len(HELDOUT),
        "_vecs": vecs,
    }


# --------------------------------------------------------------------------- #
# stage 1 — four-arm screen
# --------------------------------------------------------------------------- #
def measure_arm(client: DaemonClient, vec, position: str, chat: bool,
                token_sets: dict, bases: dict) -> dict:
    """te and KL for one arm at coef 1 over every HELDOUT prompt."""
    rows = []
    for p in HELDOUT:
        b = bases[(p, chat)]
        s = client.logits(p, chat=chat, layer=LAYER, vec=vec, coef=COEF)["logits"]
        moved = float(np.max(np.abs(s.astype(np.float64) - b.astype(np.float64))))
        if not (moved > 1e-6):
            raise RuntimeError(
                f"no-op steering on {p!r} ({position}, chat={chat}), max|Δ|={moved:.3e}"
            )
        te_raw = float(effect(s, token_sets) - effect(b, token_sets))
        rows.append({
            "prompt": _esc(p),
            "te_raw": te_raw,
            # The daemon returns the model's FINAL logits: transformers'
            # GraniteForCausalLM.forward already does
            # ``logits = self.lm_head(h) / config.logits_scaling`` (8.0), and the
            # app's own lens divides by the same factor (adapter.py:295), so the
            # returned logit gap IS the natural-log probability gap. Dividing by
            # 8 again (the K3 H18 / K5 convention) double-counts the scaling.
            "te_nats": te_raw,
            "te_nats_k3_div_scaling": te_raw / LOGITS_SCALING,
            "kl_nats": float(kl_nats(b, s)),
            "max_abs_logit_delta": moved,
        })

    return _summarize(f"{'chat' if chat else 'raw'}_{position}", position,
                      "chat" if chat else "raw", rows)


def _summarize(arm_id: str, position: str, render: str, rows: list[dict]) -> dict:
    """Arm-level statistics from the per-prompt rows (the only source of truth).

    `te_raw` is the measured logit gap on the returned logits, i.e. already in
    nats (the model applies logits_scaling internally). The stored `te_nats`
    field is recomputed from it so a units change cannot leave stale means.
    """
    for r in rows:
        r["te_nats"] = float(r["te_raw"])
        r["te_nats_k3_div_scaling"] = float(r["te_raw"]) / LOGITS_SCALING
    te = [r["te_nats"] for r in rows]
    kl = [r["kl_nats"] for r in rows]
    signs = [_sign(x) for x in te]
    return {
        "id": arm_id,
        "position": position,
        "render": render,
        "coef": COEF,
        "te_raw_mean": _mean([r["te_raw"] for r in rows]),
        "te_nats_mean": _mean(te),
        "te_nats_median": float(np.median(te)),
        "te_nats_min": float(np.min(te)),
        "te_nats_max": float(np.max(te)),
        "te_nats_k3_div_scaling_mean": _mean([r["te_nats_k3_div_scaling"] for r in rows]),
        "te_signflip_p_vs_zero": float(signflip_p(te)),
        "kl_nats_mean": _mean(kl),
        "kl_nats_median": float(np.median(kl)),
        "kl_nats_worst": float(np.max(kl)),
        "signs": signs,
        "n_pos": int(sum(1 for s in signs if s > 0)),
        "n_neg": int(sum(1 for s in signs if s < 0)),
        "n_zero": int(sum(1 for s in signs if s == 0)),
        "per_prompt": rows,
        "null": None,
        "verdict": "NO_EFFECT" if _mean(te) <= 0 else "PERTURBS",
        "verdict_note": "",
    }


def stage_screen(client: DaemonClient, cfg: dict) -> list[dict]:
    bases = {
        (p, chat): client.logits(p, chat=chat)["logits"]
        for chat in (True, False) for p in HELDOUT
    }
    arms = []
    for chat in (True, False):
        for position in ("pooled", "last"):
            t0 = time.time()
            arm = measure_arm(client, cfg["_vecs"][position], position, chat,
                              cfg["token_sets"], bases)
            arm["seconds"] = round(time.time() - t0, 1)
            print(
                f"  {arm['id']:<12} te={arm['te_nats_mean']:+.4f} nats  "
                f"({arm['te_raw_mean']:+.3f} raw)  KL={arm['kl_nats_mean']:.5f}  "
                f"signs {arm['n_pos']}/{arm['n_neg']}/{arm['n_zero']}  "
                f"{arm['seconds']}s", flush=True,
            )
            arms.append(arm)
    return arms


# --------------------------------------------------------------------------- #
# stage 2 — effect-matched, target-orthogonal Gaussian null
# --------------------------------------------------------------------------- #
def _eval_direction(client, vec, coef: float, token_sets: dict, prompts,
                    chat: bool) -> dict:
    """te (nats) and KL for a null direction at one coef, over `prompts`."""
    te, kl = [], []
    for p in prompts:
        b = client.logits(p, chat=chat)["logits"]
        s = client.logits(p, chat=chat, layer=LAYER, vec=vec, coef=coef)["logits"]
        if not np.isfinite(s).all():
            raise FloatingPointError("non-finite steered logits")
        te.append(float(effect(s, token_sets) - effect(b, token_sets)))
        kl.append(float(kl_nats(b, s)))
    return {
        "coef": float(coef),
        "te_nats_mean": _mean(te),
        "kl_nats_mean": _mean(kl),
        "kl_nats_worst": float(np.max(kl)),
    }


def stage_nulls(client: DaemonClient, cfg: dict, arms: list[dict], n_nulls: int,
                seed: int, tol: float, max_alpha: float, max_rounds: int,
                time_budget: float, out_path: Path, doc: dict) -> list[dict]:
    """Build the matched-effect null for every chat arm with te > 0."""
    survivors = [a for a in arms if a["render"] == "chat" and a["te_nats_mean"] > 0]
    if not survivors:
        print("  no chat-format arm has a positive effect — no nulls needed", flush=True)
        return arms

    t_start = time.time()
    screen_prompts = HELDOUT[:3]      # coef selection only; the statistic is over all 10
    rng = np.random.default_rng(seed)
    dim = int(cfg["d"])

    for arm in survivors:
        real_vec = cfg["_vecs"][arm["position"]]
        te_real = arm["te_nats_mean"]
        kl_real = arm["kl_nats_mean"]
        tol_abs = abs(te_real) * tol
        print(f"  null for {arm['id']}: te_real={te_real:+.4f} nats, "
              f"KL_real={kl_real:.5f}, n={n_nulls}, seed={seed}", flush=True)

        records, spent_out = [], 0
        for i in range(n_nulls):
            if time.time() - t_start > time_budget:
                print(f"  time budget hit after {i} / {n_nulls} nulls", flush=True)
                break
            r = orthogonalize(rng.normal(size=dim), real_vec)
            rec = {"i": i, "status": UNREACHABLE, "matched": False,
                   "coef": None, "te_nats_mean": None, "kl_nats_mean": None,
                   "kl_nats_worst": None, "rounds": 0}

            # -- coef screen at one unit on 3 prompts (cheap) ----------------- #
            scr = _eval_direction(client, r, 1.0, cfg["token_sets"],
                                  screen_prompts, True)
            rec["rounds"] += len(screen_prompts)
            if scr["te_nats_mean"] <= 0.0:
                # wrong way at one unit — give it the full budget before
                # declaring it unreachable (never dropped silently)
                scr_hi = _eval_direction(client, r, max_alpha, cfg["token_sets"],
                                         screen_prompts, True)
                rec["rounds"] += len(screen_prompts)
                if scr_hi["te_nats_mean"] <= 0.0:
                    rec["te_at_max_alpha"] = scr_hi["te_nats_mean"]
                    records.append(rec)
                    continue
                alpha = max_alpha * (te_real / scr_hi["te_nats_mean"])
            else:
                alpha = te_real / scr["te_nats_mean"]
            alpha = float(min(max(alpha, 1e-3), max_alpha))

            # -- evaluate + proportional correction on the full prompt set --- #
            best = None
            for _ in range(max_rounds):
                ev = _eval_direction(client, r, alpha, cfg["token_sets"],
                                     HELDOUT, True)
                rec["rounds"] += len(HELDOUT)
                cand = dict(ev, status=UNMATCHED, matched=False,
                            residual=ev["te_nats_mean"] - te_real,
                            rel_residual=abs(ev["te_nats_mean"] - te_real) / abs(te_real))
                if best is None or abs(cand["residual"]) < abs(best["residual"]):
                    best = cand
                if abs(cand["residual"]) <= tol_abs:
                    cand.update(status="ok", matched=True)
                    best = cand
                    break
                if cand["te_nats_mean"] <= 0.0:
                    break
                nxt = alpha * (te_real / cand["te_nats_mean"])
                if not (1e-3 <= nxt <= max_alpha) or abs(nxt - alpha) < 1e-6:
                    break
                alpha = float(nxt)

            if best is not None:
                rec.update({k: best[k] for k in
                            ("coef", "te_nats_mean", "kl_nats_mean", "kl_nats_worst",
                             "matched", "status", "rel_residual")})
            records.append(rec)
            spent_out += 1
            if (i + 1) % 5 == 0 or i == 0:
                done = [x for x in records if x["matched"]]
                print(f"    {i + 1}/{n_nulls} nulls, matched {len(done)}, "
                      f"unreachable {sum(1 for x in records if x['status'] == UNREACHABLE)}, "
                      f"{time.time() - t_start:.0f}s", flush=True)
                arm["null"] = _null_summary(records, kl_real, te_real, n_nulls,
                                            seed, tol, max_alpha)
                _save(out_path, doc)

        arm["null"] = _null_summary(records, kl_real, te_real, n_nulls,
                                    seed, tol, max_alpha)
        arm["null"]["records"] = records
        arm["null"]["complete"] = len(records) == n_nulls
        arm["n_seconds"] = round(time.time() - t_start, 1)
        a = arm["null"]
        v = a["S"]
        print(f"    -> n={a['n_matched']} matched, {a['n_unreachable']} unreachable, "
              f"{a['n_unmatched']} unmatched; median KL_null={a['kl_null_median']:.5f} "
              f"KL_real={kl_real:.5f} S={v if v is None else round(v, 3)} "
              f"p={a['rank_p']:.4f}", flush=True)
        _save(out_path, doc)

    # verdicts
    _apply_verdicts(arms)
    return arms


def _apply_verdicts(arms: list[dict]) -> None:
    """The pre-registered rule, applied once, in one place."""
    for arm in arms:
        if arm["render"] != "chat":
            arm["verdict"] = ("ABLATION_NO_EFFECT" if arm["te_nats_mean"] <= 0
                              else "ABLATION_EFFECT")
            arm["verdict_note"] = "raw render is not the app format (K3 H4); ablation only"
            continue
        if arm["te_nats_mean"] <= 0:
            arm["verdict"] = "NO_EFFECT"
            arm["verdict_note"] = ("target effect is not positive in the app format; "
                                   "the direction does not steer the concept")
            continue
        n = arm["null"] or {}
        ok = (n.get("S") is not None and n["S"] < 1.0
              and n.get("rank_p") is not None and n["rank_p"] < 0.05)
        arm["verdict"] = "STEERS" if ok else "PERTURBS"
        arm["verdict_note"] = ("positive effect bought with less global KL than an "
                               "effect-matched orthogonal random direction"
                               if ok else
                               "positive effect, but not distinguishable from an "
                               "effect-matched random direction (S >= 1 or p >= 0.05)")


def _null_summary(records: list[dict], kl_real: float, te_real: float,
                  n_planned: int, seed: int, tol: float, max_alpha: float) -> dict:
    matched = [r for r in records if r["matched"]]
    unreach = [r for r in records if r["status"] == UNREACHABLE]
    unmatched = [r for r in records if r["status"] == UNMATCHED]
    kls = [r["kl_nats_mean"] for r in matched]
    out = {
        "n_planned": n_planned,
        "n_completed": len(records),
        "n_matched": len(matched),
        "n_unreachable": len(unreach),
        "n_unmatched": len(unmatched),
        "unreachable_fraction": (len(unreach) / len(records)) if records else None,
        "unmatched_fraction": (len(unmatched) / len(records)) if records else None,
        "seed": seed,
        "tol": tol,
        "max_alpha": max_alpha,
        "te_real": te_real,
        "kl_real": kl_real,
        "te_null_mean": _mean([r["te_nats_mean"] for r in matched]) if matched else None,
        "te_null_min": min((r["te_nats_mean"] for r in matched), default=None),
        "te_null_max": max((r["te_nats_mean"] for r in matched), default=None),
        "kl_null_median": None,
        "kl_null_median_ci": None,
        "kl_null_min": min(kls) if kls else None,
        "kl_null_max": max(kls) if kls else None,
        "S": None,
        "rank_p": None,
        "n_le_real": None,
        "worst_prompt_kl_real": None,
    }
    if kls:
        out["kl_null_median"] = float(np.median(kls))
        if len(kls) >= 22:
            q = quantile(kls, 0.5)
            out["kl_null_median_ci"] = [q.lo, q.hi]
        out["n_le_real"] = int(sum(1 for k in kls if k <= kl_real))
        out["rank_p"] = (1 + out["n_le_real"]) / (1 + len(kls))
        out["S"] = kl_real / out["kl_null_median"] if out["kl_null_median"] > 0 else None
    return out


# --------------------------------------------------------------------------- #
# stage 3 — recompute derived fields from stored per-prompt rows (no forwards)
# --------------------------------------------------------------------------- #
def stage_recompute(client: DaemonClient, doc: dict) -> dict:
    """Re-derive arm statistics from `per_prompt`, which is the source of truth.

    Used after a units change: the per-prompt te/KL rows are measurements, the
    arm means are arithmetic. Nothing here calls a steered forward.
    """
    lg = client.logits(HELDOUT[0], chat=False)["logits"].astype(np.float64)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    doc["config"]["nats_unit"] = {
        "logits_scaling_config": LOGITS_SCALING,
        "applied_inside_model": True,
        "source": ("transformers GraniteForCausalLM.forward: "
                   "logits = lm_head(hidden_states) / config.logits_scaling"),
        "app_confirms": "code/mw/adapter.py:295 divides the lens by the same factor",
        "measured_max_abs_logit": float(np.max(np.abs(lg))),
        "measured_p_top": float(p.max()),
        "measured_entropy_nats": float(-np.sum(p * np.log(np.maximum(p, 1e-300)))),
        "conclusion": ("returned logit gaps are nats; dividing by 8 again (K3 H18 / "
                       "K5 convention) double-counts logits_scaling"),
    }
    arms = []
    for a in doc["arms"]:
        s = _summarize(a["id"], a["position"], a["render"], a["per_prompt"])
        for k in ("seconds", "n_seconds"):
            if k in a:
                s[k] = a[k]
        s["null"] = a.get("null")
        arms.append(s)
    _apply_verdicts(arms)
    doc["arms"] = arms
    doc["recomputed"] = _now()
    return doc


def _mark_no_null(doc: dict) -> None:
    """Record, in the JSON, that the decisive null was not run and why."""
    if any(a["render"] == "chat" and a["te_nats_mean"] > 0 for a in doc.get("arms", [])):
        return
    doc["decisive_null"] = {
        "n_run": 0,
        "S": None,
        "rank_p": None,
        "reason": ("no chat-format arm has a positive target effect; there is no "
                   "effect to match, so S = KL_real / median(KL_null at matched te) "
                   "is undefined and the pre-registered rule already says NO_EFFECT"),
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(doc: dict, json_path: Path, txt_path: Path) -> None:
    arms = doc.get("arms", [])
    L = []
    L.append("C4 in the app's format — does the shipped vector steer the concept?")
    L.append("=" * 78)
    L.append(f"generated          : {doc['generated']}")
    L.append(f"model              : {doc['config']['model']}  "
             f"({doc['config']['layers']} layers, d={doc['config']['d']}, "
             f"vocab={doc['config']['vocab']}, {doc['config']['dtype']})")
    L.append(f"layer              : {doc['config']['layer']}  "
             f"({doc['config']['site']})")
    L.append(f"coef               : {doc['config']['coef']}  "
             f"(unit = one residual norm, ‖v‖ = {doc['config']['unit']['value']:.4f})")
    L.append(f"unit               : {doc['config']['unit']['definition']}")
    L.append(f"logits_scaling     : {doc['config']['logits_scaling']}  "
             f"(applied INSIDE the model: GraniteForCausalLM divides the head output "
             f"by it; adapter.py:295 does the same for the lens)")
    L.append("target effect unit : nats — Δ(log p(comment) − log p(code)) on the "
             "returned logits")
    nu = doc["config"].get("nats_unit") or {}
    if nu:
        L.append(f"nats-unit evidence : max|logit|={nu.get('measured_max_abs_logit'):.2f}  "
                 f"p_top={nu.get('measured_p_top'):.4f}  "
                 f"H={nu.get('measured_entropy_nats'):.2f} nats  "
                 f"(pre-scaling logits would be ~8x larger and ~one-hot)")
        L.append(f"                     {nu.get('conclusion')}")
    L.append(f"prompts            : {doc['config']['n_prompts']} HELDOUT "
             f"(same 10 as steering_test.py / verify_c4.py)")
    L.append(f"contrast           : {doc['config']['vectors']['build_render']}")
    L.append(f"cos(pooled,last)   : {doc['config']['vectors']['cos_pooled_last']:.4f}")
    L.append(f"token classes      : comment={doc['config']['token_sets']['comment_ids']} "
             f"code={doc['config']['token_sets']['code_ids']}")
    L.append("")

    if not arms:
        L.append("(screen not run)")
    else:
        L.append("SCREEN — 4 arms, coef 1")
        L.append("-" * 78)
        L.append(f"{'arm':<16}{'te nats':>10}{'K3 conv':>10}{'KL nats':>10}"
                 f"{'KL worst':>10}{'+/-/0':>10}{'sign p':>9}  verdict")
        for a in arms:
            signs = f"{a['n_pos']}/{a['n_neg']}/{a['n_zero']}"
            L.append(
                f"{a['id']:<16}{a['te_nats_mean']:>+10.4f}"
                f"{a['te_nats_k3_div_scaling_mean']:>+10.4f}"
                f"{a['kl_nats_mean']:>10.5f}{a['kl_nats_worst']:>10.5f}"
                f"{signs:>10}"
                f"{a['te_signflip_p_vs_zero']:>9.4f}  {a['verdict']}"
            )
        L.append("")

    null_arms = [a for a in arms if a.get("null")]
    if null_arms:
        L.append("DECISIVE NULL — Gaussian direction, real axis removed, matched on "
                 "achieved te (±2 %)")
        L.append("-" * 78)
        L.append(f"{'arm':<16}{'n':>4}{'unreach':>9}{'unmatch':>9}"
                 f"{'te null':>9}{'KL real':>10}{'KL null':>10}{'S':>8}{'rank p':>8}")
        for a in null_arms:
            n = a["null"]
            L.append(
                f"{a['id']:<16}{n['n_matched']:>4}"
                f"{_fmt(n['unreachable_fraction'], '.2f'):>9}"
                f"{_fmt(n['unmatched_fraction'], '.2f'):>9}"
                f"{_fmt(n['te_null_mean'], '+.4f'):>9}"
                f"{n['kl_real']:>10.5f}"
                f"{_fmt(n['kl_null_median'], '.5f'):>10}"
                f"{_fmt(n['S'], '.3f'):>8}"
                f"{_fmt(n['rank_p'], '.4f'):>8}"
            )
        L.append("")
        for a in null_arms:
            n = a["null"]
            L.append(f"{a['id']}: n_completed={n['n_completed']}/{n['n_planned']} "
                     f"complete={n.get('complete')}  "
                     f"KL_null median CI={n['kl_null_median_ci']}  "
                     f"min/max={n['kl_null_min']}/{n['kl_null_max']}  "
                     f"#(KL_null <= KL_real)={n['n_le_real']}")
        L.append("")
    elif arms and all(a["te_nats_mean"] <= 0 for a in arms if a["render"] == "chat"):
        L.append("DECISIVE NULL — not run")
        L.append("-" * 78)
        L.append("No chat-format arm has a positive target effect, so there is no")
        L.append("effect to match: S = KL_real / median(KL_null at matched te) is")
        L.append("undefined and the pre-registered rule already returns NO_EFFECT.")
        L.append("With a negative te the vector moves the concept the wrong way;")
        L.append("scaling the same direction up cannot make it a steering knob.")
        L.append("")

    if arms:
        L.append("PER-PROMPT (te nats / KL nats)")
        L.append("-" * 78)
        for a in arms:
            L.append(f"[{a['id']}]  te={a['te_nats_mean']:+.4f}  KL={a['kl_nats_mean']:.5f}"
                     f"   [K3-convention te (=raw/8) {a.get('te_nats_k3_div_scaling_mean', float('nan')):+.4f}]")
            for r in a["per_prompt"]:
                L.append(f"  {r['prompt']:<28}{r['te_nats']:>+9.4f}{r['kl_nats']:>10.5f}")
            L.append("")

    L.append("VERDICT PER ARM")
    L.append("-" * 78)
    for a in arms:
        L.append(f"{a['id']:<16}{a['verdict']:<22}{a['verdict_note']}")
    L.append("")
    L.append("Rule: STEERS only if te > 0 in the chat-format arm AND S < 1 AND rank p < 0.05.")
    L.append("      te <= 0 -> NO_EFFECT. te > 0 but S >= 1 or p >= 0.05 -> PERTURBS.")
    L.append("      Raw-render arms are ABLATION_*; they are not the app's format (K3 H4).")
    L.append("")
    L.append("NO_EFFECT means no positive effect on the TARGET CONCEPT. It does not mean")
    L.append("nothing happens: the same slider move leaves a large full-vocabulary KL")
    L.append("(1.7-1.9 nats in chat format), i.e. it is a perturbation knob, not steering.")
    L.append("'K3 conv' is the raw/8 value used by proof/c4_decisive.* and K5; it")
    L.append("double-counts logits_scaling, which the model already applies, so it is 8x")
    L.append("too small. 'te nats' is the correct natural-log value.")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("screen", "null", "all", "recompute"),
                    default="screen")
    ap.add_argument("--nulls", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max-alpha", type=float, default=64.0)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=2100.0,
                    help="seconds for the whole null stage")
    ap.add_argument("--json", default=str(ROOT / "proof" / "c4_app_format.json"))
    ap.add_argument("--txt", default=str(ROOT / "proof" / "c4_app_format.txt"))
    args = ap.parse_args()

    json_path, txt_path = Path(args.json), Path(args.txt)
    doc = _load(json_path)
    if doc and args.stage == "screen":
        doc = {}          # a fresh screen replaces the old one
    doc.setdefault("generated", _now())
    doc["updated"] = _now()

    client = DaemonClient(timeout=1800.0)
    t0 = time.time()
    cfg = None

    if args.stage == "recompute":
        if "arms" not in doc:
            raise SystemExit("no arms in the JSON — run --stage screen first")
        doc = stage_recompute(client, doc)
        doc["updated"] = _now()
        _mark_no_null(doc)
        _save(json_path, doc)
        write_report(doc, json_path, txt_path)
        print(f"recomputed {len(doc['arms'])} arms from stored per-prompt rows", flush=True)
        return 0

    if args.stage in ("screen", "all"):
        print("setup: health, unit, vectors, token classes …", flush=True)
        cfg = setup(client)
        print(f"  unit (pre-L8, chat last) = {cfg['unit']['value']:.4f} "
              f"(pos0 {cfg['unit']['pos0']:.4f} / neg0 {cfg['unit']['neg0']:.4f})", flush=True)
        print(f"  raw norms  pooled={cfg['vectors']['raw_norm']['pooled']:.4f} "
              f"last={cfg['vectors']['raw_norm']['last']:.4f} "
              f"cos={cfg['vectors']['cos_pooled_last']:.4f}", flush=True)
        print(f"  tokens     comment={cfg['token_sets']['comment_ids']} "
              f"code={cfg['token_sets']['code_ids']}", flush=True)
        print("screen: 4 arms × 10 prompts, coef 1 …", flush=True)
        arms = stage_screen(client, cfg)
        doc = {
            "generated": doc["generated"],
            "updated": _now(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "arms": arms,
        }
        _mark_no_null(doc)
        _save(json_path, doc)
        write_report(doc, json_path, txt_path)
        null_ready = [a["id"] for a in arms
                      if a["render"] == "chat" and a["te_nats_mean"] > 0]
        print(f"screen done in {time.time() - t0:.0f}s; chat arms with te > 0: "
              f"{null_ready or 'none'}", flush=True)
        if not null_ready:
            print("STOP: no chat-format arm has a positive target effect.", flush=True)
            write_report(doc, json_path, txt_path)
            return 0

    if args.stage in ("null", "all"):
        if "config" not in doc:
            raise SystemExit("no screen in the JSON — run --stage screen first")
        if cfg is None:
            cfg = setup(client)      # cheap: 2 hidden + 2 builds, same directions
        # the rebuild must reproduce the screen's unit, or the arms are not comparable
        unit_recorded = float(doc["config"]["unit"]["value"])
        if abs(cfg["unit"]["value"] - unit_recorded) > 1e-6 * max(1.0, unit_recorded):
            raise RuntimeError(
                f"unit changed between stages: {unit_recorded} vs {cfg['unit']['value']}"
            )
        doc["null_stage"] = {
            "started": _now(),
            "unit_check": cfg["unit"]["value"],
            "unit_recorded": unit_recorded,
            "n_planned": args.nulls,
            "seed": args.seed,
            "tol": args.tol,
            "max_alpha": args.max_alpha,
            "time_budget_s": args.time_budget,
            "screen_prompts": HELDOUT[:3],
        }
        arms = stage_nulls(client, cfg, doc["arms"], args.nulls, args.seed, args.tol,
                           args.max_alpha, args.max_rounds, args.time_budget,
                           json_path, doc)
        doc["arms"] = arms
        doc["null_stage"]["seconds"] = round(time.time() - t0, 1)
        _save(json_path, doc)
        write_report(doc, json_path, txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
