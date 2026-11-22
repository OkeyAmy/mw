"""c4_common.py — shared, model-free helpers for the C4 steering measurements.

One module, one estimator, so the divergence documented in
`agents/K3_test_audit.md` (H13: two harnesses, two controls, two calibrations)
cannot recur. This file is deliberately **import-light**: nothing here loads
weights. Every model access goes through the out-of-process daemon HTTP API, so
a driver script can import it, run under pytest, or be used from a notebook
without touching the 10 GB Granite checkpoint.

What this module replaces
-------------------------
* `steering_test.py:pct`  -> `quantile()`  (H5/H25: `round(p*(n-1))` silently
  returns the maximum for n <= 21 and reports no uncertainty).
* `ctrl_eff` (abs yes/no + abs digits/comments) -> `kl_nats()` (H1/H2: the old
  "collateral" contained the target and was non-monotone; KL has a true zero at
  a no-op and is direction-sensitive).
* `mean(real) - mean(base)` + max-of-N gate -> `signflip_p()` (H5/H6: the paired
  design was analysed as if unpaired, against a threshold that is itself a
  high-variance random variable).
* norm-matched random controls -> `orthogonalize()` + `match_effect()` (§5 of K3:
  match on *achieved target effect*, not on vector norm).

Units. The daemon returns raw model logits (Granite has `logits_scaling = 8.0`).
`target_te()` and `match_effect()` work in those raw logit units because that is
what the existing artifacts report; divide by `logits_scaling` for nats.
`kl_nats()` is already in nats (it softmaxes, so the scaling cancels).

Typical use (see `agents/K5_c4_common.md` for the full driver sketch):

    from c4_common import (DaemonClient, build_token_sets, target_te,
                           match_effect, kl_nats)
    c = DaemonClient()
    vec = c.build_vector(8, POS, NEG)["vec"]
    te = target_te(c, prompt, 8, vec, 1.0, token_sets, chat=True)
    alpha, achieved, reached = match_effect(c, prompt, 8, rnd, te, token_sets, chat=True)
"""

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.request
from collections import namedtuple

import numpy as np

DEFAULT_BASE = "http://127.0.0.1:8770"

#: Return value of :func:`quantile`. Unpacks as ``(value, lo, hi)``.
Quantile = namedtuple("Quantile", "value lo hi n q alpha")


# --------------------------------------------------------------------------- #
# daemon client
# --------------------------------------------------------------------------- #
class DaemonError(RuntimeError):
    """The steering daemon is unreachable or returned an error."""


def _decode_f32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()


class DaemonClient:
    """Thin, dependency-free client for the shared Granite daemon.

    Decodes ``logits_b64`` / ``vec_b64`` into ``np.float32`` arrays. The decoded
    array is exposed under ``logits`` / ``vec``; the raw base64 field is kept so
    callers can cache or forward payloads without re-encoding.

    Unsteered logits are memoised (``cache=True``) because every effect
    measurement needs the same baseline: the naive harness does 3 forwards per
    prompt per vector (K3 H20), this one does 1 + N_vectors.
    """

    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 180.0, cache: bool = True):
        self.base = base.rstrip("/")
        self.timeout = float(timeout)
        self.cache = bool(cache)
        self._base_logits: dict[tuple[str, bool], dict] = {}

    # -- plumbing ---------------------------------------------------------- #
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # daemon up, request rejected
            body = exc.read()[:400].decode("utf-8", "replace")
            raise DaemonError(f"{self.base}{path} -> HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, OSError) as exc:  # daemon down
            raise DaemonError(
                f"steering daemon unreachable at {self.base}{path} ({exc}). "
                "Check `curl -s http://127.0.0.1:8770/health` and start the "
                "daemon before using c4_common."
            ) from exc

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.base + path, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read()[:400].decode("utf-8", "replace")
            raise DaemonError(f"{self.base}{path} -> HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise DaemonError(
                f"steering daemon unreachable at {self.base}{path} ({exc}). "
                "Check `curl -s http://127.0.0.1:8770/health` and start the "
                "daemon before using c4_common."
            ) from exc

    def clear_cache(self) -> None:
        self._base_logits.clear()

    # -- endpoints --------------------------------------------------------- #
    def health(self) -> dict:
        """``{"ok", "model", "layers", "vocab", "d", "dtype"}`` (GET /health)."""
        return self._get("/health")

    def render(self, text: str, chat: bool = False) -> dict:
        """``{"ids", "tokens", "text", "n"}`` for the (optionally chat-templated) text."""
        return self._post("/render", {"text": text, "chat": bool(chat)})

    def hidden(self, text: str, chat: bool = False) -> dict:
        """``{"norms": [layers], "n_tokens"}`` — per-layer residual norms."""
        return self._post("/hidden", {"text": text, "chat": bool(chat)})

    def logits(
        self,
        text: str,
        chat: bool = False,
        layer: int | None = None,
        vec: str | np.ndarray | None = None,
        coef: float = 0.0,
        use_cache: bool = True,
    ) -> dict:
        """Next-token logits at the last position, optionally steered.

        Returns the daemon payload plus ``logits`` (``np.float32``, vocab).
        ``vec`` may be a base64 string (as returned by :meth:`build_vector`) or a
        numpy array. Steered calls are never cached.
        """
        payload: dict = {"text": text, "chat": bool(chat)}
        steered = vec is not None and coef != 0.0
        if steered:
            if isinstance(vec, np.ndarray):
                vec = base64.b64encode(np.ascontiguousarray(vec, np.float32).tobytes()).decode()
            payload.update(layer=layer, vec=vec, coef=float(coef))
        elif layer is not None:
            payload["layer"] = layer

        key = (text, bool(chat))
        if not steered and self.cache and use_cache and key in self._base_logits:
            return dict(self._base_logits[key])  # shallow copy; array is shared
        out = self._post("/logits", payload)
        out["logits"] = _decode_f32(out["logits_b64"])
        if not steered and self.cache:
            self._base_logits[key] = out
        return out

    def build_vector(
        self, layer: int, pos, neg, chat: bool = False, position: str = "pooled"
    ) -> dict:
        """Diff-in-means vector. Returns ``{"vec", "norm", "site", "layer", ...}``."""
        if not pos or not neg:
            raise ValueError("build_vector needs non-empty pos and neg lists")
        if position not in ("pooled", "last"):
            raise ValueError(f"position must be 'pooled' or 'last', got {position!r}")
        out = self._post(
            "/build_vector",
            {
                "layer": int(layer),
                "pos": list(pos),
                "neg": list(neg),
                "chat": bool(chat),
                "position": position,
            },
        )
        out["vec"] = _decode_f32(out["vec_b64"])
        return out


def daemon_available(base: str = DEFAULT_BASE, timeout: float = 2.0) -> bool:
    """True iff ``/health`` answers and reports ``ok``. Used by pytest skipif."""
    try:
        out = DaemonClient(base, timeout=timeout).health()
    except (DaemonError, ValueError, KeyError):
        return False
    return bool(out.get("ok", False))


def build_token_sets(
    client: DaemonClient, comment: list[str], code: list[str], chat: bool = False
) -> dict:
    """Map surface strings to first-token ids, using the daemon's own tokenizer.

    Returns ``{"comment_ids": [...], "code_ids": [...]}`` (sorted, deduplicated)
    so no local tokenizer load is needed. K3 H17 warns that these surface strings
    collapse onto a handful of ids — print the sets and check them; the point of
    this helper is only to remove the duplicated `first_ids()` copies.
    """

    def ids(strings):
        seen = set()
        for s in strings:
            r = client.render(s, chat=chat)
            if r.get("ids"):
                seen.add(int(r["ids"][0]))
        if not seen:
            raise ValueError(f"no token ids for {strings!r}")
        return sorted(seen)

    return {"comment_ids": ids(comment), "code_ids": ids(code)}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _as_f64(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("empty logits array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits contain NaN/Inf")
    return arr


def _log_softmax(logits) -> np.ndarray:
    z = _as_f64(logits)
    m = z.max()
    return z - (m + math.log(np.exp(z - m).sum()))


def kl_nats(logits_a, logits_b) -> float:
    """Full-vocabulary ``KL(p_b || p_a)`` in nats.

    ``p = softmax(logits)`` in float64. This is the replacement for `ctrl_eff`
    (K3 H1/H2): it is >= 0 with a true zero at a no-op, is direction-sensitive,
    and is comparable across models. Note the argument order — the *second*
    distribution is the numerator, i.e. ``KL(steered || base)`` when called as
    ``kl_nats(base_logits, steered_logits)``.

    Float32 logits can yield ~1e-9 negatives from cancellation; they are clamped
    to 0.0.
    """
    a = _as_f64(logits_a)
    b = _as_f64(logits_b)
    if a.shape != b.shape:
        raise ValueError(f"logits shape mismatch: {a.shape} vs {b.shape}")
    log_p = _log_softmax(b)
    log_q = _log_softmax(a)
    p = np.exp(log_p)
    val = float(np.sum(p * (log_p - log_q)))
    return max(val, 0.0)


def topk_kl(logits_a, logits_b, k: int = 256) -> float:
    """Approximate ``KL(p_b || p_a)`` on the union of both top-k supports.

    Convenience for quick triage only — renormalises the truncated distributions,
    so it is *not* a lower bound. Report :func:`kl_nats` in anything decisive.
    """
    a = _as_f64(logits_a)
    b = _as_f64(logits_b)
    if a.shape != b.shape:
        raise ValueError(f"logits shape mismatch: {a.shape} vs {b.shape}")
    k = max(1, min(int(k), a.size))
    idx = np.union1d(np.argsort(a)[-k:], np.argsort(b)[-k:])
    return kl_nats(a[idx], b[idx])


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1)))


def _binom_ppf(q: float, n: int, p: float) -> int:
    """Smallest k with CDF(k) >= q (q in (0,1))."""
    if q <= 0.0:
        return 0
    if q >= 1.0:
        return n
    acc = 0.0
    for k in range(n + 1):
        acc += math.comb(n, k) * p**k * (1.0 - p) ** (n - k)
        if acc >= q:
            return k
    return n


def quantile(xs, q: float, require_n: int = 22, alpha: float = 0.05) -> Quantile:
    """Nearest-rank quantile plus a distribution-free order-statistic interval.

    Replaces `pct()` (K3 H5/H25), whose `round(q*(n-1))` index is the sample
    maximum for every ``n <= 21``. Here ``n < require_n`` raises instead of
    silently returning the max, and the returned interval makes the sampling
    error visible.

    Parameters
    ----------
    xs : sequence of float
    q : float in (0, 1) — the quantile, e.g. 0.95 for a p95 gate.
    require_n : int, default 22. The smallest n for which the nearest-rank index
        is not the maximum at q=0.95 (``ceil(0.95*n) < n`` iff ``n >= 21``; 22
        keeps a margin).
    alpha : float, default 0.05 — two-sided coverage of the interval.

    Returns
    -------
    Quantile(value, lo, hi, n, q, alpha)

    The interval is ``[X_(l), X_(u)]`` where ``l = ppf(alpha/2, n, q) + 1`` and
    ``u = ppf(1 - alpha/2, n, q) + 1``, from
    ``P(X_(r) <= xi_q) = P(Binomial(n, q) >= r)``. It brackets ``value`` by
    construction; when the interval is degenerate (small n, extreme q) that is
    itself the diagnostic.
    """
    vals = sorted(float(x) for x in xs)
    n = len(vals)
    if n < require_n:
        raise ValueError(
            f"quantile(q={q}) needs n >= {require_n} samples, got {n}; "
            "a p95 gate on n < 22 is the sample maximum (K3 H5/H25)"
        )
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rank = min(n, max(1, math.ceil(q * n)))  # 1-based nearest rank
    lo_i = min(n, max(1, _binom_ppf(alpha / 2.0, n, q) + 1))
    hi_i = min(n, max(1, _binom_ppf(1.0 - alpha / 2.0, n, q) + 1))

    value = vals[rank - 1]
    lo, hi = vals[lo_i - 1], vals[hi_i - 1]
    lo, hi = min(lo, value), max(hi, value)  # numeric belt-and-braces
    return Quantile(value, lo, hi, n, q, alpha)


def signflip_p(deltas, exact_max_n: int = 20, n_mc: int = 20000, seed: int = 20261122) -> float:
    """Exact two-sided sign-flip permutation p-value for a paired design.

    Statistic: ``sum(s_i * d_i)`` with ``s_i`` in {-1, +1}. The p-value is
    ``P(|T*| >= |T_obs|)`` over the sign-flip distribution (the paired analogue
    of a one-sample test, valid without normality).

    * ``n <= exact_max_n``: full enumeration of ``2**n`` sign patterns — exact,
      no sampling error. All-positive differences give ``2 / 2**n`` (the identity
      and its mirror); all-zero differences give ``1.0``.
    * ``n > exact_max_n``: Monte Carlo with a fixed seed and the add-one
      correction ``(1 + hits) / (1 + n_mc)``, so the result is deterministic and
      never 0.

    This replaces the mean-difference-versus-p95 comparison in
    `steering_test.py` (K3 H6).
    """
    d = np.asarray([float(x) for x in deltas], dtype=np.float64)
    n = d.size
    if n == 0:
        raise ValueError("signflip_p needs at least one difference")
    if not np.all(np.isfinite(d)):
        raise ValueError("differences contain NaN/Inf")
    obs = abs(float(d.sum()))
    scale = max(1.0, float(np.abs(d).sum()))

    if n <= exact_max_n:
        # Enumerate sign vectors as bits of an integer.
        combos = np.arange(2**n, dtype=np.int64)
        bits = ((combos[:, None] >> np.arange(n)) & 1) * 2 - 1
        stats = np.abs(bits.astype(np.float64) @ d)
        return float(np.count_nonzero(stats >= obs - 1e-12 * scale)) / float(2**n)

    rng = np.random.default_rng(seed)
    hits = 0
    remaining = n_mc
    while remaining > 0:
        chunk = min(remaining, 4096)
        signs = rng.integers(0, 2, size=(chunk, n)) * 2 - 1
        stats = np.abs(signs.astype(np.float64) @ d)
        hits += int(np.count_nonzero(stats >= obs - 1e-12 * scale))
        remaining -= chunk
    return (1.0 + hits) / (1.0 + n_mc)


def orthogonalize(r, v) -> np.ndarray:
    """``r`` with the ``v`` axis removed, renormalised to ``||v||``.

    ``r - (r.v)/(v.v) * v`` then scaled to the norm of ``v``. This is the
    null-direction construction from K3 §5.3 step 3: it blocks the cheap
    "borrow the target axis" null, so a random control cannot smuggle in the
    direction being tested.

    Raises ``ValueError`` for a zero ``v`` or an ``r`` that lies entirely along
    ``v`` (the orthogonal complement is then undefined).
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if r.shape != v.shape:
        raise ValueError(f"shape mismatch: {r.shape} vs {v.shape}")
    vv = float(v @ v)
    if vv <= 0.0:
        raise ValueError("orthogonalize needs a non-zero v")
    out = r - (float(r @ v) / vv) * v
    norm = float(np.linalg.norm(out))
    if norm <= 1e-12 * math.sqrt(vv):
        raise ValueError("r is parallel to v; the orthogonal complement is empty")
    return (out / norm) * math.sqrt(vv)


# --------------------------------------------------------------------------- #
# target effect + effect matching
# --------------------------------------------------------------------------- #
def _logsumexp(lg: np.ndarray, ids) -> float:
    idx = np.asarray(ids, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("empty token id set")
    return float(np.logaddexp.reduce(lg[idx].astype(np.float64)))


def _check_token_sets(token_sets: dict) -> tuple[list, list]:
    try:
        comment_ids = list(token_sets["comment_ids"])
        code_ids = list(token_sets["code_ids"])
    except (TypeError, KeyError) as exc:
        raise ValueError("token_sets must be a dict with 'comment_ids' and 'code_ids'") from exc
    if not comment_ids or not code_ids:
        raise ValueError("token_sets comment_ids and code_ids must be non-empty")
    return comment_ids, code_ids


def effect(logits, token_sets: dict) -> float:
    """``logsumexp(comment) - logsumexp(code)`` on raw logits."""
    lg = _as_f64(logits)
    comment_ids, code_ids = _check_token_sets(token_sets)
    return _logsumexp(lg, comment_ids) - _logsumexp(lg, code_ids)


def target_te(
    client: DaemonClient,
    prompt: str,
    layer: int,
    vec,
    coef: float,
    token_sets: dict,
    chat: bool = False,
) -> float:
    """Δ target effect versus unsteered, in raw logit units.

    ``Δ(logsumexp(comment) - logsumexp(code))`` at ``coef`` minus the same
    quantity with no steering. The baseline call is memoised by the client, so
    sweeping ``coef`` costs one forward per point, not two.
    """
    if vec is None or coef == 0.0:
        return 0.0
    _check_token_sets(token_sets)  # fail before spending a forward
    base = client.logits(prompt, chat=chat)["logits"]
    steered = client.logits(prompt, chat=chat, layer=layer, vec=vec, coef=coef)["logits"]
    if not np.isfinite(steered).all():
        raise FloatingPointError("daemon returned non-finite steered logits")
    delta = float(effect(steered, token_sets) - effect(base, token_sets))
    return 0.0 if delta == 0.0 else delta


def match_effect(
    client: DaemonClient,
    prompt: str,
    layer: int,
    vec,
    target_te: float,
    token_sets: dict,
    chat: bool = False,
    max_alpha: float = 64.0,
    tol: float = 0.02,
    max_iter: int = 40,
) -> tuple[float, float, bool]:
    """Bisect ``coef`` until the achieved target effect matches ``target_te``.

    This is the K3 §5.4 null construction: a random direction is credited with
    the *effect* it achieves, not with a matched norm, so the comparison cannot
    be won by scaling.

    Returns ``(alpha, achieved_te, reached)``.

    * ``tol`` is **relative** to ``target_te`` (default 2 %), matching the
      "``te(r, α) = te_real ± 2 %``" protocol; a floor of ``1e-6`` absolute is
      applied for tiny targets.
    * If the direction cannot reach the target before ``max_alpha``, the call
      returns ``reached=False`` with the effect at ``max_alpha`` and **does not
      raise** — the driver must report the unreachable fraction rather than
      silently dropping those directions (K3 §5.4).
    * ``target_te == 0`` is satisfied at ``alpha = 0``.
    """
    tgt = float(target_te)
    if not math.isfinite(tgt):
        raise ValueError("target_te must be finite")
    tol_abs = max(abs(tgt) * float(tol), 1e-6)
    if tgt == 0.0:
        return 0.0, 0.0, True
    if max_alpha <= 0.0:
        raise ValueError("max_alpha must be > 0")

    sign = 1.0 if tgt > 0.0 else -1.0
    goal = sign * tgt  # positive
    # `target_te` is shadowed by the parameter name, so resolve the module-level
    # function dynamically (which also keeps it monkeypatchable in tests).
    te_fn = globals()["target_te"]

    def f(alpha: float) -> float:
        return sign * te_fn(client, prompt, layer, vec, alpha, token_sets, chat)

    f_hi = f(max_alpha)
    if f_hi <= 0.0:
        # The direction moves the effect the wrong way (or not at all).
        return float(max_alpha), sign * f_hi, False
    if abs(f_hi - goal) <= tol_abs:
        return float(max_alpha), sign * f_hi, True

    lo, hi = 0.0, float(max_alpha)
    if f_hi < goal:
        return hi, sign * f_hi, False

    best = (hi, f_hi)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        best = (mid, val)
        if abs(val - goal) <= tol_abs:
            return mid, sign * val, True
        if val < goal:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, float(max_alpha)):
            break
    alpha, val = best
    return float(alpha), sign * val, abs(val - goal) <= tol_abs
