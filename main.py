"""gta6-trophy-rarity-forecast-01: a join/earn forecaster with an ensemble
median (Python 3.13, stdlib only).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target, from the house's
reads of the game's trophy page up to the case's checkpoint.

Why this shape, not persistence
--------------------------------
An earn rate is a ratio, rate = E / N, of two moving quantities:

* N(t), the players Exophase tracks, only grows: new players sync profiles.
* E_i(t), how many of them hold trophy i, grows as players play long enough
  to earn it.

Persistence ("the rate stays where it is") ignores both motions, and the
practice shows how badly that fails: launch night's page sat unchanged at
638 tracked players, then one refresh took it to 3,868 players and every
rate but the 100% one fell, because the newcomers had earned almost
nothing yet. A forecaster that models the two motions separately can see
that dilution coming; one that fits the rate alone cannot.

The median: an ensemble of two views
--------------------------------------
1. Structural (join/earn). Cumulative tracked players follow a saturating
   Weibull curve J(t) = J0 + A * (1 - exp(-(t / tau) ** k)), fitted to the
   distinct reads (consecutive identical reads carry no new information:
   Exophase refreshes in jumps, hours apart, and the page is frozen
   between refreshes). A player who joins at time s earns trophy i after
   a delay D_i (lognormal, median d_i, spread sigma_i: playtime to reach
   it varies across players), with probability p_i (the share of players
   who ever earn it). So the earned share is
       E_i(t) / J(t) = p_i * H_i(t),
   where H_i(t) is the share of the tracked population whose join time
   plus delay has passed by t, computed by convolving the fitted join
   density with the lognormal delay CDF. (p_i, d_i, sigma_i) are fitted
   per trophy on a grid, scored on log(E + 0.5). The median rate at the
   target T is 100 * p_i * H_i(T). Using a spread delay instead of a
   point delay matters: a point delay fits the middle of a trophy's rise
   but mis-extrapolates its shoulders, and the final is scored on the
   shoulders.

2. Trend. A least-squares line of logit(rate) against ln(1 + hours since
   release), fitted per trophy over all its reads, read at the target,
   capped at the trophy's description-prior eventual share (a rate cannot
   run past the share of players who ever earn it). This is the house
   curve baseline's shape, but fitted on the whole history instead of
   the last 24 hours and capped, so it cannot run away on a short rise.

The forecast median is the logit-average of the two views. Neither view
is trusted alone: the structural view can misread the join curve's tail,
the trend view can chase noise; their average was the most accurate of
the candidates on every synthetic launch we backtested (player-level
simulations with known truth, Weibull joins, lognormal delays, 4-hour
refreshes), beating persistence and both parents in each scenario.

Description priors. A trophy nobody (or almost nobody) has earned by the
deadline still has a future: its own series says nothing, but its
description says what it takes. Story chapters are earned by a large
share of players after a short delay; grinds and 100% completion by a
small share after a long one. description_prior() encodes that ordering
only -- rough magnitudes, in tenure hours at launch-week play intensity.
Trophies with real earned history use their fit; trophies with a little
history are shrunk toward the prior; trophies with none use the prior
outright, and any trophy leaning on the prior gets the maximum spread
(2.0 logit units), because the prior is a guess and the pinball loss
punishes a confident wrong median far more than an honest wide one.

The spread (logit scale). Honest sources of uncertainty, added in
quadrature, per trophy:
* fit residual: the trophy's residual std of logit(rate) around its
  structural curve, shrunk toward the pooled median across trophies;
* refresh lumpiness: the answer is the page as of Exophase's last refresh
  before the target, so the effective "as of" time is the target minus up
  to one refresh gap; the term is the trophy's logit slope at the target
  times half the median gap between distinct reads;
* model disagreement: half the logit gap between the two views' medians;
* model error over the horizon: a 0.20 logit floor.
The total spread is clamped to [0.20, 2.0] logit units. Quantiles are
q_tau = inv_logit(logit(m_i) + z_tau * s_i) with the standard normal z
at (0.05, 0.25, 0.5, 0.75, 0.95).

Flat-history fallback. When the history has too few distinct reads to fit
anything (the practice cases: every read identical), no trend can be
estimated, but the kit's own account of launch night says what happens
next: a refresh is coming and it dilutes. The fallback assumes one
refresh before the target with
   N_target = N_last * (1 + t_target_hours / 1.0)
(a prior calibrated on the documented first refresh: 638 -> 3,868 tracked
players about 4.7 h after the first read), and that each newcomer earns
the 100% trophy (if one exists, it stays at 100%) plus ~0.03 other
trophies on average, spread over the other trophies in proportion to
their current earned counts (the documented refresh added 3,333 earned
trophies over 3,230 newcomers, 3,230 of them the 100% trophy). The
spread is a wide 0.9 logit units, because the dilution's size is a
prior, not a measurement. The final case has a rich, moving history, so
this path should never fire there; it exists so a flat case still gets
an honest, diluted forecast instead of persistence.

Determinism and safety: no randomness, no network, no clock reads; every
case is processed independently; any unexpected failure in one case falls
back to a wide persistence forecast for that case rather than crashing,
because a crashed case is scored as persistence anyway and ranks below
every valid entry.

Usage (as the grader runs it): python3 main.py, reading every *.in in
$MAKEFUN_INPUT_DIR (else /input) and writing <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Delay-median grid (hours) and lognormal delay spreads for the structural fit.
D_MED_GRID = [2, 4, 6, 9, 13, 19, 28, 40, 58, 84, 120, 170, 240]
SIGMAS = [0.4, 0.7, 1.1]
# Weibull (tau hours, k) grid for the joins curve.
TAUS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
KS = [0.3, 0.45, 0.6, 0.8, 1.0, 1.3]

# Fallback (flat history) priors -- see the module docstring.
FALLBACK_N_GROWTH_PER_HOUR = 1.0   # N_target = N_last * (1 + t_target / 1h)
FALLBACK_NEW_TROPHIES_EACH = 0.0319  # non-100% trophies a newcomer earns
FALLBACK_SPREAD = 0.9
MODEL_SPREAD = 0.20   # horizon model-error term, logit units
SPREAD_MIN = 0.20
SPREAD_MAX = 2.0


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return clamp_rate(100.0 / (1.0 + math.exp(-lg)))
    e = math.exp(lg)
    return clamp_rate(100.0 * e / (1.0 + e))


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _linfit2(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b). Degenerate input -> (mean y, 0)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def fit_joins(times, ns):
    """Fit J(t) = J0 + A * (1 - exp(-(t/tau)**k)) to distinct reads.

    Returns a function J(t) or None when the history cannot support a fit.
    """
    if len(times) < 3 or max(ns) <= min(ns):
        return None
    best = None  # (score, tau, k, J0, A)
    for tau in TAUS:
        for k in KS:
            f = [1.0 - math.exp(-((t / tau) ** k)) if t > 0 else 0.0 for t in times]
            j0, a = _linfit2(f, ns)
            if a <= 0:
                continue
            j0 = max(0.0, j0)
            score = 0.0
            for t, n in zip(times, ns):
                pred = j0 + a * (1.0 - math.exp(-((t / tau) ** k)) if t > 0 else 0.0)
                score += (math.log(max(pred, 1.0)) - math.log(max(n, 1.0))) ** 2
            if best is None or score < best[0]:
                best = (score, tau, k, j0, a)
    if best is None:
        return None
    # One local refinement pass around the winning (tau, k).
    _, tau0, k0, _, _ = best
    for tau in [tau0 * f for f in (0.7, 0.85, 1.0, 1.18, 1.4)]:
        for k in [max(0.2, k0 + dk) for dk in (-0.15, -0.07, 0.0, 0.07, 0.15)]:
            f = [1.0 - math.exp(-((t / tau) ** k)) if t > 0 else 0.0 for t in times]
            j0, a = _linfit2(f, ns)
            if a <= 0:
                continue
            j0 = max(0.0, j0)
            score = 0.0
            for t, n in zip(times, ns):
                pred = j0 + a * (1.0 - math.exp(-((t / tau) ** k)) if t > 0 else 0.0)
                score += (math.log(max(pred, 1.0)) - math.log(max(n, 1.0))) ** 2
            if score < best[0]:
                best = (score, tau, k, j0, a)
    _, tau, k, j0, a = best

    def J(t):
        if t <= 0:
            return j0
        return j0 + a * (1.0 - math.exp(-((t / tau) ** k)))

    return J


# ---------------------------------------------------------------------------
# Description priors (see the module docstring for the reasoning).
# ---------------------------------------------------------------------------
_GRIND_RE = re.compile(r"(\d+)\s*(stunts?|locations?|challenges?|sets?|vehicles?|races?|vaults?|shops?|properties|animals?)")


def description_prior(trophy):
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "bronze")
    if tier == "platinum" or "100% completion" in text:
        return 0.02, 200.0
    m = re.search(r"complete chapter (\d)", text)
    if m:
        n = int(m.group(1))
        return (0.65, 8.0) if n == 1 else (0.55, 14.0) if n == 2 else (0.48, 20.0) if n == 3 else (0.42, 26.0)
    if "complete the final mission." in text or text.startswith("complete the final mission"):
        return 0.30, 45.0
    if "complete the final mission for" in text:
        return 0.18, 40.0
    if "mission strands for strangers" in text:
        return 0.10, 70.0
    if text.startswith("complete ") and ("run" in text or "heist" in text or "play" in text or "takedown" in text or "glory" in text):
        return 0.40, 28.0
    if "gold medal in all" in text:
        return 0.04, 100.0
    g = _GRIND_RE.search(text)
    if g:
        count = int(g.group(1))
        p = 0.20 if count <= 10 else 0.12 if count <= 50 else 0.08 if count <= 100 else 0.05
        return p, 20.0 + 0.6 * count
    if tier == "gold":
        return 0.06, 70.0
    if tier == "silver":
        return 0.12, 55.0
    return 0.25, 25.0


def _distinct_reads(case):
    """Deduplicated reads: (times_h, players, rates) with times in hours
    since the first read. Consecutive identical pages carry no new signal."""
    history = case.get("history") or []
    t0 = parse_time(history[0]["at"])
    times, ns, rates = [], [], []
    prev_key = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        times.append((parse_time(h["at"]) - t0).total_seconds() / 3600.0)
        ns.append(float(h["players_tracked"]))
        rates.append(list(h["rates"]))
    return times, ns, rates, t0


def structural_view(case, times, ns, rates, target_h):
    """Join/earn medians + per-trophy residual sds. Returns
    (medians, resid_sds, prior_flags) or None when the history cannot
    support the fit."""
    J = fit_joins(times, ns)
    if J is None:
        return None
    # Join-density weights on a 2h grid; the mass at s=0 is the J0 cohort.
    # The grid must reach the target, not just the last read: players who
    # join between the deadline and the target dilute H(T), and leaving
    # their mass out would overstate every trophy's median.
    t_max = max(target_h, max(times))
    grid = []
    s = 0.0
    while s <= t_max + 2.0:
        grid.append(s)
        s += 2.0
    weights = []
    prev_j = J(0.0)
    for g in grid[1:]:
        jv = J(g)
        weights.append(max(0.0, jv - prev_j))
        prev_j = jv
    w0 = J(0.0)

    def H(t, dmed, sig):
        if t <= 0:
            return 0.0
        jt = J(t)
        if jt <= 0:
            return 0.0
        ln_d = math.log(dmed)
        tot = w0 * _norm_cdf((math.log(t) - ln_d) / sig)
        for gj, wj in zip(grid[1:], weights):
            if gj >= t or wj <= 0:
                continue
            tot += wj * _norm_cdf((math.log(t - gj) - ln_d) / sig)
        return tot / jt

    # Precompute H on the read times and at the target, once per combo:
    # the per-trophy fit then only differs in p.
    combos = [(d, sg) for d in D_MED_GRID for sg in SIGMAS]
    h_reads = [[H(t, d, sg) for t in times] for (d, sg) in combos]
    h_target = [H(target_h, d, sg) for (d, sg) in combos]
    h_up = [H(target_h + 3.0, d, sg) for (d, sg) in combos]
    h_dn = [H(max(target_h - 3.0, 0.01), d, sg) for (d, sg) in combos]

    medians, sds, prior_flags, slopes = [], [], [], []
    for i, trophy in enumerate(case["trophies"]):
        earned = [0.0 if rates[j][i] is None else rates[j][i] * ns[j] / 100.0 for j in range(len(times))]
        e_max = max(earned) if earned else 0.0
        p_prior, d_prior = description_prior(trophy)
        if e_max < 1.0:
            # Nobody has earned it yet: the series is silent, use the prior.
            ci = min(range(len(combos)), key=lambda c: abs(combos[c][0] - d_prior))
            p, dmed, sig = p_prior, combos[ci][0], 0.7
            best_ci = ci
            prior_flags.append(True)
            sd = 0.0
        else:
            best = None
            for ci, (dmed, sig) in enumerate(combos):
                xs = [ns[j] * h_reads[ci][j] for j in range(len(times))]
                sxx = sum(x * x for x in xs)
                if sxx <= 0:
                    continue
                p = min(1.0, max(0.0, sum(x * y for x, y in zip(xs, earned)) / sxx))
                sc = sum((math.log(p * x + 0.5) - math.log(y + 0.5)) ** 2 for x, y in zip(xs, earned))
                if best is None or sc < best[0]:
                    best = (sc, p, ci)
            _, p, best_ci = best
            dmed, sig = combos[best_ci]
            if e_max < 5.0:
                # A handful of earners: shrink the noisy fit toward the prior.
                wgt = e_max / 5.0
                p = wgt * p + (1.0 - wgt) * p_prior
                prior_flags.append(True)
            else:
                prior_flags.append(False)
            # Logit-rate residuals of the fitted curve, for the spread.
            resids = []
            for j, t in enumerate(times):
                if ns[j] <= 0:
                    continue
                r_obs = 100.0 * earned[j] / ns[j]
                r_fit = 100.0 * p * h_reads[best_ci][j]
                if 0.0 < r_obs < 100.0 and 0.0 < r_fit < 100.0:
                    resids.append(logit(r_obs) - logit(r_fit))
            if len(resids) >= 4:
                m = sum(resids) / len(resids)
                sd = math.sqrt(sum((r - m) ** 2 for r in resids) / (len(resids) - 1))
            else:
                sd = 0.0
        m = clamp_rate(100.0 * p * h_target[best_ci])
        # Logit slope at the target (per hour), for refresh lumpiness.
        m_up = clamp_rate(100.0 * p * h_up[best_ci])
        m_dn = clamp_rate(100.0 * p * h_dn[best_ci])
        slopes.append(abs(logit(m_up) - logit(m_dn)) / 6.0)
        medians.append(m)
        sds.append(sd)
    return medians, sds, prior_flags, slopes


def trend_view(case, times, ns, rates, t0):
    """Logit-linear trend medians + spreads, capped at the prior eventual share."""
    rel_s = (case.get("game") or {}).get("release_date")
    try:
        rel = parse_time(rel_s) if rel_s and "T" in rel_s else datetime.fromisoformat(rel_s).replace(tzinfo=timezone.utc) if rel_s else t0
    except Exception:
        rel = t0
    target = parse_time(case["target"])
    tgt_h = (target - rel).total_seconds() / 3600.0
    horizon_d = case["horizon_minutes"] / 1440.0
    medians, spreads = [], []
    for i, trophy in enumerate(case["trophies"]):
        pts = []
        for j, t in enumerate(times):
            r = rates[j][i]
            if r is None or not (0.0 < r < 100.0):
                continue
            hrs = (t0.timestamp() + t * 3600.0 - rel.timestamp()) / 3600.0
            pts.append((math.log1p(max(hrs, 0.0)), logit(r)))
        if len(pts) >= 3:
            a, b = _linfit2([p[0] for p in pts], [p[1] for p in pts])
            resid = [y - (a + b * x) for x, y in pts]
            sd = math.sqrt(sum(r * r for r in resid) / len(resid)) if len(resid) > 1 else 0.0
            lg = a + b * math.log1p(max(tgt_h, 0.0))
            p_prior, _ = description_prior(trophy)
            # The eventual share is at least the share already earned.
            last_r = rates[-1][i] if rates else None
            cap_p = p_prior
            if last_r is not None and 0.0 < last_r < 100.0:
                cap_p = max(cap_p, last_r / 100.0)
            lg = min(lg, logit(100.0 * cap_p))
            s = math.sqrt((0.5 * math.sqrt(horizon_d)) ** 2 + sd ** 2)
        else:
            last = rates[-1][i] if rates else None
            lg = logit(last if last is not None else 0.0)
            s = 0.5 * math.sqrt(horizon_d)
        medians.append(clamp_rate(inv_logit(lg)))
        spreads.append(min(SPREAD_MAX, max(SPREAD_MIN, s)))
    return medians, spreads


def fallback_forecast(case, last_rates, n_last, target_h):
    """Flat-history dilution prior (see module docstring)."""
    n_target = n_last * (1.0 + target_h / FALLBACK_N_GROWTH_PER_HOUR)
    n_new = max(0.0, n_target - n_last)
    earned_last = [0.0 if r is None else r * n_last / 100.0 for r in last_rates]
    full_idx = [i for i, r in enumerate(last_rates) if r is not None and r >= 99.99]
    rest_total = sum(e for i, e in enumerate(earned_last) if i not in full_idx)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last_rates[i]
        if i in full_idx:
            rows.append((trophy["id"], 99.99, 0.05))
            continue
        e = earned_last[i]
        if rest_total > 0:
            e_new = e + FALLBACK_NEW_TROPHIES_EACH * n_new * (e / rest_total)
        else:
            e_new = e
        m = clamp_rate(100.0 * e_new / n_target)
        rows.append((trophy["id"], m, FALLBACK_SPREAD))
    return rows


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("no history")
    t0 = parse_time(history[0]["at"])
    target = parse_time(case["target"])
    target_h = (target - t0).total_seconds() / 3600.0
    times, ns, rates, _ = _distinct_reads(case)
    last_rates = list(history[-1]["rates"])
    n_last = float(history[-1]["players_tracked"])

    rows = None
    if len(times) >= 4 and max(ns) > min(ns) * 1.05:
        struct = structural_view(case, times, ns, rates, target_h)
        if struct is not None:
            s_meds, s_sds, prior_flags, slopes = struct
            t_meds, t_spreads = trend_view(case, times, ns, rates, t0)
            gaps = [b - a for a, b in zip(times, times[1:]) if b - a > 0]
            gap_med = median(gaps) if gaps else 4.0
            pool = median([s for s in s_sds if s > 0]) if any(s > 0 for s in s_sds) else 0.30
            rows = []
            for i, trophy in enumerate(case["trophies"]):
                # Ensemble median: logit-average of the two views.
                m = clamp_rate(inv_logit(0.5 * (logit(s_meds[i]) + logit(t_meds[i]))))
                n_i = len(times)
                sd_i = math.sqrt((n_i * s_sds[i] ** 2 + 4 * pool ** 2) / (n_i + 4)) if s_sds[i] > 0 else pool
                disagree = 0.5 * abs(logit(s_meds[i]) - logit(t_meds[i]))
                s = math.sqrt(sd_i ** 2 + (slopes[i] * gap_med / 2.0) ** 2 + disagree ** 2 + MODEL_SPREAD ** 2)
                s = max(s, t_spreads[i] * 0.5)
                if prior_flags[i]:
                    s = SPREAD_MAX
                s = min(SPREAD_MAX, max(SPREAD_MIN, s))
                rows.append((trophy["id"], m, s))
    if rows is None:
        rows = fallback_forecast(case, last_rates, n_last, target_h)
    out = []
    for tid, m, s in rows:
        lg = logit(m)
        q = [round4(inv_logit(lg + z * s)) for z in Z]
        q = [min(99.99, max(0.01, v)) for v in q]
        q = sorted(q)
        out.append({"id": tid, "q": q})
    if len(out) != len(case["trophies"]):
        raise ValueError("row count mismatch")
    return {"trophies": out}


def safe_forecast_case(case):
    """Never crash a case: fall back to wide persistence on any surprise."""
    try:
        return forecast_case(case)
    except Exception:
        rows = []
        last = (case.get("history") or [{}])[-1]
        last_rates = last.get("rates") or []
        for i, trophy in enumerate(case["trophies"]):
            r = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 1.0
            lg = logit(r)
            q = [round4(inv_logit(lg + z * 1.0)) for z in Z]
            rows.append({"id": trophy["id"], "q": sorted(min(99.99, max(0.01, v)) for v in q)})
        return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = safe_forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
