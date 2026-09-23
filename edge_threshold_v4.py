"""
ITEM 2: where does a shrunk edge clear the vig?

The harness established that beta is about 0.229 pooled, so roughly a quarter
of the model's disagreement with the line is real information. That tells you
HOW MUCH to shrink but not WHERE a shrunk edge is still big enough to be worth
betting after the hold. This script answers that, which is the number that
turns a measurement into a product: loud when there is something to say,
silent otherwise.

THE PIPELINE, WHICH IS PHASE 5.1 THROUGH 5.3 APPLIED

  1. blended mean = line_consensus + alpha + beta * (projection - line_consensus)
     Consensus is the anchor because it is the better estimate of the truth.
     alpha is the mean-versus-median offset per market; adding it is what puts
     the distribution's MEDIAN near the line, which is why a correctly priced
     prop comes out near P(over) = 0.5 rather than 0.5 plus the skew.
  2. distribution around that mean: gamma for yardage, negative binomial for
     receptions, moment-matched to (blended mean, residual SD). The residual
     SD comes from the regression, so it is the spread around the BLENDED
     mean, not around the model's own projection the way the shipped sigma is.
  3. P(over) evaluated at the line you would actually bet, FanDuel by default.
  4. edge = P(model) - P(breakeven implied by the actual FanDuel odds), per
     side. The better side is taken. Both sides negative means no bet.
  5. realized profit per unit staked using the real American odds.
  6. bucket by edge, and report the cumulative "bet everything above t" curve.

THREE WAYS THIS COULD LIE, AND WHAT IS DONE ABOUT EACH

  A. CIRCULAR SHRINKAGE. Fitting beta and alpha on the same rows being priced
     makes the tail look good by construction, the same defect as the old
     in-sample RMSE column. So parameters are fitted LEAVE ONE SEASON OUT:
     each season is priced with alpha and beta estimated from the other
     seasons only. With a single season available the script falls back to
     5-fold CV grouped on event_id and says so.

  B. THRESHOLD SELECTION. Scanning thirty cutoffs and reporting the best is
     selection on noise. Guards: every row of the cumulative curve carries a
     game-clustered standard error, and a separate HOLDOUT pass picks the
     threshold on the other seasons and reports what it actually earned on the
     held-out season. The holdout number is the only one to believe.

  C. SURVIVORSHIP IN THE INSTRUMENT. rushing joins at only 47 percent because
     models.rushing.build_dataset filters carries >= 5 before the harness sees
     a row, so the rushing sample conditions on the player having got work and
     its over rate of 0.538 is almost certainly inflated. A pooled row
     excluding rushing is printed alongside the pooled row, and rushing is
     flagged wherever it appears. Recompute after Phase 3.1.

Run from the repo root with the venv active:
    python edge_threshold.py --all-seasons --cache lines_cache.parquet
    python edge_threshold.py --all-seasons --cache lines_cache.parquet --line-source best
    python edge_threshold.py --all-seasons --cache lines_cache.parquet --exclude rushing

Reuses eval_harness for credentials, fetching, the cache, the collapse and the
clustered estimators, so there is one implementation of each.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

import eval_harness as eh

# markets whose outcome is a continuous right-skewed yardage total
YARDAGE = ("receiving", "rushing", "qb_passing", "qb_rushing")
# markets whose outcome is a small count
COUNT = ("receptions",)

DEFAULT_ODDS = -113.0
GRID = np.round(np.arange(0.0, 0.2501, 0.005), 4)


# ------------------------------------------------------------------ odds math

def implied_breakeven(odds):
    """American odds to the win probability needed to break even."""
    o = np.asarray(odds, float)
    out = np.full(o.shape, np.nan)
    neg = o < 0
    pos = o > 0
    out[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    out[pos] = 100.0 / (o[pos] + 100.0)
    return out


def payout(odds):
    """Profit per 1 unit staked on a win."""
    o = np.asarray(odds, float)
    out = np.full(o.shape, np.nan)
    neg = o < 0
    pos = o > 0
    out[neg] = 100.0 / (-o[neg])
    out[pos] = o[pos] / 100.0
    return out


# --------------------------------------------------------------- distribution

def p_over(market, mean, sd, line):
    """P(actual > line) for a distribution centred on `mean` with spread `sd`.

    Gamma for yardage: moment matched, shape = (mean/sd)^2, scale = sd^2/mean.
    This is what carries the right skew, and it is why a prop priced with
    alpha applied lands near 0.5 rather than above it.

    Receptions: negative binomial where the variance exceeds the mean, Poisson
    where it does not. residSD for receptions is close to sqrt(mean), so the
    count is near-Poisson and a negative binomial is not always defined. The
    line is a half integer so P(over 4.5) is P(X >= 5).
    """
    mean = np.asarray(mean, float)
    sd = np.asarray(sd, float)
    line = np.asarray(line, float)
    out = np.full(mean.shape, np.nan)
    ok = np.isfinite(mean) & np.isfinite(line) & (mean > 1e-6) & (sd > 1e-6)
    if not ok.any():
        return out

    if market in COUNT:
        m = mean[ok]
        v = sd[ok] ** 2
        k = np.floor(line[ok])          # P(X > 4.5) = P(X >= 5) = sf(4)
        res = np.empty(m.shape)
        nb = v > m * 1.0001
        if nb.any():
            # negative binomial with mean m, variance v
            p = m[nb] / v[nb]
            r = m[nb] * p / (1.0 - p)
            res[nb] = stats.nbinom.sf(k[nb], r, p)
        if (~nb).any():
            res[~nb] = stats.poisson.sf(k[~nb], m[~nb])
        out[ok] = res
        return out

    shape = (mean[ok] / sd[ok]) ** 2
    scale = sd[ok] ** 2 / mean[ok]
    out[ok] = stats.gamma.sf(line[ok], a=shape, scale=scale)
    return out


# ----------------------------------------------------------------- parameters

def fit_params(sub, min_rows=200):
    """alpha and beta per market, fitted LEAVE ONE SEASON OUT.

    Returns {(market, season): (alpha, beta, resid_sd, source)}. Pricing a
    season with parameters estimated from that same season would make the tail
    look good by construction, which is the defect this avoids.
    """
    params = {}
    seasons = sorted(sub["season"].unique())
    for m, g in sub.groupby("market"):
        full = eh.ols_clustered(g["dev_cons"], g["out_cons"], g["event_id"])
        if full is None:
            continue
        if len(seasons) > 1:
            for s in seasons:
                other = g[g["season"] != s]
                if len(other) < min_rows:
                    params[(m, s)] = (full["alpha"], full["beta"],
                                      full["resid_sd"], "all seasons")
                    continue
                f = eh.ols_clustered(other["dev_cons"], other["out_cons"],
                                     other["event_id"])
                if f is None:
                    params[(m, s)] = (full["alpha"], full["beta"],
                                      full["resid_sd"], "all seasons")
                else:
                    params[(m, s)] = (f["alpha"], np.clip(f["beta"], 0.0, 1.0),
                                      f["resid_sd"], "leave-season-out")
        else:
            # one season only: 5-fold grouped on event_id
            fold = eh._folds(g["event_id"].to_numpy(), 5)
            fits = []
            for k in range(5):
                tr = fold != k
                if tr.sum() < min_rows:
                    continue
                f = eh.ols_clustered(g["dev_cons"].to_numpy()[tr],
                                     g["out_cons"].to_numpy()[tr],
                                     g["event_id"].to_numpy()[tr])
                if f is not None:
                    fits.append((f["alpha"], np.clip(f["beta"], 0, 1),
                                 f["resid_sd"]))
            if fits:
                a = float(np.mean([f[0] for f in fits]))
                b = float(np.mean([f[1] for f in fits]))
                r = float(np.mean([f[2] for f in fits]))
                params[(m, seasons[0])] = (a, b, r, "5-fold (single season)")
            else:
                params[(m, seasons[0])] = (full["alpha"], full["beta"],
                                           full["resid_sd"], "all seasons")
    return params


def fit_sigma_sqrt(sub, params, min_rows=300):
    """sigma = a * sqrt(mean), with the exponent IMPOSED rather than fitted.

    WHY. For a count outcome the variance is roughly proportional to the mean
    (Poisson, and negative binomial with modest overdispersion), so the
    standard deviation goes as the square root and b = 0.5 by theory. The
    full-market receptions fits came back at 0.473, 0.480, 0.532 and 0.522,
    which is 0.5 within noise, so this is not an assumption being smuggled in.

    The practical reason it matters: fitting b needs LEVEL SPAN, and a
    restricted stratum does not have any. Pricing only the 2.5 and 3.5 lines
    left blended means spanning about 2.2 to 4.5, a ratio of 2.0, which fails
    the 3x span gate and falls back to a FLAT sigma. Imposing the exponent
    needs no span at all, so a stratified run keeps its level dependence.

    Returns {(market, season): (a, 0.5)}, leave-season-out.
    """
    out = {}
    seasons = sorted(sub["season"].unique())

    def fit(frame, market):
        rows = []
        for s in seasons:
            key = (market, s)
            if key not in params:
                continue
            a, b, _r, _src = params[key]
            g = frame[frame["season"] == s]
            if not len(g):
                continue
            blend = (g["line_consensus"] + a
                     + b * (g["projection"] - g["line_consensus"]))
            rows.append(pd.DataFrame({"blend": blend.to_numpy(),
                                      "actual": g["actual"].to_numpy()}))
        if not rows:
            return None
        d = pd.concat(rows, ignore_index=True)
        d = d[np.isfinite(d["blend"]) & (d["blend"] > 1e-6)]
        if len(d) < min_rows:
            return None
        resid = d["actual"] - d["blend"]
        # a is the scale such that sd = a * sqrt(mean); estimate it from the
        # standardised residuals so every level contributes equally
        z = resid / np.sqrt(d["blend"])
        a_hat = float(np.sqrt(np.mean(z ** 2)))
        return (a_hat, 0.5) if np.isfinite(a_hat) and a_hat > 0 else None

    for m, g in sub.groupby("market"):
        full = fit(g, m)
        for s in seasons:
            other = g[g["season"] != s]
            r = fit(other, m) if len(other) >= min_rows else None
            if r is None:
                r = full
            if r is not None:
                out[(m, s)] = r
    return out


def fit_sigma_level(sub, params, min_rows=400, nbuckets=10):
    """sigma as a power function of the blended mean, fitted per market.

    WHY THIS EXISTS. Using the regression residual SD gives ONE scalar per
    market, so receiving prices every prop with a spread of 27.1 yards. The
    outcome is strongly heteroskedastic: the shipped sigma is
    3.272 * proj^0.6172, which is 34.7 at a 50-yard projection and about 49 at
    90. A flat sigma is therefore far too narrow exactly where the large
    deviations live, which produces extreme probabilities on BOTH sides and
    the monotonically growing overconfidence seen in the calibration table.

    Fits log(sd) = log(a) + b * log(mean) on count-weighted level buckets,
    leave-season-out so the sigma pricing a season never saw that season.
    Returns {(market, season): (a, b)}.
    """
    out = {}
    seasons = sorted(sub["season"].unique())

    def fit(frame, market):
        rows = []
        for s in seasons:
            key = (market, s)
            if key not in params:
                continue
            a, b, _r, _src = params[key]
            g = frame[frame["season"] == s]
            if not len(g):
                continue
            blend = (g["line_consensus"] + a
                     + b * (g["projection"] - g["line_consensus"]))
            rows.append(pd.DataFrame({"blend": blend.to_numpy(),
                                      "actual": g["actual"].to_numpy()}))
        if not rows:
            return None
        d = pd.concat(rows, ignore_index=True)
        d = d[np.isfinite(d["blend"]) & (d["blend"] > 1e-6)]
        if len(d) < min_rows:
            return None
        d["resid"] = d["actual"] - d["blend"]
        try:
            d["bucket"] = pd.qcut(d["blend"], nbuckets, duplicates="drop")
        except Exception:
            return None
        agg = d.groupby("bucket", observed=True).agg(
            m=("blend", "median"), sd=("resid", "std"), n=("resid", "size"))
        agg = agg[(agg["n"] >= 25) & (agg["sd"] > 1e-6) & (agg["m"] > 1e-6)]
        if len(agg) < 4:
            return None
        # A power law needs leverage in the regressor. Every QB projects into a
        # narrow 180 to 300 band, so log(mean) barely varies and the fit is
        # unidentified: it returned a = 55, 128, 27, 67 with b near zero across
        # four seasons, and in one of them assigned sigma 127.7 against a flat
        # 70. Require the top bucket to be at least 3x the bottom, else fall
        # back to the flat residual SD.
        span = float(agg["m"].max() / max(agg["m"].min(), 1e-9))
        if span < 3.0:
            return None
        X = np.column_stack([np.ones(len(agg)), np.log(agg["m"].to_numpy())])
        y = np.log(agg["sd"].to_numpy())
        w = np.sqrt(agg["n"].to_numpy())
        coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        a_hat, b_hat = float(np.exp(coef[0])), float(coef[1])
        if not (np.isfinite(a_hat) and np.isfinite(b_hat)) or a_hat <= 0:
            return None
        return a_hat, float(np.clip(b_hat, 0.0, 1.5))

    for m, g in sub.groupby("market"):
        full = fit(g, m)
        for s in seasons:
            other = g[g["season"] != s]
            r = fit(other, m) if len(other) >= min_rows else None
            if r is None:
                r = full
            if r is not None:
                out[(m, s)] = r
    return out


def report_sigma(slevel, params):
    print("\n  LEVEL-DEPENDENT SIGMA  sigma = a * mean^b  (out of sample)")
    print(f"  {'market':<14}{'season':>8}{'a':>9}{'b':>8}"
          f"{'flat residSD':>14}")
    for (m, s), (a, b) in sorted(slevel.items() if slevel else []):
        flat = params.get((m, s), (0, 0, np.nan, ""))[2]
        print(f"  {m:<14}{int(s):>8}{a:>9.3f}{b:>8.4f}{flat:>14.1f}")
    missing = sorted({k for k in params if k not in (slevel or {})})
    for m, s in missing:
        flat = params[(m, s)][2]
        print(f"  {m:<14}{int(s):>8}{'flat':>9}{'-':>8}{flat:>14.1f}"
              f"   too little level span to identify a power law")


def report_params(params):
    print("\n  SHRINKAGE PARAMETERS (out of sample)")
    print(f"  {'market':<14}{'season':>8}{'alpha':>9}{'beta':>8}"
          f"{'residSD':>10}   source")
    for (m, s), (a, b, r, src) in sorted(params.items()):
        print(f"  {m:<14}{int(s):>8}{a:>+9.2f}{b:>8.3f}{r:>10.1f}   {src}")


# --------------------------------------------------------------- price a frame

def price(sub, params, line_source, slevel=None, sigma_mode="flat",
          sigma_scale=1.0):
    """Attach the model probability, the chosen side, the edge and the profit.

    sigma_mode 'flat'  uses the regression residual SD, one scalar per market.
    sigma_mode 'level' uses a * blend^b from fit_sigma_level, which is what a
                       heteroskedastic outcome requires.
    sigma_scale multiplies whichever sigma is chosen, for the calibration scan.
    """
    col = f"line_{line_source}"
    s = sub.dropna(subset=[col, "line_consensus", "projection", "actual"]).copy()

    key = list(zip(s["market"], s["season"]))
    s["alpha"] = [params.get(k, (np.nan,) * 4)[0] for k in key]
    s["beta"] = [params.get(k, (np.nan,) * 4)[1] for k in key]
    s["sigma"] = [params.get(k, (np.nan,) * 4)[2] for k in key]
    s = s.dropna(subset=["alpha", "beta", "sigma"])

    s["blend"] = (s["line_consensus"] + s["alpha"]
                  + s["beta"] * (s["projection"] - s["line_consensus"]))

    if sigma_mode == "level" and slevel:
        lev = np.full(len(s), np.nan)
        keys = list(zip(s["market"], s["season"]))
        blend = s["blend"].to_numpy()
        flat = s["sigma"].to_numpy()
        for i, k in enumerate(keys):
            if k in slevel and np.isfinite(blend[i]) and blend[i] > 1e-6:
                a, b = slevel[k]
                lev[i] = a * blend[i] ** b
        # clip to a sane multiple of the flat estimate so a bad power fit on
        # one bucket cannot produce an absurd spread
        lev = np.where(np.isfinite(lev),
                       np.clip(lev, 0.25 * flat, 4.0 * flat), flat)
        s["sigma"] = lev
    s["sigma"] = s["sigma"] * float(sigma_scale)

    s["bet_line"] = s[col]
    po = np.full(len(s), np.nan)
    for m, idx in s.groupby("market").indices.items():
        po[idx] = p_over(m, s["blend"].to_numpy()[idx],
                         s["sigma"].to_numpy()[idx],
                         s["bet_line"].to_numpy()[idx])
    s["p_over"] = po
    s = s.dropna(subset=["p_over"])

    oo = pd.to_numeric(s.get("over_odds"), errors="coerce").fillna(DEFAULT_ODDS)
    uo = pd.to_numeric(s.get("under_odds"), errors="coerce").fillna(DEFAULT_ODDS)
    s["be_over"] = implied_breakeven(oo)
    s["be_under"] = implied_breakeven(uo)
    s["pay_over"] = payout(oo)
    s["pay_under"] = payout(uo)

    s["edge_over"] = s["p_over"] - s["be_over"]
    s["edge_under"] = (1.0 - s["p_over"]) - s["be_under"]
    s["side_over"] = s["edge_over"] >= s["edge_under"]
    s["edge"] = np.where(s["side_over"], s["edge_over"], s["edge_under"])

    push = np.isclose(s["actual"], s["bet_line"])
    won = np.where(s["side_over"], s["actual"] > s["bet_line"],
                   s["actual"] < s["bet_line"])
    pay = np.where(s["side_over"], s["pay_over"], s["pay_under"])
    s["push"] = push
    s["won"] = won
    s["profit"] = np.where(push, 0.0, np.where(won, pay, -1.0))
    return s


# ------------------------------------------------- winner's curse correction

def curse_correct(s, min_rows=300):
    """Recalibrate the CHOSEN side's probability, leave-season-out.

    WHY. The unconditional calibration is close to unbiased (+0.2 pp overall
    on the 2.5-3.5 receptions stratum), but the conditional table, which
    conditions on the side the model picked, runs -4.5 to -12.5 points across
    the bands. That gap is not a miscalibration of P(over). It is SELECTION:
    picking the larger of two noisy edges systematically picks the estimate
    that is too extreme, and the bias grows with the displayed edge.

    Measured at roughly 4.3 points of win probability pooled, which is LARGER
    THAN THE VIG. So correcting it is not a refinement, it is potentially the
    difference between a losing and a winning rule.

    The correction fits, on OTHER seasons only:

        won = a + b * (p_bet - 0.5)

    over the post-selection rows, then maps this season's chosen probability
    through it. b below 1 shrinks toward 0.5, which is the expected direction.
    The side is NOT re-chosen, because the curse is a property of having
    already selected, and re-picking on corrected numbers would just move the
    selection problem one step along.
    """
    s = s.copy()
    live = s[~s["push"]]
    p_bet_all = np.where(s["side_over"], s["p_over"], 1.0 - s["p_over"])
    s["p_bet_raw"] = p_bet_all

    print("\n  WINNER'S CURSE CORRECTION (leave-season-out)")
    print(f"  {'market':<14}{'season':>8}{'a':>9}{'b':>8}{'n_train':>9}"
          f"{'shrink at p=0.60':>18}")
    adj = np.full(len(s), np.nan)
    for (m, season), idx in s.groupby(["market", "season"]).indices.items():
        tr = live[(live["market"] == m) & (live["season"] != season)]
        if len(tr) < min_rows:
            continue
        x = (np.where(tr["side_over"], tr["p_over"], 1.0 - tr["p_over"])
             - 0.5)
        y = tr["won"].to_numpy(float)
        f = eh.ols_clustered(x, y, tr["event_id"].to_numpy())
        if f is None:
            continue
        a_c, b_c = f["alpha"], f["beta"]
        adj[idx] = a_c + b_c * (p_bet_all[idx] - 0.5)
        at60 = a_c + b_c * 0.10
        print(f"  {m:<14}{int(season):>8}{a_c:>9.4f}{b_c:>8.3f}{f['n']:>9}"
              f"{at60:>18.4f}")
    s["p_bet_adj"] = np.where(np.isfinite(adj), adj, p_bet_all)
    n_adj = int(np.isfinite(adj).sum())
    print(f"    corrected {n_adj} of {len(s)} rows; "
          f"b below 1.0 means shrinkage toward 0.5")

    be = np.where(s["side_over"], s["be_over"], s["be_under"])
    s["edge_raw"] = s["edge"]
    s["edge"] = s["p_bet_adj"] - be
    print(f"    edge median {s['edge_raw'].median():+.4f} -> "
          f"{s['edge'].median():+.4f}, share positive "
          f"{100 * (s['edge_raw'] > 0).mean():.1f}% -> "
          f"{100 * (s['edge'] > 0).mean():.1f}%")
    return s


# -------------------------------------------------------------- calibration

def calibration_unconditional(s):
    """Calibration with NO side selection: every prop, bucketed by P(over).

    The conditional table below picks the side with the larger edge, which is
    the larger of two noisy estimates, so realized performance sits below
    predicted even under perfect pricing. That is a winner's curse in the
    probability estimate, the same effect the bets table showed at -10.2 pp
    when conditioned on the model's chosen side.

    This table conditions on nothing, so a gap here is a genuine pricing
    error. The DIFFERENCE between the two tables is the curse.
    """
    print("\n  CALIBRATION, UNCONDITIONAL (no side selection)")
    live = s[~s["push"]].copy()
    over_won = (live["actual"] > live["bet_line"]).astype(float)
    live["over_won"] = over_won
    bands = [(0.00, 0.35), (0.35, 0.45), (0.45, 0.50), (0.50, 0.55),
             (0.55, 0.65), (0.65, 1.00)]
    print(f"  {'P(over) band':<14}{'n':>7}{'predicted':>11}{'actual':>9}"
          f"{'SE':>8}{'gap (pp)':>10}{'z':>7}")
    for lo, hi in bands:
        g = live[(live["p_over"] >= lo) & (live["p_over"] < hi)]
        if len(g) < 30:
            continue
        cm = eh.cluster_mean(g["over_won"].to_numpy(), g["event_id"].to_numpy())
        if cm is None:
            continue
        pred = float(g["p_over"].mean())
        gap = cm["mean"] - pred
        z = gap / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{cm['n']:>7}{pred:>11.4f}"
              f"{cm['mean']:>9.4f}{cm['se']:>8.4f}{100 * gap:>+10.1f}"
              f"{z:>+7.2f}")
    cm = eh.cluster_mean(live["over_won"].to_numpy(), live["event_id"].to_numpy())
    if cm is not None:
        pred = float(live["p_over"].mean())
        gap = cm["mean"] - pred
        print(f"  {'OVERALL':<14}{cm['n']:>7}{pred:>11.4f}{cm['mean']:>9.4f}"
              f"{cm['se']:>8.4f}{100 * gap:>+10.1f}"
              f"{gap / cm['se'] if cm['se'] > 0 else np.nan:>+7.2f}")
    print("    A gap here is a real pricing error. A gap in the CONDITIONAL")
    print("    table that is larger is the winner's curse from side selection.")


def calibration(s):
    """Does the priced probability match the realized rate?

    This is the sanity check on the whole pipeline. If the blended mean plus
    alpha plus the gamma is right, a bucket priced at 0.55 should win near 55
    percent of the time. A systematic gap means the distribution or the
    residual SD is wrong and the threshold below is measuring the wrong thing.
    """
    print("\n  CALIBRATION OF THE PRICED PROBABILITY")
    live = s[~s["push"]].copy()
    p = np.where(live["side_over"], live["p_over"], 1.0 - live["p_over"])
    live["p_bet"] = p
    bands = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
             (0.60, 0.70), (0.70, 1.00)]
    print(f"  {'band':<14}{'n':>7}{'predicted':>11}{'actual':>9}{'SE':>8}"
          f"{'gap (pp)':>10}{'z':>7}")
    for lo, hi in bands:
        g = live[(live["p_bet"] >= lo) & (live["p_bet"] < hi)]
        if len(g) < 30:
            continue
        cm = eh.cluster_mean(g["won"].astype(float).to_numpy(),
                             g["event_id"].to_numpy())
        if cm is None:
            continue
        pred = float(g["p_bet"].mean())
        gap = cm["mean"] - pred
        z = gap / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{cm['n']:>7}{pred:>11.4f}"
              f"{cm['mean']:>9.4f}{cm['se']:>8.4f}{100 * gap:>+10.1f}"
              f"{z:>+7.2f}")


# ------------------------------------------------- zero-deviation pricing

def zero_dev_check(s):
    """What the pricing layer says when the model agrees with the line exactly.

    At dev = 0 the blended mean is just line + alpha, so this isolates the
    SKEW correction from anything the model contributes. Compare the average
    priced P(over) against the realized over rate on the same rows.

    A priced P(over) BELOW the realized rate means the gamma is over-skewed
    (or alpha is under-applied), and the layer carries a structural under-lean
    before the model says anything. That bias is present on every prop, so it
    contaminates every edge and every threshold.

    The fix is not a fudge: choose alpha so the priced P(over) at dev = 0
    matches the measured over rate, i.e. calibrate alpha to the MEDIAN rather
    than reading it off the regression mean offset.
    """
    print("\n  PRICING AT ZERO DEVIATION (skew check, model contributes nothing)")
    print(f"  {'market':<14}{'n':>7}{'priced p_over':>15}{'actual over':>13}"
          f"{'gap (pp)':>10}{'SE':>8}{'z':>7}{'alpha used':>12}")
    for m, g in s.groupby("market"):
        line = g["bet_line"].to_numpy(float)
        alpha = g["alpha"].to_numpy(float)
        sig = g["sigma"].to_numpy(float)
        blend0 = line + alpha
        p0 = p_over(m, blend0, sig, line)
        ok = np.isfinite(p0)
        if ok.sum() < 30:
            continue
        won = (g["actual"].to_numpy(float)[ok] > line[ok]).astype(float)
        cm = eh.cluster_mean(won, g["event_id"].to_numpy()[ok])
        if cm is None:
            continue
        pred = float(np.mean(p0[ok]))
        gap = cm["mean"] - pred
        z = gap / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {m:<14}{int(ok.sum()):>7}{pred:>15.4f}{cm['mean']:>13.4f}"
              f"{100 * gap:>+10.1f}{cm['se']:>8.4f}{z:>+7.2f}"
              f"{float(np.mean(alpha)):>+12.2f}")
    print("    A positive gap means the layer under-prices the over on every")
    print("    prop, which is a structural lean the model never asked for.")


# ------------------------------------------------------------- sigma scan

def sigma_scan(joined, params, line_source, slevel, sigma_mode,
               grid=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5,
                     1.6, 1.8, 2.0)):
    """Find the sigma multiplier that actually calibrates the probabilities.

    A single number that quantifies how far off the spread is. If the overall
    calibration gap closes at 1.3, the distributions are 30 percent too narrow
    and the threshold analysis cannot be trusted until that is fixed at the
    source rather than by a fudge factor.

    'share +edge' is the second diagnostic: against a roughly 6 percent hold a
    calibrated layer should find positive edge on a minority of props, so a
    figure near 60 percent is itself proof of overconfidence.
    """
    print("\n" + "=" * 96)
    print(f"SIGMA MULTIPLIER SCAN (sigma_mode={sigma_mode})")
    print("=" * 96)
    print(f"  {'mult':>6}{'n':>8}{'predicted':>11}{'actual':>9}"
          f"{'gap (pp)':>10}{'z':>8}{'share +edge':>13}{'ROI at 0':>10}"
          f"{'ROI at .10':>12}")
    best = None
    for k in grid:
        s = price(joined, params, line_source, slevel, sigma_mode, k)
        live = s[~s["push"]]
        if len(live) < 100:
            continue
        p = np.where(live["side_over"], live["p_over"], 1.0 - live["p_over"])
        cm = eh.cluster_mean(live["won"].astype(float).to_numpy(),
                             live["event_id"].to_numpy())
        if cm is None:
            continue
        pred = float(np.mean(p))
        gap = cm["mean"] - pred
        z = gap / cm["se"] if cm["se"] > 0 else np.nan
        share = float((s["edge"] > 0).mean())
        r0 = float(live["profit"].mean())
        g10 = live[live["edge"] >= 0.10]
        r10 = float(g10["profit"].mean()) if len(g10) >= 30 else np.nan
        print(f"  {k:>6.2f}{len(live):>8}{pred:>11.4f}{cm['mean']:>9.4f}"
              f"{100 * gap:>+10.1f}{z:>+8.2f}{100 * share:>12.1f}%"
              f"{r0:>+10.4f}{r10:>+12.4f}")
        if best is None or abs(gap) < abs(best[1]):
            best = (k, gap)
    if best:
        print(f"\n    closest to calibrated at multiplier {best[0]:.2f} "
              f"(gap {100 * best[1]:+.1f} pp)")
        print("    A multiplier far from 1.00 means the SOURCE of sigma is "
              "wrong. Fix the")
        print("    functional form rather than shipping the fudge factor.")
    return best


# ----------------------------------------------------------- threshold curve

def curve(s, label, weeks):
    """Cumulative 'bet everything with edge above t' with clustered SEs."""
    print(f"\n  THRESHOLD CURVE: {label}")
    print(f"  {'min edge':>9}{'bets':>7}{'per wk':>8}{'win rate':>10}"
          f"{'ROI':>9}{'SE':>8}{'t':>7}{'95% CI on ROI':>20}")
    best = None
    for t in GRID:
        g = s[s["edge"] >= t]
        live = g[~g["push"]]
        if len(live) < 30:
            continue
        cm = eh.cluster_mean(live["profit"].to_numpy(), live["event_id"].to_numpy())
        if cm is None:
            continue
        wr = float(live["won"].mean())
        tstat = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        lo = cm["mean"] - 1.96 * cm["se"]
        hi = cm["mean"] + 1.96 * cm["se"]
        print(f"  {t:>9.3f}{len(live):>7}{len(live) / max(weeks, 1):>8.1f}"
              f"{wr:>10.4f}{cm['mean']:>+9.4f}{cm['se']:>8.4f}{tstat:>+7.2f}"
              f"  [{lo:>+7.4f},{hi:>+8.4f}]")
        if best is None or cm["mean"] > best[1]:
            best = (t, cm["mean"], len(live))
    if best:
        print(f"\n    highest ROI at min edge {best[0]:.3f}: "
              f"{best[1]:+.4f} on {best[2]} bets")
        if best[0] >= GRID[-1]:
            print("    WARNING: that is the top of the grid, so the search was "
                  "truncated. Raise GRID.")
        print("    THIS NUMBER IS SELECTED AND BIASED UPWARD. The holdout "
              "table below is the")
        print("    one to act on.")
    return best


# ------------------------------------------------------------- tail profile

def tail_profile(s, thresholds=(0.0, 0.05, 0.10, 0.15)):
    """What the bets above a cutoff actually LOOK like.

    Survivorship in a joined sample does not show up as a calibration failure,
    it shows up as profit, so the calibration table cannot detect it. What does
    detect it is the composition of the tail. If the profitable bets are all
    low-line overs in a market whose build_dataset filters on volume, the
    'edge' is the filter. If they are spread across the line distribution and
    split between sides, it is more likely real.
    """
    print("\n" + "=" * 96)
    print("TAIL PROFILE: what the bets above each cutoff consist of")
    print("=" * 96)
    print(f"  {'market':<13}{'min edge':>9}{'bets':>6}{'% over':>8}"
          f"{'line p10':>10}{'line p50':>10}{'line p90':>10}"
          f"{'mean |dev|':>12}{'mean sigma':>12}{'ROI':>9}")
    for m, g in s.groupby("market"):
        for t in thresholds:
            b = g[(g["edge"] >= t) & (~g["push"])]
            if len(b) < 20:
                continue
            ln = b["bet_line"].to_numpy(float)
            dev = np.abs(b["projection"] - b["line_consensus"]).to_numpy(float)
            print(f"  {m:<13}{t:>9.2f}{len(b):>6}"
                  f"{100 * b['side_over'].mean():>7.0f}%"
                  f"{np.quantile(ln, 0.10):>10.1f}"
                  f"{np.quantile(ln, 0.50):>10.1f}"
                  f"{np.quantile(ln, 0.90):>10.1f}"
                  f"{np.nanmean(dev):>12.1f}"
                  f"{b['sigma'].mean():>12.1f}"
                  f"{b['profit'].mean():>+9.4f}")
        print()
    print("    Compare the tail rows against the min-edge 0.00 row for the same")
    print("    market. A tail that collapses onto one side and one end of the")
    print("    line distribution is the signature of a filtered sample.")


# ---------------------------------------------------------------- the holdout

def holdout_by_market(s, weeks_by_season, min_rows=800):
    """Per-market holdout. The pooled version lets one market's tail choose a
    threshold that does not suit another, which is a silent averaging error
    when the markets have different edge scales."""
    for m, g in s.groupby("market"):
        if len(g) < min_rows:
            print(f"\n  {m}: {len(g)} rows, too few for a per-market holdout")
            continue
        print(f"\n  --- {m} ---")
        holdout(g, weeks_by_season, quiet=True)


def holdout(s, weeks_by_season, quiet=False):
    """Choose the threshold on other seasons, then spend it on the held out one.

    This is the only honest estimate of what a threshold rule would have
    earned, because the choice of threshold never sees the season it is
    evaluated on.
    """
    if not quiet:
        print("\n" + "=" * 96)
        print("HOLDOUT: threshold chosen on other seasons, "
              "evaluated on the held out one")
        print("=" * 96)
    seasons = sorted(s["season"].unique())
    if len(seasons) < 2:
        print("  needs at least two seasons")
        return
    print(f"  {'held out':>9}{'chosen t':>10}{'bets':>7}{'per wk':>8}"
          f"{'win rate':>10}{'ROI':>9}{'SE':>8}{'t':>7}")
    rows = []
    for s_out in seasons:
        tr = s[s["season"] != s_out]
        te = s[s["season"] == s_out]
        bt, bv = None, -np.inf
        for t in GRID:
            g = tr[(tr["edge"] >= t) & (~tr["push"])]
            if len(g) < 100:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[(te["edge"] >= bt) & (~te["push"])]
        cm = (eh.cluster_mean(g["profit"].to_numpy(), g["event_id"].to_numpy())
              if len(g) >= 20 else None)
        if cm is None:
            print(f"  {int(s_out):>9}{bt:>10.3f}{len(g):>7}"
                  f"   too few bets for a clustered SE")
            continue
        wr = float(g["won"].mean())
        tstat = cm["mean"] / cm["se"] if cm and cm["se"] > 0 else np.nan
        wk = max(weeks_by_season.get(s_out, 1), 1)
        flag = "  <- AT GRID MAX, search truncated" if bt >= GRID[-1] else ""
        print(f"  {int(s_out):>9}{bt:>10.3f}{len(g):>7}{len(g) / wk:>8.1f}"
              f"{wr:>10.4f}{cm['mean']:>+9.4f}{cm['se']:>8.4f}{tstat:>+7.2f}"
              f"{flag}")
        rows.append((s_out, bt, len(g), cm["mean"], g))
    if rows and not quiet:
        allg = pd.concat([r[4] for r in rows], ignore_index=True)
        cm = eh.cluster_mean(allg["profit"].to_numpy(),
                             allg["event_id"].to_numpy())
        tstat = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {'POOLED':>9}{'':>10}{len(allg):>7}{'':>8}"
              f"{allg['won'].mean():>10.4f}{cm['mean']:>+9.4f}"
              f"{cm['se']:>8.4f}{tstat:>+7.2f}")
        print("\n    A pooled holdout ROI whose CI excludes zero is the first "
              "honest evidence of")
        print("    a profitable rule in this project. A CI spanning zero means "
              "the threshold is")
        print("    not yet established and the app should stay silent.")
    elif rows:
        allg = pd.concat([r[4] for r in rows], ignore_index=True)
        cm = eh.cluster_mean(allg["profit"].to_numpy(),
                             allg["event_id"].to_numpy())
        if cm is not None:
            t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
            print(f"  {'POOLED':>9}{'':>10}{len(allg):>7}{'':>8}"
                  f"{allg['won'].mean():>10.4f}{cm['mean']:>+9.4f}"
                  f"{cm['se']:>8.4f}{t:>+7.2f}")


# ------------------------------------------------------------ stratum filter

def add_quartiles(j):
    """Line-level and player-frequency quartiles, computed WITHIN market.

    Same construction as attention_strata.py so the buckets mean the same
    thing in both scripts. B1 is the LOWEST quartile.
    """
    j = j.copy()
    freq = j.groupby("_key").size().rename("player_freq")
    j = j.merge(freq, left_on="_key", right_index=True, how="left")
    for col, name in (("line_consensus", "lineq"), ("player_freq", "freqq")):
        r = pd.Series(np.nan, index=j.index)
        for m, g in j.groupby("market"):
            if g[col].notna().sum() < 240:
                continue
            try:
                q = pd.qcut(g[col], 4, duplicates="drop")
            except Exception:
                continue
            r.loc[g.index] = q.cat.codes.replace(-1, np.nan).to_numpy() + 1
        j[name] = r
    return j


def apply_line_values(j, vals, lo, hi):
    """Restrict to EXACT line values, or a numeric range.

    WHY THIS EXISTS, AND WHY THE QUARTILE VERSION WAS NOT ENOUGH.
    attention_strata found receptions Q1 at beta +0.282 (t +4.60) out of
    sample with a MEDIAN LINE OF 2.5. Pricing "quartile 1" in v2 gave a
    median line of 1.5, because reception lines take only a handful of
    discrete values (1.5, 2.5, 3.5, ...) and qcut puts every tie in the same
    bucket. p10, p50 and p90 all came back 1.5, so the stratum was ONE line
    value rather than a quartile, and it was the wrong one.

    Two further things went wrong as a consequence, both visible in the v2
    output and both avoided by selecting values directly:

      - Sigma lost its level dependence. With a single line value there is no
        span to fit a power law against, so the gate correctly fell back to
        flat and every prop got the same 1.4 spread.
      - The tail went to 99 and 100 percent overs. On a 1.5 line the over is
        the cheap side, alpha rose from +0.12 to +0.42, and the layer just
        recommended the over on everything. The 2024 holdout at -0.2454 is
        that failing.

    Selecting line values directly keeps the population well defined and lets
    you exclude the degenerate bottom line while keeping the one that carried
    the signal.
    """
    before = len(j)
    if vals:
        j = j[np.isin(np.round(j["line_consensus"].to_numpy(), 2), vals)]
        print(f"  line values {vals}: {len(j)} of {before} rows")
    if lo is not None:
        j = j[j["line_consensus"] >= lo]
    if hi is not None:
        j = j[j["line_consensus"] <= hi]
    if lo is not None or hi is not None:
        print(f"  line range [{lo}, {hi}]: {len(j)} of {before} rows")
    for m, g in j.groupby("market"):
        if len(g):
            q = g["line_consensus"].quantile([0.1, 0.5, 0.9]).to_numpy()
            print(f"    {m}: {len(g)} rows, line p10/p50/p90 "
                  f"{q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}, "
                  f"{g['line_consensus'].nunique()} distinct line values")
    return j.copy()


def apply_stratum(j, line_q, freq_q):
    """Restrict to the requested quartiles.

    WHY THIS EXISTS. attention_strata.py found beta varies enormously by line
    level: receptions Q1 came in at +0.282 (t +4.60) out of sample against a
    pooled 0.275, and receiving Q1 at +0.195 against a pooled 0.093. A higher
    beta is not an edge on its own though, because it still has to clear the
    hold, and receptions carries a WIDER hold (0.0608) than the flat -113
    markets (0.0476) precisely where the signal is. This flag prices a single
    stratum so that question can be answered directly.

    Shrinkage parameters are fitted AFTER this filter, so alpha, beta and
    sigma all come from the stratum rather than being borrowed from the
    market as a whole. That is the point: a pooled beta applied to a
    stratified population is the same category of error as a pooled beta
    applied to four markets.
    """
    if not line_q and not freq_q:
        return j
    before = len(j)
    if line_q:
        j = j[j["lineq"].isin(line_q)]
        print(f"  line quartile filter {line_q}: {len(j)} of {before} rows")
    if freq_q:
        n = len(j)
        j = j[j["freqq"].isin(freq_q)]
        print(f"  player-frequency quartile filter {freq_q}: "
              f"{len(j)} of {n} rows")
    for m, g in j.groupby("market"):
        if len(g):
            print(f"    {m}: {len(g)} rows, median line "
                  f"{g['line_consensus'].median():.1f}, median player "
                  f"frequency {g['player_freq'].median():.0f}")
    return j.copy()


# ----------------------------------------------------------------------- data

def build_joined(args):
    """Load lines, score models walk-forward and join. Mirrors the harness."""
    sec = eh._secrets()
    box = {}

    def factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(x) for x in args.seasons.split(",") if x.strip()]
    else:
        seasons = [2025]

    lines = eh.load_lines(factory, seasons, markets, args.cache, args.refresh)
    if lines.empty:
        print("  no lines found")
        sys.exit(1)
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)
    print(f"  {len(props)} distinct player-week props")

    window = None
    if args.lead_window:
        p = [float(x) for x in args.lead_window.split(",")]
        window = (p[0], p[1])
    props = eh.report_lead(props, window)

    print("\nscoring models walk-forward")
    scored = []
    for m in markets:
        print(f"  {m}")
        sc = eh.score_market(m, seasons, mode="walk_forward",
                             population=args.score_population)
        if len(sc):
            scored.append(sc)
    if not scored:
        print("  nothing scored")
        sys.exit(1)
    proj = pd.concat(scored, ignore_index=True)

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)

    joined = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(joined)} rows "
          f"({100.0 * len(joined) / max(len(props), 1):.1f}% of props)")
    joined["dev_cons"] = joined["projection"] - joined["line_consensus"]
    joined["out_cons"] = joined["actual"] - joined["line_consensus"]
    return joined.dropna(subset=["dev_cons", "out_cons"])


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default=",".join(eh.DEFAULT_MARKETS))
    ap.add_argument("--exclude", default="",
                    help="markets to drop after scoring, e.g. rushing")
    ap.add_argument("--line-source", default="fanduel",
                    choices=["fanduel", "consensus", "best"],
                    help="the line you would actually bet")
    ap.add_argument("--lead-window", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"],
                    help="all = build_all_rows where available (honest); "
                         "board = build_dataset (reproduces the old numbers)")
    ap.add_argument("--line-quartile", default=None,
                    help="restrict to these line-level quartiles within "
                         "market, 1 is LOWEST. e.g. 1 or 1,2")
    ap.add_argument("--freq-quartile", default=None,
                    help="restrict to these player-prop-frequency quartiles, "
                         "1 is least often quoted. e.g. 4")
    ap.add_argument("--line-value", default=None,
                    help="restrict to EXACT consensus line values, comma "
                         "separated. e.g. 2.5,3.5  Preferred over "
                         "--line-quartile for discrete markets like "
                         "receptions, where quartiles collapse onto ties.")
    ap.add_argument("--line-min", type=float, default=None,
                    help="minimum consensus line to keep")
    ap.add_argument("--line-max", type=float, default=None,
                    help="maximum consensus line to keep")
    ap.add_argument("--sigma-mode", default="flat",
                    choices=["flat", "level"],
                    help="flat = one residual SD per market; level = a*mean^b")
    ap.add_argument("--sigma-scale", type=float, default=1.0,
                    help="multiply sigma, for testing the calibration miss")
    ap.add_argument("--sigma-scan", action="store_true",
                    help="scan multipliers to find what would calibrate")
    ap.add_argument("--sigma-sqrt", action="store_true",
                    help="impose sigma = a*sqrt(mean) instead of fitting the "
                         "exponent. Correct for count markets and needs no "
                         "level span, so it survives a restricted stratum.")
    ap.add_argument("--curse-correct", action="store_true",
                    help="recalibrate the chosen side's probability "
                         "leave-season-out, to remove the side-selection bias")
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()
    if args.season and not args.seasons:
        args.seasons = str(args.season)

    print("=" * 96)
    print("EDGE THRESHOLD v4: where a shrunk edge clears the vig")
    print(f"  betting at line_{args.line_source}")
    if args.line_value:
        print(f"  EXACT line values {args.line_value}")
    if args.line_min is not None or args.line_max is not None:
        print(f"  line range [{args.line_min}, {args.line_max}]")
    if args.line_quartile:
        print(f"  line quartiles {args.line_quartile} (1 = lowest lines)")
    if args.freq_quartile:
        print(f"  player-frequency quartiles {args.freq_quartile}")
    print("=" * 96)

    joined = build_joined(args)

    # 'best' needs the side, which needs the projection, so it is built here
    # for completeness. Note it is the shopping price, not a forecast.
    over = joined["projection"] > joined["line_consensus"]
    joined["line_best"] = np.where(over, joined["line_min"], joined["line_max"])

    lvals = ([round(float(x), 2) for x in args.line_value.split(",")]
             if args.line_value else None)
    if lvals or args.line_min is not None or args.line_max is not None:
        joined = apply_line_values(joined, lvals, args.line_min, args.line_max)
        if len(joined) < 300:
            print(f"  only {len(joined)} rows after the line filter, "
                  f"too few to price")
            return

    lq = [int(x) for x in args.line_quartile.split(",")] if args.line_quartile else None
    fq = [int(x) for x in args.freq_quartile.split(",")] if args.freq_quartile else None
    if lq or fq:
        joined = add_quartiles(joined)
        joined = apply_stratum(joined, lq, fq)
        if len(joined) < 300:
            print(f"  only {len(joined)} rows after the stratum filter, "
                  f"too few to price")
            return

    drop = [m.strip() for m in args.exclude.split(",") if m.strip()]
    if drop:
        joined = joined[~joined["market"].isin(drop)].copy()
        print(f"  excluded {drop}: {len(joined)} rows remain")

    params = fit_params(joined)
    if not params:
        print("  could not fit shrinkage parameters")
        return
    report_params(params)

    slevel = None
    if args.sigma_sqrt:
        slevel = fit_sigma_sqrt(joined, params)
        print("\n  SIGMA = a * sqrt(mean), exponent imposed at 0.5")
        print(f"  {'market':<14}{'season':>8}{'a':>9}{'b':>8}"
              f"{'flat residSD':>14}")
        for (m, s_) in sorted(slevel):
            a_, b_ = slevel[(m, s_)]
            flat = params.get((m, s_), (0, 0, np.nan, ""))[2]
            print(f"  {m:<14}{int(s_):>8}{a_:>9.3f}{b_:>8.2f}{flat:>14.1f}")
    elif args.sigma_mode == "level" or args.sigma_scan:
        slevel = fit_sigma_level(joined, params)
        report_sigma(slevel, params)

    if args.sigma_scan:
        sigma_scan(joined, params, args.line_source, slevel, args.sigma_mode)

    mode = "level" if (args.sigma_sqrt or args.sigma_mode == "level") else "flat"
    s = price(joined, params, args.line_source, slevel, mode,
              args.sigma_scale)
    if args.curse_correct:
        s = curse_correct(s)
    print(f"\n  sigma_mode={args.sigma_mode}, sigma_scale={args.sigma_scale}")
    print(f"\n  priced {len(s)} rows, {int(s['push'].sum())} pushes")
    print(f"  edge distribution: median {s['edge'].median():+.4f}, "
          f"90th {s['edge'].quantile(0.90):+.4f}, "
          f"99th {s['edge'].quantile(0.99):+.4f}, "
          f"max {s['edge'].max():+.4f}")
    pos = float((s["edge"] > 0).mean())
    print(f"  {100 * pos:.1f}% of props show a positive edge after shrinkage")

    zero_dev_check(s)
    calibration_unconditional(s)
    calibration(s)

    weeks_by_season = (s.groupby("season")["week"].nunique().to_dict())
    total_weeks = int(sum(weeks_by_season.values()))

    print("\n" + "=" * 96)
    print("THRESHOLD CURVES")
    print("=" * 96)
    curve(s, "ALL MARKETS", total_weeks)
    if "rushing" in set(s["market"]):
        nr = s[s["market"] != "rushing"]
        if len(nr) > 200:
            curve(nr, "EXCLUDING RUSHING (survivorship, see Phase 3.1)",
                  total_weeks)
    for m, g in s.groupby("market"):
        if len(g) < 300:
            continue
        note = "  [survivorship suspect]" if m == "rushing" else ""
        curve(g, f"{m}{note}", total_weeks)

    tail_profile(s)
    holdout(s, weeks_by_season)
    print("\n" + "=" * 96)
    print("HOLDOUT BY MARKET")
    print("=" * 96)
    holdout_by_market(s, weeks_by_season)

    if args.save_rows:
        s.to_csv(args.save_rows, index=False)
        print(f"\n  wrote {args.save_rows}")

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  CALIBRATION first. If the priced probabilities do not match the")
    print("  realized rates, nothing below means anything and the fix is the")
    print("  distribution or the residual SD, not the threshold.")
    print("  The threshold curve shows what betting everything above a cutoff")
    print("  would have returned per unit staked. 'per wk' is how often the")
    print("  app would speak, which matters as much as the ROI.")
    print("  The 'highest ROI' line is SELECTED and biased upward. Use the")
    print("  holdout table, which never lets the threshold see the season it")
    print("  is scored on.")
    print("  rushing is flagged throughout because its 47 percent join rate")
    print("  and 0.538 over rate point at the Phase 3.1 filter, which would")
    print("  inflate any threshold computed on it. Recompute after the fix.")
    print("  Staking is out of scope here. This measures edge per unit")
    print("  staked, not how much to stake.")


if __name__ == "__main__":
    main()
