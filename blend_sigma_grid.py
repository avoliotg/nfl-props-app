"""
Price the same rows across a GRID of (mean, sigma) choices, without touching
production code.

WHY BOTH AT ONCE
    PARAMS sigma was fitted as residual spread around the model's OWN
    projection, so it includes projection error. Variant B prices around the
    blend, which has had most of that error removed. Applying the old sigma to
    the new mean leaves sigma too wide, which biases P(over) DOWN uniformly.
    eval_harness says this in its own legend. So the mean change and the sigma
    change are not separable, and shipping one alone ships a bias whose sign
    we can already predict.

MEANS
  A raw        projection                    ships today
  B blend      line + alpha + beta*dev       Phase 4.4

SIGMAS
  P params     whatever PARAMS says          ships today
  R resid      flat residSD from the harness the blend-appropriate scalar
  S sqrt       sqrt(k * mean), receptions    the one-parameter form

Sigma is injected by monkeypatching mc_pricing.sigma_for for the duration of
one variant, then restored. Everything else (family, floor, moment match,
zero gate) stays the real code path.

ALPHA AND BETA come from the DIRECT measurement at zero deviation, not the
regression intercept, because beta is clipped to zero in three markets and
the intercept of a free-slope fit is the wrong parameter for a zero-slope
model. qb_passing alpha is 0 because its direct alpha is +0.55 with an SE of
2.55, which is not a measured parameter.

Reads all_markets.csv (eval_harness --save-rows output).
"""

import argparse

import numpy as np
import pandas as pd

import mc_pricing

# direct alpha at zero deviation, beta clipped where the clustered CI
# includes zero. See alpha_caliber.py and fit_ab.py.
BLEND = {
    "receptions": dict(alpha=0.1491, beta=0.2261, resid=2.1, sqrt_k=1.22),
    "receiving":  dict(alpha=3.4623, beta=0.0, resid=28.4, sqrt_k=None),
    "rushing":    dict(alpha=4.0067, beta=0.0, resid=30.6, sqrt_k=None),
    "qb_passing": dict(alpha=0.0, beta=0.0, resid=72.8, sqrt_k=None),
}

BANDS = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]
_REAL_SIGMA_FOR = mc_pricing.sigma_for


class sigma_override:
    """Temporarily replace mc_pricing.sigma_for. Restores on exit."""

    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        mc_pricing.sigma_for = self.fn
        return self

    def __exit__(self, *a):
        mc_pricing.sigma_for = _REAL_SIGMA_FOR
        return False


def make_flat(value):
    def f(market, proj, stage_mult=1.0):
        return max(value * float(stage_mult), mc_pricing.SIGMA_MIN)
    return f


def make_sqrt(k):
    def f(market, proj, stage_mult=1.0):
        m = max(float(proj), 0.0)
        return max(np.sqrt(k * m) * float(stage_mult), mc_pricing.SIGMA_MIN)
    return f


def tier(e):
    if e is None or not np.isfinite(e):
        return "unpriced"
    if e < 2:
        return "Pass"
    if e < 4:
        return "Lean"
    if e < 7:
        return "Strong"
    return "Max"


def price_vector(market, means, lines):
    out = np.full(len(means), np.nan)
    for i, (m, ln) in enumerate(zip(means, lines)):
        p = mc_pricing.p_over(market, m, ln)
        if p is not None:
            out[i] = p
    return out


def breakeven(over_odds, under_odds):
    be = np.full(len(over_odds), np.nan)
    for i, (o, u) in enumerate(zip(over_odds, under_odds)):
        v = mc_pricing.breakeven_from_odds(o, u)
        if v is not None:
            be[i] = v
    return be


def calib(pv, won):
    """Worst band gap and the full band row, predicted minus actual."""
    cells, worst, worst_where = [], 0.0, ""
    for i in range(len(BANDS) - 1):
        sel = np.isfinite(pv) & (pv >= BANDS[i]) & (pv < BANDS[i + 1])
        n = int(sel.sum())
        if n < 30:
            cells.append("%13s" % ("n=%d" % n))
            continue
        gap = 100.0 * (pv[sel].mean() - won[sel].mean())
        cells.append("%13s" % ("%+.1f(%d)" % (gap, n)))
        if abs(gap) > abs(worst):
            worst = gap
            worst_where = "%.2f-%.2f" % (BANDS[i], BANDS[i + 1])
    return cells, worst, worst_where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all_markets.csv")
    ap.add_argument("--source", default="line_fanduel",
                    choices=["line_fanduel", "line_consensus"])
    args = ap.parse_args()

    d = pd.read_csv(args.rows)
    d = d[d[args.source].notna() & d["projection"].notna()
          & d["actual"].notna()].copy()
    d["line"] = d[args.source]
    d["dev"] = d["projection"] - d["line"]

    print("=" * 118)
    print("MEAN x SIGMA GRID   source=%s" % args.source)
    print("=" * 118)

    for market, g in d.groupby("market"):
        if market not in BLEND:
            continue
        p = BLEND[market]
        line = g["line"].to_numpy(float)
        dev = g["dev"].to_numpy(float)
        proj = g["projection"].to_numpy(float)
        won = (g["actual"].to_numpy(float) > line).astype(float)
        be = breakeven(g["over_odds"].to_numpy(), g["under_odds"].to_numpy())

        means = {
            "A raw": proj,
            "B blend": line + p["alpha"] + p["beta"] * dev,
        }
        sigmas = {
            "P params": None,
            "R resid%.0f" % p["resid"]: make_flat(p["resid"]),
        }
        if p["sqrt_k"]:
            sigmas["S sqrt%.2f" % p["sqrt_k"]] = make_sqrt(p["sqrt_k"])

        print("\n" + "-" * 118)
        print("%s   n=%d   alpha=%+.4f  beta=%.4f   actual over rate %.4f"
              % (market, len(g), p["alpha"], p["beta"], won.mean()))
        print("  PARAMS sigma at the median mean: %.2f"
              % _REAL_SIGMA_FOR(market, float(np.median(proj))))
        print("-" * 118)
        hdr = "  %-22s %7s %7s %7s %8s   %s" % (
            "variant", "priced", "mean_p", "worst", "where",
            "tiers P/L/S/M")
        print(hdr)

        results = []
        for mname, mu in means.items():
            for sname, sfn in sigmas.items():
                ctx = sigma_override(sfn) if sfn else sigma_override(_REAL_SIGMA_FOR)
                with ctx:
                    pv = price_vector(market, mu, line)
                ok = np.isfinite(pv)
                if ok.sum() == 0:
                    print("  %-22s all unpriced" % (mname + " / " + sname))
                    continue
                edge_o = (pv - be) * 100.0
                edge_u = ((1 - pv) - (1 - be)) * 100.0
                best = np.where(edge_o >= edge_u, edge_o, edge_u)
                t = pd.Series([tier(e) for e in best[ok]]).value_counts()
                tstr = "/".join(str(int(t.get(k, 0)))
                                for k in ("Pass", "Lean", "Strong", "Max"))
                cells, worst, where = calib(pv, won)
                print("  %-22s %7d %7.4f %+7.1f %8s   %s"
                      % (mname + " / " + sname, int(ok.sum()),
                         np.nanmean(pv[ok]), worst, where, tstr))
                results.append((mname + " / " + sname, cells))

        print("\n  CALIBRATION BY BAND, pred - actual in points")
        print("  %-22s" % "variant" + "".join(
            "%13s" % ("%.2f-%.2f" % (BANDS[i], BANDS[i + 1]))
            for i in range(len(BANDS) - 1)))
        for label, cells in results:
            print("  %-22s" % label + "".join(cells))

    print("\n" + "=" * 118)
    print("HOW TO READ")
    print("=" * 118)
    print("  'worst' is the largest absolute band gap, which is the criterion")
    print("  sigma_refit used. Lower is better. 'where' names the band.")
    print("  Expected ordering if the reasoning is right:")
    print("    B beats A on the TILT (bottom band negative, top positive)")
    print("    R beats P once the mean is the blend, because PARAMS sigma")
    print("      includes projection error the blend has removed")
    print("    S beats R for receptions, because flat is the wrong form")
    print("  A uniform sign across every band is a MEAN offset, fixable with")
    print("  alpha. A tilt is mean error that alpha cannot fix. A shift in")
    print("  one direction everywhere as sigma changes is the sigma effect,")
    print("  and dP/dsigma is negative, so a wider sigma lowers every band.")
    print("  If B/P shows a uniform NEGATIVE gap and B/R shrinks it, that is")
    print("  the predicted too-wide-sigma bias and it confirms 4.2 and 4.4")
    print("  are not separable.")


if __name__ == "__main__":
    main()
