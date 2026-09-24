"""
Price the same rows four ways and compare, WITHOUT touching production code.

Variants, all priced through mc_pricing.p_over so the distribution family,
the projection floor and the moment match are the real ones:

  A  raw            mean = projection                  what ships today
  B  blend_ab       mean = line + alpha + beta*dev      Phase 4.4 as written
  C  blend_b        mean = line + beta*dev              beta only, alpha dropped
  D  line_only      mean = line                         the null

For receptions, beta is nonzero so C and D differ. For the three markets
where beta clips to zero, B collapses to line+alpha and C collapses to D,
which is exactly the decision we are trying to see.

Sigma is NOT changed here. This isolates the mean change. Run the sigma
change separately or the two effects are confounded.

Reads all_markets.csv (eval_harness --save-rows output).
"""

import argparse

import numpy as np
import pandas as pd

import mc_pricing

# Full-sample fit on line_fanduel, from fit_ab.py. Beta clipped to zero
# where the harness's game-clustered CI includes zero.
BLEND = {
    "receptions": dict(alpha=0.1243, beta=0.2261, beta_raw=0.2261, se=0.035),
    "receiving":  dict(alpha=4.4462, beta=0.0,    beta_raw=0.0658, se=0.044),
    "rushing":    dict(alpha=3.4632, beta=0.0,    beta_raw=-0.0429, se=0.048),
    "qb_passing": dict(alpha=2.3671, beta=0.0,    beta_raw=0.0851, se=0.067),
}

BANDS = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]


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

    print("=" * 96)
    print("BLEND COMPARISON   source=%s   sigma UNCHANGED (PARAMS as shipped)"
          % args.source)
    print("=" * 96)

    for market, g in d.groupby("market"):
        if market not in BLEND:
            print("\n%s: no blend parameters, skipped" % market)
            continue
        p = BLEND[market]
        a, b, braw = p["alpha"], p["beta"], p["beta_raw"]
        clipped = " (beta clipped from %+.4f, t=%+.1f)" % (
            braw, braw / p["se"]) if b == 0.0 and braw != 0.0 else ""

        line = g["line"].to_numpy(float)
        dev = g["dev"].to_numpy(float)
        proj = g["projection"].to_numpy(float)
        act = g["actual"].to_numpy(float)

        means = {
            "A raw": proj,
            "B line+a+b*dev": line + a + b * dev,
            "C line+b*dev": line + b * dev,
            "D line only": line,
        }

        be = breakeven(g["over_odds"].to_numpy(), g["under_odds"].to_numpy())
        won = (act > line).astype(float)

        print("\n" + "-" * 96)
        print("%s   n=%d   alpha=%+.4f  beta=%.4f%s"
              % (market, len(g), a, b, clipped))
        print("  actual over rate %.4f" % np.nanmean(won))
        print("-" * 96)
        print("  %-16s %8s %8s %8s %8s %8s   %s"
              % ("variant", "priced", "mean_p", "med_p", "p<.35", "p>.65",
                 "tiers (Pass/Lean/Strong/Max)"))

        for name, mu in means.items():
            pv = price_vector(market, mu, line)
            ok = np.isfinite(pv)
            n_ok = int(ok.sum())
            if n_ok == 0:
                print("  %-16s %8d   all unpriced" % (name, 0))
                continue
            edge_o = (pv - be) * 100.0
            edge_u = ((1 - pv) - (1 - be)) * 100.0
            best = np.where(edge_o >= edge_u, edge_o, edge_u)
            tiers = pd.Series([tier(e) for e in best[ok]]).value_counts()
            tstr = "/".join(str(int(tiers.get(k, 0)))
                            for k in ("Pass", "Lean", "Strong", "Max"))
            print("  %-16s %8d %8.4f %8.4f %8d %8d   %s"
                  % (name, n_ok, np.nanmean(pv[ok]), np.nanmedian(pv[ok]),
                     int((pv[ok] < 0.35).sum()), int((pv[ok] > 0.65).sum()),
                     tstr))

        # calibration on the shipped bands, predicted minus actual
        print("\n  CALIBRATION, pred - actual in points (negative = priced too low)")
        print("  %-16s" % "variant" + "".join(
            "%14s" % ("%.2f-%.2f" % (BANDS[i], BANDS[i + 1]))
            for i in range(len(BANDS) - 1)))
        for name, mu in means.items():
            pv = price_vector(market, mu, line)
            cells = []
            for i in range(len(BANDS) - 1):
                sel = np.isfinite(pv) & (pv >= BANDS[i]) & (pv < BANDS[i + 1])
                if sel.sum() < 30:
                    cells.append("%14s" % ("n=%d" % int(sel.sum())))
                else:
                    gap = 100.0 * (pv[sel].mean() - won[sel].mean())
                    cells.append("%14s" % ("%+.1f (%d)" % (gap, int(sel.sum()))))
            print("  %-16s" % name + "".join(cells))

    print("\n" + "=" * 96)
    print("HOW TO READ")
    print("=" * 96)
    print("  Bands are on EACH variant's own predictions, so the row sets")
    print("  differ between variants. That is deliberate: the question is")
    print("  whether a variant's stated probabilities are honest, not how")
    print("  the same rows move.")
    print("  A uniform sign across every band is a MEAN offset. A tilt from")
    print("  positive at the bottom to negative at the top is the")
    print("  over-dispersion pattern and sigma cannot fix it.")
    print("  'priced' below n means the projection floor returned None. For")
    print("  variants B, C and D the mean is near the line, so the floor")
    print("  bites far less often than it does for A.")


if __name__ == "__main__":
    main()
