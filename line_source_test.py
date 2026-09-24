"""
Does the choice of LINE SOURCE change alpha and beta?

Every constant we fitted today used line_fanduel from the captured lines
table. The app prices a line Ted TYPES IN. This script cannot close that gap,
because the typed line lives in the bets table, not here. What it CAN do is
show how much the pricing parameters move across the captured anchors, which
bounds how much the typed-line question could matter.

SOURCES
  line_fanduel      one book, the sharpest available anchor (hold 0.0608)
  line_consensus    MEDIAN across books. This is the harness's anchor.
  line_mean         mean across books, computed here as (min+max)/2 proxy
  line_min          the lowest line any book posted
  line_max          the highest line any book posted
  line_best         REFERENCE ONLY, selected using the projection, so its
                    beta is inflated by construction. Do not read it as a
                    candidate. It is here to show what selection bias looks
                    like next to honest anchors.

WHY min AND max ARE NOT SILLY
    They are not anchors anyone would price against, but they bracket the
    range. If alpha and beta are stable from min to max, the anchor choice
    cannot matter much and the typed-line gap is bounded. If they swing, the
    typed line needs verifying before any constant is trusted.

NOTE ON line_mean
    all_markets.csv does not carry a true mean across books, only min, max
    and the median. (min+max)/2 is a midrange, not a mean, and it is more
    sensitive to a single outlying book than either. Labelled honestly.

Clustered on event_id throughout. Alpha reported BOTH ways: regression
intercept, and direct at zero deviation, because beta clips to zero in three
markets and the intercept of a free-slope fit is the wrong parameter there.
"""

import argparse

import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except Exception:
    _st = None

# proportional zero-deviation windows, about 8% of a typical line
ZD = {"receptions": 0.25, "receiving": 3.5, "rushing": 5.0, "qb_passing": 18.0}


def ols_clustered(x, y, cluster):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], np.asarray(cluster)[ok]
    n = len(x)
    if n < 50:
        return None
    X = np.column_stack([np.ones(n), x])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    meat = np.zeros((2, 2))
    for c in np.unique(cl):
        sel = cl == c
        s = X[sel].T @ resid[sel]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return dict(alpha=float(coef[0]), beta=float(coef[1]),
                se_alpha=float(np.sqrt(max(V[0, 0], 0.0))),
                se_beta=float(np.sqrt(max(V[1, 1], 0.0))),
                n=n, g=int(len(np.unique(cl))),
                resid_sd=float(resid.std(ddof=2)))


def cluster_mean_se(v, cluster):
    v = np.asarray(v, float)
    c = np.asarray(cluster)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) < 50:
        return None
    m = v.mean()
    grp = pd.Series(v).groupby(pd.Series(c)).agg(["sum", "count"])
    g = len(grp)
    dev = grp["sum"].to_numpy() - m * grp["count"].to_numpy()
    var = np.sum(dev ** 2) / (len(v) ** 2) * (g / max(g - 1, 1))
    return dict(mean=float(m), se=float(np.sqrt(max(var, 0.0))), n=len(v), g=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all_markets.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.rows)
    d["line_mid"] = (d["line_min"] + d["line_max"]) / 2.0

    sources = ["line_fanduel", "line_consensus", "line_mid",
               "line_min", "line_max", "line_best"]

    print("=" * 120)
    print("LINE SOURCE COMPARISON")
    print("=" * 120)
    print("  line_best is REFERENCE ONLY: selected using the projection, so")
    print("  its beta is inflated by construction. Not a candidate.")

    for market, g0 in d.groupby("market"):
        if market not in ZD:
            continue
        zd = ZD[market]
        print("\n" + "-" * 120)
        print("%s   n=%d   zero-dev window %.2f" % (market, len(g0), zd))
        print("-" * 120)
        print("  %-16s %6s %5s %19s %19s %21s %8s"
              % ("source", "n", "games", "beta", "alpha (intercept)",
                 "alpha (direct)", "residSD"))

        agree = {}
        for src in sources:
            g = g0[g0[src].notna() & g0["projection"].notna()
                   & g0["actual"].notna()].copy()
            if len(g) < 50:
                print("  %-16s too few rows" % src)
                continue
            dev = g["projection"] - g[src]
            out = g["actual"] - g[src]
            f = ols_clustered(dev, out, g["event_id"])
            zsel = dev.abs() < zd
            z = cluster_mean_se(out[zsel], g["event_id"][zsel])
            if f is None:
                print("  %-16s fit failed" % src)
                continue
            bstr = "%+.4f (%.4f)" % (f["beta"], f["se_beta"])
            astr = "%+.4f (%.4f)" % (f["alpha"], f["se_alpha"])
            zstr = ("%+.4f (%.4f) n=%d" % (z["mean"], z["se"], z["n"])) \
                if z else "n=%d too few" % int(zsel.sum())
            flag = "  <- inflated" if src == "line_best" else ""
            print("  %-16s %6d %5d %19s %19s %21s %8.2f%s"
                  % (src, f["n"], f["g"], bstr, astr, zstr,
                     f["resid_sd"], flag))
            if src != "line_best":
                agree[src] = (f["beta"], f["se_beta"],
                              z["mean"] if z else None,
                              z["se"] if z else None)

        # how much does the anchor choice move things, excluding line_best
        if len(agree) >= 2:
            betas = np.array([v[0] for v in agree.values()])
            print("\n  honest anchors only (line_best excluded):")
            print("    beta  range %+.4f to %+.4f, spread %.4f"
                  % (betas.min(), betas.max(), betas.max() - betas.min()))
            dirs = [(k, v[2], v[3]) for k, v in agree.items()
                    if v[2] is not None]
            if dirs:
                dv = np.array([x[1] for x in dirs])
                print("    direct alpha range %+.4f to %+.4f, spread %.4f"
                      % (dv.min(), dv.max(), dv.max() - dv.min()))
            # is fanduel vs consensus a real difference or noise?
            if "line_fanduel" in agree and "line_consensus" in agree:
                b1, s1, _, _ = agree["line_fanduel"]
                b2, s2, _, _ = agree["line_consensus"]
                zstat = (b1 - b2) / np.sqrt(s1 ** 2 + s2 ** 2)
                print("    fanduel vs consensus beta: %+.4f vs %+.4f, "
                      "z %+.2f (NOT independent, same props, so this z is "
                      "conservative)" % (b1, b2, zstat))

    print("\n" + "=" * 120)
    print("HOW TO READ")
    print("=" * 120)
    print("  The honest-anchor SPREAD is the number that matters. If beta")
    print("  moves by less than one SE across min, median, FanDuel and max,")
    print("  the anchor choice is not material and the typed-line question")
    print("  is bounded to something small.")
    print("  FanDuel and consensus are computed on the SAME props, so the z")
    print("  between them assumes independence it does not have. Read it as")
    print("  an upper bound on significance, not a test.")
    print("  A large alpha difference between min and max is expected and is")
    print("  not a finding: alpha is defined relative to the line, so moving")
    print("  the line mechanically moves alpha by the same amount.")


if __name__ == "__main__":
    main()
