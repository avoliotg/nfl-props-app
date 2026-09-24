"""
Does ALPHA depend on player caliber?

Alpha is the mean-versus-median offset: the outcome's expected value sits
above a line placed at the median because the distribution is right-skewed.
It is a valid PRICING parameter and an invalid betting signal. We established
today that alpha is what removes the over-dispersion tilt, so if it varies by
caliber, a single pooled value is leaving calibration on the table.

TWO MEASUREMENTS, because they fail differently:

  1. REGRESSION INTERCEPT of out on dev. This is what caliber_split.py
     reported. Its SE depends on how far the tier's dev values sit from zero,
     so a tier with few near-zero rows gets a wide intercept mechanically,
     not because alpha is uncertain there.

  2. DIRECT at zero deviation, |dev| < ZD. The mean of (actual - line) on
     rows where the projection roughly agrees with the line. No extrapolation.
     This is the quantity plan item 4.1 actually wants, and it is what the
     app would use.

If the two disagree within a tier, trust 2 and distrust the tier's slope.

CONFOUND TO KEEP IN MIND
    alpha_by_line_v4 already found alpha unstable ACROSS SEASONS at four of
    six line values (2.5 p=0.003, 4.5 p=0.010, 5.5 p=0.000). That is a
    different cut than across tiers. A tier that disagrees here might be
    picking up season instability rather than a caliber effect, so the
    per-season breakdown prints for any tier that fails.

Clustered on event_id throughout, matching eval_harness.
"""

import argparse
import importlib

import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except Exception:
    _st = None


def cluster_mean_se(values, cluster):
    """Mean of values with a cluster-robust SE. Clusters are games."""
    v = np.asarray(values, float)
    c = np.asarray(cluster)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    n = len(v)
    if n < 30:
        return None
    m = v.mean()
    groups = pd.Series(v).groupby(pd.Series(c)).agg(["sum", "count"])
    g = len(groups)
    # variance of the mean under clustering: sum of within-cluster
    # deviations squared, scaled
    dev = groups["sum"].to_numpy() - m * groups["count"].to_numpy()
    var = np.sum(dev ** 2) / (n ** 2) * (g / max(g - 1, 1))
    return dict(mean=float(m), se=float(np.sqrt(max(var, 0.0))),
                n=n, n_clusters=g)


def ols_clustered(x, y, cluster):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], np.asarray(cluster)[ok]
    n = len(x)
    if n < 30:
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
                n=n, n_clusters=int(len(np.unique(cl))))


def homogeneity(vals, ses):
    pairs = [(v, s) for v, s in zip(vals, ses)
             if v is not None and s is not None and s > 0]
    if len(pairs) < 2:
        return None
    b = np.array([p[0] for p in pairs])
    se = np.array([p[1] for p in pairs])
    w = 1.0 / se ** 2
    bbar = float(np.sum(w * b) / np.sum(w))
    chi2 = float(np.sum(w * (b - bbar) ** 2))
    df = len(pairs) - 1
    p = float(_st.chi2.sf(chi2, df)) if _st is not None else float("nan")
    return dict(chi2=chi2, df=df, p=p, pooled=bbar)


def run_cut(d, label, tier_col, zd, show_seasons=False):
    print("\n" + "-" * 108)
    print("CUT: %s" % label)
    print("-" * 108)
    print("  %-26s %20s %26s" % ("tier", "intercept alpha", "direct alpha (|dev|<%.2f)" % zd))
    ints, int_ses, dirs, dir_ses, labels = [], [], [], [], []

    for lab, sub in d.groupby(tier_col, observed=True):
        if len(sub) < 100:
            continue
        f = ols_clustered(sub["dev"], sub["out"], sub["event_id"])
        zsub = sub[sub["dev"].abs() < zd]
        z = cluster_mean_se(zsub["out"], zsub["event_id"])
        istr = ("%+.4f (SE %.4f)" % (f["alpha"], f["se_alpha"])) if f else "n/a"
        zstr = ("%+.4f (SE %.4f) n=%d" % (z["mean"], z["se"], z["n"])) if z \
            else "n=%d too few" % len(zsub)
        print("  %-26s %20s %26s" % (str(lab), istr, zstr))
        labels.append(str(lab))
        ints.append(f["alpha"] if f else None)
        int_ses.append(f["se_alpha"] if f else None)
        dirs.append(z["mean"] if z else None)
        dir_ses.append(z["se"] if z else None)

    for nm, v, s in (("intercept", ints, int_ses), ("direct", dirs, dir_ses)):
        h = homogeneity(v, s)
        if h:
            print("  %-10s homogeneity chi2=%6.2f df=%d p=%.4f  pooled=%+.4f  -> %s"
                  % (nm, h["chi2"], h["df"], h["p"], h["pooled"],
                     "TIERS DISAGREE" if h["p"] < 0.05 else "consistent"))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all_markets.csv")
    ap.add_argument("--market", default="receptions")
    ap.add_argument("--source", default="line_fanduel",
                    choices=["line_fanduel", "line_consensus"])
    ap.add_argument("--zero-dev", type=float, default=0.25)
    args = ap.parse_args()

    d = pd.read_csv(args.rows)
    d = d[d["market"] == args.market].copy()
    d = d[d[args.source].notna() & d["projection"].notna()
          & d["actual"].notna()].copy()
    d["line"] = d[args.source]
    d["dev"] = d["projection"] - d["line"]
    d["out"] = d["actual"] - d["line"]

    print("=" * 108)
    print("ALPHA BY CALIBER   market=%s   source=%s   n=%d   zero-dev window %.2f"
          % (args.market, args.source, len(d), args.zero_dev))
    print("=" * 108)

    f = ols_clustered(d["dev"], d["out"], d["event_id"])
    z = cluster_mean_se(d[d["dev"].abs() < args.zero_dev]["out"],
                        d[d["dev"].abs() < args.zero_dev]["event_id"])
    print("\nPOOLED")
    print("  intercept alpha  %+.4f (SE %.4f)" % (f["alpha"], f["se_alpha"]))
    print("  direct alpha     %+.4f (SE %.4f) on %d rows, %d games"
          % (z["mean"], z["se"], z["n"], z["n_clusters"]))
    print("  the two agree to %.4f" % abs(f["alpha"] - z["mean"]))

    # ---- proxy 1, target share
    try:
        mod = importlib.import_module("models.%s" % args.market)
        frame = mod.build_dataset()
        keep = [c for c in ["season", "week", "player_display_name",
                            "target_share_roll", "position"]
                if c in frame.columns]
        frame = frame[keep].copy()
        norm = lambda s: (s.astype(str).str.lower()
                          .str.replace(r"[^a-z ]", "", regex=True).str.strip())
        frame["_pk"] = norm(frame["player_display_name"])
        d["_pk"] = norm(d["player"])
        d = d.merge(frame.drop(columns=["player_display_name"]),
                    on=["season", "week", "_pk"], how="left")
        rate = d["target_share_roll"].notna().mean()
        print("\n  target_share join rate %.1f%%" % (100 * rate))
        if rate >= 0.5:
            g = d[d["target_share_roll"].notna()].copy()
            q = g["target_share_roll"].quantile([0.25, 0.5, 0.75]).to_list()
            g["ts_tier"] = pd.cut(g["target_share_roll"],
                                  [-np.inf] + q + [np.inf],
                                  labels=["Q1 low", "Q2", "Q3", "Q4 high"])
            run_cut(g, "target_share_roll quartile (the honest cut)",
                    "ts_tier", args.zero_dev)
            if "position" in g.columns:
                g["pos"] = g["position"].where(
                    g["position"].isin(["WR", "TE", "RB"]), "other")
                run_cut(g, "position", "pos", args.zero_dev)
    except Exception as e:
        print("  proxy 1 failed: %s: %s" % (type(e).__name__, e))

    # ---- proxy 2, line value
    run_cut(d, "line value (the market's own caliber read)", "line",
            args.zero_dev)

    # ---- season, to separate a caliber effect from the known season instability
    run_cut(d, "season (the known confound, for reference)", "season",
            args.zero_dev)

    print("\n" + "=" * 108)
    print("HOW TO READ")
    print("=" * 108)
    print("  The DIRECT column is the decision. It is what plan item 4.1")
    print("  wants and it needs no extrapolation to dev=0.")
    print("  Where intercept and direct disagree inside a tier, the tier's")
    print("  dev values sit far from zero and the intercept is an")
    print("  extrapolation. Distrust the slope there too.")
    print("  A per-tier alpha table is only worth shipping if the DIRECT")
    print("  row says TIERS DISAGREE. Otherwise pooled alpha is the honest")
    print("  default and one fewer parameter to defend.")
    print("  If the season cut also disagrees, a tier difference may be")
    print("  season composition rather than caliber. alpha_by_line_v4")
    print("  already found season instability at four of six line values.")


if __name__ == "__main__":
    main()
