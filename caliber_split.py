"""
Does beta or alpha depend on player CALIBER?

Three independent caliber proxies, because each has a different failure mode:

  1. target_share_roll  role, pre-game, from the model frame. The honest one.
  2. line               the market's own caliber assessment. No join needed.
  3. n_books            quote frequency. Already showed spearman +1.00 on beta.

All three are knowable before kickoff, so none of them is the volume-filter
contamination that erased the rushing and qb_passing betas.

WHY THIS MIGHT BE A MIRAGE
    The within-tier correlation finding says the model's headline correlation
    largely reflects separating high-volume from low-volume players, which the
    market already encodes in the line. So a beta that rises with caliber may
    be saying the LINE is a better forecast for obscure players, not that the
    model is smarter about stars. Read the per-tier residual SD column beside
    beta before concluding anything.

Standard errors are clustered on event_id, matching eval_harness, because
props from one game share a game state and iid SEs would be too small.

Reads all_markets.csv (eval_harness --save-rows) and joins the model frame
for target_share_roll.
"""

import argparse
import importlib

import numpy as np
import pandas as pd


def ols_clustered(x, y, cluster):
    """Slope, intercept and cluster-robust SE on the slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cluster = x[ok], y[ok], np.asarray(cluster)[ok]
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
    for c in np.unique(cluster):
        sel = cluster == c
        Xc, rc = X[sel], resid[sel]
        s = Xc.T @ rc
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return dict(alpha=coef[0], beta=coef[1],
                se_beta=float(np.sqrt(max(V[1, 1], 0.0))),
                se_alpha=float(np.sqrt(max(V[0, 0], 0.0))),
                n=n, n_clusters=int(len(np.unique(cluster))),
                resid_sd=float(resid.std(ddof=2)))


def report(label, f):
    if f is None:
        print("  %-22s too few rows" % label)
        return
    t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
    print("  %-22s n=%5d g=%4d  beta=%+.4f (SE %.4f, t %+5.2f)  "
          "alpha=%+.4f (SE %.4f)  residSD=%6.2f"
          % (label, f["n"], f["n_clusters"], f["beta"], f["se_beta"], t,
             f["alpha"], f["se_alpha"], f["resid_sd"]))


def homogeneity(fits):
    """Chi-square test that the betas across tiers are the same."""
    fits = [f for f in fits if f is not None and f["se_beta"] > 0]
    if len(fits) < 2:
        return None
    b = np.array([f["beta"] for f in fits])
    se = np.array([f["se_beta"] for f in fits])
    w = 1.0 / se ** 2
    bbar = np.sum(w * b) / np.sum(w)
    chi2 = float(np.sum(w * (b - bbar) ** 2))
    df = len(fits) - 1
    # survival function of chi2 without scipy
    from math import exp, gamma
    def sf(x, k):
        # regularized upper incomplete gamma, series fallback
        if x <= 0:
            return 1.0
        terms, t, s = 200, 1.0, 0.0
        a = k / 2.0
        # use scipy if present, it is a dependency of this repo anyway
        try:
            from scipy import stats as _st
            return float(_st.chi2.sf(x, k))
        except Exception:
            return float("nan")
    return dict(chi2=chi2, df=df, p=sf(chi2, df), pooled=bbar)


def cut(series, edges, labels):
    return pd.cut(series, edges, labels=labels, include_lowest=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all_markets.csv")
    ap.add_argument("--market", default="receptions")
    ap.add_argument("--source", default="line_fanduel",
                    choices=["line_fanduel", "line_consensus"])
    args = ap.parse_args()

    d = pd.read_csv(args.rows)
    d = d[d["market"] == args.market].copy()
    d = d[d[args.source].notna() & d["projection"].notna()
          & d["actual"].notna()].copy()
    d["line"] = d[args.source]
    d["dev"] = d["projection"] - d["line"]
    d["out"] = d["actual"] - d["line"]

    print("=" * 104)
    print("CALIBER SPLIT   market=%s   source=%s   n=%d"
          % (args.market, args.source, len(d)))
    print("=" * 104)

    base = ols_clustered(d["dev"], d["out"], d["event_id"])
    print("\nPOOLED (the number we were about to hardcode)")
    report("all rows", base)

    # ---------------------------------------------------------------- proxy 1
    print("\n" + "-" * 104)
    print("PROXY 1: target_share_roll, joined from the model frame. The honest cut.")
    print("-" * 104)
    joined = False
    try:
        mod = importlib.import_module("models.%s" % args.market)
        frame = mod.build_dataset()
        keep = ["season", "week", "player_display_name",
                "target_share_roll", "snap_roll", "targets_roll"]
        keep = [c for c in keep if c in frame.columns]
        frame = frame[keep].copy()
        frame["_pk"] = (frame["player_display_name"].astype(str).str.lower()
                        .str.replace(r"[^a-z ]", "", regex=True).str.strip())
        d["_pk"] = (d["player"].astype(str).str.lower()
                    .str.replace(r"[^a-z ]", "", regex=True).str.strip())
        d2 = d.merge(frame.drop(columns=["player_display_name"]),
                     on=["season", "week", "_pk"], how="left")
        rate = d2["target_share_roll"].notna().mean()
        print("  join rate %.1f%% (%d of %d rows)"
              % (100 * rate, int(d2["target_share_roll"].notna().sum()), len(d2)))
        if rate < 0.5:
            print("  join rate too low to read. Skipping proxy 1.")
        else:
            g = d2[d2["target_share_roll"].notna()].copy()
            q = g["target_share_roll"].quantile([0.25, 0.5, 0.75]).to_list()
            g["tier"] = cut(g["target_share_roll"],
                            [-np.inf] + q + [np.inf],
                            ["Q1 lowest", "Q2", "Q3", "Q4 highest"])
            fits = []
            for lab, sub in g.groupby("tier", observed=True):
                f = ols_clustered(sub["dev"], sub["out"], sub["event_id"])
                fits.append(f)
                ts = sub["target_share_roll"]
                report("%s  [%.3f-%.3f]" % (lab, ts.min(), ts.max()), f)
            h = homogeneity(fits)
            if h:
                print("  homogeneity chi2=%.2f df=%d p=%.4f  pooled=%+.4f  -> %s"
                      % (h["chi2"], h["df"], h["p"], h["pooled"],
                         "TIERS DISAGREE" if h["p"] < 0.05 else "consistent"))
            joined = True
    except Exception as e:
        print("  join failed: %s: %s" % (type(e).__name__, e))

    # ---------------------------------------------------------------- proxy 2
    print("\n" + "-" * 104)
    print("PROXY 2: the line itself. The market's own caliber assessment.")
    print("-" * 104)
    fits = []
    for lab, sub in d.groupby("line", observed=True):
        if len(sub) < 200:
            continue
        f = ols_clustered(sub["dev"], sub["out"], sub["event_id"])
        fits.append(f)
        report("line %.1f" % lab, f)
    h = homogeneity(fits)
    if h:
        print("  homogeneity chi2=%.2f df=%d p=%.4f  pooled=%+.4f  -> %s"
              % (h["chi2"], h["df"], h["p"], h["pooled"],
                 "TIERS DISAGREE" if h["p"] < 0.05 else "consistent"))

    # ---------------------------------------------------------------- proxy 3
    print("\n" + "-" * 104)
    print("PROXY 3: n_books, the quote-frequency proxy.")
    print("-" * 104)
    fits = []
    for lab, sub in d.groupby("n_books", observed=True):
        if len(sub) < 200:
            continue
        f = ols_clustered(sub["dev"], sub["out"], sub["event_id"])
        fits.append(f)
        report("n_books %d" % lab, f)
    h = homogeneity(fits)
    if h:
        print("  homogeneity chi2=%.2f df=%d p=%.4f  pooled=%+.4f  -> %s"
              % (h["chi2"], h["df"], h["p"], h["pooled"],
                 "TIERS DISAGREE" if h["p"] < 0.05 else "consistent"))

    print("\n" + "=" * 104)
    print("HOW TO READ")
    print("=" * 104)
    print("  A beta that RISES with caliber may only mean the LINE is the")
    print("  better forecast for obscure players. Check residSD: if it rises")
    print("  with caliber too, the beta difference is partly scale, not skill.")
    print("  The chi-square is the decision. A single pooled beta is")
    print("  defensible when the tiers do not disagree, and that is the")
    print("  null we should prefer given how many betas in this project")
    print("  turned out to be artifacts.")
    print("  If proxy 1 and proxy 2 disagree with each other, trust proxy 1:")
    print("  the line is chosen partly in response to the same information")
    print("  the projection uses, so conditioning on it is closer to")
    print("  conditioning on the answer.")


if __name__ == "__main__":
    main()
