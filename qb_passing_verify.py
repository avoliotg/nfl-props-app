"""
QB_PASSING VERIFICATION: plateau or selection cliff?

WHAT IS BEING VERIFIED

  side_picker_search_v2 found, in qb_passing:

    model 3, logistic on book_p + line + features
      disagreement slope  +0.832   SE 0.312   t +2.67   p 0.0076
      2024                +0.947   t 1.85
      2025                +0.734   t 1.72
      placebo             centred, t +0.26
      dLogLoss            +0.00034  (positive)
      HOLDOUT ROI         +0.1278   SE 0.0611   t +2.09   248 bets

  That is the first result today that shows signal AND pays. Both scorable
  seasons are positive with similar slopes, which is what the receiving
  signal lacked (0.290 / 1.314 / 0.239, almost entirely 2025, ROI -0.0076).

  Feature leakage was checked and is absent: attempts_roll and def_pass_roll
  are both shift(1).rolling(6), def_pass_roll's fillna uses
  expanding().mean().shift(1) rather than a full-sample mean, and wind_eff is
  game conditions rather than a player result.

THE SPECIFIC WAY THIS COULD STILL BE NOTHING

  The ROI was produced by an argmax over 31 thresholds, exactly like the
  receptions candidate rule. When that rule was put through a fixed-cutoff
  neighbourhood test, it did not show a plateau: the non-cumulative bins ran

    0.050 to 0.065   -0.04 pp
    0.065 to 0.080   -4.97
    0.080 to 0.090   -2.07
    0.090 to 0.100   +5.02      <- the selected point
    0.100 to 0.115  +13.10
    0.115 to 0.140   +8.93

  Nothing below the threshold was positive and everything above it was. A
  13-point swing across a 0.01 boundary with no gradient leading up to it is
  a cliff, and a cliff where an argmax was pointed is what selection
  produces. The pricing layer is smooth in the projection, so nothing in it
  switches on at a threshold.

  THIS SCRIPT ASKS WHETHER QB_PASSING DOES THE SAME THING OR NOT.

  If the edge rises gradually across fixed bins, that is a dose response and
  the signal is real. If it steps at the selected point, it is the same
  artifact in a new market and should be closed the same way.

THE MULTIPLICITY POSITION, STATED HONESTLY UP FRONT

  side_picker_search_v2 printed a Bonferroni threshold of 0.0100, which was
  per market. The honest count is 5 challengers across 4 markets, so 20
  comparisons and a threshold of 0.0025. At p 0.0076 qb_passing DOES NOT
  clear that. Neither does receiving at 0.0066.

  So this signal is not established by its p-value. What makes it worth
  verifying rather than dismissing is the combination: a stable slope in both
  scorable seasons, a centred placebo, a positive log-loss difference, and an
  ROI that clears the hold. Those are four partly independent facts rather
  than one p-value, and the tests below are about whether they hold up.

ONE MORE CAVEAT WORTH KEEPING IN VIEW

  qb_passing is priced at flat odds both sides, so book_p is near-constant
  and the baseline is effectively a base rate (over rate 0.4982). The
  challenger only has to beat a coin flip, which is a far lower bar than
  receptions faced, where the price itself is an efficient forecast with a
  measured slope of 1.008. So this result reads as "features predict
  over/under at the line, nonlinearly" rather than "we beat the book's
  price". That is still a side-picker if it survives, but it is a weaker
  claim than the number alone suggests.

WHAT IS TESTED

  G1. THRESHOLD NEIGHBOURHOOD, fixed cutoffs, no selection anywhere.
  G2. NON-CUMULATIVE BINS, so a step reads as a step rather than being
      smoothed by the cumulative sum.
  G3. PER SEASON AT A FIXED CUTOFF, no argmax.
  G4. GAME-CLUSTERED BOOTSTRAP at the fixed cutoff, which removes the
      selection step that is the whole worry.
  G5. SIDE AND LINE COMPOSITION. If every bet is one side, or clusters at
      one end of the line distribution, that is the signature of a filtered
      sample rather than a general edge.

PRE-REGISTERED PREDICTIONS

  G1a. NO plateau. The edge peaks near the argmax-selected cutoff and falls
       away either side, as receptions did.
  G2a. The bins below the selected cutoff are flat or negative.
  G3a. Both seasons stay positive at a fixed cutoff, because the slope
       already was, but with intervals that overlap zero.
  G4a. The fixed-cutoff bootstrap CI includes zero.
  G5a. The bets skew heavily to one side.

  I would rather be wrong about G1a. A plateau here, in a market with two
  positive seasons and a centred placebo, would be the first thing in this
  project that survives the test that killed the receptions rule.

Run from the repo root with the venv active:
    python qb_passing_verify.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

PUB_BETA = 0.085
PUB_SLOPE = 0.832
PUB_ROI = 0.1278
PUB_BETS = 248
GATE_TOL = 0.030
GATE_TOL_SLOPE = 0.030
GATE_TOL_ROI = 0.010

MARKET = "qb_passing"
# 5 challengers x 4 markets in the search that produced this signal, plus the
# cuts in this script. Counted generously on purpose.
CUTS_TOTAL = 20 + 10 + 6 + 2


def american_to_decimal(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 1.0 + o / 100.0 if o > 0 else 1.0 + 100.0 / (-o)


def devig(a, b):
    da, db = american_to_decimal(a), american_to_decimal(b)
    if not (np.isfinite(da) and np.isfinite(db)):
        return np.nan
    ra, rb = 1.0 / da, 1.0 / db
    return ra / (ra + rb) if (ra + rb) > 0 else np.nan


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def mde(se):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


def build(seasons, cache, refresh):
    import importlib
    from models import data_utils
    mod_name, actual_col = eh.MARKET_SPEC[MARKET]
    mod = importlib.import_module(f"models.{mod_name}")
    feats = list(getattr(mod, "LEAN_FEATS", None)
                 or getattr(mod, "FEATS", None) or [])
    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    full = builder()
    keep = feats + [actual_col, "season", "week", "player_display_name"]
    full = full[[c for c in keep if c in full.columns]].dropna(
        subset=feats + [actual_col]).copy()
    full["_key"] = data_utils.norm_join_name(full["player_display_name"])
    full["week"] = full["week"].astype(int)

    def factory():
        raise SystemExit("cache incomplete; run eval_harness once first")

    lines = eh.load_lines(factory, seasons, [MARKET], cache, refresh)
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)
    props["_key"] = data_utils.norm_join_name(props["player"])
    props["week"] = props["week"].astype(int)

    j = props.merge(full, on=["season", "week", "_key"], how="inner")
    j["book_p"] = [devig(o, u)
                   for o, u in zip(j["over_odds"], j["under_odds"])]
    j["dec_over"] = [american_to_decimal(o) for o in j["over_odds"]]
    j["dec_under"] = [american_to_decimal(o) for o in j["under_odds"]]
    j = j.dropna(subset=["line_fanduel", "book_p", actual_col,
                         "dec_over", "dec_under"]).copy()
    j = j[j[actual_col] != j["line_fanduel"]].copy()
    j["over"] = (j[actual_col] > j["line_fanduel"]).astype(int)
    return j, full, feats, actual_col


def fit_model3(j, feats):
    """Walk-forward logistic on book_p + line + features, as in the search."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cols = ["book_p", "line_fanduel"] + feats
    d = j.reset_index(drop=True).copy()
    p = np.full(len(d), np.nan)
    for s in sorted(d["season"].unique()):
        tr = d.index[d["season"] < s]
        te = d.index[d["season"] == s]
        if len(tr) < 500 or not len(te):
            continue
        y = d.loc[tr, "over"].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000))
        m.fit(d.loc[tr, cols].to_numpy(float), y)
        p[te] = m.predict_proba(d.loc[te, cols].to_numpy(float))[:, 1]
    d["p"] = p
    d = d.dropna(subset=["p"]).copy()
    d["be_over"] = 1.0 / d["dec_over"]
    d["be_under"] = 1.0 / d["dec_under"]
    d["e_over"] = d["p"] - d["be_over"]
    d["e_under"] = (1.0 - d["p"]) - d["be_under"]
    d["side_over"] = d["e_over"] >= d["e_under"]
    d["edge"] = np.where(d["side_over"], d["e_over"], d["e_under"])
    won = np.where(d["side_over"], d["over"] == 1, d["over"] == 0)
    pay = np.where(d["side_over"], d["dec_over"] - 1.0, d["dec_under"] - 1.0)
    d["won_f"] = won.astype(float)
    d["profit"] = np.where(won, pay, -1.0)
    be_ch = np.where(d["side_over"], d["be_over"], d["be_under"])
    be_ot = np.where(d["side_over"], d["be_under"], d["be_over"])
    d["fair_chosen"] = be_ch / (be_ch + be_ot)
    return d


def directional(sub):
    dis = (sub["p"] - sub["book_p"]).to_numpy(float)
    X = np.column_stack([dis, sub["book_p"].to_numpy(float)])
    y = sub["over"].to_numpy(float)
    c = sub["event_id"].to_numpy()
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, c = X[ok], y[ok], c[ok]
    n = len(y)
    if n < 100:
        return None
    D = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    inv = np.linalg.inv(D.T @ D)
    meat = np.zeros((3, 3))
    o = np.argsort(c)
    Ds, rs, cs = D[o], resid[o], c[o]
    st = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(st) - 1
    for i in range(g):
        sl = slice(st[i], st[i + 1])
        u = Ds[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - 3, 1))
    se = np.sqrt(np.diag(inv @ meat @ inv * scale))
    return {"slope": coef[1], "se": se[1], "n": n,
            "t": coef[1] / se[1] if se[1] > 0 else np.nan}


def gap_stats(g):
    if len(g) < 20:
        return None
    cw = eh.cluster_mean(g["won_f"].to_numpy(), g["event_id"].to_numpy())
    if cw is None:
        return None
    fair = float(g["fair_chosen"].mean())
    return {"n": len(g), "wr": cw["mean"], "fair": fair,
            "gap": cw["mean"] - fair, "se": cw["se"],
            "t": (cw["mean"] - fair) / cw["se"] if cw["se"] > 0 else np.nan,
            "roi": float(g["profit"].mean())}


def select_threshold(d):
    """The argmax the search used, reported so the neighbourhood is centred."""
    grid = np.round(np.arange(0.0, 0.151, 0.005), 4)
    out = {}
    for s_out in sorted(d["season"].unique()):
        tr = d[d["season"] != s_out]
        bt, bv = None, -np.inf
        for t in grid:
            g = tr[tr["edge"] >= t]
            if len(g) < 100:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is not None:
            out[s_out] = bt
    return out


def gate(j, full, d, feats, actual_col):
    from sklearn.linear_model import LinearRegression
    print("\n" + "=" * 96)
    print("GATE")
    print("=" * 96)
    out = []
    for s in sorted(j["season"].unique()):
        tr = full[full["season"] < s]
        te = j[j["season"] == s]
        if len(tr) < 300 or not len(te):
            continue
        m = LinearRegression().fit(tr[feats], tr[actual_col])
        te = te.copy()
        te["proj"] = m.predict(te[feats])
        out.append(te)
    g = pd.concat(out, ignore_index=True)
    f = eh.ols_clustered(g["proj"] - g["line_fanduel"],
                         g[actual_col] - g["line_fanduel"], g["event_id"])
    r = directional(d)
    th = select_threshold(d)
    rows = [d[(d["season"] == k) & (d["edge"] >= v)] for k, v in th.items()]
    rows = [x for x in rows if len(x) >= 20]
    bets = pd.concat(rows, ignore_index=True)
    cm = eh.cluster_mean(bets["profit"].to_numpy(),
                         bets["event_id"].to_numpy())

    checks = [
        ("beta", f["beta"], PUB_BETA, GATE_TOL),
        ("model 3 slope", r["slope"], PUB_SLOPE, GATE_TOL_SLOPE),
        ("holdout ROI", cm["mean"], PUB_ROI, GATE_TOL_ROI),
        ("holdout bets", float(len(bets)), float(PUB_BETS), 0.5),
    ]
    fails = []
    print(f"  {'quantity':<18}{'got':>11}{'published':>12}{'diff':>10}"
          f"   verdict")
    for name, got, pub, tol in checks:
        dd = got - pub
        ok = np.isfinite(dd) and abs(dd) <= tol
        print(f"  {name:<18}{got:>11.4f}{pub:>12.4f}{dd:>+10.4f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    if fails:
        print(f"\n  GATE FAILED for {fails}. Stop.")
        sys.exit(2)
    print("  GATE PASSED.")
    print(f"\n  thresholds the argmax chose: "
          f"{ {int(k): float(v) for k, v in th.items()} }")
    return th, bets


def g1(d, th):
    print("\n" + "=" * 96)
    print("G1. THRESHOLD NEIGHBOURHOOD, fixed cutoffs, NO selection")
    print("=" * 96)
    sel = float(np.median(list(th.values())))
    print(f"  the argmax sat near {sel:.3f}. A PLATEAU across the")
    print(f"  neighbourhood means a real edge; a SPIKE at the selected")
    print(f"  point means the argmax found a cliff.\n")
    print(f"  {'cutoff':>8}{'rows':>7}{'winrt':>9}{'fair':>9}{'gap pp':>9}"
          f"{'SE pp':>8}{'t':>7}{'ROI':>9}")
    for t in np.round(np.arange(0.01, 0.101, 0.01), 3):
        r = gap_stats(d[d["edge"] >= t])
        if r is None:
            print(f"  {t:>8.3f}   too few rows")
            continue
        print(f"  {t:>8.3f}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    print("\n  Cumulative, so heavily autocorrelated. G2 removes that.")
    return sel


def g2(d):
    print("\n" + "=" * 96)
    print("G2. NON-CUMULATIVE BINS")
    print("=" * 96)
    edges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 1.0]
    print(f"  {'bin':<18}{'rows':>7}{'winrt':>9}{'fair':>9}{'gap pp':>9}"
          f"{'SE pp':>8}{'t':>7}{'ROI':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = d[(d["edge"] >= lo) & (d["edge"] < hi)]
        r = gap_stats(g)
        lab = f"{lo:.2f} to {hi:.2f}" if hi < 1 else f"{lo:.2f} and up"
        if r is None:
            print(f"  {lab:<18}{len(g):>7}   too few rows")
            continue
        print(f"  {lab:<18}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    print("\n  A gradual rise is a dose response. A flat run then a jump at")
    print("  the selected point is the receptions cliff in a new market.")


def g3(d, sel):
    print("\n" + "=" * 96)
    print(f"G3. PER SEASON AT A FIXED {sel:.3f}, no argmax")
    print("=" * 96)
    sub = d[d["edge"] >= sel]
    print(f"  {'season':>8}{'rows':>7}{'winrt':>9}{'fair':>9}{'gap pp':>9}"
          f"{'SE pp':>8}{'t':>7}{'ROI':>9}")
    for s, g in sub.groupby("season"):
        r = gap_stats(g)
        if r is None:
            print(f"  {int(s):>8}{len(g):>7}   too few rows")
            continue
        print(f"  {int(s):>8}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    r = gap_stats(sub)
    if r:
        print(f"  {'POOLED':>8}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    return sub


def g4(sub, n_boot, seed=0):
    print("\n" + "=" * 96)
    print(f"G4. GAME-CLUSTERED BOOTSTRAP at the fixed cutoff, {n_boot} draws")
    print("=" * 96)
    if len(sub) < 50:
        print("  too few rows")
        return
    rng = np.random.default_rng(seed)
    ev = sub["event_id"].to_numpy()
    won = sub["won_f"].to_numpy()
    fair = sub["fair_chosen"].to_numpy()
    prof = sub["profit"].to_numpy()
    gr = {}
    for i, e in enumerate(ev):
        gr.setdefault(e, []).append(i)
    keys = list(gr)
    idx = [np.asarray(gr[k]) for k in keys]
    gaps, rois = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx[i] for i in pick])
        gaps.append(won[sel].mean() - fair[sel].mean())
        rois.append(prof[sel].mean())
    gaps, rois = np.asarray(gaps), np.asarray(rois)
    print(f"  {len(keys)} games hold {len(sub)} bets")
    print(f"  gap  mean {100 * gaps.mean():+.2f} pp  "
          f"95% CI [{100 * np.percentile(gaps, 2.5):+.2f}, "
          f"{100 * np.percentile(gaps, 97.5):+.2f}]  "
          f"{100 * (gaps > 0).mean():.1f}% positive")
    print(f"  ROI  mean {rois.mean():+.4f}  "
          f"95% CI [{np.percentile(rois, 2.5):+.4f}, "
          f"{np.percentile(rois, 97.5):+.4f}]  "
          f"{100 * (rois > 0).mean():.1f}% positive")
    print("\n  No threshold was selected here, so a CI excluding zero is the")
    print("  strongest statement available. Read it against G5.")


def g5(sub, d):
    print("\n" + "=" * 96)
    print("G5. SIDE AND LINE COMPOSITION")
    print("=" * 96)
    n = len(sub)
    print(f"  bets {n}   overs {int(sub['side_over'].sum())}"
          f"   unders {int((~sub['side_over']).sum())}")
    print(f"  line p10 / p50 / p90  "
          f"{np.percentile(sub['line_fanduel'], 10):.1f} / "
          f"{np.percentile(sub['line_fanduel'], 50):.1f} / "
          f"{np.percentile(sub['line_fanduel'], 90):.1f}")
    print(f"  all rows for comparison  "
          f"{np.percentile(d['line_fanduel'], 10):.1f} / "
          f"{np.percentile(d['line_fanduel'], 50):.1f} / "
          f"{np.percentile(d['line_fanduel'], 90):.1f}")
    print(f"  mean decimal odds taken {1 + np.where(sub['side_over'], sub['dec_over'] - 1, sub['dec_under'] - 1).mean():.4f}")
    print("\n  A tail that collapses onto one side and one end of the line")
    print("  distribution is the signature of a filtered sample, not a")
    print("  general edge.")
    print(f"\n  MULTIPLICITY: {CUTS_TOTAL} comparisons counted across the")
    print(f"  search and this script. Bonferroni at 0.05: "
          f"{0.05 / CUTS_TOTAL:.5f}.")
    print(f"  The signal's p was 0.0076, which does not clear it. What")
    print(f"  makes it worth verifying is the combination of a stable")
    print(f"  per-season slope, a centred placebo, a positive log-loss")
    print(f"  difference and an ROI above the hold, not the p-value.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    seasons = (list(eh.ALL_SEASONS) if args.all_seasons or not args.seasons
               else [int(x) for x in args.seasons.split(",") if x.strip()])

    print("=" * 96)
    print("QB_PASSING VERIFICATION: plateau or selection cliff?")
    print(f"  seasons {seasons}")
    print("=" * 96)

    j, full, feats, actual_col = build(seasons, args.cache, args.refresh)
    print(f"  joined {len(j)} props, full dataset {len(full)} rows")
    print(f"  features {feats}")
    d = fit_model3(j, feats)
    print(f"  scored {len(d)} rows out of sample")
    th, bets = gate(j, full, d, feats, actual_col)
    sel = g1(d, th)
    g2(d)
    sub = g3(d, sel)
    g4(sub, args.boot)
    g5(sub, d)

    print("\n" + "=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  G2 first. Gradual rise means a dose response and the signal is")
    print("  real. Flat then a jump at the selected point means this is the")
    print("  receptions cliff again, and the honest move is to close it the")
    print("  same way. Then G4, which removes the selection step. G5 says")
    print("  whether the bets are a general edge or one filtered corner.")
    print("=" * 96)


if __name__ == "__main__":
    main()
