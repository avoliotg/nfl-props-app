"""
SIDE-PICKER SEARCH: is the market wrong in POCKETS our features can find?

THE QUESTION THIS ASKS, AND WHY IT IS NOT ANOTHER CUT OF THE SAME DATA

  Every test in odds_space_test v1 to v4 asked whether (projection - line)
  predicts (actual - line) LINEARLY. Beta is one scalar. That specification
  assumes the market's error is proportional to our disagreement with it, so
  it CANNOT see a market that is efficient on average and wrong in
  identifiable pockets: high target share with a falling snap rate, big
  underdogs in high-total games, and so on. Those are interactions.

  So: is there any function of our features, interactions included, that
  beats the book's own devigged probability out of sample?

  If nothing does, the market is efficient with respect to everything this
  project can measure, and picking a side needs NEW information rather than
  a better arrangement of the features we have. If something does, that
  function is the side-picker, in every market including the flat-odds ones.

TWO DESIGNS WERE TESTED AGAINST A PLANTED POCKET. THE FIRST ONE FAILED.

  I planted a market error of 6 points concentrated in a 5 percent pocket
  defined by a feature interaction, which is exactly the object this test
  exists to find, and checked whether each metric could see it.

  AVERAGE LOG LOSS CANNOT. Boosting returned t -5.49 with no pocket and
  t -5.45 with the pocket. Identical. The model's own variance costs about
  0.012 of log loss, which swamps a real pocket worth a small fraction of
  that. A script using average log loss would have returned a confident
  negative that meant nothing.

  THE DIRECTIONAL SLOPE CAN, above a size. Regressing the outcome on the
  model's DISAGREEMENT with the book, controlling for the book, gives:

      planted pocket    slope      t
      none              +0.025   +0.27
      6 points          -0.007   -0.07
      12 points         +0.200   +2.17

  Noise that is uncorrelated with the outcome contributes zero slope, which
  is why this survives model variance and log loss does not. So the
  directional slope is the primary metric here and log loss is reported only
  as a secondary, with its demonstrated blindness stated beside it.

  SEED AVERAGING IS A NO-OP for this estimator and should not be tried
  again. Five seeds returned slopes identical to three decimals, because
  HistGradientBoostingClassifier is deterministic unless rows are
  subsampled: random_state only affects binning.

THE HONEST POWER LIMIT, STATED BEFORE ANY RESULT

  At about 8,000 rows this design detects a 12-point pocket covering 5
  percent of rows and misses a 6-point one. The market's hold is about 5.7
  percent, so a pocket must exceed roughly 6 points to be profitable at all.

  DETECTION LIMIT ABOUT 10 TO 12 POINTS, PROFITABILITY LIMIT ABOUT 6.

  There is therefore a band of pockets that would be profitable and that
  this data cannot find. A negative result here means "no pocket large
  enough for us to see", not "no pocket". That band cannot be closed by a
  better method: The Odds API's historical player props begin 2023-05-03, so
  there is no more data to buy at any price, and the four seasons in hand are
  all there will ever be for the backtest.

THE CHALLENGERS, all fitted walk-forward and scored out of sample

    1 logit: book_p only            sanity check, should tie the baseline
    2 logit: book_p + line
    3 logit: book_p + features      the plan's LINE-AS-FEATURE model
    4 boost: book_p + features      allows interactions, the real candidate
    5 boost: features only          our features with no market input

  Model 4 is the one that could find a pocket. Model 5 asks whether the
  features carry anything at all on their own.

PRE-REGISTERED PREDICTIONS

  E1. Model 1 ties the baseline: slope within noise of zero, log loss
      difference within noise. If it does not, the fitting is broken and
      nothing below counts.
  E2. No challenger's disagreement slope clears t +2 in any market.
  E3. Model 5 is clearly worse than the baseline in every market.
  E4. If anything wins it is model 4 on receptions, the one market with a
      real price signal for a booster to correct.

  I have been wrong five times today, twice with confidence, and the log
  loss design above was wrong in a way that would have produced a false
  negative. E2 is a prediction, not a conclusion.

THE GATE

  Reproduce the published FanDuel-anchored betas, 0.226 / 0.066 / -0.043 /
  0.085, from this script's own feature-carrying join before any model is
  fitted. This join is new code, so it has to be verified against a
  published number before it produces one.

Run from the repo root with the venv active:
    python side_picker_search.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

PUBLISHED_FD_BETA = {
    "receptions": 0.226, "receiving": 0.066,
    "rushing": -0.043, "qb_passing": 0.085,
}
GATE_TOL = 0.030

# From the planted-pocket calibration above. Printed beside every result so
# a null is read as a detection limit rather than as an absence.
PLANTED = {"null_t": 0.27, "six_pt_t": -0.07, "twelve_pt_t": 2.17}
HOLD = 0.057


def american_to_decimal(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 1.0 + o / 100.0 if o > 0 else 1.0 + 100.0 / (-o)


def devig(over_odds, under_odds):
    do = american_to_decimal(over_odds)
    du = american_to_decimal(under_odds)
    if not (np.isfinite(do) and np.isfinite(du)):
        return np.nan
    ro, ru = 1.0 / do, 1.0 / du
    return ro / (ro + ru) if (ro + ru) > 0 else np.nan


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def mde(se):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


def build(market, seasons, cache, refresh):
    """Join props to the module's OWN feature frame, not just projections."""
    import importlib
    from models import data_utils

    mod_name, actual_col = eh.MARKET_SPEC[market]
    mod = importlib.import_module(f"models.{mod_name}")
    feats = list(getattr(mod, "LEAN_FEATS", None)
                 or getattr(mod, "FEATS", None) or [])
    if not feats:
        return None, None, None

    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    df = builder()
    if actual_col not in df.columns:
        return None, None, None

    keep = feats + [actual_col, "season", "week", "player_display_name"]
    df = df[[c for c in keep if c in df.columns]].dropna(
        subset=feats + [actual_col]).copy()

    def factory():
        raise SystemExit("cache incomplete; run eval_harness once to fill it")

    lines = eh.load_lines(factory, seasons, [market], cache, refresh)
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)

    props["_key"] = data_utils.norm_join_name(props["player"])
    df["_key"] = data_utils.norm_join_name(df["player_display_name"])
    props["week"] = props["week"].astype(int)
    df["week"] = df["week"].astype(int)

    j = props.merge(df, on=["season", "week", "_key"], how="inner")
    j["book_p"] = [devig(o, u)
                   for o, u in zip(j["over_odds"], j["under_odds"])]
    j["dec_over"] = [american_to_decimal(o) for o in j["over_odds"]]
    j["dec_under"] = [american_to_decimal(o) for o in j["under_odds"]]
    j = j.dropna(subset=["line_fanduel", "book_p", actual_col,
                         "dec_over", "dec_under"]).copy()
    j = j[j[actual_col] != j["line_fanduel"]].copy()
    j["over"] = (j[actual_col] > j["line_fanduel"]).astype(int)
    return j, feats, actual_col


def gate(j, market, feats, actual_col):
    from sklearn.linear_model import LinearRegression
    out = []
    for s in sorted(j["season"].unique()):
        tr = j[j["season"] < s]
        te = j[j["season"] == s]
        if len(tr) < 300 or not len(te):
            continue
        m = LinearRegression().fit(tr[feats], tr[actual_col])
        te = te.copy()
        te["proj"] = m.predict(te[feats])
        out.append(te)
    if not out:
        return None, None
    d = pd.concat(out, ignore_index=True)
    f = eh.ols_clustered(d["proj"] - d["line_fanduel"],
                         d[actual_col] - d["line_fanduel"], d["event_id"])
    return f, d


def fit_challengers(d, feats, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    specs = {
        "1 logit book_p only": (["book_p"], "logit"),
        "2 logit book_p+line": (["book_p", "line_fanduel"], "logit"),
        "3 logit book_p+feats": (["book_p", "line_fanduel"] + feats, "logit"),
        "4 boost book_p+feats": (["book_p", "line_fanduel"] + feats, "boost"),
        "5 boost feats only": (["line_fanduel"] + feats, "boost"),
    }
    d = d.reset_index(drop=True)
    preds = {k: np.full(len(d), np.nan) for k in specs}
    for s in sorted(d["season"].unique()):
        tr_i = d.index[d["season"] < s]
        te_i = d.index[d["season"] == s]
        if len(tr_i) < 500 or not len(te_i):
            continue
        y_tr = d.loc[tr_i, "over"].to_numpy(int)
        if len(np.unique(y_tr)) < 2:
            continue
        for name, (cols, kind) in specs.items():
            X_tr = d.loc[tr_i, cols].to_numpy(float)
            X_te = d.loc[te_i, cols].to_numpy(float)
            if kind == "logit":
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000))
            else:
                m = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=200, learning_rate=0.05,
                    l2_regularization=1.0, random_state=seed)
            m.fit(X_tr, y_tr)
            preds[name][te_i] = m.predict_proba(X_te)[:, 1]
    for k, v in preds.items():
        d[k] = v
    return d, list(specs)


def logloss(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def directional(sub, name):
    """PRIMARY METRIC. Slope of the outcome on the model's disagreement
    with the book, controlling for the book. Validated against a planted
    pocket; average log loss was not able to see the same object.
    """
    dis = (sub[name] - sub["book_p"]).to_numpy(float)
    if np.allclose(dis, dis[0]):
        return None
    X = np.column_stack([dis, sub["book_p"].to_numpy(float)])
    y = sub["over"].to_numpy(float)
    c = sub["event_id"].to_numpy()
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, c = X[ok], y[ok], c[ok]
    n = len(y)
    if n < 200:
        return None
    D = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    inv = np.linalg.inv(D.T @ D)
    meat = np.zeros((3, 3))
    order = np.argsort(c)
    Ds, rs, cs = D[order], resid[order], c[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(starts) - 1
    for i in range(g):
        sl = slice(starts[i], starts[i + 1])
        u = Ds[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - 3, 1))
    se = np.sqrt(np.diag(inv @ meat @ inv * scale))
    return {"n": n, "clusters": g, "slope": coef[1], "se": se[1],
            "t": coef[1] / se[1] if se[1] > 0 else np.nan,
            "sd_dis": float(dis[ok].std())}


def report(d, names, market):
    sub = d.dropna(subset=names + ["book_p", "over"]).copy()
    if len(sub) < 300:
        print(f"  too few scored rows ({len(sub)})")
        return None, []
    y = sub["over"].to_numpy(int)
    base_ll = logloss(y, sub["book_p"])
    thresh = 0.05 / len(names)
    print(f"  {len(sub)} out-of-sample rows, "
          f"{sub['event_id'].nunique()} games, over rate {y.mean():.4f}")
    print(f"  baseline book_p log loss {base_ll.mean():.5f}")
    print(f"  Bonferroni over {len(names)} challengers: {thresh:.4f}\n")
    print(f"  {'challenger':<24}{'slope':>9}{'SE':>8}{'t':>7}{'p':>9}"
          f"{'MDE':>8}{'sd dis':>9}{'dLogLoss':>11}   verdict")
    winners = []
    for n in names:
        r = directional(sub, n)
        if r is None:
            print(f"  {n:<24}   no variation in disagreement")
            continue
        dll = float((base_ll - logloss(y, sub[n])).mean())
        win = np.isfinite(r["t"]) and r["t"] > 2.0
        print(f"  {n:<24}{r['slope']:>9.3f}{r['se']:>8.3f}{r['t']:>7.2f}"
              f"{two_sided_p(r['t']):>9.4f}{mde(r['se']):>8.3f}"
              f"{r['sd_dis']:>9.4f}{dll:>+11.5f}"
              f"   {'SIGNAL' if win else 'no'}")
        if win:
            winners.append(n)
    print(f"\n  PLANTED-POCKET CALIBRATION for this design:")
    print(f"    no pocket t {PLANTED['null_t']:+.2f},  "
          f"6-point pocket t {PLANTED['six_pt_t']:+.2f},  "
          f"12-point pocket t {PLANTED['twelve_pt_t']:+.2f}")
    print(f"    So a null above means no pocket larger than about 10 to 12")
    print(f"    points covering 5 percent of rows. The hold is "
          f"{HOLD:.3f}, so")
    print(f"    pockets between about 6 and 10 points would pay and would")
    print(f"    not be visible here.")
    print(f"    The dLogLoss column is reported for completeness only: the")
    print(f"    same calibration showed it blind to a planted pocket.")
    return sub, winners


def roi_if_winner(sub, name):
    s = sub.copy()
    s["be_over"] = 1.0 / s["dec_over"]
    s["be_under"] = 1.0 / s["dec_under"]
    s["e_over"] = s[name] - s["be_over"]
    s["e_under"] = (1.0 - s[name]) - s["be_under"]
    s["side_over"] = s["e_over"] >= s["e_under"]
    s["edge"] = np.where(s["side_over"], s["e_over"], s["e_under"])
    won = np.where(s["side_over"], s["over"] == 1, s["over"] == 0)
    pay = np.where(s["side_over"], s["dec_over"] - 1.0, s["dec_under"] - 1.0)
    s["profit"] = np.where(won, pay, -1.0)
    grid = np.round(np.arange(0.0, 0.151, 0.005), 4)
    rows = []
    for s_out in sorted(s["season"].unique()):
        tr = s[s["season"] != s_out]
        te = s[s["season"] == s_out]
        bt, bv = None, -np.inf
        for t in grid:
            g = tr[tr["edge"] >= t]
            if len(g) < 100:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[te["edge"] >= bt]
        if len(g) >= 20:
            rows.append(g)
    if not rows:
        print("      holdout selected nothing")
        return
    b = pd.concat(rows, ignore_index=True)
    cm = eh.cluster_mean(b["profit"].to_numpy(), b["event_id"].to_numpy())
    t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
    print(f"      HOLDOUT ROI {cm['mean']:+.4f}  SE {cm['se']:.4f}"
          f"  t {t:+.2f}  bets {len(b)}")
    print(f"      Beating the book on the slope but losing money means the")
    print(f"      signal is real and smaller than the hold.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--markets", default=",".join(eh.DEFAULT_MARKETS))
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    seasons = (list(eh.ALL_SEASONS) if args.all_seasons or not args.seasons
               else [int(x) for x in args.seasons.split(",") if x.strip()])
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    print("=" * 96)
    print("SIDE-PICKER SEARCH: is the market wrong in findable pockets?")
    print(f"  seasons {seasons}   markets {markets}")
    print("  primary metric: slope of the outcome on the model's")
    print("  disagreement with the book, controlling for the book")
    print("=" * 96)

    any_winner = False
    for m in markets:
        print(f"\n{'=' * 96}\n{m.upper()}\n{'=' * 96}")
        j, feats, actual_col = build(m, seasons, args.cache, args.refresh)
        if j is None or len(j) < 500:
            print("  too few rows or no feature list, skipping")
            continue
        print(f"  joined {len(j)} rows")
        print(f"  features {feats}")
        f, d = gate(j, m, feats, actual_col)
        if f is None:
            print("  gate could not fit, skipping")
            continue
        pub = PUBLISHED_FD_BETA.get(m)
        diff = f["beta"] - pub
        ok = abs(diff) <= GATE_TOL
        print(f"  GATE beta {f['beta']:+.3f} vs published {pub:+.3f}"
              f"  diff {diff:+.3f}  {'PASS' if ok else 'FAIL'}\n")
        if not ok:
            print("  gate failed, this market's results are not comparable")
            continue
        d, names = fit_challengers(d, feats)
        sub, winners = report(d, names, m)
        for w in winners:
            any_winner = True
            print(f"\n  {w} shows signal. Does it pay?")
            roi_if_winner(sub, w)

    print("\n" + "=" * 96)
    print("WHAT THIS SETTLES")
    print("=" * 96)
    if any_winner:
        print("  At least one challenger's disagreement with the book")
        print("  predicts the outcome out of sample. That is a side-picker.")
        print("  Refit it deliberately rather than trusting these defaults,")
        print("  and check the ROI line for whether it clears the hold.")
    else:
        print("  No function of these features, interactions included,")
        print("  beat the book in any market. Against the planted-pocket")
        print("  calibration that means: no pocket larger than about 10 to")
        print("  12 points covering 5 percent of rows.")
        print("  Pockets between about 6 and 10 points would be profitable")
        print("  and are below this data's detection limit. That band")
        print("  cannot be closed with more analysis: player-prop history")
        print("  starts 2023-05-03 and these four seasons are all there is.")
        print("  Picking a side would need NEW information, not a better")
        print("  arrangement of these features.")
    print("=" * 96)


if __name__ == "__main__":
    main()
