"""
SIDE-PICKER SEARCH v2: two bugs fixed, and the receiving signal re-tested

WHAT v1 REPORTED

  receiving, model 3 (logit on book_p + line + features): disagreement slope
  +1.191, SE 0.358, t +3.33, p 0.0009, clearing its Bonferroni threshold of
  0.0100. Holdout ROI +0.0234, SE 0.0394, t +0.59 on 566 bets.

  And two of four markets FAILED the gate: rushing +0.014 against a
  published -0.043, qb_passing +0.368 against +0.085.

THE TWO BUGS, BOTH MINE

  BUG 1, which explains the gate failures. eval_harness.score_market fits
  its projection model on the module's FULL dataset and then joins to props
  for scoring. v1 fitted on the JOINED frame instead, so its training
  population was conditioned on a FanDuel prop existing. That conditions on
  the book choosing to post a line, which selects for starters and volume.
  The drift scales with how much the join discards, which is the signature:

      market       joined / full      published    v1 got
      receptions   7850 / 22540         +0.226     +0.241
      receiving    8676 / 22540         +0.066     +0.073
      rushing      2671 / 6635          -0.043     +0.014
      qb_passing   1670 / 2817          +0.085     +0.368

  BUG 2, which matters more for the receiving result. v1 fitted the
  challengers on `d`, the GATE's OUTPUT, rather than on the joined frame.
  `d` already excludes the first season because walk-forward needs a prior
  one, so for 2024 the challenger training set was EMPTY and only 2025 and
  2026 were scored. That is why receiving reported 3,148 out-of-sample rows
  instead of roughly 6,500, and why the 2025 model trained on 2024 alone.

  So the +3.33 rests on two scored seasons with one season of training. It
  may survive a correct run. It had not been tested.

WHAT I WAS WRONG ABOUT, AND WHICH IS WORTH RECORDING

  I suspected `ypt_roll` of leaking a current-game result, because receiving
  carries it and receptions does not. It does not leak. receiving.py computes
  `ypt_game` and then `shift(1).ewm(...)`, so the feature only ever sees
  prior games. The concern was unfounded.

FIXES IN v2

  1. The gate's projection model is fitted on the module's full dataset, the
     harness's way, so the gate either passes or the join is genuinely wrong.
  2. Challengers are fitted directly on the joined frame, walk-forward by
     season, so every season after the first is scored and training grows.
  3. A per-season breakdown of any winning slope, because a signal resting on
     one season is a different object from one present in all of them.
  4. A PLACEBO on the winning slope: the outcome shuffled within season,
     refit, to locate the null of this whole procedure. v2 of the odds-space
     test found a real artifact this way and it cost nothing to add.

PRE-REGISTERED PREDICTIONS

  F1. All four markets pass the gate once the training population is fixed.
      If rushing or qb_passing still fails, the join itself is wrong and
      nothing in this script counts.
  F2. The receiving slope SHRINKS substantially with four seasons scored
      instead of two, and does not clear its Bonferroni threshold.
      Reasoning: v1's estimate came from the thinnest possible training
      arm, and a slope of +1.19 on a disagreement with sd 0.024 implies the
      model reorders props more confidently than a market with an
      efficient price should allow.
  F3. The placebo centres on zero. If it does not, the slope metric carries
      a mechanical component in this design and the honest value is the
      difference.
  F4. If receiving survives, its holdout ROI still does not clear the hold.

  F2 is a prediction and I would rather be wrong about it. A surviving slope
  in a market currently displayed as reference-only would be the first
  positive result in this project in three sessions.

Run from the repo root with the venv active:
    python side_picker_search_v2.py --all-seasons --cache lines_cache.parquet
    python side_picker_search_v2.py --all-seasons --cache lines_cache.parquet --placebo 200
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
    """Return (joined props frame, module's FULL dataset, feats, actual_col).

    The full dataset is returned separately because the projection model must
    be fitted on it, not on the join. That was bug 1.
    """
    import importlib
    from models import data_utils

    mod_name, actual_col = eh.MARKET_SPEC[market]
    mod = importlib.import_module(f"models.{mod_name}")
    feats = list(getattr(mod, "LEAN_FEATS", None)
                 or getattr(mod, "FEATS", None) or [])
    if not feats:
        return None, None, None, None

    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    full = builder()
    if actual_col not in full.columns:
        return None, None, None, None
    keep = feats + [actual_col, "season", "week", "player_display_name"]
    full = full[[c for c in keep if c in full.columns]].dropna(
        subset=feats + [actual_col]).copy()
    full["_key"] = data_utils.norm_join_name(full["player_display_name"])
    full["week"] = full["week"].astype(int)

    def factory():
        raise SystemExit("cache incomplete; run eval_harness once to fill it")

    lines = eh.load_lines(factory, seasons, [market], cache, refresh)
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


def gate(j, full, market, feats, actual_col):
    """Harness parity: fit the projection on the FULL dataset, score the join.

    This is bug 1 fixed. Fitting on the join conditions the training
    population on a prop existing, which selects for starters and volume.
    """
    from sklearn.linear_model import LinearRegression
    out = []
    for s in sorted(j["season"].unique()):
        tr = full[full["season"] < s]
        if len(tr) < 300:
            continue
        te = j[j["season"] == s]
        if not len(te):
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


def fit_challengers(j, feats, seed=0):
    """Walk-forward on the JOINED frame. Bug 2 fixed.

    v1 fitted on the gate's output, which had already lost the first season,
    so the second season trained on nothing and was never scored.
    """
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
    d = j.reset_index(drop=True).copy()
    preds = {k: np.full(len(d), np.nan) for k in specs}
    scored_seasons = []
    for s in sorted(d["season"].unique()):
        tr_i = d.index[d["season"] < s]
        te_i = d.index[d["season"] == s]
        if len(tr_i) < 500 or not len(te_i):
            continue
        y_tr = d.loc[tr_i, "over"].to_numpy(int)
        if len(np.unique(y_tr)) < 2:
            continue
        scored_seasons.append((int(s), len(tr_i), len(te_i)))
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
    return d, list(specs), scored_seasons


def logloss(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def directional(sub, name, ycol="over"):
    """Slope of the outcome on the model's disagreement with the book."""
    dis = (sub[name] - sub["book_p"]).to_numpy(float)
    if not np.isfinite(dis).any() or np.allclose(dis, dis[np.isfinite(dis)][0]):
        return None
    X = np.column_stack([dis, sub["book_p"].to_numpy(float)])
    y = sub[ycol].to_numpy(float)
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


def report(d, names, market, scored_seasons):
    sub = d.dropna(subset=names + ["book_p", "over"]).copy()
    if len(sub) < 300:
        print(f"  too few scored rows ({len(sub)})")
        return None, []
    y = sub["over"].to_numpy(int)
    base_ll = logloss(y, sub["book_p"])
    thresh = 0.05 / len(names)
    print(f"  walk-forward scoring plan (season, train rows, test rows):")
    for s, ntr, nte in scored_seasons:
        print(f"    {s}  train {ntr:>6}  test {nte:>6}")
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
        win = np.isfinite(r["t"]) and two_sided_p(r["t"]) < thresh \
            and r["slope"] > 0
        print(f"  {n:<24}{r['slope']:>9.3f}{r['se']:>8.3f}{r['t']:>7.2f}"
              f"{two_sided_p(r['t']):>9.4f}{mde(r['se']):>8.3f}"
              f"{r['sd_dis']:>9.4f}{dll:>+11.5f}"
              f"   {'SIGNAL' if win else 'no'}")
        if win:
            winners.append(n)
    print(f"\n  planted-pocket calibration: no pocket t "
          f"{PLANTED['null_t']:+.2f}, 6-point t {PLANTED['six_pt_t']:+.2f}, "
          f"12-point t {PLANTED['twelve_pt_t']:+.2f}")
    print(f"  So a null means no pocket above roughly 10 to 12 points in 5")
    print(f"  percent of rows. The hold is {HOLD:.3f}, so 6 to 10 point")
    print(f"  pockets would pay and are below the detection limit.")
    return sub, winners


def per_season(sub, name):
    print(f"    PER SEASON, {name}")
    print(f"      {'season':>8}{'rows':>7}{'slope':>9}{'SE':>8}{'t':>7}")
    for s, g in sub.groupby("season"):
        r = directional(g, name)
        if r is None:
            print(f"      {int(s):>8}{len(g):>7}   too few rows")
            continue
        print(f"      {int(s):>8}{r['n']:>7}{r['slope']:>9.3f}"
              f"{r['se']:>8.3f}{r['t']:>7.2f}")
    print("      A slope present in one season only is not the same object")
    print("      as one present in all of them.")


def placebo_slope(sub, name, n_draws, seed=0):
    if not n_draws:
        return
    print(f"    PLACEBO, outcome shuffled within season, {n_draws} draws")
    rng = np.random.default_rng(seed)
    vals = []
    s2 = sub.copy()
    for _ in range(n_draws):
        s2["y_shuf"] = (s2.groupby("season")["over"]
                        .transform(lambda v: rng.permutation(v.to_numpy())))
        r = directional(s2, name, ycol="y_shuf")
        if r is not None:
            vals.append(r["slope"])
    if not vals:
        print("      no usable draws")
        return
    v = np.asarray(vals)
    se_mean = v.std() / np.sqrt(len(v))
    t = v.mean() / se_mean if se_mean > 0 else np.nan
    print(f"      mean {v.mean():+.4f}  sd {v.std():.4f}  "
          f"SE of mean {se_mean:.4f}  t vs 0 {t:+.2f}")
    print(f"      95% range [{np.percentile(v, 2.5):+.4f}, "
          f"{np.percentile(v, 97.5):+.4f}]")
    if abs(t) > 3:
        print("      PLACEBO OFFSET. Subtract this mean from the slope.")
    else:
        print("      Placebo centred. The slope is honest as printed.")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--markets", default=",".join(eh.DEFAULT_MARKETS))
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--placebo", type=int, default=0)
    args = ap.parse_args()

    seasons = (list(eh.ALL_SEASONS) if args.all_seasons or not args.seasons
               else [int(x) for x in args.seasons.split(",") if x.strip()])
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    print("=" * 96)
    print("SIDE-PICKER SEARCH v2: training population and walk-forward fixed")
    print(f"  seasons {seasons}   markets {markets}")
    print("=" * 96)

    any_winner = False
    gate_fails = []
    for m in markets:
        print(f"\n{'=' * 96}\n{m.upper()}\n{'=' * 96}")
        j, full, feats, actual_col = build(m, seasons, args.cache,
                                           args.refresh)
        if j is None or len(j) < 500:
            print("  too few rows or no feature list, skipping")
            continue
        print(f"  joined {len(j)} props against a full dataset of "
              f"{len(full)} rows")
        print(f"  features {feats}")
        f, d = gate(j, full, m, feats, actual_col)
        if f is None:
            print("  gate could not fit, skipping")
            continue
        pub = PUBLISHED_FD_BETA.get(m)
        diff = f["beta"] - pub
        ok = abs(diff) <= GATE_TOL
        print(f"  GATE beta {f['beta']:+.3f} vs published {pub:+.3f}"
              f"  diff {diff:+.3f}  {'PASS' if ok else 'FAIL'}\n")
        if not ok:
            gate_fails.append(m)
            print("  gate failed, this market's results are not comparable")
            continue
        d2, names, plan = fit_challengers(j, feats)
        sub, winners = report(d2, names, m, plan)
        if sub is None:
            continue
        for w in winners:
            any_winner = True
            print(f"\n  {w} shows signal.")
            per_season(sub, w)
            placebo_slope(sub, w, args.placebo)
            print(f"    Does it pay?")
            roi_if_winner(sub, w)

    print("\n" + "=" * 96)
    print("WHAT THIS SETTLES")
    print("=" * 96)
    if gate_fails:
        print(f"  GATE STILL FAILING for {gate_fails}. The join is wrong")
        print("  for those markets and their results mean nothing. Fix that")
        print("  before reading anything else.")
    if any_winner:
        print("  At least one challenger's disagreement with the book")
        print("  predicts the outcome out of sample, on a corrected")
        print("  walk-forward. Check the per-season table and the placebo")
        print("  before treating it as real, then the ROI line for whether")
        print("  it clears the hold.")
    else:
        print("  No function of these features beat the book in any market")
        print("  that passed the gate. Against the planted calibration that")
        print("  means no pocket above roughly 10 to 12 points in 5 percent")
        print("  of rows. Player-prop history starts 2023-05-03, so these")
        print("  four seasons are all there will ever be for the backtest.")
    print("=" * 96)


if __name__ == "__main__":
    main()
