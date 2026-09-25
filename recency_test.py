"""recency_test.py - does the LINE overreact to recent games?

Reports only. Writes nothing, places no bet.

THE HYPOTHESIS, STATED SO IT CAN FAIL

    The argument is not "the model cannot see intangibles". It is sharper
    than that and it is about how the LINE is built: the market overweights
    the last game or two relative to what actually predicts the next one, so
    a player the recent discourse hates is priced too low and the over is
    underpriced.

    That is falsifiable with data already in hand, and it does not require
    out-forecasting anyone. It requires only that the market's weighting
    differ from the outcome's weighting.

THE DIRECT TEST, WHICH IS SECTION 2

    Two regressions on the same rows, same regressors:

        actual ~ prev1 + prev8      the TRUE weights
        line   ~ prev1 + prev8      the MARKET's weights

    prev1 is the player's last game, prev8 the mean of the eight before the
    current one. Both are pre-game, both lagged. Because line and actual are
    in identical units, the coefficients are directly comparable with no
    scaling.

    If the market's coefficient on prev1 EXCEEDS the true one, the line
    overweights the last game and fading recent form pays. If they match,
    the market weights recency correctly and the hypothesis is dead in its
    strongest form.

    This is a better test than regressing actual-minus-line on a shock,
    because it says WHERE any disagreement lives rather than only that one
    exists. Section 3 runs the shock version too, since it converts directly
    into a bet.

WHAT WOULD MAKE IT ACTIONABLE RATHER THAN MERELY TRUE

    A weight difference is not money. Section 4 prices the implied rule at
    the ACTUAL posted odds, with a cluster bootstrap and a season split,
    because the measured hold is 4.55 percent on yardage and that is the bar
    any edge has to clear.

CONFOUNDS THIS CANNOT REMOVE

    A player whose last game was poor may genuinely be declining: an injury,
    a benching, a role change. So a negative prev1 shock legitimately
    predicts lower output some of the time, and this test measures the NET of
    real decline against market overreaction. A null result therefore does
    not prove the market is efficient, only that overreaction does not
    dominate real decline on average.

    Rolling windows run across season boundaries, as the models' own
    features do. A player's first games of a season are informed by the prior
    season, which is a modelling choice rather than a fact.

Run from the repo root:
    python recency_test.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
from models import data_utils

MARKET_STAT = {
    "receiving": "receiving_yards",
    "receptions": "receptions",
    "rushing": "rushing_yards",
    "qb_rushing": "rushing_yards",
    "qb_passing": "passing_yards",
}
SEASONS_STATS = [2022, 2023, 2024, 2025, 2026]


def _rule(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def ols_cluster(X, y, clusters):
    """Multivariate OLS with cluster-robust covariance.

    eval_harness.ols_clustered handles ONE regressor. The headline test needs
    two, so this exists. Section 1 gates it against the harness's function on
    the single-regressor case, because a hand-rolled sandwich estimator is
    exactly the kind of thing that silently returns plausible wrong numbers.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    g = np.asarray(clusters)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y, g = X[keep], y[keep], g[keep]
    n, k = X.shape
    if n <= k + 1:
        return None
    XtX = X.T @ X
    try:
        XtXi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta

    meat = np.zeros((k, k))
    order = np.argsort(g, kind="stable")
    Xs, es, gs = X[order], e[order], g[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    G = len(bounds) - 1
    for i in range(G):
        sl = slice(bounds[i], bounds[i + 1])
        u = Xs[sl].T @ es[sl]
        meat += np.outer(u, u)
    if G <= 1:
        return None
    adj = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
    V = XtXi @ meat @ XtXi * adj
    return {"beta": beta, "se": np.sqrt(np.diag(V)), "n": n, "G": G,
            "V": V}


def build_history():
    """Per player-week lagged form, one column set per stat."""
    ps = data_utils.load_player_stats(SEASONS_STATS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    stats = sorted(set(MARKET_STAT.values()))
    have = [c for c in stats if c in ps.columns]
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps["_key"] = data_utils.norm_join_name(ps["player_display_name"])

    idc = next((c for c in ("player_id", "gsis_id") if c in ps.columns), None)
    if idc:
        nid = ps.groupby(["season", "week", "_key"])[idc].transform("nunique")
        n_bad = int((nid > 1).sum())
        if n_bad:
            print(f"  dropped {n_bad} rows on normalized-name collisions")
        ps = ps[nid == 1]

    # A player with a stat row was active, so a blank stat is a real zero.
    for c in have:
        ps[c] = ps[c].fillna(0.0)

    key = idc if idc else "_key"
    ps = ps.sort_values([key, "season", "week"]).reset_index(drop=True)
    for c in have:
        gb = ps.groupby(key)[c]
        ps[f"{c}__prev1"] = gb.shift(1)
        ps[f"{c}__prev3"] = gb.transform(
            lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        ps[f"{c}__prev8"] = gb.transform(
            lambda s: s.shift(1).rolling(8, min_periods=4).mean())
        # prev8 EXCLUDING the last game, so prev1 and the baseline do not
        # share the observation whose weight is being tested.
        ps[f"{c}__base"] = gb.transform(
            lambda s: s.shift(2).rolling(7, min_periods=4).mean())
    cols = ["season", "week", "_key"] + have + [
        f"{c}__{suf}" for c in have for suf in ("prev1", "prev3", "prev8", "base")]
    return ps[cols]


def build(args, seasons):
    _rule("LOADING")
    markets = sorted({eh.FETCH_MARKET.get(m, m) for m in MARKET_STAT})
    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, markets, args.cache, args.refresh)
    lines, rep = data_utils.split_qb_rushing(lines, min_match_rate=0.98)
    lines = lines[lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    if args.book != "all":
        lines = lines[lines["book"] == args.book]
    keys = ["season", "week", "market", "player", "book"]
    if "captured_at" in lines.columns:
        lines = lines.sort_values("captured_at").drop_duplicates(keys, keep="last")
    print(f"  {len(lines):,} rows, book={args.book}")

    lines["dec_over"] = [devig.american_to_decimal(o) for o in lines["over_odds"]]
    lines["dec_under"] = [devig.american_to_decimal(o) for o in lines["under_odds"]]
    lines = lines[lines["dec_over"].notna() & lines["dec_under"].notna()].copy()
    lines["_key"] = data_utils.norm_join_name(lines["player"])

    hist = build_history()
    out = []
    for market, stat in MARKET_STAT.items():
        sel = lines[lines["market"] == market]
        if not len(sel) or f"{stat}__prev1" not in hist.columns:
            continue
        cols = ["season", "week", "_key", stat,
                f"{stat}__prev1", f"{stat}__prev3", f"{stat}__prev8",
                f"{stat}__base"]
        m = sel.merge(hist[cols], on=["season", "week", "_key"], how="inner")
        m = m.rename(columns={stat: "actual",
                              f"{stat}__prev1": "prev1",
                              f"{stat}__prev3": "prev3",
                              f"{stat}__prev8": "prev8",
                              f"{stat}__base": "base"})
        out.append(m)
    p = pd.concat(out, ignore_index=True)
    p = p.dropna(subset=["actual", "line", "prev1", "base"]).copy()

    # Walk-forward projections, so section 3 can ask whether the shock adds
    # anything BEYOND what the model already sees. Without this the control
    # column silently returns nan, which is how a missing test looks like a
    # passing one.
    proj = []
    for market in sorted(p["market"].unique()) if len(p) else []:
        try:
            s = eh.score_market(market, seasons, mode="walk_forward",
                                population="all")
        except Exception as e:
            print(f"  could not score {market}: {type(e).__name__}: {e}")
            continue
        if len(s):
            proj.append(s)
    if proj:
        proj = pd.concat(proj, ignore_index=True)
        proj["_key"] = data_utils.norm_join_name(proj["player"])
        proj["week"] = proj["week"].astype(int)
        proj = proj.drop_duplicates(
            subset=["season", "week", "market", "_key"])
        p = p.merge(proj[["season", "week", "market", "_key", "projection"]],
                    on=["season", "week", "market", "_key"], how="left")
        print(f"  {int(p['projection'].notna().sum()):,} of {len(p):,} rows "
              f"carry a walk-forward projection")
    else:
        print("  no projections available; the shock-versus-model control "
              "in section 3 will be skipped and reported as skipped")

    p = p[p["actual"] != p["line"]].copy()
    p["over"] = (p["actual"] > p["line"]).astype(int)
    p["roi_under"] = np.where(p["over"] == 0, p["dec_under"] - 1.0, -1.0)
    p["roi_over"] = np.where(p["over"] == 1, p["dec_over"] - 1.0, -1.0)

    # shock, standardized WITHIN market so a yard and a reception are
    # comparable, and so the threshold rule in section 4 means the same thing
    # in every market.
    p["shock_raw"] = p["prev1"] - p["base"]
    p["shock"] = p.groupby("market")["shock_raw"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    print(f"  {len(p):,} settled props with lagged form")
    print(f"  markets: {sorted(p['market'].unique())}")
    return p


def gate(p):
    _rule("SECTION 1: GATE on the hand-rolled clustered covariance")
    print("ols_cluster is hand-written, so it must reproduce")
    print("eval_harness.ols_clustered exactly on the single-regressor case")
    print("before any two-regressor result from it is trusted.")
    print()
    g = p[p["market"] == p["market"].mode().iloc[0]].dropna(
        subset=["prev1", "actual"])
    ref = eh.ols_clustered(g["prev1"], g["actual"], g["event_id"])
    X = np.column_stack([np.ones(len(g)), g["prev1"].to_numpy()])
    mine = ols_cluster(X, g["actual"].to_numpy(), g["event_id"].to_numpy())
    if ref is None or mine is None:
        print("  ABORT: could not fit the comparison.")
        sys.exit(1)
    db = abs(mine["beta"][1] - ref["beta"])
    ds = abs(mine["se"][1] - ref["se_beta"])
    print(f"  {'':22}{'harness':>14}{'ols_cluster':>14}{'diff':>12}")
    print(f"  {'slope':22}{ref['beta']:>14.9f}{mine['beta'][1]:>14.9f}{db:>12.2e}")
    print(f"  {'clustered SE':22}{ref['se_beta']:>14.9f}{mine['se'][1]:>14.9f}{ds:>12.2e}")
    print(f"  {'n':22}{ref['n']:>14}{mine['n']:>14}")
    print(f"  {'clusters':22}{ref['clusters']:>14}{mine['G']:>14}")
    if db > 1e-8 or ds > 1e-8 or ref["n"] != mine["n"]:
        print("\n  ABORT: the covariance estimator disagrees with the harness.")
        sys.exit(1)
    print("\n  GATE PASSED.")


def section_weights(p, n_boot):
    _rule("SECTION 2: DOES THE LINE OVERWEIGHT THE LAST GAME?")
    print("Same rows, same regressors, two dependent variables.")
    print("  actual ~ prev1 + base   the weights that PREDICT the outcome")
    print("  line   ~ prev1 + base   the weights the MARKET applies")
    print()
    print("'w_prev1 diff' is market minus true. POSITIVE means the line")
    print("leans on the last game MORE than the outcome justifies, which is")
    print("the overreaction hypothesis. Intervals are a game-cluster")
    print("bootstrap of the difference, because the two fits share rows.")
    print()
    print(f"  {'market':<13}{'n':>7}{'true w1':>9}{'mkt w1':>9}{'diff':>9}"
          f"{'lo95':>9}{'hi95':>9}{'true w8':>9}{'mkt w8':>9}")
    res = {}
    rng = np.random.default_rng(0)
    for market, g in p.groupby("market"):
        g = g.dropna(subset=["prev1", "base", "line", "actual"])
        if len(g) < 400:
            continue
        X = np.column_stack([np.ones(len(g)), g["prev1"], g["base"]])
        fa = ols_cluster(X, g["actual"].to_numpy(), g["event_id"].to_numpy())
        fl = ols_cluster(X, g["line"].to_numpy(), g["event_id"].to_numpy())
        if fa is None or fl is None:
            continue
        diff = fl["beta"][1] - fa["beta"][1]

        groups = [d for _, d in g.groupby("event_id")]
        idx = np.arange(len(groups))
        ds = []
        for _ in range(max(300, n_boot // 2)):
            pick = rng.choice(idx, size=len(idx), replace=True)
            rep = pd.concat([groups[i] for i in pick], ignore_index=True)
            Xr = np.column_stack([np.ones(len(rep)), rep["prev1"], rep["base"]])
            a = ols_cluster(Xr, rep["actual"].to_numpy(),
                            np.arange(len(rep)))
            l = ols_cluster(Xr, rep["line"].to_numpy(), np.arange(len(rep)))
            if a and l:
                ds.append(l["beta"][1] - a["beta"][1])
        lo, hi = (np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan))
        star = "  <--" if np.isfinite(lo) and (lo > 0 or hi < 0) else ""
        print(f"  {market:<13}{fa['n']:>7,}{fa['beta'][1]:>9.4f}"
              f"{fl['beta'][1]:>9.4f}{diff:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
              f"{fa['beta'][2]:>9.4f}{fl['beta'][2]:>9.4f}{star}")
        res[market] = (diff, lo, hi)
    print()
    print("  A positive, interval-excluding-zero diff means the market")
    print("  overweights the last game. A NEGATIVE one means the market")
    print("  UNDERweights it, which would be the opposite of the hypothesis")
    print("  and would imply fading recency loses money.")
    return res


def section_shock(p):
    _rule("SECTION 3: THE SHOCK REGRESSION, AND THE OVER RATE BY SHOCK")
    print("(actual - line) regressed on the standardized shock")
    print("(prev1 minus baseline). A NEGATIVE slope means a player coming off")
    print("a bad game beats his line, which is the tradeable form of")
    print("overreaction.")
    print()
    print("The last three columns repeat the slope CONTROLLING for the")
    print("model's own view (projection minus line). If the shock survives")
    print("that control, it carries information the model does not already")
    print("have. If it collapses, the model already encodes it.")
    print()
    print(f"  {'market':<13}{'n':>7}{'slope':>9}{'SE':>8}{'t':>7}"
          f"{'ctrl':>9}{'SE':>8}{'t':>7}")
    for market, g in p.groupby("market"):
        g = g.dropna(subset=["shock", "line", "actual"])
        if len(g) < 400:
            continue
        y = (g["actual"] - g["line"]).to_numpy()
        X = np.column_stack([np.ones(len(g)), g["shock"]])
        f1 = ols_cluster(X, y, g["event_id"].to_numpy())
        # and controlling for the model's own view, where available
        t2 = s2 = b2 = np.nan
        has_proj = "projection" in g.columns and g["projection"].notna().any()
        if has_proj:
            gg = g.dropna(subset=["projection"])
            X2 = np.column_stack([np.ones(len(gg)), gg["shock"],
                                  gg["projection"] - gg["line"]])
            f2 = ols_cluster(X2, (gg["actual"] - gg["line"]).to_numpy(),
                             gg["event_id"].to_numpy())
            if f2:
                b2, s2 = f2["beta"][1], f2["se"][1]
                t2 = b2 / s2 if s2 else np.nan
        if f1 is None:
            continue
        t1 = f1["beta"][1] / f1["se"][1] if f1["se"][1] else np.nan
        if np.isfinite(b2):
            print(f"  {market:<13}{f1['n']:>7,}{f1['beta'][1]:>+9.4f}"
                  f"{f1['se'][1]:>8.4f}{t1:>7.2f}{b2:>+9.4f}{s2:>8.4f}"
                  f"{t2:>7.2f}")
        else:
            print(f"  {market:<13}{f1['n']:>7,}{f1['beta'][1]:>+9.4f}"
                  f"{f1['se'][1]:>8.4f}{t1:>7.2f}{'skipped':>9}"
                  f"{'-':>8}{'-':>7}")

    print()
    print("  over rate by shock quintile (1 = worst recent slump):")
    print(f"  {'market':<13}{'q':>3}{'n':>7}{'mean shock':>12}{'over':>8}")
    for market, g in p.groupby("market"):
        g = g.dropna(subset=["shock"]).copy()
        if len(g) < 500:
            continue
        try:
            g["_q"] = pd.qcut(g["shock"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        for q, gg in g.groupby("_q"):
            print(f"  {market:<13}{int(q) + 1:>3}{len(gg):>7,}"
                  f"{gg['shock'].mean():>12.3f}{gg['over'].mean():>8.4f}")
        print()


def boot_ci(profit, clusters, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"p": np.asarray(profit, float),
                       "g": np.asarray(clusters)})
    groups = [g["p"].to_numpy() for _, g in df.groupby("g")]
    if len(groups) < 10:
        return np.nan, np.nan
    idx = np.arange(len(groups))
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        out[b] = np.concatenate([groups[i] for i in pick]).mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def section_roi(p, n_boot):
    _rule("SECTION 4: WHAT THE FADE RULE PAYS AT REAL ODDS")
    print("Bet the OVER on the biggest recent slumps and the UNDER on the")
    print("biggest recent spikes. Real posted odds. The bar is the measured")
    print("hold, 4.55 percent median on yardage and 5.73 on receptions.")
    print()
    hits = []
    for thr in (0.5, 1.0, 1.5, 2.0):
        print(f"  |shock| >= {thr}")
        print(f"    {'market':<13}{'side':<7}{'bets':>7}{'win':>8}{'ROI':>9}"
              f"{'lo95':>9}{'hi95':>9}")
        for market, g in p.groupby("market"):
            lo_g = g[g["shock"] <= -thr]
            hi_g = g[g["shock"] >= thr]
            for lab, gg, col, win in (("over", lo_g, "roi_over", "over"),
                                      ("under", hi_g, "roi_under", None)):
                if len(gg) < 120:
                    continue
                roi = gg[col].mean()
                wr = gg["over"].mean() if win else 1 - gg["over"].mean()
                lo, hi = boot_ci(gg[col], gg["event_id"], n_boot)
                star = "  <--" if np.isfinite(lo) and lo > 0 else ""
                print(f"    {market:<13}{lab:<7}{len(gg):>7,}{wr:>8.4f}"
                      f"{roi:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}{star}")
                if np.isfinite(lo) and lo > 0:
                    hits.append((market, lab, thr, len(gg), roi, lo, hi))
        print()
    return hits


def section_seasons(p, hits, n_boot):
    if not hits:
        _rule("SECTION 5: SEASON SPLIT")
        print("  Nothing cleared, so there is nothing to split. That is the")
        print("  result: on this data the line does not misweight recency by")
        print("  enough to beat the vig with a mechanical fade.")
        return
    _rule("SECTION 5: DOES EACH HIT HOLD ITS SIGN EVERY SEASON?")
    for market, side, thr, n, roi, lo, hi in hits:
        col = "roi_over" if side == "over" else "roi_under"
        g = p[(p["market"] == market)]
        g = g[g["shock"] <= -thr] if side == "over" else g[g["shock"] >= thr]
        print()
        print(f"  {market} {side} |shock|>={thr}  pooled ROI {roi:+.4f}")
        print(f"    {'season':<8}{'bets':>7}{'ROI':>9}{'lo95':>9}{'hi95':>9}")
        signs = []
        for season, gg in g.groupby("season"):
            if len(gg) < 40:
                continue
            r = gg[col].mean()
            l, h = boot_ci(gg[col], gg["event_id"], max(200, n_boot // 4))
            print(f"    {season:<8}{len(gg):>7,}{r:>+9.4f}{l:>+9.4f}{h:>+9.4f}")
            signs.append(np.sign(r))
        print("    SIGN CONSISTENT" if signs and all(s > 0 for s in signs)
              else "    SIGN FLIPS, treat the pooled figure as unreliable")


def section_honesty(p, weights, hits):
    _rule("SECTION 6: READING THIS HONESTLY")
    n_tests = 4 * len(p["market"].unique()) * 2
    print(f"  fade-rule cells tested     {n_tests}")
    print(f"  expected 1-sided hits      {n_tests * 0.025:.1f}")
    print(f"  cleared                    {len(hits)}")
    print()
    print("  Section 2 is the finding worth keeping regardless of section 4.")
    print("  A weight difference that excludes zero says something TRUE about")
    print("  how the line is built, even if a mechanical rule cannot harvest")
    print("  it after vig. That is the useful case for a screen: it tells you")
    print("  which direction the line is likely wrong, and leaves the")
    print("  selection of WHICH props to a read.")
    print()
    print("  What section 2 cannot tell you is whether YOUR read picks the")
    print("  right subset. That is not in the data. It is what paper trading")
    print("  with both components logged separately would measure.")
    print()
    if weights:
        pos = [m for m, (d, lo, hi) in weights.items() if lo > 0]
        neg = [m for m, (d, lo, hi) in weights.items() if hi < 0]
        if pos:
            print(f"  markets where the line OVERweights the last game: {pos}")
        if neg:
            print(f"  markets where the line UNDERweights it: {neg}")
            print("  (in those, fading recency LOSES; leaning into it is the")
            print("   direction the data supports)")
        if not pos and not neg:
            print("  no market shows a weight difference distinguishable from")
            print("  zero. The market weights recency about right, and the")
            print("  hypothesis fails in its strongest form.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=600)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 88)
    print("RECENCY: does the line overweight the last game?")
    print(f"  seasons {seasons}   book {args.book}   boot {args.boot}")
    print("=" * 88)

    p = build(args, seasons)
    if len(p) < 2000:
        print("too few rows.")
        sys.exit(1)
    gate(p)
    weights = section_weights(p, args.boot)
    section_shock(p)
    hits = section_roi(p, args.boot)
    section_seasons(p, hits, args.boot)
    section_honesty(p, weights, hits)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
