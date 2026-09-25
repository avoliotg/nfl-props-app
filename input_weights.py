"""input_weights.py - does the line misweight the INPUTS, not the output?

Reports only. Writes nothing, places no bet.

WHY THIS IS THE RIGHT FOLLOW-UP TO recency_test.py

    recency_test measured whether the line overweights recent YARDS and found
    it does not: FanDuel's weight on the last game matched the true
    predictive weight in all five markets, every interval straddling zero,
    and the over rate was flat across shock quintiles.

    But that test used the OUTPUT stat as both the signal and the target. The
    interesting version of the argument is that a player can produce yards
    while the underlying process diverges from them: target volume, air
    yards, snap share, carry share. A receiver who is seeing the same targets
    but dropping them has bad recent yards and unchanged opportunity. A
    receiver whose target count is quietly collapsing has the same recent
    yards and a genuinely worse outlook. Recent yards cannot tell those
    apart. The market obviously tracks recent yards. Whether it correctly
    weights OPPORTUNITY is a separate and much less obvious question.

THE METHOD, SIMPLIFIED FROM recency_test SECTION 2

    That section fitted two regressions and bootstrapped the difference of a
    coefficient. That machinery was unnecessary. When two regressions share
    the same design matrix X, the difference of their coefficients is exactly
    the coefficient from regressing the DIFFERENCE of the dependent
    variables. So:

        (actual - line) ~ recent_input + base_input

    gives the weight gap directly, with a proper clustered standard error and
    no bootstrap at all. Section 1 GATES this identity numerically against
    the two-regression route before anything is reported, because "these two
    procedures are algebraically the same" is exactly the sort of claim this
    project has learned to verify rather than assert.

    Sign convention: the dependent variable is actual MINUS line, so a
    NEGATIVE coefficient on recent_input means the line moves MORE with that
    input than the outcome justifies, which is overweighting. A POSITIVE
    coefficient means the line underweights it, and that is the direction
    worth hoping for, because underweighted opportunity is unpriced
    information rather than a mispriced reaction.

    recent and base windows do NOT overlap: recent is the last three games,
    base the eight before those. So the two regressors are not sharing
    observations, and the coefficient on recent is a genuine partial.

WHAT WOULD BE A FINDING

    A coefficient whose clustered interval excludes zero, on an input a book
    plausibly underuses, that holds its sign across seasons, and that
    survives controlling for the model's own projection. The last control
    matters most: the models already carry targets_roll, carries_roll,
    attempts_roll and snap_roll, so an input that only looks informative
    until the projection is added is something the project already has.

CONFOUNDS

    Opportunity is endogenous. A player whose targets are falling is often
    falling for a reason the market can see and this test cannot, so a
    positive coefficient is an upper bound on exploitable information.

    Snap counts come from a separate nflverse table and the join has a
    documented history of silently dropping rows on name formatting. The
    join rate is reported rather than assumed.

Run from the repo root:
    python input_weights.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
import recency_test as rt
from models import data_utils

SEASONS_STATS = [2022, 2023, 2024, 2025, 2026]

MARKET_STAT = {
    "receiving": "receiving_yards",
    "receptions": "receptions",
    "rushing": "rushing_yards",
    "qb_rushing": "rushing_yards",
    "qb_passing": "passing_yards",
}

# Inputs to test per market. These are OPPORTUNITY measures, not outputs.
# The output stat is deliberately included once per market as a control, so
# the table shows the known-null result next to the new ones.
MARKET_INPUTS = {
    "receiving": ["targets", "receiving_air_yards",
                  "receiving_yards_after_catch", "target_share",
                  "air_yards_share", "wopr", "offense_pct", "receiving_yards"],
    "receptions": ["targets", "receiving_air_yards", "target_share",
                   "air_yards_share", "wopr", "offense_pct", "receptions"],
    "rushing": ["carries", "offense_pct", "rushing_yards"],
    "qb_rushing": ["carries", "attempts", "rushing_yards"],
    "qb_passing": ["attempts", "completions", "offense_pct", "passing_yards"],
}


def _rule(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


def load_history(wanted):
    """Lagged recent and base windows for every requested input column."""
    ps = data_utils.load_player_stats(SEASONS_STATS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    have = [c for c in wanted if c in ps.columns]
    missing = [c for c in wanted if c not in ps.columns]
    if missing:
        print(f"  not in load_player_stats, skipped: {missing}")
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps["_key"] = data_utils.norm_join_name(ps["player_display_name"])

    idc = next((c for c in ("player_id", "gsis_id") if c in ps.columns), None)
    if idc:
        nid = ps.groupby(["season", "week", "_key"])[idc].transform("nunique")
        nb = int((nid > 1).sum())
        if nb:
            print(f"  dropped {nb} rows on normalized-name collisions")
        ps = ps[nid == 1]

    # snap share lives in a different table, and this join has a documented
    # history of dropping rows on name formatting
    if "offense_pct" in wanted:
        try:
            sc = data_utils.load_snap_counts(SEASONS_STATS)
            sc = sc.to_pandas() if hasattr(sc, "to_pandas") else sc
            name_col = next((c for c in ("player", "player_name",
                                         "player_display_name")
                             if c in sc.columns), None)
            if name_col and "offense_pct" in sc.columns:
                sc["_key"] = data_utils.norm_join_name(sc[name_col])
                sc = sc.drop_duplicates(subset=["season", "week", "_key"])
                before = len(ps)
                ps = ps.merge(sc[["season", "week", "_key", "offense_pct"]],
                              on=["season", "week", "_key"], how="left")
                rate = ps["offense_pct"].notna().mean()
                print(f"  snap-count join: {rate:.1%} of {before:,} rows "
                      f"matched")
                if "offense_pct" not in have:
                    have.append("offense_pct")
            else:
                print("  snap counts lack a usable name or offense_pct column")
        except Exception as e:
            print(f"  snap counts unavailable: {type(e).__name__}: {e}")

    for c in have:
        if c != "offense_pct":
            ps[c] = ps[c].fillna(0.0)

    key = idc if idc else "_key"
    ps = ps.sort_values([key, "season", "week"]).reset_index(drop=True)
    out_cols = ["season", "week", "_key"]
    for c in have:
        gb = ps.groupby(key)[c]
        ps[f"{c}__recent"] = gb.transform(
            lambda s: s.shift(1).rolling(3, min_periods=3).mean())
        ps[f"{c}__base"] = gb.transform(
            lambda s: s.shift(4).rolling(8, min_periods=4).mean())
        out_cols += [f"{c}__recent", f"{c}__base"]
    stats = [c for c in set(MARKET_STAT.values()) if c in ps.columns]
    return ps[out_cols + stats], have


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
    lines, _ = data_utils.split_qb_rushing(lines, min_match_rate=0.98)
    lines = lines[lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    if args.book != "all":
        lines = lines[lines["book"] == args.book]
    keys = ["season", "week", "market", "player", "book"]
    if "captured_at" in lines.columns:
        lines = lines.sort_values("captured_at").drop_duplicates(keys, keep="last")
    lines["dec_over"] = [devig.american_to_decimal(o) for o in lines["over_odds"]]
    lines["dec_under"] = [devig.american_to_decimal(o) for o in lines["under_odds"]]
    lines = lines[lines["dec_over"].notna() & lines["dec_under"].notna()].copy()
    lines["_key"] = data_utils.norm_join_name(lines["player"])
    print(f"  {len(lines):,} priced rows, book={args.book}")

    wanted = sorted({c for v in MARKET_INPUTS.values() for c in v})
    hist, have = load_history(wanted)

    proj = []
    for market in sorted(lines["market"].unique()):
        try:
            s = eh.score_market(market, seasons, mode="walk_forward",
                                population="all")
        except Exception:
            continue
        if len(s):
            proj.append(s)
    if proj:
        proj = pd.concat(proj, ignore_index=True)
        proj["_key"] = data_utils.norm_join_name(proj["player"])
        proj["week"] = proj["week"].astype(int)
        proj = proj.drop_duplicates(subset=["season", "week", "market", "_key"])
    else:
        proj = None

    out = []
    for market, stat in MARKET_STAT.items():
        sel = lines[lines["market"] == market]
        if not len(sel) or stat not in hist.columns:
            continue
        cols = ["season", "week", "_key", stat]
        for c in MARKET_INPUTS.get(market, []):
            for suf in ("recent", "base"):
                nm = f"{c}__{suf}"
                if nm in hist.columns:
                    cols.append(nm)
        m = sel.merge(hist[sorted(set(cols))], on=["season", "week", "_key"],
                      how="inner")
        m = m.rename(columns={stat: "actual"})
        out.append(m)
    p = pd.concat(out, ignore_index=True)
    p = p.dropna(subset=["actual", "line"]).copy()
    p = p[p["actual"] != p["line"]].copy()
    p["over"] = (p["actual"] > p["line"]).astype(int)
    p["roi_under"] = np.where(p["over"] == 0, p["dec_under"] - 1.0, -1.0)
    p["roi_over"] = np.where(p["over"] == 1, p["dec_over"] - 1.0, -1.0)
    p["resid"] = p["actual"] - p["line"]
    if proj is not None:
        p = p.merge(proj[["season", "week", "market", "_key", "projection"]],
                    on=["season", "week", "market", "_key"], how="left")
    print(f"  {len(p):,} settled props with lagged inputs")
    return p, have


def gate(p):
    _rule("SECTION 1: GATE on the algebraic shortcut")
    print("Claim: regressing (actual - line) on X gives exactly the")
    print("difference between the coefficients from regressing line on X and")
    print("actual on X. If true, recency_test's bootstrap of that difference")
    print("was unnecessary machinery and this script can use clustered SEs")
    print("directly. Verify rather than assert.")
    print()
    g = p[p["market"] == "receiving"].dropna(
        subset=["targets__recent", "targets__base", "line", "actual"])
    if len(g) < 500:
        print("  ABORT: not enough receiving rows to gate.")
        sys.exit(1)
    X = np.column_stack([np.ones(len(g)), g["targets__recent"],
                         g["targets__base"]])
    ev = g["event_id"].to_numpy()
    fa = rt.ols_cluster(X, g["actual"].to_numpy(), ev)
    fl = rt.ols_cluster(X, g["line"].to_numpy(), ev)
    fd = rt.ols_cluster(X, (g["actual"] - g["line"]).to_numpy(), ev)
    if not (fa and fl and fd):
        print("  ABORT: a fit failed.")
        sys.exit(1)
    print(f"  {'':<24}{'actual~X':>13}{'line~X':>13}{'diff of two':>14}"
          f"{'direct':>13}{'gap':>11}")
    ok = True
    for i, nm in enumerate(["intercept", "targets recent", "targets base"]):
        d = fa["beta"][i] - fl["beta"][i]
        gap = abs(d - fd["beta"][i])
        print(f"  {nm:<24}{fa['beta'][i]:>13.8f}{fl['beta'][i]:>13.8f}"
              f"{d:>14.8f}{fd['beta'][i]:>13.8f}{gap:>11.2e}")
        if gap > 1e-8:
            ok = False
    if not ok:
        print("\n  ABORT: the identity does not hold numerically.")
        sys.exit(1)
    print("\n  GATE PASSED. Using the direct regression with clustered SEs.")


def section_weights(p, have):
    _rule("SECTION 2: WEIGHT GAP PER INPUT")
    print("(actual - line) ~ recent_input + base_input, clustered on game.")
    print()
    print("NEGATIVE recent coefficient = the line moves MORE with that input")
    print("than the outcome justifies (OVERweights it).")
    print("POSITIVE = the line UNDERweights it, which is unpriced")
    print("opportunity and the more interesting direction.")
    print()
    print("'ctrl' repeats the recent coefficient controlling for the model's")
    print("projection minus line. The models already carry targets_roll,")
    print("carries_roll, attempts_roll and snap_roll, so a coefficient that")
    print("dies under that control is something the project already has.")
    print()
    print(f"  {'market':<12}{'input':<28}{'n':>7}{'recent':>10}{'SE':>9}"
          f"{'t':>7}{'base':>10}{'ctrl':>10}{'SE':>9}{'t':>7}")
    findings = []
    for market, g0 in p.groupby("market"):
        for inp in MARKET_INPUTS.get(market, []):
            rc, bc = f"{inp}__recent", f"{inp}__base"
            if rc not in g0.columns or bc not in g0.columns:
                continue
            g = g0.dropna(subset=[rc, bc, "resid"])
            if len(g) < 400:
                continue
            X = np.column_stack([np.ones(len(g)), g[rc], g[bc]])
            f = rt.ols_cluster(X, g["resid"].to_numpy(),
                               g["event_id"].to_numpy())
            if f is None:
                continue
            t = f["beta"][1] / f["se"][1] if f["se"][1] else np.nan
            cb = cs = ct = np.nan
            if "projection" in g.columns and g["projection"].notna().any():
                gg = g.dropna(subset=["projection"])
                if len(gg) > 400:
                    X2 = np.column_stack([np.ones(len(gg)), gg[rc], gg[bc],
                                          gg["projection"] - gg["line"]])
                    f2 = rt.ols_cluster(X2, gg["resid"].to_numpy(),
                                        gg["event_id"].to_numpy())
                    if f2:
                        cb, cs = f2["beta"][1], f2["se"][1]
                        ct = cb / cs if cs else np.nan
            star = "  <--" if abs(t) > 2.5 else ""
            print(f"  {market:<12}{inp:<28}{f['n']:>7,}{f['beta'][1]:>+10.4f}"
                  f"{f['se'][1]:>9.4f}{t:>7.2f}{f['beta'][2]:>+10.4f}"
                  f"{cb:>+10.4f}{cs:>9.4f}{ct:>7.2f}{star}")
            if abs(t) > 2.5:
                findings.append((market, inp, f["beta"][1], f["se"][1], t,
                                 cb, ct))
        print()
    n_tests = sum(len(MARKET_INPUTS.get(m, [])) for m in p["market"].unique())
    print(f"  tests run {n_tests}, expected |t|>2.5 by luck "
          f"{n_tests * 0.0124:.1f}, flagged {len(findings)}")
    return findings


def section_seasons(p, findings):
    if not findings:
        _rule("SECTION 3: SEASON SPLIT")
        print("  Nothing flagged. The line weights opportunity about as well")
        print("  as it weights recent output, which closes this question the")
        print("  same way recency_test closed the other one.")
        return
    _rule("SECTION 3: DO THE FLAGS HOLD SIGN ACROSS SEASONS?")
    for market, inp, b, se, t, cb, ct in findings:
        rc, bc = f"{inp}__recent", f"{inp}__base"
        g0 = p[p["market"] == market].dropna(subset=[rc, bc, "resid"])
        print()
        print(f"  {market} / {inp}   pooled {b:+.4f} (t {t:+.2f}), "
              f"controlled {cb:+.4f} (t {ct:+.2f})")
        print(f"    {'season':<8}{'n':>7}{'recent':>10}{'SE':>9}{'t':>7}")
        signs = []
        for season, g in g0.groupby("season"):
            if len(g) < 150:
                continue
            X = np.column_stack([np.ones(len(g)), g[rc], g[bc]])
            f = rt.ols_cluster(X, g["resid"].to_numpy(),
                               g["event_id"].to_numpy())
            if f is None:
                continue
            tt = f["beta"][1] / f["se"][1] if f["se"][1] else np.nan
            print(f"    {season:<8}{f['n']:>7,}{f['beta'][1]:>+10.4f}"
                  f"{f['se'][1]:>9.4f}{tt:>7.2f}")
            signs.append(np.sign(f["beta"][1]))
        print("    SIGN CONSISTENT" if signs and len(set(signs)) == 1
              else "    SIGN FLIPS, unreliable")


def section_roi(p, findings, n_boot):
    if not findings:
        return
    _rule("SECTION 4: WHAT THE IMPLIED RULE PAYS AT REAL ODDS")
    print("A weight gap is not money. Bet the side the gap implies, on the")
    print("rows where the input deviates most from its own baseline, at the")
    print("actual posted price.")
    import under_bias as ub
    for market, inp, b, se, t, cb, ct in findings:
        rc, bc = f"{inp}__recent", f"{inp}__base"
        g0 = p[p["market"] == market].dropna(subset=[rc, bc]).copy()
        if len(g0) < 400:
            continue
        g0["_dev"] = g0[rc] - g0[bc]
        g0["_z"] = (g0["_dev"] - g0["_dev"].mean()) / g0["_dev"].std(ddof=0)
        # positive b means the line UNDERweights the input, so a HIGH recent
        # input is under-reflected in the line and the OVER is the side.
        over_high = b > 0
        print()
        print(f"  {market} / {inp}   coefficient {b:+.4f} -> "
              f"{'high input means OVER' if over_high else 'high input means UNDER'}")
        print(f"    {'cut':<24}{'bets':>7}{'ROI':>9}{'lo95':>9}{'hi95':>9}")
        for thr in (0.5, 1.0, 1.5):
            hi = g0[g0["_z"] >= thr]
            lo = g0[g0["_z"] <= -thr]
            for lab, gg, col in ((f"z>=+{thr}", hi,
                                  "roi_over" if over_high else "roi_under"),
                                 (f"z<=-{thr}", lo,
                                  "roi_under" if over_high else "roi_over")):
                if len(gg) < 120:
                    continue
                roi = gg[col].mean()
                l, h, _ = ub.boot_ci(gg[col], gg["event_id"], n_boot)
                star = "  <--" if np.isfinite(l) and l > 0 else ""
                print(f"    {lab:<24}{len(gg):>7,}{roi:>+9.4f}"
                      f"{l:>+9.4f}{h:>+9.4f}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=800)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 92)
    print("INPUT WEIGHTS: does the line misweight OPPORTUNITY?")
    print(f"  seasons {seasons}   book {args.book}")
    print("=" * 92)

    p, have = build(args, seasons)
    if len(p) < 2000:
        print("too few rows.")
        sys.exit(1)
    gate(p)
    findings = section_weights(p, have)
    section_seasons(p, findings)
    section_roi(p, findings, args.boot)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
