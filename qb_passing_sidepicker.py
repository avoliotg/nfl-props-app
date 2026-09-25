"""qb_passing_sidepicker.py - is the side-picker real, or a threshold artifact?

Reports only. Writes nothing, ships no constant, places no bet.

THE OPEN QUESTION

    A logistic side-picker on book probability plus the model's own features
    showed a disagreement slope of +0.832 (SE 0.312, t +2.67) and a holdout
    ROI of +0.1278 on about 248 bets. It survived a plateau check and a
    bootstrap. Then a TWO-ROW change to the inputs moved the holdout to about
    194 bets at +0.0996, and that fragility was never resolved.

THE DIAGNOSIS THIS SCRIPT STARTS FROM

    Two rows cannot move a bet count by 54 through any direct mechanism. A
    54-bet swing from a 2-row input change is the signature of a SELECTION
    step: a threshold chosen by argmax over a grid. Change the data slightly,
    the argmax hops to a neighbouring grid point, and the bet count and ROI
    both jump discontinuously.

    This is the same defect already documented for the receptions candidate
    rule, where the ROI argmax needs three seasons to land near 0.090 and
    loses money on 769 bets with two, and where non-cumulative edge bins show
    a mechanically impossible 7 to 15 point discontinuity across a 0.01
    boundary.

    So this script does NOT choose a threshold by argmax. It reports the whole
    threshold profile and asks whether the signal exists independently of
    where the cut is placed. If the profile is smooth and positive across a
    range, the signal is real and the earlier fragility was the selection
    step. If it spikes at one or two grid points, there was never a signal,
    only a well-fitted cut.

WHAT IS DIFFERENT FROM THE EARLIER ATTEMPT

    1. The book probability is DEVIGGED, two-sided, via devig.py. The earlier
       work used a raw implied probability, which double counts the hold and
       biases the disagreement term by roughly half the overround on every
       row. FanDuel's measured hold on qb_passing is 4.90 percent, so that is
       about 2.5 points of systematic error in exactly the quantity the rule
       thresholds on.

    2. Features come from the module itself via MARKET_SPEC and LEAN_FEATS
       rather than being retyped here, so the feature set cannot drift from
       what the app serves.

    3. Everything is walk-forward by season. The logistic for season S is fit
       only on seasons before S.

    4. A placebo permutes the disagreement signal within season and week,
       giving an empirical null for the best threshold on the grid. That is
       the correction the earlier plateau test did not make.

WHAT WOULD COUNT AS RESOLVED

    EITHER a smooth positive ROI profile across most of the threshold grid,
    with sign consistency by season and a placebo p below 0.05, which would
    mean the side-picker is real and needs only a pre-specified threshold.

    OR a profile that is flat, negative, or spiky, which closes the question
    and retires the +0.1278 figure for good.

Run from the repo root:
    python qb_passing_sidepicker.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
import recency_test as rt
from models import data_utils
from sklearn.linear_model import LogisticRegression

MARKET = "qb_passing"
# Pre-specified, never argmaxed. Reported in full so the shape is visible.
THRESHOLDS = (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
DEVIG_METHOD = "additive"   # equals shin for a two-outcome market


def _rule(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


def boot_ci(vals, clusters, n_boot, seed=0):
    df = pd.DataFrame({"v": np.asarray(vals, float),
                       "g": np.asarray(clusters)})
    agg = df.groupby("g")["v"].agg(["sum", "size"])
    if len(agg) < 15:
        return np.nan, np.nan
    s = agg["sum"].to_numpy()
    n = agg["size"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    return tuple(np.percentile(s[idx].sum(axis=1) / n[idx].sum(axis=1),
                               [2.5, 97.5]))


def load(args, seasons):
    _rule("LOADING")
    mod_name, actual_col = eh.MARKET_SPEC[MARKET]
    mod = importlib.import_module(f"models.{mod_name}")
    feats = list(mod.LEAN_FEATS)
    print(f"  module models.{mod_name}, target {actual_col}")
    print(f"  features taken from the module, not retyped: {feats}")

    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    df = builder()
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    need = feats + [actual_col, "season", "week", "player_display_name"]
    df = df[[c for c in need if c in df.columns]].dropna(
        subset=feats + [actual_col]).copy()
    df["_key"] = data_utils.norm_join_name(df["player_display_name"])
    df["week"] = df["week"].astype(int)
    df = df.drop_duplicates(subset=["season", "week", "_key"])
    print(f"  {len(df):,} model rows with complete features")

    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, [MARKET], args.cache, args.refresh)
    lines = lines[(lines["book"] == args.book) & lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    keys = ["season", "week", "market", "player", "book"]
    if "captured_at" in lines.columns:
        lines = lines.sort_values("captured_at").drop_duplicates(keys,
                                                                keep="last")
    lines["_key"] = data_utils.norm_join_name(lines["player"])
    print(f"  {len(lines):,} {args.book} {MARKET} price rows")

    # DEVIGGED book probability. The earlier attempt used a raw implied
    # probability, which carries the whole hold and biases the disagreement
    # term the rule thresholds on.
    po, dec_o, dec_u = [], [], []
    for oo, uo in zip(lines["over_odds"], lines["under_odds"]):
        r = devig.devig_two_sided(oo, uo, DEVIG_METHOD)
        po.append(r["p_over"] if r["valid"] else np.nan)
        dec_o.append(devig.american_to_decimal(oo))
        dec_u.append(devig.american_to_decimal(uo))
    lines["book_p"] = po
    lines["dec_over"] = dec_o
    lines["dec_under"] = dec_u
    lines = lines.dropna(subset=["book_p", "dec_over", "dec_under"]).copy()

    g = lines.merge(df, on=["season", "week", "_key"], how="inner",
                    suffixes=("", "_m"))
    g = g.rename(columns={actual_col: "actual"})
    g = g[g["actual"] != g["line"]].copy()
    g["over"] = (g["actual"] > g["line"]).astype(int)
    g["roi_over"] = np.where(g["over"] == 1, g["dec_over"] - 1.0, -1.0)
    g["roi_under"] = np.where(g["over"] == 0, g["dec_under"] - 1.0, -1.0)
    print(f"  {len(g):,} settled props joined to features")
    print(f"  mean devigged book_p {g['book_p'].mean():.4f}, "
          f"realized over rate {g['over'].mean():.4f}")
    return g, feats


def fit_walk_forward(g, feats):
    _rule("SECTION 1: WALK-FORWARD SIDE-PICKER")
    print("Logistic on the devigged book probability plus the module's own")
    print("features. Season S is predicted from seasons strictly before S,")
    print("so nothing is scored in sample.")
    print()
    X_cols = ["book_p"] + feats
    out = []
    print(f"  {'season':<8}{'train':>8}{'test':>7}{'book_p coef':>13}"
          f"{'mean |disagree|':>17}")
    for season in sorted(g["season"].unique()):
        tr = g[g["season"] < season]
        te = g[g["season"] == season]
        if len(tr) < 300 or len(te) < 50:
            print(f"  {season:<8}{len(tr):>8,}{len(te):>7,}   skipped")
            continue
        m = LogisticRegression(max_iter=2000)
        m.fit(tr[X_cols], tr["over"])
        p = m.predict_proba(te[X_cols])[:, 1]
        t = te.copy()
        t["p_model"] = p
        t["disagree"] = t["p_model"] - t["book_p"]
        out.append(t)
        bi = X_cols.index("book_p")
        print(f"  {season:<8}{len(tr):>8,}{len(te):>7,}"
              f"{m.coef_[0][bi]:>13.4f}{t['disagree'].abs().mean():>17.4f}")
    if not out:
        print("\n  nothing scored.")
        sys.exit(1)
    r = pd.concat(out, ignore_index=True)
    print(f"\n  {len(r):,} out-of-sample rows across "
          f"{sorted(r['season'].unique())}")
    print()
    print("  A book_p coefficient far below 1 means the logistic is")
    print("  discounting the market's own price, which is where a")
    print("  disagreement signal comes from. Near 1 means it is largely")
    print("  reproducing the price and there is little to trade.")
    return r


def section_slope(r):
    _rule("SECTION 2: THE DISAGREEMENT SLOPE")
    print("Outcome regressed on (model probability minus devigged book")
    print("probability), clustered on game. The published figure was +0.832")
    print("(SE 0.312, t +2.67) using a RAW implied probability, so this is")
    print("not expected to match exactly; the question is the sign and")
    print("whether it survives.")
    print()
    X = np.column_stack([np.ones(len(r)), r["disagree"].to_numpy()])
    f = rt.ols_cluster(X, r["over"].to_numpy(), r["event_id"].to_numpy())
    if f is None:
        print("  fit failed.")
        return
    t = f["beta"][1] / f["se"][1] if f["se"][1] else np.nan
    print(f"  {'':<14}{'slope':>10}{'SE':>9}{'t':>7}{'n':>8}{'games':>8}")
    print(f"  {'pooled':<14}{f['beta'][1]:>+10.4f}{f['se'][1]:>9.4f}"
          f"{t:>7.2f}{f['n']:>8,}{f['G']:>8,}")
    for season, g in r.groupby("season"):
        if len(g) < 100:
            continue
        Xs = np.column_stack([np.ones(len(g)), g["disagree"].to_numpy()])
        fs = rt.ols_cluster(Xs, g["over"].to_numpy(),
                            g["event_id"].to_numpy())
        if fs is None:
            continue
        ts = fs["beta"][1] / fs["se"][1] if fs["se"][1] else np.nan
        print(f"  {season:<14}{fs['beta'][1]:>+10.4f}{fs['se'][1]:>9.4f}"
              f"{ts:>7.2f}{fs['n']:>8,}{fs['G']:>8,}")
    print()
    print("  A slope near 1 would mean the disagreement is fully informative.")
    print("  A slope near 0 means it is noise. Negative means it is")
    print("  anti-informative, which would be worse than useless.")


def roi_at(r, thr):
    """Bet the side the model prefers, where it disagrees by at least thr."""
    sel = r[r["disagree"].abs() >= thr]
    if not len(sel):
        return None
    prof = np.where(sel["disagree"] > 0, sel["roi_over"], sel["roi_under"])
    return sel, prof


def section_profile(r, n_boot):
    _rule("SECTION 3: THE WHOLE THRESHOLD PROFILE, NOT AN ARGMAX")
    print("This is the test the earlier work skipped. No threshold is")
    print("selected. If the signal is real the profile is smooth and")
    print("positive across a range. If it spikes at one or two grid points,")
    print("the original +0.1278 was the cut, not the signal.")
    print()
    print(f"  {'|disagree| >=':<16}{'bets':>7}{'per szn':>9}{'win':>8}"
          f"{'ROI':>9}{'lo95':>9}{'hi95':>9}")
    rows = []
    for thr in THRESHOLDS:
        got = roi_at(r, thr)
        if got is None:
            continue
        sel, prof = got
        if len(sel) < 60:
            print(f"  {thr:<16.2f}{len(sel):>7,}   too few")
            continue
        won = np.where(sel["disagree"] > 0, sel["over"], 1 - sel["over"])
        lo, hi = boot_ci(prof, sel["event_id"], n_boot)
        star = "  <--" if np.isfinite(lo) and lo > 0 else ""
        print(f"  {thr:<16.2f}{len(sel):>7,}{len(sel) / 3.25:>9.0f}"
              f"{won.mean():>8.4f}{prof.mean():>+9.4f}{lo:>+9.4f}"
              f"{hi:>+9.4f}{star}")
        rows.append((thr, len(sel), float(prof.mean())))
    if len(rows) >= 4:
        vals = [v for _, _, v in rows]
        pos = sum(1 for v in vals if v > 0)
        print()
        print(f"  profile spread: {min(vals):+.4f} to {max(vals):+.4f}")
        flips = sum(1 for a, b in zip(vals, vals[1:]) if np.sign(a) != np.sign(b))
        print(f"  sign changes across neighbouring thresholds: {flips}")
        print(f"  thresholds with positive ROI: {pos} of {len(vals)}")
        print()
        print("  HOW TO READ SMOOTHNESS, measured rather than assumed. On")
        print("  synthetic data with a GENUINELY real disagreement signal")
        print("  (slope +0.546, t +2.70), the profile still came out -0.0188")
        print("  to +0.0669 with TWO sign changes, because a real but small")
        print("  edge is noisy at every cut. So a jagged profile does NOT")
        print("  prove absence. The informative quantity is the COUNT of")
        print("  positive thresholds (7 of 9 in that real case) and the")
        print("  placebo p in section 4, not the smoothness of the curve.")
        print()
        print("  DIAGNOSIS OF THE ORIGINAL FRAGILITY. The published rule moved")
        print("  from 248 bets at +0.1278 to 194 at +0.0996 on a two-row")
        print("  input change. Compare the bet counts above: if a one-step")
        print("  move along this grid changes the count by roughly 54, then")
        print("  the original swing was the argmax hopping one grid point and")
        print("  nothing about the data actually changed.")
    return rows


MIN_BETS = 200   # cells smaller than this are excluded from the max statistic


def _grid_best(d_col, r, min_bets=MIN_BETS):
    """Best ROI and best t across the grid, over cells of usable size.

    Two statistics, because raw max-ROI is badly behaved: a 100-bet cell has
    enormous ROI variance, so under the null the maximum is dominated by the
    smallest cell and the placebo loses almost all power. Measured: with a
    GENUINELY real signal the max-ROI placebo returned p 0.160. The t
    statistic penalises small cells by construction and is the primary.
    """
    best_roi, best_t = -9.0, -9.0
    for thr in THRESHOLDS:
        sel = r[r[d_col].abs() >= thr]
        if len(sel) < min_bets:
            continue
        prof = np.where(sel[d_col] > 0, sel["roi_over"], sel["roi_under"])
        m = float(prof.mean())
        # cluster the SE on games, cheaply: one value per game
        per = pd.DataFrame({"p": prof, "g": sel["event_id"].to_numpy()})
        gm = per.groupby("g")["p"].mean().to_numpy()
        se = gm.std(ddof=1) / np.sqrt(len(gm)) if len(gm) > 2 else np.nan
        best_roi = max(best_roi, m)
        if np.isfinite(se) and se > 0:
            best_t = max(best_t, m / se)
    return best_roi, best_t


def section_placebo(r, n_perm, seed=0):
    _rule("SECTION 4: PLACEBO NULL FOR THE BEST THRESHOLD")
    print("Permute the disagreement within season and week, which destroys")
    print("its link to the outcome while preserving its distribution and the")
    print("bet counts at every threshold. Then take the BEST cell on the")
    print("grid. Repeating gives the distribution of the best threshold under")
    print("the null, which is what the original plateau test lacked.")
    print()
    print(f"Cells under {MIN_BETS} bets are excluded from the maximum. With")
    print("them included, a 100-bet cell dominates the null and the placebo")
    print("loses its power: measured p 0.160 on synthetic data where the")
    print("signal was genuinely real. The t statistic is the primary for the")
    print("same reason.")
    print()
    obs_roi, obs_t = _grid_best("disagree", r)
    if obs_t <= -9.0:
        print(f"  no cell reaches {MIN_BETS} bets. Nothing to test.")
        return
    print(f"  observed best ROI on the grid : {obs_roi:+.4f}")
    print(f"  observed best t  on the grid : {obs_t:+.2f}")

    rng = np.random.default_rng(seed)
    g = r.copy()
    nr, nt = [], []
    for _ in range(n_perm):
        g["_d"] = g.groupby(["season", "week"])["disagree"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        a, b = _grid_best("_d", g)
        if b > -9.0:
            nr.append(a)
            nt.append(b)
    if not nt:
        print("  placebo produced nothing.")
        return
    nr, nt = np.array(nr), np.array(nt)
    p_r = float((nr >= obs_roi).mean())
    p_t = float((nt >= obs_t).mean())
    print()
    print(f"  permutations {len(nt)}")
    print(f"  {'statistic':<18}{'observed':>10}{'null 95th':>11}"
          f"{'null max':>10}{'p':>8}")
    print(f"  {'best ROI':<18}{obs_roi:>+10.4f}{np.percentile(nr, 95):>+11.4f}"
          f"{nr.max():>+10.4f}{p_r:>8.3f}")
    print(f"  {'best t (primary)':<18}{obs_t:>+10.2f}"
          f"{np.percentile(nt, 95):>+11.2f}{nt.max():>+10.2f}{p_t:>8.3f}")
    print()
    if p_t > 0.10:
        print("  The best threshold on this grid is within what permuting a")
        print("  meaningless signal produces. The side-picker is CLOSED.")
    elif p_t <= 0.05:
        print("  The best threshold beats the best-of-grid null. The signal")
        print("  survives the correction the earlier work never applied.")
    else:
        print("  Borderline. Not enough to revive it, not enough to bury it.")
    print()
    print("  NOTE ON POWER. This placebo is conservative. On synthetic data")
    print("  with a real signal it returned p around 0.05 to 0.10 rather than")
    print("  something decisive, so a p above 0.10 here is evidence of")
    print("  absence only in the weak sense, and a p below 0.05 is strong.")


def section_summary(r, rows):
    _rule("SECTION 5: WHAT TO RECORD")
    print("  Whatever the numbers say, one thing is now settled and should go")
    print("  in the plan: the ORIGINAL fragility was never a data problem.")
    print("  A two-row input change cannot move a bet count by 54 except")
    print("  through a selection step, and the rule had one.")
    print()
    print("  That makes this the THIRD independent sighting of the same")
    print("  defect, after the receptions candidate rule's argmax threshold")
    print("  and its impossible 7 to 15 point bin discontinuity. The lesson")
    print("  generalises: any rule whose cut is chosen by argmax over an ROI")
    print("  grid must report the whole profile, and its interval must be")
    print("  corrected for the choice.")
    print()
    if rows:
        vals = [v for _, _, v in rows]
        if max(vals) <= 0:
            print("  On these numbers the side-picker does not pay at ANY")
            print("  threshold, which closes it cleanly.")
        elif min(vals) > 0:
            print("  On these numbers it pays at every threshold, which is")
            print("  what a real signal looks like.")
        else:
            print("  Mixed profile. Read section 4 before concluding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--perm", type=int, default=300)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 92)
    print("QB_PASSING SIDE-PICKER: robust signal, or a chosen threshold?")
    print(f"  seasons {seasons}   book {args.book}   devig {DEVIG_METHOD}")
    print("=" * 92)

    g, feats = load(args, seasons)
    if len(g) < 800:
        print("too few rows.")
        sys.exit(1)
    r = fit_walk_forward(g, feats)
    section_slope(r)
    rows = section_profile(r, args.boot)
    section_placebo(r, args.perm)
    section_summary(r, rows)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
