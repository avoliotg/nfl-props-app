"""qb_beta_compare.py - the harness's refit against the SHIPPED model, paired.

ITEM 3 of the qb_rushing sequence, option C.

WHY THIS SCRIPT EXISTS

  eval_harness does not call load_model. It fits its own LinearRegression at
  the line its docstring flags as future work, borrowing only build_dataset
  and the feature list. So every beta the project has ever published
  (receptions 0.275, receiving 0.093, rushing -0.032, qb_passing 0.075)
  describes a REFITTED model, not the model the app serves.

  That gap is documented as "about 0.1 receptions on served projections". It
  has never been measured in BETA units, which is the unit every decision in
  this project is made in.

  Rewiring the harness was considered and rejected for a good reason:
  load_model was a single fit pinned to season <= 2024, so calling it would
  score 2023 and 2024 with a model trained on them and destroy the
  leave-season-out discipline. That objection no longer applies to
  qb_rushing, whose load_model now takes train_max_season. So this market is
  the one place the gap can be measured honestly, walk-forward, on both arms
  at once.

WHAT IT DOES

  Builds the prop set ONCE, scores it two ways on the SAME rows, and fits the
  SAME regression to both:

    arm A  eval_harness.score_market, imported not restated. The refit.
    arm B  models.qb_rushing.load_model(train_max_season=season-1). Shipped.

  Then reports each beta, the paired difference, and a cluster bootstrap
  interval for the difference.

WHAT IT HONESTLY DOES NOT DO

  The bootstrap resamples game clusters and refits the REGRESSION, holding
  each arm's projections fixed. It therefore gives the sampling variance of
  the beta difference on this row set, and does NOT propagate the variance of
  refitting the underlying models on resampled data. Doing that properly
  means refitting two LinearRegressions per replicate per season, which is
  affordable but would take minutes rather than seconds. Read the interval as
  "given these projections, do the two betas differ", not as a full
  accounting.

  And one data point is one data point. This measures the gap for a
  two-feature QB model. It does not establish the gap for receptions, which
  is the market that matters. If the gap is small here that is reassuring
  rather than conclusive.

Run from the repo root:
    python qb_beta_compare.py --all-seasons --cache lines_cache.parquet
    python qb_beta_compare.py --all-seasons --cache lines_cache.parquet --boot 500
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh
from models import data_utils, qb_rushing

MARKET = "qb_rushing"


def _rule(t):
    print()
    print("=" * 80)
    print(t)
    print("=" * 80)


def build_props(args, seasons):
    _rule("PROP SET (built once, shared by both arms)")
    stored = eh.FETCH_MARKET.get(MARKET, MARKET)
    print(f"fetching stored market '{stored}' to derive '{MARKET}'")

    sec = eh._secrets()
    box = {}

    def client_factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(client_factory, seasons, [stored], args.cache,
                          args.refresh)
    if lines.empty:
        print("no lines found.")
        sys.exit(1)

    lines, rep = data_utils.split_qb_rushing(lines, min_match_rate=0.98)
    print(f"split: {rep['reassigned']} of {rep['rows_in_from_market']} "
          f"reassigned ({rep['reassigned_share']:.1%}), match rate "
          f"{rep['match_rate']:.4f}, {rep['unmatched']} unmatched")

    lines = lines[lines["market"] == MARKET].reset_index(drop=True)
    if lines.empty:
        print("nothing labelled qb_rushing after the split.")
        sys.exit(1)
    nullw = int(lines["week"].isna().sum())
    if nullw:
        print(f"dropping {nullw} rows with a null week")
        lines = lines[lines["week"].notna()]

    props = eh.collapse_books(lines)
    props = eh.report_lead(props, None)
    print(f"{len(props)} distinct player-week props")
    return props


def arm_b(seasons_scored, population):
    """Score with the SHIPPED model, walk-forward via train_max_season.

    Mirrors eval_harness.score_market's population handling deliberately, but
    replaces the refit with qb_rushing.load_model. The population is built the
    same way so the two arms are comparable; only the FIT differs, which is
    the entire point.
    """
    feats = list(qb_rushing.LEAN_FEATS)
    actual_col = "rushing_yards"

    builder = qb_rushing.build_dataset
    if population == "all" and hasattr(qb_rushing, "build_all_rows"):
        builder = qb_rushing.build_all_rows
    df = builder()
    need = feats + [actual_col, "season", "week", "player_display_name"]
    have = [c for c in need if c in df.columns]
    df = df[have].dropna(subset=feats + [actual_col]).copy()

    out = []
    for season in seasons_scored:
        test = df[df["season"] == season]
        if len(test) == 0:
            continue
        # The shipped fit, trained strictly before the scored season. This is
        # the call that was impossible before load_model took a parameter.
        model, mfeats = qb_rushing.load_model(train_max_season=season - 1)
        if list(mfeats) != feats:
            print(f"ABORT: load_model returned features {list(mfeats)}, "
                  f"expected {feats}")
            sys.exit(1)
        pred = model.predict(test[feats])
        out.append(pd.DataFrame({
            "season": test["season"].to_numpy(),
            "week": test["week"].to_numpy(),
            "player": test["player_display_name"].to_numpy(),
            "projection": pred,
            "actual": test[actual_col].to_numpy(),
        }))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["market"] = MARKET
    return res


def gate_paired(a, b):
    """Both arms must cover exactly the same player-weeks, or it is not paired."""
    _rule("GATE: the two arms must score identical rows")
    ka = set(zip(a["season"], a["week"], a["_key"]))
    kb = set(zip(b["season"], b["week"], b["_key"]))
    print(f"arm A rows {len(a):,}   distinct keys {len(ka):,}")
    print(f"arm B rows {len(b):,}   distinct keys {len(kb):,}")
    only_a, only_b = ka - kb, kb - ka
    print(f"only in A: {len(only_a):,}    only in B: {len(only_b):,}")
    if only_a or only_b:
        print()
        print("ABORT: the arms do not cover the same rows, so any difference")
        print("in beta could be a population difference rather than a model")
        print("difference. That is the confusion this whole project has been")
        print("burned by. Fix the population handling before comparing.")
        sys.exit(1)
    print("\nGATE PASSED. Same rows, so the only difference is the fit.")


def fit_arm(sub, col):
    return eh.ols_clustered(sub["projection"] - sub[col],
                            sub["actual"] - sub[col], sub["event_id"])


def boot_diff(sub, col, n_boot, seed=0):
    """Cluster bootstrap of (beta_A - beta_B), resampling games."""
    rng = np.random.default_rng(seed)
    events = sub["event_id"].dropna().unique()
    by_event = {e: g for e, g in sub.groupby("event_id")}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(events, size=len(events), replace=True)
        rep = pd.concat([by_event[e] for e in pick], ignore_index=True)
        # give each drawn game a distinct cluster id, or repeated draws of the
        # same game would be treated as one cluster and shrink the interval
        rep["event_id"] = np.repeat(
            np.arange(len(pick)), [len(by_event[e]) for e in pick])
        fa = eh.ols_clustered(rep["projection_a"] - rep[col],
                              rep["actual"] - rep[col], rep["event_id"])
        fb = eh.ols_clustered(rep["projection_b"] - rep[col],
                              rep["actual"] - rep[col], rep["event_id"])
        if fa and fb:
            diffs.append(fa["beta"] - fb["beta"])
    return np.array(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--boot", type=int, default=200,
                    help="cluster bootstrap replicates for the beta difference")
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 80)
    print("qb_rushing: HARNESS REFIT versus SHIPPED MODEL, paired")
    print(f"  seasons {seasons}   population {args.score_population}")
    print("=" * 80)

    props = build_props(args, seasons)

    _rule("ARM A: eval_harness.score_market (the refit)")
    a = eh.score_market(MARKET, seasons, mode="walk_forward",
                        population=args.score_population)
    if not len(a):
        print("arm A scored nothing.")
        sys.exit(1)
    seasons_scored = sorted(a["season"].unique().tolist())
    print(f"{len(a):,} player-weeks, seasons {seasons_scored}")

    _rule("ARM B: qb_rushing.load_model(train_max_season=season-1) (shipped)")
    print("scoring only the seasons arm A produced, so the row sets match")
    b = arm_b(seasons_scored, args.score_population)
    if not len(b):
        print("arm B scored nothing.")
        sys.exit(1)
    print(f"{len(b):,} player-weeks")

    for f in (a, b, props):
        f["_key"] = data_utils.norm_join_name(f["player"])
        f["week"] = f["week"].astype(int)

    gate_paired(a, b)

    # Projection-level agreement, which is the claim the plan makes in
    # RECEPTION units and has never been checked here.
    _rule("PROJECTION AGREEMENT (before any regression)")
    mrg = a.merge(b[["season", "week", "_key", "projection"]],
                  on=["season", "week", "_key"], how="inner",
                  suffixes=("_a", "_b"))
    d = (mrg["projection_a"] - mrg["projection_b"]).abs()
    print(f"paired rows      {len(mrg):,}")
    print(f"mean abs diff    {d.mean():.4f} rushing yards")
    print(f"median abs diff  {d.median():.4f}")
    print(f"max abs diff     {d.max():.4f}")
    print(f"rows over 1.0    {int((d > 1.0).sum()):,}")
    print(f"rows over 5.0    {int((d > 5.0).sum()):,}")
    print(f"correlation      {np.corrcoef(mrg['projection_a'], mrg['projection_b'])[0,1]:.6f}")

    identical = float(d.max()) < 1e-9
    if identical:
        _rule("THE TWO ARMS ARE IDENTICAL BY CONSTRUCTION, AND THAT IS THE ANSWER")
        print("Zero difference on every row is the Phase 3.4 byte-identical")
        print("signature, which is normally a bug. Here it is arithmetic:")
        print()
        print("  arm A trains on build_dataset() rows with season <  S")
        print("  arm B trains on build_dataset() rows with season <= S-1")
        print()
        print("Those are the same set. Same features, same target, same")
        print("LinearRegression with the same defaults, so the same fit.")
        print()
        print("qb_rushing.load_model applies NO training gate. It is a bare")
        print("LinearRegression over the module's own dataset, which is")
        print("exactly what the harness reimplements. So for THIS market the")
        print("harness gap is provably zero rather than merely small.")
        print()
        print("WHAT THAT DOES AND DOES NOT ESTABLISH")
        print()
        print("  It does NOT generalise by itself. The gap exists wherever")
        print("  load_model does something the harness does not replicate:")
        print("    rushing     gates training on carries_roll >= 1.5 while")
        print("                the harness scores build_all_rows ungated")
        print("    receptions  gates on targets_roll >= 3")
        print("    receiving   same gate")
        print()
        print("  So the harness gap is not one number, it is one number per")
        print("  market, and it equals whatever that market's training gate")
        print("  does to the coefficients.")
        print()
        print("  For receptions that is measurable and partly measured: Phase")
        print("  3.4 found that moving the gate from 3 to 1.5 shifts served")
        print("  projections by at most 0.137 receptions. The harness is")
        print("  effectively ungated, so the harness-versus-shipped gap there")
        print("  is of that order. That is an INFERENCE, not a measurement,")
        print("  because 3-to-1.5 is not 3-to-none. Measuring it directly")
        print("  needs this same script pointed at receptions, which requires")
        print("  parameterising receptions.load_model the way qb_rushing's")
        print("  now is.")
        print()
        print("The beta tables below will therefore be identical between arms")
        print("to the last digit. The bootstrap is skipped: resampling to")
        print("estimate the spread of an exact zero would burn time to print")
        print("a zero.")

    # Join both arms to the props, once, so every table below is paired.
    # NOTE: `suffixes` only renames OVERLAPPING columns, so `actual` came
    # through from arm A unsuffixed. Asking for `actual_a` here was a bug.
    joined = props.merge(
        mrg[["season", "week", "market", "_key", "projection_a",
             "projection_b", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print()
    print(f"joined to props: {len(joined):,} rows "
          f"({100.0 * len(joined) / max(len(props), 1):.1f}% of props)")
    if len(joined) < 200:
        print("too few joined rows to fit. Stopping.")
        sys.exit(1)

    for src in ("fanduel", "consensus"):
        col = f"line_{src}"
        if col not in joined.columns:
            continue
        sub = joined.dropna(subset=[col, "projection_a", "projection_b"]).copy()
        if len(sub) < 200:
            print(f"\n{src}: only {len(sub)} rows, skipping")
            continue

        _rule(f"LINE SOURCE: {src}   ({len(sub):,} rows)")
        eh.header()
        for label, pcol in (("A refit", "projection_a"),
                            ("B shipped", "projection_b")):
            s2 = sub.rename(columns={pcol: "projection"})
            eh.report_fit(label, fit_arm(s2, col))

        print()
        print("  per season")
        for season, g in sub.groupby("season"):
            for label, pcol in (("A refit", "projection_a"),
                                ("B shipped", "projection_b")):
                g2 = g.rename(columns={pcol: "projection"})
                eh.report_fit(f"{season} {label}", fit_arm(g2, col))

        fa = fit_arm(sub.rename(columns={"projection_a": "projection"}), col)
        fb = fit_arm(sub.rename(columns={"projection_b": "projection"}), col)
        if not (fa and fb):
            continue
        diff = fa["beta"] - fb["beta"]
        print()
        print(f"  BETA DIFFERENCE (refit minus shipped): {diff:+.4f}")
        print(f"  naive independent SE would be "
              f"{np.hypot(fa['se_beta'], fb['se_beta']):.4f}, which is TOO "
              f"WIDE here")
        print("  because the two arms are fitted on the same rows and are")
        print("  highly correlated. Bootstrapping the difference directly.")

        if args.boot > 0 and not identical:
            d2 = boot_diff(sub, col, args.boot)
            if len(d2):
                lo, hi = np.percentile(d2, [2.5, 97.5])
                print()
                print(f"  cluster bootstrap, {len(d2)} replicates")
                print(f"    point       {diff:+.4f}")
                print(f"    boot median {np.median(d2):+.4f}")
                print(f"    95% CI      [{lo:+.4f}, {hi:+.4f}]")
                print(f"    share > 0   {float((d2 > 0).mean()):.3f}")
                straddles = lo <= 0 <= hi
                print()
                if straddles:
                    print("    INTERPRETATION: the interval includes zero, so")
                    print("    the refit and the shipped model are not")
                    print("    distinguishable in beta on this market. That is")
                    print("    evidence the harness proxy is adequate, on ONE")
                    print("    market with two features.")
                else:
                    print("    INTERPRETATION: the interval EXCLUDES zero. The")
                    print("    harness measures a materially different model")
                    print("    than the app serves, and every published beta")
                    print("    needs that caveat attached in beta units, not")
                    print("    just in projection units.")

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
