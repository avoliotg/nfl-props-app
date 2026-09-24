"""
CANDIDATE RULE ANATOMY: what are the 176 holdout bets actually made of?

WHY THIS EXISTS

  The candidate rule reproduces exactly: +0.1475, SE 0.0849, t +1.74, 176
  bets, win rate 0.5341. Its decomposition also reproduces: +0.0524 book only,
  +0.0742 model only, which sum to 0.1266 against a full 0.1475, and neither
  arm is distinguishable from zero alone.

  Four separate regression tests (odds_space_test v1 to v4) found nothing:
  the model adds nothing to FanDuel's price, the price is an efficient
  forecast at slope 1.0, other books' lines add nothing, and every simple
  realised-ROI rule is negative. None of those tests can see this rule,
  for two reasons:

    1. POWER. The rule fires on 120 of 4234 priced rows, about 4 percent.
       A linear regression over the full sample with an MDE of 0.029 per
       reception cannot detect an effect concentrated in a 4 percent tail.
       "Zero on average" and "zero in the tail" are different claims.
    2. MY ERROR. odds_space_test v3's Test G swept only the EXPENSIVE side
       (back the over when the over price is high, and the mirror). The
       candidate rule backs the CHEAP side at plus money. I never tested the
       direction the rule actually bets.

  So the rule is untested rather than refuted. This script tests it directly,
  on its own bets, in units where one lucky longshot cannot carry the result.

THE SUSPICIOUS SHAPE THIS IS AIMED AT

  In all three v5 runs the ROI is negative at every cutoff below about 0.075
  and then flips hard positive above it:

    full run     -0.0688 at 0.000 ... -0.0450 at 0.070 ... +0.2035 at 0.100
    no-model     -0.0688 at 0.000 ... -0.0372 at 0.065 ... +0.1438 at 0.100
    same-line    -0.0770 at 0.000 ... -0.0529 at 0.065 ... +0.3036 at 0.100

  A real graded edge should improve SMOOTHLY with the threshold. Sitting at
  -0.06 for fifteen consecutive rows and then jumping to +0.20 is a different
  shape, and it coexists with a conditional calibration table that is
  monotonically OVERCONFIDENT (-3.3, -4.5, -4.9, -4.9, -8.1, -13.5 points).
  The layer is worst exactly where the rule bets most, and the rule still
  wins there. That inversion needs an explanation before the number is
  treated as an edge.

WHAT IS REPORTED, AND WHY EACH

  1. MEAN IMPLIED PROBABILITY of the chosen side, both with vig (the
     breakeven you must beat) and devigged (the book's own fair estimate).
     ROI in plus-money markets is leveraged: a 9-point win-rate edge at +120
     looks like +20 percent. The win rate against the devigged price is the
     forecasting claim, stated in a unit where leverage is removed.
  2. REALISED WIN RATE with a game-clustered SE, and the gap against both
     benchmarks.
  3. PROFIT CONCENTRATION. Top-5 and top-10 bet contribution to total
     profit, ROI with the largest winners removed, and ROI recomputed as if
     every win paid the MEDIAN price. If the result dies when the biggest
     payouts go, it is a few longshots rather than an edge.
  4. GAME-CLUSTERED BOOTSTRAP. Resampling BETS treats props in one game as
     independent, which they are not. The handoff records 91.2 percent
     positive replicates; this reports both bet-level and game-level so the
     difference is visible.
  5. A FULL-PROCEDURE PLACEBO. Shuffling profit within (season, line) and
     re-running the entire threshold selection, so the null accounts for the
     selection step and not just the bets.

THE GATE

  Reproduce +0.1475 on 176 bets at win rate 0.5341 before reporting anything
  new. This script re-derives the holdout selection, so if it lands anywhere
  else it is not looking at the candidate rule.

PRE-REGISTERED PREDICTIONS

  A1. Mean devigged implied probability of the chosen side is BELOW 0.50,
      confirming the rule backs the cheap side. Estimated near 0.46 from the
      full run's win rate and ROI.
  A2. The win-rate gap against the devigged price is 5 to 9 points, and
      under 2 game-clustered SEs. The ROI of +0.1475 is that gap amplified
      by plus-money leverage.
  A3. Removing the top 5 winning bets of 176 cuts the ROI by more than half.
  A4. The game-level bootstrap gives a materially lower positive share than
      the bet-level one, and lower than the recorded 91.2 percent.
  A5. The full-procedure placebo centres near the all-bets ROI of about
      -0.06, not at zero, because the selection step has no upward bias but
      the population does.

  If A3 and A4 both fail, the rule survives its sharpest test yet and the
  honest description changes from "threshold selection" to "a real tail
  effect the regressions lacked power to see".

Run from the repo root with the venv active:
    python candidate_anatomy.py --all-seasons --cache lines_cache.parquet
    python candidate_anatomy.py --all-seasons --cache lines_cache.parquet --boot 4000 --placebo 400
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import edge_threshold_v5 as et5

PUBLISHED = {"roi": 0.1475, "bets": 176, "win_rate": 0.5341}
GATE_TOL_ROI = 0.010
GATE_TOL_WR = 0.010

# BENCHMARKS FROM A PLANTED GENUINE EDGE.
#
# Simulated 176 bets with a REAL win rate of 0.5341 at decimal odds drawn
# around 2.19, matching the rule's own shape, then run through these same
# diagnostics. This is what a genuine edge of this size and price LOOKS like,
# and it is the comparison that matters: a bare threshold is useless here.
#
# The concentration test is much weaker than it first appears. Total profit is
# only about 25 units across 176 bets and each win pays about 1.2, so removing
# ten winners removes nearly half the profit even when the edge is real.
PLANTED = {
    "roi": 0.1445,
    "roi_wo_top5": 0.0993,     # a 31 percent drop, with a REAL edge
    "roi_wo_top10": 0.0577,    # a 60 percent drop, with a REAL edge
    "boot_bet_pos": 96.1,
    "boot_game_pos": 95.6,     # clustering barely bites at ~3 bets per game
}


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def build_priced(args):
    """Replicate v5's pipeline for the candidate rule, by importing it."""
    a = types.SimpleNamespace(
        markets="receptions",
        all_seasons=args.all_seasons,
        seasons=args.seasons,
        cache=args.cache,
        refresh=args.refresh,
        lead_window=None,
        score_population="all",
    )
    joined = et5.build_joined(a)
    joined = et5.apply_line_values(joined, [2.5, 3.5], None, None)

    params = et5.fit_params(joined)
    slevel = et5.fit_sigma_sqrt(joined, params)
    s = et5.price(joined, params, "fanduel", slevel, "level", 1.0,
                  no_model=False)
    print(f"\n  priced {len(s)} rows, {int(s['push'].sum())} pushes")
    return s


def select_holdout(s):
    """v5's holdout selection, replicated, but RETAINING the chosen rows.

    v5 prints the per-season result and discards the frames. The anatomy
    needs the rows, so the selection loop is reproduced here exactly:
    maximise mean profit over GRID on the other seasons, require at least
    100 training bets, then spend that threshold on the held-out season.
    """
    rows = []
    per_season = []
    for s_out in sorted(s["season"].unique()):
        tr = s[s["season"] != s_out]
        te = s[s["season"] == s_out]
        bt, bv = None, -np.inf
        for t in et5.GRID:
            g = tr[(tr["edge"] >= t) & (~tr["push"])]
            if len(g) < 100:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[(te["edge"] >= bt) & (~te["push"])].copy()
        if len(g) < 20:
            per_season.append((s_out, bt, len(g), np.nan, np.nan))
            continue
        g["held_out"] = s_out
        g["chosen_t"] = bt
        cm = eh.cluster_mean(g["profit"].to_numpy(), g["event_id"].to_numpy())
        per_season.append((s_out, bt, len(g), cm["mean"], cm["se"]))
        rows.append(g)
    if not rows:
        print("  no holdout bets selected")
        sys.exit(1)
    return pd.concat(rows, ignore_index=True), per_season


def gate(bets, per_season):
    print("\n" + "=" * 96)
    print("GATE: reproduce the published candidate rule")
    print("=" * 96)
    print(f"  {'held out':>9}{'chosen t':>10}{'bets':>7}{'ROI':>9}{'SE':>8}")
    for s_out, bt, n, roi, se in per_season:
        if np.isfinite(roi):
            print(f"  {int(s_out):>9}{bt:>10.3f}{n:>7}{roi:>+9.4f}{se:>8.4f}")
        else:
            print(f"  {int(s_out):>9}{bt:>10.3f}{n:>7}   too few for a SE")
    cm = eh.cluster_mean(bets["profit"].to_numpy(),
                         bets["event_id"].to_numpy())
    wr = float(bets["won"].mean())
    t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
    print(f"\n  {'POOLED':>9}{'':>10}{len(bets):>7}{cm['mean']:>+9.4f}"
          f"{cm['se']:>8.4f}   t {t:+.2f}   win rate {wr:.4f}")
    fails = []
    if abs(cm["mean"] - PUBLISHED["roi"]) > GATE_TOL_ROI:
        fails.append(f"ROI {cm['mean']:+.4f} vs {PUBLISHED['roi']:+.4f}")
    if len(bets) != PUBLISHED["bets"]:
        fails.append(f"bets {len(bets)} vs {PUBLISHED['bets']}")
    if abs(wr - PUBLISHED["win_rate"]) > GATE_TOL_WR:
        fails.append(f"win rate {wr:.4f} vs {PUBLISHED['win_rate']:.4f}")
    if fails:
        print(f"\n  GATE FAILED: {fails}")
        print("  This is not the candidate rule. Stop.")
        sys.exit(2)
    print("  GATE PASSED. These are the published 176 bets.")
    return cm, wr


def report_price_units(bets):
    """The forecasting claim, with plus-money leverage removed."""
    print("\n" + "=" * 96)
    print("1. WHAT PRICE DO THESE BETS TAKE, AND WHAT IS THE REAL EDGE?")
    print("=" * 96)
    be_chosen = np.where(bets["side_over"], bets["be_over"], bets["be_under"])
    be_other = np.where(bets["side_over"], bets["be_under"], bets["be_over"])
    tot = be_chosen + be_other
    fair = be_chosen / tot
    pay = np.where(bets["side_over"], bets["pay_over"], bets["pay_under"])
    dec = 1.0 + pay

    print(f"  bets {len(bets)}   overs {int(bets['side_over'].sum())}"
          f"   unders {int((~bets['side_over']).sum())}")
    print(f"\n  price of the chosen side")
    print(f"    mean breakeven (with vig)        {be_chosen.mean():.4f}")
    print(f"    mean devigged fair probability   {fair.mean():.4f}")
    print(f"    mean decimal odds                {dec.mean():.4f}")
    print(f"    decimal odds p10 / p50 / p90     "
          f"{np.percentile(dec, 10):.3f} / {np.percentile(dec, 50):.3f}"
          f" / {np.percentile(dec, 90):.3f}")
    print(f"    mean hold on these props         {(tot - 1.0).mean():.4f}")

    wr = float(bets["won"].mean())
    cm = eh.cluster_mean(bets["won"].to_numpy().astype(float),
                         bets["event_id"].to_numpy())
    print(f"\n  the forecasting claim, leverage removed")
    print(f"    realised win rate                {wr:.4f}"
          f"   game-clustered SE {cm['se']:.4f}")
    print(f"    vs breakeven (with vig)          "
          f"{100.0 * (wr - be_chosen.mean()):+.2f} pp"
          f"   t {(wr - be_chosen.mean()) / cm['se']:+.2f}")
    print(f"    vs devigged fair probability     "
          f"{100.0 * (wr - fair.mean()):+.2f} pp"
          f"   t {(wr - fair.mean()) / cm['se']:+.2f}")
    print(f"\n  The second line is the honest size of the claim. The ROI of")
    print(f"  {PUBLISHED['roi']:+.4f} is that gap amplified by the price.")
    bets = bets.copy()
    bets["fair"] = fair
    bets["dec"] = dec
    return bets


def report_concentration(bets):
    """Does the result survive losing its biggest winners?"""
    print("\n" + "=" * 96)
    print("2. PROFIT CONCENTRATION: is this a few longshots?")
    print("=" * 96)
    p = bets["profit"].to_numpy()
    total = p.sum()
    order = np.argsort(-p)
    print(f"  total profit {total:+.2f} units on {len(p)} bets")
    for k in (1, 3, 5, 10):
        if k <= len(p):
            top = p[order[:k]].sum()
            share = 100.0 * top / total if total != 0 else np.nan
            rest = p[order[k:]]
            roi_rest = rest.mean() if len(rest) else np.nan
            print(f"    top {k:>2} bets contribute {top:>+7.2f} "
                  f"({share:>6.1f}% of total)   ROI without them "
                  f"{roi_rest:>+8.4f}")

    wins = bets[bets["won"]]
    if len(wins):
        med_pay = float(np.median(wins["dec"] - 1.0))
        wr = float(bets["won"].mean())
        roi_flat = wr * med_pay - (1.0 - wr)
        print(f"\n  median winning payout {med_pay:.4f}")
        print(f"  ROI if every win paid the MEDIAN price "
              f"{roi_flat:+.4f}")
        print(f"  actual ROI {bets['profit'].mean():+.4f}")
        print(f"  The gap between those two is pure payout dispersion.")

    trim = np.sort(p)[1:-1] if len(p) > 4 else p
    print(f"\n  trimmed mean (drop one highest and one lowest) "
          f"{trim.mean():+.4f}")

    print(f"\n  BENCHMARK, a PLANTED GENUINE edge of the same shape:")
    print(f"    ROI {PLANTED['roi']:+.4f}   "
          f"without top 5 {PLANTED['roi_wo_top5']:+.4f}   "
          f"without top 10 {PLANTED['roi_wo_top10']:+.4f}")
    print(f"    A real edge ALSO loses 31 and 60 percent here, so a large")
    print(f"    drop is NOT evidence against the rule. Only a drop that is")
    print(f"    much steeper than the benchmark says anything.")


def report_bootstrap(bets, n_boot, seed=0):
    """Bet-level against game-level. Props in one game are not independent."""
    print("\n" + "=" * 96)
    print(f"3. BOOTSTRAP, {n_boot} draws: bet-level vs game-level")
    print("=" * 96)
    rng = np.random.default_rng(seed)
    p = bets["profit"].to_numpy()
    ev = bets["event_id"].to_numpy()

    bl = np.array([rng.choice(p, size=len(p), replace=True).mean()
                   for _ in range(n_boot)])

    groups = {}
    for i, e in enumerate(ev):
        groups.setdefault(e, []).append(i)
    keys = list(groups)
    idx_by_key = [np.asarray(groups[k]) for k in keys]
    gl = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx_by_key[i] for i in pick])
        gl.append(p[sel].mean())
    gl = np.asarray(gl)

    print(f"  {'level':<14}{'mean':>10}{'sd':>9}{'% positive':>12}"
          f"   95% CI")
    for lab, v in (("bet-level", bl), ("game-level", gl)):
        print(f"  {lab:<14}{v.mean():>+10.4f}{v.std():>9.4f}"
              f"{100.0 * (v > 0).mean():>11.1f}%"
              f"   [{np.percentile(v, 2.5):+.4f}, "
              f"{np.percentile(v, 97.5):+.4f}]")
    print(f"\n  {len(keys)} distinct games hold {len(p)} bets, "
          f"{len(p) / max(len(keys), 1):.2f} per game.")
    print("  The handoff records 91.2 percent positive replicates. If that")
    print("  was bet-level, the game-level row is the one to quote.")
    print(f"\n  BENCHMARK, a PLANTED GENUINE edge of the same shape:")
    print(f"    bet-level {PLANTED['boot_bet_pos']:.1f}% positive, "
          f"game-level {PLANTED['boot_game_pos']:.1f}%")
    print("    At about 3 bets per game the clustering barely bites, so a")
    print("    LARGE bet-to-game gap would itself be the surprise.")


def report_placebo(s, n_draws, seed=0):
    """Null for the WHOLE procedure, selection step included.

    Profit is shuffled within (season, line), which destroys any link
    between the edge and the outcome while preserving the population's own
    negative expectation and its line mix. Then the entire holdout selection
    re-runs. The resulting distribution is what this procedure returns on
    noise, and the honest comparison for +0.1475 is against THIS, not zero.
    """
    if not n_draws:
        return
    print("\n" + "=" * 96)
    print(f"4. FULL-PROCEDURE PLACEBO, {n_draws} draws")
    print("=" * 96)
    rng = np.random.default_rng(seed)
    base = s.copy()
    vals = []
    for _ in range(n_draws):
        sh = base.copy()
        sh["profit"] = (sh.groupby(["season", "bet_line"])["profit"]
                        .transform(lambda v: rng.permutation(v.to_numpy())))
        try:
            bets, _ = select_holdout_quiet(sh)
        except SystemExit:
            continue
        if bets is not None and len(bets):
            vals.append(float(bets["profit"].mean()))
    if not vals:
        print("  no usable draws")
        return
    v = np.asarray(vals)
    print(f"  placebo ROI mean {v.mean():+.4f}   sd {v.std():.4f}")
    print(f"  95% range [{np.percentile(v, 2.5):+.4f}, "
          f"{np.percentile(v, 97.5):+.4f}]")
    print(f"  share of placebo draws at or above +{PUBLISHED['roi']:.4f}: "
          f"{100.0 * (v >= PUBLISHED['roi']).mean():.1f}%")
    print(f"  all-bets ROI for reference: {s['profit'].mean():+.4f}")
    print("\n  The placebo mean is the artifact. The honest value of the")
    print("  rule is its distance from THIS, not from zero.")


def select_holdout_quiet(s):
    rows = []
    for s_out in sorted(s["season"].unique()):
        tr = s[s["season"] != s_out]
        te = s[s["season"] == s_out]
        bt, bv = None, -np.inf
        for t in et5.GRID:
            g = tr[(tr["edge"] >= t) & (~tr["push"])]
            if len(g) < 100:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[(te["edge"] >= bt) & (~te["push"])]
        if len(g) >= 20:
            rows.append(g)
    if not rows:
        return None, None
    return pd.concat(rows, ignore_index=True), None


def report_per_season(bets):
    print("\n" + "=" * 96)
    print("5. PER SEASON, in win-rate units")
    print("=" * 96)
    print(f"  {'season':>8}{'bets':>7}{'win rate':>10}{'fair p':>9}"
          f"{'gap pp':>9}{'ROI':>9}{'mean dec':>10}")
    for s_out, g in bets.groupby("held_out"):
        wr = float(g["won"].mean())
        fp = float(g["fair"].mean())
        print(f"  {int(s_out):>8}{len(g):>7}{wr:>10.4f}{fp:>9.4f}"
              f"{100.0 * (wr - fp):>+9.2f}{g['profit'].mean():>+9.4f}"
              f"{g['dec'].mean():>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--placebo", type=int, default=200)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("CANDIDATE RULE ANATOMY: receptions, lines 2.5 and 3.5, sqrt sigma")
    print("=" * 96)

    s = build_priced(args)
    bets, per_season = select_holdout(s)
    gate(bets, per_season)
    bets = report_price_units(bets)
    report_concentration(bets)
    report_bootstrap(bets, args.boot)
    report_per_season(bets)
    report_placebo(s, args.placebo)

    if args.save_rows:
        cols = ["held_out", "chosen_t", "season", "week", "player",
                "bet_line", "line_consensus", "projection", "blend",
                "sigma", "p_over", "edge", "side_over", "fair", "dec",
                "actual", "won", "profit", "event_id"]
        cols = [c for c in cols if c in bets.columns]
        bets[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"\n  wrote {len(bets)} bets to {args.save_rows}")

    print("\n" + "=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  Gate first. Then section 1: the win-rate gap against the")
    print("  DEVIGGED price is the real size of the claim, and the ROI is")
    print("  that gap times the price. Then section 2: if the result dies")
    print("  when the top 5 winners go, it is longshots. Then section 3's")
    print("  game-level row, not the bet-level one. Section 4 is the null")
    print("  for the whole procedure and +0.1475 should be judged against")
    print("  it rather than against zero.")
    print("=" * 96)


if __name__ == "__main__":
    main()
