"""
MODEL-FREE TEST v3: is the effect a PLATEAU or a SELECTION CLIFF?

THE BUG IN v2, WHICH I INTRODUCED AND WHICH PASSED A GATE BY 0.0001

  v2's gate 3 reported the price-rule ROI as -0.0302 against a published
  -0.0401, a difference of +0.0099 against a tolerance of 0.010. It passed by
  one ten-thousandth.

  It was wrong. I passed `et5.GRID`, which runs 0 to 0.25 and is the EDGE
  grid, as the threshold grid for a DECIMAL ODDS column. Every row satisfies
  `dec_under >= 0.25`, so the selection took everything, and -0.0302 is
  exactly v1's blanket-unders figure. The correctly ranged grid I had defined
  for it sat two lines above, unused.

  A gate that passes because a tolerance was marginally too loose is worse
  than one that fails. Fixed here, and the tolerance tightened to 0.002 so
  the same accident cannot repeat.

WHAT v2 FOUND, AND WHY IT CUTS BOTH WAYS

  C1 CONFIRMED, C3 CONFIRMED, C2 REFUTED. C2 was the one with power.

  FOR the rule. The rows it SKIPS have a win-rate gap of -0.30 pp at t -0.37
  and an ROI of -0.0668: a perfectly calibrated population losing exactly the
  vig. The rows it TAKES have a gap of +8.53 pp at t +2.23 and an ROI of
  +0.1410. The ROI difference is +0.2078 at t +2.41.

  AGAINST the rule. The dose response across all 4,234 rows is NEGATIVE,
  -0.3765 at t -1.70, and the ten edge deciles are flat:

    +0.06  +1.05  +2.69  -0.20  -2.89  -0.88  +1.01  +0.46  -1.32  +0.70

  The top decile spans edge 0.0720 to 0.2301 with an overall gap of +0.70 pp.
  The rule's threshold of 0.090 to 0.100 sits INSIDE that decile and takes
  177 of its 424 rows at +8.53 pp, so the other 247 rows must run about
  -4.9 pp. A thirteen-point swing inside one decile with no gradient leading
  up to it is a CLIFF, not an edge, and a cliff located exactly where a
  fine-grained argmax was pointed is what selection produces.

WHY THE EXISTING PLACEBO DOES NOT SETTLE IT

  The anatomy's placebo reached 0 of 400 draws above +0.1475, which looks
  decisive. But it shuffles profit, which destroys the cliff along with
  everything else. It therefore answers "could an argmax find this in pure
  noise" when the question is "could it find a cliff this stable in noise
  with THIS fold structure". The four folds share three of four seasons, so
  the held-out seasons are nowhere near independent.

WHAT v3 TESTS

  D1. THRESHOLD NEIGHBOURHOOD, no selection anywhere. The gap at every FIXED
      cutoff from 0.05 to 0.15 in 0.01 steps, with game-clustered SEs. A real
      edge gives a PLATEAU: roughly stable or rising across the whole
      neighbourhood. Selection gives a SPIKE at 0.09 to 0.10 and little
      either side.

  D2. NON-CUMULATIVE narrow bins above 0.05, so a step is distinguishable
      from a wander. D1 is cumulative and therefore autocorrelated by
      construction; this is the same data without that.

  D3. PER SEASON AT A FIXED 0.09, no argmax at all. Four numbers with
      clustered SEs. The holdout's +2.73, +10.04, +18.09 were each produced
      under a selected threshold; these are not.

  D4. GAME-CLUSTERED BOOTSTRAP ON THE FIXED-0.09 RULE. This removes the
      selection step entirely, which is the candidate rule's one identified
      weakness. If the fixed-threshold gap holds up with a CI excluding zero,
      the rule stands independently of the argmax. If it does not, the argmax
      was the whole result.

  D5. A MULTIPLICITY ACCOUNTING for the session. By this point we have
      examined many cuts of this same population. A raw p of 0.026 is not
      worth what it looks like, and the script prints the Bonferroni
      threshold against an explicit count.

PRE-REGISTERED PREDICTIONS

  D1a. The gap does NOT form a plateau. It will peak near 0.09 to 0.10 and
       fall away on both sides. Reasoning: v2's deciles are flat, so there is
       no gradient for a plateau to sit on.
  D2a. The narrow bins wander rather than step, with no bin below 0.09
       clearly positive.
  D3a. The fixed-0.09 per-season gaps are LESS monotone than the selected
       ones, because part of that +2.73 / +10.04 / +18.09 trend is the
       threshold moving between seasons.
  D4a. The fixed-0.09 bootstrap CI includes zero, or excludes it only
       marginally, and does not survive the multiplicity threshold in D5.

  If D1a is REFUTED and a plateau appears, that is strong evidence the effect
  is real and my cliff reading is wrong. That is the outcome I would most
  like to be wrong about, and it is why D1 is a fixed grid with no selection.

Run from the repo root with the venv active:
    python model_free_test_v3.py --all-seasons --cache lines_cache.parquet
    python model_free_test_v3.py --all-seasons --cache lines_cache.parquet --boot 4000
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import edge_threshold_v5 as et5

PUB_RULE = {"roi": 0.1475, "bets": 176, "win_rate": 0.5341}
PUB_PRICE = {"fair": 0.4458, "gap_pp": 8.83}
PUB_V1_PRICE_RULE = -0.0401
PUB_V2_TAKES = {"gap_pp": 8.53, "n": 177}
PUB_V2_SKIPS = {"gap_pp": -0.30, "n": 4057}
TOL = 0.010
TOL_PP = 0.30
TOL_GATE3 = 0.002

# The decimal-odds grid for the price rule. THIS is what v2 failed to pass.
DEC_GRID = np.round(np.arange(1.50, 2.81, 0.05), 2)

FIXED_T = 0.090

# Cuts examined in this script, for the multiplicity accounting in D5.
# Deliberately generous: undercounting the look-elsewhere effect is the
# failure mode this exists to prevent.
CUTS_THIS_SCRIPT = 11 + 7 + 4      # D1 grid, D2 bins, D3 seasons
CUTS_EARLIER_SESSION = 4 + 4 + 7 + 8 + 14 + 6 + 4 + 9 + 10


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def mde(se):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


def gap_stats(g):
    """Win-rate gap against the devigged fair price, game-clustered."""
    if len(g) < 20:
        return None
    cw = eh.cluster_mean(g["won_f"].to_numpy(), g["event_id"].to_numpy())
    if cw is None:
        return None
    fair = float(g["fair_chosen"].mean())
    gap = cw["mean"] - fair
    return {"n": len(g), "wr": cw["mean"], "fair": fair, "gap": gap,
            "se": cw["se"], "t": gap / cw["se"] if cw["se"] > 0 else np.nan,
            "roi": float(g["profit"].mean())}


def build_priced(args):
    a = types.SimpleNamespace(
        markets="receptions", all_seasons=args.all_seasons,
        seasons=args.seasons, cache=args.cache, refresh=args.refresh,
        lead_window=None, score_population="all")
    joined = et5.build_joined(a)
    joined = et5.apply_line_values(joined, [2.5, 3.5], None, None)
    params = et5.fit_params(joined)
    slevel = et5.fit_sigma_sqrt(joined, params)
    s = et5.price(joined, params, "fanduel", slevel, "level", 1.0,
                  no_model=False)
    s = s.copy()
    s["dec_under"] = 1.0 + s["pay_under"]
    s["won_under"] = (s["actual"] < s["bet_line"]).astype(float)
    s["prof_under"] = np.where(s["won_under"] > 0.5, s["pay_under"], -1.0)
    be_ch = np.where(s["side_over"], s["be_over"], s["be_under"])
    be_ot = np.where(s["side_over"], s["be_under"], s["be_over"])
    s["fair_chosen"] = be_ch / (be_ch + be_ot)
    s["won_f"] = s["won"].astype(float)
    s = s[~s["push"]].copy()
    print(f"\n  priced {len(s)} rows, pushes dropped")
    return s


def select_thresholds(s, profit_col, edge_col, grid, min_train=100):
    out = {}
    for s_out in sorted(s["season"].unique()):
        tr = s[s["season"] != s_out]
        bt, bv = None, -np.inf
        for t in grid:
            g = tr[tr[edge_col] >= t]
            if len(g) < min_train:
                continue
            v = float(g[profit_col].mean())
            if v > bv:
                bt, bv = t, v
        if bt is not None:
            out[s_out] = bt
    return out


def gate(s):
    print("\n" + "=" * 96)
    print("GATES 1 to 4")
    print("=" * 96)
    th = select_thresholds(s, "profit", "edge", et5.GRID)
    rows = []
    for s_out, t in th.items():
        g = s[(s["season"] == s_out) & (s["edge"] >= t)].copy()
        if len(g) >= 20:
            rows.append(g)
    bets = pd.concat(rows, ignore_index=True)
    cm = eh.cluster_mean(bets["profit"].to_numpy(),
                         bets["event_id"].to_numpy())
    wr = float(bets["won"].mean())
    fair = float(bets["fair_chosen"].mean())
    gap = 100.0 * (wr - fair)

    # gate 3, with the CORRECT decimal grid this time
    pth = select_thresholds(s, "prof_under", "dec_under", DEC_GRID)
    prows = [s[(s["season"] == k) & (s["dec_under"] >= v)]
             for k, v in pth.items()]
    prows = [p for p in prows if len(p) >= 20]
    price_roi = (pd.concat(prows, ignore_index=True)["prof_under"].mean()
                 if prows else np.nan)

    # gate 4, v2's taken and skipped gaps
    trows = []
    for s_out, t in th.items():
        g = s[s["season"] == s_out].copy()
        g["above"] = (g["edge"] >= t).astype(float)
        trows.append(g)
    d = pd.concat(trows, ignore_index=True)
    gt = gap_stats(d[d["above"] > 0.5])
    gs = gap_stats(d[d["above"] < 0.5])

    checks = [
        ("ROI", cm["mean"], PUB_RULE["roi"], TOL),
        ("bets", float(len(bets)), float(PUB_RULE["bets"]), 0.5),
        ("win rate", wr, PUB_RULE["win_rate"], TOL),
        ("fair probability", fair, PUB_PRICE["fair"], TOL),
        ("win-rate gap pp", gap, PUB_PRICE["gap_pp"], TOL_PP),
        ("price rule ROI", price_roi, PUB_V1_PRICE_RULE, TOL_GATE3),
        ("v2 TAKES gap pp", 100 * gt["gap"], PUB_V2_TAKES["gap_pp"], TOL_PP),
        ("v2 SKIPS gap pp", 100 * gs["gap"], PUB_V2_SKIPS["gap_pp"], TOL_PP),
    ]
    fails = []
    print(f"  {'quantity':<22}{'got':>11}{'published':>12}{'diff':>10}"
          f"{'tol':>8}   verdict")
    for name, got, pub, tol in checks:
        dd = got - pub
        ok = np.isfinite(dd) and abs(dd) <= tol
        print(f"  {name:<22}{got:>11.4f}{pub:>12.4f}{dd:>+10.4f}"
              f"{tol:>8.4f}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    if fails:
        print(f"\n  GATES FAILED for {fails}. Stop. Do not widen tolerance.")
        sys.exit(2)
    print("  GATES PASSED, gate 3 now on the correct decimal grid at a")
    print(f"  tolerance of {TOL_GATE3:.3f} rather than {TOL:.3f}.")
    print(f"\n  price-rule thresholds chosen: "
          f"{ {int(k): float(v) for k, v in pth.items()} }")
    return d, th


def d1_neighbourhood(s):
    """Gap at every FIXED cutoff. Plateau means real, spike means selected."""
    print("\n" + "=" * 96)
    print("D1. THRESHOLD NEIGHBOURHOOD, fixed cutoffs, NO selection")
    print("=" * 96)
    grid = np.round(np.arange(0.05, 0.151, 0.01), 3)
    thresh = 0.05 / (CUTS_THIS_SCRIPT)
    print(f"  {len(grid)} cutoffs. Bonferroni for this table alone "
          f"{0.05 / len(grid):.4f}.")
    print(f"  A PLATEAU across the neighbourhood means a real edge.")
    print(f"  A SPIKE at 0.09 to 0.10 means the argmax found a cliff.\n")
    print(f"  {'cutoff':>8}{'rows':>7}{'winrt':>9}{'fair':>9}{'gap pp':>9}"
          f"{'SE pp':>8}{'t':>7}{'ROI':>9}")
    for t in grid:
        r = gap_stats(s[s["edge"] >= t])
        if r is None:
            print(f"  {t:>8.3f}   too few rows")
            continue
        print(f"  {t:>8.3f}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    print("\n  These rows are CUMULATIVE and therefore heavily")
    print("  autocorrelated. D2 shows the same data without that.")


def d2_bins(s):
    """Non-cumulative narrow bins, so a step is visible as a step."""
    print("\n" + "=" * 96)
    print("D2. NON-CUMULATIVE BINS above 0.05")
    print("=" * 96)
    edges = [0.05, 0.065, 0.08, 0.09, 0.10, 0.115, 0.14, 1.0]
    print(f"  {len(edges) - 1} bins. Bonferroni for this table alone "
          f"{0.05 / (len(edges) - 1):.4f}.\n")
    print(f"  {'bin':<18}{'rows':>7}{'winrt':>9}{'fair':>9}{'gap pp':>9}"
          f"{'SE pp':>8}{'t':>7}{'ROI':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = s[(s["edge"] >= lo) & (s["edge"] < hi)]
        r = gap_stats(g)
        lab = f"{lo:.3f} to {hi:.3f}" if hi < 1 else f"{lo:.3f} and up"
        if r is None:
            print(f"  {lab:<18}{len(g):>7}   too few rows")
            continue
        print(f"  {lab:<18}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    print("\n  If the bins below 0.09 are flat and only the bins at and")
    print("  above 0.09 are positive, that is the cliff. If the rise is")
    print("  gradual across bins, that is a real dose response that the")
    print("  linear slope in v2 was too crude to see.")


def d3_seasons(s):
    """Per season at a FIXED threshold. No argmax anywhere."""
    print("\n" + "=" * 96)
    print(f"D3. PER SEASON AT A FIXED {FIXED_T:.3f}, no selection")
    print("=" * 96)
    print("  The holdout's +2.73 / +10.04 / +18.09 were each produced under")
    print("  a SELECTED threshold that differed by season. These are not.\n")
    print(f"  {'season':>8}{'rows':>7}{'winrt':>9}{'fair':>9}{'gap pp':>9}"
          f"{'SE pp':>8}{'t':>7}{'ROI':>9}")
    sub = s[s["edge"] >= FIXED_T]
    for s_out, g in sub.groupby("season"):
        r = gap_stats(g)
        if r is None:
            print(f"  {int(s_out):>8}{len(g):>7}   too few rows")
            continue
        print(f"  {int(s_out):>8}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    r = gap_stats(sub)
    if r:
        print(f"  {'POOLED':>8}{r['n']:>7}{r['wr']:>9.4f}{r['fair']:>9.4f}"
              f"{100 * r['gap']:>+9.2f}{100 * r['se']:>8.2f}{r['t']:>7.2f}"
              f"{r['roi']:>+9.4f}")
    return sub


def d4_bootstrap(sub, n_boot, seed=0):
    """Game-clustered bootstrap on the FIXED-threshold rule.

    This is the version with the selection step removed entirely, which is
    the candidate rule's one identified weakness. If the gap survives here
    the rule does not depend on the argmax.
    """
    print("\n" + "=" * 96)
    print(f"D4. GAME-CLUSTERED BOOTSTRAP at a fixed {FIXED_T:.3f}, "
          f"{n_boot} draws")
    print("=" * 96)
    if len(sub) < 50:
        print("  too few rows")
        return
    rng = np.random.default_rng(seed)
    ev = sub["event_id"].to_numpy()
    won = sub["won_f"].to_numpy()
    fair = sub["fair_chosen"].to_numpy()
    prof = sub["profit"].to_numpy()
    groups = {}
    for i, e in enumerate(ev):
        groups.setdefault(e, []).append(i)
    keys = list(groups)
    idx = [np.asarray(groups[k]) for k in keys]
    gaps, rois = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx[i] for i in pick])
        gaps.append(won[sel].mean() - fair[sel].mean())
        rois.append(prof[sel].mean())
    gaps = np.asarray(gaps)
    rois = np.asarray(rois)
    print(f"  {len(keys)} games hold {len(sub)} bets\n")
    print(f"  win-rate gap   mean {100 * gaps.mean():+.2f} pp   "
          f"sd {100 * gaps.std():.2f}")
    print(f"                 95% CI [{100 * np.percentile(gaps, 2.5):+.2f}, "
          f"{100 * np.percentile(gaps, 97.5):+.2f}] pp   "
          f"{100 * (gaps > 0).mean():.1f}% positive")
    print(f"  ROI            mean {rois.mean():+.4f}   sd {rois.std():.4f}")
    print(f"                 95% CI [{np.percentile(rois, 2.5):+.4f}, "
          f"{np.percentile(rois, 97.5):+.4f}]   "
          f"{100 * (rois > 0).mean():.1f}% positive")
    print("\n  A CI on the GAP that excludes zero here is the strongest")
    print("  statement available for this rule, because no threshold was")
    print("  selected. Read it against D5.")


def d5_multiplicity():
    print("\n" + "=" * 96)
    print("D5. MULTIPLICITY ACCOUNTING for the whole session")
    print("=" * 96)
    tot = CUTS_THIS_SCRIPT + CUTS_EARLIER_SESSION
    print(f"  cuts of this same population examined in this script   "
          f"{CUTS_THIS_SCRIPT}")
    print(f"  cuts examined earlier in the session (v1 to v4, anatomy) "
          f"{CUTS_EARLIER_SESSION}")
    print(f"  total                                                   {tot}")
    print(f"\n  Bonferroni threshold at 0.05 over {tot} cuts: "
          f"{0.05 / tot:.5f}")
    print(f"  a raw p of 0.026 corresponds to t about 2.23")
    print(f"  the t required at that threshold is about "
          f"{4.06 if tot > 100 else 3.5:.2f}")
    print("\n  This is deliberately generous in counting. The point is not")
    print("  the exact number: it is that a t of 2.2 found after this much")
    print("  searching of one population is weak evidence, and the rule's")
    print("  own out-of-sample 2026 firing is worth more than any further")
    print("  cut of the same four seasons.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("MODEL-FREE TEST v3: plateau or selection cliff?")
    print("=" * 96)

    s = build_priced(args)
    d, th = gate(s)
    d1_neighbourhood(s)
    d2_bins(s)
    sub = d3_seasons(s)
    d4_bootstrap(sub, args.boot)
    d5_multiplicity()

    if args.save_rows:
        cols = ["season", "week", "player", "bet_line", "projection",
                "blend", "sigma", "p_over", "edge", "side_over",
                "fair_chosen", "actual", "won_f", "profit", "event_id"]
        cols = [c for c in cols if c in s.columns]
        s[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"\n  wrote {len(s)} rows to {args.save_rows}")

    print("\n" + "=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  D2 first, because it is the same data as D1 without the")
    print("  cumulative autocorrelation. A gradual rise across bins means")
    print("  a real dose response that v2's linear slope was too crude to")
    print("  detect. A flat run below 0.09 with everything above it")
    print("  positive means a cliff at the selected point.")
    print("  Then D4, which is the rule with the selection step removed.")
    print("  Then D5, which says what a t of 2.2 is worth after this much")
    print("  searching. The answer is: less than one clean 2026 season.")
    print("=" * 96)


if __name__ == "__main__":
    main()
