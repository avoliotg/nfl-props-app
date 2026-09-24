"""
MODEL-FREE TEST: is the candidate rule just "back unders at plus money"?

WHAT THE ANATOMY ESTABLISHED (run 2026-09-24, gate passed)

  The candidate rule survives its sharpest tests. In win-rate units the claim
  is +8.83 points (win rate 0.5341 against a devigged fair probability of
  0.4458) at t +2.30 game-clustered, which is a HIGHER t than the ROI's
  +1.74 because ROI carries payout dispersion the win rate does not.

  The longshot explanation is dead and specifically dead. ROI without the top
  5 winners is +0.1024 against a planted-genuine-edge benchmark of +0.0993,
  and without the top 10 it is +0.0584 against +0.0577. Both match a real
  edge almost exactly. ROI if every win paid the MEDIAN price is +0.1937,
  ABOVE the actual +0.1475, so payout dispersion works against this rule.

  The full-procedure placebo, selection step included, gives mean -0.0437 with
  sd 0.0454 and 0 of 400 draws reaching +0.1475. About 4.2 placebo SDs.

  AND THE COMPOSITION IS THE CLUE: 150 of 176 bets are UNDERS, at mean
  decimal odds 2.1626. The rule almost always backs the under at plus money.

WHY THAT MATTERS

  Three separate earlier findings point the same way:
    v1 Test A   every receptions price band had overs hitting LESS often
                than priced, six bands, all negative
    v3 Test G   blanket unders returned -0.0212 against blanket overs at
                -0.0974, a seven-point asymmetry
    anatomy     150 of 176 bets are unders

  So there is a real level bias toward overs in this market, and the rule
  concentrates it. The question this script settles is whether the rule is
  ANYTHING MORE than that bias.

THE DECISIVE TEST IS SECTION 3, NOT THE ROI COMPARISON

  If "back unders at plus money" works on its own, the edge is simpler and
  more robust than the candidate rule and the model is decoration.

  But comparing two ROIs does not answer it, because the model-free rule and
  the candidate rule bet overlapping populations. The clean question is
  CONDITIONAL: restrict to rows the model-free rule would take, then ask
  whether the model's edge filter separates winners from losers INSIDE that
  population. That is a difference in win rates on the same rows, which is
  where the model either earns its place or does not.

  Note one subtlety that makes the model-free rule interesting rather than
  trivial. v1 Test B measured the price slope at 1.008, meaning the price is
  an efficient forecast AT THE MARGIN. Fading it should therefore be neutral
  to negative. A level bias is a different object from a marginal slope, and
  a rule can exploit the first while the second stays efficient.

PRE-REGISTERED PREDICTIONS

  B1. Blanket unders on 2.5 and 3.5 returns about -0.03, negative. Taken
      from v3 Test G's -0.0281 on this exact population. Not a rule.
  B2. Unders restricted to plus money beats blanket unders but lands BELOW
      the candidate rule, holdout ROI under +0.08.
  B3. Overlap is high: more than 60 percent of the candidate rule's 176 bets
      are also unders at plus money.
  B4. THE DECISIVE ONE. Inside the model-free population, the model's edge
      filter adds a POSITIVE win-rate difference of 3 points or more. If it
      comes back at zero with a tight SE, the model is decoration and the
      honest rule is the price rule. If it comes back positive, the model is
      doing real selection and the +8.83 points is not merely the level bias.

  I genuinely do not know which way B4 goes. Everything in v1 to v4 says the
  model adds nothing on average; the anatomy says the rule works. Both can be
  true only if the model's contribution lives entirely in the tail, and this
  is the test that looks there.

THE GATES

  Gate 1: the candidate rule, +0.1475 on 176 bets at win rate 0.5341.
  Gate 2: the anatomy's price figures, fair probability 0.4458 and win-rate
          gap +8.83 points.

  Both are published. No gate may be widened to make a run pass.

Run from the repo root with the venv active:
    python model_free_test.py --all-seasons --cache lines_cache.parquet
    python model_free_test.py --all-seasons --cache lines_cache.parquet --placebo 400
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
TOL = 0.010
TOL_PP = 0.30

# Price cutoffs for the model-free sweep, in DECIMAL odds on the under.
# 2.00 is exactly plus money. Below that the under is the favourite.
DEC_CUTS = [1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50]


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def fmt_p(p, thresh):
    if p is None or not np.isfinite(p):
        return "     na"
    return f"{p:>7.4f}{'*' if p < thresh else ' '}"


def mde(se):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


# ------------------------------------------------------------------ the build

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

    # model-free quantities: the UNDER, always, at FanDuel's own price
    s = s.copy()
    s["dec_under"] = 1.0 + s["pay_under"]
    s["dec_over"] = 1.0 + s["pay_over"]
    tot = s["be_over"] + s["be_under"]
    s["fair_under"] = s["be_under"] / tot
    s["fair_over"] = s["be_over"] / tot
    s["won_under"] = (s["actual"] < s["bet_line"]).astype(float)
    s["prof_under"] = np.where(s["won_under"] > 0.5, s["pay_under"], -1.0)
    s["won_over"] = (s["actual"] > s["bet_line"]).astype(float)
    s["prof_over"] = np.where(s["won_over"] > 0.5, s["pay_over"], -1.0)
    print(f"\n  priced {len(s)} rows, {int(s['push'].sum())} pushes")
    return s


def select_holdout(s, edge_col, profit_col, won_col, grid, min_train=100):
    """The v5 holdout procedure, generalised to any edge column and grid.

    Threshold is chosen by maximising mean profit on the OTHER seasons, then
    spent on the held-out one. Identical discipline to the candidate rule, so
    the comparison between rules is fair rather than one rule getting a
    looser selection than another.
    """
    rows = []
    per = []
    for s_out in sorted(s["season"].unique()):
        tr = s[s["season"] != s_out]
        te = s[s["season"] == s_out]
        bt, bv = None, -np.inf
        for t in grid:
            g = tr[(tr[edge_col] >= t) & (~tr["push"])]
            if len(g) < min_train:
                continue
            v = float(g[profit_col].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[(te[edge_col] >= bt) & (~te["push"])].copy()
        if len(g) < 20:
            per.append((s_out, bt, len(g), np.nan, np.nan))
            continue
        g["held_out"] = s_out
        g["chosen_t"] = bt
        cm = eh.cluster_mean(g[profit_col].to_numpy(),
                             g["event_id"].to_numpy())
        per.append((s_out, bt, len(g), cm["mean"], cm["se"]))
        rows.append(g)
    if not rows:
        return None, per
    return pd.concat(rows, ignore_index=True), per


def summarise(label, bets, profit_col, won_col, fair_col):
    """One row: bets, ROI, win rate, fair probability, gap, both t values."""
    if bets is None or len(bets) < 20:
        print(f"  {label:<38}{'too few bets':>40}")
        return None
    cm_p = eh.cluster_mean(bets[profit_col].to_numpy(),
                           bets["event_id"].to_numpy())
    cm_w = eh.cluster_mean(bets[won_col].to_numpy(),
                           bets["event_id"].to_numpy())
    wr = cm_w["mean"]
    fair = float(bets[fair_col].mean())
    gap = 100.0 * (wr - fair)
    t_roi = cm_p["mean"] / cm_p["se"] if cm_p["se"] > 0 else np.nan
    t_gap = (wr - fair) / cm_w["se"] if cm_w["se"] > 0 else np.nan
    print(f"  {label:<38}{len(bets):>6}{cm_p['mean']:>+9.4f}"
          f"{t_roi:>7.2f}{wr:>9.4f}{fair:>9.4f}{gap:>+8.2f}{t_gap:>7.2f}")
    return {"n": len(bets), "roi": cm_p["mean"], "se_roi": cm_p["se"],
            "wr": wr, "se_wr": cm_w["se"], "fair": fair, "gap": gap,
            "t_roi": t_roi, "t_gap": t_gap}


# ---------------------------------------------------------------------- gates

def gate(s):
    print("\n" + "=" * 96)
    print("GATE 1 and 2: the candidate rule and its price figures")
    print("=" * 96)
    bets, per = select_holdout(s, "edge", "profit", "won", et5.GRID)
    if bets is None:
        print("  no bets selected. GATE FAILED.")
        sys.exit(2)
    cm = eh.cluster_mean(bets["profit"].to_numpy(),
                         bets["event_id"].to_numpy())
    wr = float(bets["won"].mean())
    be_ch = np.where(bets["side_over"], bets["be_over"], bets["be_under"])
    be_ot = np.where(bets["side_over"], bets["be_under"], bets["be_over"])
    fair = (be_ch / (be_ch + be_ot)).mean()
    gap = 100.0 * (wr - fair)

    checks = [
        ("ROI", cm["mean"], PUB_RULE["roi"], TOL),
        ("bets", float(len(bets)), float(PUB_RULE["bets"]), 0.5),
        ("win rate", wr, PUB_RULE["win_rate"], TOL),
        ("fair probability", fair, PUB_PRICE["fair"], TOL),
        ("win-rate gap pp", gap, PUB_PRICE["gap_pp"], TOL_PP),
    ]
    fails = []
    print(f"  {'quantity':<20}{'got':>11}{'published':>12}{'diff':>10}"
          f"   verdict")
    for name, got, pub, tol in checks:
        d = got - pub
        ok = abs(d) <= tol
        print(f"  {name:<20}{got:>11.4f}{pub:>12.4f}{d:>+10.4f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    if fails:
        print(f"\n  GATE FAILED for {fails}. Stop. Do not widen tolerance.")
        sys.exit(2)
    print("  GATES PASSED. This is the published candidate rule.")
    return bets


# ------------------------------------------------------------------- section 1

def section_1(s, cand):
    """Model-free rules, same holdout discipline, on the same population."""
    print("\n" + "=" * 96)
    print("1. MODEL-FREE RULES: no projection anywhere")
    print("=" * 96)
    print(f"  {'rule':<38}{'bets':>6}{'ROI':>9}{'t':>7}{'winrt':>9}"
          f"{'fair':>9}{'gap pp':>8}{'t gap':>7}")

    summarise("CANDIDATE RULE (for reference)", cand, "profit", "won",
              "_fair_chosen") if "_fair_chosen" in cand.columns else None
    # recompute the candidate's fair column for the shared summariser
    c = cand.copy()
    be_ch = np.where(c["side_over"], c["be_over"], c["be_under"])
    be_ot = np.where(c["side_over"], c["be_under"], c["be_over"])
    c["_fair"] = be_ch / (be_ch + be_ot)
    summarise("CANDIDATE RULE (model, reference)", c, "profit", "won",
              "_fair")

    all_rows = s[~s["push"]].copy()
    summarise("blanket unders, every row", all_rows, "prof_under",
              "won_under", "fair_under")
    summarise("blanket overs, every row", all_rows, "prof_over",
              "won_over", "fair_over")

    print()
    for cdec in DEC_CUTS:
        g = all_rows[all_rows["dec_under"] >= cdec]
        summarise(f"unders at decimal >= {cdec:.2f}", g, "prof_under",
                  "won_under", "fair_under")

    print("\n  IN-SAMPLE, no holdout. These sweep cells are selected and")
    print("  biased upward. Section 2 applies the same holdout discipline")
    print("  the candidate rule was held to.")
    return all_rows


# ------------------------------------------------------------------- section 2

def section_2(all_rows):
    """The price rule under the candidate rule's own holdout discipline."""
    print("\n" + "=" * 96)
    print("2. THE PRICE RULE, HOLDOUT SELECTED (same discipline as the rule)")
    print("=" * 96)
    grid = np.round(np.arange(1.50, 2.81, 0.05), 2)
    bets, per = select_holdout(all_rows, "dec_under", "prof_under",
                               "won_under", grid)
    print(f"  {'held out':>9}{'chosen dec':>12}{'bets':>7}{'ROI':>9}"
          f"{'SE':>8}")
    for s_out, bt, n, roi, se in per:
        if np.isfinite(roi):
            print(f"  {int(s_out):>9}{bt:>12.2f}{n:>7}{roi:>+9.4f}"
                  f"{se:>8.4f}")
        else:
            print(f"  {int(s_out):>9}{bt:>12.2f}{n:>7}   too few for a SE")
    if bets is None:
        print("  nothing selected")
        return None
    print()
    print(f"  {'rule':<38}{'bets':>6}{'ROI':>9}{'t':>7}{'winrt':>9}"
          f"{'fair':>9}{'gap pp':>8}{'t gap':>7}")
    r = summarise("PRICE RULE, holdout", bets, "prof_under", "won_under",
                  "fair_under")
    print("\n  Compare against the candidate rule's +0.1475, t +1.74,")
    print("  win-rate gap +8.83 pp at t +2.30.")
    return bets


# ------------------------------------------------------------------- section 3

def section_3(all_rows):
    """THE DECISIVE TEST. Does the model separate winners INSIDE the
    model-free population?

    Restricted to unders at plus money, split by whether the candidate
    rule's edge clears a threshold chosen on the other seasons. The
    difference in win rates is fitted as a clustered regression of the
    outcome on the indicator, so the SE accounts for props sharing a game.

    A positive difference means the model earns its place. A zero with a
    tight SE means the model is decoration and the price rule is the honest
    rule. The MDE is printed either way.
    """
    print("\n" + "=" * 96)
    print("3. DECISIVE: inside the price rule's population, does the model")
    print("   separate winners from losers?")
    print("=" * 96)
    pop = all_rows[all_rows["dec_under"] >= 2.00].copy()
    print(f"  population: unders at plus money, {len(pop)} rows")
    if len(pop) < 200:
        print("  too few rows")
        return

    # threshold chosen leave-season-out on the MODEL edge, profit on the under
    rows = []
    for s_out in sorted(pop["season"].unique()):
        tr = pop[pop["season"] != s_out]
        te = pop[pop["season"] == s_out]
        bt, bv = None, -np.inf
        for t in et5.GRID:
            g = tr[tr["edge"] >= t]
            if len(g) < 100:
                continue
            v = float(g["prof_under"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        te = te.copy()
        te["above"] = (te["edge"] >= bt).astype(float)
        te["chosen_t"] = bt
        rows.append(te)
    if not rows:
        print("  no seasons selectable")
        return
    d = pd.concat(rows, ignore_index=True)

    n_hi = int(d["above"].sum())
    n_lo = int((1 - d["above"]).sum())
    print(f"  thresholds chosen leave-season-out: "
          f"{sorted(d['chosen_t'].unique())}")
    print(f"  above threshold {n_hi} rows, below {n_lo} rows\n")

    print(f"  {'group':<38}{'bets':>6}{'ROI':>9}{'t':>7}{'winrt':>9}"
          f"{'fair':>9}{'gap pp':>8}{'t gap':>7}")
    hi = summarise("model says BET (edge >= t)", d[d["above"] > 0.5],
                   "prof_under", "won_under", "fair_under")
    lo = summarise("model says SKIP (edge < t)", d[d["above"] < 0.5],
                   "prof_under", "won_under", "fair_under")

    f = eh.ols_clustered(d["above"], d["won_under"], d["event_id"])
    if f is not None:
        t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
        print(f"\n  DIFFERENCE IN WIN RATES, clustered on game")
        print(f"    model BET minus model SKIP: "
              f"{100.0 * f['beta']:+.2f} pp   SE {100.0 * f['se_beta']:.2f}"
              f"   t {t:+.2f}   p {two_sided_p(t):.4f}")
        print(f"    MDE {100.0 * mde(f['se_beta']):.2f} pp")
        if abs(t) < 2:
            print(f"\n    The model does NOT separate winners here at this n.")
            print(f"    Read the MDE before calling it zero: an effect up to")
            print(f"    {100.0 * mde(f['se_beta']):.1f} pp is not excluded.")
        else:
            print(f"\n    The model DOES separate winners inside the price")
            print(f"    rule's population. It is not decoration.")

    # and the mirror: does the PRICE separate winners inside the MODEL's pop?
    print("\n  MIRROR: inside the model's own selected population, does the")
    print("  price add anything?")
    mp = all_rows[all_rows["edge"] >= 0.090].copy()
    if len(mp) >= 200:
        mp["plus"] = (mp["dec_under"] >= 2.00).astype(float)
        f2 = eh.ols_clustered(mp["plus"], mp["won_under"], mp["event_id"])
        if f2 is not None:
            t2 = f2["beta"] / f2["se_beta"] if f2["se_beta"] > 0 else np.nan
            print(f"    plus money minus minus money: "
                  f"{100.0 * f2['beta']:+.2f} pp   "
                  f"SE {100.0 * f2['se_beta']:.2f}   t {t2:+.2f}")
            print(f"    MDE {100.0 * mde(f2['se_beta']):.2f} pp")
            print(f"    (edge >= 0.090 fixed here, not holdout selected, so")
            print(f"     this is descriptive rather than a rule)")


# ------------------------------------------------------------------- section 4

def section_4(cand, all_rows):
    """How much do the two rules actually overlap?"""
    print("\n" + "=" * 96)
    print("4. OVERLAP: are these the same bets?")
    print("=" * 96)
    n = len(cand)
    unders = int((~cand["side_over"]).sum())
    plus = int((cand["dec_under"] >= 2.00).sum())
    both = int(((~cand["side_over"]) & (cand["dec_under"] >= 2.00)).sum())
    print(f"  candidate rule bets                     {n}")
    print(f"    of which UNDER                        {unders} "
          f"({100.0 * unders / n:.1f}%)")
    print(f"    of which the under is plus money      {plus} "
          f"({100.0 * plus / n:.1f}%)")
    print(f"    both under AND plus money             {both} "
          f"({100.0 * both / n:.1f}%)")
    pool = all_rows[(all_rows["dec_under"] >= 2.00)]
    print(f"\n  price-rule population (unders at plus money) {len(pool)} rows")
    print(f"  candidate rule selects {both} of them, "
          f"{100.0 * both / max(len(pool), 1):.1f}%")
    print("\n  A high overlap does not settle anything by itself. Section 3")
    print("  is what decides whether the model adds to the price.")


# ------------------------------------------------------------------- placebo

def placebo(all_rows, n_draws, seed=0):
    """Null for the PRICE rule's whole procedure, matching the anatomy's."""
    if not n_draws:
        return
    print("\n" + "=" * 96)
    print(f"5. FULL-PROCEDURE PLACEBO for the price rule, {n_draws} draws")
    print("=" * 96)
    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(1.50, 2.81, 0.05), 2)
    vals = []
    for _ in range(n_draws):
        sh = all_rows.copy()
        sh["prof_under"] = (sh.groupby(["season", "bet_line"])["prof_under"]
                            .transform(
                                lambda v: rng.permutation(v.to_numpy())))
        b, _ = select_holdout(sh, "dec_under", "prof_under", "won_under",
                              grid)
        if b is not None and len(b):
            vals.append(float(b["prof_under"].mean()))
    if not vals:
        print("  no usable draws")
        return
    v = np.asarray(vals)
    print(f"  placebo ROI mean {v.mean():+.4f}   sd {v.std():.4f}")
    print(f"  95% range [{np.percentile(v, 2.5):+.4f}, "
          f"{np.percentile(v, 97.5):+.4f}]")
    print(f"  all-rows under ROI for reference: "
          f"{all_rows['prof_under'].mean():+.4f}")
    print("\n  The candidate rule's own placebo was mean -0.0437, sd 0.0454,")
    print("  with 0 of 400 draws reaching +0.1475.")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--placebo", type=int, default=0)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("MODEL-FREE TEST: receptions, lines 2.5 and 3.5, FanDuel")
    print("=" * 96)

    s = build_priced(args)
    cand = gate(s)
    cand = cand.copy()
    cand["dec_under"] = 1.0 + cand["pay_under"]
    all_rows = section_1(s, cand)
    section_2(all_rows)
    section_3(all_rows)
    section_4(cand, all_rows)
    placebo(all_rows, args.placebo)

    if args.save_rows:
        cols = ["season", "week", "player", "bet_line", "projection",
                "blend", "sigma", "p_over", "edge", "side_over",
                "dec_under", "dec_over", "fair_under", "actual",
                "won_under", "prof_under", "event_id"]
        cols = [c for c in cols if c in all_rows.columns]
        all_rows[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"\n  wrote {len(all_rows)} rows to {args.save_rows}")

    print("\n" + "=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  Gates first. Section 1 is in-sample and biased upward, so use")
    print("  it only to see the shape. Section 2 is the price rule held to")
    print("  the same discipline as the candidate rule. Section 3 DECIDES:")
    print("  if the model separates winners inside the price rule's own")
    print("  population, it earns its place; if not, the price rule is the")
    print("  honest rule and the model is decoration. Read the MDE either")
    print("  way, because 'no separation at this n' is not 'no separation'.")
    print("=" * 96)


if __name__ == "__main__":
    main()
