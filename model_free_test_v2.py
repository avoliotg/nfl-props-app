"""
MODEL-FREE TEST v2: section 3 rebuilt, plus an honest power accounting

WHAT v1 SETTLED (run 2026-09-24, gates passed to four decimals)

  B1 CONFIRMED. Blanket unders on 2.5 and 3.5 return -0.0302, t -2.07.
  B2 CONFIRMED and then some. The price rule under the candidate rule's own
     holdout discipline returns -0.0401, not merely below the +0.08 I
     predicted. "Back unders at plus money" is NOT a rule.
  B3 CONFIRMED. 61.9 percent of the candidate rule's bets are unders at plus
     money.

  So the level bias toward overs, real though it is (v1 Test A had all six
  price bands negative), does not pay. The candidate rule is not that.

WHY v1's SECTION 3 WAS BROKEN, AND IT WAS MY ERROR

  It selected the threshold by maximising `prof_under`, not the rule's own
  profit on the rule's own chosen side. The thresholds it picked came back
  0.025, 0.035, 0.085, 0.090, so two of four sat near 0.03, and the split was
  653 rows above against 639 below.

  The candidate rule's thresholds are 0.090 to 0.100 and it takes 4 percent
  of rows. The overlap table says it directly: 109 of 1292 rows in that
  population, 8.4 percent.

  So v1 compared the top HALF against the bottom half and called it the tail
  test. It is not the rule's tail. Its +0.09 pp difference with an MDE of
  8.03 was uninformative rather than negative, and I should not have framed
  it as a decisive test that came back null.

WHAT v2 DOES INSTEAD

  3A. The rule's ACTUAL selection. Thresholds chosen exactly as v5 does
      (maximise mean profit on the rule's own chosen side, over the other
      seasons), then within the full population compare the rows the rule
      TAKES against the rows it SKIPS, scoring both on the side the rule
      would have taken. That is the real tail question.

  3B. DOSE RESPONSE, which has more power than a binary split because it
      uses the continuous variation in the edge. Regress the outcome on the
      edge, clustered on game, across all rows and then within the taken
      rows only. If the edge is informative, higher edge means higher win
      rate, and that is testable without any threshold at all.

  3C. AN EXPLICIT POWER ACCOUNTING, printed BEFORE the results.
      This matters more than either test. At a base win rate near 0.46, the
      SE of a win-rate difference is about sqrt(p(1-p)/n) for the smaller
      arm, so detecting the observed +8.83 pp at 80 percent power needs
      roughly 250 to 300 bets in that arm. The rule has 176 holdout bets.
      The tail test is therefore UNDERPOWERED BY CONSTRUCTION, and saying so
      in advance is the difference between a null result and an absence of
      evidence.

PRE-REGISTERED PREDICTIONS

  C1. 3A comes back POSITIVE in point estimate, somewhere between +3 and
      +10 pp, and does NOT clear t 2. Reasoning: the rule's ROI is positive
      and its win-rate gap is +8.83, so a positive point estimate is close
      to arithmetic; and 3C says the n cannot resolve it.
  C2. 3B's dose response is POSITIVE across all rows. This is the one test
      here with real power, about 4200 rows, and it is the cleanest single
      statement of whether the edge means anything.
  C3. 3B's dose response WITHIN the taken rows is indistinguishable from
      zero, because within a selected tail the remaining variation is small.

  C2 is the one to watch. If the dose response is positive and significant
  across the full population, then the edge is informative and v3's Test E
  was measuring the wrong thing (a linear mean-space slope) rather than
  measuring nothing. If it is flat, the rule's 176 bets are the only
  evidence that anything is there.

THE GATES

  Gate 1: the candidate rule, +0.1475 on 176 bets at win rate 0.5341.
  Gate 2: the anatomy's price figures, fair 0.4458 and gap +8.83 pp.
  Gate 3: v1's own published price-rule holdout, -0.0401.

Run from the repo root with the venv active:
    python model_free_test_v2.py --all-seasons --cache lines_cache.parquet
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
TOL = 0.010
TOL_PP = 0.30


def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return np.nan
    from math import erfc, sqrt
    return erfc(abs(t) / sqrt(2.0))


def mde(se):
    if se is None or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.80 * se


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
    print(f"\n  priced {len(s)} rows, {int(s['push'].sum())} pushes")
    return s


def select_thresholds(s, profit_col="profit", edge_col="edge",
                      min_train=100):
    """v5's threshold choice, returning {season: threshold}.

    Maximise mean profit on the OTHER seasons over v5's own GRID, requiring
    at least 100 training bets. This is the rule's actual selection rule and
    v1's section 3 did not use it.
    """
    out = {}
    for s_out in sorted(s["season"].unique()):
        tr = s[s["season"] != s_out]
        bt, bv = None, -np.inf
        for t in et5.GRID:
            g = tr[(tr[edge_col] >= t) & (~tr["push"])]
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
    print("GATES 1 to 3")
    print("=" * 96)
    th = select_thresholds(s)
    rows = []
    for s_out, t in th.items():
        g = s[(s["season"] == s_out) & (s["edge"] >= t) & (~s["push"])].copy()
        if len(g) >= 20:
            g["held_out"] = s_out
            g["chosen_t"] = t
            rows.append(g)
    bets = pd.concat(rows, ignore_index=True)
    cm = eh.cluster_mean(bets["profit"].to_numpy(),
                         bets["event_id"].to_numpy())
    wr = float(bets["won"].mean())
    fair = float(bets["fair_chosen"].mean())
    gap = 100.0 * (wr - fair)

    # v1's price rule, for gate 3
    grid = np.round(np.arange(1.50, 2.81, 0.05), 2)
    pth = select_thresholds(s, profit_col="prof_under",
                            edge_col="dec_under")
    prows = []
    for s_out, t in pth.items():
        g = s[(s["season"] == s_out) & (s["dec_under"] >= t)
              & (~s["push"])]
        if len(g) >= 20:
            prows.append(g)
    price_roi = (pd.concat(prows, ignore_index=True)["prof_under"].mean()
                 if prows else np.nan)

    checks = [
        ("ROI", cm["mean"], PUB_RULE["roi"], TOL),
        ("bets", float(len(bets)), float(PUB_RULE["bets"]), 0.5),
        ("win rate", wr, PUB_RULE["win_rate"], TOL),
        ("fair probability", fair, PUB_PRICE["fair"], TOL),
        ("win-rate gap pp", gap, PUB_PRICE["gap_pp"], TOL_PP),
        ("v1 price rule ROI", price_roi, PUB_V1_PRICE_RULE, TOL),
    ]
    fails = []
    print(f"  {'quantity':<22}{'got':>11}{'published':>12}{'diff':>10}"
          f"   verdict")
    for name, got, pub, tol in checks:
        d = got - pub
        ok = np.isfinite(d) and abs(d) <= tol
        print(f"  {name:<22}{got:>11.4f}{pub:>12.4f}{d:>+10.4f}"
              f"   {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    if fails:
        print(f"\n  GATES FAILED for {fails}. Stop.")
        sys.exit(2)
    print("  GATES PASSED.")
    print(f"\n  thresholds the rule actually uses: "
          f"{ {int(k): round(v, 3) for k, v in th.items()} }")
    return bets, th


def section_3c(s, th, bets):
    """Power accounting, printed BEFORE the results. This is the point."""
    print("\n" + "=" * 96)
    print("3C. POWER ACCOUNTING, read this before any result below")
    print("=" * 96)
    taken = 0
    skipped = 0
    for s_out, t in th.items():
        g = s[(s["season"] == s_out) & (~s["push"])]
        taken += int((g["edge"] >= t).sum())
        skipped += int((g["edge"] < t).sum())
    p = float(bets["fair_chosen"].mean())
    var = p * (1.0 - p)
    print(f"  base win probability of the chosen side  {p:.4f}")
    print(f"  rows the rule TAKES                      {taken}")
    print(f"  rows the rule SKIPS                      {skipped}")
    se_diff = np.sqrt(var * (1.0 / max(taken, 1) + 1.0 / max(skipped, 1)))
    print(f"\n  analytic SE of a win-rate difference     {100 * se_diff:.2f} pp")
    print(f"  minimum detectable difference, 80% power "
          f"{100 * 2.80 * se_diff:.2f} pp")
    obs = PUB_PRICE["gap_pp"] / 100.0
    need = var * (2.80 / obs) ** 2 if obs > 0 else np.nan
    print(f"\n  the observed gap is                      "
          f"{PUB_PRICE['gap_pp']:.2f} pp")
    print(f"  bets needed in the small arm to resolve  {need:.0f}")
    print(f"  bets the rule has                        {len(bets)}")
    print(f"  weeks at 2.5 bets per week to get there  "
          f"{max(need - len(bets), 0) / 2.5:.0f}")
    print("\n  So the binary tail test below is UNDERPOWERED BY")
    print("  CONSTRUCTION. A point estimate near zero there is an absence")
    print("  of evidence, not evidence of absence. 3B has more power.")


def section_3a(s, th):
    """The rule's real selection: rows TAKEN against rows SKIPPED."""
    print("\n" + "=" * 96)
    print("3A. THE RULE'S ACTUAL TAIL: taken vs skipped, same side scored")
    print("=" * 96)
    rows = []
    for s_out, t in th.items():
        g = s[(s["season"] == s_out) & (~s["push"])].copy()
        g["above"] = (g["edge"] >= t).astype(float)
        rows.append(g)
    d = pd.concat(rows, ignore_index=True)

    print(f"  {'group':<34}{'rows':>7}{'ROI':>9}{'t':>7}{'winrt':>9}"
          f"{'fair':>9}{'gap pp':>8}{'t gap':>7}")
    for lab, sub in (("rule TAKES (edge >= t)", d[d["above"] > 0.5]),
                     ("rule SKIPS (edge < t)", d[d["above"] < 0.5])):
        if len(sub) < 20:
            print(f"  {lab:<34}{len(sub):>7}   too few")
            continue
        cp = eh.cluster_mean(sub["profit"].to_numpy(),
                             sub["event_id"].to_numpy())
        cw = eh.cluster_mean(sub["won_f"].to_numpy(),
                             sub["event_id"].to_numpy())
        fair = float(sub["fair_chosen"].mean())
        gap = 100.0 * (cw["mean"] - fair)
        t_roi = cp["mean"] / cp["se"] if cp["se"] > 0 else np.nan
        t_gap = (cw["mean"] - fair) / cw["se"] if cw["se"] > 0 else np.nan
        print(f"  {lab:<34}{len(sub):>7}{cp['mean']:>+9.4f}{t_roi:>7.2f}"
              f"{cw['mean']:>9.4f}{fair:>9.4f}{gap:>+8.2f}{t_gap:>7.2f}")

    f = eh.ols_clustered(d["above"], d["won_f"], d["event_id"])
    if f is not None:
        t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
        print(f"\n  win-rate difference, TAKES minus SKIPS")
        print(f"    {100 * f['beta']:+.2f} pp   SE {100 * f['se_beta']:.2f}"
              f"   t {t:+.2f}   p {two_sided_p(t):.4f}"
              f"   MDE {100 * mde(f['se_beta']):.2f} pp")
    fp = eh.ols_clustered(d["above"], d["profit"], d["event_id"])
    if fp is not None:
        t = fp["beta"] / fp["se_beta"] if fp["se_beta"] > 0 else np.nan
        print(f"  ROI difference, TAKES minus SKIPS")
        print(f"    {fp['beta']:+.4f}   SE {fp['se_beta']:.4f}"
              f"   t {t:+.2f}   MDE {mde(fp['se_beta']):.4f}")


def section_3b(s, th):
    """Dose response. More power than any split, and no threshold needed."""
    print("\n" + "=" * 96)
    print("3B. DOSE RESPONSE: does a bigger edge mean a higher win rate?")
    print("=" * 96)
    d = s[~s["push"]].copy()
    print(f"  Outcome is the win on the side the layer would take, so a")
    print(f"  positive slope means the edge is informative.\n")
    print(f"  {'population':<34}{'rows':>7}{'slope':>10}{'SE':>9}"
          f"{'t':>7}{'p':>9}{'MDE':>9}")

    f = eh.ols_clustered(d["edge"], d["won_f"], d["event_id"])
    if f is not None:
        t = f["beta"] / f["se_beta"]
        print(f"  {'all rows':<34}{f['n']:>7}{f['beta']:>10.4f}"
              f"{f['se_beta']:>9.4f}{t:>7.2f}"
              f"{two_sided_p(t):>9.4f}{mde(f['se_beta']):>9.4f}")

    rows = []
    for s_out, t_ in th.items():
        g = d[(d["season"] == s_out) & (d["edge"] >= t_)]
        rows.append(g)
    tk = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(tk) >= 50:
        f2 = eh.ols_clustered(tk["edge"], tk["won_f"], tk["event_id"])
        if f2 is not None:
            t2 = f2["beta"] / f2["se_beta"]
            print(f"  {'within the taken rows only':<34}{f2['n']:>7}"
                  f"{f2['beta']:>10.4f}{f2['se_beta']:>9.4f}{t2:>7.2f}"
                  f"{two_sided_p(t2):>9.4f}{mde(f2['se_beta']):>9.4f}")

    print("\n  BY EDGE DECILE, all rows, to see the shape rather than a slope")
    d["dec"] = pd.qcut(d["edge"], 10, labels=False, duplicates="drop")
    print(f"  {'decile':>7}{'rows':>7}{'edge lo':>9}{'edge hi':>9}"
          f"{'winrt':>9}{'fair':>9}{'gap pp':>8}{'SE pp':>8}{'ROI':>9}")
    for q, g in d.groupby("dec"):
        cw = eh.cluster_mean(g["won_f"].to_numpy(),
                             g["event_id"].to_numpy())
        if cw is None:
            continue
        fair = float(g["fair_chosen"].mean())
        print(f"  {int(q):>7}{len(g):>7}{g['edge'].min():>9.4f}"
              f"{g['edge'].max():>9.4f}{cw['mean']:>9.4f}{fair:>9.4f}"
              f"{100 * (cw['mean'] - fair):>+8.2f}{100 * cw['se']:>8.2f}"
              f"{g['profit'].mean():>+9.4f}")
    print("\n  A monotone rise in the gap column is what a real edge looks")
    print("  like. A flat set of deciles with only the top one positive is")
    print("  what threshold selection on noise looks like.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("MODEL-FREE TEST v2: section 3 rebuilt on the rule's own threshold")
    print("=" * 96)

    s = build_priced(args)
    bets, th = gate(s)
    section_3c(s, th, bets)
    section_3a(s, th)
    section_3b(s, th)

    if args.save_rows:
        cols = ["season", "week", "player", "bet_line", "projection",
                "blend", "sigma", "p_over", "edge", "side_over",
                "fair_chosen", "dec_under", "actual", "won_f", "profit",
                "event_id"]
        cols = [c for c in cols if c in s.columns]
        s[cols].to_csv(args.save_rows, index=False, encoding="utf-8")
        print(f"\n  wrote {len(s)} rows to {args.save_rows}")

    print("\n" + "=" * 96)
    print("READING ORDER")
    print("=" * 96)
    print("  3C first, so the results are read against the power available.")
    print("  Then 3B, which has about 4200 rows and is the only test here")
    print("  with real power. The decile table is the most informative")
    print("  single object: monotone means a real edge, flat with only the")
    print("  top decile positive means selection on noise.")
    print("  3A is underpowered and its point estimate is what matters,")
    print("  not its p-value.")
    print("=" * 96)


if __name__ == "__main__":
    main()
