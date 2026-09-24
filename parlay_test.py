"""
SINGLES VERSUS PARLAYS, and the cold-streak hypothesis.

TWO QUESTIONS, BOTH PROMPTED BY A REDDIT MLB MODEL

The post: filter to good hitters who play daily, list the ones who have gone
two games without a hit, parlay two of them to get a hit. Claimed 80 percent
on 2-leg parlays, posted record 2-2.

The stated reasoning is the gambler's fallacy and does not survive contact.
For a 2-leg parlay to hit 80 percent each leg must win 89.4 percent, while the
legs were priced at 68 to 69 percent and a good hitter's true rate is 65 to 71.

But two things in it are worth testing properly, and one of them corrects an
error I made when first dismissing it.

QUESTION 1: DOES PARLAYING HELP OR HURT?

I said parlaying two uncorrelated legs is strictly worse EV. THAT IS WRONG.
Edges MULTIPLY. Write r = p * d for the return multiplier of a leg at decimal
odds d with true probability p. A single bet returns r - 1 per unit. A 2-leg
parlay returns r1*r2 - 1. Two legs at r = 1.05 give a parlay at 1.1025, so
10.25 percent rather than 5.

What parlaying really does is explode the variance. So the comparison cannot
be made on ROI, it has to be made on GROWTH RATE, which is
E[log(1 + f * profit)] maximised over the bankroll fraction f. A strategy with
higher ROI and far higher variance can have a lower growth rate, because the
Kelly-optimal stake shrinks faster than the edge grows.

And the asymmetry matters: if the legs are NEGATIVE EV, multiplying makes it
much worse. Two legs at r = 0.95 give 0.9025, so -9.75 percent instead of -5.

So the honest answer depends on whether the candidate rule's legs are really
+EV, which is exactly what is still uncertain at t +1.74.

IMPLEMENTATION NOTE ON FORMING PARLAYS. Taking all possible pairs inflates the
sample and correlates observations, since one bet appears in many pairs.
Instead the qualifying bets in each week are randomly paired into DISJOINT
pairs, repeated many times, and the results averaged. Legs are required to come
from DIFFERENT GAMES, because two props in the same game are correlated
through game script and pace, which would break the independence the parlay
price assumes.

QUESTION 2: DOES A COLD STREAK PREDICT ANYTHING?

Strip out the "due for a hit" reasoning and a defensible version remains: does
the market OVERREACT to short-term slumps? If a book shades a price after a
cold run beyond what the underlying rate justifies, buying cold players is
buying the overcorrection. That has nothing to do with regression being owed.

The football analogue of "gets a hit" is a 1+ reception prop. And the ladder
calibration found exactly one anomalous rung out of ten:

    rung k=1   predicted 0.8917   realized 0.9030   gap +1.1 pp   z +3.62

Every other rung was inside half a point. The model UNDERSTATES P(at least one
catch), and it is the only significant miss in the table. That is the rung the
Reddit idea points at.

So this script computes games since last reception, lagged so it is known
before kickoff, and asks whether beta or realized ROI varies with it. If the
coefficient is flat, the effect is the fallacy. If cold players beat the line,
the market overreacts and there is something there.

Run from the repo root:
    python parlay_test.py --all-seasons --cache lines_cache.parquet
    python parlay_test.py --all-seasons --cache lines_cache.parquet --min-edge 0.09
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh
import edge_threshold_v5 as et


# --------------------------------------------------------------------- data

def load(args):
    sec = eh._secrets()
    box = {}

    def factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    seasons = (list(eh.ALL_SEASONS) if args.all_seasons
               else [int(x) for x in (args.seasons or "2025").split(",")])
    lines = eh.load_lines(factory, seasons, ["receptions"], args.cache,
                          args.refresh)
    if lines.empty:
        sys.exit("  no lines found")
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)

    print("\nscoring receptions walk-forward")
    proj = eh.score_market("receptions", seasons, mode="walk_forward",
                           population=args.score_population)
    if not len(proj):
        sys.exit("  nothing scored")

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)
    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"  joined {len(j)} rows")
    j["dev_cons"] = j["projection"] - j["line_consensus"]
    j["out_cons"] = j["actual"] - j["line_consensus"]
    j = j.dropna(subset=["dev_cons", "out_cons"]).copy()

    vals = [round(float(x), 2) for x in args.line_value.split(",")]
    j = j[np.isin(np.round(j["line_consensus"], 2), vals)]
    print(f"  consensus line in {vals}: {len(j)} rows")
    return j


def add_cold_streak(j):
    """Games since this player last recorded a reception, LAGGED.

    Must be lagged by one game or it leaks the current result. A value of 0
    means he caught one last time out; 3 means three straight games without a
    catch.
    """
    j = j.sort_values(["_key", "season", "week"]).copy()
    out = np.zeros(len(j), dtype=float)
    run = {}
    for i, (k, act) in enumerate(zip(j["_key"].to_numpy(),
                                     j["actual"].to_numpy())):
        out[i] = run.get(k, 0)          # value BEFORE this game
        run[k] = 0 if act > 0 else run.get(k, 0) + 1
    j["cold"] = out
    print(f"\n  cold streak distribution (games since last catch, lagged):")
    for v in range(0, 5):
        n = int((j["cold"] == v).sum())
        print(f"    {v} games: {n:>6} ({100.0 * n / len(j):.1f}%)")
    n = int((j["cold"] >= 5).sum())
    print(f"    5+ games: {n:>6} ({100.0 * n / len(j):.1f}%)")
    return j


# --------------------------------------------------------------- question 2

def cold_streak_test(s, min_n=200):
    print("\n" + "=" * 96)
    print("QUESTION 2: does a cold streak predict anything?")
    print("=" * 96)
    print("  If the market overreacts to slumps, cold players beat the line.")
    print("  If it is the gambler's fallacy, these rows are flat.\n")
    s = s.copy()
    s["bucket"] = pd.cut(s["cold"], [-0.1, 0.1, 1.1, 2.1, 99],
                         labels=["0 (caught last game)", "1 game", "2 games",
                                 "3+ games"])
    print(f"  {'streak':<24}{'n':>7}{'beta':>8}{'SE':>7}{'t':>7}"
          f"{'over rate':>11}{'ROI':>9}{'SE':>8}{'t':>7}")
    for b in s["bucket"].cat.categories:
        g = s[s["bucket"] == b]
        if len(g) < min_n:
            print(f"  {str(b):<24}{len(g):>7}   too few")
            continue
        f = eh.ols_clustered(g["dev_cons"], g["out_cons"], g["event_id"])
        live = g[~g["push"]]
        cm = eh.cluster_mean(live["profit"].to_numpy(),
                             live["event_id"].to_numpy())
        won = (g["actual"] > g["line_consensus"]).astype(float)
        orate = float(won.mean())
        tb = f["beta"] / f["se_beta"] if f and f["se_beta"] > 0 else np.nan
        tr = cm["mean"] / cm["se"] if cm and cm["se"] > 0 else np.nan
        print(f"  {str(b):<24}{len(g):>7}"
              f"{f['beta'] if f else np.nan:>8.3f}"
              f"{f['se_beta'] if f else np.nan:>7.3f}{tb:>+7.2f}"
              f"{orate:>11.4f}"
              f"{cm['mean'] if cm else np.nan:>+9.4f}"
              f"{cm['se'] if cm else np.nan:>8.4f}{tr:>+7.2f}")
    print("\n  Also the 1+ RECEPTION question directly, which is the football")
    print("  analogue of 'gets a hit':")
    print(f"  {'streak':<24}{'n':>7}{'P(1+ catch)':>13}{'SE':>8}")
    for b in s["bucket"].cat.categories:
        g = s[s["bucket"] == b]
        if len(g) < min_n:
            continue
        hit = (g["actual"] >= 1).astype(float).to_numpy()
        cm = eh.cluster_mean(hit, g["event_id"].to_numpy())
        if cm:
            print(f"  {str(b):<24}{cm['n']:>7}{cm['mean']:>13.4f}"
                  f"{cm['se']:>8.4f}")
    print("\n    A FALLING P(1+ catch) as the streak lengthens means cold")
    print("    players are genuinely worse, not due. A flat or rising one")
    print("    means the streak carries no information about the next game.")


# --------------------------------------------------------------- question 1

def growth(profits, fmax=1.0, steps=200):
    """Kelly growth rate E[log(1 + f*profit)] and the optimal fraction."""
    p = np.asarray(profits, float)
    best = (0.0, 0.0)
    for f in np.linspace(0.001, fmax, steps):
        v = 1.0 + f * p
        if (v <= 0).any():
            break
        g = float(np.mean(np.log(v)))
        if g > best[1]:
            best = (f, g)
    return best


def parlay_test(s, n_iter=400, seed=0):
    print("\n" + "=" * 96)
    print("QUESTION 1: singles versus 2-leg parlays")
    print("=" * 96)
    live = s[~s["push"]].copy()
    if len(live) < 50:
        print("  too few bets")
        return
    rng = np.random.default_rng(seed)

    # singles
    cm = eh.cluster_mean(live["profit"].to_numpy(), live["event_id"].to_numpy())
    f1, g1 = growth(live["profit"].to_numpy())
    print(f"\n  SINGLES")
    print(f"    bets            {len(live)}")
    print(f"    win rate        {live['won'].mean():.4f}")
    print(f"    ROI per unit    {cm['mean']:+.4f}  SE {cm['se']:.4f}  "
          f"t {cm['mean'] / cm['se']:+.2f}")
    print(f"    SD of profit    {live['profit'].std():.4f}")
    print(f"    optimal Kelly f {f1:.3f}, growth rate {g1:.5f} per bet")

    # parlays: disjoint random pairs within a week, different games
    live["dec"] = 1.0 + np.where(live["side_over"], live["pay_over"],
                                 live["pay_under"])
    rois, hits, alln = [], [], []
    pooled = []
    for it in range(n_iter):
        prof = []
        for (se, wk), g in live.groupby(["season", "week"]):
            idx = rng.permutation(g.index.to_numpy())
            used = set()
            for a in idx:
                if a in used:
                    continue
                ev_a = g.loc[a, "event_id"]
                partner = None
                for b in idx:
                    if b in used or b == a:
                        continue
                    if g.loc[b, "event_id"] != ev_a:
                        partner = b
                        break
                if partner is None:
                    continue
                used.update([a, partner])
                w = bool(g.loc[a, "won"]) and bool(g.loc[partner, "won"])
                d = g.loc[a, "dec"] * g.loc[partner, "dec"]
                prof.append(d - 1.0 if w else -1.0)
        if prof:
            rois.append(float(np.mean(prof)))
            hits.append(float(np.mean([p > 0 for p in prof])))
            alln.append(len(prof))
            if it == 0:
                pooled = list(prof)
            else:
                pooled.extend(prof)
    if not rois:
        print("\n  could not form any parlays")
        return
    pooled = np.array(pooled)
    f2, g2 = growth(pooled)
    print(f"\n  2-LEG PARLAYS, disjoint random pairs, different games, "
          f"{n_iter} resamples")
    print(f"    parlays/resample {np.mean(alln):.0f}")
    print(f"    hit rate         {np.mean(hits):.4f}")
    print(f"    ROI per unit     {np.mean(rois):+.4f}  "
          f"resample SD {np.std(rois):.4f}")
    print(f"    SD of profit     {pooled.std():.4f}")
    print(f"    optimal Kelly f  {f2:.3f}, growth rate {g2:.5f} per bet")

    print("\n  VERDICT")
    print(f"    ROI:    singles {cm['mean']:+.4f}   parlays "
          f"{np.mean(rois):+.4f}   "
          f"{'parlays higher' if np.mean(rois) > cm['mean'] else 'singles higher'}")
    print(f"    GROWTH: singles {g1:.5f}      parlays {g2:.5f}      "
          f"{'parlays better' if g2 > g1 else 'singles better'}")
    print()
    print("    ROI is the wrong comparison on its own. Growth rate accounts")
    print("    for the variance, and variance is what parlays add. If ROI")
    print("    rises while growth falls, the parlay is a worse bet despite")
    print("    the bigger number, because the stake that survives the")
    print("    variance is smaller than the extra edge is worth.")
    print("    If BOTH are higher, the legs are genuinely +EV and multiplying")
    print("    them compounds a real edge. If both are lower, the legs are")
    print("    -EV and parlaying compounds the loss.")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--line-value", default="2.5,3.5")
    ap.add_argument("--min-edge", type=float, default=0.09,
                    help="the candidate rule's threshold")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    print("=" * 96)
    print("SINGLES VS PARLAYS, and the cold-streak hypothesis")
    print(f"  receptions, lines {args.line_value}, min edge {args.min_edge}")
    print("=" * 96)

    j = load(args)
    j = add_cold_streak(j)

    params = et.fit_params(j)
    if not params:
        sys.exit("  could not fit shrinkage parameters")
    et.report_params(params)
    sig = et.fit_sigma_sqrt(j, params)
    s = et.price(j, params, "fanduel", sig, "level", 1.0)
    print(f"\n  priced {len(s)} rows")

    cold_streak_test(s)

    q = s[s["edge"] >= args.min_edge].copy()
    print(f"\n  {len(q)} bets clear the {args.min_edge} edge threshold")
    parlay_test(q, n_iter=args.iters)


if __name__ == "__main__":
    main()
