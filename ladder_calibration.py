"""
LADDER CALIBRATION: can the distribution be trusted away from the posted line?

WHY THIS COMES BEFORE ANY ALT-LINE BETTING

Books post an alternate ladder for receptions: over 1.5, 2.5, 3.5, 4.5 and up,
each with its own price. The ladder is generated from one base distribution
plus a vig schedule, and the upper rungs attract far less money than the main
line, so that is where a book's shape assumptions are least defended.

Exploiting that requires YOUR distribution to be right where theirs is wrong.
And every calibration check in this project so far has evaluated the model
probability AT THE POSTED LINE ONLY. A negative binomial that nails
P(X >= 3) when the line is 2.5 tells you nothing about P(X >= 6), because the
line sits near the median and the tails were never tested.

So this script asks the question the alt idea actually rests on: for every
rung k, is the predicted P(X >= k) matched by the realized frequency?

  - If calibration holds across the ladder, the alt idea is live and the
    upper rungs are a genuine second surface to bet.
  - If it degrades in the tails, alt lines are a trap. You would be betting
    the part of your own distribution you have never validated, at longer
    odds, which turns a small shape error into a large loss.

WHAT THE TAIL ERROR WOULD LOOK LIKE

A negative binomial with the right mean and variance can still have the wrong
tail, because two parameters fix two moments and nothing else. If receptions
have more upside spikes than an NB allows, predicted P(X >= 7) comes out too
low and betting overs at the top of the ladder looks profitable when it is
not. The table below would show that as a positive gap growing with k.

WHAT THIS SCRIPT PRODUCES

  PART A, historical. Per-rung calibration over four seasons: predicted
  P(X >= k) against realized frequency, with a game-clustered standard error,
  plus the same split by whether the rung is below, at, or above the posted
  line. That last split is the one that matters: a rung three catches above
  the line is where the alt ladder lives.

  PART B, actionable this week. For a chosen season and week, the model's
  implied FAIR American odds at every rung, written to CSV. Compare those
  against FanDuel's actual alt prices by hand. Any rung where their price
  implies a probability materially below your fair one is a candidate bet, and
  the comparison costs nothing because you are reading it off their app.

NOTHING HERE NEEDS CREDITS. The ladder is computed from the model. Capturing
real alt odds is a separate step and costs about one extra credit per event.

Run from the repo root:
    python ladder_calibration.py --all-seasons --cache lines_cache.parquet
    python ladder_calibration.py --all-seasons --cache lines_cache.parquet \\
        --export-ladder 2026,2 --out ladder_2026wk2.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

import eval_harness as eh
import edge_threshold_v5 as et

MAXK = 10


def fair_american(p):
    """Probability to fair American odds, no vig."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = (p > 1e-6) & (p < 1 - 1e-6)
    hi = ok & (p >= 0.5)
    lo = ok & (p < 0.5)
    out[hi] = -100.0 * p[hi] / (1.0 - p[hi])
    out[lo] = 100.0 * (1.0 - p[lo]) / p[lo]
    return out


def sf_at(market, mean, sigma, k):
    """P(X >= k) for a count market, or P(X > k) for a continuous one."""
    mean = np.asarray(mean, float)
    sigma = np.asarray(sigma, float)
    out = np.full(mean.shape, np.nan)
    ok = np.isfinite(mean) & (mean > 1e-6) & (sigma > 1e-6)
    if not ok.any():
        return out
    if market in et.COUNT:
        m, v = mean[ok], sigma[ok] ** 2
        res = np.empty(m.shape)
        nb = v > m * 1.0001
        if nb.any():
            pp = m[nb] / v[nb]
            r = m[nb] * pp / (1.0 - pp)
            res[nb] = stats.nbinom.sf(k - 1, r, pp)
        if (~nb).any():
            res[~nb] = stats.poisson.sf(k - 1, m[~nb])
        out[ok] = res
    else:
        shape = (mean[ok] / sigma[ok]) ** 2
        scale = sigma[ok] ** 2 / mean[ok]
        out[ok] = stats.gamma.sf(k, a=shape, scale=scale)
    return out


# --------------------------------------------------------------------- data

def load(args):
    sec = eh._secrets()
    box = {}

    def factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(x) for x in args.seasons.split(",") if x.strip()]
    else:
        seasons = [2025]

    lines = eh.load_lines(factory, seasons, markets, args.cache, args.refresh)
    if lines.empty:
        sys.exit("  no lines found")
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)
    print(f"\n  {len(props)} distinct player-week props")

    print("\nscoring models walk-forward")
    scored = []
    for m in markets:
        print(f"  {m}")
        sc = eh.score_market(m, seasons, mode="walk_forward",
                             population=args.score_population)
        if len(sc):
            scored.append(sc)
    if not scored:
        sys.exit("  nothing scored")
    proj = pd.concat(scored, ignore_index=True)

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)

    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(j)} rows "
          f"({100.0 * len(j) / max(len(props), 1):.1f}%)")
    j["dev_cons"] = j["projection"] - j["line_consensus"]
    j["out_cons"] = j["actual"] - j["line_consensus"]
    return j.dropna(subset=["dev_cons", "out_cons", "line_consensus"]).copy()


def attach_distribution(j, params, sig):
    j = j.copy()
    key = list(zip(j["market"], j["season"]))
    j["alpha"] = [params.get(k, (np.nan,) * 4)[0] for k in key]
    j["beta"] = [params.get(k, (np.nan,) * 4)[1] for k in key]
    j["sig_a"] = [sig.get(k, (np.nan, 0.5))[0] for k in key]
    j = j.dropna(subset=["alpha", "beta", "sig_a"])
    j["blend"] = (j["line_consensus"] + j["alpha"]
                  + j["beta"] * (j["projection"] - j["line_consensus"]))
    j["sigma"] = j["sig_a"] * np.sqrt(j["blend"].clip(lower=1e-6))
    return j[j["blend"] > 0].copy()


# ------------------------------------------------------------------- PART A

def ladder_calibration(j, min_n=150):
    print("\n" + "=" * 100)
    print("PART A: per-rung calibration, P(X >= k) predicted against realized")
    print("=" * 100)
    for m, g in j.groupby("market"):
        print(f"\n  {m}   ({len(g)} props)")
        print(f"  {'rung k':>7}{'n':>7}{'predicted':>11}{'realized':>10}"
              f"{'SE':>8}{'gap (pp)':>10}{'z':>7}{'fair odds':>11}")
        for k in range(1, MAXK + 1):
            p = sf_at(m, g["blend"].to_numpy(), g["sigma"].to_numpy(), k)
            ok = np.isfinite(p)
            if ok.sum() < min_n:
                continue
            hit = (g["actual"].to_numpy()[ok] >= k).astype(float)
            cm = eh.cluster_mean(hit, g["event_id"].to_numpy()[ok])
            if cm is None:
                continue
            pred = float(np.mean(p[ok]))
            if pred < 0.01 or pred > 0.99:
                continue
            gap = cm["mean"] - pred
            z = gap / cm["se"] if cm["se"] > 0 else np.nan
            fo = fair_american(np.array([pred]))[0]
            print(f"  {k:>7}{int(ok.sum()):>7}{pred:>11.4f}{cm['mean']:>10.4f}"
                  f"{cm['se']:>8.4f}{100 * gap:>+10.1f}{z:>+7.2f}"
                  f"{fo:>+11.0f}")
    print("\n    A gap that stays near zero across the rungs means the whole")
    print("    distribution is trustworthy, not just its middle. A gap that")
    print("    GROWS with k means the tail is wrong, and the alt ladder would")
    print("    be betting on the part of the model nobody has validated.")


def by_distance_from_line(j, min_n=120):
    """The split that decides the alt idea: calibration BY DISTANCE from the
    posted line. The main line is d = 0. The alt ladder lives at d >= 2."""
    print("\n" + "=" * 100)
    print("PART A2: calibration by DISTANCE of the rung from the posted line")
    print("=" * 100)
    print("  d is rung minus posted line, rounded. d near 0 is the main line,")
    print("  which every earlier check already covered. d of 2 or more is")
    print("  where the alternate ladder actually sits.")
    for m, g in j.groupby("market"):
        print(f"\n  {m}")
        print(f"  {'d':>5}{'n':>8}{'predicted':>11}{'realized':>10}{'SE':>8}"
              f"{'gap (pp)':>10}{'z':>7}{'fair odds':>11}")
        rows = {}
        for k in range(1, MAXK + 1):
            p = sf_at(m, g["blend"].to_numpy(), g["sigma"].to_numpy(), k)
            d = np.round(k - g["line_consensus"].to_numpy()).astype(int)
            hit = (g["actual"].to_numpy() >= k).astype(float)
            ok = np.isfinite(p) & (p > 0.005) & (p < 0.995)
            for dd in np.unique(d[ok]):
                sel = ok & (d == dd)
                if not sel.any():
                    continue
                rows.setdefault(int(dd), [[], [], []])
                rows[int(dd)][0].extend(p[sel])
                rows[int(dd)][1].extend(hit[sel])
                rows[int(dd)][2].extend(g["event_id"].to_numpy()[sel])
        for dd in sorted(rows):
            pv, hv, ev = rows[dd]
            if len(pv) < min_n:
                continue
            cm = eh.cluster_mean(np.array(hv), np.array(ev))
            if cm is None:
                continue
            pred = float(np.mean(pv))
            gap = cm["mean"] - pred
            z = gap / cm["se"] if cm["se"] > 0 else np.nan
            fo = fair_american(np.array([pred]))[0]
            print(f"  {dd:>+5}{len(pv):>8}{pred:>11.4f}{cm['mean']:>10.4f}"
                  f"{cm['se']:>8.4f}{100 * gap:>+10.1f}{z:>+7.2f}"
                  f"{fo:>+11.0f}")
    print("\n    READ THE d = +2 AND d = +3 ROWS. Those are the alt-over rungs")
    print("    the idea targets. A positive gap there means the model")
    print("    UNDER-states the upside, so its fair odds are too long and an")
    print("    alt over would look cheap when it is not. A negative gap means")
    print("    the model over-states the upside, which is the direction that")
    print("    makes alt overs genuinely attractive.")


# ------------------------------------------------------------------- PART B

def export_ladder(j, season, week, path):
    print("\n" + "=" * 100
          + f"\nPART B: fair-odds ladder for {season} week {week}\n"
          + "=" * 100)
    g = j[(j["season"] == season) & (j["week"] == week)]
    if not len(g):
        print(f"  no rows for {season} week {week}. Available: "
              f"{sorted(j['season'].unique())}")
        return
    out = []
    for _, r in g.iterrows():
        row = {"player": r["player"], "market": r["market"],
               "posted_line": r["line_consensus"],
               "model_mean": round(float(r["blend"]), 2),
               "sigma": round(float(r["sigma"]), 2)}
        for k in range(1, MAXK + 1):
            p = sf_at(r["market"], np.array([r["blend"]]),
                      np.array([r["sigma"]]), k)[0]
            if not np.isfinite(p) or p < 0.02 or p > 0.98:
                continue
            row[f"p_over_{k - 0.5}"] = round(float(p), 4)
            row[f"fair_{k - 0.5}"] = int(round(fair_american(np.array([p]))[0]))
        out.append(row)
    df = pd.DataFrame(out).sort_values("model_mean", ascending=False)
    df.to_csv(path, index=False)
    print(f"  wrote {path}, {len(df)} players")
    print(f"\n  top 12 by model mean, fair odds at each rung:")
    cols = ["player", "posted_line", "model_mean"] + [
        c for c in df.columns if c.startswith("fair_")]
    print(df[cols].head(12).to_string(index=False))
    print("\n  HOW TO USE THIS WITHOUT SPENDING A CREDIT. Open FanDuel, find a")
    print("  player above, and read their alternate reception ladder. Compare")
    print("  each rung to the fair odds here. If FanDuel offers +250 where")
    print("  this says +180, their implied probability is well below the")
    print("  model's and that rung is a candidate. Do a dozen by hand before")
    print("  building any capture for it.")
    print("  And read PART A2 first. If the model is not calibrated two or")
    print("  three catches above the line, these fair odds are fiction.")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default="receptions")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--export-ladder", default=None,
                    help="season,week to export, e.g. 2026,2")
    ap.add_argument("--out", default="ladder.csv")
    args = ap.parse_args()
    if args.season and not args.seasons:
        args.seasons = str(args.season)

    print("=" * 100)
    print("LADDER CALIBRATION: is the distribution trustworthy away from "
          "the line?")
    print("=" * 100)

    j = load(args)
    params = et.fit_params(j)
    if not params:
        sys.exit("  could not fit shrinkage parameters")
    et.report_params(params)
    sig = et.fit_sigma_sqrt(j, params)
    print("\n  SIGMA = a * sqrt(mean), exponent imposed at 0.5")
    for k in sorted(sig):
        print(f"    {k[0]} {int(k[1])}: a = {sig[k][0]:.3f}")

    j = attach_distribution(j, params, sig)
    print(f"\n  {len(j)} props with a usable distribution")

    ladder_calibration(j)
    by_distance_from_line(j)

    if args.export_ladder:
        s, w = [int(x) for x in args.export_ladder.split(",")]
        export_ladder(j, s, w, args.out)


if __name__ == "__main__":
    main()
