"""banded_under_test.py - does the low-line under effect exist outside rushing?

Reports only. Writes nothing, ships no constant, places no bet.

THE QUESTION

    Rule 1 bets the UNDER on rushing where line <= 46.5, ROI +0.0589. The
    mechanism we believe is distributional: alpha measured at zero deviation
    is +4.0067, so the MEAN sits about four yards above the line, while the
    under wins 55.86 percent, so the MEDIAN sits below it. Mean above,
    median below, is right skew straddling the line, and it is largest where
    the outcome has a hard floor at zero and a long right tail.

    If that is the mechanism it should not care what the stat is called. A
    WR4 and a backup running back are the same animal.

WHY THE EXISTING 28-OF-28 CONTROL DOES NOT ANSWER THIS

    Those 28 cells took the under in the non-rushing markets and all came
    back negative, which is what made Rule 1 credible. But they pooled over
    EVERY line value. Rushing's own effect dies above 46.5, with terciles
    falling +0.0571, +0.0076, -0.0273, so a pooled rushing cell would very
    likely have read negative too. 28 of 28 rules out a BROAD under bias. It
    does not rule out a BANDED one, and the banded test has never been run.

    Note also that mean-above-median is NECESSARY BUT NOT SUFFICIENT.
    receiving alpha is +3.4623, nearly rushing's +4.0067, yet receiving
    unders lose in all seven book cells. So the operative variable is where
    the BOOK PUTS THE LINE, not the shape of the distribution alone. This
    test is therefore genuinely open, not a formality.

THE GATE, WHICH RUNS FIRST

    rushing below 46.5 must reproduce a positive ROI near the published
    +0.0589 before any new cell is computed. A script reimplementing an
    existing procedure reproduces that procedure's published output before
    producing any new number. If the control does not come back, the
    population or the grading differs from the published work and every new
    number here is uninterpretable.

BEST-OF-N, AND WHY THE PLACEBO SHUFFLES THE LINE

    This is a threshold grid across several markets, which is exactly the
    setup that returned p = 0.160 on a signal known to be real before the
    placebo was built properly. So the reported p is a PERMUTATION p on the
    MAX statistic over the whole grid, not a raw t on the best cell.

    The permutation shuffles the LINE within market and season, keeping each
    row's outcome and odds attached to it. That breaks the line-to-outcome
    relationship, which is the null, while leaving the outcome vector
    completely intact so that game-level correlation survives into the null
    distribution. Shuffling OUTCOMES instead would destroy that clustering
    and give a null that is too narrow, which would flatter the result.

CELL FLOOR

    200 bets. Below that a cell is not reported and cannot be the best cell,
    in the observed data or in any placebo draw. Applying the floor to both
    sides is the point: a floor applied only to the observed data would let
    placebo draws win on tiny cells and understate the p.

Run from the repo root with the venv active:

    python banded_under_test.py
    python banded_under_test.py --placebo 500
    python banded_under_test.py --markets receptions,qb_rushing
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

CACHE = "lines_cache.parquet"
BOOK = "fanduel"
CELL_FLOOR = 200

# The control. Published figure from rushing_lowline.py.
CONTROL_MARKET = "rushing"
CONTROL_CUT = 46.5
CONTROL_ROI = 0.0589
CONTROL_TOL = 0.040

DEFAULT_MARKETS = ("rushing", "receptions", "receiving", "qb_rushing",
                   "qb_passing")


def american_to_decimal(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / (-o))


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def clustered_se(payout, cluster):
    """CR0 cluster-robust SE of the mean. Games are the cluster.

    Props in one game share weather, pace, game script and blowout status,
    so treating them as independent would overstate precision. This is the
    same discipline as the game-clustered errors used elsewhere.
    """
    x = np.asarray(payout, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    d = x - x.mean()
    s = pd.Series(d).groupby(np.asarray(cluster)).sum().to_numpy()
    var = (s ** 2).sum() / (n ** 2)
    return float(np.sqrt(var)) if var > 0 else np.nan


def load_market(market, cache):
    """FanDuel closing rows for one market, joined to the actual outcome."""
    import importlib
    from models import data_utils as du

    fetch = eh.FETCH_MARKET.get(market, market)
    C = pd.read_parquet(cache)
    L = C[C["market"] == fetch].copy()
    if market in eh.FETCH_MARKET:
        L, rep = du.split_qb_rushing(L, min_match_rate=0.98)
        L = L[L["market"] == market].copy()
        print("  split: %d of %d reassigned, match rate %.4f"
              % (rep["reassigned"], rep["rows_in_from_market"],
                 rep["match_rate"]))

    L = L[L["book"] == BOOK].copy()
    L = L[L["line"].notna() & L["under_odds"].notna()
          & L["week"].notna()].copy()
    if L.empty:
        return None

    # One row per player-week: the LAST snapshot captured before kickoff.
    # commence_time is populated on every cached row, so this is a direct
    # comparison rather than a reconstruction.
    if "commence_time" in L.columns:
        ct = pd.to_datetime(L["commence_time"], utc=True, errors="coerce")
        cap = pd.to_datetime(L["captured_at"], utc=True, errors="coerce")
        pre = cap < ct
        print("  dropped %d rows captured at or after kickoff"
              % int((~pre).sum()))
        L = L[pre].copy()
        L["_cap"] = cap[pre].to_numpy()
    else:
        L["_cap"] = pd.to_datetime(L["captured_at"], utc=True,
                                   errors="coerce")

    L["_key"] = du.norm_join_name(L["player"])
    L["week"] = L["week"].astype(int)
    L = (L.sort_values("_cap")
          .drop_duplicates(subset=["season", "week", "_key"], keep="last"))

    mod_name, actual_col = eh.MARKET_SPEC[market]
    mod = importlib.import_module(f"models.{mod_name}")
    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    full = builder()
    if actual_col not in full.columns:
        return None
    keep = [actual_col, "season", "week", "player_display_name"]
    if "game_id" in full.columns:
        keep.append("game_id")
    full = full[[c for c in keep if c in full.columns]].dropna(
        subset=[actual_col]).copy()
    full["_key"] = du.norm_join_name(full["player_display_name"])
    full["week"] = full["week"].astype(int)

    j = L.merge(full, on=["season", "week", "_key"], how="inner")
    print("  %d FanDuel closing rows joined to an outcome" % len(j))

    # Pushes are not bets. Excluded and counted, never graded as losses.
    push = j[actual_col] == j["line"]
    if push.any():
        print("  %d pushes excluded" % int(push.sum()))
    j = j[~push].copy()

    j["under_won"] = (j[actual_col] < j["line"]).astype(int)
    dec = american_to_decimal(j["under_odds"])
    j["payout"] = np.where(j["under_won"] == 1, dec - 1.0, -1.0)
    j["_game"] = (j["game_id"] if "game_id" in j.columns
                  else j["season"].astype(str) + "_" + j["week"].astype(str))
    return j


def grid_for(j):
    """Cumulative cutoffs at line deciles, coarsest cell first.

    Deciles rather than a fixed ladder because the markets differ by an
    order of magnitude in scale: receptions lines sit near 3, qb_passing
    near 230. A shared numeric grid would be meaningless.
    """
    qs = np.unique(np.round(j["line"].quantile(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to_numpy(), 2))
    return [c for c in qs if (j["line"] <= c).sum() >= CELL_FLOOR]


def cells(j, cuts):
    out = []
    for c in cuts:
        s = j[j["line"] <= c]
        if len(s) < CELL_FLOOR:
            continue
        roi = float(s["payout"].mean())
        se = clustered_se(s["payout"], s["_game"])
        out.append({"cut": c, "n": len(s),
                    "games": int(s["_game"].nunique()),
                    "under_rate": float(s["under_won"].mean()),
                    "roi": roi, "se": se,
                    "t": roi / se if se and np.isfinite(se) else np.nan})
    return out


def best_roi(j, cuts):
    """Max cell ROI over the grid, floor applied. None if no cell qualifies."""
    best = None
    for c in cuts:
        s = j[j["line"] <= c]
        if len(s) < CELL_FLOOR:
            continue
        r = float(s["payout"].mean())
        if best is None or r > best:
            best = r
    return best


def placebo(j, cuts, draws, rng):
    """Shuffle the LINE within market and season; outcomes stay put.

    This is the null that the line carries no information about which side
    wins. Because the outcome and odds never move, every game-level
    correlation in the outcome vector is preserved in each draw, which a
    shuffle of the outcome would destroy.
    """
    got = []
    base = j.copy()
    for _ in range(draws):
        shuffled = (base.groupby("season")["line"]
                    .transform(lambda s: rng.permutation(s.to_numpy())))
        d = base.assign(line=shuffled)
        b = best_roi(d, cuts)
        if b is not None:
            got.append(b)
    return np.asarray(got, dtype=float)


def run_market(market, cache, draws, rng):
    hr(market.upper())
    try:
        j = load_market(market, cache)
    except Exception as e:
        print("  could not load: %s: %s" % (type(e).__name__, e))
        return None
    if j is None or len(j) < CELL_FLOOR:
        print("  too few rows")
        return None

    print("  line range %.1f to %.1f, seasons %s"
          % (j["line"].min(), j["line"].max(),
             sorted(j["season"].unique().tolist())))
    print("  overall under rate %.4f on %d bets, ROI %+.4f"
          % (j["under_won"].mean(), len(j), j["payout"].mean()))

    cuts = grid_for(j)
    if not cuts:
        print("  no cell reaches the %d-bet floor" % CELL_FLOOR)
        return None

    rows = cells(j, cuts)
    print()
    print("  CUMULATIVE UNDER, line <= cut")
    print("    %8s %6s %6s %10s %9s %8s %7s"
          % ("cut", "n", "games", "under rate", "ROI", "clus SE", "t"))
    for r in rows:
        print("    %8.2f %6d %6d %10.4f %+9.4f %8.4f %+7.2f"
              % (r["cut"], r["n"], r["games"], r["under_rate"], r["roi"],
                 r["se"], r["t"]))

    obs = max(r["roi"] for r in rows)
    bestrow = [r for r in rows if r["roi"] == obs][0]
    print()
    print("  best cell: line <= %.2f, %d bets, ROI %+.4f, clustered t %+.2f"
          % (bestrow["cut"], bestrow["n"], obs, bestrow["t"]))
    print("  MDE at that cell: %+.4f" % (2.80 * bestrow["se"]))

    if draws:
        null = placebo(j, cuts, draws, rng)
        if len(null) == 0:
            print("  placebo produced no qualifying draws")
            return None
        p = float((null >= obs).sum() + 1) / (len(null) + 1)
        print()
        print("  PLACEBO on the MAX cell, line shuffled within season, "
              "%d draws" % len(null))
        print("    null mean %+.4f  sd %.4f  95th pct %+.4f  max %+.4f"
              % (null.mean(), null.std(ddof=1),
                 np.percentile(null, 95), null.max()))
        print("    observed %+.4f   permutation p = %.4f" % (obs, p))
        if p <= 0.05:
            print("    SURVIVES the best-of-grid correction.")
        else:
            print("    does NOT survive. The best cell is within what this")
            print("    grid produces by chance.")
        return {"market": market, "cut": bestrow["cut"], "n": bestrow["n"],
                "roi": obs, "t": bestrow["t"], "p": p}
    return {"market": market, "cut": bestrow["cut"], "n": bestrow["n"],
            "roi": obs, "t": bestrow["t"], "p": np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--placebo", type=int, default=200)
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    ap.add_argument("--skip-gate", action="store_true",
                    help="run without the rushing control. Results are not "
                         "comparable to the published rule.")
    args = ap.parse_args()
    rng = np.random.default_rng(20260928)

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in eh.MARKET_SPEC]
    if bad:
        print("unknown markets: %s" % bad)
        sys.exit(1)

    hr("GATE: reproduce the published rushing low-line under")
    if args.skip_gate:
        print("  SKIPPED by flag. Nothing below is comparable to the")
        print("  published rule and should not be treated as such.")
    else:
        try:
            g = load_market(CONTROL_MARKET, args.cache)
        except Exception as e:
            print("  could not load %s: %s: %s"
                  % (CONTROL_MARKET, type(e).__name__, e))
            sys.exit(1)
        if g is None:
            print("  no rushing rows. Nothing written.")
            sys.exit(1)
        s = g[g["line"] <= CONTROL_CUT]
        roi = float(s["payout"].mean())
        se = clustered_se(s["payout"], s["_game"])
        print("  rushing, line <= %.1f, FanDuel, %d bets across %d games"
              % (CONTROL_CUT, len(s), s["_game"].nunique()))
        print("  ROI %+.4f  clustered SE %.4f  t %+.2f"
              % (roi, se, roi / se if se else float("nan")))
        print("  published %+.4f, tolerance %.3f" % (CONTROL_ROI, CONTROL_TOL))
        if abs(roi - CONTROL_ROI) > CONTROL_TOL:
            print()
            print("  GATE FAILED. This script does not reproduce the")
            print("  published control, so its population or its grading")
            print("  differs from the work Rule 1 rests on and every new")
            print("  number below would be uninterpretable. Nothing else run.")
            print("  Check: book filter, snapshot selection, push handling,")
            print("  and whether the published figure used these seasons.")
            sys.exit(1)
        print("  GATE PASSED")

    found = []
    for m in markets:
        r = run_market(m, args.cache, args.placebo, rng)
        if r:
            found.append(r)

    hr("SUMMARY")
    print("  %-12s %8s %7s %9s %7s %8s" % ("market", "cut", "n", "ROI", "t",
                                           "perm p"))
    for r in found:
        print("  %-12s %8.2f %7d %+9.4f %+7.2f %8.4f"
              % (r["market"], r["cut"], r["n"], r["roi"], r["t"], r["p"]))
    print()
    survivors = [r for r in found
                 if np.isfinite(r["p"]) and r["p"] <= 0.05
                 and r["market"] != CONTROL_MARKET]
    if survivors:
        print("  Surviving NON-control cells: %s"
              % ", ".join(r["market"] for r in survivors))
        print("  These are candidates, not rules. Next steps before any of")
        print("  them ships: a pre-registered holdout chosen BEFORE the")
        print("  threshold, a per-season breakdown, and a cross-book check,")
        print("  because Rule 1's credibility came from appearing in six of")
        print("  seven books rather than from its pooled ROI.")
    else:
        print("  No non-control market survives the best-of-grid correction.")
        print("  Read that with the MDE lines above: a null on a cell whose")
        print("  MDE exceeds any plausible effect is an underpowered test,")
        print("  not an absence.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
