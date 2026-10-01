"""player_streak_test.py - does a PLAYER's prop streak carry information?

READ ONLY. Writes nothing, changes no constant.

    python player_streak_test.py --cache lines_cache.parquet

THE AXIS FLIP. Every sequence test so far asked whether a DEFENSIVE unit
carries a persistent exploitable trait. All of them were null:

    unconditional defensive streaks   best +0.1206, null max +0.1176
    conditioned on defence quality    best +0.1249, null max +0.1591
    conditioned on bye weeks          best +0.1626, null max +0.1549
    recent change in allowance        line does not respond AND the
                                      residual does not either

This flips to the player. A receiver who has cleared his line four weeks
running is a different object from a defence that has conceded four.

WHY THIS IS NOT recency_test.py. That script measured FanDuel's weight on
a player's LAST GAME against the weight that actually predicts his next
one. They matched in all five markets, every interval covering zero, zero
of 40 specifications clearing. So the line handles a player's recent FORM
correctly.

    A STREAK is a different quantity from a weighted average of form, and
    the proposed mechanism is different: not "he has been productive" but
    "his usage has shifted", a target share or role change the line may
    lag. A four-game streak of clearing can happen at a flat production
    level if the LINE has not kept up, which is precisely the case a
    form-weighting test cannot see.

    The prior is still poor. recency_test is strong evidence that this
    market handles player recency well.

WHAT A PLAYER-GAME STREAK IS. Per player, per market, in a season: did he
clear his line. Consecutive clears build an over-streak, consecutive
failures an under-streak, and a push leaves both untouched. The streak
entering a game is computed from PRIOR games only.

    Same season only. A player's role resets across an offseason, and the
    whole point is to capture a role change rather than a career tendency.

THE FAMILY, enumerated before looking
    streak length   2, 3, 4, 5+
    direction       consecutive clears, consecutive failures
    market          receiving, receptions, rushing, qb_passing
    32 templates.

PRE-REGISTERED
    P1. NULL. The max-statistic permutation over 32 templates does not
        clear 0.05. Reason: recency_test already established the line
        handles player recency correctly, and five defensive-sequence
        tests today have all been null with the null maximum rising as
        cells narrow.

    P2. The null maximum will land between +0.09 and +0.16. Player cells
        are smaller than defence cells, since one player generates one
        prop per game while a defence faces several.

    P3. NO monotone gradient in streak length, in either direction.

    P4. If any cell looks strong it will be a long streak on a thin
        sample, and the permutation will dismiss it. Stated in advance so
        a 60-bet cell at +0.20 is read as noise.

    ALIVE only if the family maximum clears 0.05, the holdout keeps the
    sign, and the template has 150+ props in both periods.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
LENGTHS = [2, 3, 4, 5]     # 5 means "5 or more"
N_PERM = 2000
MIN_CELL = 60
DEV = [2023, 2024]
HOLD = [2025, 2026]


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def profit_if_win(odds):
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / np.abs(o))


def roi_of(win, profit):
    return float(np.where(win, profit, -1.0).mean()) if len(win) else np.nan


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy() - (len(x) + 1) / 2
    ry = pd.Series(y).rank().to_numpy() - (len(y) + 1) / 2
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args()

    hr("DO PLAYER PROP STREAKS CARRY INFORMATION?")
    print("  P1 predicts NULL: recency_test already found the line handles")
    print("     player recency correctly, 0 of 40 specifications clearing")
    print("  P2 predicts the null maximum lands between +0.09 and +0.16")
    print("  P3 predicts no monotone gradient in streak length")
    print("  P4 predicts any strong-looking cell is a long streak on a")
    print("     thin sample. Stated now so a 60-bet cell at +0.20 is read")
    print("     as noise.")

    hr("LOADING")
    C = pd.read_parquet(args.cache)
    C = C[(C["book"] == BOOK) & (C["market"].isin(MARKETS))].copy()
    if "snapshot_label" in C.columns:
        C = C[C["snapshot_label"] == "closing"]
    from models import data_utils
    seasons = sorted(C["season"].dropna().unique().astype(int).tolist())
    stats = data_utils.load_player_stats(seasons)
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    stats["_k"] = data_utils.norm_join_name(stats["player_display_name"])
    C["_k"] = data_utils.norm_join_name(C["player"])
    frames = []
    for mk in MARKETS:
        col = STAT_COL[mk]
        s = stats[["_k", "season", "week", col]].copy()
        g = C[C["market"] == mk].merge(s, on=["_k", "season", "week"],
                                       how="inner")
        frames.append(g.rename(columns={col: "actual"}))
    D = pd.concat(frames, ignore_index=True)
    D = D[D["actual"].notna() & D["line"].notna()
          & D["under_odds"].notna() & D["over_odds"].notna()].copy()
    D["over"] = (D["actual"] > D["line"]).astype(int)
    D["under"] = (D["actual"] < D["line"]).astype(int)
    print("  %d props with an outcome" % len(D))

    hr("1. BUILDING PLAYER STREAKS")
    print("  Per player, per market, within a season. A push leaves both")
    print("  runs untouched. The streak entering a game uses PRIOR games")
    print("  only, so no row ever sees its own result.")
    D = D.sort_values(["_k", "market", "season", "week"]).reset_index(
        drop=True)
    D["run_over"] = 0
    D["run_under"] = 0
    for _, idx in D.groupby(["_k", "market", "season"]).groups.items():
        ro = ru = 0
        for i in idx:
            D.at[i, "run_over"] = ro
            D.at[i, "run_under"] = ru
            if D.at[i, "over"] == 1:
                ro, ru = ro + 1, 0
            elif D.at[i, "under"] == 1:
                ro, ru = 0, ru + 1
    print()
    print("  clear-streak distribution entering a prop:")
    print("    %s" % D["run_over"].value_counts().sort_index().head(9)
          .to_dict())
    print("  failure-streak distribution:")
    print("    %s" % D["run_under"].value_counts().sort_index().head(9)
          .to_dict())
    print()
    print("  If these halve cleanly, the sequences are coin flips before")
    print("  any ROI is computed. That is how the defensive version came")
    print("  out: 2432 / 968 / 425 / 194 / 92 / 39 / 14 / 3.")

    win = D["under"].to_numpy().astype(bool)
    profit = profit_if_win(D["under_odds"].to_numpy())
    print()
    print("  pooled under ROI %+.4f on %d bets" % (roi_of(win, profit),
                                                   len(D)))

    hr("2. THE TEMPLATE FAMILY")
    templates = []
    for mk in MARKETS:
        for L in LENGTHS:
            for direction in ("clears", "fails"):
                col = "run_over" if direction == "clears" else "run_under"
                run = (D[col] >= L) if L == max(LENGTHS) else (D[col] == L)
                lab = "%d+" % L if L == max(LENGTHS) else "%d" % L
                m = ((D["market"] == mk) & run).to_numpy()
                templates.append((mk, direction, lab, L, m))
    print("  %d templates: %d markets x %d lengths x 2 directions"
          % (len(templates), len(MARKETS), len(LENGTHS)))

    def table(w):
        return np.array([roi_of(w[m], profit[m]) if m.sum() >= MIN_CELL
                         else np.nan for *_, m in templates])

    obs = table(win)
    hr("3. THE TABLE  (uncorrected, so not evidence)")
    print("  'after N straight CLEARS, bet the UNDER' is the fade; a")
    print("  positive number there means the streak reverted.")
    print()
    print("  %-12s %-8s %-5s %10s %7s %9s"
          % ("market", "after", "len", "under ROI", "props", "win rate"))
    print("  " + "-" * 58)
    for i, (mk, d, lab, L, m) in enumerate(templates):
        n = int(m.sum())
        if n == 0:
            continue
        rs = ("%+10.4f" % obs[i]) if np.isfinite(obs[i]) else "         -"
        print("  %-12s %-8s %-5s %s %7d %9.4f%s"
              % (mk, d, lab, rs, n, float(win[m].mean()),
                 "" if n >= MIN_CELL else "  thin"))

    usable = np.isfinite(obs)
    if not usable.any():
        print("  nothing clears the floor")
        sys.exit(0)
    omax = float(np.nanmax(obs[usable]))
    imax = int(np.nanargmax(np.where(usable, obs, -np.inf)))
    print()
    print("  %d of %d clear the %d-prop floor"
          % (int(usable.sum()), len(templates), MIN_CELL))
    print("  best: %s after %s %s, ROI %+.4f on %d props"
          % (templates[imax][0], templates[imax][2], templates[imax][1],
             omax, int(templates[imax][4].sum())))

    hr("4. MAX-STATISTIC PERMUTATION")
    print("  %d permutations, win/loss shuffled globally with every prop"
          % args.perm)
    print("  keeping its own template memberships and its own price")
    rng = np.random.default_rng(0)
    maxes = np.empty(args.perm)
    for b in range(args.perm):
        t = table(rng.permutation(win))
        u = np.isfinite(t)
        maxes[b] = np.nanmax(t[u])
        if (b + 1) % 500 == 0:
            print("\r    %d of %d" % (b + 1, args.perm), end="", flush=True)
    print()
    p = float(np.mean(maxes >= omax))
    print()
    print("  null maximum: median %+.4f  95th %+.4f  sd %.4f"
          % (np.median(maxes), np.quantile(maxes, 0.95), maxes.std()))
    print("  P2 range was +0.09 to +0.16: %s"
          % ("inside" if 0.09 <= np.median(maxes) <= 0.16 else "OUTSIDE"))
    print("  observed %+.4f -> p = %.4f   %s"
          % (omax, p, "CLEARS" if p < 0.05 else "does not clear"))

    hr("5. P3: MONOTONICITY")
    print("  Any single template is a best-of-%d. A gradient across"
          % int(usable.sum()))
    print("  lengths pools their evidence into one statistic.")
    print()
    for mk in MARKETS:
        for d in ("clears", "fails"):
            rs, ls = [], []
            for i, (m2, d2, lab, L, m) in enumerate(templates):
                if m2 == mk and d2 == d and np.isfinite(obs[i]):
                    rs.append(obs[i])
                    ls.append(L)
            if len(rs) < 3:
                print("  %-12s after %-8s too few lengths (%d)"
                      % (mk, d, len(rs)))
                continue
            print("  %-12s after %-8s rho %+.4f over %d lengths"
                  % (mk, d, spearman(ls, rs), len(rs)))

    hr("6. HOLDOUT ON THE BEST TEMPLATE")
    mk, d, lab, L, m = templates[imax]
    sub = D[m]
    for label, ss in (("dev %s" % DEV, DEV), ("holdout %s" % HOLD, HOLD)):
        s2 = sub[sub["season"].isin(ss)]
        if len(s2) == 0:
            print("  %-18s no rows" % label)
            continue
        w = s2["under"].to_numpy().astype(bool)
        pr = profit_if_win(s2["under_odds"].to_numpy())
        print("  %-18s ROI %+.4f on %d props  %s"
              % (label, roi_of(w, pr), len(s2),
                 "" if len(s2) >= 150 else "<- under the 150 floor"))

    hr("VERDICT")
    if p >= 0.05:
        print("  NULL. Best of %d templates is %+.4f; chance's best on the"
              % (int(usable.sum()), omax))
        print("  same data is %+.4f." % np.median(maxes))
        print()
        print("  P1 predicted this. With the defensive axis already null")
        print("  four ways, the player axis null closes sequence testing")
        print("  on this data. The null-maximum schedule is the thing to")
        print("  keep: it measures how fast the bar rises as cells narrow.")
    else:
        print("  CLEARS at 0.05: %s after %s %s, ROI %+.4f."
              % (mk, lab, d, omax))
        print("  Check section 6 and the prop count before treating this")
        print("  as a candidate. P1 predicted failure, so this would be a")
        print("  result that beat a pre-registered expectation.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
