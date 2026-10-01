"""diag_teammate_corr.py - how do teammates' outcomes actually co-move?

READ ONLY. Descriptive, no pre-registration, no decision rule. The point is
to find out whether there is anything here before any hypothesis is written.

WHY THIS IS NOT AN SGP TEST, and the distinction matters.

    FanDuel does not price SGP legs independently. It applies a correlation
    model: positively correlated legs are marked UP, negatively correlated
    ones marked down or blocked. So "find correlated legs" is never the
    edge, because the book already knows. The only edge would be pairs
    where the book's correlation ASSUMPTION is wrong, and testing that
    needs SGP quotes.

    We have none. The Odds API captures individual props, so nothing in the
    184,782 historical rows or the 150,000 live captures is an SGP price.
    Any SGP claim is therefore unbacktestable with current data, which is a
    hard constraint rather than a difficulty.

    Also: a two-leg SGP typically holds 6-10 percent against 4.9 percent on
    a straight FanDuel rushing prop. Rule 1's measured 3-point gap over
    breakeven would be consumed by that margin before any correlation
    argument started.

WHAT IS TESTABLE, AND WHY IT BEARS ON RULE 1 RATHER THAN ON SGPs

    Rule 1 bets the under on every rushing line at or below 46.5. On a
    Sunday that can fire on BOTH backs in one backfield. If two backs'
    outcomes are strongly negatively correlated, because carries are a
    fixed pie, then those two unders are not two independent bets: one
    back's under failing makes the other's more likely to succeed.

    That changes STAKING and variance, not expected value. Negative
    correlation between two positive-EV bets is good for a portfolio: it
    reduces the variance of the pair. Positive correlation is the harmful
    case, because it concentrates risk.

    So the practical question is: when Rule 1 fires twice in one backfield,
    are those two bets diversifying or doubling down?

METHOD, AND THE CONFOUND IT AVOIDS

    Raw correlation between two teammates' yardage across games is
    confounded: a good offence lifts both, so two backs can look positively
    correlated purely because their team is sometimes dominant.

    So this correlates DEVIATIONS from each player's own expectation,
    using his own prior-games rolling mean with the current game excluded.
    That asks the right question: when A is above his usual, is B below
    his usual?

Run:  python diag_teammate_corr.py
"""
import itertools
import sys

import numpy as np
import pandas as pd

from models import data_utils

SEASONS = [2023, 2024, 2025, 2026]
MIN_SHARED_GAMES = 8      # below this a pair correlation is noise
HALFLIFE = 6.0            # for the player's own expectation


def hr(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


def load():
    s = data_utils.load_player_stats(SEASONS)
    s = s.to_pandas() if hasattr(s, "to_pandas") else s
    keep = ["player_id", "player_display_name", "position", "team",
            "season", "week", "carries", "rushing_yards", "targets",
            "receptions", "receiving_yards", "attempts", "passing_yards"]
    keep = [c for c in keep if c in s.columns]
    return s[keep].copy()


def add_deviation(df, col):
    """Deviation from the player's own prior-games EWMA.

    shift(1) so the current game is excluded from its own expectation. A
    raw correlation between teammates is confounded by team quality: a
    dominant offence lifts both players, which looks positive regardless
    of how they split the work.
    """
    g = df.sort_values(["player_id", "season", "week"]).groupby("player_id")[col]
    exp = g.transform(lambda x: x.shift(1).ewm(halflife=HALFLIFE,
                                               min_periods=3).mean())
    return df[col] - exp


def pair_corrs(df, col, positions, label, min_mean=0.0):
    """Correlation of deviations for every same-team-game pair."""
    d = df[df["position"].isin(positions)].copy()
    d["dev"] = add_deviation(d, col)
    d = d[d["dev"].notna()]
    if d.empty:
        print("  no rows for %s" % label)
        return pd.DataFrame()

    # a player must be a real contributor, or the pair is noise
    means = d.groupby("player_id")[col].mean()
    good = set(means[means >= min_mean].index)
    d = d[d["player_id"].isin(good)]

    rows = []
    for (season, team), g in d.groupby(["season", "team"]):
        ids = sorted(g["player_id"].unique())
        for a, b in itertools.combinations(ids, 2):
            ga = g[g["player_id"] == a].set_index("week")["dev"]
            gb = g[g["player_id"] == b].set_index("week")["dev"]
            common = ga.index.intersection(gb.index)
            if len(common) < MIN_SHARED_GAMES:
                continue
            x, y = ga.loc[common].to_numpy(), gb.loc[common].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            na = g[g["player_id"] == a]["player_display_name"].iloc[0]
            nb = g[g["player_id"] == b]["player_display_name"].iloc[0]
            rows.append({"season": season, "team": team, "a": na, "b": nb,
                         "n": len(common), "r": r,
                         "mean_a": float(g[g["player_id"] == a][col].mean()),
                         "mean_b": float(g[g["player_id"] == b][col].mean())})
    out = pd.DataFrame(rows)
    print()
    print("  %s: %d pairs with %d+ shared games"
          % (label, len(out), MIN_SHARED_GAMES))
    if len(out) == 0:
        return out
    print("    mean r   %+.4f" % out["r"].mean())
    print("    median r %+.4f" % out["r"].median())
    print("    share negative %.1f%%" % (100.0 * (out["r"] < 0).mean()))
    print("    quartiles %+.3f / %+.3f / %+.3f"
          % (out["r"].quantile(0.25), out["r"].median(),
             out["r"].quantile(0.75)))
    # a one-sample t on the pair correlations, which is approximate because
    # pairs from one team-season are not independent of each other
    if len(out) > 2:
        t = out["r"].mean() / (out["r"].std() / np.sqrt(len(out)))
        print("    approximate t vs 0: %+.2f\n"
              "    (pairs within one team-season are not independent, so\n"
              "     treat this as indicative rather than a test)" % t)
    return out


def main():
    hr("LOADING")
    df = load()
    print("  %d player-game rows, seasons %s" % (len(df), SEASONS))

    hr("1. TWO BACKS IN ONE BACKFIELD: rushing yards")
    print("  Mechanically this should be NEGATIVE: carries are a fixed pie.")
    print("  Correlating DEVIATIONS from each back's own prior-game EWMA,")
    print("  so a dominant offence lifting both does not read as positive.")
    rb = pair_corrs(df, "rushing_yards", ["RB"], "RB-RB rushing yards",
                    min_mean=20.0)
    if len(rb):
        print()
        print("  most negative pairs:")
        for _, r in rb.nsmallest(8, "r").iterrows():
            print("    %s %s  %-22s %-22s n=%2d  r %+.3f  (%.0f / %.0f ypg)"
                  % (r["season"], r["team"], r["a"][:22], r["b"][:22],
                     r["n"], r["r"], r["mean_a"], r["mean_b"]))
        print()
        print("  most positive pairs:")
        for _, r in rb.nlargest(5, "r").iterrows():
            print("    %s %s  %-22s %-22s n=%2d  r %+.3f"
                  % (r["season"], r["team"], r["a"][:22], r["b"][:22],
                     r["n"], r["r"]))

    hr("2. TWO RECEIVERS ON ONE TEAM: receiving yards")
    print("  AMBIGUOUS a priori. Target competition pushes negative, team")
    print("  passing volume pushes positive, and which dominates is an")
    print("  empirical question nobody in this project has answered.")
    wr = pair_corrs(df, "receiving_yards", ["WR", "TE"],
                    "WR/TE pairs receiving yards", min_mean=25.0)

    hr("3. CARRIES RATHER THAN YARDS")
    print("  Carries are the actual fixed resource; yards add efficiency")
    print("  noise on top. If the pie story is right, the carry")
    print("  correlation should be MORE negative than the yardage one.")
    car = pair_corrs(df, "carries", ["RB"], "RB-RB carries", min_mean=5.0)
    if len(car) and len(rb):
        print()
        print("  carries mean r %+.4f against yards mean r %+.4f"
              % (car["r"].mean(), rb["r"].mean()))
        # CORRECTED after the first run. This originally printed "carries
        # ARE more negative, consistent with the fixed pie" whenever the
        # carry mean was merely LOWER than the yardage mean. On the real
        # data that fired at +0.0060 against +0.1010: arithmetically true
        # and substantively backwards, since neither figure is negative.
        # A fixed pie requires a NEGATIVE correlation, not a smaller
        # positive one.
        if car["r"].mean() >= 0:
            print("  carries are NOT negative, so there is no fixed pie to")
            print("  find. Game script is the likely reason: a team that is")
            print("  winning runs more, so total carries EXPAND and both")
            print("  backs gain. That volume channel cancels the split.")
        elif car["r"].mean() < rb["r"].mean():
            print("  carries are negative AND more negative than yards,")
            print("  which is what a fixed pie looks like.")
        else:
            print("  carries are negative but not more so than yards,")
            print("  which is against the fixed-pie story.")

    hr("4. WHAT THIS MEANS FOR RULE 1 STAKING")
    if len(rb):
        share_neg = 100.0 * (rb["r"] < 0).mean()
        print("  %.0f%% of RB-RB pairs are negatively correlated, mean r %+.3f."
              % (share_neg, rb["r"].mean()))
        if rb["r"].mean() > 0:
            print()
            print("  MEASURED POSITIVE, which is the unhelpful direction.")
            print("  Two backs' unders are slightly positively correlated,")
            print("  so betting both CONCENTRATES risk rather than")
            print("  diversifying it. Small at this size, but it argues for")
            print("  treating a backfield pair as closer to one bet than")
            print("  two when sizing.")
            print()
            print("  And it closes the constraint-violation idea: that")
            print("  needed carries to be a fixed pie, so that two high")
            print("  lines on one backfield would be jointly implausible.")
            print("  They are not, so there is no violation to find.")
        print()
        print("  Rule 1 fires on EVERY rushing line at or below 46.5, so on")
        print("  a given Sunday it can fire on both backs in one backfield.")
        print()
        print("  Negative correlation between two POSITIVE-EV bets is")
        print("  GOOD for a portfolio: it lowers the variance of the pair")
        print("  without touching the expected value. Positive correlation")
        print("  is the harmful case, because it concentrates risk.")
        print()
        print("  So this says nothing about whether to TAKE both bets. It")
        print("  says something about how much the pair swings, which is a")
        print("  staking question rather than a selection one.")
    print()
    print("  NOTHING WAS WRITTEN. No rule was changed.")
    print()


if __name__ == "__main__":
    main()
