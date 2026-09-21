"""
Quantify the two effects of the `carries >= 5` filter in rushing.build_dataset.

1. FEATURE INFLATION: carries_roll is currently a rolling mean over only the
   games where a back got 5+ carries. Games with 0-4 carries are deleted
   before the rolling mean is taken, so a backup's workload feature reflects
   only the weeks the starter was hurt or resting.

2. SILENT GRADING LOSS: actual_result reads the same filtered frame, so a
   player-week with fewer than 5 carries has no row and returns None. The bet
   never grades. Because the inflated projections sit on backups who often bust,
   the lost rows are disproportionately ones the model got WRONG.
   The position == "RB" filter also means no QB can ever grade.

Run from the repo root with the venv active:
    python rushing_filter_diag.py
"""
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

from models import data_utils

SEASONS = [2022, 2023, 2024, 2025, 2026]

# players flagged from the bet log as suspiciously over-projected
SUSPECTS = [
    "George Holani", "Raheim Sanders", "Justice Hill", "Samaje Perine",
    "Chris Rodriguez Jr.", "Chris Brooks", "Braelon Allen",
    "Jonathon Brooks", "Tyjae Spears", "Alvin Kamara",
]


def _roll(df, col="carries", window=6):
    """Lagged rolling mean, exactly as build_dataset does it."""
    return (df.groupby("player_id")[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()))


def main():
    print("=" * 74)
    print("RUSHING SURVIVORSHIP FILTER: how much does `carries >= 5` distort things?")
    print("=" * 74)

    ps = data_utils.load_player_stats(SEASONS)
    if hasattr(ps, "to_pandas"):
        ps = ps.to_pandas()

    rb_all = ps[ps["position"] == "RB"].copy()
    rb_all["carries"] = rb_all["carries"].fillna(0)
    rb_all = rb_all.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # honest feature: rolling mean over every active game
    rb_all["carries_roll_all"] = _roll(rb_all)

    # current behaviour: filter first, then roll
    rb_filt = rb_all[rb_all["carries"] >= 5].copy()
    rb_filt = rb_filt.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    rb_filt["carries_roll_filt"] = _roll(rb_filt)

    # ---------------------------------------------------------------- section 1
    print("\n" + "-" * 74)
    print("1. HOW MANY ROWS DOES THE FILTER DELETE?")
    print("-" * 74)
    for season in SEASONS:
        a = rb_all[rb_all["season"] == season]
        f = rb_filt[rb_filt["season"] == season]
        if len(a) == 0:
            continue
        pct = 100.0 * (1 - len(f) / len(a))
        print(f"  {season}: {len(a):>5} RB player-weeks, {len(f):>5} survive "
              f"({pct:>5.1f}% deleted)")

    qb_rows = ps[ps["position"] == "QB"]
    print(f"\n  QB player-weeks in player_stats: {len(qb_rows)}")
    print("  All of them are invisible to rushing.build_dataset (position == 'RB'),")
    print("  so actual_result returns None for every QB and no QB bet can grade.")

    # ---------------------------------------------------------------- section 2
    print("\n" + "-" * 74)
    print("2. FEATURE INFLATION ON THE FLAGGED PLAYERS (latest available week)")
    print("-" * 74)
    print(f"  {'player':<24}{'season':>7}{'wk':>4}{'roll_all':>10}"
          f"{'roll_filt':>11}{'inflation':>11}{'proj_delta':>12}")

    YARDS_PER_CARRY = 4.0  # rough LinearRegression coefficient on carries_roll
    rows = []
    for name in SUSPECTS:
        sub_a = rb_all[rb_all["player_display_name"] == name]
        if len(sub_a) == 0:
            print(f"  {name:<24}  not found in player_stats as an RB")
            continue
        last = sub_a.iloc[-1]
        ra = last["carries_roll_all"]
        sub_f = rb_filt[rb_filt["player_display_name"] == name]
        rf = sub_f.iloc[-1]["carries_roll_filt"] if len(sub_f) else np.nan
        infl = rf - ra if (pd.notna(rf) and pd.notna(ra)) else np.nan
        print(f"  {name:<24}{int(last['season']):>7}{int(last['week']):>4}"
              f"{ra:>10.2f}{rf:>11.2f}{infl:>11.2f}"
              f"{infl * YARDS_PER_CARRY:>12.1f}")
        rows.append((name, ra, rf, infl))

    valid = [r for r in rows if pd.notna(r[3])]
    if valid:
        mean_infl = np.mean([r[3] for r in valid])
        print(f"\n  mean inflation across flagged players: {mean_infl:+.2f} carries "
              f"(~{mean_infl * YARDS_PER_CARRY:+.1f} yards of projection)")

    # ---------------------------------------------------------------- section 3
    print("\n" + "-" * 74)
    print("3. POPULATION-WIDE INFLATION, BY HOW MUCH WORK A BACK ACTUALLY GETS")
    print("-" * 74)
    merged = rb_all.merge(
        rb_filt[["player_id", "season", "week", "carries_roll_filt"]],
        on=["player_id", "season", "week"], how="left")
    m = merged.dropna(subset=["carries_roll_all", "carries_roll_filt"]).copy()
    if len(m):
        m["inflation"] = m["carries_roll_filt"] - m["carries_roll_all"]
        m["bucket"] = pd.cut(m["carries_roll_all"], [-0.1, 2, 5, 10, 15, 100],
                             labels=["0-2", "2-5", "5-10", "10-15", "15+"])
        print(f"  {'roll_all':<10}{'n':>6}{'mean_infl':>12}{'proj_delta':>12}")
        for b, g in m.groupby("bucket", observed=True):
            print(f"  {str(b):<10}{len(g):>6}{g['inflation'].mean():>12.2f}"
                  f"{g['inflation'].mean() * YARDS_PER_CARRY:>12.1f}")
        print("\n  The inflation should be largest for the low-workload backs,")
        print("  which is exactly the population the betting strategy targets.")

    # ---------------------------------------------------------------- section 4
    print("\n" + "-" * 74)
    print("4. SILENT GRADING LOSS: bust games are the ones that vanish")
    print("-" * 74)
    recent = rb_all[rb_all["season"] >= 2025]
    if len(recent):
        lost = recent[recent["carries"] < 5]
        print(f"  2025+ RB player-weeks: {len(recent)}")
        print(f"  under 5 carries (ungradeable): {len(lost)} "
              f"({100.0 * len(lost) / len(recent):.1f}%)")
        if len(lost):
            print(f"  mean rushing yards in those lost rows: "
                  f"{lost['rushing_yards'].fillna(0).mean():.1f}")
        kept = recent[recent["carries"] >= 5]
        if len(kept):
            print(f"  mean rushing yards in the rows that DO grade: "
                  f"{kept['rushing_yards'].fillna(0).mean():.1f}")
        print("\n  A bet on a backup that busts produces a low actual, so the UNDER")
        print("  wins and the OVER loses. Those are the rows that silently fail to")
        print("  grade, so the measured rushing record is biased in the model's")
        print("  favour. The true record is worse than the -12.0 pp already seen.")

    print("\n" + "=" * 74)
    print("WHAT THIS IMPLIES")
    print("=" * 74)
    print("  If section 2 shows large positive inflation on the flagged players,")
    print("  the fake Max edges are explained by the feature, not by the pricing")
    print("  and not by a missing depth chart. The fix is to compute carries_roll")
    print("  over every active game and filter afterwards.")
    print("  If section 4 shows a large share of ungradeable rows, the rushing")
    print("  calibration numbers are measured on a biased subsample.")


if __name__ == "__main__":
    main()
