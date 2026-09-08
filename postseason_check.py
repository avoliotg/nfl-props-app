"""
OpalScales - do postseason games distort the week 1 bridge?

THE OBSERVATION
    TreVeyon Henderson projects 1.6 receiving yards against a FanDuel line of
    8.5. His targets_roll is 1.0, built from 2025 weeks 17-22 with targets of
    0, 0, 1, 2, 0, 3. Week numbers past 18 are PLAYOFF games. In week 15 his
    targets_roll was 3.33, so his regular-season usage was roughly triple what
    the bridge is using.

THE QUESTION
    build_upcoming_week bridges features from the prior season's tail. For a
    team that went deep in January, that tail is postseason football:
    different opponents, different game scripts, often different rotations.
    Every player on a deep-run team could be carrying a distorted week 1
    projection.

WHAT THIS MEASURES
  1. How many players' bridge tails include postseason games, and how many
     teams are affected.
  2. For those players, regular-season-only rolling features vs the
     as-built tail, and what that does to the projection.
  3. Whether postseason usage is SYSTEMATICALLY different or just noisier.
     If it is only noisier, excluding it costs sample size for no gain.
  4. A concrete before/after table for the affected 2026 week 1 players.

WHAT IT DOES NOT DO
    Change anything. This is a read-only diagnostic. Excluding the postseason
    would alter every projection for deep-run teams, so it needs evidence
    first.

HOW TO RUN
    python postseason_check.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

REG_MAX = {2022: 18, 2023: 18, 2024: 18, 2025: 18}   # last regular-season week
ROLL_N = 6


def load():
    from models import receiving as r
    d = r.build_dataset()
    d = d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    d["is_post"] = [w > REG_MAX.get(int(s), 18) for s, w in
                    zip(d["season"], d["week"])]
    return d, r


def stage1(d):
    print("\n" + "=" * 74)
    print("STAGE 1 - how much postseason is in the data?")
    print("=" * 74)
    print(f"  total rows {len(d):,} | postseason rows {int(d['is_post'].sum()):,} "
          f"({100 * d['is_post'].mean():.1f}%)")
    for s in sorted(d["season"].unique()):
        sub = d[d["season"] == s]
        post = sub[sub["is_post"]]
        print(f"    {int(s)}: weeks {sub['week'].min()}-{sub['week'].max()} | "
              f"postseason rows {len(post):,} | "
              f"teams {post['team'].nunique()} | players {post['player_id'].nunique()}")

    # who has postseason games in their LAST 6 appearances of a season?
    print("\n  players whose final-6 tail includes postseason, by season:")
    for s in sorted(d["season"].unique()):
        sub = d[d["season"] == s]
        tail = sub.groupby("player_id").tail(ROLL_N)
        aff = tail.groupby("player_id")["is_post"].any()
        n_aff = int(aff.sum())
        print(f"    {int(s)}: {n_aff} of {aff.size} players "
              f"({100 * n_aff / max(aff.size, 1):.0f}%)")
        if int(s) == 2025:
            teams = (tail[tail["is_post"]].groupby("team")["player_id"]
                     .nunique().sort_values(ascending=False))
            print(f"      2025 affected teams: "
                  + ", ".join(f"{t}({n})" for t, n in teams.items()))


def stage2(d):
    print("\n" + "=" * 74)
    print("STAGE 2 - is postseason usage systematically different?")
    print("=" * 74)
    # per player-season, compare regular-season mean vs postseason mean
    rows = []
    for (pid, s), g in d.groupby(["player_id", "season"]):
        reg = g[~g["is_post"]]
        post = g[g["is_post"]]
        if len(reg) < 6 or len(post) < 2:
            continue
        rows.append(dict(
            player=g["player_display_name"].iloc[0], season=int(s),
            n_reg=len(reg), n_post=len(post),
            tgt_reg=reg["targets"].mean(), tgt_post=post["targets"].mean(),
            yds_reg=reg["receiving_yards"].mean(),
            yds_post=post["receiving_yards"].mean(),
            snap_reg=reg["snap_roll"].mean(), snap_post=post["snap_roll"].mean(),
        ))
    t = pd.DataFrame(rows)
    if t.empty:
        print("  not enough paired data")
        return t
    t["d_tgt"] = t["tgt_post"] - t["tgt_reg"]
    t["d_yds"] = t["yds_post"] - t["yds_reg"]
    print(f"  {len(t):,} player-seasons with 6+ regular and 2+ postseason games")
    for col, lbl in (("d_tgt", "targets/game"), ("d_yds", "receiving yards/game")):
        v = t[col].dropna()
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"    {lbl:22s} post minus reg: {v.mean():+7.3f} +- {se:.3f} SE  "
              f"(z = {v.mean() / max(se, 1e-9):+.2f})")
    print("\n    |z| under 2 means postseason usage is NOT systematically")
    print("    different on average, so excluding it buys nothing on average.")
    print("    The distortion would then be per-player variance, not bias.")

    # variance: is postseason usage noisier relative to a player's own norm?
    t["ratio"] = t["tgt_post"] / t["tgt_reg"].replace(0, np.nan)
    r = t["ratio"].dropna()
    print(f"\n    per-player targets ratio (post/reg): median {r.median():.3f}, "
          f"IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}")
    print(f"    share of players whose postseason targets were under half "
          f"their regular-season rate: {100 * float((r < 0.5).mean()):.0f}%")
    print(f"    share over 1.5x: {100 * float((r > 1.5).mean()):.0f}%")
    return t


def stage3(d, r):
    print("\n" + "=" * 74)
    print("STAGE 3 - 2026 week 1: which players would move, and by how much?")
    print("=" * 74)
    try:
        proj = r.project_week(2026, 1, min_targets=0.0)
    except Exception as exc:
        print(f"  project_week failed: {type(exc).__name__}: {exc}")
        return

    prior = d[d["season"] == 2025]
    tails = prior.groupby("player_id").tail(ROLL_N)
    aff_ids = set(tails[tails["is_post"]]["player_id"].unique())

    # recompute a regular-season-only tail for the same players
    reg = prior[~prior["is_post"]]
    reg_tail = reg.groupby("player_id").tail(ROLL_N)
    alt = reg_tail.groupby("player_id").agg(
        alt_tgt=("targets", "mean"),
        alt_snap=("snap_roll", "mean"),
        name=("player_display_name", "last"),
        team=("team", "last"),
    )
    built = tails.groupby("player_id").agg(
        cur_tgt=("targets", "mean"),
        n_post=("is_post", "sum"),
    )
    cmp = alt.join(built, how="inner")
    cmp = cmp[cmp.index.isin(aff_ids)]
    cmp["d_tgt"] = cmp["alt_tgt"] - cmp["cur_tgt"]

    # keep only players who are actually on the 2026 board
    on_board = set(proj["player_display_name"])
    cmp = cmp[cmp["name"].isin(on_board)]
    cmp = cmp.sort_values("d_tgt", ascending=False)

    print(f"  {len(cmp)} board players whose 2025 tail includes postseason games")
    if len(cmp) == 0:
        return
    print("\n  biggest upward moves if postseason were excluded:")
    print("    player                team  post_gms  tail_tgt  regonly_tgt   diff")
    for pid, row in cmp.head(15).iterrows():
        print(f"    {str(row['name'])[:20]:20s} {str(row['team']):4s} "
              f"{int(row['n_post']):8d} {row['cur_tgt']:9.2f} "
              f"{row['alt_tgt']:12.2f} {row['d_tgt']:+6.2f}")
    print("\n  biggest downward moves:")
    for pid, row in cmp.tail(8).iloc[::-1].iterrows():
        print(f"    {str(row['name'])[:20]:20s} {str(row['team']):4s} "
              f"{int(row['n_post']):8d} {row['cur_tgt']:9.2f} "
              f"{row['alt_tgt']:12.2f} {row['d_tgt']:+6.2f}")

    print(f"\n  mean |change| in tail targets: {cmp['d_tgt'].abs().mean():.3f}")
    print(f"  players moving by more than 1 target/game: "
          f"{int((cmp['d_tgt'].abs() > 1).sum())}")


def main():
    print("=" * 74)
    print("POSTSEASON BRIDGE CHECK")
    print("=" * 74)
    d, r = load()
    stage1(d)
    stage2(d)
    stage3(d, r)

    print("\n" + "=" * 74)
    print("HOW TO DECIDE")
    print("=" * 74)
    print("  Change the bridge ONLY if stage 2 shows a systematic difference")
    print("  (|z| above 2) AND stage 3 shows enough affected board players to")
    print("  matter. If postseason usage is merely noisier, dropping those games")
    print("  shortens the tail for exactly the players who played the most")
    print("  football, which trades one bias for another.")
    print("\n  Note the asymmetry: only deep-run teams are affected, so any")
    print("  distortion is concentrated rather than league-wide.")


if __name__ == "__main__":
    main()
