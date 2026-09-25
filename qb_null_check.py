"""qb_null_check.py - does the missing fillna(0) in qb_rushing matter?

ITEM 1 of the qb_rushing measurement sequence. Reports only. Writes nothing,
ships nothing, changes no constant.

THE QUESTION

  models/rushing.py fills carries and rushing_yards with 0 before computing
  rolling features, on the reasoning in its docstring note 4: a player with a
  stat row was ACTIVE, so a missing rushing line is a genuine zero and not
  absent data. models/qb_rushing.py does not do this.

  For running backs that fix was worth real rows. For quarterbacks it could
  matter MORE or not at all, and the direction is not obvious:

    - more, because a pocket passer with zero carries is the NORMAL case, and
      dropna in load_model would delete exactly the low-rushing population
      the module docstring says it wants to project;

    - not at all, because nflverse may store 0.0 rather than null for active
      players, which is what the receptions check found (0 nulls in 22,540
      rows). Phase 3.4 was called the highest-value fix in three consecutive
      handoffs and measured as a no-op. This could be the same.

  MEASURE IT. Do not assume either way.

THE MECHANISM, IF THERE ARE NULLS

  build_dataset computes the rolling mean BEFORE anything fills nulls, and
  pandas' rolling mean SKIPS NaN rather than treating it as zero. So a week
  with a null stat is excluded from the player's own history, and
  rush_yds_roll becomes the mean over only the weeks the stat was recorded.
  That is the same SHAPE as bug pattern 1 (a feature computed over a filtered
  population), with nullness as the filter instead of a volume threshold.

WHY THIS SCRIPT CARRIES A GATE

  To compare two versions of the rolling features it has to build them, which
  means reimplementing part of build_dataset. That is the exact thing that
  produced this project's worst errors, so per the working principle it must
  REPRODUCE qb_rushing.build_dataset()'s output before it is allowed to
  report anything new, and it aborts rather than warning.

  It also follows the other Phase 3.4 lesson: judge the change on the SERVED
  PROJECTION, unrounded, not on the fitted coefficients and not on
  project_week's rounded output, which produced a fake result last time.

Run from the repo root:
    python qb_null_check.py
"""
import sys

import numpy as np
import pandas as pd

from models import data_utils, qb_rushing
from sklearn.linear_model import LinearRegression

SEASONS = qb_rushing.SEASONS
FEATS = qb_rushing.LEAN_FEATS
TARGET = "rushing_yards"
KEYS = ["player_id", "season", "week"]


def _rule(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def build(fill_zero):
    """qb_rushing.build_dataset's feature pipeline, optionally with fillna(0).

    fill_zero=False must be byte-equivalent to the shipped function. That is
    what the gate checks.
    """
    ps = data_utils.load_player_stats(SEASONS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    qb = ps[ps["position"] == "QB"].copy()

    if fill_zero:
        qb["carries"] = qb["carries"].fillna(0)
        qb["rushing_yards"] = qb["rushing_yards"].fillna(0.0)

    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for col in ["rushing_yards", "carries"]:
        qb[f"{col}_roll"] = (qb.groupby("player_id")[col]
                             .transform(lambda s: s.shift(1)
                                        .rolling(6, min_periods=1).mean()))
    qb = qb.rename(columns={"rushing_yards_roll": "rush_yds_roll"})
    return qb


def gate(mine):
    _rule("GATE: reproduce qb_rushing.build_dataset() before reporting anything")
    shipped = qb_rushing.build_dataset()
    print(f"shipped build_dataset : {len(shipped):,} rows")
    print(f"this script, fill=off : {len(mine):,} rows")

    if len(shipped) != len(mine):
        print("\nABORT: row counts differ. The pipeline in this script is not")
        print("the pipeline in qb_rushing.build_dataset, so no number it")
        print("produces can be compared to anything.")
        sys.exit(1)

    a = shipped.sort_values(KEYS).reset_index(drop=True)
    b = mine.sort_values(KEYS).reset_index(drop=True)

    worst = {}
    for c in FEATS:
        d = (a[c] - b[c]).abs()
        # NaN in both places is agreement, not a difference.
        both_nan = a[c].isna() & b[c].isna()
        d = d[~both_nan]
        one_nan = int((a[c].isna() != b[c].isna()).sum())
        worst[c] = (float(d.max()) if len(d) else 0.0, one_nan)

    for c, (mx, one_nan) in worst.items():
        print(f"  {c:16s} max abs diff {mx:.12f}   null mismatches {one_nan}")

    if any(mx > 1e-9 or one_nan for mx, one_nan in worst.values()):
        print("\nABORT: the reimplementation does not reproduce the shipped")
        print("features. Fix this script before trusting any figure below.")
        sys.exit(1)

    print("\nGATE PASSED. The pipeline matches, so the fill=on variant is a")
    print("genuine one-change comparison.")
    return shipped


def section_nulls(off):
    _rule("SECTION 1: ARE THERE ANY NULLS AT ALL?")
    n = len(off)
    rows = []
    for c in ["carries", TARGET]:
        nulls = int(off[c].isna().sum())
        rows.append({"column": c, "rows": n, "nulls": nulls,
                     "null_pct": round(100 * nulls / max(1, n), 4),
                     "zeros": int((off[c] == 0).sum())})
    print(pd.DataFrame(rows).to_string(index=False))

    tot = sum(r["nulls"] for r in rows)
    if tot == 0:
        print()
        print("VERDICT: zero nulls. nflverse stores 0.0 for active QBs, same")
        print("as it does for receptions. The missing fillna is a LATENT")
        print("defect with no current effect. Match the pattern anyway to")
        print("remove the possibility, but expect a no-op, and do not record")
        print("it as a fix that moved anything.")
        return False

    print()
    print(f"{tot:,} null cells present, so this is NOT a no-op. Continuing.")

    # If nulls exist, WHO has them matters more than how many. A null
    # concentrated on low-rushing QBs is the survivorship case.
    nl = off[off[TARGET].isna() | off["carries"].isna()]
    print()
    print(f"rows with a null: {len(nl):,} across {nl['player_id'].nunique():,} QBs")
    if "player_display_name" in nl.columns:
        print("top 15 by null row count:")
        print(nl["player_display_name"].value_counts().head(15).to_string())
    print()
    print("null rows by season:")
    print(nl.groupby("season").size().to_string())
    return True


def section_dropna(off, on):
    _rule("SECTION 2: WHAT load_model's dropna DELETES")
    for label, df in (("as shipped", off), ("with fillna(0)", on)):
        tr = df[df["season"] <= 2024]
        kept = tr.dropna(subset=FEATS + [TARGET])
        print(f"{label:16s} train rows {len(tr):,} -> {len(kept):,} kept "
              f"({len(tr) - len(kept):,} dropped, "
              f"{(len(tr)-len(kept))/max(1,len(tr)):.2%})")
        if len(kept):
            print(f"{'':16s} mean {TARGET} of kept rows: "
                  f"{kept[TARGET].mean():.2f}")
        dropped = tr[~tr.index.isin(kept.index)]
        if len(dropped) and dropped[TARGET].notna().any():
            print(f"{'':16s} mean {TARGET} of DROPPED rows: "
                  f"{dropped[TARGET].mean():.2f}   <- if this is far below")
            print(f"{'':16s} the kept mean, the drop is survivorship")


def section_features(off, on):
    _rule("SECTION 3: HOW FAR DO THE ROLLING FEATURES MOVE?")
    a = off.sort_values(KEYS).reset_index(drop=True)
    b = on.sort_values(KEYS).reset_index(drop=True)
    if len(a) != len(b):
        print(f"row counts differ ({len(a):,} vs {len(b):,}); fillna changed")
        print("the population, which it should not. Investigate.")
        return
    for c in FEATS:
        d = (a[c] - b[c]).abs().dropna()
        if not len(d):
            print(f"  {c}: nothing comparable")
            continue
        print(f"  {c:16s} mean abs {d.mean():.4f}  max abs {d.max():.4f}  "
              f"rows moved >0.01: {int((d > 0.01).sum()):,}")


def section_projection(off, on):
    _rule("SECTION 4: THE ONLY NUMBER THAT DECIDES THIS")
    print("Phase 3.4's lesson: coefficients can move a lot while the SERVED")
    print("projection does not. Judge on the projection, unrounded, because")
    print("the first Phase 3.4 measurement used rounded output and produced a")
    print("result that was entirely quantization.")
    print()

    models = {}
    for label, df in (("as shipped", off), ("with fillna(0)", on)):
        tr = df[df["season"] <= 2024].dropna(subset=FEATS + [TARGET])
        m = LinearRegression().fit(tr[FEATS], tr[TARGET])
        models[label] = m
        coefs = ", ".join(f"{f}={c:+.4f}" for f, c in zip(FEATS, m.coef_))
        print(f"{label:16s} intercept {m.intercept_:+.4f}   {coefs}")

    # Score the SAME rows under both models, so the comparison is paired.
    common = on.dropna(subset=FEATS).sort_values(KEYS).reset_index(drop=True)
    off_idx = off.set_index(KEYS)
    common = common[common.set_index(KEYS).index.isin(off_idx.index)]
    if not len(common):
        print("\nno rows with usable features under both. Nothing to compare.")
        return

    p_off = models["as shipped"].predict(common[FEATS])
    p_on = models["with fillna(0)"].predict(common[FEATS])
    d = np.abs(p_off - p_on)
    print()
    print(f"paired rows scored : {len(common):,}")
    print(f"mean abs diff      : {d.mean():.4f} rushing yards")
    print(f"max abs diff       : {d.max():.4f} rushing yards")
    print(f"rows moving > 1.0  : {int((d > 1.0).sum()):,}")
    print(f"rows moving > 5.0  : {int((d > 5.0).sum()):,}")
    print()
    print("HOW TO READ IT. QB rushing lines sit on a yardage grid, so a move")
    print("of a fraction of a yard is operationally irrelevant the same way")
    print("0.137 receptions was. A move of several yards is not.")

    if len(common):
        common = common.copy()
        common["_d"] = d
        top = common.nlargest(10, "_d")
        cols = [c for c in ("player_display_name", "season", "week") if c in top.columns]
        print()
        print("10 largest movers:")
        print(top[cols + ["_d"]].to_string(index=False))


def main():
    off = build(fill_zero=False)
    shipped = gate(off)
    # Use the shipped frame from here on where possible, so the reported
    # numbers describe the real object rather than the reproduction.
    has_nulls = section_nulls(shipped)
    on = build(fill_zero=True)
    section_dropna(off, on)
    section_features(off, on)
    section_projection(off, on)

    _rule("VERDICT")
    if not has_nulls:
        print("No nulls. Expect every section above to show zero movement.")
        print("Item 2 (parameterising the train season) is unaffected and is")
        print("still required, because that blocks leave-season-out scoring")
        print("regardless of what nulls do.")
    else:
        print("Nulls present. Read section 4 first: if the served projection")
        print("barely moves, this is a latent defect to fix for hygiene, not")
        print("a finding. If it moves by yards, it changes the market.")
    print()
    print("Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
