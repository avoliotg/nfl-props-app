"""qb_split_report.py - measure data_utils.split_qb_rushing on REAL rows.

Reports only. Writes nothing, changes nothing, fits nothing.

Two questions, and the second one is the one that decides whether plan item
3.5 was worth doing:

  1. What share of real 'rushing' line rows can be matched to a position?
     That number is what `min_match_rate` should be set from. Until it is
     measured, the gate is off and the function reports instead of aborting.

  2. Do the reassigned rows actually JOIN to qb_rushing.build_dataset()?
     The plan claims this recovers about 40 percent of rushing's join rate.
     If QB lines relabel cleanly but still fail to join, the market stays
     unmeasurable and the label bought nothing. This is the payoff check and
     it has never been run.

Run from the repo root:
    python qb_split_report.py
    python qb_split_report.py --cache lines_cache.parquet
    python qb_split_report.py --no-join        (skip section 2, faster)
"""
import argparse
import os
import sys

import pandas as pd

try:
    from models import data_utils
except ImportError:
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "models", "data_utils.py")
    _spec = importlib.util.spec_from_file_location("data_utils", _p)
    data_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(data_utils)


def _rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_lines(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("The 354,554 historical rows live in lines_cache.parquet, which")
        print("eval_harness writes via --cache. Point --cache at it, or pass")
        print("--cache <other.parquet>.")
        sys.exit(2)
    df = pd.read_parquet(path)
    print(f"loaded {len(df):,} rows from {path}")
    print(f"columns: {sorted(df.columns)}")
    return df


def resolve_cols(df, args):
    """Find the market / player / season columns without guessing silently."""
    cands = {
        "market": [args.market_col, "market", "market_key", "mkt"],
        "player": [args.player_col, "player", "player_name", "description",
                   "player_display_name"],
        "season": [args.season_col, "season", "year"],
    }
    out = {}
    for role, names in cands.items():
        hit = next((c for c in names if c and c in df.columns), None)
        if hit is None:
            print(f"ERROR: no column found for '{role}'. Tried {names}.")
            print(f"Pass --{role}-col explicitly. Columns present: "
                  f"{sorted(df.columns)}")
            sys.exit(2)
        out[role] = hit
    print(f"using columns: market='{out['market']}', player='{out['player']}', "
          f"season='{out['season']}'")
    return out


def section_split(df, cols):
    _rule("SECTION 1: THE SPLIT")

    mk = df[cols["market"]].astype(str).str.strip().str.lower()
    print("market value counts (raw, before the split):")
    print(mk.value_counts().to_string())

    out, report = data_utils.split_qb_rushing(
        df, market_col=cols["market"], player_col=cols["player"],
        season_col=cols["season"])

    print()
    print("report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if report.get("match_rate") is None:
        print()
        print("No 'rushing' rows in this frame. Nothing to measure.")
        return out, report

    print()
    print("SET THE GATE FROM THIS:")
    print(f"  measured match_rate = {report['match_rate']}")
    print(f"  suggested min_match_rate, rounded DOWN to leave headroom for")
    print(f"  inactive players in future weeks: "
          f"{max(0.0, (int(report['match_rate'] * 20) / 20) - 0.05):.2f}")
    print("  (that is a starting point, not a measurement. Judge it against")
    print("   the unmatched count and the unmatched names below.)")

    # Per-season, because book composition and roster churn both vary by year
    # and a split that works in 2025 and fails in 2023 is worth seeing.
    tgt = df[cols["market"]].astype(str).str.strip().str.lower() == "rushing"
    per = pd.DataFrame({
        "season": pd.to_numeric(df.loc[tgt, cols["season"]], errors="coerce"),
        "reassigned": (out.loc[tgt, cols["market"]] == "qb_rushing").values,
    })
    tab = per.groupby("season").agg(
        rushing_rows=("reassigned", "size"),
        reassigned=("reassigned", "sum"))
    tab["share"] = (tab["reassigned"] / tab["rushing_rows"]).round(4)
    print()
    print("per season:")
    print(tab.to_string())

    # The unmatched names are the actionable output here: if they are mostly
    # one spelling variant, that is a name-normalisation fix (plan 3.6), not a
    # position-lookup problem.
    sub = df.loc[tgt, [cols["season"], cols["player"]]].copy()
    sub["norm"] = data_utils.norm_join_name(sub[cols["player"]])
    per_season, pooled = data_utils.player_position_map(
        sorted(pd.to_numeric(sub[cols["season"]], errors="coerce")
               .dropna().astype(int).unique()))
    known = set(pooled["player_norm"])
    unmatched = sub[~sub["norm"].isin(known)]
    print()
    print(f"unmatched rows: {len(unmatched):,} "
          f"({len(unmatched) / max(1, len(sub)):.2%} of rushing rows), "
          f"{unmatched['norm'].nunique():,} distinct names")
    if len(unmatched):
        print("top 20 unmatched names by row count:")
        print(unmatched[cols["player"]].value_counts().head(20).to_string())

    top = (out.loc[tgt & (out[cols["market"]] == "qb_rushing"), cols["player"]]
           .value_counts().head(15))
    if len(top):
        print()
        print("top 15 reassigned players by row count:")
        print(top.to_string())

    return out, report


def section_join(out, cols):
    _rule("SECTION 2: DO THE REASSIGNED ROWS ACTUALLY JOIN?")
    print("This is the payoff check. A clean label that still fails to join")
    print("leaves qb_rushing exactly as unmeasurable as it is today.")
    print()

    try:
        from models import qb_rushing
    except Exception as e:
        print(f"could not import models.qb_rushing: {type(e).__name__}: {e}")
        print("skipping section 2.")
        return

    try:
        qb = qb_rushing.build_dataset()
    except Exception as e:
        print(f"qb_rushing.build_dataset() failed: {type(e).__name__}: {e}")
        print("skipping section 2.")
        return

    qb = qb.copy()
    qb["norm"] = data_utils.norm_join_name(qb["player_display_name"])
    qb["_s"] = pd.to_numeric(qb["season"], errors="coerce").astype("Int64")
    qb["_w"] = pd.to_numeric(qb["week"], errors="coerce").astype("Int64")
    keys = set(zip(qb["_s"], qb["_w"], qb["norm"]))
    season_keys = set(zip(qb["_s"], qb["norm"]))
    print(f"qb_rushing.build_dataset(): {len(qb):,} rows, "
          f"{qb['norm'].nunique():,} distinct QBs, "
          f"seasons {sorted(qb['_s'].dropna().unique().tolist())}")

    q = out[out[cols["market"]] == "qb_rushing"].copy()
    if len(q) == 0:
        print("no qb_rushing rows after the split. Nothing to join.")
        return

    week_col = next((c for c in ("week", "wk") if c in q.columns), None)
    q["norm"] = data_utils.norm_join_name(q[cols["player"]])
    q["_s"] = pd.to_numeric(q[cols["season"]], errors="coerce").astype("Int64")

    if week_col is None:
        print("no week column in the line frame, so only a season-level join")
        print("can be checked. Pass a frame with a week column for the exact")
        print("player-week join rate.")
        hit = pd.Series(list(zip(q["_s"], q["norm"]))).isin(season_keys)
        print(f"season-level join: {hit.sum():,} of {len(q):,} "
              f"({hit.mean():.2%})")
        return

    q["_w"] = pd.to_numeric(q[week_col], errors="coerce").astype("Int64")
    exact = pd.Series(list(zip(q["_s"], q["_w"], q["norm"]))).isin(keys)
    loose = pd.Series(list(zip(q["_s"], q["norm"]))).isin(season_keys)

    print()
    print(f"reassigned line rows:            {len(q):,}")
    print(f"join on season+week+name:        {exact.sum():,} ({exact.mean():.2%})")
    print(f"join on season+name only:        {loose.sum():,} ({loose.mean():.2%})")
    print()
    print("A large gap between the two means the names match but the")
    print("player-weeks do not, which is a bye/inactive/DNP question rather")
    print("than a normalisation one. A low figure on BOTH is a name problem")
    print("and points at plan item 3.6.")

    miss = q.loc[~loose.values, cols["player"]].value_counts().head(20)
    if len(miss):
        print()
        print("top 20 reassigned-but-unjoinable names:")
        print(miss.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--market-col", default=None)
    ap.add_argument("--player-col", default=None)
    ap.add_argument("--season-col", default=None)
    ap.add_argument("--no-join", action="store_true",
                    help="skip section 2 (it loads nflverse QB stats)")
    args = ap.parse_args()

    df = load_lines(args.cache)
    cols = resolve_cols(df, args)
    out, report = section_split(df, cols)
    if not args.no_join and report.get("reassigned"):
        section_join(out, cols)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
