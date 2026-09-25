"""name_resolve.py - measure the collision risk, then PROPOSE aliases.

Three jobs, in order of importance:

  1. MEASURE THE COLLISION COUNT that parenthetical-tag stripping creates.
     FanDuel writes "Lamar Jackson (BAL)" because two players in league
     history normalize to "lamar jackson". norm_join_name now strips the tag,
     which turns a guaranteed miss into a match that is correct only if the
     normalized name is unique. The position lookup survives a collision
     because it takes the modal position; actual_result does NOT, because it
     takes iloc[0]. This section says how many names are affected and which.

  2. RE-MEASURE the unmatched set under the new normalizer, so the accent and
     tag fixes are verified against real rows rather than assumed.

  3. PROPOSE aliases for whatever is still unmatched, with evidence attached
     (position, seasons, row counts) and printed as paste-ready dict lines.

It PROPOSES. It does not write. Adding an alias on this script's say-so
without reading the evidence column is exactly the error class the project
has a hard-gate rule about.

Run from the repo root:
    python name_resolve.py
    python name_resolve.py --cache lines_cache.parquet --market rushing
    python name_resolve.py --market all
"""
import argparse
import difflib
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

SEASONS = [2022, 2023, 2024, 2025, 2026]


def _rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_nflverse():
    ps = data_utils.load_player_stats(SEASONS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    need = ["player_display_name", "position", "season"]
    missing = [c for c in need if c not in ps.columns]
    if missing:
        print(f"ERROR: load_player_stats missing {missing}")
        sys.exit(2)
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps["norm"] = data_utils.norm_join_name(ps["player_display_name"])
    id_col = next((c for c in ("player_id", "gsis_id", "pfr_id")
                   if c in ps.columns), None)
    return ps, id_col


def section_collisions(ps, id_col):
    _rule("SECTION 1: COLLISION RISK FROM TAG STRIPPING")

    if id_col is None:
        print("no player id column found, so collisions cannot be measured")
        print("by identity. Falling back to distinct display names, which")
        print("OVERSTATES collisions because one player can be spelled two")
        print("ways in the same table.")
        grp = ps.groupby("norm")["player_display_name"].nunique()
    else:
        print(f"measuring identity collisions on '{id_col}'")
        grp = ps.groupby("norm")[id_col].nunique()

    colliding = grp[grp > 1].sort_values(ascending=False)
    print()
    print(f"distinct normalized names : {len(grp):,}")
    print(f"names mapping to >1 player: {len(colliding):,} "
          f"({len(colliding)/max(1,len(grp)):.2%})")

    if len(colliding) == 0:
        print()
        print("VERDICT: no collisions. Tag stripping is safe, and so is")
        print("actual_result's iloc[0].")
        return set()

    print()
    print("colliding names, with the players behind each:")
    for nm in colliding.head(25).index:
        sub = ps[ps["norm"] == nm]
        who = sub.groupby(
            [c for c in (id_col, "position") if c]).size() if id_col else \
            sub.groupby("position").size()
        detail = "; ".join(
            f"{k} n={v}" for k, v in who.items())
        print(f"  {nm}: {detail}")

    print()
    print("HOW TO READ THIS. A collision matters in two places only:")
    print("  - actual_result, which takes iloc[0] and can therefore grade a")
    print("    bet against the wrong player. That is a MONEY bug.")
    print("  - the position lookup, which takes the modal position and is")
    print("    robust unless the two players are both high-volume.")
    print("If any colliding name is a player you would actually bet, the tag")
    print("strip needs a team-aware join rather than a name-only one.")
    return set(colliding.index)


def section_unmatched(args, ps):
    _rule("SECTION 2: WHAT IS STILL UNMATCHED UNDER THE NEW NORMALIZER")

    if not os.path.exists(args.cache):
        print(f"{args.cache} not found, skipping sections 2 and 3.")
        return None
    df = pd.read_parquet(args.cache)
    if args.market != "all":
        df = df[df["market"].astype(str).str.strip().str.lower() == args.market]
    print(f"{len(df):,} rows in scope (market={args.market})")

    df = df.copy()
    df["norm"] = data_utils.norm_join_name(df["player"])
    known = set(ps["norm"])
    df["matched"] = df["norm"].isin(known)

    n = len(df)
    n_un = int((~df["matched"]).sum())
    print(f"matched   : {n - n_un:,} ({(n-n_un)/max(1,n):.4%})")
    print(f"unmatched : {n_un:,} ({n_un/max(1,n):.4%}), "
          f"{df.loc[~df['matched'], 'norm'].nunique():,} distinct names")

    if args.market == "rushing":
        print()
        print("BASELINE FOR COMPARISON. qb_split_report.py measured 153")
        print("unmatched rushing rows and 11 distinct names BEFORE the accent")
        print("and tag fixes. If the number above is not lower, the fixes did")
        print("not do what the smoke test says they do, and the normalizer in")
        print("models/ is not the one being imported here.")

    if n_un == 0:
        return df
    print()
    print("unmatched names by row count:")
    print(df.loc[~df["matched"], "player"].value_counts().head(30).to_string())
    return df


def _candidates(target, pool, ps, id_col, limit=4):
    """Propose nflverse names for one unmatched name, with evidence."""
    out = []

    # Strategy A: overall string similarity. Catches Mitch/Mitchell,
    # Josh/Joshua, Eli/Elijah, Cameron/Cam.
    for c in difflib.get_close_matches(target, pool, n=limit, cutoff=0.72):
        out.append((c, "similar"))

    # Strategy B: same surname plus compatible first token. Catches the
    # abbreviated forms ("m rudolph" -> "mason rudolph") that string
    # similarity scores too low to find.
    parts = target.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        for c in pool:
            cp = c.split()
            if len(cp) >= 2 and cp[-1] == last and cp[0].startswith(first[0]):
                if c not in [o[0] for o in out]:
                    out.append((c, "surname+initial"))

    rows = []
    for cand, how in out[:limit + 2]:
        sub = ps[ps["norm"] == cand]
        pos = sub["position"].mode()
        rows.append({
            "candidate": cand,
            "via": how,
            "position": pos.iloc[0] if len(pos) else "?",
            "seasons": ",".join(str(int(s)) for s in
                                sorted(sub["season"].dropna().unique())),
            "nflverse_rows": len(sub),
            "spelled": sub["player_display_name"].mode().iloc[0]
            if len(sub) else "?",
        })
    return rows


def section_propose(df, ps, id_col):
    _rule("SECTION 3: PROPOSED ALIASES (evidence attached, nothing written)")

    if df is None:
        return
    un = df.loc[~df["matched"]]
    if len(un) == 0:
        print("nothing unmatched. No aliases needed.")
        return

    pool = sorted(set(ps["norm"]))
    counts = un["norm"].value_counts()
    accepted = []

    for nm in counts.index:
        raw = un.loc[un["norm"] == nm, "player"].mode().iloc[0]
        print()
        print(f"UNMATCHED: {nm!r}  (as written: {raw!r}, {counts[nm]} rows)")
        rows = _candidates(nm, pool, ps, id_col)
        if not rows:
            print("  no candidate found. Either the player is absent from")
            print("  nflverse entirely, or the name is too abbreviated to")
            print("  resolve without a team. Leave it.")
            continue
        print(pd.DataFrame(rows).to_string(index=False))
        # Only a single high-confidence candidate is worth pre-writing, and
        # even then it is a proposal.
        if len(rows) == 1:
            accepted.append((nm, rows[0]["candidate"], rows[0]["position"],
                             counts[nm]))

    print()
    print("-" * 72)
    print("PASTE-READY, FOR THE CANDIDATES WITH EXACTLY ONE MATCH.")
    print("Read the position and season columns above before accepting any")
    print("line. A candidate at the right position in the right seasons is")
    print("probably the player; one at the wrong position is a coincidence of")
    print("spelling and would create a WRONG join, which is worse than the")
    print("missing one it replaces.")
    print("-" * 72)
    print("# add inside _NAME_ALIASES in models/data_utils.py")
    for nm, cand, pos, n in accepted:
        print(f'    "{nm}": "{cand}",   # {pos}, {n} rows')
    if not accepted:
        print("    (none had a single unambiguous candidate)")
    print()
    print("Names with more than one candidate need a human decision. Names")
    print("with none should stay unmatched.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--market", default="rushing",
                    help="market to scope sections 2 and 3, or 'all'")
    args = ap.parse_args()

    ps, id_col = load_nflverse()
    print(f"nflverse: {len(ps):,} player-week rows, "
          f"{ps['norm'].nunique():,} distinct normalized names, "
          f"seasons {SEASONS}")

    section_collisions(ps, id_col)
    df = section_unmatched(args, ps)
    section_propose(df, ps, id_col)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
