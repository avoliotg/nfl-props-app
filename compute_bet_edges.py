"""
OpalScales - compute projection / edge / tier for specific bets.

WHY
    The Thursday SF @ LA snapshot stored NULL projections, because import_lines
    matched against project_week, which had switched to the played-games path
    once the Wednesday game finished and so contained only NE and SEA players.
    That is fixed for future imports, but the stored nulls cannot be recovered.

    This recomputes edge and tier for a handful of bets so the tracker sheet has
    no gaps. Note the projections come from the CURRENT model on data that now
    includes played games, so they are close to but not identical to what the
    model said pre-kickoff. Good enough for a bet log; not a historical record.

EDIT
    The BETS list below. Then: python compute_bet_edges.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# market, player, side, line, over_odds, under_odds
BETS = [
    ("receiving", "Christian McCaffrey", "OVER", 38.5, -114, -114),
    ("receiving", "Christian McCaffrey", "OVER", 39.5, -113, -113),
    ("receiving", "Puka Nacua", "UNDER", 90.5, -113, -113),
    ("rushing", "Blake Corum", "OVER", 44.5, -113, -113),
    ("qb_passing", "Brock Purdy", "UNDER", 243.5, -113, -113),
    ("receptions", "George Kittle", "OVER", 3.5, -106, -106),
    ("qb_rushing", "Drake Maye", "OVER", 24.5, -114, -114),
    ("qb_passing", "Aaron Rodgers", "OVER", 210.5, -114, -114),
    ("receiving", "Dallas Goedert", "OVER", 34.5, -114, -114),
]
SEASON, WEEK = 2026, 1


def main():
    import pandas as pd
    import mc
    import db
    from models import (receiving, receptions, rushing, qb_passing,
                        anytime_td, qb_rushing)

    MODULES = {"receiving": receiving, "receptions": receptions,
               "rushing": rushing, "qb_passing": qb_passing,
               "anytime_td": anytime_td, "qb_rushing": qb_rushing}

    def board(market):
        """Played rows unioned with the assembler, so every team is covered
        regardless of which games have finished."""
        mod = MODULES[market]
        frames = []
        for getter in ("project_week", "build_upcoming_week"):
            fn = getattr(mod, getter, None)
            if fn is None:
                continue
            try:
                df = fn(SEASON, WEEK)
            except Exception:
                continue
            if df is not None and len(df):
                frames.append(df)
        if market == "rushing":
            for getter in ("project_week", "build_upcoming_week"):
                fn = getattr(qb_rushing, getter, None)
                if fn is None:
                    continue
                try:
                    df = fn(SEASON, WEEK)
                except Exception:
                    continue
                if df is not None and len(df):
                    df = df.copy()
                    df["is_qb_model"] = True
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(subset=["player_display_name"], keep="first")

    cache = {}
    print(f"{'market':12s} {'player':22s} {'side':6s} {'line':>7s} "
          f"{'proj':>7s} {'edge':>7s} {'tier':8s}")
    rows = []
    for mkt, player, side, line, oo, uo in BETS:
        if mkt not in cache:
            cache[mkt] = board(mkt)
        bd = cache[mkt]
        proj = edge = tier = None
        if len(bd):
            m = bd[bd["player_display_name"].apply(db._norm_name)
                   == db._norm_name(player)]
            if len(m):
                r = m.iloc[0]
                proj = float(r["projection"])
                eff = "qb_rushing" if r.get("is_qb_model") else mkt
                res = mc.edge_calc(eff, proj, line, 0,
                                   over_odds=oo, under_odds=uo)
                if res:
                    # take the edge for the side actually BET, not the model's
                    # preferred side, since a losing side still has an edge
                    # figure and the log should record what was wagered
                    edge = (res["edge_over"] if side == "OVER"
                            else res["edge_under"])
                    tier = mc.tier_for_edge(edge)
        ps = f"{proj:7.1f}" if proj is not None else "      -"
        es = f"{edge:+7.1f}" if edge is not None else "      -"
        print(f"{mkt:12s} {player[:22]:22s} {side:6s} {line:7.1f} "
              f"{ps} {es} {str(tier or '-'):8s}")
        rows.append((mkt, player, side, line, proj, edge, tier))

    missing = [r for r in rows if r[4] is None]
    if missing:
        print(f"\n{len(missing)} player(s) not found on their board:")
        for r in missing:
            print(f"  {r[1]} ({r[0]})")
        print("  Check the spelling against the app's board.")

    print("\nTSV of edge and tier, in BETS order:")
    for mkt, player, side, line, proj, edge, tier in rows:
        print(f"{player}\t{mkt}\t{side}\t{line}\t"
              f"{'' if proj is None else round(proj,1)}\t"
              f"{'' if edge is None else round(edge,1)}\t{tier or ''}")


if __name__ == "__main__":
    main()
