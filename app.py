import streamlit as st
import pandas as pd
import betlog

st.set_page_config(page_title="NFL Props", layout="wide")

MARKETS = {
    "Receiving Yards": "receiving",
    "Receptions": "receptions",
    "Rushing Yards": "rushing",
    "QB Passing Yards": "qb_passing",
    "Anytime TD": "anytime_td",
}

from models import receiving, receptions, rushing, qb_passing, anytime_td
MODULES = {
    "receiving": receiving,
    "receptions": receptions,
    "rushing": rushing,
    "qb_passing": qb_passing,
    "anytime_td": anytime_td,
}

st.markdown("<style>h1{color:#e0873a;}</style>", unsafe_allow_html=True)
st.title("🔮 OpalScales")
st.caption("An OpalScales modeling project · 🏈 NFL player prop projections")

market_name = st.selectbox("Market", list(MARKETS.keys()),
                           help="Which prop market to project.")
market_key = MARKETS[market_name]
module = MODULES[market_key]

# Is this a probability market (Anytime TD) rather than a yardage/number market?
IS_PROB = getattr(module, "IS_PROBABILITY", False)
# market-aware label for the projection column
if IS_PROB:
    PROJ_LABEL = "Proj TD%"
elif market_key == "receptions":
    PROJ_LABEL = "Proj Catches"
else:
    PROJ_LABEL = "Proj Yds"
st.divider()

tab_board, tab_player, tab_scorecard, tab_top = st.tabs(
    ["📋 Board", "🔍 Player", "📊 Scorecard", "🎯 Top Plays"])

TIER_COLORS = {"Pass": "#8a7f70", "Lean": "#e6c14d",
               "Strong": "#4caf72", "Max": "#f0964a"}

HEADER_HELP = {
    "Player": "Qualifying player.",
    "Proj": "Model projection.",
    "Model %": "Model's probability the player scores a TD.",
    "Line": "The sportsbook over/under line you entered.",
    "Odds": "The American odds you entered (e.g. +150).",
    "Implied %": "Probability implied by the odds you entered.",
    "Gap": "Edge. Yardage: proj − line. TD: model% − implied%.",
    "Conf": "Relative confidence 0-100. NOT a win probability.",
    "Side": "Which side the model favors.",
    "Tier": "Pass / Lean / Strong / Max — bigger edge = stronger.",
}


def render_html_table(df, cols, aligns):
    def conf_color(v):
        if v >= 85: return "#f0964a"
        if v >= 65: return "#e6c14d"
        if v >= 45: return "#c0b090"
        return "#8a7f70"

    rows = ""
    for i, r in df.reset_index(drop=True).iterrows():
        bg = "#1e1912" if i % 2 == 0 else "#2a2318"
        cells = ""
        for c, a in zip(cols, aligns):
            val = r[c]
            style = f"padding:9px 12px;text-align:{a};"
            if c == "Tier":
                style += f"font-weight:800;text-transform:uppercase;color:{TIER_COLORS.get(val,'#8a7f70')};"
                cells += f'<td style="{style}">{val}</td>'
            elif c == "Side":
                sc = "#4caf72" if val == "OVER" else "#e0655a" if val == "UNDER" else "#c0b090"
                style += f"font-weight:700;color:{sc};"
                cells += f'<td style="{style}">{val}</td>'
            elif c == "Conf":
                style += f"font-weight:800;color:{conf_color(val)};"
                cells += f'<td style="{style}">{val:.0f}</td>'
            elif c == "Gap":
                gc = "#4caf72" if val > 0 else "#e0655a" if val < 0 else "#c0b090"
                style += f"font-weight:700;color:{gc};"
                cells += f'<td style="{style}">{val:+.1f}</td>'
            elif isinstance(val, (int, float)) and pd.notna(val):
                cells += f'<td style="{style}">{val:.1f}</td>'
            else:
                if c == "Player":
                    style += "font-weight:600;"
                cells += f'<td style="{style}">{val}</td>'
        rows += f'<tr style="background:{bg};">{cells}</tr>'

    header = "".join(
        f'<th title="{HEADER_HELP.get(h,h)}" style="padding:11px 12px;text-align:{a};'
        f'background:#e0873a;color:#161310;font-weight:800;font-size:0.8rem;'
        f'text-transform:uppercase;letter-spacing:0.05em;cursor:help;">{h}</th>'
        for h, a in zip(cols, aligns))

    return f"""<div style="border-radius:10px;overflow:hidden;border:1px solid #3a2f1e;
        box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-top:8px;max-width:1000px;">
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;
        font-family:'Source Sans Pro',sans-serif;">
        <thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"""


# ============ BOARD ============
with tab_board:
    st.subheader(f"{market_name} — Board")

    c1, c2 = st.columns(2)
    with c1:
        season = st.selectbox("Season", module.available_seasons(),
                              index=len(module.available_seasons()) - 1,
                              help="NFL season to view.")
    with c2:
        weeks = module.available_weeks(season)
        week = st.selectbox("Week", weeks, index=len(weeks) - 1,
                            help="Week within the season. Weeks 19+ are playoffs.")

    board = module.project_week(season, week)

    if len(board) == 0:
        st.warning("No projections available for that week yet.")
    else:
        # figure out the market's volume column (targets/carries/attempts/touches)
        vol_cols = [c for c in board.columns if c.endswith("_roll")]

        if IS_PROB:
            st.info("✏️ **Type the sportsbook's American odds (e.g. +150, -200)** "
                    "in the 'Your Odds' column to see the edge.")
            input_label = "Your Odds ✏️"
        else:
            st.info("✏️ **Type a sportsbook line in the 'Your Line' column** to grade an edge.")
            input_label = "Your Line ✏️"

        board = board.copy()
        board["your_input"] = None

        colcfg = {
            "player_display_name": st.column_config.TextColumn("Player", width="medium"),
            "team": st.column_config.TextColumn("Tm", width="small"),
            "opponent_team": st.column_config.TextColumn("Opp", width="small"),
            "position": st.column_config.TextColumn("Pos", width="small"),
            "your_input": st.column_config.NumberColumn(input_label, format="%.1f", width="small"),
        }
        colcfg["projection"] = st.column_config.NumberColumn(PROJ_LABEL, format="%.1f", width="small")
        # volume columns get simple labels
        vol_labels = {"targets_roll": "Recent Tgts", "snap_roll": "Recent Snap%",
                      "carries_roll": "Recent Car", "attempts_roll": "Recent Att",
                      "touches_roll": "Recent Tch"}
        for vc in vol_cols:
            fmt = "%.0f%%" if vc == "snap_roll" else "%.1f"
            colcfg[vc] = st.column_config.NumberColumn(vol_labels.get(vc, vc), format=fmt, width="small")

        display_cols = ["player_display_name", "team", "opponent_team", "position",
                        "projection"] + vol_cols + ["your_input"]
        edited = st.data_editor(
            board[display_cols], use_container_width=False, width=1000, hide_index=True,
            disabled=[c for c in display_cols if c != "your_input"],
            column_config=colcfg, key="board_editor")

        graded = edited[edited["your_input"].notna()].copy()

        # sanity guard (yardage only; odds have their own valid ranges)
        if not IS_PROB and len(graded) > 0:
            bad = graded[(graded["your_input"] < 0) | (graded["your_input"] > 150)]
            if len(bad) > 0:
                names = ", ".join(f"{r['player_display_name']} ({r['your_input']:.0f})"
                                  for _, r in bad.iterrows())
                override = st.checkbox("☑ I've checked — grade the flagged lines anyway",
                                       key="sanity_override")
                if not override:
                    st.warning(f"⚠️ These lines look off (outside 0-150): {names}. "
                               f"Tick the box to grade anyway.")
                    graded = graded[(graded["your_input"] >= 0) & (graded["your_input"] <= 150)]

        if len(graded) > 0:
            if IS_PROB:
                graded["implied"] = graded["your_input"].apply(module.american_to_prob)
                graded["gap"] = (graded["projection"] - graded["implied"]).round(1)
                graded["side"] = graded["gap"].apply(lambda g: "OVER" if g > 0 else "UNDER")
                graded["tier"] = graded["gap"].apply(module.tier_for_gap)
                graded["confidence"] = graded["gap"].apply(module.confidence_for_gap)
                graded = graded.sort_values("gap", key=lambda s: s.abs(),
                                            ascending=False).reset_index(drop=True)
                show = graded.rename(columns={
                    "player_display_name": "Player", "projection": "Model %",
                    "your_input": "Odds", "implied": "Implied %", "gap": "Gap",
                    "confidence": "Conf", "side": "Side", "tier": "Tier"})
                cols = ["Player", "Model %", "Odds", "Implied %", "Gap", "Conf", "Side", "Tier"]
                aligns = ["left", "left", "left", "left", "left", "left", "left", "left"]
            else:
                graded["gap"] = (graded["projection"] - graded["your_input"]).round(1)
                graded["side"] = graded["gap"].apply(lambda g: "OVER" if g > 0 else "UNDER")
                graded["tier"] = graded["gap"].apply(module.tier_for_gap)
                graded["confidence"] = graded["gap"].apply(module.confidence_for_gap)
                graded = graded.sort_values("gap", key=lambda s: s.abs(),
                                            ascending=False).reset_index(drop=True)
                show = graded.rename(columns={
                    "player_display_name": "Player", "projection": "Proj",
                    "your_input": "Line", "gap": "Gap", "confidence": "Conf",
                    "side": "Side", "tier": "Tier"})
                cols = ["Player", "Proj", "Line", "Gap", "Conf", "Side", "Tier"]
                aligns = ["left", "left", "left", "left", "left", "left", "left"]

            st.markdown("### Your entered lines")
            st.markdown(render_html_table(show, cols, aligns), unsafe_allow_html=True)
            st.caption("Hover any column header (ⓘ) for what it means.")

            # ---- save to log ----
            grid = graded[["player_display_name", "projection", "your_input",
                           "gap", "confidence", "side", "tier"]].copy()
            grid["bet"] = False
            logged_view = st.data_editor(
                grid, use_container_width=False, width=1000, hide_index=True,
                disabled=["player_display_name", "projection", "your_input",
                          "gap", "confidence", "side", "tier"],
                column_config={
                    "player_display_name": st.column_config.TextColumn("Player", width="medium"),
                    "bet": st.column_config.CheckboxColumn("Bet?", width="small",
                        help="Check if you actually placed this bet."),
                }, key="graded_editor")

            if st.button("💾 Save to Log", type="primary"):
                entries = logged_view.copy()
                entries["logged_at"] = betlog.now_stamp()
                entries["market"] = market_key
                entries["season"] = season
                entries["week"] = week
                entries["result_yards"] = None
                entries["outcome"] = None
                entries = entries.rename(columns={"player_display_name": "player",
                                                  "your_input": "line"})
                entries = entries[betlog.COLUMNS]
                betlog.append_entries(entries)
                st.success(f"Saved {len(entries)} pick(s) to the log.")

# ============ PLAYER ============
with tab_player:
    st.subheader(f"{market_name} — Player Lookup")

    pseason = st.selectbox("Season", module.available_seasons(),
                           index=len(module.available_seasons()) - 1,
                           key="player_season", help="Season to look up.")
    players = module.all_players(pseason)
    if not players:
        st.info("No players available for that season.")
    else:
        player = st.selectbox("Player", players, key="player_pick")
        hist = module.player_history(pseason, player)

        if len(hist) == 0:
            st.info("No projection history for this player/season.")
        else:
            st.markdown(f"#### {player} — model vs. actual ({pseason})")
            base_cols = ["week", "opponent_team", "projection", "actual"]
            extra_cols = [c for c in hist.columns if c not in base_cols]

            def render_hist(df):
                rows = ""
                for i, r in df.iterrows():
                    bg = "#1e1912" if i % 2 == 0 else "#2a2318"
                    if pd.notna(r["actual"]):
                        miss = r["actual"] - r["projection"]
                        mc = "#4caf72" if abs(miss) <= 15 else "#e0655a"
                        actual_cell = f'<td style="padding:8px 12px;text-align:left;color:#ede4d8;">{r["actual"]:.1f}</td>'
                        miss_cell = f'<td style="padding:8px 12px;text-align:left;color:{mc};font-weight:600;">{miss:+.1f}</td>'
                    else:
                        actual_cell = '<td style="padding:8px 12px;text-align:left;color:#8a7f70;">—</td>'
                        miss_cell = '<td style="padding:8px 12px;text-align:left;color:#8a7f70;">—</td>'
                    extra = "".join(
                        f'<td style="padding:8px 12px;text-align:left;">{r[c]:.1f}</td>'
                        if isinstance(r[c], (int, float)) and pd.notna(r[c])
                        else f'<td style="padding:8px 12px;text-align:left;">{r[c]}</td>'
                        for c in extra_cols)
                    rows += f"""<tr style="background:{bg};">
                      <td style="padding:8px 12px;text-align:left;">{int(r['week'])}</td>
                      <td style="padding:8px 12px;text-align:left;">{r['opponent_team']}</td>
                      <td style="padding:8px 12px;text-align:left;">{r['projection']:.1f}</td>
                      {actual_cell}{miss_cell}{extra}</tr>"""
                vol_nice = {"targets_roll": "Recent Tgts", "snap_roll": "Recent Snap%",
                            "carries_roll": "Recent Car", "attempts_roll": "Recent Att",
                            "touches_roll": "Recent Tch"}
                heads = ["Wk", "Opp", PROJ_LABEL, "Actual", "Miss"] + [vol_nice.get(c, c.replace("_", " ").title()) for c in extra_cols]
                header = "".join(
                    f'<th style="padding:10px 12px;text-align:left;background:#e0873a;'
                    f'color:#161310;font-weight:800;font-size:0.78rem;text-transform:uppercase;'
                    f'letter-spacing:0.04em;">{h}</th>' for h in heads)
                return f"""<div style="border-radius:10px;overflow:hidden;border:1px solid #3a2f1e;
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-top:8px;max-width:900px;">
                    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;
                    font-family:'Source Sans Pro',sans-serif;">
                    <thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"""

            st.markdown(render_hist(hist), unsafe_allow_html=True)
            graded_h = hist[hist["actual"].notna()]
            if len(graded_h) > 0:
                mae = (graded_h["projection"] - graded_h["actual"]).abs().mean()
                unit = "prob pts" if IS_PROB else ""
                st.caption(f"Avg miss for {player}: **{mae:.1f} {unit}** over {len(graded_h)} games.")

        log = betlog.load_log()
        plog = log[(log["market"] == market_key) & (log["player"] == player)] if len(log) else log
        if len(plog) > 0:
            st.markdown(f"#### Your logged picks on {player}")
            st.dataframe(plog[["season", "week", "line", "projection", "gap",
                               "side", "tier", "bet", "result_yards", "outcome"]],
                         use_container_width=True, hide_index=True)

# ============ SCORECARD ============
with tab_scorecard:
    st.subheader(f"{market_name} — Season Scorecard")

    log = betlog.load_log()
    log = log[log["market"] == market_key] if len(log) else log

    if len(log) == 0:
        st.info("No picks logged yet. Enter lines on the Board and Save to Log.")
    else:
        total = len(log)
        bet_count = int(log["bet"].sum()) if "bet" in log else 0
        graded = log[log["outcome"].notna()] if "outcome" in log else log.iloc[0:0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Logged picks", total)
        m2.metric("Actually bet", bet_count)
        m3.metric("Graded (have result)", len(graded))

        if st.button("🎯 Grade picks (pull actual results)"):
            betlog.grade_log(
                lambda s, w, p: module.actual_result(int(s), int(w), p),
                is_prob=IS_PROB)
            st.success("Graded. Refreshing...")
            st.rerun()

        st.markdown("#### Picks by tier")
        by_tier = (log["tier"].value_counts()
                   .reindex(["Max", "Strong", "Lean", "Pass"]).fillna(0).astype(int))
        st.dataframe(by_tier.rename("Count").reset_index().rename(columns={"index": "Tier"}),
                     use_container_width=False, hide_index=True)

        st.markdown("#### All logged picks")
        st.dataframe(log[["season", "week", "player", "projection", "line",
                          "gap", "confidence", "side", "tier", "bet",
                          "result_yards", "outcome"]],
                     use_container_width=True, hide_index=True)

        gr = log[log["outcome"].isin(["WIN", "LOSS"])]
        if len(gr) > 0:
            st.markdown("#### Hit rate")
            def rate(df):
                w = (df["outcome"] == "WIN").sum()
                n = len(df)
                return f"{w}/{n} ({w/n*100:.0f}%)" if n else "—"
            h1, h2 = st.columns(2)
            h1.metric("Model — all graded", rate(gr))
            bets = gr[gr["bet"] == True]
            h2.metric("Your actual bets", rate(bets) if len(bets) else "—")
            # ============ TOP PLAYS (merged across all markets) ============
with tab_top:
    st.subheader("🎯 Top Plays — all markets, ranked by confidence")
    st.caption("Pulls every pick you've saved to the log for the selected week, "
               "across all markets, ranked by tier and confidence. "
               "Enter + save picks in each market's Board first.")

    log = betlog.load_log()
    if len(log) == 0:
        st.info("No saved picks yet. Enter lines/odds on each market's Board and Save to Log.")
    else:
        # week picker (union of weeks present in the log)
        seasons_avail = sorted(log["season"].dropna().unique().tolist())
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            tseason = st.selectbox("Season", seasons_avail,
                                   index=len(seasons_avail) - 1, key="top_season")
        wk_avail = sorted(log[log["season"] == tseason]["week"].dropna().unique().tolist())
        with tc2:
            tweek = st.selectbox("Week", wk_avail, index=len(wk_avail) - 1, key="top_week")
        with tc3:
            min_tier = st.selectbox("Min tier", ["Pass", "Lean", "Strong", "Max"],
                                    index=1, key="top_tier",
                                    help="Only show plays at this tier or stronger.")

        tier_rank = {"Pass": 0, "Lean": 1, "Strong": 2, "Max": 3}
        view = log[(log["season"] == tseason) & (log["week"] == tweek)].copy()
        view["tier_rank"] = view["tier"].map(tier_rank).fillna(-1)
        view = view[view["tier_rank"] >= tier_rank[min_tier]]
        # nice market label
        key_to_name = {v: k for k, v in MARKETS.items()}
        view["market_label"] = view["market"].map(key_to_name).fillna(view["market"])
        view = view.sort_values(["tier_rank", "confidence"], ascending=False).reset_index(drop=True)

        if len(view) == 0:
            st.warning(f"No {min_tier}+ plays saved for {tseason} Week {tweek}.")
        else:
            def render_top(df):
                rows = ""
                for i, r in df.iterrows():
                    bg = "#1e1912" if i % 2 == 0 else "#2a2318"
                    tier_c = TIER_COLORS.get(r["tier"], "#8a7f70")
                    side_c = "#4caf72" if r["side"] == "OVER" else "#e0655a"
                    conf = r["confidence"]
                    cc = "#f0964a" if conf >= 85 else "#e6c14d" if conf >= 65 else "#c0b090" if conf >= 45 else "#8a7f70"
                    bet_mark = "✅" if r.get("bet") in (True, "True", "true") else ""
                    rows += f"""<tr style="background:{bg};">
                      <td style="padding:9px 12px;font-weight:800;text-transform:uppercase;color:{tier_c};">{r['tier']}</td>
                      <td style="padding:9px 12px;font-weight:800;color:{cc};">{conf:.0f}</td>
                      <td style="padding:9px 12px;font-weight:600;">{r['player']}</td>
                      <td style="padding:9px 12px;color:#c0b090;">{r['market_label']}</td>
                      <td style="padding:9px 12px;text-align:right;">{r['projection']:.1f}</td>
                      <td style="padding:9px 12px;text-align:right;">{r['line']:.1f}</td>
                      <td style="padding:9px 12px;text-align:right;font-weight:700;color:{'#4caf72' if r['gap']>0 else '#e0655a'};">{r['gap']:+.1f}</td>
                      <td style="padding:9px 12px;font-weight:700;color:{side_c};">{r['side']}</td>
                      <td style="padding:9px 12px;text-align:center;">{bet_mark}</td>
                    </tr>"""
                heads = ["Tier", "Conf", "Player", "Market", "Proj", "Line/Odds", "Gap", "Side", "Bet"]
                aligns = ["left", "left", "left", "left", "right", "right", "right", "left", "center"]
                header = "".join(
                    f'<th style="padding:11px 12px;text-align:{a};background:#e0873a;'
                    f'color:#161310;font-weight:800;font-size:0.8rem;text-transform:uppercase;'
                    f'letter-spacing:0.04em;">{h}</th>' for h, a in zip(heads, aligns))
                return f"""<div style="border-radius:10px;overflow:hidden;border:1px solid #3a2f1e;
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-top:8px;">
                    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;
                    font-family:'Source Sans Pro',sans-serif;">
                    <thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"""

            st.markdown(f"**{len(view)} play(s)** for {tseason} Week {tweek}, {min_tier}+ tier:")
            st.markdown(render_top(view), unsafe_allow_html=True)
            st.caption("Ranked by tier then confidence. Confidence is normalized per-market, "
                       "so tiers are comparable across markets. 'Line/Odds' = your entered value "
                       "(yards/catches for yardage markets, American odds for TD).")