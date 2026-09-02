import streamlit as st
import pandas as pd
import betlog
import db
import mc
ADMIN_EMAIL = "avoliotg@gmail.com"
OPAL_BANNER = ("📣 **Note from OpalScales:** Be wary of huge edges early in the season, "
               "they're often the model's early season blind spots, not real value. See the Guide for details.")

st.set_page_config(page_title="NFL Props", layout="wide")
# ---------- Login gate ----------
if "user" not in st.session_state:
    st.session_state.user = None


def _show_login():
    st.title("🔮 OpalScales")
    st.caption("Log in or sign up to continue.")
    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log In", type="primary", key="login_btn"):
            user, msg = db.sign_in(email, pw)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        email2 = st.text_input("Email", key="signup_email")
        pw2 = st.text_input("Password (8+ characters)", type="password", key="signup_pw")
        if st.button("Sign Up", type="primary", key="signup_btn"):
            if len(pw2) < 8:
                st.warning("Password must be at least 8 characters.")
            else:
                ok, msg = db.sign_up(email2, pw2)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

if st.session_state.user is None:
    _show_login()
    st.stop()
IS_ADMIN = st.session_state.user["email"] == ADMIN_EMAIL
with st.sidebar:
    st.caption(f"Logged in as {st.session_state.user['email']}")
    if st.button("Log out"):
        st.session_state.user = None
        st.rerun()

MARKETS = {
    "Receiving Yards": "receiving",
    "Receptions": "receptions",
    "Rushing Yards": "rushing",
    "QB Passing Yards": "qb_passing",
    "Anytime TD": "anytime_td",
}

from models import receiving, receptions, rushing, qb_passing, anytime_td, qb_rushing
MODULES = {
    "receiving": receiving,
    "receptions": receptions,
    "rushing": rushing,
    "qb_passing": qb_passing,
    "anytime_td": anytime_td,
}

st.markdown("<style>h1{color:#e0873a;}</style>", unsafe_allow_html=True)
st.title("🔮 OpalScales")
st.caption("An OpalScales modeling project · 🏈 Player prop projections")
if OPAL_BANNER.strip():
    st.info(OPAL_BANNER)

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

tab_labels = ["📋 Board", "📊 Scorecard", "🎯 Top Plays", "🔍 Market History", "📖 Guide"]
if IS_ADMIN:
    tab_labels.append("📥 Import")
_tabs = st.tabs(tab_labels)
tab_board, tab_scorecard, tab_top, tab_player, tab_guide = _tabs[0], _tabs[1], _tabs[2], _tabs[3], _tabs[4]

TIER_COLORS = {"Pass": "#8a7f70", "Lean": "#e6c14d",
               "Strong": "#4caf72", "Max": "#f0964a"}

HEADER_HELP = {
    "Player": "Qualifying player.",
    "Proj": "Model projection (yards or catches).",
    "Model %": "Model's probability the player scores a TD.",
    "Line": "The sportsbook over/under line.",
    "Odds": "The American odds (e.g. +150).",
    "P(over)%": "Model's probability the result lands OVER the line.",
    "Edge": "Model probability minus the vig-adjusted breakeven, in points. Positive = value.",
    "Side": "Which side the edge favors (— means no positive-edge side = pass).",
    "Tier": "Pass / Lean / Strong / Max — bigger edge = stronger.",
    "Captured": "When this line was imported.",
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
            elif c == "Edge":
                ec = "#4caf72" if val > 0 else "#e0655a" if val < 0 else "#c0b090"
                style += f"font-weight:800;color:{ec};"
                cells += f'<td style="{style}">{val:+.1f}</td>'
            elif isinstance(val, (int, float)) and pd.notna(val):
                cells += f'<td style="{style}">{val:.1f}</td>'
            else:
                if c == "Player":
                    style += "font-weight:600;"
                    parts = str(val).split(" ", 1)
                    first = parts[0]
                    last = parts[1] if len(parts) > 1 else ""
                    team = r["team"] if "team" in r and pd.notna(r.get("team")) else ""
                    tag = (f'<span style="font-size:0.7rem;font-weight:700;color:#e0873a;'
                           f'background:#2a2318;border:1px solid #3a2f1e;border-radius:4px;'
                           f'padding:1px 5px;margin-left:6px;vertical-align:middle;">{team}</span>'
                           if team else "")
                    qb_tag = (f'<span style="font-size:0.65rem;font-weight:600;color:#c0b090;'
                              f'margin-left:5px;vertical-align:middle;">🏃 QB model</span>'
                              if r.get("is_qb_model") else "")
                    name_html = f'{first}<br>{last} {tag} {qb_tag}' if last else f'{first} {tag} {qb_tag}'
                    cells += f'<td style="{style}">{name_html}</td>'
                else:
                    cells += f'<td style="{style}">{val}</td>'
        rows += f'<tr style="background:{bg};">{cells}</tr>'

    header = "".join(
        f'<th title="{HEADER_HELP.get(h,h)}" style="padding:11px 12px;text-align:{a};'
        f'background:#e0873a;color:#161310;font-weight:800;font-size:0.8rem;'
        f'text-transform:uppercase;letter-spacing:0.05em;cursor:help;'
        f'position:sticky;top:0;z-index:2;">{h}</th>'
        for h, a in zip(cols, aligns))

    return f"""<div style="border-radius:10px;overflow-x:auto;overflow-y:auto;max-height:70vh;
        -webkit-overflow-scrolling:touch;
        border:1px solid #3a2f1e;box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-top:8px;max-width:1000px;">
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
    if market_key == "rushing":
        qb_board = qb_rushing.project_week(season, week)
        if len(qb_board) > 0:
            qb_board = qb_board.copy()
            qb_board["is_qb_model"] = True
            board = board.copy()
            board["is_qb_model"] = False
            board = pd.concat([board, qb_board], ignore_index=True)

    if len(board) == 0:
        st.warning("No projections available for that week yet.")
    else:
        # figure out the market's volume column (targets/carries/attempts/touches)
        vol_cols = [c for c in board.columns if c.endswith("_roll")]

        if IS_PROB:
            st.info("✏️ Enter the sportsbook's American odds in the **Your Odds** column "
                    "(auto-filled from imported lines when available). Edge appears below.")
        else:
            st.info("✏️ Enter the **Line** and **Over/Under odds** "
                    "(auto-filled from imported lines when available). "
                    "Edge is computed against the real vig; blank odds fall back to −110.")

        board = board.copy()
        # pre-fill inputs from the imported lines pool (if any)
        pool = db.get_lines(season, week, market_key, st.session_state.user)

        def _prefill_field(player_name, field):
            entry = pool.get(db._norm_name(player_name))
            return entry.get(field) if entry else None

        vol_labels = {"targets_roll": "Recent Tgts", "snap_roll": "Recent Snap%",
                      "carries_roll": "Recent Car", "attempts_roll": "Recent Att",
                      "touches_roll": "Recent Tch"}

        colcfg = {
            "player_display_name": st.column_config.TextColumn("Player", width="medium"),
            "team": st.column_config.TextColumn("Tm", width="small"),
            "opponent_team": st.column_config.TextColumn("Opp", width="small"),
            "position": st.column_config.TextColumn("Pos", width="small"),
        }
        colcfg["projection"] = st.column_config.NumberColumn(PROJ_LABEL, format="%.1f", width="small")
        for vc in vol_cols:
            fmt = "%.0f%%" if vc == "snap_roll" else "%.1f"
            colcfg[vc] = st.column_config.NumberColumn(vol_labels.get(vc, vc), format=fmt, width="small")

        if IS_PROB:
            # TD: single odds input (over/yes price)
            board["over_odds"] = board["player_display_name"].apply(
                lambda p: _prefill_field(p, "over_odds"))
            colcfg["over_odds"] = st.column_config.NumberColumn(
                "Your Odds ✏️", format="%.0f", width="small")
            input_cols = ["over_odds"]
        else:
            # yardage/receptions: line + over odds + under odds (all editable)
            board["line"] = board["player_display_name"].apply(
                lambda p: _prefill_field(p, "line"))
            board["over_odds"] = board["player_display_name"].apply(
                lambda p: _prefill_field(p, "over_odds"))
            board["under_odds"] = board["player_display_name"].apply(
                lambda p: _prefill_field(p, "under_odds"))
            board["captured_at"] = board["player_display_name"].apply(
                lambda p: _prefill_field(p, "captured_at"))
            colcfg["line"] = st.column_config.NumberColumn("Line ✏️", format="%.1f", width="small")
            colcfg["over_odds"] = st.column_config.NumberColumn("Over ✏️", format="%.0f", width="small")
            colcfg["under_odds"] = st.column_config.NumberColumn("Under ✏️", format="%.0f", width="small")
            input_cols = ["line", "over_odds", "under_odds"]

        display_cols = (["player_display_name", "team", "opponent_team", "position",
                         "projection"] + vol_cols + input_cols)
        edited = st.data_editor(
            board[display_cols], width='stretch', hide_index=True,
            disabled=[c for c in display_cols if c not in input_cols],
            column_config=colcfg, key=f"board_editor_{market_key}")

        # rows with the key input present (line for yardage, odds for TD)
        key_input = "line" if not IS_PROB else "over_odds"
        graded = edited[edited[key_input].notna()].copy()
        if "is_qb_model" in board.columns:
            qb_flag_map = board.set_index("player_display_name")["is_qb_model"].to_dict()
            graded["is_qb_model"] = graded["player_display_name"].map(qb_flag_map)
        else:
            graded["is_qb_model"] = False
        if "captured_at" in board.columns:
            cap_map = board.set_index("player_display_name")["captured_at"].to_dict()
            graded["captured_at"] = graded["player_display_name"].map(cap_map)
        else:
            graded["captured_at"] = None

        # sanity guard (yardage only; odds have their own valid ranges)
        if not IS_PROB and len(graded) > 0:
            max_line = 500 if market_key == "qb_passing" else 150
            bad = graded[(graded["line"] < 0) | (graded["line"] > max_line)]
            if len(bad) > 0:
                names = ", ".join(f"{r['player_display_name']} ({r['line']:.0f})"
                                  for _, r in bad.iterrows())
                override = st.checkbox("☑ I've checked — grade the flagged lines anyway",
                                       key="sanity_override")
                if not override:
                    st.warning(f"⚠️ These lines look off (outside 0-{max_line}): {names}. "
                               f"Tick the box to grade anyway.")
                    graded = graded[(graded["line"] >= 0) & (graded["line"] <= max_line)]

        if len(graded) > 0:
            # ---- MC edge layer ----
            # TODO: games_played is hardcoded 0 (correct for Week 1). Generalize
            # to real current-season volume-qualifying games before Week 2.
            GAMES_PLAYED = 0

            def _row_edge(r):
                if IS_PROB:
                    # TD: model already outputs a probability; edge = model% - implied%
                    implied = module.american_to_prob(r["over_odds"])
                    if implied is None:
                        return pd.Series({"p_over": r["projection"], "edge": None,
                                          "side": "", "tier": "", "approx": False})
                    e = round(r["projection"] - implied, 1)
                    return pd.Series({
                        "p_over": r["projection"], "edge": e,
                        "side": "OVER" if e > 0 else "—",
                        "tier": mc.tier_for_edge(e), "approx": False})
                else:
                    effective_market = "qb_rushing" if r.get("is_qb_model") else market_key
                    res = mc.edge_calc(effective_market, r["projection"], r["line"],
                                       GAMES_PLAYED,
                                       over_odds=r.get("over_odds"),
                                       under_odds=r.get("under_odds"))
                    if res is None:
                        return pd.Series({"p_over": None, "edge": None,
                                          "side": "", "tier": "", "approx": False})
                    best = res["best_edge"]
                    # both sides negative → it's a pass; side is moot
                    side = res["best_side"] if best >= 0 else "—"
                    return pd.Series({
                        "p_over": res["p_over"], "edge": best,
                        "side": side, "tier": mc.tier_for_edge(best),
                        "approx": res["approx_odds"]})

            mc_cols = graded.apply(_row_edge, axis=1)
            graded = pd.concat([graded, mc_cols], axis=1)
            graded = graded[graded["edge"].notna()].copy()
            graded = graded.sort_values("edge", ascending=False).reset_index(drop=True)

            # format the import timestamp for display (compact, human-readable)
            graded["captured_display"] = pd.to_datetime(
                graded["captured_at"], errors="coerce", utc=True
            ).dt.strftime("%m/%d %I:%M%p")
            graded["captured_display"] = graded["captured_display"].fillna("—")

            if IS_PROB:
                show = graded.rename(columns={
                    "player_display_name": "Player", "projection": "Model %",
                    "over_odds": "Odds", "p_over": "P(over)%", "edge": "Edge",
                    "side": "Side", "tier": "Tier", "captured_display": "Captured"})
                cols = ["Player", "Model %", "Odds", "Edge", "Side", "Tier", "Captured"]
            else:
                show = graded.rename(columns={
                    "player_display_name": "Player", "projection": "Proj",
                    "line": "Line", "p_over": "P(over)%", "edge": "Edge",
                    "side": "Side", "tier": "Tier", "captured_display": "Captured"})
                cols = ["Player", "Proj", "Line", "P(over)%", "Edge", "Side", "Tier", "Captured"]
            aligns = ["left"] * len(cols)

            # summary stats (market-agnostic: edge/tier mean the same everywhere)
            n_lines = len(graded)
            n_positive = int((graded["edge"] > 0).sum())
            tier_counts = graded["tier"].value_counts()
            n_max = int(tier_counts.get("Max", 0))
            n_strong = int(tier_counts.get("Strong", 0))
            n_lean = int(tier_counts.get("Lean", 0))
            st.markdown(
                f"**{n_lines}** lines entered · **{n_positive}** positive edges · "
                f"{n_max} Max, {n_strong} Strong, {n_lean} Lean")

            st.markdown("### Your entered lines")
            if graded["approx"].any():
                st.caption("⚠️ Rows without imported odds use a −110 assumption "
                           "(edge is approximate).")
            st.markdown(render_html_table(show, cols, aligns), unsafe_allow_html=True)
            st.caption("Hover any column header (ⓘ) for what it means. "
                       "Edge = model probability minus the vig-adjusted breakeven, in points. "
                       "Captured = when that line was imported.")

            # ---- save to log ----
            save_cols = ["player_display_name", "projection", "line",
                         "over_odds", "under_odds", "edge", "p_over",
                         "side", "tier"]
            if IS_PROB:
                # TD has no 'line'/'under_odds'; fill them so the schema is uniform
                graded["line"] = None
                graded["under_odds"] = None
            grid = graded[save_cols].copy()
            grid["bet"] = False
            logged_view = st.data_editor(
                grid, width='stretch', hide_index=True,
                disabled=[c for c in save_cols],
                column_config={
                    "player_display_name": st.column_config.TextColumn("Player", width="medium"),
                    "bet": st.column_config.CheckboxColumn("Bet?", width="small",
                        help="Check if you actually placed this bet."),
                }, key=f"graded_editor_{market_key}")

            if st.button("💾 Save to Log", type="primary"):
                entries = logged_view.copy()
                entries["logged_at"] = betlog.now_stamp()
                entries["market"] = market_key
                entries["season"] = season
                entries["week"] = week
                entries["result_yards"] = None
                entries["outcome"] = None
                entries = entries.rename(columns={"player_display_name": "player"})
                entries = entries[betlog.COLUMNS]
                betlog.append_entries(entries, st.session_state.user)
                st.success(f"Saved {len(entries)} pick(s) to the log.")

# ============ PLAYER ============
with tab_player:
    st.subheader(f"{market_name} — Market History")

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
                    f'letter-spacing:0.04em;position:sticky;top:0;z-index:2;">{h}</th>' for h in heads)
                return f"""<div style="border-radius:10px;overflow-x:auto;overflow-y:auto;max-height:70vh;
                    -webkit-overflow-scrolling:touch;border:1px solid #3a2f1e;
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

        log = betlog.load_log(st.session_state.user)
        plog = log[(log["market"] == market_key) & (log["player"] == player)] if len(log) else log
        if len(plog) > 0:
            st.markdown(f"#### Your logged picks on {player}")
            st.dataframe(plog[["season", "week", "line", "projection", "p_over",
                               "edge", "side", "tier", "bet", "result_yards", "outcome"]],
                         width='stretch', hide_index=True)

# ============ SCORECARD ============
with tab_scorecard:
    st.subheader(f"{market_name} — Season Scorecard")

    log = betlog.load_log(st.session_state.user)
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
                is_prob=IS_PROB, user=st.session_state.user)
            st.success("Graded. Refreshing...")
            st.rerun()

        st.markdown("#### Picks by tier")
        by_tier = (log["tier"].value_counts()
                   .reindex(["Max", "Strong", "Lean", "Pass"]).fillna(0).astype(int))
        st.dataframe(by_tier.rename("Count").reset_index().rename(columns={"index": "Tier"}),
                     width='stretch', hide_index=True)

        st.markdown("#### All logged picks")
        st.dataframe(log[["logged_at", "season", "week", "player", "projection", "line",
                          "edge", "p_over", "side", "tier", "bet",
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
    st.subheader("🎯 Top Plays — all markets, ranked by edge")
    st.caption("Pulls every pick you've saved to the log for the selected week, "
               "across all markets, ranked by edge (model probability minus the "
               "vig-adjusted breakeven). Enter + save picks in each market's Board first.")

    log = betlog.load_log(st.session_state.user)
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
        view = view.sort_values(["tier_rank", "edge"], ascending=False).reset_index(drop=True)

        if len(view) == 0:
            st.warning(f"No {min_tier}+ plays saved for {tseason} Week {tweek}.")
        else:
            def render_top(df):
                df = df.copy()
                df["logged_disp"] = pd.to_datetime(
                    df["logged_at"], errors="coerce"
                ).dt.strftime("%m/%d %I:%M%p")
                df["logged_disp"] = df["logged_disp"].fillna("—")

                rows = ""
                for i, r in df.iterrows():
                    bg = "#1e1912" if i % 2 == 0 else "#2a2318"
                    tier_c = TIER_COLORS.get(r["tier"], "#8a7f70")
                    side_c = "#4caf72" if r["side"] == "OVER" else "#e0655a" if r["side"] == "UNDER" else "#c0b090"
                    edge_v = r["edge"] if pd.notna(r["edge"]) else 0
                    edge_c = "#4caf72" if edge_v > 0 else "#e0655a" if edge_v < 0 else "#c0b090"
                    bet_mark = "✅" if r.get("bet") in (True, "True", "true") else ""
                    line_disp = f"{r['line']:.1f}" if pd.notna(r["line"]) else "—"
                    rows += f"""<tr style="background:{bg};">
                      <td style="padding:9px 12px;font-weight:800;text-transform:uppercase;color:{tier_c};">{r['tier']}</td>
                      <td style="padding:9px 12px;text-align:right;font-weight:800;color:{edge_c};">{edge_v:+.1f}</td>
                      <td style="padding:9px 12px;font-weight:600;">{r['player']}</td>
                      <td style="padding:9px 12px;color:#c0b090;">{r['market_label']}</td>
                      <td style="padding:9px 12px;text-align:right;">{r['projection']:.1f}</td>
                      <td style="padding:9px 12px;text-align:right;">{line_disp}</td>
                      <td style="padding:9px 12px;font-weight:700;color:{side_c};">{r['side']}</td>
                      <td style="padding:9px 12px;text-align:center;">{bet_mark}</td>
                      <td style="padding:9px 12px;text-align:right;color:#8a7f70;font-size:0.8rem;">{r['logged_disp']}</td>
                    </tr>"""
                heads = ["Tier", "Edge", "Player", "Market", "Proj", "Line", "Side", "Bet", "Logged"]
                aligns = ["left", "right", "left", "left", "right", "right", "left", "center", "right"]
                header = "".join(
                    f'<th style="padding:10px 12px;text-align:left;background:#e0873a;'
                    f'color:#161310;font-weight:800;font-size:0.78rem;text-transform:uppercase;'
                    f'letter-spacing:0.04em;">{h}</th>' for h in heads)
                return f"""<div style="border-radius:10px;overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid #3a2f1e;
                    box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-top:8px;max-width:900px;">
                    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;
                    font-family:'Source Sans Pro',sans-serif;">
                    <thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"""

            st.markdown(f"**{len(view)} play(s)** for {tseason} Week {tweek}, {min_tier}+ tier:")
            st.markdown(render_top(view), unsafe_allow_html=True)
            st.caption("Ranked by tier then edge. Edge (model probability minus the "
                       "vig-adjusted breakeven, in points) is the same unit across all "
                       "markets, so plays are directly comparable.")
                       # ============ GUIDE ============
with tab_guide:
    st.subheader("📖 OpalScales Guide")

    st.markdown("""
### 🔮 What OpalScales Is

OpalScales projects player prop outcomes using statistical models built on years of data,
then compares those projections to the sportsbook line to find where there might be an edge.
It currently covers five NFL markets: receiving yards, receptions, rushing yards, QB passing
yards, and anytime TD. The same approach can extend to other sports and markets down the road.

Think of it as a second opinion, not a crystal ball. It does the math so you can bring the
football brain.

---

### How to use it

1. **Pick a market** from the top dropdown.
2. **Pick the season and week.**
3. On the **Board**, enter the sportsbook's numbers. Lines auto-fill if they've been imported,
   but you can type or adjust them yourself:
   - Yardage and receptions markets: enter the **line** plus the **over and under odds**.
   - Anytime TD: enter the **American odds** (e.g. 150 for +150, or -200).
4. The app computes the **edge** for you, tells you which side it favors, and sorts it into a tier.
5. **Save picks to the Log**, tick the ones you actually bet, and the **Scorecard** tracks how
   they turn out over time. The **Player** tab shows any player's projection history versus real
   results, and **Top Plays** ranks your best edges across every market at once.

A note on your edits: anything you type on the Board lives only in your session. It never changes
the shared line data, so feel free to test alternate lines or numbers from a different book. Your
tinkering stays yours.

---

### How to read the columns

- **Proj**: the model's projection. Units depend on the market: *yards* (receiving, rushing,
  passing), *catches* (receptions), or a *probability %* (anytime TD).
- **Line**: the sportsbook's over/under number for that prop.
- **Over / Under**: the American odds on each side of the line. These matter more than they look,
  because the price is how the book takes its cut (see the edge example below).
- **P(over)%**: the model's estimated probability the result lands over the line. This is the
  honest "how likely" number, already accounting for how much uncertainty there is early in the
  season.
- **Edge**: the heart of the whole tool. It is your model probability minus the break-even
  probability the odds require. Positive means the model thinks the bet is worth more than its
  price. Negative means the price is too steep, even if the projection "agrees" with you. Measured
  in percentage points, and it means the same thing across every market, so a +4 on a rushing prop
  is directly comparable to a +4 on a TD.
- **Side**: which side the edge favors (Over or Under). A dash means neither side clears break-even,
  so it is a pass.
- **Tier**: the plain-English verdict. **Pass** (skip it), **Lean** (small edge), **Strong** (solid
  edge), **Max** (biggest edge). Bigger edge, stronger tier.

---

### A real example: why "the projection disagrees" is not enough

Say the board shows a receiver projected for 7.1 receptions, and the line is 7.5, priced at +116
over and -154 under. (These numbers are pulled from a real prop.)

Your gut says: model says 7.1, line says 7.5, so bet the under. That gut is a trap, and the tool
is built to catch it.

The model does lean under, giving him about a 55% chance to land below 7.5. But look at the price.
The under is -154, which means you need to win about 61% of the time just to break even after the
book's cut. The model only gives you 55%. So the under is likely, but not likely *enough* to beat
what you are paying for it. Both sides come back with a negative edge, so the honest call is a pass.

The lesson: a projection that "disagrees" with the line is not automatically a bet. Most props,
most of the time, should be a pass. That is what an efficient market looks like. The edge number
tells you the rare times the line is actually soft enough to be worth it. Chase the edge, not the
gut feeling.

---

### The season arc: trust it more as the year goes on

The model gets stronger as the season progresses, and it is honest about that.

- **Weeks 1 to 3**: weakest. There is little current-season form yet, so it leans on last season's
  data and wraps every projection in extra uncertainty. Edges will look smaller and more cautious
  on purpose. Bet light, if at all.
- **Midseason onward**: strongest. It is working off rich current-season data, and the extra
  caution fades away. Trust it most here.

That early-season caution is not a bug, it is the point. When the model cannot see clearly, it
tells you so by pulling its probabilities toward a coin flip. It earns your trust as real data
piles up.

---

### Beware of Week 1

The very start of the season is the trickiest stretch, so a few things to know:

- **Rookies will not appear in Week 1.** They have no NFL history to project from. They show up
  once they have banked a game or two of real data, and stay volatile until they have a few under
  their belt.
- **Players returning from a lost season may not appear either.** If someone missed all of last
  year, there is no recent data to build on, so the model sits them out rather than guess.
- **Players who changed teams** are projected on their new team and matchup, but their underlying
  form still comes from last season, so read them with a little extra skepticism early.
- **Stable veterans in the same role** are your safest ground in Week 1. Last year's data transfers
  cleanly for them, so lean on those and use your judgment on the rest.

---

### Honest limitations (please read this part)

- Projections are **guides, not guarantees.** Football is noisy, and a perfect matchup can still
  bust. No model captures that.
- The model is near the **ceiling of what pre-game data can predict.** The rest is genuine
  randomness. Anyone claiming certainty is selling something.
- It does not see **injuries, benchings, weather surprises, or coverage matchups** in real time.
  That context is your job. Bring your football brain and layer it on top.
- Markets are efficient. The edge, if it exists, is small and lives in the details, which is
  exactly why the tool measures it so carefully.

---

### The OpalScales approach

Small edges. Obscure players over stars, because the market is sharpest on the names everyone
watches. Discipline over hype. Track everything honestly, and let the results tell you what is
actually working.

*Bet responsibly. This is a tool for informed decisions, not financial advice.*
""")
    # ============ IMPORT ============
if IS_ADMIN:
    with _tabs[5]:
        st.subheader("📥 Import Lines from CSV")
        st.caption("Paste CSV from the extraction prompt. Columns: "
                   "player, market, line, over_odds, under_odds. "
                   "Imports are cumulative — re-importing a player updates their line.")

        if "import_result" in st.session_state:
            r = st.session_state.pop("import_result")
            st.success(f"✅ Imported {r['imported']} line(s) for {r['season']} Week {r['week']}.")
            if r["by_market"]:
                breakdown = " · ".join(f"{k}: {v}" for k, v in r["by_market"].items())
                st.caption(f"By market — {breakdown}")
            if r["bad_market"]:
                st.warning("Unrecognized market(s) skipped: "
                           + ", ".join(sorted(set(r["bad_market"]))))

        ic1, ic2 = st.columns(2)
        with ic1:
            imp_season = st.number_input("Season", min_value=2020, max_value=2030,
                                         value=2025, step=1, key="imp_season")
        with ic2:
            imp_week = st.number_input("Week", min_value=1, max_value=25,
                                       value=1, step=1, key="imp_week")

        csv_text = st.text_area(
            "Paste CSV here", height=200, key="imp_csv",
            placeholder="player,market,line,over_odds,under_odds\n"
                        "Ja'Marr Chase,receiving_yards,74.5,-115,-105\n"
                        "...")

        if st.button("📥 Import", type="primary", key="imp_btn"):
            if not csv_text.strip():
                st.warning("Paste some CSV first.")
            else:
                import io, csv as _csv
                try:
                    reader = _csv.DictReader(io.StringIO(csv_text.strip()))
                    rows = [dict(r) for r in reader]
                except Exception as e:
                    rows = None
                    st.error(f"Couldn't parse CSV: {e}")
                if rows is not None:
                    if len(rows) == 0:
                        st.warning("No data rows found (need a header row + at least one line).")
                    else:
                        result = db.import_lines(rows, imp_season, imp_week, st.session_state.user)
                        st.session_state["import_result"] = {
                            "imported": result["imported"],
                            "season": imp_season, "week": imp_week,
                            "by_market": result["by_market"],
                            "bad_market": result["bad_market"],
                        }
                        st.cache_data.clear()
                        st.rerun()