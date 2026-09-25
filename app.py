import streamlit as st
import pandas as pd
import betlog
# PERFORMANCE, September 25. Streamlit re-runs this whole file on every
# widget interaction, and every Supabase read used to repeat. Two of them
# sat inside per-market LOOPS, so one widget change cost up to twelve
# network round trips. app_cache wraps those reads with a TTL plus a
# version token that writes bump, so re-runs are free and a write still
# invalidates. See app_cache.py for the measurements.
import app_cache
import rules
from models import data_utils
import db
import mc
import mc_pricing
import devig


def _default_season():
    """NFL seasons are labelled by their starting year and run Sep-Feb, so
    January and February belong to the prior year's season."""
    from datetime import date
    t = date.today()
    return t.year - 1 if t.month <= 2 else t.year

def _login_backdrop(path="assets/opal_banner.jpg", top=0.38, bottom=0.55):
    """Set the login page background to the opal image, dimmed for legibility.

    Streamlit has no background-image API, so this injects CSS and inlines the
    file as base64. The image is compressed first because it is embedded in the
    page on every load. A dark gradient sits over it so the form stays readable
    regardless of what the artwork is doing underneath.

    Targets .stApp, which has been a stable selector. If a Streamlit upgrade
    ever breaks this, the page still works and just loses the backdrop.
    """
    import base64
    import os
    if not os.path.exists(path):
        return
    try:
        b64 = base64.b64encode(open(path, "rb").read()).decode()
    except Exception:
        return
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(14,12,20,{top}), rgba(14,12,20,{bottom})),
                url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            /* 30% keeps the orb in frame on narrow screens; `fixed` breaks on
               iOS Safari, which sizes the background to the viewport rather
               than the document and pushes the orb out of view. */
            background-position: 30% center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

ADMIN_EMAIL = "avoliotg@gmail.com"
OPAL_BANNER = ("📣 **Note from OpalScales:** Be wary of huge edges early in the season, "
               "they're often the model's early season blind spots, not real value. See the Guide for details.")

WELCOME_BANNER = ("👋 **New here?** There's a lot of data on this page. Head to the "
                   "**📖 Guide** tab first, it explains what everything means and how to use it.")

st.set_page_config(page_title="NFL Props", layout="wide")
# ---------- Login gate ----------
if "user" not in st.session_state:
    st.session_state.user = None


def _show_login():
    _login_backdrop()
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
# Same artwork behind the app, dimmed far harder than the login page. The
# interior is dense data and the tier / toward-away colours carry meaning, so
# this needs to be a texture rather than a picture.
_login_backdrop(top=0.75, bottom=0.85)
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
if WELCOME_BANNER.strip():
    st.info(WELCOME_BANNER)

st.divider()

tab_labels = ["📋 Board", "📊 Scorecard", "🎯 Top Plays", "🔍 Market History",
              "📈 Line Movement & Side Picker", "📖 Guide"]
if IS_ADMIN:
    tab_labels.append("📥 Import")
    tab_labels.append("📤 Export")
_tabs = st.tabs(tab_labels)
tab_board, tab_scorecard, tab_top, tab_player, tab_movement, tab_guide = _tabs[0], _tabs[1], _tabs[2], _tabs[3], _tabs[4], _tabs[5]

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
    "Tier": "Pass / Lean / Strong / Max. A bigger edge is a stronger tier.",
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
                # A missing edge is a real state, not an error: a
                # reference-only row carries no edge by design. Guard
                # here as well as in the column selection, because a
                # single such row mixed into an otherwise-priced
                # market would not set reference_only_market and would
                # crash the entire board.
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    style += "color:#8a7f70;"
                    cells += f'<td style="{style}">—</td>'
                else:
                    ec = ("#4caf72" if val > 0 else
                          "#e0655a" if val < 0 else "#c0b090")
                    style += f"font-weight:800;color:{ec};"
                    cells += f'<td style="{style}">{val:+.1f}</td>'
            elif isinstance(val, (int, float)) and pd.notna(val):
                cells += f'<td style="{style}">{val:.1f}</td>'
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                # A numeric column with no value. This previously
                # fell through to the string branch and printed the
                # literal "None" or "nan".
                style += "color:#8a7f70;"
                cells += f'<td style="{style}">—</td>'
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
    market_name = st.selectbox("Market", list(MARKETS.keys()),
                               help="Which prop market to project.", key="board_market")
    market_key = MARKETS[market_name]
    module = MODULES[market_key]
    IS_PROB = getattr(module, "IS_PROBABILITY", False)
    if IS_PROB:
        PROJ_LABEL = "Proj TD%"
    elif market_key == "receptions":
        PROJ_LABEL = "Proj Catches"
    else:
        PROJ_LABEL = "Proj Yds"

    # ... rest of the existing Board tab code continues below, unchanged ...

with tab_board:
    st.subheader(f"{market_name} · Board")

    c1, c2 = st.columns(2)
    with c1:
        season = st.selectbox("Season", module.available_seasons(),
                              index=len(module.available_seasons()) - 1,
                              help="NFL season to view.")
    with c2:
        weeks = module.available_weeks(season)
        week = st.selectbox("Week", weeks,
                            index=data_utils.default_week_index(weeks, season),
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
        pool = app_cache.get_lines(season, week, market_key,
                                   st.session_state.user)

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
                override = st.checkbox("☑ I have checked. Grade the flagged lines anyway",
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
                    # A probability market can declare its own verdict
                    # unsafe. J1's reference-only path keys off BETA being
                    # clipped to zero, and a probability market has no beta,
                    # so without this there is nothing to stop the board
                    # recommending Max on a miscalibrated row. anytime_td set
                    # CALIBRATION_SUSPENDED on September 25 after three
                    # defects were found inflating low-volume players.
                    if getattr(module, "CALIBRATION_SUSPENDED", False):
                        # Carry the IMPLIED probability even though the
                        # verdict is withheld, so the comparison can be made
                        # by eye without the app asserting one. Note it is
                        # VIG-INCLUSIVE: a one-sided market cannot be
                        # devigged (devig.py refuses it), so this overstates
                        # the true probability, and the model reading lower
                        # is expected rather than a disagreement.
                        return pd.Series({"p_over": r["projection"],
                                          "edge": None, "side": "",
                                          "tier": "", "approx": False,
                                          "reference_only": True,
                                          "implied": module.american_to_prob(
                                              r["over_odds"])})
                    # TD: model already outputs a probability; edge = model% - implied%
                    implied = module.american_to_prob(r["over_odds"])
                    if implied is None:
                        return pd.Series({"p_over": r["projection"], "edge": None,
                                          "side": "", "tier": "", "approx": False,
                                          "implied": None})
                    e = round(r["projection"] - implied, 1)
                    return pd.Series({
                        "p_over": r["projection"], "edge": e,
                        "implied": implied,
                        "side": "OVER" if e > 0 else "—",
                        "tier": mc.tier_for_edge(e), "approx": False})
                else:
                    effective_market = "qb_rushing" if r.get("is_qb_model") else market_key
                    # BELT AND BRACES. mc.edge_calc sets reference_only from
                    # its own logic, which lives in mc.py. Consulting
                    # mc_pricing.has_model_signal here as well means the
                    # single beta in BLEND is AUTHORITATIVE: retiring a market
                    # by setting its beta to 0.0 suppresses the verdict even
                    # if mc.py's own check ever drifts. Receptions was retired
                    # this way on 2026-09-25.
                    if not mc_pricing.has_model_signal(effective_market):
                        _p = mc.prob_over(effective_market, r["projection"],
                                          r["line"], GAMES_PLAYED)
                        # With beta clipped to zero the blended mean is just
                        # line + alpha, so the model's P(over) depends ONLY on
                        # the LINE and is identical for every player. It looked
                        # like information and was not. The market's DEVIGGED
                        # probability is player-specific (sd 0.0659 across 277
                        # distinct values on receptions) and is the best
                        # calibrated number available for these props: book_p
                        # coefficient +1.0890, bin biases -0.008 to +0.043.
                        # The whole conclusion of 2026-09-25 is that the PRICE
                        # is the sharp quantity, so that is what to show.
                        _d = devig.devig_two_sided(r.get("over_odds"),
                                                   r.get("under_odds"),
                                                   "additive")
                        return pd.Series({"p_over": _p, "edge": None,
                                          "side": "", "tier": "",
                                          "approx": False,
                                          "reference_only": True,
                                          "mkt_p": (_d["p_over"]
                                                    if _d["valid"] else None)})
                    res = mc.edge_calc(effective_market, r["projection"], r["line"],
                                       GAMES_PLAYED,
                                       over_odds=r.get("over_odds"),
                                       under_odds=r.get("under_odds"))
                    if res is None:
                        return pd.Series({"p_over": None, "edge": None,
                                          "side": "", "tier": "", "approx": False,
                                          "reference_only": False})
                    # REFERENCE ONLY: beta is clipped to zero for this market,
                    # so the projection does not enter the price and the model
                    # makes no player-specific claim. Show the probability,
                    # withhold the verdict. Plan item J1 in code.
                    if res.get("reference_only"):
                        return pd.Series({
                            "p_over": res["p_over"], "edge": None,
                            "side": "", "tier": "",
                            "approx": res["approx_odds"],
                            "reference_only": True})
                    best = res["best_edge"]
                    # both sides negative → it's a pass; side is moot
                    side = res["best_side"] if best >= 0 else "—"
                    return pd.Series({
                        "p_over": res["p_over"], "edge": best,
                        "side": side, "tier": mc.tier_for_edge(best),
                        "approx": res["approx_odds"],
                        "reference_only": False})

            mc_cols = graded.apply(_row_edge, axis=1)
            graded = pd.concat([graded, mc_cols], axis=1)

            # Drop rows we could not price at all, but KEEP reference-only
            # rows: those are priced fine, they just carry no edge. The old
            # unconditional edge.notna() filter would have emptied the board
            # for every market except receptions.
            if "reference_only" not in graded.columns:
                graded["reference_only"] = False
            graded["reference_only"] = graded["reference_only"].fillna(False)
            is_ref = graded["reference_only"].astype(bool)
            graded = graded[graded["edge"].notna()
                            | (is_ref & graded["p_over"].notna())].copy()
            reference_only_market = bool(len(graded) > 0
                                         and graded["reference_only"].all())
            # Sorting a reference-only board by the model's p_over is sorting
            # by the line, since that is all it now depends on. The market's
            # devigged probability is the informative ordering.
            _sort_by = "edge"
            if reference_only_market:
                _sort_by = ("mkt_p" if "mkt_p" in graded.columns
                            and graded["mkt_p"].notna().any() else "p_over")
            graded = graded.sort_values(
                _sort_by, ascending=False).reset_index(drop=True)

            # format the import timestamp for display (compact, human-readable)
            graded["captured_display"] = pd.to_datetime(
                graded["captured_at"], errors="coerce", utc=True
            ).dt.strftime("%m/%d %I:%M%p")
            graded["captured_display"] = graded["captured_display"].fillna("—")

            # PERCENT SCALING. mc returns P(over) on 0-1 while the column has
            # always been labelled "P(over)%", so a 50 percent probability
            # displayed as "0.5". Fixed here rather than in mc, because the
            # saved log and the downstream calculations expect 0-1.
            # anytime_td is excluded: its projection is already a percentage.
            if not IS_PROB:
                for _src, _dst in (("p_over", "p_over_pct"),
                                   ("mkt_p", "mkt_p_pct")):
                    if _src in graded.columns:
                        graded[_dst] = pd.to_numeric(
                            graded[_src], errors="coerce") * 100.0

            if IS_PROB:
                show = graded.rename(columns={
                    "player_display_name": "Player", "projection": "Model %",
                    "over_odds": "Odds", "implied": "Implied %",
                    "p_over": "P(over)%", "edge": "Edge",
                    "side": "Side", "tier": "Tier", "captured_display": "Captured"})
                if reference_only_market:
                    # Same reasoning as the non-probability branch below: an
                    # absent column says the market makes no claim, a blank
                    # one looks broken. This branch built Edge, Side and Tier
                    # unconditionally, so suppressing anytime_td set edge to
                    # None and then asked render_html_table to colour it,
                    # raising TypeError: '>' not supported between NoneType
                    # and int.
                    cols = ["Player", "Model %", "Odds", "Implied %",
                            "Captured"]
                else:
                    cols = ["Player", "Model %", "Odds", "Implied %", "Edge",
                            "Side", "Tier", "Captured"]
            else:
                show = graded.rename(columns={
                    "player_display_name": "Player", "projection": "Proj",
                    "line": "Line", "p_over_pct": "P(over)%",
                    "mkt_p_pct": "Mkt P(over)%", "edge": "Edge",
                    "side": "Side", "tier": "Tier", "captured_display": "Captured"})
                if reference_only_market:
                    # No Edge, Side or Tier: a blank column invites the reader
                    # to wonder what is missing, an absent one says the market
                    # makes no such claim.
                    #
                    # The MODEL's P(over) is dropped too, because with beta at
                    # zero it is line + alpha for every player and therefore
                    # identical down the whole column. The MARKET's devigged
                    # probability replaces it: player-specific and well
                    # calibrated.
                    cols = ["Player", "Proj", "Line", "Mkt P(over)%",
                            "Captured"]
                else:
                    cols = ["Player", "Proj", "Line", "P(over)%", "Edge",
                            "Side", "Tier", "Captured"]
            aligns = ["left"] * len(cols)

            # summary stats (market-agnostic: edge/tier mean the same everywhere)
            n_lines = len(graded)
            if reference_only_market:
                st.markdown(
                    f"**{n_lines}** lines entered · **reference only**, no edge "
                    f"shown for this market")
                st.info(
                    getattr(module, "SUSPENSION_REASON", None) or
                    "This market is shown for reference: the projection does "
                    "not enter the price, so no edge is claimed. The "
                    "probability is the line plus a measured skew correction, "
                    "priced through the market's own distribution. "
                    "NO market currently carries a model verdict. Receptions "
                    "was the last one and was retired on 2026-09-25: its beta "
                    "of 0.2261 was real but measured against the LINE, and "
                    "receptions is 93 percent flat on the line because "
                    "FanDuel moves it through the ODDS. Tested against the "
                    "devigged price, the shipped probability lost on log loss "
                    "and returned an encompassing t of +0.60. Side-picking "
                    "now lives in the Line Movement & Side Picker tab, where "
                    "two pricing-bias rules with measured out-of-sample "
                    "support fire on specific props.")
            else:
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
            if reference_only_market:
                cap = ("Hover any column header (ⓘ) for what it means. "
                       "No Edge, Side or Tier is shown for this market. "
                       "Captured = when that line was imported.")
                if IS_PROB:
                    cap += (" Implied % comes straight from the odds and "
                            "INCLUDES the vig, so it overstates the true "
                            "probability. A one-sided market cannot be "
                            "devigged, so the model reading lower than "
                            "Implied % is expected and is not by itself a "
                            "disagreement.")
                st.caption(cap)
            else:
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
    market_name = st.selectbox("Market", list(MARKETS.keys()),
                               help="Which prop market to project.", key="history_market")
    market_key = MARKETS[market_name]
    module = MODULES[market_key]
    IS_PROB = getattr(module, "IS_PROBABILITY", False)
    if IS_PROB:
        PROJ_LABEL = "Proj TD%"
    elif market_key == "receptions":
        PROJ_LABEL = "Proj Catches"
    else:
        PROJ_LABEL = "Proj Yds"

    # ... rest of the existing Market History tab code continues below, unchanged ...

with tab_player:
    st.subheader(f"{market_name} · Market History")

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
            st.markdown(f"#### {player} · model vs. actual ({pseason})")
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
    market_name = st.selectbox("Market", list(MARKETS.keys()),
                               help="Which prop market to project.", key="scorecard_market")
    market_key = MARKETS[market_name]
    module = MODULES[market_key]
    IS_PROB = getattr(module, "IS_PROBABILITY", False)
    if IS_PROB:
        PROJ_LABEL = "Proj TD%"
    elif market_key == "receptions":
        PROJ_LABEL = "Proj Catches"
    else:
        PROJ_LABEL = "Proj Yds"

    # ... rest of the existing Scorecard tab code continues below, unchanged ...

with tab_scorecard:
    st.subheader(f"{market_name} · Season Scorecard")

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

        if st.button("🎯 Grade picks (all markets)"):
            # One lookup per market, so each row grades against its own model.
            grade_lookups = {
                mk: (lambda m: lambda s, w, p: m.actual_result(int(s), int(w), p))(mod)
                for mk, mod in MODULES.items() if hasattr(mod, "actual_result")
            }
            if hasattr(qb_rushing, "actual_result"):
                grade_lookups["qb_rushing"] = (
                    lambda s, w, p: qb_rushing.actual_result(int(s), int(w), p))
            betlog.grade_log(None, user=st.session_state.user,
                             lookups=grade_lookups,
                             prob_markets=("anytime_td",))
            st.success("Graded. Refreshing...")
            st.rerun()

        st.markdown("#### Picks by tier")
        by_tier = (log["tier"].value_counts()
                   .reindex(["Max", "Strong", "Lean", "Pass"]).fillna(0).astype(int))
        st.dataframe(by_tier.rename("Count").reset_index().rename(columns={"index": "Tier"}),
                     width='stretch', hide_index=True)

        log_display = log.copy()
        log_display["logged_at"] = pd.to_datetime(
            log_display["logged_at"], errors="coerce", utc=True
        ).dt.strftime("%m/%d %I:%M%p")

        st.markdown("#### All logged picks")
        st.dataframe(log_display[["logged_at", "season", "week", "player", "projection", "line",
                          "edge", "p_over", "side", "tier", "bet",
                          "result_yards", "outcome"]],
                     width='stretch', hide_index=True)

        gr = log[log["outcome"].isin(["WIN", "LOSS"])]
        if len(gr) > 0:
            st.markdown("#### Hit rate")
            def rate(df):
                w = (df["outcome"] == "WIN").sum()
                n = len(df)
                return f"{w}/{n} ({w/n*100:.0f}%)" if n else "—"
            h1, h2 = st.columns(2)
            h1.metric("Model, all graded", rate(gr))
            bets = gr[gr["bet"] == True]
            h2.metric("Your actual bets", rate(bets) if len(bets) else "—")
            # ============ TOP PLAYS (merged across all markets) ============
with tab_top:
    st.subheader("🎯 Top Plays · all markets, ranked by edge")
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


# ============ LINE MOVEMENT ============
with tab_movement:
    st.subheader("📈 Line Movement & Side Picker")
    st.caption("Every captured snapshot for each player, across all markets. "
               "🟢 toward = the line moved toward the model's read (market agreeing). "
               "🔴 away = it moved against the model (be more skeptical). "
               "Sparklines need 3+ snapshots to render. Check **Bet?** and hit Save to log picks.")
    st.caption("The **Rule** column flags props where a rule with measured "
               "out-of-sample support fires, and on which side. Two rules are "
               "live and both say UNDER; they are PRICING BIASES, not model "
               "verdicts, so a rule can fire where the model has no opinion. "
               "The column is absent for markets no rule covers.")

    lm_season = st.selectbox("Season", module.available_seasons(),
                             index=len(module.available_seasons()) - 1, key="lm_season")
    lm_weeks = module.available_weeks(lm_season)
    lm_week = st.selectbox("Week", lm_weeks,
                           index=data_utils.default_week_index(lm_weeks, lm_season),
                           key="lm_week")

    TIER_EMOJI = {"Pass": "⚪ Pass", "Lean": "🟡 Lean",
                  "Strong": "🟢 Strong", "Max": "🔥 Max"}
    TA_EMOJI = {"toward": "🟢 toward", "away": "🔴 away", "flat": "⚪ flat"}

    import game_export
    lm_games = game_export.load_games(lm_season, lm_week)
    if len(lm_games) == 0:
        st.caption("No schedule found for that week, showing all players.")
        lm_picked, lm_filter_on = [], False
    else:
        lm_slot_opts = [s for s in game_export.SLOT_ORDER
                        if s in set(lm_games["slot"])]
        lm_slots = st.multiselect("Kickoff slots", lm_slot_opts,
                                  default=lm_slot_opts, key="lm_slots")
        lm_avail = lm_games[lm_games["slot"].isin(lm_slots)]
        lm_lab = {f"{r['matchup']}  ({r['slot']})": r["matchup"]
                  for _, r in lm_avail.iterrows()}
        lm_picked_labels = st.multiselect(
            f"Games ({len(lm_lab)} in these slots)", list(lm_lab),
            default=list(lm_lab), key="lm_games_pick")
        lm_picked = [lm_lab[l] for l in lm_picked_labels]
        lm_filter_on = True
        st.caption("Filtering applies to every market below. Rows whose player "
                   "could not be matched to a team are hidden while filtering.")

    lm_rule_hits, lm_rule_errors = [], []

    for mkt_key, mkt_label in MARKETS.items():
        st.markdown(f"#### {mkt_key}")
        mv = app_cache.get_line_movement(lm_season, lm_week, mkt_label,
                                         st.session_state.user)

        # ---- side-picker rules for this market ----
        # rules.py is the decision layer: it consumes a prop's market, line
        # and projection and reports whether a rule with measured
        # out-of-sample support fires. It is separate from mc_pricing on
        # purpose, because both live rules are PRICING BIASES and rushing's
        # beta is -0.041, so routing them through the pricing layer would
        # invent a projection-based story for an effect that has none.
        rule_side = {}
        if len(mv):
            try:
                _props = pd.DataFrame({
                    "market": mkt_label,
                    "player": mv["player"],
                    # raw_line is the actual line; latest_line is an implied
                    # probability for anytime_td, which no rule covers anyway
                    "line": mv["raw_line"] if "raw_line" in mv.columns
                            else mv["latest_line"],
                    "season": lm_season,
                    "week": lm_week,
                    "projection": mv["raw_projection"]
                                  if "raw_projection" in mv.columns else None,
                })
                _fired, _rep = rules.evaluate(_props)
                for _, _fr in _fired.iterrows():
                    rule_side[str(_fr["player"])] = _fr["side"]
                lm_rule_hits.append((mkt_key, int(len(_fired)), _rep))
            except Exception as _e:
                # A rule layer failure must not take the tab down. Recorded
                # so a silent zero cannot be mistaken for "nothing fired".
                lm_rule_errors.append(f"{mkt_key}: {type(_e).__name__}: {_e}")

        if len(mv) == 0:
            st.caption(f"No {mkt_key} players with 2+ snapshots yet.")
            continue

        is_td = (mkt_label == "anytime_td")
        line_word = "Prob%" if is_td else "Line"

        mv = mv.copy()
        mv["abs_move"] = mv["line_move"].abs().fillna(0)
        mv = mv.sort_values("abs_move", ascending=False).reset_index(drop=True)

        # Attach team/game, then filter. Filtering mv itself (rather than the
        # display grid) keeps the Bet?/Save path consistent: savable rows and
        # the checkbox map both derive from mv, so a filtered table cannot try
        # to save a player who is not visible.
        if lm_filter_on:
            extra = [qb_rushing] if mkt_label == "rushing" else []
            tmap = game_export.team_map(lm_season, lm_week, mkt_label, MODULES,
                                        db._norm_name, extra_modules=extra)
            mv, n_unmatched, _ = game_export.attach_games(
                mv, "player", tmap, lm_games, db._norm_name)
            mv = game_export.filter_games(mv, lm_picked, include_unassigned=False)
            if len(mv) == 0:
                st.caption(f"No {mkt_key} rows in the selected games.")
                continue
        else:
            mv["Game"] = ""

        cap_disp = (pd.to_datetime(mv["latest_captured"], errors="coerce", utc=True)
                    .dt.tz_convert("America/New_York")
                    .dt.strftime("%m/%d %I:%M%p").fillna("—"))

        grid = pd.DataFrame({
            "Player": mv["player"],
            "Game": mv["Game"],
            "Tier": mv["latest_tier"].map(lambda t: TIER_EMOJI.get(t, "—")),
            "vs. Model": mv["toward_away"].map(lambda t: TA_EMOJI.get(t, "—")),
            "Proj": mv["raw_projection"],
            "Edge": mv["latest_edge"],
            "Side": mv["latest_side"].replace("", "—") if not is_td else "—",
            f"First {line_word}": mv["first_line"],
            f"Latest {line_word}": mv["latest_line"],
            "Move": mv["line_move"],
            "Line Captures": mv["snapshots"],
            "Trend": mv["series"],
            "Captured": cap_disp,
            "Rule": mv["player"].map(lambda p: rule_side.get(str(p), "")),
            "Bet?": False,
        })
        for numcol in ["Proj", "Edge", f"First {line_word}", f"Latest {line_word}", "Move"]:
            grid[numcol] = pd.to_numeric(grid[numcol], errors="coerce").astype("float64")

        # Drop the MODEL columns when this market has nothing to put in
        # them. Captured rows carry a line and no projection, and a market
        # whose beta is clipped to zero is reference-only by design (plan
        # item J1), so Tier, vs. Model, Proj, Edge and Side rendered as a
        # wall of "None" and "—". An empty column reads as a broken value
        # rather than an absent one. Any column with a real value is kept.
        for _c in ["Tier", "vs. Model", "Proj", "Edge", "Side", "Rule"]:
            if _c not in grid.columns:
                continue
            _vals = grid[_c].dropna()
            if len(_vals) == 0 or set(_vals.astype(str)) <= {"—", "", "None",
                                                             "nan"}:
                grid = grid.drop(columns=[_c])

        edited = st.data_editor(
            grid, width='stretch', hide_index=True,
            disabled=[c for c in grid.columns if c != "Bet?"],
            column_config={
                "Player": st.column_config.TextColumn("Player", pinned=True, width="medium"),
                "Game": st.column_config.TextColumn("Game", width="small"),
                "Tier": st.column_config.TextColumn("Tier", width="small"),
                "Proj": st.column_config.NumberColumn("Proj", format="%.1f", width="small"),
                "Edge": st.column_config.NumberColumn("Edge", format="%+.1f", width="small"),
                "Side": st.column_config.TextColumn("Side", width="small"),
                "Trend": st.column_config.LineChartColumn(f"{line_word} Trend", width="small"),
                f"First {line_word}": st.column_config.NumberColumn(f"First {line_word}", format="%.1f", width="small"),
                f"Latest {line_word}": st.column_config.NumberColumn(f"Latest {line_word}", format="%.1f", width="small"),
                "Move": st.column_config.NumberColumn("Move", format="%+.1f", width="small"),
                "vs. Model": st.column_config.TextColumn("vs. Model", width="small"),
                "Line Captures": st.column_config.NumberColumn("Line Captures", format="%d", width="small"),
                "Captured": st.column_config.TextColumn("Captured", width="small"),
                "Bet?": st.column_config.CheckboxColumn("Bet?", width="small"),
            },
            key=f"movement_editor_{mkt_label}")

        with st.expander(f"📋 Copy {mkt_key} as text"):
            copy_df = grid.drop(columns=["Trend", "Bet?"])
            fmt = st.radio("Format", ["Markdown (Reddit)", "TSV (Sheets/Excel)"],
                           horizontal=True, key=f"copy_fmt_{mkt_label}")
            if fmt.startswith("Markdown"):
                txt = copy_df.to_markdown(index=False)
            else:
                txt = copy_df.to_csv(index=False, sep="\t")
            st.code(txt, language=None)

        if st.button(f"💾 Save {mkt_key} to Log", key=f"movement_save_btn_{mkt_label}"):
            savable = mv[mv["latest_edge"].notna()].copy()
            bet_flags = edited.set_index("Player")["Bet?"].to_dict()
            entries = pd.DataFrame({
                "player": savable["player"],
                "projection": savable["raw_projection"],
                "line": savable["raw_line"],
                "over_odds": savable["raw_over_odds"],
                "under_odds": savable["raw_under_odds"],
                "edge": savable["latest_edge"],
                "p_over": savable["p_over"] if not is_td else None,
                "side": savable["latest_side"] if not is_td else "",
                "tier": savable["latest_tier"],
                "bet": savable["player"].map(lambda p: bool(bet_flags.get(p, False))),
            })
            entries["logged_at"] = betlog.now_stamp()
            entries["market"] = mkt_label
            entries["season"] = lm_season
            entries["week"] = lm_week
            entries["result_yards"] = None
            entries["outcome"] = None
            entries = entries[betlog.COLUMNS]
            betlog.append_entries(entries, st.session_state.user)
            st.success(f"Saved {len(entries)} {mkt_key} pick(s) to the log.")

        with st.expander(f"📊 View a player's full {mkt_key} trend"):
            players_with_series = mv[mv["series"].apply(lambda s: len(s) >= 2)]["player"].tolist()
            if not players_with_series:
                st.caption("No players with enough snapshots yet.")
            else:
                picked = st.selectbox("Player", players_with_series, key=f"trend_pick_{mkt_label}")
                row = mv[mv["player"] == picked].iloc[0]
                series = row["series"]
                chart_df = pd.DataFrame({line_word: series})
                st.line_chart(chart_df)
                st.caption(f"{picked}: {len(series)} snapshot(s) captured, "
                           f"from {row['first_line']:.1f} to {row['latest_line']:.1f} {line_word.lower()}.")
            
    # ---- side-picker summary for the week ----
    # A rule that fires zero times and a rule that could not be evaluated
    # look identical in the grids above, so the counts are reported
    # explicitly. rules.describe() carries the provenance and the caveats,
    # which belong next to the output rather than in a commit message.
    if lm_rule_errors:
        st.warning("Side-picker could not be evaluated for: "
                   + "; ".join(lm_rule_errors))
    if lm_rule_hits:
        _tot = sum(n for _, n, _ in lm_rule_hits)
        _parts = [f"{k}: {n}" for k, n, _ in lm_rule_hits if n]
        st.markdown(f"#### Side picker: {_tot} rule firing(s) this week"
                    + (f" · {chr(0x2022).join(_parts)}" if _parts else ""))
        if _tot == 0:
            st.caption("No rule fired. That is a normal outcome, not a "
                       "failure: the rushing rule needs a line at or below "
                       f"{rules.RUSHING_MAX_LINE:g}, and the receiving rule "
                       "needs an air-yards shock of z >= "
                       f"{rules.AIR_YARDS_Z_TRIGGER:g} with the projection "
                       "below the veto.")
        with st.expander("Why these rules, and what they are not"):
            st.code(rules.describe(), language=None)
            _nf = []
            for _k, _n, _rep in lm_rule_hits:
                for _rname, _rr in (_rep.get("rules") or {}).items():
                    if _rr.get("fired"):
                        continue
                    _why = _rr.get("reason") or "no reason recorded"
                    _nf.append(f"{_k} / {_rname}: {_why}")
            if _nf:
                st.caption("Rules that did not fire, and why:")
                for _line in _nf[:12]:
                    st.caption(f"  {_line}")
                       # ============ GUIDE ============
with tab_guide:
    st.subheader("OpalScales Guide")

    st.markdown("""
### What this is

OpalScales prices NFL player props and looks for places the sportsbook is
wrong. It covers six markets: receiving yards, receptions, rushing yards, QB
rushing yards, QB passing yards, and anytime touchdown.

It is a research tool that happens to have a betting interface. Most of what
it has established is negative, and that is the point. Knowing which ideas do
not work is what makes the remaining ones worth anything.

**Read this part before anything else.** As of 25 September 2026, the
projection models do not beat the market in any of the six markets. Not one
board shows a model edge, because none is supported by the evidence. What the
app does show is two specific rules that do not depend on the projection at
all, and a lot of reference data.

---

### The short version of what was learned

The models are good at estimating a player's expected output. They are not
good at beating a sportsbook, and those are different problems.

Every measurement of model skill in this project compared the projection to
the **line**. The line turns out to be the stale number. FanDuel moves
reception prices mainly through the **odds**, leaving the line flat about 93
percent of the time. So a model that improves on the line can add nothing to
the price you actually bet against.

Receptions was the last market with a model verdict. Its blend weight of
0.2261 was real, measured with a clustered t of +6.5 and confirmed four
separate ways. Tested against FanDuel's own devigged price, it lost on log
loss and returned an encompassing t of +0.60. That test had the power to
detect a real advantage at t +7.45, and the model's own blend constants were
fitted on the very rows being scored, so the test was tilted in its favour and
it still failed. The verdict was retired the same day.

What survived is different in kind. Two **pricing biases**, where the book
shades a price rather than misjudging a player. Neither uses the projection to
pick a side.

---

### The two live rules

Both say UNDER. That is not a preference, it is what survived. Every over side
tested failed. Across qb_passing, qb_rushing, receiving and receptions, 28 of
28 book and market combinations had the under below breakeven, and only
rushing came out positive. The likely reason is well documented elsewhere:
casual money prefers overs and books price accordingly.

**Rushing, low line, under.** When a rushing line is 46.5 or below, bet the
under. Return was 5.9 percent per unit staked, with a 95 percent interval of
1.2 to 10.8 percent, over roughly 438 bets a season. It was positive in all
four seasons, decayed smoothly as the line threshold rose rather than spiking
at one value, held up at ordinary prices instead of only on short ones, and
appeared at six of seven sportsbooks with under win rates between 54.8 and
56.1 percent. That last point matters: the bias is in the market, not in one
book. FanDuel is the place to play it only because its hold is 4.9 percent
against 6.0 to 7.1 elsewhere. The model contributes nothing here and is not
consulted.

**Receiving, air yards, under.** When a receiver's recent air yards jump well
above his own baseline, bet the under, unless the model projects him well
above the line. Roughly 250 bets a season, about 8.4 percent over the market's
own baseline. The mechanism is that the line keeps weighting recent downfield
usage as heavily as it used to while that usage has become less predictive, so
the price leans too far toward a couple of deep targets.

The model's job in this rule is narrow and worth being precise about. Inside
the rule, returns by projection quintile run +10.0, +7.9, +12.7, +8.6, then
**-11.7** percent in the top quintile. So the model is not picking a
direction, it is identifying one bad bucket. It vetoes, it does not confirm.

Both rules appear in the **Line Movement and Side Picker** tab, in the
**Rule** column, with a weekly summary underneath.

---

### Honest limits on those rules

The rushing rule is the stronger of the two and I would still not call it
settled. Both were found by searching this data, so their intervals are
optimistic no matter how carefully each individual test was run. The receiving
rule scores 0.072 on a best of grid placebo, which is borderline rather than
clear, and its returns decay by season: 16.5 percent, then 10.2, then 7.0.
That pattern is consistent with a market slowly correcting a stale weight,
which would mean the edge closes on its own.

At a few dollars a bet, 438 rushing bets a season at 5.9 percent is roughly 25
to 130 dollars of expected profit. The honest return here is knowledge and
enjoyment rather than income.

---

### How to use the app

1. Pick a market, season and week at the top of the **Board** tab.
2. Enter the sportsbook's numbers, or let imported lines fill in. Yardage and
   receptions markets want the line plus both sides' odds. Anytime TD wants
   the American odds on its single side.
3. Read the board as reference. No market shows a model edge.
4. Go to **Line Movement and Side Picker** for the actual picks. The **Rule**
   column is where a validated rule has fired.
5. Save picks to the log, tick the ones you really bet, and the **Scorecard**
   follows them over time.

Anything you type on the Board stays in your session. It never changes the
shared line data, so you can test alternate numbers from another book freely.

---

### How to read the columns

**Proj** is the model's projection, in yards, catches, or a probability for
anytime TD. Useful as an estimate of expected output. Not a betting signal.

**Line** is the sportsbook's number.

**Over and Under** are the American odds on each side. These matter more than
they look, because the price is where the book takes its cut, and because on
several markets the price carries more information than the line does.

**Mkt P(over)%** is the market's own probability that the result lands over,
with the sportsbook's margin removed. This is the best calibrated number on
the board and it is not the model's. Bin it against real outcomes and it comes
back with biases between -0.8 and +4.3 percentage points, and a slope of 1.09
against 1.00 for a perfect forecast. When a well measured rule fires and this
number disagrees, the rule has the better track record on these specific
props, but that is the whole of the claim.

**Implied %** appears on anytime TD instead. It comes straight from the odds
and still **includes the book's margin**, because a one sided market cannot
have the margin removed cleanly. So it overstates the true probability, and the
model reading lower than it is expected rather than a disagreement.

**Rule** in the Line Movement and Side Picker tab shows UNDER where a
validated rule has fired.

**Edge, Side and Tier** are absent from every market. They were removed
deliberately, one market at a time, as each was measured and failed. An absent
column says the app makes no claim. A blank one would just look broken.

---

### Anytime TD is showing a probability and nothing else

Three defects were found in that model on 25 September 2026, all inflating the
numbers for low usage players.

The model was trained on players averaging 3 or more touches but served to
players at 1.5 or more, so it was fitted on regular contributors and then
asked about fringe ones. The week one bridge measured a player's scoring rate
over his busiest weeks only. And it enforced no minimum game count at all, so
one prior game with a touchdown became a 100 percent scoring rate. Nine
players out of 460 were in exactly that state.

That is how the board came to show fringe players at 27 to 40 percent against
market prices of 3 to 10, every row tagged as a maximum confidence play.

All three are fixed. A residual bias of about 3.8 percentage points remains in
the low usage band, and the model still cannot tell a blocking tight end from
a receiving one, because it only sees touches, scoring rate, and two position
flags. The probability is shown. The verdict stays off until a full season
re-measures it.

---

### Ideas that were tested and do not work

Worth listing so nobody spends another weekend on them.

**Line shopping across books.** Settling at the best available price returned
-10.0 percent against +14.8 for FanDuel alone. The best line is best because
it is the one most likely to be wrong in your favour, which is adverse
selection rather than an edge.

**Fading recent performance.** The idea that the line overreacts to a bad game
is appealing and false. FanDuel's weight on a player's last game matches the
weight that actually predicts his next one, in all five two sided markets,
with every interval covering zero. Zero of 40 versions of the rule cleared.

**The model as a general side picker.** Measured directly against the devigged
price in the market where it is strongest, it adds nothing.

**Alpha as a signal.** The gap between a projection's mean and median is a
real distributional feature and a valid pricing input. It is not an edge, and
treating it as one is a mistake this project made and corrected.

**Obscure props being softer.** The theory that the rushing edge lives in props
few books post is wrong. Returns at the most obscure end were **negative**.
The edge is spread across the low line band, not concentrated in unstakeable
corners.

---

### Why so much of this is negative

Roughly a third of the findings in this project have been artifacts caught
before they became beliefs. A few examples, because the pattern is worth
recognising.

A blend weight of 0.655 on QB rushing yards looked like the second strongest
signal in the project. It came entirely from props FanDuel never posted, where
the reference line was noise. On a line that carries no information, the
arithmetic drives that number toward 1.0 mechanically.

A promising QB passing rule swung from 248 bets to 194 on a **two row** change
to its inputs. Two rows cannot do that directly. It happened because the rule
picked its own threshold by searching for the best one, so a tiny change made
the search jump to a neighbouring value. Three separate findings in this
project have had that same defect.

The lesson that generalises: a good looking number is a hypothesis about a
bug until it survives a test designed to kill it. Sign consistency across
seasons, a smooth response to nearby thresholds, and a placebo that accounts
for how many things were tried.

---

### What is still open

Whether the receiving rule's decay continues, which would mean the market has
closed it.

Whether the QB passing side picker is real. It scores 0.010 on the same
placebo the receiving rule scores 0.072 on, which is better, but it has only
two scorable seasons and its returns thin out sharply as the sample grows. It
is not wired into the app.

Whether adding air yards to the receiving and receptions models helps. It is
the one input measured as mispriced by the market and it is absent from both
models.

Forward results. Everything here was found by looking at 2023 to 2026 data,
which no amount of care fully corrects for. The next genuinely new information
arrives on Sunday.
""")
