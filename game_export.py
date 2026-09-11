"""
OpalScales - game/matchup filtering for exports.

THE PROBLEM THIS SOLVES
    The lines table stores only player, market, line and odds, so
    get_line_movement has no team information. Exporting "the NE vs SEA game"
    therefore needs teams attached from somewhere else.

    Teams come from each market module's project_week(), which returns `team`
    and `opponent_team`. Kickoff slots come from nfl.load_schedules(), which
    carries gameday and gametime - the same source build_upcoming_week already
    uses for spread and total.

    Matching is by normalized player name (db._norm_name), the same normalizer
    that got the suffix and nickname fixes. Players who fail to match are
    COUNTED AND REPORTED rather than silently dropped, because a silent drop is
    how you export a file that is quietly missing half a game.
"""

import pandas as pd


SLOT_ORDER = ["Wed", "Thu", "Fri", "Sat", "Sun AM", "Sun 1pm", "Sun 4pm",
              "SNF", "MNF", "Tue", "Other"]

# team_map now calls build_upcoming_week as well as project_week, and the
# assembler hits load_rosters and load_schedules each time. Five markets per
# rerun made that noticeable, so results are memoised per (season, week,
# market). The dict persists across Streamlit reruns because the module stays
# loaded. Team assignments do not change within a week, so staleness is not a
# concern; a full app reboot clears it.
_TEAM_MAP_CACHE = {}


def _slot(gameday, gametime):
    """Classify a kickoff into a viewing slot. gametime is ET in the schedule."""
    try:
        d = pd.to_datetime(gameday, errors="coerce")
        dow = d.day_name() if pd.notna(d) else ""
    except Exception:
        dow = ""
    hour = None
    if gametime is not None and str(gametime) not in ("", "nan", "None"):
        try:
            hour = int(str(gametime).split(":")[0])
        except Exception:
            hour = None

    if dow == "Wednesday":
        return "Wed"
    if dow == "Thursday":
        return "Thu"
    if dow == "Friday":
        return "Fri"
    if dow == "Saturday":
        return "Sat"
    if dow == "Monday":
        return "MNF"
    if dow == "Tuesday":
        return "Tue"
    if dow == "Sunday":
        if hour is None:
            return "Other"
        if hour < 12:
            return "Sun AM"        # London / early kickoffs
        if hour < 15:
            return "Sun 1pm"
        if hour < 19:
            return "Sun 4pm"
        return "SNF"
    return "Other"


def load_games(season, week):
    """One row per game: matchup label, both teams, slot, kickoff."""
    import nflreadpy as nfl
    sched = nfl.load_schedules([int(season)])
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    wk = sched[sched["week"] == int(week)].copy()
    if len(wk) == 0:
        return pd.DataFrame(columns=["matchup", "away_team", "home_team",
                                     "slot", "gameday", "gametime"])
    gd = wk["gameday"] if "gameday" in wk.columns else pd.Series([None] * len(wk))
    gt = wk["gametime"] if "gametime" in wk.columns else pd.Series([None] * len(wk))
    wk["slot"] = [_slot(a, b) for a, b in zip(gd, gt)]
    wk["matchup"] = wk["away_team"].astype(str) + " @ " + wk["home_team"].astype(str)
    cols = ["matchup", "away_team", "home_team", "slot"]
    for extra in ("gameday", "gametime"):
        if extra in wk.columns:
            cols.append(extra)
    out = wk[cols].copy()
    out["_slot_rank"] = out["slot"].map(
        {s: i for i, s in enumerate(SLOT_ORDER)}).fillna(99)
    return (out.sort_values(["_slot_rank", "matchup"])
            .drop(columns=["_slot_rank"]).reset_index(drop=True))


def team_map(season, week, market_key, modules, norm, extra_modules=()):
    """norm_name -> (team, opponent) for a market.

    `norm` is the name normalizer (pass db._norm_name). Injected rather than
    imported so this module is testable outside the app.

    extra_modules covers markets served by more than one model, e.g. rushing
    also carries QB rushing players through qb_rushing.
    """
    ck = (int(season), int(week), market_key,
          tuple(getattr(m, "__name__", str(m)) for m in extra_modules))
    if ck in _TEAM_MAP_CACHE:
        return _TEAM_MAP_CACHE[ck]

    mods = [modules[market_key]] + list(extra_modules)
    m = {}
    for mod in mods:
        # BOTH sources, and this is the point of the function.
        #
        # project_week has two paths: played games, or build_upcoming_week as a
        # fallback when the requested week has no rows yet. Mid-week those
        # differ. Once the Wednesday game posted to nflverse, project_week
        # started taking the played path and returned ONLY New England and
        # Seattle players, so every other team failed the team match and got
        # hidden by the filter. A game plainly full of rows read as empty.
        #
        # Calling the assembler as well covers every team regardless of what
        # has been played. project_week runs first and setdefault keeps it
        # authoritative where both have a player, since played-game rows carry
        # the real current team.
        for getter in ("project_week", "build_upcoming_week"):
            fn = getattr(mod, getter, None)
            if fn is None:
                continue
            try:
                proj = fn(int(season), int(week))
            except Exception:
                continue
            if proj is None or len(proj) == 0 or "team" not in proj.columns:
                continue
            for _, r in proj.iterrows():
                key = norm(r.get("player_display_name"))
                if not key:
                    continue
                m.setdefault(key, (r.get("team"), r.get("opponent_team")))

    _TEAM_MAP_CACHE[ck] = m
    return m


def attach_games(frame, player_col, tmap, games, norm):
    """Add Team / Opp / Game columns. Unmatched rows get empty strings.

    Returns (enriched_frame, n_unmatched, unmatched_names).
    """
    lookup = {}
    for _, g in games.iterrows():
        lookup[str(g["away_team"])] = g["matchup"]
        lookup[str(g["home_team"])] = g["matchup"]

    teams, opps, gms = [], [], []
    unmatched = []
    for nm in frame[player_col]:
        t, o = tmap.get(norm(nm), (None, None))
        if t is None:
            unmatched.append(nm)
            teams.append("")
            opps.append("")
            gms.append("")
        else:
            teams.append(t)
            opps.append(o if o is not None else "")
            gms.append(lookup.get(str(t), ""))
    out = frame.copy()
    out.insert(1, "Game", gms)
    out.insert(2, "Team", teams)
    out.insert(3, "Opp", opps)
    return out, len(unmatched), unmatched


def filter_games(frame, selected_matchups, include_unassigned=False):
    """Keep rows whose Game is in the selection."""
    if not selected_matchups:
        return frame.iloc[0:0]
    keep = frame["Game"].isin(selected_matchups)
    if include_unassigned:
        keep = keep | (frame["Game"] == "")
    return frame[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bet tracker export
# ---------------------------------------------------------------------------

TRACKER_COLS = ["Game Date", "Kickoff", "Game", "Logged", "Season", "Week",
                "Market", "Player", "Side", "Line", "Odds", "Edge", "Tier",
                "Outcome"]


def _to_eastern(series):
    """betlog stores logged_at in UTC (the server clock). The tracker wants
    Eastern, so a bet placed at 10:47 ET does not read 14:47."""
    import pandas as pd
    t = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        t = t.dt.tz_convert("America/New_York")
    except Exception:
        pass
    return t.dt.strftime("%Y-%m-%d %H:%M").fillna("")


def tracker_rows(bets, games_by_week, tmap_by_key, norm):
    """Build the 14 tracker columns from betlog rows.

    bets           betlog rows already filtered to bet == True
    games_by_week  {(season, week): games DataFrame from load_games}
    tmap_by_key    {(season, week, market): team map from team_map}
    norm           name normalizer (db._norm_name)

    Odds are picked by side, because betlog stores over_odds and under_odds
    separately and there is no single odds column. Taking the wrong one would
    silently produce wrong profit figures in the sheet.
    """
    import pandas as pd

    out = []
    logged = _to_eastern(bets["logged_at"]) if "logged_at" in bets.columns \
        else pd.Series([""] * len(bets), index=bets.index)

    for idx, r in bets.iterrows():
        season, week = r.get("season"), r.get("week")
        market, player = r.get("market"), r.get("player")
        side = str(r.get("side") or "").upper()

        gdf = games_by_week.get((season, week))
        tmap = tmap_by_key.get((season, week, market), {})
        team = (tmap.get(norm(player)) or (None, None))[0]

        gdate, kick, matchup = "", "", ""
        if gdf is not None and team:
            hit = gdf[(gdf["home_team"].astype(str) == str(team))
                      | (gdf["away_team"].astype(str) == str(team))]
            if len(hit):
                g = hit.iloc[0]
                matchup = g.get("matchup", "")
                gdate = str(g.get("gameday", "") or "")
                kick = str(g.get("gametime", "") or "")

        odds = r.get("over_odds") if side == "OVER" else r.get("under_odds")
        if odds is None or (isinstance(odds, float) and odds != odds):
            odds = r.get("over_odds") if r.get("over_odds") is not None \
                else r.get("under_odds")

        # `x or ""` is not enough: a pandas NaN is TRUTHY, so a pending bet
        # came through as the string "NAN" and would not match the sheet's
        # WIN / LOSS / VOID / blank convention.
        ov = r.get("outcome")
        outcome = "" if (ov is None or pd.isna(ov)) else str(ov).strip().upper()

        out.append({
            "Game Date": gdate,
            "Kickoff": kick,
            "Game": matchup,
            "Logged": logged.get(idx, ""),
            "Season": season,
            "Week": week,
            "Market": market,
            "Player": player,
            "Side": side,
            "Line": r.get("line"),
            "Odds": odds,
            "Edge": r.get("edge"),
            "Tier": r.get("tier"),
            "Outcome": outcome,
        })

    df = pd.DataFrame(out, columns=TRACKER_COLS)
    if len(df):
        # Rows whose team could not be resolved have an empty Game Date, which
        # sorts first as a string. Push them to the end so the sheet reads
        # chronologically and the odd unmatched row is obvious at the bottom.
        df["_blank"] = (df["Game Date"].astype(str) == "").astype(int)
        df = (df.sort_values(["_blank", "Game Date", "Game", "Player"])
              .drop(columns=["_blank"]).reset_index(drop=True))
    return df
