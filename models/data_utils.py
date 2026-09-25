"""
Shared data-loading helpers that gracefully skip seasons whose data
doesn't exist yet (e.g. a future season before it has been played).
"""
import nflreadpy as nfl
import os

from datetime import date


def _current_nfl_season():
    """The season whose data should already be posted.

    From mid-September onward the current season is under way, so its data
    must exist and a failure to load it is a real failure. Before that, only
    completed seasons can be required.
    """
    t = date.today()
    if t.month > 9 or (t.month == 9 and t.day >= 15):
        return t.year
    return t.year - 1

# nflreadpy's cache directory did not exist, so every call re-downloaded from
# the network and a transient failure changed the dataset from run to run.
# Filesystem caching makes the frames reproducible across runs and makes
# build_dataset much faster. update_config works whatever the import order,
# where the NFLREADPY_CACHE env var only applies before config is built.
from nflreadpy.config import update_config
update_config(cache_mode="filesystem")


def _try_seasons(loader, seasons, required_through=None):
    """Call an nflreadpy loader one season at a time.

    A season that genuinely has no data yet (a future season) is skipped.
    A season that FAILS for any other reason raises, because the old
    behaviour swallowed every exception and returned a short frame. A
    transient network failure on one season silently produced a dataset
    missing roughly a third of its rows, and since no caller kept the second
    return value, nothing surfaced it. The model then trained on whatever
    arrived. Same failure class as the snap-count merge and the null
    projections: a wrong answer with no error.

    required_through: the last season that MUST load. Defaults to the latest
    completed season inferred from the request, so history is mandatory and
    only the newest season is allowed to be absent.
    """
    import pandas as pd
    seasons = list(seasons)
    if required_through is None and seasons:
        # Derive the requirement from the calendar, not from the request.
        # Defaulting to max(seasons) - 1 left the CURRENT season as a soft
        # skip, which is backwards: losing 2026 silently means projecting off
        # the prior-season bridge alone with only a printed note to show it.
        required_through = min(max(seasons), _current_nfl_season())

    frames = []
    loaded = []
    failures = []
    for s in seasons:
        try:
            raw = loader([s])
            df = raw.to_pandas() if hasattr(raw, "to_pandas") else raw
        except Exception as e:
            failures.append((s, f"{type(e).__name__}: {e}"))
            continue
        if len(df) > 0:
            frames.append(df)
            loaded.append(s)
        else:
            failures.append((s, "returned 0 rows"))

    # A season at or below required_through is history and must be present.
    # Anything above it may legitimately not be posted yet.
    hard_failures = [(s, msg) for s, msg in failures if s <= required_through]
    if hard_failures:
        detail = "; ".join(f"{s}: {msg}" for s, msg in hard_failures)
        raise RuntimeError(
            f"{getattr(loader, '__name__', 'loader')} failed for season(s) "
            f"{[s for s, _ in hard_failures]} and returned partial data. "
            f"Refusing to train on a short frame. Details: {detail}")

    if failures:
        skipped = [s for s, _ in failures]
        print(f"NOTE: {getattr(loader, '__name__', 'loader')} skipped "
              f"season(s) {skipped} (not posted yet)")

    if not frames:
        raise RuntimeError(
            f"{getattr(loader, '__name__', 'loader')} loaded no seasons at all "
            f"from {seasons}. Check network access and nflverse availability.")

    return pd.concat(frames, ignore_index=True), loaded


def load_player_stats(seasons):
    df, _ = _try_seasons(nfl.load_player_stats, seasons)
    return df


def load_schedules(seasons):
    df, _ = _try_seasons(nfl.load_schedules, seasons)
    return df


def load_snap_counts(seasons):
    df, _ = _try_seasons(nfl.load_snap_counts, seasons)
    return df


def load_pbp(seasons):
    df, _ = _try_seasons(nfl.load_pbp, seasons)
    return df


# nflverse tables disagree on nicknames vs legal names, which suffix stripping
# cannot reconcile. Both spellings map to one canonical key. Keys and values are
# already normalized (lowercase, punctuation and suffixes stripped), so add any
# new entries in that form.
_NAME_ALIASES = {
    "chig okonkwo": "chigoziem okonkwo",
    "kenny gainwell": "kenneth gainwell",
    "gabe davis": "gabriel davis",
    "bam knight": "zonovan knight",
    "dee eskridge": "dwayne eskridge",
    "mike woods": "michael woods",
    # Added September 24 2026 from name_resolve.py, run against 33,885
    # rushing line rows and 78,014 nflverse player-weeks. Each was accepted
    # on POSITION plus SEASON agreement, not on string similarity alone:
    # a candidate at the wrong position is a spelling coincidence and would
    # create a wrong join, which is worse than the missing one it replaces.
    "cameron ward": "cam ward",              # QB, 2025-2026, 38 line rows
    "josh dobbs": "joshua dobbs",            # QB, 2022-2025, 15 line rows
    "mitch trubisky": "mitchell trubisky",   # QB, 2022-2025, 8 line rows
    "eli mitchell": "elijah mitchell",       # RB, 2022-2025, 7 line rows
    "phillip walker": "pj walker",           # QB, 2022-2023, 2 line rows
    #
    # DELIBERATELY NOT ADDED: the four abbreviated-initial forms
    # ("j hill", "d evans", "d henderson", "m rudolph"), 8 line rows total.
    # An alias is a PERMANENT GLOBAL REWRITE, and an initial form is not
    # unique to one player: a future Evans would silently become Darrynton
    # Evans. "j hill" resolved to four candidates of which only Justice Hill
    # carries the ball, so it is probably right, but "probably" earned on 5
    # rows does not justify a permanent rule. These stay unmatched on
    # purpose. Resolving them needs a team column, not a name rule.
}


def norm_join_name(s):
    """Normalize a name column for joining across nflverse tables.

    The stats and snap-count tables format names differently. Periods were
    already handled ("DK Metcalf" vs "D.K. Metcalf"), but SUFFIXES were not,
    which silently dropped 546 rows (4.1%) from receiving: Michael Pittman vs
    Michael Pittman Jr., Deebo Samuel Sr., Chris Godwin Jr., Brian Thomas Jr.,
    Luther Burden III and others. A failed join left offense_pct null, so
    snap_roll came out all-NaN, and because snap_roll is a LEAN_FEAT the row
    was dropped from the dataset entirely.

    Nicknames are handled separately via _NAME_ALIASES, since no string rule
    turns "Chig Okonkwo" into "Chigoziem Okonkwo".

    SEPTEMBER 24 2026, two additions, both found by qb_split_report.py on the
    184,782-row line cache:

      ACCENTS. Punctuation was stripped but diacritics were not, so "Audric
      Estime" as FanDuel spells it never matched nflverse's "Audric Estime"
      with an acute e. That was 58 rushing rows on one player, and since the
      rule is general it was silently costing rows in every market. NFKD
      decomposition plus removal of the combining range folds them.

      PARENTHETICAL TEAM TAGS. FanDuel writes "Lamar Jackson (BAL)" where
      nflverse has "Lamar Jackson", 17 rows. Note WHY the tag is there: two
      players in league history normalize to "lamar jackson", and FanDuel is
      disambiguating deliberately rather than decorating. Stripping the tag
      therefore converts a guaranteed miss into a match that is correct only
      if the normalized name is unique in nflverse. Run name_resolve.py to
      measure the collision count before relying on this; where a collision
      exists, the position lookup survives it (it takes the modal position)
      but actual_result does not, because it takes iloc[0].

    Order matters. Parentheticals are removed BEFORE the punctuation class,
    because that class does not contain brackets, so "(BAL)" would otherwise
    survive as a bare token.

    Takes a pandas Series, returns a pandas Series.
    """
    import pandas as pd

    # FORCE THE OBJECT DTYPE BEFORE ANY REGEX WORK. This is not cosmetic.
    #
    # pandas may back a string column with pyarrow, and when it does,
    # .str.replace(regex=True) is executed by RE2 rather than by Python's
    # re. The two engines do not accept the same syntax. RE2 rejects \uXXXX
    # escapes outright, so a raw-string pattern that Python resolves happily
    # raises ArrowInvalid: "invalid escape sequence: \u".
    #
    # That is exactly what broke the deployed app on September 25 while every
    # local run passed: the local frame was object dtype and used re, the
    # Streamlit Cloud frame was pyarrow-backed and used RE2. Converting here
    # pins the engine to the one these patterns were written and tested
    # against, and closes the whole class of divergence rather than the one
    # escape that happened to surface.
    out = pd.Series(s.astype(str).to_numpy(dtype=object), index=s.index)

    out = (out
           .str.normalize("NFKD")
           # NOT a raw string, deliberately. Python resolves these escapes so
           # the engine receives the two literal combining characters rather
           # than the six-character text "\u0300". A raw string here was the
           # bug above.
           .str.replace("[\u0300-\u036f]", "", regex=True)
           .str.replace(r"\s*\([^)]*\)", "", regex=True)
           .str.replace("[.'\u2019\u2018`,-]", "", regex=True)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.lower())
    out = out.str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
    return out.replace(_NAME_ALIASES)


# ---------------------------------------------------------------------------
# PLAN ITEM 3.5: THE qb_rushing LABEL, DERIVED AT ANALYSIS TIME
# ---------------------------------------------------------------------------
#
# WHY THIS IS A DERIVATION AND NOT AN IMPORT FIX
#
#   The Odds API uses ONE market key for rushing yards regardless of who is
#   rushing, and the FanDuel transcription schema does the same. There is no
#   upstream label to map, so `MARKET_MAP` in db.py is not missing an entry:
#   there is nothing to put in it. Position lives only in nflverse, which is
#   reached after the join, so after the join is the only place the split can
#   be made.
#
#   Deriving rather than relabelling at write time also buys two things:
#
#     1. It applies RETROACTIVELY to all 354,554 historical rows in
#        lines_cache.parquet and to every existing row in `lines`. Fixing
#        db.import_lines would only label rows captured from that moment on,
#        which would leave the market unmeasurable historically, which is the
#        entire point of doing this.
#
#     2. It touches no app behaviour. `db.get_lines` filters on
#        `.eq("market", market)`, so writing 'qb_rushing' into the table would
#        stop the rushing board prefilling QB rows, and would also break the
#        Line Movement market filter and the betlog market filter. Four call
#        sites, none of them in need of changing.
#
#   Note the app ALREADY prices these rows correctly: app.py and
#   db._get_board concat the QB board onto rushing with `is_qb_model=True`,
#   and mc.edge_calc is called with "qb_rushing" as the effective market. The
#   flag is computed and then discarded rather than persisted. So this is a
#   MEASUREMENT gap, not a pricing gap.
#
# WHY THERE IS NO DEFAULT MATCH-RATE GATE
#
#   Lines exist for players who were later inactive and therefore have no
#   nflverse stat row, so the name match rate is legitimately below 100 and
#   nobody has measured what it actually is. Picking a threshold here would be
#   planting a number. `min_match_rate` therefore defaults to None: run it
#   once, read `report["match_rate"]`, then set the gate from the measurement
#   and it will abort rather than warn from then on.

from functools import lru_cache

QB_RUSHING_MARKET = "qb_rushing"


@lru_cache(maxsize=8)
def _position_rows(seasons_key):
    """Cached (season, normalized name) -> position rows from nflverse.

    lru_cache rather than st.cache_data on purpose: this module is imported by
    eval_harness, by the edge_threshold scripts and from Colab, none of which
    run inside Streamlit. Adding a streamlit import here would make every
    analysis script depend on it.
    """
    import pandas as pd
    ps = load_player_stats(list(seasons_key))
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    keep = [c for c in ("season", "position", "player_display_name") if c in ps.columns]
    if len(keep) < 3:
        raise RuntimeError(
            "load_player_stats is missing one of season / position / "
            f"player_display_name. Got columns: {sorted(ps.columns)[:40]}")
    out = ps[keep].dropna(subset=["position", "player_display_name"]).copy()
    out["player_norm"] = norm_join_name(out["player_display_name"])
    return out[["season", "player_norm", "position"]]


def player_position_map(seasons, position_rows=None):
    """Two position lookups: within-season, and pooled across seasons.

    A player can be listed at more than one position across a career, so both
    maps take the MODE of the position rather than the first row. The pooled
    map is the fallback for a line on a player with no stat row in that
    season, which happens when a player is listed and then never plays.

    Returns (per_season, pooled), both pandas frames.
    """
    import pandas as pd
    rows = _position_rows(tuple(sorted(seasons))) if position_rows is None else position_rows
    rows = rows.copy()

    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else None

    per_season = (rows.groupby(["season", "player_norm"])["position"]
                  .agg(_mode).reset_index()
                  .rename(columns={"position": "position_season"}))
    pooled = (rows.groupby("player_norm")["position"]
              .agg(_mode).reset_index()
              .rename(columns={"position": "position_pooled"}))
    return per_season, pooled


def split_qb_rushing(df, seasons=None, market_col="market", player_col="player",
                     season_col="season", from_market="rushing",
                     to_market=QB_RUSHING_MARKET, position_rows=None,
                     min_match_rate=None):
    """Relabel rushing lines belonging to quarterbacks as `qb_rushing`.

    Call this on any joined or pre-join line frame BEFORE grouping by market.
    It is idempotent: rows already labelled `to_market` are left alone, so
    calling it twice cannot double-count.

    Rows whose player cannot be matched to a position KEEP their original
    market. That is deliberate and it is the one quiet failure mode here: an
    unmatched quarterback stays labelled 'rushing' and is then dropped by the
    RB position filter in models/rushing.py exactly as it is today. The report
    exists so that share is visible rather than assumed.

    position_rows: inject a frame with columns season / player_norm / position
    to run without network access. Used by smoke_qb_split.py.

    Returns (out, report). `report` is a plain dict, so it prints and it
    serialises.
    """
    import pandas as pd

    out = df.copy()
    for col in (market_col, player_col, season_col):
        if col not in out.columns:
            raise KeyError(
                f"split_qb_rushing needs column '{col}'. "
                f"Present: {sorted(out.columns)[:40]}")

    target = out[market_col].astype(str).str.strip().str.lower() == from_market
    n_target = int(target.sum())
    report = {
        "rows_total": int(len(out)),
        "rows_in_from_market": n_target,
        "from_market": from_market,
        "to_market": to_market,
    }

    if n_target == 0:
        report.update({"matched_season": 0, "matched_pooled": 0, "unmatched": 0,
                       "match_rate": None, "reassigned": 0, "distinct_qbs": 0,
                       "note": f"no rows with market == '{from_market}'"})
        return out, report

    if seasons is None:
        seasons = sorted(int(s) for s in out.loc[target, season_col].dropna().unique())
    per_season, pooled = player_position_map(seasons, position_rows=position_rows)

    sub = out.loc[target, [season_col, player_col]].copy()
    sub["player_norm"] = norm_join_name(sub[player_col])
    sub["_season_key"] = pd.to_numeric(sub[season_col], errors="coerce").astype("Int64")

    per_season = per_season.copy()
    per_season["_season_key"] = pd.to_numeric(
        per_season["season"], errors="coerce").astype("Int64")

    sub = sub.merge(per_season[["_season_key", "player_norm", "position_season"]],
                    on=["_season_key", "player_norm"], how="left")
    sub = sub.merge(pooled, on="player_norm", how="left")

    matched_season = sub["position_season"].notna()
    position = sub["position_season"].where(matched_season, sub["position_pooled"])
    matched_pooled = (~matched_season) & sub["position_pooled"].notna()
    unmatched = position.isna()

    is_qb = (position.astype(str).str.upper() == "QB").to_numpy()

    idx = out.index[target]
    out.loc[idx[is_qb], market_col] = to_market

    report.update({
        "matched_season": int(matched_season.sum()),
        "matched_pooled": int(matched_pooled.sum()),
        "unmatched": int(unmatched.sum()),
        "match_rate": round(float(1.0 - unmatched.mean()), 4),
        "reassigned": int(is_qb.sum()),
        "reassigned_share": round(float(is_qb.mean()), 4),
        "distinct_qbs": int(sub.loc[is_qb, "player_norm"].nunique()),
        "seasons": list(seasons),
    })

    if min_match_rate is not None and report["match_rate"] < min_match_rate:
        raise RuntimeError(
            f"split_qb_rushing matched only {report['match_rate']:.4f} of "
            f"{n_target} '{from_market}' rows to a position, below the "
            f"required {min_match_rate}. Refusing to return a frame whose "
            f"market labels are mostly unresolved. Full report: {report}")

    return out, report


# ---------------------------------------------------------------------------
# SHARED GRADING, WITH A COLLISION GUARD. Plan items 6.2 and 3.6.
# ---------------------------------------------------------------------------
#
# THE BUG THIS FIXES, AND IT IS A MONEY BUG
#
#   Every market module implements its own actual_result, and each one does
#   some version of:
#
#       m = ps[(season) & (week) & (player_display_name == name)]
#       val = m.iloc[0][stat_col]
#
#   iloc[0] assumes the name identifies ONE player. name_resolve.py measured
#   that assumption failing for 21 normalized names in nflverse, and the
#   worst case is not marginal:
#
#       michael carter   CB with 51 rows   against   RB with 47 rows
#
#   Michael Carter the running back carries real props. In any week both
#   played, iloc[0] picks whichever row pandas happens to return first, so a
#   bet can be graded against a cornerback's rushing yards. The bet log is
#   supposed to be ground truth, and a silently wrong grade corrupts every
#   figure computed from it afterwards.
#
#   Other exposed names with bettable players: aj green, michael thomas,
#   dj turner, spencer brown, tyler davis, anthony brown (a QB), and
#   lamar jackson (QB with 64 rows against a CB with 3).
#
#   Note this is NOT caused by norm_join_name. Both Michael Carters are
#   spelled identically in FanDuel and in nflverse, so no normalizer is
#   involved. The defect predates the September 24 name work entirely.
#
# WHY THIS REFUSES RATHER THAN GUESSING
#
#   A missing grade is a visible gap that shows up as an ungraded row. A
#   wrong grade is invisible and permanent. So when a name resolves to more
#   than one player_id, actual_stat returns None and the row stays ungraded.
#
# WHY IT LIVES HERE RATHER THAN IN FIVE MODULES
#
#   Copying a guard into five files is the "import, do not restate" failure
#   that produced this project's two worst errors. One implementation, five
#   callers.

# Seasons the grading frame covers. Derived from the calendar rather than
# hardcoded, so it does not silently stop covering the current season.
def _stat_seasons():
    return tuple(range(2022, _current_nfl_season() + 2))


@lru_cache(maxsize=4)
def _stats_frame(seasons_key):
    """Unfiltered player stats for grading, with collision counts attached.

    Deliberately NOT filtered by position or volume. rushing.build_dataset
    drops every week under 5 carries and every non-RB, which is why a backup
    who got 2 carries used to return None and the bet silently never graded.
    """
    ps = load_player_stats(list(seasons_key))
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps["_norm"] = norm_join_name(ps["player_display_name"])
    id_col = next((c for c in ("player_id", "gsis_id") if c in ps.columns),
                  None)
    # Carry the identity itself rather than a precomputed collision count.
    # A count computed here would be stale after a POSITION filter: narrowing
    # to QB can leave one unambiguous row that still carries a flag saying
    # two players share the name. The smoke test caught exactly that.
    ps["_pid"] = ps[id_col] if id_col else ps["player_display_name"]
    return ps


def colliding_names(seasons=None):
    """Normalized names that map to more than one player. For auditing."""
    ps = _stats_frame(tuple(seasons) if seasons else _stat_seasons())
    n = ps.groupby(["season", "week", "_norm"])["_pid"].transform("nunique")
    bad = ps[n > 1]
    if not len(bad):
        import pandas as pd
        return pd.DataFrame(columns=["_norm", "rows", "seasons"])
    return (bad.groupby("_norm")
            .agg(rows=("_norm", "size"),
                 seasons=("season", lambda s: sorted(set(s))))
            .sort_values("rows", ascending=False).reset_index())


def actual_stat(season, week, player_name, col, seasons=None, position=None):
    """One settled stat for grading. The number, or None.

    Returns None, never a guess, when:
      - the player-week has no stat row at all (did not play)
      - the name resolves to more than one player_id (COLLISION GUARD)
      - the requested column is absent from nflverse

    Returns 0.0 rather than None when the row exists and the stat is null,
    because a player with a stat row was ACTIVE and a blank stat is a real
    zero. Measured true for receptions (0 nulls in 22,540 rows) and QB
    rushing (0 nulls in 2,817), so this is insurance rather than a fix.

    position: a single position string, or a LIST of them, narrowing the
    frame before the name match. This makes cross-position collisions
    unreachable rather than merely guarded, and it matters for more than
    tidiness: receiving.build_dataset and receptions.build_dataset filter to
    WR/TE/RB, which already excludes Michael Carter the CORNERBACK, so those
    markets grade Michael Carter the running back correctly today. Calling
    this function unfiltered would make the guard fire and return None, which
    would be a regression. Pass ["WR", "TE", "RB"] to preserve the behaviour
    and gain the fallback and the zero handling.

    Leave it None only where the population is genuinely mixed, as in rushing
    (running backs, receivers taking carries, quarterbacks).
    """
    import pandas as pd
    ps = _stats_frame(tuple(seasons) if seasons else _stat_seasons())
    if col not in ps.columns:
        return None

    wk = ps[(ps["season"] == season) & (ps["week"] == week)]
    if position is not None and "position" in wk.columns:
        want = [position] if isinstance(position, str) else list(position)
        wk = wk[wk["position"].isin(want)]
    if not len(wk):
        return None

    m = wk[wk["player_display_name"] == player_name]
    if not len(m):
        # FanDuel and nflverse spell the same player differently, so fall
        # back to the normalized form. norm_join_name is vectorised and needs
        # a Series on both sides.
        target = norm_join_name(pd.Series([player_name])).iloc[0]
        m = wk[wk["_norm"] == target]
    if not len(m):
        return None

    # Count identities on the MATCHED rows, after any position narrowing, so
    # a filter that resolves the ambiguity is allowed to resolve it.
    if m["_pid"].nunique() > 1:
        # Ambiguous. A missing grade is a visible gap; a wrong grade is a
        # silent error in the table that is supposed to be ground truth.
        return None

    val = m.iloc[0][col]
    return 0.0 if pd.isna(val) else float(val)


def current_week(season):
    """The week currently IN PROGRESS, or the most recently played. None if
    the season has not started.

    WHY THIS EXISTS. available_weeks returns the played weeks PLUS the next
    one, so the board can project an upcoming game. app.py then selected
    `index=len(weeks) - 1`, which is always that appended future week. On
    September 25 2026, with one Week 3 game complete, the board opened on
    Week 4: no lines, no snapshots, an empty board, and the Line Movement
    game picker came up blank so every game had to be selected by hand.

    "Played" is judged from the SCHEDULE rather than from any market module,
    so the answer is the same for all six boards and does not depend on
    whether a particular player has a stat row yet. A week counts as started
    once ANY game in it has a result, which is why a Thursday-night opener
    puts you in that week rather than the previous one.
    """
    import pandas as pd
    sched = load_schedules([season])
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    col = next((c for c in ("result", "home_score", "away_score")
                if c in sched.columns), None)
    if col is None:
        return None
    done = sched[pd.to_numeric(sched[col], errors="coerce").notna()]
    if not len(done):
        return None
    wk = pd.to_numeric(done["week"], errors="coerce").dropna()
    return int(wk.max()) if len(wk) else None


def default_week_index(weeks, season):
    """Index into `weeks` that app.py should preselect.

    Prefers the current week. Falls back to the last entry, which is the old
    behaviour, when the current week is not in the list (a completed season,
    or a schedule that failed to load).
    """
    if not weeks:
        return 0
    cur = None
    try:
        cur = current_week(season)
    except Exception:
        cur = None
    if cur is not None and cur in weeks:
        return weeks.index(cur)
    return len(weeks) - 1
