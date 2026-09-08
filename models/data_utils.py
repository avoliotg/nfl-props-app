"""
Shared data-loading helpers that gracefully skip seasons whose data
doesn't exist yet (e.g. a future season before it has been played).
"""
import nflreadpy as nfl


def _try_seasons(loader, seasons):
    """Call an nflreadpy loader one season at a time; skip any that 404
    (data not posted yet). Returns a combined pandas DataFrame of the
    seasons that succeeded, plus the list of seasons actually loaded."""
    import pandas as pd
    frames = []
    loaded = []
    for s in seasons:
        try:
            df = loader([s]).to_pandas()
            if len(df) > 0:
                frames.append(df)
                loaded.append(s)
        except Exception:
            # season data not available yet (e.g. future season) — skip it
            continue
    if frames:
        return pd.concat(frames, ignore_index=True), loaded
    # nothing loaded — return an empty frame
    return pd.DataFrame(), loaded


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

    Takes a pandas Series, returns a pandas Series.
    """
    out = (s.astype(str)
           .str.replace("[.'\u2019\u2018`,-]", "", regex=True)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.lower())
    out = out.str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
    return out.replace(_NAME_ALIASES)