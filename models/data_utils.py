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