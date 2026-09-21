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