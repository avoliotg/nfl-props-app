"""
PHASE 2: the evaluation harness.

This is the gate for every model change from here on. It answers one question
on real market prices instead of 596 graded bets:

    does the model's disagreement with the line predict the outcome?

by fitting, per market and pooled:

    (actual - line) = alpha + beta * (projection - line) + error

  beta ~ 1   the projection carries the information, the line adds nothing
  beta ~ 0   the line is sufficient, the deviations are noise
  beta ~ 0.3 only 30 percent of the apparent edge is real

WHAT CHANGED IN THIS VERSION, AND WHY

  1. EMPIRICAL OVER RATE, not an inferred one.
     alpha is mean(actual - line). Books charge flat -113 both sides on
     yardage, which asserts P(over) is about 0.5, so the line sits near the
     MEDIAN. For a right-skewed outcome the mean sits above the median, so a
     positive alpha is expected by construction and is NOT a betting edge.
     Converting alpha to an over rate by dividing by sigma assumes symmetry,
     which is the one assumption that is false. So the over rate is now
     COUNTED directly, with a game-clustered standard error. If receiving
     comes back near 0.500 despite alpha +3.08, the alpha edge is dead and
     plan item 4.8 closes. Note that alpha remains a VALID PRICING parameter
     (Phase 5.3): setting the distribution mean to line + alpha puts the
     median near the line, which is what makes P(over) land near 0.5. Valid
     for pricing, invalid as a signal.

  2. OUT OF SAMPLE BLEND RMSE.
     The old version fitted beta on the same rows it then scored, and beta = 0
     is nested inside that fit, so the blend almost could not lose to the line.
     The reported margin was what beta and R-squared algebraically imply, not
     independent confirmation. beta is now fitted by 5-fold CV grouped on
     event_id, and both the honest and the in-sample numbers are printed side
     by side.

  3. PER-SEASON POOLING IN SIGMA UNITS.
     The old per-season rows pooled markets in raw units, so passing yards
     (sd ~69) swamped receptions (sd ~2) and the coefficient was meaningless.
     Scales are computed once per market across all seasons, so differences
     between seasons are not driven by changing scales.

  4. TRAINING COUNTS REPORTED.
     Walk-forward on 2023 trains on 2022 alone; 2026 trains on four seasons.
     A beta that rises with training size is NOT instability, and the two look
     identical unless the counts are visible. n_train is now carried through
     the join and printed per season and market. --fixed-train holds the model
     constant across seasons, which is the clean test of whether beta varies
     by season rather than by training size.

  5. HONEST IN-SAMPLE COMPARISON.
     The old version hardcoded "train on season <= 2024", so on --all-seasons
     it trained on three seasons for 2023 against one for walk-forward and the
     gap mixed the leak with a 3x size difference. It now trains on <= s
     against < s, a one-season difference in both arms.

  6. LEAD TIME FILTER.
     The 2023-2025 backfill computed the closing timestamp from a September
     enumeration, so games whose kickoff later moved got snapshots at the
     wrong time. 38 events are outside a 20 to 90 minute window, 5 of them
     captured AFTER kickoff, which are in-play prices. Gate with
     --lead-window before deciding whether to re-pull them.

  7. BEST-LINE PLACEBO.
     line_best is selected USING the projection and then appears in both dev
     and out with a negative sign, which inflates beta mechanically. Same
     shared-term structure as the TD movement artifact. --placebo N shuffles
     projections, reselects line_best, and refits; the placebo mean is the
     artifact and the honest value is the difference.

  8. SHOPPING VALUE AS A REALIZED WIN RATE.
     beta against line_best is the wrong instrument for shopping: shopping
     does not improve the forecast, it improves the price. The side is now
     fixed using the consensus line, then the win rate is counted at each
     source. No shared term, and it answers the operational question.

  9. LOCAL CACHE. 354,554 rows at 1,000 a page is 355 round trips, and Phase 3
     runs this before and after every fix. --cache stores the fetched rows and
     appends missing seasons rather than re-paging.

CAVEAT THAT THIS VERSION DOES NOT FIX

  build_dataset() is called ONCE over all seasons, then split by season. So
  walk-forward applies to the model COEFFICIENTS only. Any feature built from
  a full-sample statistic (a player career mean, a league-wide normalisation)
  leaks future information into earlier seasons regardless of the split.
  Backward-looking shifted rolling features are fine. Auditing this is the
  prerequisite for item 3, switching to each module's own load_model.

Run from the repo root with the venv active:
    python eval_harness.py --season 2025
    python eval_harness.py --all-seasons --in-sample --cache lines_cache.parquet
    python eval_harness.py --all-seasons --lead-window 20,90
    python eval_harness.py --all-seasons --fixed-train 2022
    python eval_harness.py --season 2025 --placebo 200
    python eval_harness.py --seasons 2026 --weeks 1,2

Credentials come from .streamlit/secrets.toml or the environment.
"""
import argparse
import os
import sys
from getpass import getpass

import numpy as np
import pandas as pd

SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
PAGE = 1000

# app market -> (module name, nflverse actual column)
MARKET_SPEC = {
    "receiving": ("receiving", "receiving_yards"),
    "receptions": ("receptions", "receptions"),
    "rushing": ("rushing", "rushing_yards"),
    "qb_passing": ("qb_passing", "passing_yards"),
    "qb_rushing": ("qb_rushing", "rushing_yards"),
}
# qb_rushing has NO STORED LABEL, and this is a property of the data source
# rather than an oversight. The Odds API uses one market key for rushing yards
# regardless of who is carrying, and the FanDuel transcription schema does the
# same, so QB props sit in the table as market = 'rushing'. MARKET_MAP in db.py
# is not missing an entry; there is nothing to put in it.
#
# FIXED September 24 2026 (plan item 3.5). The label is DERIVED after the fact
# from nflverse position by data_utils.split_qb_rushing, which is the only
# place the information exists. Deriving rather than relabelling at write time
# means it applies retroactively to all 184,782 cached rows, and it leaves the
# app's storage and every query against it untouched.
#
# Measured on the full cache: 11,126 of 33,885 rushing rows are QBs (32.8
# percent, stable across all four seasons), 99.98 percent of rushing rows
# resolve to a position, and the reassigned rows join qb_rushing's own dataset
# at 99.99 percent on season, week and name.
#
# FETCH_MARKET maps a DERIVED market to the label it is STORED under. Anything
# absent is stored under its own name. This exists because fetch_lines filters
# the query with .in_("market", ...) and load_lines filters the cache the same
# way, so requesting a derived market directly would return zero rows.
FETCH_MARKET = {
    "qb_rushing": "rushing",
}

DEFAULT_MARKETS = ("receiving", "receptions", "rushing", "qb_passing",
                   "qb_rushing")
ALL_SEASONS = (2023, 2024, 2025, 2026)
LINE_SOURCES = ("fanduel", "consensus", "best")

# flat -113 both sides is what FanDuel charges on yardage, so a bet needs
# 113 / 213 = 0.5305 to break even. Used only for the shopping table.
BREAKEVEN = 113.0 / 213.0


# ------------------------------------------------------------------- plumbing

def _secrets():
    if not os.path.exists(SECRETS_PATH):
        return {}
    try:
        import tomllib
        with open(SECRETS_PATH, "rb") as fh:
            return tomllib.load(fh)
    except Exception as e:
        print(f"  could not parse the secrets file: {e}")
        return {}


def connect(sec):
    from supabase import create_client
    url = sec.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = sec.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not (url and key):
        print("  missing Supabase credentials")
        sys.exit(1)
    email = (os.environ.get("OPAL_EMAIL") or sec.get("OPAL_EMAIL")
             or sec.get("ADMIN_EMAIL") or input("email: ").strip())
    pw = (os.environ.get("OPAL_PASSWORD") or sec.get("OPAL_PASSWORD")
          or getpass("password (hidden): "))
    client = create_client(url, key)
    res = client.auth.sign_in_with_password({"email": email, "password": pw})
    if not res.user:
        print("  login failed")
        sys.exit(1)
    print(f"  authenticated as {res.user.email}")
    return client


# ---------------------------------------------------------------------- cache

def _cache_read(path):
    """Read a cached frame. Parquet if pyarrow is present, pickle otherwise."""
    if not path:
        return None
    for p, reader in ((path, pd.read_parquet), (path + ".pkl", pd.read_pickle)):
        if os.path.exists(p):
            try:
                df = reader(p)
                print(f"  cache hit: {len(df)} rows from {p}")
                return df
            except Exception as e:
                print(f"  cache at {p} unreadable ({e}), ignoring")
    return None


def _cache_write(path, df):
    if not path:
        return
    try:
        df.to_parquet(path, index=False)
        print(f"  cache written: {len(df)} rows to {path}")
    except Exception:
        df.to_pickle(path + ".pkl")
        print(f"  cache written: {len(df)} rows to {path}.pkl "
              f"(parquet unavailable, install pyarrow for the faster path)")


# ------------------------------------------------------------------- fetching

def fetch_lines(client, seasons, markets):
    """Historical closing lines, paged, with progress so it never looks hung."""
    frames = []
    cols = ("season,week,market,player,book,line,over_odds,under_odds,"
            "event_id,commence_time,captured_at,snapshot_label")
    for season in seasons:
        rows = []
        start = 0
        while True:
            resp = (client.table("historical_lines").select(cols)
                    .eq("season", int(season))
                    .eq("snapshot_label", "closing")
                    .in_("market", list(markets))
                    .order("id", desc=False)
                    .range(start, start + PAGE - 1).execute())
            batch = resp.data or []
            rows.extend(batch)
            print(f"\r    {season}: {len(rows)} rows", end="", flush=True)
            if len(batch) < PAGE:
                break
            start += PAGE
        print()
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_lines(client_factory, seasons, markets, cache_path, refresh):
    """Cache-aware fetch. Appends only the seasons the cache is missing.

    The cache is keyed on nothing: it just holds rows, and the season and
    market columns are used to work out what is already present. That means a
    cache built for 2025 receiving is reused as-is and only the gap is pulled.
    """
    cached = None if refresh else _cache_read(cache_path)
    have_pairs = set()
    if cached is not None and len(cached):
        have_pairs = set(zip(cached["season"].astype(int), cached["market"]))

    missing = {}
    for s in seasons:
        need = [m for m in markets if (int(s), m) not in have_pairs]
        if need:
            missing[s] = need
    if missing:
        client = client_factory()
        print("\nfetching historical closing lines")
        for s, mk in missing.items():
            print(f"  season {s}, markets {mk}")
        fetched = pd.concat(
            [fetch_lines(client, [s], mk) for s, mk in missing.items()],
            ignore_index=True) if missing else pd.DataFrame()
        if cached is not None and len(cached):
            combined = pd.concat([cached, fetched], ignore_index=True)
        else:
            combined = fetched
        if cache_path and len(combined):
            _cache_write(cache_path, combined)
    else:
        print("\nall requested seasons and markets already cached")
        combined = cached

    if combined is None or not len(combined):
        return pd.DataFrame()
    keep = combined["season"].astype(int).isin([int(s) for s in seasons]) & \
        combined["market"].isin(list(markets))
    return combined[keep].reset_index(drop=True)


# ------------------------------------------------------------ line aggregation

def collapse_books(lines):
    """One row per (season, week, market, player) with three line sources.

    Eight books quote the same prop at different numbers, so a single 'the
    line' does not exist. Each source answers a different question:
      fanduel    what was actually available where the money is
      consensus  the market's best estimate, robust to one stale book
      best       the shopping edge, taken AFTER the model picks a side
    'best' depends on the model's side so only min and max are carried here.

    lead_min is added: minutes between the snapshot and kickoff. Negative
    means the snapshot was taken after the game started, which is an in-play
    price rather than a close.
    """
    lines = lines.copy()
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines = lines.dropna(subset=["line"])

    for c in ("commence_time", "captured_at"):
        if c in lines.columns:
            lines[c] = pd.to_datetime(lines[c], errors="coerce", utc=True)

    keys = ["season", "week", "market", "player"]
    grp = lines.groupby(keys, sort=False)
    agg = {
        "n_books": ("book", "nunique"),
        "line_consensus": ("line", "median"),
        "line_min": ("line", "min"),
        "line_max": ("line", "max"),
        "event_id": ("event_id", "first"),
        "commence_time": ("commence_time", "first"),
    }
    if "captured_at" in lines.columns:
        agg["captured_at"] = ("captured_at", "min")
    out = grp.agg(**agg).reset_index()

    if "captured_at" in out.columns:
        out["lead_min"] = (
            (out["commence_time"] - out["captured_at"]).dt.total_seconds() / 60.0)
    else:
        out["lead_min"] = np.nan

    fd = lines[lines["book"] == "fanduel"][
        keys + ["line", "over_odds", "under_odds"]]
    fd = fd.rename(columns={"line": "line_fanduel"})
    # a closing snapshot should give one FanDuel row per prop; if alternate
    # lines ever appear this takes the first rather than failing silently
    dupes = int(fd.duplicated(subset=keys).sum())
    if dupes:
        print(f"  note: {dupes} duplicate FanDuel rows collapsed to the first")
    fd = fd.drop_duplicates(subset=keys)
    out = out.merge(fd, on=keys, how="left")
    out["book_spread"] = out["line_max"] - out["line_min"]
    return out


def report_lead(props, window):
    """Print the lead-time distribution and optionally filter on it."""
    lead = props["lead_min"]
    if lead.isna().all():
        print("  lead time unavailable (no captured_at in the cache); "
              "refresh the cache with --refresh to enable the filter")
        return props
    print(f"\n  snapshot lead time, minutes before kickoff:")
    print(f"    median {lead.median():.1f}   "
          f"after kickoff {int((lead < 0).sum())} props   "
          f"under 20 {int(((lead >= 0) & (lead < 20)).sum())}   "
          f"over 90 {int((lead > 90).sum())}")
    if window is None:
        print("    no filter applied. Pass --lead-window 20,90 to gate it.")
        return props
    lo, hi = window
    keep = lead.between(lo, hi)
    print(f"    filtering to [{lo}, {hi}]: keeping {int(keep.sum())} of "
          f"{len(props)} props ({100.0 * keep.mean():.1f}%)")
    print("    NOTE: the excluded events are disproportionately prime time, "
          "week 18 and playoff")
    print("    games, which are the sharpest-priced on the board. Dropping "
          "them is not a random deletion.")
    return props[keep].copy()


# --------------------------------------------------------------- model scoring

def score_market(market, seasons, mode="walk_forward", fixed_train=None,
                 population="all"):
    """Projections for every historical player-week in this market.

    mode:
      walk_forward  for season S train on seasons strictly before S. Honest.
      in_sample     for season S train on seasons up to and INCLUDING S. The
                    gap against walk-forward is the overfitting measure, and
                    the two arms differ by exactly one season of training data
                    so the gap is not confounded with training size.
      fixed         train once on seasons <= fixed_train and score every
                    season with that one model. Holds the model constant, so
                    beta differences across seasons are about the seasons and
                    the market rather than about how much data the model had.

    population, and this matters as much as the TRAINING population:
      all    score build_all_rows() where the module exposes it, else
             build_dataset(). FanDuel posts a rushing line on every active back
             before kickoff, so the population the line was set for is every
             active RB player-week. models.rushing.build_dataset filters
             carries >= 5, which is an IN-GAME result, so scoring only those
             rows conditions the EVALUATION on the single fact that most
             determines the outcome. That is what produced a rushing tail of 98
             percent overs at a median line of 13.5 and a holdout ROI of +0.48
             in a market whose beta is 0.21. Fixing the model's features and
             training gate did NOT fix this, because the filter still decides
             which rows get scored and joined.
      board  score build_dataset(), matching the app display. Use only to
             reproduce the old numbers.

    Relies on each module exposing build_dataset, load_model and LEAN_FEATS or
    FEATS. Returns season, week, player, projection, actual, n_train.
    """
    from sklearn.linear_model import LinearRegression
    import importlib

    mod_name, actual_col = MARKET_SPEC[market]
    try:
        mod = importlib.import_module(f"models.{mod_name}")
    except Exception as e:
        print(f"    could not import models.{mod_name}: {e}")
        return pd.DataFrame()

    feats = getattr(mod, "LEAN_FEATS", None) or getattr(mod, "FEATS", None)
    if not feats:
        print(f"    {mod_name} exposes no feature list, skipping")
        return pd.DataFrame()

    builder = mod.build_dataset
    builder_name = "build_dataset"
    if population == "all" and hasattr(mod, "build_all_rows"):
        builder = mod.build_all_rows
        builder_name = "build_all_rows (unfiltered)"
    try:
        df = builder()
    except Exception as e:
        print(f"    {builder_name} failed for {mod_name}: {e}")
        return pd.DataFrame()
    print(f"    scoring population: {builder_name}, {len(df)} rows")

    need = list(feats) + [actual_col, "season", "week", "player_display_name"]
    have = [c for c in need if c in df.columns]
    if actual_col not in df.columns:
        print(f"    {mod_name} has no column {actual_col}, skipping")
        return pd.DataFrame()

    # how many rows the dropna costs, and whether they are zero-outcome rows.
    # Dropping rows where an active player recorded a null stat where 0 is
    # correct is the Phase 3.4 bug appearing inside the measuring instrument,
    # and it would bias alpha UPWARD by removing the worst outcomes.
    pre = len(df)
    null_actual = int(df[actual_col].isna().sum())
    df = df[have].dropna(subset=list(feats) + [actual_col]).copy()
    if null_actual:
        print(f"    {market}: {null_actual} of {pre} rows have a null "
              f"{actual_col} and were dropped")
        print(f"      if those are active players who recorded zero, this "
              f"biases alpha upward (Phase 3.4)")

    out = []
    for season in seasons:
        if mode == "walk_forward":
            train = df[df["season"] < season]
        elif mode == "in_sample":
            train = df[df["season"] <= season]
        elif mode == "fixed":
            train = df[df["season"] <= int(fixed_train)]
        else:
            raise ValueError(f"unknown mode {mode}")
        test = df[df["season"] == season]
        if len(train) < 300 or len(test) == 0:
            print(f"    {market} {season}: train {len(train)}, "
                  f"test {len(test)} -> skipped")
            continue
        model = LinearRegression().fit(train[list(feats)], train[actual_col])
        pred = model.predict(test[list(feats)])
        out.append(pd.DataFrame({
            "season": test["season"].to_numpy(),
            "week": test["week"].to_numpy(),
            "player": test["player_display_name"].to_numpy(),
            "projection": pred,
            "actual": test[actual_col].to_numpy(),
            "n_train": len(train),
            "train_seasons": train["season"].nunique(),
        }))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["market"] = market
    return res


# --------------------------------------------------------------------- the fit

def ols_clustered(x, y, cluster):
    """y = a + b*x with standard errors clustered on `cluster`.

    Props in one game share information (game script, weather, pace), so
    independent-row standard errors are too narrow. Clustering fixes that.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(cluster)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    n = len(x)
    if n < 20 or np.allclose(x, x[0]):
        return None
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    xtx_inv = np.linalg.inv(X.T @ X)

    # cluster-robust meat matrix
    meat = np.zeros((2, 2))
    order = np.argsort(c)
    Xs, rs, cs = X[order], resid[order], c[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(starts) - 1
    for i in range(g):
        sl = slice(starts[i], starts[i + 1])
        u = Xs[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - 2, 1))
    cov = xtx_inv @ meat @ xtx_inv * scale
    se = np.sqrt(np.diag(cov))

    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "n": n, "clusters": g,
        "alpha": coef[0], "beta": coef[1],
        "se_alpha": se[0], "se_beta": se[1],
        "resid_sd": np.sqrt((resid @ resid) / (n - 2)),
        "r2": 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan,
    }


def cluster_mean(y, cluster):
    """Mean of y with a standard error clustered on `cluster`.

    Used for the over rate. A plain binomial standard error would be too
    narrow because props in one game move together, and the over rate is the
    quantity that decides whether the alpha finding is real.
    """
    y = np.asarray(y, float)
    c = np.asarray(cluster)
    ok = np.isfinite(y)
    y, c = y[ok], c[ok]
    n = len(y)
    if n < 20:
        return None
    m = y.mean()
    resid = y - m
    order = np.argsort(c)
    rs, cs = resid[order], c[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(starts) - 1
    tot = 0.0
    for i in range(g):
        u = rs[starts[i]:starts[i + 1]].sum()
        tot += u * u
    var = tot / (n * n) * (g / max(g - 1, 1))
    return {"n": n, "clusters": g, "mean": m, "se": float(np.sqrt(var))}


def report_fit(label, f):
    if f is None:
        print(f"  {label:<30} too few rows")
        return
    lo = f["beta"] - 1.96 * f["se_beta"]
    hi = f["beta"] + 1.96 * f["se_beta"]
    t0 = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
    t1 = (f["beta"] - 1) / f["se_beta"] if f["se_beta"] > 0 else np.nan
    print(f"  {label:<30}{f['n']:>7}{f['clusters']:>7}{f['beta']:>8.3f}"
          f"{f['se_beta']:>7.3f}  [{lo:>+6.3f},{hi:>+7.3f}]{t0:>8.2f}"
          f"{t1:>8.2f}{f['alpha']:>8.2f}{f['resid_sd']:>8.1f}")


def header():
    print(f"  {'':<30}{'n':>7}{'games':>7}{'beta':>8}{'SE':>7}"
          f"  {'95% CI':<16}{'t(b=0)':>7}{'t(b=1)':>8}{'alpha':>8}{'residSD':>8}")


# ------------------------------------------------------------ sigma-unit pool

def sigma_scales(sub):
    """One scale per market, computed across ALL seasons in the run.

    Computing the scale within each season would let a season with a wider
    outcome distribution look like a season with a different beta, which is
    exactly the confusion the per-season table is meant to resolve.
    """
    scale = {}
    for m, g in sub.groupby("market"):
        s = g["out"].std()
        scale[m] = s if s and s > 0 else 1.0
    return scale


def pooled_fit(sub, scale):
    dz = sub["dev"] / sub["market"].map(scale)
    oz = sub["out"] / sub["market"].map(scale)
    return ols_clustered(dz, oz, sub["event_id"])


# ------------------------------------------------------------- the over rate

def over_rate_table(sub, source):
    """Count P(actual > line). This is the test of the alpha finding.

    alpha is mean(actual - line), which for a right-skewed outcome is positive
    whenever the line sits at the median, and books put it there by charging
    flat -113 both sides. So alpha says nothing about which side wins. The
    over rate does.
    """
    print(f"\n  EMPIRICAL OVER RATE at line_{source} "
          f"(breakeven at -113 is {BREAKEVEN:.4f})")
    print(f"  {'market':<14}{'n':>7}{'push':>6}{'over rate':>11}{'SE':>8}"
          f"{'95% CI':>18}{'vs breakeven':>14}{'alpha':>8}")
    rows = []
    for m, g in sub.groupby("market"):
        line = g[f"line_{source}"].to_numpy(float)
        act = g["actual"].to_numpy(float)
        push = np.isclose(act, line)
        live = ~push & np.isfinite(line) & np.isfinite(act)
        if live.sum() < 20:
            print(f"  {m:<14}{int(live.sum()):>7}  too few")
            continue
        won = (act[live] > line[live]).astype(float)
        cm = cluster_mean(won, g["event_id"].to_numpy()[live])
        if cm is None:
            continue
        lo = cm["mean"] - 1.96 * cm["se"]
        hi = cm["mean"] + 1.96 * cm["se"]
        edge = cm["mean"] - BREAKEVEN
        alpha = float(np.mean(act[live] - line[live]))
        print(f"  {m:<14}{cm['n']:>7}{int(push.sum()):>6}"
              f"{cm['mean']:>11.4f}{cm['se']:>8.4f}"
              f"  [{lo:>6.4f},{hi:>7.4f}]{edge:>+14.4f}{alpha:>+8.2f}")
        rows.append((m, cm["mean"], cm["se"], alpha))
    print("\n    An over rate near 0.5000 alongside a large positive alpha "
          "means the alpha is")
    print("    distributional skew, not an edge. Plan item 4.8 closes on "
          "that reading.")
    print("    An over rate above the breakeven with a CI excluding it would "
          "be a real,")
    print("    model-free edge. Pushes are excluded from the rate and counted "
          "separately.")
    return rows


# --------------------------------------------------------------- RMSE, honest

def _folds(event_ids, k=5):
    """Assign folds by event so a game never straddles the split."""
    uniq = pd.unique(pd.Series(event_ids).astype(str))
    lookup = {e: i % k for i, e in enumerate(uniq)}
    return pd.Series(event_ids).astype(str).map(lookup).to_numpy()


def rmse_table(sub, k=5):
    """RMSE of four predictors, with beta fitted OUT of sample.

    The old version fitted beta on the same rows it scored. beta = 0 is nested
    in that fit, so the blend essentially could not lose and the margin was
    what beta and R-squared already imply. Here beta and alpha come from k-1
    folds and are applied to the held-out fold, grouped on event_id so props
    in one game stay together.
    """
    print(f"\n  {'market':<14}{'n':>7}{'RMSE line':>11}{'RMSE proj':>11}"
          f"{'blend oos':>11}{'blend ins':>11}{'beta ins':>10}"
          f"{'beta oos':>10}")
    for m, g in sub.groupby("market"):
        y = g["out"].to_numpy(float)
        x = g["dev"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        y, x = y[ok], x[ok]
        ev = g["event_id"].to_numpy()[ok]
        if len(y) < 100:
            continue

        ins = ols_clustered(x, y, ev)
        b_ins = float(np.clip(ins["beta"], 0, 1)) if ins else 0.0

        fold = _folds(ev, k)
        pred = np.full(len(y), np.nan)
        betas = []
        for f in range(k):
            tr, te = fold != f, fold == f
            if tr.sum() < 50 or te.sum() == 0:
                continue
            fit = ols_clustered(x[tr], y[tr], ev[tr])
            if fit is None:
                continue
            b = float(np.clip(fit["beta"], 0, 1))
            betas.append(b)
            pred[te] = fit["alpha"] + b * x[te]
        have = np.isfinite(pred)
        rmse_oos = np.sqrt(np.mean((y[have] - pred[have]) ** 2)) \
            if have.any() else np.nan
        rmse_ins = np.sqrt(np.mean((y - b_ins * x) ** 2))
        print(f"  {m:<14}{len(y):>7}"
              f"{np.sqrt(np.mean(y ** 2)):>11.2f}"
              f"{np.sqrt(np.mean((y - x) ** 2)):>11.2f}"
              f"{rmse_oos:>11.2f}{rmse_ins:>11.2f}"
              f"{b_ins:>10.3f}{np.mean(betas) if betas else np.nan:>10.3f}")
    print("\n    'blend oos' is the honest number. If it does not beat "
          "'RMSE line', the")
    print("    blend adds nothing usable even where beta is significantly "
          "positive, because")
    print("    a real but tiny R-squared moves RMSE by almost nothing.")


# ------------------------------------------------------------------- shopping

def shopping_table(joined):
    """Realized win rate at each line source, with the SIDE held fixed.

    Shopping does not improve the forecast, it improves the price, so beta
    against line_best is the wrong instrument and is inflated anyway because
    line_best is chosen using the projection and then appears on both sides of
    the regression. Here the side is chosen once from the consensus line, then
    the win rate is counted at each source. No shared term.
    """
    print("\n" + "=" * 96)
    print("SHOPPING VALUE: same side, different price")
    print("=" * 96)
    need = ["line_fanduel", "line_consensus", "line_best"]
    sub = joined.dropna(subset=need + ["actual", "projection"]).copy()
    if len(sub) < 200:
        print(f"  only {len(sub)} rows with all three sources, skipping")
        return
    wants_over = sub["projection"] > sub["line_consensus"]
    print(f"  {'market':<14}{'n':>7}{'side':>8}"
          f"{'fanduel':>10}{'consensus':>11}{'best':>9}{'best - fd':>11}")
    for m, g in sub.groupby("market"):
        over = wants_over.loc[g.index].to_numpy()
        act = g["actual"].to_numpy(float)
        res = {}
        for src in ("fanduel", "consensus", "best"):
            ln = g[f"line_{src}"].to_numpy(float)
            push = np.isclose(act, ln)
            win = np.where(over, act > ln, act < ln).astype(float)
            win = win[~push]
            cm = cluster_mean(win, g["event_id"].to_numpy()[~push])
            res[src] = cm["mean"] if cm else np.nan
        print(f"  {m:<14}{len(g):>7}{f'{over.mean():.0%} over':>8}"
              f"{res['fanduel']:>10.4f}{res['consensus']:>11.4f}"
              f"{res['best']:>9.4f}{res['best'] - res['fanduel']:>+11.4f}")
    print("\n    'best - fd' is the shopping gain in win probability on the "
          "same bet. At -113")
    print("    a gain of 0.03 is roughly 3 percent of turnover. Caveat: "
          "'best' spans eight")
    print("    books including ones with stale or limited prices, so part of "
          "this is not")
    print("    actually available to you. Restrict to books you hold accounts "
          "with to size it.")


# -------------------------------------------------------------- best placebo

def placebo_best(joined, n_shuffles, seed=0):
    """How much of the 'best line' beta is the shared-term artifact.

    line_best is selected using the projection and then appears in both dev
    and out with a negative sign, so Var(line_best) enters the covariance and
    pushes beta up mechanically. Same structure as td_reversion_check.py.

    WHAT IS PERMUTED, AND WHY IT MATTERS. The permutation is of the DEVIATION
    (projection - line_consensus), with the projection then rebuilt as
    line_consensus + permuted deviation. Permuting the projection directly
    does NOT work: it breaks the pairing between a projection and its own
    line, so placebo deviations come out several times wider than real ones,
    the selection mechanism changes size, and the placebo under-states the
    artifact badly. Verified on synthetic data with a known-zero signal, where
    permuting the projection recovered only a seventh of the artifact while
    permuting the deviation recovered it in full.
    """
    print("\n" + "=" * 96)
    print(f"BEST-LINE PLACEBO ({n_shuffles} shuffles)")
    print("=" * 96)
    sub = joined.dropna(
        subset=["line_consensus", "line_min", "line_max", "projection",
                "actual"]).copy()
    if len(sub) < 200:
        print("  too few rows")
        return
    rng = np.random.default_rng(seed)
    cons = sub["line_consensus"].to_numpy(float)
    dev0 = sub["projection"].to_numpy(float) - cons
    scale_ref = None

    def fit_with(dev_vec):
        s = sub.copy()
        s["projection"] = cons + dev_vec
        over = dev_vec > 0
        s["line_best"] = np.where(over, s["line_min"], s["line_max"])
        s["dev"] = s["projection"] - s["line_best"]
        s["out"] = s["actual"] - s["line_best"]
        sc = scale_ref if scale_ref is not None else sigma_scales(s)
        return pooled_fit(s, sc), sc

    base, scale_ref = fit_with(dev0)
    if base is None:
        print("  base fit failed")
        return
    print(f"  observed pooled beta against line_best: {base['beta']:+.4f} "
          f"(SE {base['se_beta']:.4f})")

    betas = []
    grp = sub.groupby(["market", "season"]).indices
    for i in range(n_shuffles):
        dev = dev0.copy()
        for _, idx in grp.items():
            dev[idx] = rng.permutation(dev[idx])
        f, _ = fit_with(dev)
        if f is not None:
            betas.append(f["beta"])
        if (i + 1) % 25 == 0:
            print(f"\r    {i + 1}/{n_shuffles}", end="", flush=True)
    print()
    if not betas:
        print("  no placebo fits succeeded")
        return
    b = np.array(betas)
    print(f"  placebo mean {b.mean():+.4f}, sd {b.std():.4f}, "
          f"95th pct {np.percentile(b, 95):+.4f}")
    print(f"  ARTIFACT-CORRECTED best-line beta: "
          f"{base['beta'] - b.mean():+.4f}")
    print(f"  consensus beta for comparison is printed above. If the "
          f"corrected number is")
    print(f"  no higher than the consensus beta, the 0.345 was the artifact "
          f"and there is no")
    print(f"  forecast gain from shopping. The shopping table measures the "
          f"PRICE gain, which")
    print(f"  is the real and separate thing.")


# ------------------------------------------------------------- join diagnostic

def join_diagnostic(props, joined):
    """Which props failed to find a scored player-week, and who they are.

    The known systematic cause is the qb_rushing label bug: player_rush_yds
    covers QBs, those rows store as market = 'rushing', and models.rushing
    filters to RBs, so every QB rushing prop fails silently. If QBs top the
    rushing list below, that is confirmed and it is Phase 3.7 rather than
    'players with no stat row'.
    """
    print("\n  JOIN DIAGNOSTIC")
    key = ["season", "week", "market", "_key"]
    matched = joined[key].drop_duplicates()
    miss = props.merge(matched, on=key, how="left", indicator=True)
    miss = miss[miss["_merge"] == "left_only"]
    print(f"  {'market':<14}{'props':>8}{'joined':>8}{'rate':>8}")
    for m in sorted(props["market"].unique()):
        p = int((props["market"] == m).sum())
        j = p - int((miss["market"] == m).sum())
        print(f"  {m:<14}{p:>8}{j:>8}{100.0 * j / max(p, 1):>7.1f}%")
    for m in sorted(miss["market"].unique()):
        top = miss[miss["market"] == m]["player"].value_counts().head(10)
        if len(top):
            print(f"\n    most frequent unjoined in {m}:")
            print("      " + ", ".join(f"{k} ({v})" for k, v in top.items()))


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", default=None,
                    help="comma separated, e.g. 2024,2025")
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    ap.add_argument("--weeks", default=None,
                    help="comma separated week filter, e.g. 1,2")
    ap.add_argument("--regular-season-only", action="store_true",
                    help="drop weeks above 18, whose population differs")
    ap.add_argument("--lead-window", default=None,
                    help="min,max minutes before kickoff, e.g. 20,90")
    ap.add_argument("--save-rows", default=None,
                    help="write the joined row set to CSV for further analysis")
    ap.add_argument("--in-sample", action="store_true",
                    help="also report the in-sample fit, to measure overfitting")
    ap.add_argument("--fixed-train", type=int, default=None,
                    help="train once on seasons <= this and score all seasons")
    ap.add_argument("--placebo", type=int, default=0,
                    help="shuffles for the best-line placebo, e.g. 200")
    ap.add_argument("--cache", default=None,
                    help="path to a local cache of fetched rows")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cache and re-fetch")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"],
                    help="all = build_all_rows where available (honest); "
                         "board = build_dataset (reproduces the old numbers)")
    args = ap.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in MARKET_SPEC]
    if bad:
        print(f"unknown markets: {bad}")
        sys.exit(1)

    if args.all_seasons:
        seasons = list(ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [2025]

    weeks = None
    if args.weeks:
        weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    window = None
    if args.lead_window:
        parts = [float(p) for p in args.lead_window.split(",")]
        if len(parts) != 2:
            print("--lead-window takes two numbers, e.g. 20,90")
            sys.exit(1)
        window = (parts[0], parts[1])

    print("=" * 96)
    print("EVALUATION HARNESS: model versus the closing line")
    print(f"  seasons {seasons}   markets {markets}")
    if weeks:
        print(f"  weeks {weeks}")
    print("=" * 96)
    print("  NOTE: build_dataset is called once over all seasons, so the")
    print("  walk-forward split applies to model COEFFICIENTS only. Any")
    print("  feature built from a full-sample statistic still leaks.")

    sec = _secrets()
    client_box = {}

    def client_factory():
        if "c" not in client_box:
            client_box["c"] = connect(sec)
        return client_box["c"]

    # Fetch under the STORED market labels, derive the split, then filter to
    # what was actually asked for. Doing it in this order matters: asking the
    # database for 'qb_rushing' returns nothing, because that label does not
    # exist upstream. See the FETCH_MARKET comment.
    fetch_markets = sorted({FETCH_MARKET.get(m, m) for m in markets})
    derived = [m for m in markets if m in FETCH_MARKET]
    if derived:
        print(f"  derived markets {derived} are stored as "
              f"{[FETCH_MARKET[m] for m in derived]}; fetching "
              f"{fetch_markets} and splitting on nflverse position")

    lines = load_lines(client_factory, seasons, fetch_markets, args.cache,
                       args.refresh)
    if lines.empty:
        print("  no lines found. Has fill_weeks.py been run?")
        return

    if derived:
        from models import data_utils as _du
        # The gate is set from measurement, not from a guess: the full cache
        # resolves 99.98 percent of rushing rows to a position, and the eight
        # that fail are abbreviated first initials that cannot be resolved by
        # name at all. 0.98 leaves headroom for a rookie whose nflverse
        # spelling has not been aliased yet, and ABORTS rather than quietly
        # scoring a market whose labels are mostly unresolved.
        lines, split_report = _du.split_qb_rushing(lines, min_match_rate=0.98)
        print(f"  split: {split_report['reassigned']} of "
              f"{split_report['rows_in_from_market']} "
              f"{split_report['from_market']} rows reassigned to "
              f"{split_report['to_market']} "
              f"({split_report['reassigned_share']:.1%}), "
              f"{split_report['distinct_qbs']} distinct players, "
              f"match rate {split_report['match_rate']:.4f}, "
              f"{split_report['unmatched']} unmatched")

        before = len(lines)
        lines = lines[lines["market"].isin(markets)].reset_index(drop=True)
        if len(lines) != before:
            print(f"  dropped {before - len(lines)} rows in fetched-but-not-"
                  f"requested markets")
        if lines.empty:
            print("  no lines left after the market filter")
            return
    print(f"  {len(lines)} prop rows, {lines['book'].nunique()} books")
    if lines["week"].isna().all():
        print("\n  >>> every week is null. Run fill_weeks.py first, or the")
        print("  >>> join to model projections cannot be made.")
        return
    nullw = int(lines["week"].isna().sum())
    if nullw:
        print(f"  dropping {nullw} rows with a null week")
        lines = lines[lines["week"].notna()]

    props = collapse_books(lines)
    print(f"  {len(props)} distinct player-week props")
    print(f"  median books per prop: {props['n_books'].median():.0f}, "
          f"mean book spread: {props['book_spread'].mean():.2f}")

    props = report_lead(props, window)
    if weeks:
        props = props[props["week"].astype(int).isin(weeks)].copy()
        print(f"  week filter: {len(props)} props remain")
    if args.regular_season_only:
        before = len(props)
        props = props[props["week"].astype(int) <= 18].copy()
        print(f"  regular season only: dropped {before - len(props)} props")
    if len(props) < 200:
        print("  too few props after filtering")
        return

    print("\nscoring models walk-forward")
    mode = "fixed" if args.fixed_train else "walk_forward"
    if args.fixed_train:
        print(f"  FIXED training window: seasons <= {args.fixed_train}, "
              f"one model for every scored season")
    scored = []
    for m in markets:
        print(f"  {m}")
        s = score_market(m, seasons, mode=mode, fixed_train=args.fixed_train,
                         population=args.score_population)
        if len(s):
            print(f"    {len(s)} player-weeks scored")
            scored.append(s)
    if not scored:
        print("  nothing scored")
        return
    proj = pd.concat(scored, ignore_index=True)

    # join on normalised names, since the API and nflverse spell differently
    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)

    joined = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual",
              "n_train", "train_seasons"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(joined)} rows "
          f"({100.0 * len(joined) / max(len(props), 1):.1f}% of props)")
    if len(joined) < 200:
        print("  too few joined rows to fit. Check the week fill and name join.")
        return

    join_diagnostic(props, joined)

    # training size per season and market, so a beta that rises with training
    # data is not mistaken for an unstable beta
    print("\n  TRAINING SIZE (walk-forward)")
    tt = (joined.groupby(["market", "season"])
          .agg(scored=("projection", "size"),
               n_train=("n_train", "first"),
               train_seasons=("train_seasons", "first"))
          .reset_index())
    print(f"  {'market':<14}{'season':>8}{'scored':>9}{'n_train':>10}"
          f"{'seasons':>9}")
    for _, r in tt.iterrows():
        print(f"  {r['market']:<14}{int(r['season']):>8}{int(r['scored']):>9}"
              f"{int(r['n_train']):>10}{int(r['train_seasons']):>9}")
    print("    A beta that rises monotonically with n_train is a training-size")
    print("    effect, not instability. Use --fixed-train to separate them.")

    # 'best' line: the most favourable number for the side the model likes.
    # Model above the consensus means it wants the over, so the LOWEST line is
    # best; below means the under, so the HIGHEST. This is what makes the
    # line_best beta an artifact, and why the placebo exists.
    wants_over = joined["projection"] > joined["line_consensus"]
    joined["line_best"] = np.where(wants_over, joined["line_min"],
                                   joined["line_max"])

    if args.save_rows:
        joined.to_csv(args.save_rows, index=False)
        print(f"  wrote {args.save_rows}")

    for src in LINE_SOURCES:
        col = f"line_{src}"
        if col not in joined.columns:
            continue
        sub = joined.dropna(subset=[col]).copy()
        if len(sub) < 200:
            print(f"\n  {src}: only {len(sub)} rows, skipping")
            continue
        sub["dev"] = sub["projection"] - sub[col]
        sub["out"] = sub["actual"] - sub[col]

        print("\n" + "=" * 96)
        print(f"LINE SOURCE: {src}   ({len(sub)} rows)")
        if src == "best":
            print("  REMINDER: line_best is selected using the projection and")
            print("  appears in both dev and out, so this beta is inflated.")
            print("  Run --placebo 200 for the corrected number.")
        print("=" * 96)
        header()
        scale = sigma_scales(sub)
        for m, g in sub.groupby("market"):
            report_fit(m, ols_clustered(g["dev"], g["out"], g["event_id"]))
        # per season, in SIGMA UNITS so markets with different units combine
        for season, g in sub.groupby("season"):
            report_fit(f"pooled {season} (sigma)", pooled_fit(g, scale))
        report_fit("POOLED ALL (sigma units)", pooled_fit(sub, scale))

        over_rate_table(sub, src)
        rmse_table(sub)

    shopping_table(joined)

    if args.placebo:
        placebo_best(joined, args.placebo)

    # overfitting check
    if args.in_sample:
        print("\n" + "=" * 96)
        print("IN SAMPLE COMPARISON (train on seasons <= S, score S)")
        print("=" * 96)
        print("  The walk-forward arm trains on seasons < S. These two differ")
        print("  by exactly one season of training data, so the gap is the")
        print("  leak rather than a training-size difference.")
        ins = []
        for m in markets:
            s = score_market(m, seasons, mode="in_sample",
                             population=args.score_population)
            if len(s):
                ins.append(s)
        if ins:
            ip = pd.concat(ins, ignore_index=True)
            ip["_key"] = data_utils.norm_join_name(ip["player"])
            ip["week"] = ip["week"].astype(int)
            j2 = props.merge(
                ip[["season", "week", "market", "_key", "projection",
                    "actual", "n_train"]],
                on=["season", "week", "market", "_key"], how="inner")
            j2 = j2.dropna(subset=["line_consensus"])
            j2["dev"] = j2["projection"] - j2["line_consensus"]
            j2["out"] = j2["actual"] - j2["line_consensus"]
            header()
            scale2 = sigma_scales(j2)
            for m, g in j2.groupby("market"):
                report_fit(f"{m} (in sample)",
                           ols_clustered(g["dev"], g["out"], g["event_id"]))
            for season, g in j2.groupby("season"):
                report_fit(f"pooled {season} in sample", pooled_fit(g, scale2))
            report_fit("POOLED ALL in sample", pooled_fit(j2, scale2))
            print("\n  A beta clearly higher in sample than walk-forward means")
            print("  the model is fitting noise, and the walk-forward number is")
            print("  the honest one.")

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  t(b=1) tests the CURRENT pricing assumption, that the projection")
    print("  should be used as the distribution mean unchanged. A large")
    print("  negative value rejects it.")
    print("  t(b=0) tests whether the model adds anything at all to the line.")
    print("  OVER RATE, not alpha, decides whether there is a model-free edge.")
    print("  alpha is the mean-versus-median offset, positive by construction")
    print("  for a right-skewed outcome priced at the median. It is a valid")
    print("  PRICING parameter for Phase 5.3 and an invalid betting signal.")
    print("  'blend oos' in the RMSE table is the honest comparison. The")
    print("  in-sample column fits beta on the rows it then scores and")
    print("  therefore almost cannot lose.")
    print("  resid_sd is the correct sigma for pricing around a blended mean,")
    print("  where the shipped sigma was fitted around the model's own")
    print("  projection and therefore includes the projection error.")
    print("  The SHOPPING table, not the line_best beta, measures shopping.")


if __name__ == "__main__":
    main()
