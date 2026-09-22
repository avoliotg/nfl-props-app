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

WHAT MAKES THIS DIFFERENT FROM beta_check.py

  1. 340,000 historical prop rows from eight books, not 596 graded bets.
  2. WALK-FORWARD scoring. For each season, models are retrained on only the
     seasons before it, so every row is honestly out of sample. The in-sample
     fit is reported alongside, and the GAP between them measures overfitting
     directly.
  3. THREE LINE SOURCES per player-week, because eight books disagree:
       fanduel    the book actually bet, so the operational number
       consensus  median across books, the best single estimate of the truth
       best       most favourable line for the model's side, the shopping edge
     The differences between these three ARE the line-shopping question.
  4. STANDARD ERRORS CLUSTERED BY GAME. Props in one game share information,
     so treating rows as independent understates uncertainty. event_id makes
     this straightforward.

Run from the repo root with the venv active:
    python eval_harness.py --season 2025
    python eval_harness.py --all-seasons
    python eval_harness.py --all-seasons --markets receiving,receptions
    python eval_harness.py --season 2025 --save-rows eval_2025.csv

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
YARDAGE = tuple(MARKET_SPEC.keys())
LINE_SOURCES = ("fanduel", "consensus", "best")


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


def fetch_lines(client, seasons, markets):
    """Historical closing lines, paged, with progress so it never looks hung."""
    frames = []
    cols = ("season,week,market,player,book,line,over_odds,under_odds,"
            "event_id,commence_time,snapshot_label")
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


# ------------------------------------------------------------ line aggregation

def collapse_books(lines):
    """One row per (season, week, market, player) with three line sources.

    Eight books quote the same prop at different numbers, so a single 'the
    line' does not exist. Each source answers a different question:
      fanduel    what was actually available where the money is
      consensus  the market's best estimate, robust to one stale book
      best       the shopping edge, taken AFTER the model picks a side
    'best' cannot be computed here because it depends on the model's side, so
    the min and max are carried and the choice is made later.
    """
    lines = lines.copy()
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines = lines.dropna(subset=["line"])

    keys = ["season", "week", "market", "player"]
    grp = lines.groupby(keys, sort=False)
    out = grp.agg(
        n_books=("book", "nunique"),
        line_consensus=("line", "median"),
        line_min=("line", "min"),
        line_max=("line", "max"),
        event_id=("event_id", "first"),
        commence_time=("commence_time", "first"),
    ).reset_index()

    fd = lines[lines["book"] == "fanduel"][keys + ["line", "over_odds", "under_odds"]]
    fd = fd.rename(columns={"line": "line_fanduel"})
    fd = fd.drop_duplicates(subset=keys)
    out = out.merge(fd, on=keys, how="left")
    out["book_spread"] = out["line_max"] - out["line_min"]
    return out


# --------------------------------------------------------------- model scoring

def score_market(market, seasons, walk_forward=True):
    """Projections for every historical player-week in this market.

    WALK-FORWARD: for season S the model trains on seasons strictly before S,
    so no row is scored by a model that has seen it. The in-sample variant
    trains once on everything up to 2024, matching the shipped behaviour, and
    the gap between the two measures overfitting.

    Relies on each module exposing build_dataset, load_model and LEAN_FEATS or
    FEATS. Returns a frame of season, week, player, projection, actual.
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

    try:
        df = mod.build_dataset()
    except Exception as e:
        print(f"    build_dataset failed for {mod_name}: {e}")
        return pd.DataFrame()

    need = list(feats) + [actual_col, "season", "week", "player_display_name"]
    have = [c for c in need if c in df.columns]
    if actual_col not in df.columns:
        print(f"    {mod_name} has no column {actual_col}, skipping")
        return pd.DataFrame()
    df = df[have].dropna(subset=list(feats) + [actual_col]).copy()

    out = []
    for season in seasons:
        if walk_forward:
            train = df[df["season"] < season]
        else:
            train = df[df["season"] <= 2024]
        test = df[df["season"] == season]
        if len(train) < 300 or len(test) == 0:
            print(f"    {market} {season}: train {len(train)}, test {len(test)}"
                  f" -> skipped")
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


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default=",".join(YARDAGE))
    ap.add_argument("--save-rows", default=None,
                    help="write the joined row set to CSV for further analysis")
    ap.add_argument("--in-sample", action="store_true",
                    help="also report the in-sample fit, to measure overfitting")
    args = ap.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in MARKET_SPEC]
    if bad:
        print(f"unknown markets: {bad}")
        sys.exit(1)

    if args.all_seasons:
        seasons = [2023, 2024, 2025]
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [2025]

    print("=" * 96)
    print("EVALUATION HARNESS: model versus the closing line")
    print(f"  seasons {seasons}   markets {markets}")
    print("=" * 96)

    sec = _secrets()
    client = connect(sec)

    print("\nfetching historical closing lines")
    lines = fetch_lines(client, seasons, markets)
    if lines.empty:
        print("  no lines found. Has fill_weeks.py been run?")
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

    print("\nscoring models walk-forward")
    scored = []
    for m in markets:
        print(f"  {m}")
        s = score_market(m, seasons, walk_forward=True)
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
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(joined)} rows "
          f"({100.0 * len(joined) / max(len(props), 1):.1f}% of props)")
    if len(joined) < 200:
        print("  too few joined rows to fit. Check the week fill and name join.")
        return

    # 'best' line: the most favourable number for the side the model likes.
    # Model above the consensus means it wants the over, so the LOWEST line is
    # best; below means the under, so the HIGHEST.
    wants_over = joined["projection"] > joined["line_consensus"]
    joined["line_best"] = np.where(wants_over, joined["line_min"], joined["line_max"])

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
        print("=" * 96)
        header()
        for m, g in sub.groupby("market"):
            report_fit(m, ols_clustered(g["dev"], g["out"], g["event_id"]))
        for season, g in sub.groupby("season"):
            report_fit(f"all markets {season}",
                       ols_clustered(g["dev"], g["out"], g["event_id"]))

        # pooled, scaled so markets with different units can be combined
        scale = {}
        for m, g in sub.groupby("market"):
            s = g["out"].std()
            scale[m] = s if s and s > 0 else 1.0
        dz = sub["dev"] / sub["market"].map(scale)
        oz = sub["out"] / sub["market"].map(scale)
        report_fit("POOLED (sigma units)",
                   ols_clustered(dz, oz, sub["event_id"]))

        # out of sample RMSE of three predictors of the actual
        print(f"\n  {'market':<14}{'n':>7}{'RMSE line':>11}{'RMSE proj':>11}"
              f"{'RMSE blend':>12}{'beta used':>11}")
        for m, g in sub.groupby("market"):
            f = ols_clustered(g["dev"], g["out"], g["event_id"])
            if f is None:
                continue
            y = g["out"].to_numpy()
            x = g["dev"].to_numpy()
            b = np.clip(f["beta"], 0, 1)
            print(f"  {m:<14}{len(g):>7}"
                  f"{np.sqrt(np.mean(y ** 2)):>11.2f}"
                  f"{np.sqrt(np.mean((y - x) ** 2)):>11.2f}"
                  f"{np.sqrt(np.mean((y - b * x) ** 2)):>12.2f}"
                  f"{b:>11.3f}")

    # overfitting check
    if args.in_sample:
        print("\n" + "=" * 96)
        print("IN SAMPLE COMPARISON (models trained through 2024, scoring all)")
        print("=" * 96)
        ins = []
        for m in markets:
            s = score_market(m, seasons, walk_forward=False)
            if len(s):
                ins.append(s)
        if ins:
            ip = pd.concat(ins, ignore_index=True)
            ip["_key"] = data_utils.norm_join_name(ip["player"])
            ip["week"] = ip["week"].astype(int)
            j2 = props.merge(
                ip[["season", "week", "market", "_key", "projection", "actual"]],
                on=["season", "week", "market", "_key"], how="inner")
            j2 = j2.dropna(subset=["line_consensus"])
            j2["dev"] = j2["projection"] - j2["line_consensus"]
            j2["out"] = j2["actual"] - j2["line_consensus"]
            header()
            for m, g in j2.groupby("market"):
                report_fit(f"{m} (in sample)",
                           ols_clustered(g["dev"], g["out"], g["event_id"]))
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
    print("  Positive and significant would be the first real evidence of an")
    print("  edge in this project.")
    print("  Compare the THREE LINE SOURCES. If beta against 'best' is")
    print("  materially better than against 'fanduel', the shopping edge is")
    print("  real and worth the operational cost of multiple accounts.")
    print("  alpha is the mean-versus-median offset per market, estimated from")
    print("  real lines rather than guessed. It feeds Phase 5.3.")
    print("  resid_sd is the correct sigma for pricing around a blended mean,")
    print("  where the shipped sigma was fitted around the model's own")
    print("  projection and therefore includes the projection error.")
    print("  RMSE proj above RMSE line means using the projection as the mean")
    print("  is WORSE than ignoring the model, which is what the 596-row")
    print("  sample already suggested.")


if __name__ == "__main__":
    main()
