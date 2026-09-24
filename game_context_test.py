"""
Do alpha and beta depend on GAME CONTEXT?

Every variable here is knowable BEFORE kickoff. That restriction is the whole
point: conditioning on an in-game result is the volume-filter contamination
that erased the rushing and qb_passing betas, and the receptions frame is
full of in-game columns (receptions, targets, carries, air yards) that would
do exactly that. This script hard-restricts itself to a whitelist so the
mistake is not available to make.

VARIABLES
  home            derived from nflverse game_id, no fetch needed
  team_spread     already a model feature. Favourite vs underdog.
  total_line      already a model feature. Expected scoring.
  roof            schedule join. dome vs outdoors vs retractable.
  rest            schedule join. days since the team's last game.
  weekday         schedule join. Thu / Sun / Mon.
  div_game        schedule join. divisional opponent.

MULTIPLICITY, stated up front
    This script plus line_source_test.py run roughly twenty cuts. At a
    per-cut threshold of 0.05 that is about one false positive expected by
    chance alone. A Bonferroni threshold is printed beside every raw p so a
    borderline result is not read as a finding. The project has already been
    burned by a tool that flagged 2 of 4 line values as unstable on data that
    was flat by construction.

WHAT A POSITIVE RESULT WOULD EVEN MEAN
    A beta that differs by game context is only actionable if the context is
    known at pricing time, which all of these are. But a beta difference may
    also mean the LINE is better or worse in that context rather than the
    model being better or worse, which is the within-tier correlation problem
    again. Read residSD beside beta before believing anything.

Clustered on event_id throughout.
"""

import argparse

import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except Exception:
    _st = None

ZD = {"receptions": 0.25, "receiving": 3.5, "rushing": 5.0, "qb_passing": 18.0}

# only these may be used as conditioning variables. Anything not on this
# list is either an in-game result or unknown before kickoff.
WHITELIST = {"home", "team_spread", "total_line", "roof", "rest",
             "weekday", "div_game", "spread_tier", "total_tier", "rest_tier"}


def ols_clustered(x, y, cluster):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], np.asarray(cluster)[ok]
    n = len(x)
    if n < 100:
        return None
    X = np.column_stack([np.ones(n), x])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    meat = np.zeros((2, 2))
    for c in np.unique(cl):
        sel = cl == c
        s = X[sel].T @ resid[sel]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return dict(alpha=float(coef[0]), beta=float(coef[1]),
                se_alpha=float(np.sqrt(max(V[0, 0], 0.0))),
                se_beta=float(np.sqrt(max(V[1, 1], 0.0))),
                n=n, g=int(len(np.unique(cl))),
                resid_sd=float(resid.std(ddof=2)))


def cluster_mean_se(v, cluster):
    v = np.asarray(v, float)
    c = np.asarray(cluster)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) < 60:
        return None
    m = v.mean()
    grp = pd.Series(v).groupby(pd.Series(c)).agg(["sum", "count"])
    g = len(grp)
    dev = grp["sum"].to_numpy() - m * grp["count"].to_numpy()
    var = np.sum(dev ** 2) / (len(v) ** 2) * (g / max(g - 1, 1))
    return dict(mean=float(m), se=float(np.sqrt(max(var, 0.0))), n=len(v))


def homogeneity(vals, ses):
    pairs = [(v, s) for v, s in zip(vals, ses)
             if v is not None and s is not None and s > 0]
    if len(pairs) < 2:
        return None
    b = np.array([p[0] for p in pairs])
    se = np.array([p[1] for p in pairs])
    w = 1.0 / se ** 2
    bbar = float(np.sum(w * b) / np.sum(w))
    chi2 = float(np.sum(w * (b - bbar) ** 2))
    df = len(pairs) - 1
    p = float(_st.chi2.sf(chi2, df)) if _st is not None else float("nan")
    # minimum detectable spread: roughly 2 SE of the least precise tier
    mds = 2.0 * float(se.max())
    return dict(chi2=chi2, df=df, p=p, pooled=bbar, mds=mds)


def run_cut(d, market, col, zd, n_tests):
    if col not in WHITELIST:
        print("  %s is NOT whitelisted, refusing to condition on it" % col)
        return None
    if col not in d.columns:
        print("  %-14s column absent" % col)
        return None
    sub0 = d[d[col].notna()]
    if len(sub0) < 200:
        print("  %-14s only %d rows with the column present"
              % (col, len(sub0)))
        return None

    print("\n  CUT: %s" % col)
    print("    %-22s %6s %19s %21s %8s"
          % ("tier", "n", "beta", "alpha (direct)", "residSD"))
    bv, bs, dv, ds = [], [], [], []
    for lab, sub in sub0.groupby(col, observed=True):
        if len(sub) < 100:
            continue
        f = ols_clustered(sub["dev"], sub["out"], sub["event_id"])
        if f is None:
            continue
        zsel = sub["dev"].abs() < zd
        z = cluster_mean_se(sub["out"][zsel], sub["event_id"][zsel])
        zstr = ("%+.4f (%.4f) n=%d" % (z["mean"], z["se"], z["n"])) \
            if z else "n=%d too few" % int(zsel.sum())
        print("    %-22s %6d %+9.4f (%.4f) %21s %8.2f"
              % (str(lab), f["n"], f["beta"], f["se_beta"], zstr,
                 f["resid_sd"]))
        bv.append(f["beta"]); bs.append(f["se_beta"])
        dv.append(z["mean"] if z else None)
        ds.append(z["se"] if z else None)

    thresh = 0.05 / max(n_tests, 1)
    out = {}
    for nm, v, s in (("beta", bv, bs), ("alpha", dv, ds)):
        h = homogeneity(v, s)
        if not h:
            continue
        verdict = "DISAGREE" if h["p"] < thresh else (
            "borderline" if h["p"] < 0.05 else "consistent")
        print("    %-6s chi2=%6.2f df=%d  p=%.4f  (Bonferroni %.4f)  "
              "pooled=%+.4f  min detectable spread ~%.3f  -> %s"
              % (nm, h["chi2"], h["df"], h["p"], thresh, h["pooled"],
                 h["mds"], verdict))
        out[nm] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="all_markets.csv")
    ap.add_argument("--source", default="line_fanduel",
                    choices=["line_fanduel", "line_consensus"])
    ap.add_argument("--market", default="receptions")
    ap.add_argument("--no-schedule", action="store_true",
                    help="skip the nflreadpy schedule join")
    args = ap.parse_args()

    d = pd.read_csv(args.rows)
    d = d[d["market"] == args.market].copy()
    d = d[d[args.source].notna() & d["projection"].notna()
          & d["actual"].notna()].copy()
    d["line"] = d[args.source]
    d["dev"] = d["projection"] - d["line"]
    d["out"] = d["actual"] - d["line"]
    zd = ZD[args.market]

    print("=" * 120)
    print("GAME CONTEXT   market=%s  source=%s  n=%d"
          % (args.market, args.source, len(d)))
    print("=" * 120)

    # join the model frame for spread and total, which are already features
    import importlib
    mod = importlib.import_module("models.%s" % args.market)
    frame = mod.build_dataset()
    keep = [c for c in ["season", "week", "player_display_name", "game_id",
                        "team", "team_spread", "total_line"]
            if c in frame.columns]
    frame = frame[keep].copy()
    norm = lambda s: (s.astype(str).str.lower()
                      .str.replace(r"[^a-z ]", "", regex=True).str.strip())
    frame["_pk"] = norm(frame["player_display_name"])
    d["_pk"] = norm(d["player"])
    d = d.merge(frame.drop(columns=["player_display_name"]),
                on=["season", "week", "_pk"], how="left",
                suffixes=("", "_mf"))
    print("  model-frame join rate %.1f%%"
          % (100 * d["game_id"].notna().mean()))

    # home / away from the nflverse game_id: SEASON_WK_AWAY_HOME
    def is_home(row):
        gid, team = row.get("game_id"), row.get("team")
        if not isinstance(gid, str) or not isinstance(team, str):
            return np.nan
        parts = gid.split("_")
        if len(parts) < 4:
            return np.nan
        return 1.0 if parts[3] == team else (0.0 if parts[2] == team else np.nan)

    d["home"] = d.apply(is_home, axis=1)
    d["home"] = d["home"].map({1.0: "home", 0.0: "away"})
    print("  home/away derived: %d home, %d away, %d unresolved"
          % (int((d["home"] == "home").sum()),
             int((d["home"] == "away").sum()),
             int(d["home"].isna().sum())))

    # tiers on the continuous pre-game features
    if "team_spread" in d.columns and d["team_spread"].notna().any():
        q = d["team_spread"].quantile([0.25, 0.5, 0.75]).to_list()
        d["spread_tier"] = pd.cut(d["team_spread"], [-np.inf] + q + [np.inf],
                                  labels=["big fav", "fav", "dog", "big dog"])
    if "total_line" in d.columns and d["total_line"].notna().any():
        q = d["total_line"].quantile([0.33, 0.67]).to_list()
        d["total_tier"] = pd.cut(d["total_line"], [-np.inf] + q + [np.inf],
                                 labels=["low total", "mid", "high total"])

    # schedule join for roof, rest, weekday, divisional
    if not args.no_schedule:
        try:
            import nflreadpy as nfl
            sch = nfl.load_schedules()
            if hasattr(sch, "to_pandas"):
                sch = sch.to_pandas()
            cols = [c for c in ["game_id", "roof", "weekday", "div_game",
                                "home_rest", "away_rest", "home_team"]
                    if c in sch.columns]
            sch = sch[cols].copy()
            d = d.merge(sch, on="game_id", how="left", suffixes=("", "_s"))
            print("  schedule join rate %.1f%%"
                  % (100 * d["roof"].notna().mean()
                     if "roof" in d.columns else 0.0))
            if "home_rest" in d.columns and "away_rest" in d.columns:
                d["rest"] = np.where(d["home"] == "home",
                                     d["home_rest"], d["away_rest"])
                q = d["rest"].quantile([0.5]).to_list()
                d["rest_tier"] = pd.cut(
                    d["rest"], [-np.inf, 6.5, 7.5, np.inf],
                    labels=["short rest", "normal 7", "long rest"])
        except Exception as e:
            print("  schedule join failed: %s: %s  (rerun with "
                  "--no-schedule to skip)" % (type(e).__name__, e))

    candidates = [c for c in ["home", "spread_tier", "total_tier", "roof",
                              "weekday", "div_game", "rest_tier"]
                  if c in d.columns]
    print("\n  cuts to run: %s" % ", ".join(candidates))
    n_tests = len(candidates) * 2

    for col in candidates:
        run_cut(d, args.market, col, zd, n_tests)

    print("\n" + "=" * 120)
    print("HOW TO READ")
    print("=" * 120)
    print("  Every p is compared against a BONFERRONI threshold, not 0.05,")
    print("  because this script runs many cuts. 'borderline' means it would")
    print("  have passed at 0.05 alone and does not survive the correction.")
    print("  'min detectable spread' is roughly what a difference would have")
    print("  to be for these sample sizes to see it. A 'consistent' verdict")
    print("  with a large minimum detectable spread means UNDERPOWERED, not")
    print("  identical. Say so in any writeup.")
    print("  residSD beside beta matters: if beta differs between tiers and")
    print("  residSD differs too, part of the beta gap is scale rather than")
    print("  skill, which is the within-tier correlation problem again.")
    print("  Only whitelisted pre-game variables are allowed as cuts. The")
    print("  frame is full of in-game columns and conditioning on one would")
    print("  reproduce the bug that erased two betas.")


if __name__ == "__main__":
    main()
