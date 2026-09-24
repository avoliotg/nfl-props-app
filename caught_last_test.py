"""
IS "CAUGHT LAST GAME" INFORMATION THE MARKET MISSES?

WHERE THIS CAME FROM

The cold-streak test found that P(1+ reception) drops from 0.9354 to 0.8818
after a single blank game, about 5.4 points at roughly 2.3 standard errors. It
also found football has almost no cold streaks: 94.2 percent of props are on
players who caught a pass last time out, and only 26 rows in four seasons have
two consecutive blanks. So the Reddit "due for a hit" idea does not translate,
and the signal that IS there runs the other way: cold players are worse, not
due.

Separately, the ladder calibration found exactly one bad rung out of ten:

    k = 1   predicted 0.8917   realized 0.9030   gap +1.1 pp   z +3.62

Those two facts fit together. A single distribution fitted to the pooled data
sits between two populations, 0.9354 and 0.8818, and is wrong for both. So the
k = 1 miss may be a missing feature rather than a distributional problem.

THREE TESTS, IN THE ORDER THAT MATTERS

  TEST A, the one that decides it. Regress (actual - line) on
  (projection - line) AND the indicator. If the indicator's coefficient is
  significantly nonzero AFTER controlling for the line, the market has not
  priced it and it is genuine new information. If it is zero, the market
  already knows and adding it to the model will improve the projection without
  improving the EDGE. That distinction is the whole game and it is the plan's
  item 5.1 applied to one feature.

  TEST B, the model test. Refit receptions walk-forward with the feature added
  and compare beta, RMSE and the shape of the fit. A feature can help the
  projection and do nothing for beta, which is exactly what happened to every
  feature-engineering round earlier in this project.

  TEST C, the calibration test. Does adding it close the k = 1 gap? That is
  the specific defect it was nominated to fix.

WHY THE FEATURE IS COMPUTED FROM THE GAME LOG, NOT THE PROPS

A player may have props in week 3 and week 5 while having played week 4.
Lagging within the props data would skip the intervening game and mislabel the
streak. The indicator is therefore built from the full receptions game log and
joined on, so "last game" means his last actual game.

Also computed and tested: games since last catch as a continuous variable, and
receptions in the previous game as a count, since the binary may be a crude
version of something the rolling features already carry.

Run from the repo root:
    python caught_last_test.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

import eval_harness as eh
import edge_threshold_v5 as et


# ------------------------------------------------------------- the feature

def build_feature():
    """caught_last, games_since_catch and rec_last_game from the game log."""
    from models import receptions
    df = receptions.build_dataset()
    need = ["player_id", "player_display_name", "season", "week", "receptions"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"  receptions.build_dataset lacks {miss}")
    g = df[need].copy()
    g["receptions"] = g["receptions"].fillna(0)
    g = g.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # lag by one GAME within player, across seasons
    g["rec_last_game"] = g.groupby("player_id")["receptions"].shift(1)
    g["caught_last"] = (g["rec_last_game"] > 0).astype(float)
    g.loc[g["rec_last_game"].isna(), "caught_last"] = np.nan

    # running count of consecutive blanks BEFORE this game
    out = np.zeros(len(g))
    run, prev = {}, {}
    for i, (pid, rec) in enumerate(zip(g["player_id"].to_numpy(),
                                       g["receptions"].to_numpy())):
        out[i] = run.get(pid, 0)
        run[pid] = 0 if rec > 0 else run.get(pid, 0) + 1
    g["games_since_catch"] = out

    print(f"\n  feature built on {len(g)} player-weeks from the game log")
    print(f"    caught_last = 1: {100.0 * g['caught_last'].mean(skipna=True):.1f}%")
    print(f"    mean receptions last game: {g['rec_last_game'].mean():.2f}")
    return g[["player_display_name", "season", "week", "caught_last",
              "games_since_catch", "rec_last_game"]]


def load(args):
    sec = eh._secrets()
    box = {}

    def factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    seasons = (list(eh.ALL_SEASONS) if args.all_seasons
               else [int(x) for x in (args.seasons or "2025").split(",")])
    lines = eh.load_lines(factory, seasons, ["receptions"], args.cache,
                          args.refresh)
    if lines.empty:
        sys.exit("  no lines found")
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)

    print("\nscoring receptions walk-forward (baseline feature set)")
    proj = eh.score_market("receptions", seasons, mode="walk_forward",
                           population=args.score_population)
    if not len(proj):
        sys.exit("  nothing scored")

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)
    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")

    feat = build_feature()
    feat["_key"] = data_utils.norm_join_name(feat["player_display_name"])
    feat["week"] = feat["week"].astype(int)
    j = j.merge(feat[["season", "week", "_key", "caught_last",
                      "games_since_catch", "rec_last_game"]],
                on=["season", "week", "_key"], how="left")
    before = len(j)
    j = j.dropna(subset=["caught_last"])
    print(f"\n  joined {len(j)} rows with the feature "
          f"({before - len(j)} dropped, mostly first career games)")
    j["dev"] = j["projection"] - j["line_consensus"]
    j["out"] = j["actual"] - j["line_consensus"]
    return j.dropna(subset=["dev", "out"]).copy()


# ------------------------------------------------------------------- TEST A

def ols_multi(y, X, cluster, names):
    """OLS with cluster-robust SEs, several regressors."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    c = np.asarray(cluster)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, c = y[ok], X[ok], c[ok]
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    xtx = np.linalg.inv(X.T @ X)
    order = np.argsort(c)
    Xs, rs, cs = X[order], resid[order], c[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    gN = len(starts) - 1
    meat = np.zeros((k, k))
    for i in range(gN):
        sl = slice(starts[i], starts[i + 1])
        u = Xs[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (gN / max(gN - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(xtx @ meat @ xtx * scale))
    return {"n": n, "clusters": gN,
            "names": ["const"] + names,
            "coef": coef, "se": se}


def show(fit, title):
    print(f"\n  {title}   n={fit['n']}, games={fit['clusters']}")
    print(f"  {'term':<22}{'coef':>10}{'SE':>9}{'t':>8}{'p (two-sided)':>15}")
    for nm, b, s in zip(fit["names"], fit["coef"], fit["se"]):
        t = b / s if s > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(t))) if np.isfinite(t) else np.nan
        print(f"  {nm:<22}{b:>+10.4f}{s:>9.4f}{t:>+8.2f}{p:>15.4f}")


def test_a(j):
    print("\n" + "=" * 96)
    print("TEST A: does the feature predict the outcome AFTER the line?")
    print("=" * 96)
    print("  A nonzero coefficient here means the MARKET has not priced it.")
    print("  A zero coefficient means the market already knows, and adding it")
    print("  to the model would improve the projection without improving the")
    print("  edge. That is what happened to every earlier feature round.")

    show(ols_multi(j["out"], [j["dev"]], j["event_id"], ["dev"]),
         "baseline, line only")
    show(ols_multi(j["out"], [j["dev"], j["caught_last"]], j["event_id"],
                   ["dev", "caught_last"]),
         "plus caught_last (binary)")
    show(ols_multi(j["out"], [j["dev"], j["games_since_catch"]],
                   j["event_id"], ["dev", "games_since_catch"]),
         "plus games_since_catch (continuous)")
    show(ols_multi(j["out"], [j["dev"], j["rec_last_game"]], j["event_id"],
                   ["dev", "rec_last_game"]),
         "plus rec_last_game (count, tests whether the binary is just a "
         "crude version of volume)")
    print("\n    If caught_last survives but rec_last_game does not, the")
    print("    signal is the BLANK GAME itself rather than recent volume,")
    print("    which the rolling features already carry.")


# ------------------------------------------------------------------- TEST B

def test_b(j, seasons):
    print("\n" + "=" * 96)
    print("TEST B: refit the model with the feature and compare beta")
    print("=" * 96)
    from sklearn.linear_model import LinearRegression
    from models import receptions, data_utils

    df = receptions.build_dataset()
    feats = list(receptions.LEAN_FEATS)
    g = df.copy()
    g["receptions"] = g["receptions"].fillna(0)
    g = g.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g["rec_last_game"] = g.groupby("player_id")["receptions"].shift(1)
    g["caught_last"] = (g["rec_last_game"] > 0).astype(float)
    g.loc[g["rec_last_game"].isna(), "caught_last"] = np.nan

    rows = []
    for s in seasons:
        tr = g[(g["season"] < s) & (g["targets_roll"] >= 3)].dropna(
            subset=feats + ["caught_last", "receptions"])
        te = g[g["season"] == s].dropna(subset=feats + ["caught_last"])
        if len(tr) < 300 or not len(te):
            continue
        m0 = LinearRegression().fit(tr[feats], tr["receptions"])
        m1 = LinearRegression().fit(tr[feats + ["caught_last"]],
                                    tr["receptions"])
        rows.append(pd.DataFrame({
            "season": te["season"].to_numpy(), "week": te["week"].to_numpy(),
            "player": te["player_display_name"].to_numpy(),
            "proj_base": m0.predict(te[feats]),
            "proj_new": m1.predict(te[feats + ["caught_last"]]),
            "actual": te["receptions"].to_numpy()}))
        if s == seasons[-1]:
            print(f"    {s} model coefficients:")
            for nm, c in zip(feats + ["caught_last"], m1.coef_):
                print(f"      {nm:<22}{c:>+9.4f}")
    if not rows:
        print("  could not refit")
        return None
    p = pd.concat(rows, ignore_index=True)
    p["_key"] = data_utils.norm_join_name(p["player"])
    return p


def compare(j, p):
    m = j[["season", "week", "_key", "line_consensus", "event_id",
           "actual"]].merge(
        p[["season", "week", "_key", "proj_base", "proj_new"]],
        on=["season", "week", "_key"], how="inner")
    if len(m) < 300:
        print("  too few matched rows")
        return
    m["out"] = m["actual"] - m["line_consensus"]
    print(f"\n  matched {len(m)} rows for a like-for-like comparison")
    print(f"  {'model':<16}{'beta':>8}{'SE':>7}{'t':>7}{'MAE':>8}{'RMSE':>8}")
    for tag, col in (("baseline", "proj_base"), ("+caught_last", "proj_new")):
        dev = m[col] - m["line_consensus"]
        f = eh.ols_clustered(dev, m["out"], m["event_id"])
        mae = float(np.mean(np.abs(m["actual"] - m[col])))
        rmse = float(np.sqrt(np.mean((m["actual"] - m[col]) ** 2)))
        t = f["beta"] / f["se_beta"] if f and f["se_beta"] > 0 else np.nan
        print(f"  {tag:<16}{f['beta']:>8.3f}{f['se_beta']:>7.3f}{t:>+7.2f}"
              f"{mae:>8.4f}{rmse:>8.4f}")
    print("\n    MAE improving while beta does not is the signature of a")
    print("    feature the market already prices. Both improving is a real")
    print("    gain. This is the distinction that killed the XGBoost work and")
    print("    both qb_passing feature rounds.")


# ------------------------------------------------------------------- TEST C

def test_c(j, params, sig):
    print("\n" + "=" * 96)
    print("TEST C: does it close the k = 1 calibration gap?")
    print("=" * 96)
    key = list(zip(j["market"], j["season"]))
    a = np.array([params.get(k, (np.nan,) * 4)[0] for k in key])
    b = np.array([params.get(k, (np.nan,) * 4)[1] for k in key])
    sa = np.array([sig.get(k, (np.nan, 0.5))[0] for k in key])
    blend = j["line_consensus"].to_numpy() + a + b * j["dev"].to_numpy()
    sigma = sa * np.sqrt(np.clip(blend, 1e-6, None))
    ok = np.isfinite(blend) & (blend > 0) & np.isfinite(sigma)

    print(f"  {'group':<26}{'n':>7}{'predicted':>11}{'realized':>10}"
          f"{'SE':>8}{'gap (pp)':>10}{'z':>7}")
    groups = [("all props", np.ones(len(j), bool)),
              ("caught last game", j["caught_last"].to_numpy() == 1),
              ("blanked last game", j["caught_last"].to_numpy() == 0)]
    for tag, sel in groups:
        s = sel & ok
        if s.sum() < 100:
            print(f"  {tag:<26}{int(s.sum()):>7}   too few")
            continue
        m, v = blend[s], sigma[s] ** 2
        res = np.empty(m.shape)
        nb = v > m * 1.0001
        if nb.any():
            pp = m[nb] / v[nb]
            r = m[nb] * pp / (1 - pp)
            res[nb] = stats.nbinom.sf(0, r, pp)
        if (~nb).any():
            res[~nb] = stats.poisson.sf(0, m[~nb])
        hit = (j["actual"].to_numpy()[s] >= 1).astype(float)
        cm = eh.cluster_mean(hit, j["event_id"].to_numpy()[s])
        pred = float(np.mean(res))
        gap = cm["mean"] - pred
        z = gap / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {tag:<26}{cm['n']:>7}{pred:>11.4f}{cm['mean']:>10.4f}"
              f"{cm['se']:>8.4f}{100 * gap:>+10.1f}{z:>+7.2f}")
    print("\n    If the two subgroups sit on OPPOSITE sides of zero while the")
    print("    pooled row is off by +1.1, the single distribution is averaging")
    print("    two populations and the feature is the fix. If both subgroups")
    print("    are off in the same direction, the problem is distributional")
    print("    and the feature will not help.")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("CAUGHT LAST GAME: information the market misses, or already priced?")
    print("=" * 96)

    j = load(args)
    seasons = sorted(j["season"].unique())

    test_a(j)

    p = test_b(j, seasons)
    if p is not None:
        compare(j, p)

    params = et.fit_params(j.rename(columns={"dev": "dev_cons",
                                             "out": "out_cons"}))
    if params:
        sig = et.fit_sigma_sqrt(j.rename(columns={"dev": "dev_cons",
                                                  "out": "out_cons"}), params)
        test_c(j, params, sig)

    print("\n" + "=" * 96)
    print("HOW TO DECIDE")
    print("=" * 96)
    print("  TEST A is the gate. A significant caught_last coefficient means")
    print("  the market missed it and the feature can move the EDGE. A null")
    print("  there means anything TEST B shows is a better projection with no")
    print("  betting value, which this project has produced several times.")
    print("  TEST C is narrower: whether it fixes the one rung out of ten")
    print("  where the ladder calibration failed. That matters for alt-line")
    print("  work at the 0.5 rung even if the main-line edge does not move.")


if __name__ == "__main__":
    main()
