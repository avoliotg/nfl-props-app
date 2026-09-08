"""
OpalScales - should recent games get more weight in the rolling features?

WHAT IS ACTUALLY UNTESTED
    Two different things get called "recency weighting":
      1. Weighting recent TRAINING ROWS. Already tested in qb_passing round 2
         (`recency_wt`): improved bias, did nothing for MAE.
      2. Weighting recent games INSIDE the rolling window. Untested.
    The features use rolling(6, min_periods=1).mean(), so last week and six
    weeks ago count equally. The window length of 6 has never been tested
    either.

THE GATE (a measurement, not a search)
    Rather than sweeping halflives and keeping whatever wins - which will always
    produce a winner whether or not the effect is real - STAGE 1 fits the
    outcome on the six lagged games SEPARATELY with standard errors.

      lag1 coefficient much larger than lag6  -> recency matters, weighting
                                                 should help
      lags statistically indistinguishable    -> equal weighting is already
                                                 right, and any halflife that
                                                 appears to win is fitting noise

    This matters because a sweep over 6 halflives x 5 windows is 30
    comparisons, and at those odds something always looks good.

STAGE 2 (only meaningful if the gate fires)
    Rebuild the rolling features under: equal-weight windows of 3/4/6/8/10, and
    EWMA halflives of 1/2/3/4/6 games. Evaluate by rolling-origin CV, reported
    overall and by week bucket, then paired-bootstrap the best against base.

HOW TO RUN
    python recency_weight_test.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

TARGET = "receiving_yards"
FILTER = ("targets_roll", 3.0)
GAME_FEATS = ["team_spread", "total_line"]
CV_SEASONS = [2024, 2025]
N_BOOT = 1500
SEED = 20260908

WINDOWS = [3, 4, 6, 8, 10]
HALFLIVES = [1, 2, 3, 4, 6]


def load():
    from models import receiving as r
    d = r.build_dataset()
    keep = ["player_id", "player_display_name", "season", "week", TARGET,
            "targets", "target_share", "ypt_game", "targets_roll"] + GAME_FEATS
    have = [c for c in keep if c in d.columns]
    missing = [c for c in keep if c not in d.columns]
    snap_src = "offense_pct" if "offense_pct" in d.columns else None
    if snap_src:
        have.append(snap_src)
    d = d[have].copy()
    d = d.dropna(subset=[TARGET, "season", "week", "player_id"] + GAME_FEATS)
    d["season"] = d["season"].astype(int)
    d["week"] = d["week"].astype(int)
    d = d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    print(f"  {len(d):,} rows | snap source: {snap_src or 'MISSING'}")
    if missing:
        print(f"  note: columns not present: {missing}")
    return d, snap_src


# ----------------------------------------------------------------------------
# STAGE 1 - THE GATE
# ----------------------------------------------------------------------------


def stage1(d):
    print("\n" + "=" * 74)
    print("STAGE 1 - THE GATE: do recent games predict better than older ones?")
    print("=" * 74)

    x = d.copy()
    for L in range(1, 7):
        x[f"tgt_lag{L}"] = x.groupby("player_id")["targets"].shift(L)
        x[f"shr_lag{L}"] = x.groupby("player_id")["target_share"].shift(L)
    lag_t = [f"tgt_lag{L}" for L in range(1, 7)]
    lag_s = [f"shr_lag{L}" for L in range(1, 7)]

    for lbl, lags in (("targets", lag_t), ("target_share", lag_s)):
        sub = x.dropna(subset=lags + GAME_FEATS + [TARGET])
        if len(sub) < 500:
            print(f"\n  {lbl}: only {len(sub)} complete rows, skipped")
            continue
        A = np.column_stack([np.ones(len(sub)),
                             sub[lags].to_numpy(float),
                             sub[GAME_FEATS].to_numpy(float)])
        y = sub[TARGET].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        dof = len(sub) - A.shape[1]
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.diag(cov))

        print(f"\n  {lbl} lags (n={len(sub):,}), one coefficient per game back:")
        print("    lag   coef      SE      t")
        coefs, ses = [], []
        for i, L in enumerate(range(1, 7), start=1):
            b, e = beta[i], se[i]
            coefs.append(b)
            ses.append(e)
            print(f"    {L:3d} {b:+8.3f} {e:7.3f} {b / max(e, 1e-9):+6.2f}")

        c = np.array(coefs)
        e = np.array(ses)
        d16 = c[0] - c[5]
        se16 = float(np.sqrt(e[0] ** 2 + e[5] ** 2))   # ignores covariance, rough
        print(f"    lag1 - lag6: {d16:+.3f} +- {se16:.3f} (rough SE) "
              f"-> z = {d16 / max(se16, 1e-9):+.2f}")

        # is there a monotone decline at all?
        rank_corr = float(pd.Series(c).corr(pd.Series([-1, -2, -3, -4, -5, -6]),
                                            method="spearman"))
        print(f"    spearman(coef, recency) = {rank_corr:+.2f} "
              f"(+1 means strictly more weight on recent games)")

        if d16 / max(se16, 1e-9) > 2.0:
            print("    => RECENCY MATTERS for this feature. Weighting may help.")
        else:
            print("    => lag1 and lag6 are NOT distinguishable. Equal weighting")
            print("       is already appropriate; a winning halflife would be noise.")


# ----------------------------------------------------------------------------
# FEATURE REBUILDS
# ----------------------------------------------------------------------------


def rebuild(d, snap_src, mode, param):
    """Recompute rolling features under a given weighting scheme."""
    x = d.copy()
    src = {"targets_roll": "targets", "target_share_roll": "target_share",
           "ypt_roll": "ypt_game"}
    if snap_src:
        src["snap_roll"] = snap_src

    for out, col in src.items():
        if col not in x.columns:
            continue
        g = x.groupby("player_id")[col]
        if mode == "window":
            x[out] = g.transform(
                lambda s: s.shift(1).rolling(param, min_periods=1).mean())
        else:
            x[out] = g.transform(
                lambda s: s.shift(1).ewm(halflife=param, min_periods=1).mean())
    return x


def rolling_origin(x, feats, keep_keys=None):
    """OOS predictions week by week.

    keep_keys, when given, restricts evaluation to a fixed set of
    (season, week, player_id) rows so different feature specs are scored on
    IDENTICAL rows. Without it, min_periods interacts with each spec
    differently and the row counts diverge (4,762 vs 4,986), which makes both
    the MAE column and any paired bootstrap invalid.
    """
    from sklearn.linear_model import LinearRegression
    use = [f for f in feats if f in x.columns]
    x = x.dropna(subset=use + [TARGET])
    if FILTER[0] in x.columns:
        x = x[x[FILTER[0]] >= FILTER[1]]
    keys = (x[["season", "week"]].drop_duplicates()
            .sort_values(["season", "week"]).to_numpy())
    P, Y, W, K = [], [], [], []
    for s, w in keys:
        s, w = int(s), int(w)
        if s not in CV_SEASONS:
            continue
        tr = x[(x["season"] < s) | ((x["season"] == s) & (x["week"] < w))]
        te = x[(x["season"] == s) & (x["week"] == w)]
        if keep_keys is not None:
            te = te[[(s, w, p) in keep_keys for p in te["player_id"]]]
        if len(tr) < 300 or len(te) < 3:
            continue
        lm = LinearRegression().fit(tr[use], tr[TARGET])
        P.append(lm.predict(te[use]))
        Y.append(te[TARGET].to_numpy(float))
        W.append(te["week"].to_numpy())
        K.extend((s, w, p) for p in te["player_id"])
    if not P:
        return None
    return dict(pred=np.concatenate(P), y=np.concatenate(Y),
                week=np.concatenate(W), keys=K)


def summarize(r):
    b = np.where(r["week"] <= 4, "wk1-4", np.where(r["week"] <= 9, "wk5-9", "wk10+"))
    out = dict(n=len(r["y"]),
               mae=float(np.mean(np.abs(r["pred"] - r["y"]))),
               corr=float(np.corrcoef(r["pred"], r["y"])[0, 1]))
    for k in ("wk1-4", "wk5-9", "wk10+"):
        m = b == k
        out[f"mae_{k}"] = (float(np.mean(np.abs(r["pred"][m] - r["y"][m])))
                           if m.sum() > 20 else np.nan)
    return out


def stage2(d, snap_src):
    print("\n" + "=" * 74)
    print("STAGE 2 - window length and EWMA halflife sweep")
    print("=" * 74)
    feats = ["target_share_roll", "targets_roll", "snap_roll", "ypt_roll"] + GAME_FEATS
    specs = ([("window", w, f"window_{w}") for w in WINDOWS]
             + [("ewm", h, f"ewm_hl{h}") for h in HALFLIVES])

    # PASS 1: build each spec once, collect the rows each can score.
    built, keysets = {}, {}
    for mode, param, name in specs:
        x = rebuild(d, snap_src, mode, param)
        r = rolling_origin(x, feats)
        if r is None:
            continue
        built[name] = x
        keysets[name] = set(r["keys"])
        print(f"  {name:11s} scorable rows: {len(r['keys']):,}")

    if "window_6" not in built:
        print("  base spec missing, cannot compare")
        return

    common = set.intersection(*keysets.values())
    print(f"\n  COMMON row set across all {len(keysets)} specs: {len(common):,}")
    print("  Every spec is now scored on these identical rows, so the MAE")
    print("  column and the bootstrap are like-for-like.")

    # PASS 2: rescore restricted to the common rows.
    runs, rows = {}, []
    for name, x in built.items():
        r = rolling_origin(x, feats, keep_keys=common)
        if r is None:
            continue
        runs[name] = r
        rows.append(dict(spec=name, **summarize(r)))

    t = pd.DataFrame(rows).set_index("spec")
    print("\n" + t.round(4).to_string())
    print("\n  window_6 is the CURRENT setup, so that is the row to beat.")

    base = runs.get("window_6")
    if base is None:
        return
    n = len(base["y"])
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    print(f"\n  paired bootstrap vs window_6 ({N_BOOT:,} resamples, n={n:,}):")
    print("    spec           d_MAE      95% CI            verdict")
    for k, r in runs.items():
        if k == "window_6":
            continue
        if len(r["y"]) != n:
            print(f"    {k:14s} row mismatch ({len(r['y'])} vs {n}), skipped")
            continue
        dm = []
        for ii in idx:
            yy = base["y"][ii]
            dm.append(np.mean(np.abs(r["pred"][ii] - yy))
                      - np.mean(np.abs(base["pred"][ii] - yy)))
        dm = np.array(dm)
        lo, hi = np.percentile(dm, [2.5, 97.5])
        v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
        print(f"    {k:14s} {dm.mean():+8.3f} [{lo:+.3f},{hi:+.3f}]  {v}")
    print("\n    Negative means lower MAE than current. A CI crossing zero means")
    print("    the data cannot tell them apart, whatever the point estimate says.")
    print("    NOTE: comparing 9 alternatives, so expect roughly one CI to")
    print("    exclude zero by chance alone. A whole monotone RUN of them")
    print("    excluding zero is much harder to explain as chance.")


def main():
    print("=" * 74)
    print("RECENCY WEIGHTING IN THE ROLLING WINDOW")
    print("=" * 74)
    d, snap_src = load()
    stage1(d)
    stage2(d, snap_src)
    print("\n" + "=" * 74)
    print("HOW TO DECIDE")
    print("=" * 74)
    print("  Change the feature construction only if BOTH hold:")
    print("    - stage 1 shows lag1 significantly above lag6 (z > 2)")
    print("    - stage 2 shows a bootstrap CI excluding zero on MAE")
    print("  If the gate says the lags are indistinguishable, a winner in")
    print("  stage 2 is a multiple-comparisons artifact. Nine sweeps produce")
    print("  one 'significant' result about as often as not.")


if __name__ == "__main__":
    main()
