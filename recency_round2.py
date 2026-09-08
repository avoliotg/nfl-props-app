"""
OpalScales - recency weighting, round 2. Exhaustive.

ROUND 1 FOUND
    The gate fired: lag1 carries 2.2x the weight of lag6 (z = +9.13 on targets,
    +8.68 on target_share), monotone decline, spearman +1.00. And on a common
    row set, MAE fell monotonically with EWMA halflife:
        window_6 (current) 24.82 | ewm_hl3 24.56 | ewm_hl4 24.48 | ewm_hl6 24.42
    with ewm_hl3/4/6 all excluding zero in a paired bootstrap. An ordered RUN
    of winners is much harder to explain as chance than one isolated winner.

    Interpretation: the gain comes from USING MORE HISTORY with a sensible
    taper, not from sharper recency. ewm_hl1 - pure recency, almost no history -
    was indistinguishable from base.

WHAT ROUND 1 DID NOT DO, AND THIS DOES
  1. HALFLIFE 6 WAS THE EDGE OF THE GRID. Extended to 8/12/16/24 plus
     `expanding` (all history, equal weight) as the limit case. If the optimum
     sits at 24 the story is "use everything"; if it turns at 8 it is a real
     interior optimum.
  2. SELECTION DISCIPLINE. Round 1 selected and reported on the same pooled
     2024+2025 folds. With 10 specs that is optimistic. Here: select on 2024,
     touch 2025 exactly once.
  3. SEASON BOUNDARIES. The EWMA currently groups by player_id, so history
     carries across the offseason - effectively the prior-season bridge.
     Resetting per season has never been compared.
  4. PER-FEATURE HALFLIVES. target_share is a role statistic; ypt is noisy
     efficiency. No reason they want the same taper. One coordinate pass.
  5. ALL FIVE MARKETS, since each model would take the change separately.

HOW TO RUN
    python recency_round2.py
    Several minutes. Set QUICK = True to cut the grid.
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

QUICK = False
SEED = 20260908
N_BOOT = 2000
SELECT_SEASON = [2024]      # spec chosen here
FINAL_SEASON = [2025]       # touched once

HALFLIVES = [1, 2, 3, 4, 6] if QUICK else [1, 2, 3, 4, 6, 8, 12, 16, 24]
WINDOWS = [6] if QUICK else [3, 6, 10, 14, 20]

# roll feature -> source column, per market
MARKETS = {
    "receiving": dict(
        module="receiving", target="receiving_yards",
        rolls={"target_share_roll": "target_share", "targets_roll": "targets",
               "snap_roll": "offense_pct", "ypt_roll": "ypt_game"},
        game=["team_spread", "total_line"], filt=("targets_roll", 3.0)),
    "receptions": dict(
        module="receptions", target="receptions",
        rolls={"target_share_roll": "target_share", "targets_roll": "targets",
               "snap_roll": "offense_pct"},
        game=["team_spread", "total_line"], filt=("targets_roll", 3.0)),
    "rushing": dict(
        module="rushing", target="rushing_yards",
        rolls={"carries_roll": "carries"},
        game=["team_spread", "total_line"], filt=("carries_roll", 5.0)),
    "qb_passing": dict(
        module="qb_passing", target="passing_yards",
        rolls={"attempts_roll": "attempts"},
        game=["team_spread", "total_line", "wind_eff", "def_pass_roll"],
        filt=("attempts_roll", 10.0)),
    "qb_rushing": dict(
        module="qb_rushing", target="rushing_yards",
        rolls={"rush_yds_roll": "rushing_yards", "carries_roll": "carries"},
        game=["team_spread", "total_line"], filt=None),
}


def load(name):
    cfg = MARKETS[name]
    mod = __import__(f"models.{cfg['module']}", fromlist=["*"])
    d = mod.build_dataset()
    rolls = {k: v for k, v in cfg["rolls"].items() if v in d.columns}
    dropped = [k for k in cfg["rolls"] if k not in rolls]
    game = [g for g in cfg["game"] if g in d.columns]
    need = ["player_id", "season", "week", cfg["target"]] + list(rolls.values()) + game
    d = d.dropna(subset=[c for c in need if c in d.columns]).copy()
    d["season"] = d["season"].astype(int)
    d["week"] = d["week"].astype(int)
    d = d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    print(f"  {name}: {len(d):,} rows | rolls {list(rolls)} | game {game}")
    if dropped:
        print(f"    NOTE source column missing, cannot rebuild: {dropped}")
    return d, rolls, game, cfg


def rebuild(d, rolls, spec, per_feature=None, reset_season=False, adjust=True):
    """spec: ('window', n) | ('ewm', halflife) | ('expanding', None).
    per_feature overrides spec for named roll features."""
    x = d.copy()
    keys = ["player_id", "season"] if reset_season else ["player_id"]
    for out, col in rolls.items():
        s = (per_feature or {}).get(out, spec)
        g = x.groupby(keys)[col]
        mode, p = s
        if mode == "window":
            x[out] = g.transform(lambda v: v.shift(1).rolling(p, min_periods=1).mean())
        elif mode == "ewm":
            x[out] = g.transform(
                lambda v: v.shift(1).ewm(halflife=p, min_periods=1, adjust=adjust).mean())
        else:
            x[out] = g.transform(lambda v: v.shift(1).expanding(min_periods=1).mean())
    return x


def cv(x, feats, target, filt, seasons, keep=None):
    from sklearn.linear_model import LinearRegression
    use = [f for f in feats if f in x.columns]
    x = x.dropna(subset=use + [target])
    if filt and filt[0] in x.columns:
        x = x[x[filt[0]] >= filt[1]]
    ks = (x[["season", "week"]].drop_duplicates()
          .sort_values(["season", "week"]).to_numpy())
    P, Y, W, K = [], [], [], []
    for s, w in ks:
        s, w = int(s), int(w)
        if s not in seasons:
            continue
        tr = x[(x["season"] < s) | ((x["season"] == s) & (x["week"] < w))]
        te = x[(x["season"] == s) & (x["week"] == w)]
        if keep is not None:
            te = te[[(s, w, p) in keep for p in te["player_id"]]]
        if len(tr) < 200 or len(te) < 3:
            continue
        lm = LinearRegression().fit(tr[use], tr[target])
        P.append(lm.predict(te[use]))
        Y.append(te[target].to_numpy(float))
        W.append(te["week"].to_numpy())
        K.extend((s, w, p) for p in te["player_id"])
    if not P:
        return None
    return dict(pred=np.concatenate(P), y=np.concatenate(Y),
                week=np.concatenate(W), keys=K)


def score(r):
    return dict(n=len(r["y"]),
                mae=float(np.mean(np.abs(r["pred"] - r["y"]))),
                corr=float(np.corrcoef(r["pred"], r["y"])[0, 1]))


def aligned_runs(d, rolls, feats, cfg, specs, seasons, **kw):
    """Two passes: find the common row set, then rescore all specs on it."""
    built, keysets = {}, {}
    for label, spec in specs.items():
        x = rebuild(d, rolls, spec, **kw)
        r = cv(x, feats, cfg["target"], cfg["filt"], seasons)
        if r is None:
            continue
        built[label] = x
        keysets[label] = set(r["keys"])
    if not keysets:
        return {}, set()
    common = set.intersection(*keysets.values())
    runs = {}
    for label, x in built.items():
        r = cv(x, feats, cfg["target"], cfg["filt"], seasons, keep=common)
        if r:
            runs[label] = r
    return runs, common


def boot_vs(runs, base_key, n_boot=N_BOOT, seed=SEED):
    base = runs.get(base_key)
    if base is None:
        return {}
    n = len(base["y"])
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = {}
    for k, r in runs.items():
        if k == base_key or len(r["y"]) != n:
            continue
        dm = np.array([np.mean(np.abs(r["pred"][ii] - base["y"][ii]))
                       - np.mean(np.abs(base["pred"][ii] - base["y"][ii]))
                       for ii in idx])
        lo, hi = np.percentile(dm, [2.5, 97.5])
        out[k] = (float(dm.mean()), float(lo), float(hi))
    return out


# ----------------------------------------------------------------------------


def stage_a(d, rolls, feats, cfg):
    print("\n" + "=" * 74)
    print("STAGE A - extended grid, SELECTED ON 2024 ONLY")
    print("=" * 74)
    specs = {f"window_{w}": ("window", w) for w in WINDOWS}
    specs.update({f"ewm_hl{h}": ("ewm", h) for h in HALFLIVES})
    specs["expanding"] = ("expanding", None)

    runs, common = aligned_runs(d, rolls, feats, cfg, specs, SELECT_SEASON)
    if not runs:
        print("  no runs")
        return None
    t = pd.DataFrame([dict(spec=k, **score(v)) for k, v in runs.items()]).set_index("spec")
    print(f"  common rows: {len(common):,}")
    print("\n" + t.sort_values("mae").round(4).to_string())

    b = boot_vs(runs, "window_6")
    print("\n  paired bootstrap vs window_6 (current):")
    print("    spec           d_MAE      95% CI            verdict")
    for k in t.sort_values("mae").index:
        if k not in b:
            continue
        m, lo, hi = b[k]
        v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
        print(f"    {k:14s} {m:+8.3f} [{lo:+.3f},{hi:+.3f}]  {v}")

    best = t["mae"].idxmin()
    print(f"\n  >>> best on 2024: {best}")
    hl = [k for k in t.sort_values('mae').index if k.startswith('ewm')]
    if hl:
        print(f"      best EWMA: {hl[0]}")
        if hl[0].endswith(str(max(HALFLIVES))):
            print("      WARNING still at the grid edge; the optimum may be further out")
        else:
            print("      interior optimum, so the grid is wide enough")
    return specs.get(best, ("window", 6)), best


def stage_b(d, rolls, feats, cfg, best_spec):
    print("\n" + "=" * 74)
    print("STAGE B - season reset, and ewm adjust flag")
    print("=" * 74)
    variants = {
        "carry_adjustT": dict(reset_season=False, adjust=True),
        "carry_adjustF": dict(reset_season=False, adjust=False),
        "reset_adjustT": dict(reset_season=True, adjust=True),
        "reset_adjustF": dict(reset_season=True, adjust=False),
    }
    built, keysets = {}, {}
    for lbl, kw in variants.items():
        x = rebuild(d, rolls, best_spec, **kw)
        r = cv(x, feats, cfg["target"], cfg["filt"], SELECT_SEASON)
        if r:
            built[lbl] = x
            keysets[lbl] = set(r["keys"])
    if not keysets:
        print("  none ran")
        return dict(reset_season=False, adjust=True)
    common = set.intersection(*keysets.values())
    runs = {}
    for lbl, x in built.items():
        r = cv(x, feats, cfg["target"], cfg["filt"], SELECT_SEASON, keep=common)
        if r:
            runs[lbl] = r
    t = pd.DataFrame([dict(spec=k, **score(v)) for k, v in runs.items()]).set_index("spec")
    print(f"  common rows: {len(common):,} (note: reset drops week-1 rows)")
    print("\n" + t.sort_values("mae").round(4).to_string())
    b = boot_vs(runs, "carry_adjustT")
    for k, (m, lo, hi) in b.items():
        v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
        print(f"    {k:14s} vs carry_adjustT: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  {v}")
    print("\n  Carry-over IS the prior-season bridge. If reset wins, the bridge")
    print("  is hurting rather than helping, which would be a real finding.")
    best = t["mae"].idxmin()
    return variants[best]


def stage_c(d, rolls, feats, cfg, best_spec, kw):
    print("\n" + "=" * 74)
    print("STAGE C - per-feature halflife (one coordinate pass)")
    print("=" * 74)
    if len(rolls) < 2:
        print("  only one roll feature, nothing to vary")
        return None
    cand = [("ewm", h) for h in (2, 3, 6, 12, 24)] + [("expanding", None)]
    per = {}
    for feat in rolls:
        rows, built, keysets = [], {}, {}
        for c in cand:
            trial = dict(per)
            trial[feat] = c
            x = rebuild(d, rolls, best_spec, per_feature=trial, **kw)
            r = cv(x, feats, cfg["target"], cfg["filt"], SELECT_SEASON)
            if r:
                lbl = f"{c[0]}_{c[1]}"
                built[lbl] = x
                keysets[lbl] = set(r["keys"])
        if not keysets:
            continue
        common = set.intersection(*keysets.values())
        for lbl, x in built.items():
            r = cv(x, feats, cfg["target"], cfg["filt"], SELECT_SEASON, keep=common)
            if r:
                rows.append(dict(setting=lbl, **score(r)))
        t = pd.DataFrame(rows).set_index("setting").sort_values("mae")
        print(f"\n  {feat}:")
        print("    " + t.round(4).to_string().replace("\n", "\n    "))
        winner = t.index[0]
        mode, p = winner.split("_", 1)
        per[feat] = ("expanding", None) if mode == "expanding" else ("ewm", float(p))
        print(f"    -> {feat}: {winner}")
    print(f"\n  per-feature selection: "
          + ", ".join(f"{k}={v[0]}{'' if v[1] is None else v[1]}" for k, v in per.items()))
    return per


def stage_d(d, rolls, feats, cfg, best_spec, kw, per):
    print("\n" + "=" * 74)
    print("STAGE D - FINAL: score on 2025 once")
    print("=" * 74)
    specs = {
        "current_window6": (("window", 6), None),
        "global_best": (best_spec, None),
    }
    if per:
        specs["per_feature"] = (best_spec, per)

    built, keysets = {}, {}
    for lbl, (spec, pf) in specs.items():
        base_kw = dict(reset_season=False, adjust=True) if lbl == "current_window6" else kw
        x = rebuild(d, rolls, spec, per_feature=pf, **base_kw)
        r = cv(x, feats, cfg["target"], cfg["filt"], FINAL_SEASON)
        if r:
            built[lbl] = (x, base_kw, pf, spec)
            keysets[lbl] = set(r["keys"])
    if not keysets:
        print("  none ran")
        return None
    common = set.intersection(*keysets.values())
    runs = {}
    for lbl, (x, base_kw, pf, spec) in built.items():
        r = cv(x, feats, cfg["target"], cfg["filt"], FINAL_SEASON, keep=common)
        if r:
            runs[lbl] = r
    t = pd.DataFrame([dict(spec=k, **score(v)) for k, v in runs.items()]).set_index("spec")
    print(f"  common rows on 2025: {len(common):,}")
    print("\n" + t.round(4).to_string())
    b = boot_vs(runs, "current_window6")
    print("\n  paired bootstrap vs current:")
    for k, (m, lo, hi) in b.items():
        v = "BETTER" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
        print(f"    {k:16s} {m:+8.3f} [{lo:+.3f},{hi:+.3f}]  {v}")
    return t


def main():
    print("=" * 74)
    print("RECENCY WEIGHTING - ROUND 2 (exhaustive)")
    print(f"select on {SELECT_SEASON} | final on {FINAL_SEASON} | QUICK={QUICK}")
    print("=" * 74)

    summary = []
    for name in MARKETS:
        print("\n" + "#" * 74)
        print(f"# MARKET: {name}")
        print("#" * 74)
        try:
            d, rolls, game, cfg = load(name)
            if not rolls:
                print("  no rebuildable roll features, skipped")
                continue
            feats = list(rolls) + game
            got = stage_a(d, rolls, feats, cfg)
            if got is None:
                continue
            best_spec, best_label = got
            kw = stage_b(d, rolls, feats, cfg, best_spec)
            per = stage_c(d, rolls, feats, cfg, best_spec, kw)
            t = stage_d(d, rolls, feats, cfg, best_spec, kw, per)
            if t is not None:
                base = float(t.loc["current_window6", "mae"])
                bestrow = t["mae"].idxmin()
                summary.append(dict(market=name, chosen_on_2024=best_label,
                                    final_best=bestrow,
                                    mae_current=base,
                                    mae_best=float(t.loc[bestrow, "mae"]),
                                    delta=float(t.loc[bestrow, "mae"]) - base))
        except Exception as exc:
            import traceback
            print(f"  FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if summary:
        print("\n" + "=" * 74)
        print("SUMMARY - 2025 holdout")
        print("=" * 74)
        print(pd.DataFrame(summary).set_index("market").round(4).to_string())
        print("\n  Ship per market only where the 2025 bootstrap CI excluded zero.")
        print("  A market whose point estimate improves but whose CI straddles")
        print("  zero should keep window_6.")
    print("\n  AFTER ANY CHANGE: re-run atm_check.py. Rebuilding the features")
    print("  moves every projection, and the pricing coefficients were fitted")
    print("  against the old construction.")


if __name__ == "__main__":
    main()
