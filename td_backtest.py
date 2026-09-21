"""
Gate for the anytime_td survivorship fix. Fit on 2022-2024, score 2025 once.

Three variants, each a strictly larger change than the last:

  A  ORIGINAL          filter touches >= 3 FIRST, then roll; train on the
                       filtered rows. Estimates P(score | 3+ touches).
  B  FEATURE FIXED     roll over every active game, filter afterwards; still
                       train on the filtered rows.
  C  B + POPULATION    roll over every active game AND train on every active
                       game. Estimates P(score) unconditionally, which is what
                       an anytime-TD prop actually pays on.

Evaluation set is deliberately UNCONDITIONAL: every 2025 player-week where the
player was active and had enough history to be projectable. Scoring only on
3+ touch games is the bug itself, one level up, because it conditions on a
variable correlated with the outcome.

Run from the repo root with the venv active:
    python td_backtest.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models import data_utils

SEASONS = [2022, 2023, 2024, 2025, 2026]
POSITIONS = ["WR", "TE", "RB", "QB"]
FEATS = ["touches_roll", "td_rate_roll", "is_rb", "is_te"]
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
BANDS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 1.01]


def _base_frame():
    """Every active player-week at the relevant positions, with outcomes."""
    ps = data_utils.load_player_stats(SEASONS)
    if hasattr(ps, "to_pandas"):
        ps = ps.to_pandas()
    s = ps[ps["position"].isin(POSITIONS)].copy()
    s["total_td"] = s["rushing_tds"].fillna(0) + s["receiving_tds"].fillna(0)
    s["scored"] = (s["total_td"] > 0).astype(int)
    s["touches"] = s["carries"].fillna(0) + s["targets"].fillna(0)
    s["is_rb"] = (s["position"] == "RB").astype(int)
    s["is_te"] = (s["position"] == "TE").astype(int)
    return s.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _add_new_features(s):
    """Roll over every active game, which is the corrected ordering."""
    s = s.copy()
    s["td_rate_roll_new"] = (s.groupby("player_id")["scored"]
                             .transform(lambda x: x.shift(1)
                                        .rolling(10, min_periods=4).mean()))
    s["touches_roll_new"] = (s.groupby("player_id")["touches"]
                             .transform(lambda x: x.shift(1)
                                        .rolling(6, min_periods=3).mean()))
    return s


def _add_old_features(s):
    """Reproduce the original ordering: filter to 3+ touches, then roll.

    The resulting features only exist on surviving rows, so they are carried
    forward within each player to the rows the filter removed. That mirrors
    serve time, where a player carries whatever feature value his surviving
    history produced.
    """
    f = s[s["touches"] >= 3].copy()
    f = f.sort_values(["player_id", "season", "week"])
    f["td_rate_roll_old"] = (f.groupby("player_id")["scored"]
                             .transform(lambda x: x.shift(1)
                                        .rolling(10, min_periods=4).mean()))
    f["touches_roll_old"] = (f.groupby("player_id")["touches"]
                             .transform(lambda x: x.shift(1)
                                        .rolling(6, min_periods=3).mean()))
    keep = ["player_id", "season", "week",
            "td_rate_roll_old", "touches_roll_old"]
    out = s.merge(f[keep], on=["player_id", "season", "week"], how="left")
    out = out.sort_values(["player_id", "season", "week"])
    for c in ("td_rate_roll_old", "touches_roll_old"):
        out[c] = out.groupby("player_id")[c].ffill()
    return out.reset_index(drop=True)


def _fit_predict(df, feat_map, train_mask, eval_mask):
    """Fit a logistic model on train_mask, predict on eval_mask."""
    cols = [feat_map.get(f, f) for f in FEATS]
    train = df[train_mask].dropna(subset=cols + ["scored"])
    if len(train) < 200:
        return None, None, len(train)
    model = LogisticRegression(max_iter=1000).fit(train[cols], train["scored"])
    ev = df[eval_mask].dropna(subset=cols + ["scored"])
    p = model.predict_proba(ev[cols])[:, 1]
    return p, ev["scored"].to_numpy(), len(train)


def _auc(p, y):
    """Rank-based AUC, no sklearn dependency for ties handling."""
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties
    df = pd.DataFrame({"p": p, "r": ranks})
    ranks = df.groupby("p")["r"].transform("mean").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _report(name, p, y, n_train):
    if p is None:
        print(f"\n  {name}: not enough training rows ({n_train})")
        return
    n = len(p)
    pred, act = p.mean(), y.mean()
    se = np.sqrt((p * (1 - p)).sum()) / n
    z = (act - pred) / se if se > 0 else np.nan
    ll = -np.mean(y * np.log(np.clip(p, 1e-9, 1)) +
                  (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
    brier = np.mean((p - y) ** 2)

    print(f"\n  {name}")
    print(f"    trained on {n_train} rows, evaluated on {n}")
    print(f"    mean predicted {pred:.4f}  actual {act:.4f}  "
          f"gap {(act - pred) * 100:+.1f} pp  (SE {se * 100:.1f}, z {z:+.2f})")
    print(f"    log loss {ll:.4f}   Brier {brier:.4f}   AUC {_auc(p, y):.4f}")
    print(f"    {'band':<14}{'n':>6}{'pred':>8}{'actual':>8}{'gap_pp':>9}{'z':>7}")
    bucket = pd.cut(p, BANDS, right=False)
    for b, idx in pd.Series(range(n)).groupby(bucket, observed=True):
        i = idx.to_numpy()
        if len(i) < 5:
            continue
        bp, ba = p[i].mean(), y[i].mean()
        bse = np.sqrt((p[i] * (1 - p[i])).sum()) / len(i)
        bz = (ba - bp) / bse if bse > 0 else np.nan
        print(f"    {str(b):<14}{len(i):>6}{bp:>8.3f}{ba:>8.3f}"
              f"{(ba - bp) * 100:>+9.1f}{bz:>+7.2f}")


def main():
    print("=" * 74)
    print("ANYTIME TD SURVIVORSHIP FIX: fit 2022-2024, score 2025")
    print("=" * 74)

    s = _base_frame()
    s = _add_new_features(s)
    s = _add_old_features(s)

    # Evaluation set: active 2025 player-weeks that would have been projectable.
    # touches_roll_new is the honest workload history, so it gates who appears
    # on a board without conditioning on the current game's outcome.
    eval_mask = ((s["season"] == TEST_SEASON) & (s["touches_roll_new"] >= 3))

    in_train = s["season"].isin(TRAIN_SEASONS)
    train_filtered = in_train & (s["touches"] >= 3)

    print(f"\n  evaluation rows: {int(eval_mask.sum())}")
    print(f"  actual scored rate on eval set: {s.loc[eval_mask, 'scored'].mean():.4f}")
    print(f"  (for contrast, rate among 3+ touch eval rows: "
          f"{s.loc[eval_mask & (s['touches'] >= 3), 'scored'].mean():.4f})")

    old_map = {"td_rate_roll": "td_rate_roll_old",
               "touches_roll": "touches_roll_old"}
    new_map = {"td_rate_roll": "td_rate_roll_new",
               "touches_roll": "touches_roll_new"}

    print("\n" + "-" * 74)
    print("VARIANT RESULTS")
    print("-" * 74)

    p, y, nt = _fit_predict(s, old_map, train_filtered, eval_mask)
    _report("A  ORIGINAL (filter then roll, train filtered)", p, y, nt)

    p, y, nt = _fit_predict(s, new_map, train_filtered, eval_mask)
    _report("B  FEATURE FIXED (roll then filter, train filtered)", p, y, nt)

    p, y, nt = _fit_predict(s, new_map, in_train, eval_mask)
    _report("C  B + POPULATION (train on every active game)", p, y, nt)

    # D is the variant that should have been written first. The illegitimate
    # filter is `touches >= 3`, which conditions on the CURRENT game: a player
    # who ends up with 2 touches is likelier not to have scored, so filtering
    # on it leaks the outcome. Filtering on `touches_roll >= 3` conditions only
    # on LAGGED history, which is exactly what the board does when it decides
    # who to display. Training on that population matches the population served.
    # C trains on every active game including 0-2 touch players, whose TD rate
    # is far below the served base rate, which is why C under-predicts.
    train_roll = in_train & (s["touches_roll_new"] >= 3)
    p, y, nt = _fit_predict(s, new_map, train_roll, eval_mask)
    _report("D  B + SERVING POPULATION (train on touches_roll >= 3)", p, y, nt)

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("  Compare variants on LOG LOSS first. It rewards both calibration and")
    print("  discrimination, where the band table alone can hide a ranking gain.")
    print("  A is the shipped behaviour and the baseline everything else must beat.")
    print("  B changes only the feature ordering. If it beats A on log loss and")
    print("  AUC while holding the overall gap, the honest features rank better")
    print("  even where calibration is unchanged.")
    print("  C and D both change the TRAINING POPULATION. The test that matters")
    print("  is whether the training population matches the SERVED population:")
    print("  the eval set here is players with touches_roll >= 3, so a variant")
    print("  trained on a lower-base-rate population will under-predict, and one")
    print("  trained on a higher-base-rate population will over-predict.")
    print("  Watch the overall gap for overcorrection. A variant that flips from")
    print("  over-stating to under-stating has not fixed anything, it has moved")
    print("  the error, and a band structure that inverts across the range is the")
    print("  clearest sign of a population mismatch rather than a real gain.")


if __name__ == "__main__":
    main()
