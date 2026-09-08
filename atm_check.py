"""
OpalScales - is the new pricing's at-the-money P(over) empirically right?

THE CONCERN
    After shipping, 33 of 74 receiving rows on the board came back Max tier and
    23 of the top 30 were unders. At-the-money rows read P(over) about 0.41:
    Mike Evans 49.5 proj / 49.5 line -> 41.2, Christian Watson 51.6 / 51.5 ->
    41.4. Across dozens of independent FanDuel lines, that is a strong claim
    that the market is systematically wrong in one direction.

THE TESTABLE PART
    The projection is the conditional MEAN by construction (OLS). For a
    right-skewed outcome the MEDIAN sits well below the mean, so
    P(actual > projection) should be BELOW 0.50 in historical data. The
    validation run already hinted at this: receiving median residual was -5.94
    on fit, -5.62 on 2024, -8.05 on 2025.

    So: what fraction of historical games had actual > projection?
      near 0.41  -> the gamma is calibrated, unders really are the better side
                    of an at-the-money line, and the Max count is not an artifact
      near 0.50  -> the pricing is over-skewed and the Max rows are an artifact

THE UNTESTABLE PART (state it plainly)
    Whether the model BEATS THE MARKET cannot be tested here, because there is
    no historical line data - only two imports exist, both for 2026 week 1 with
    no results yet. Internal calibration and market-beating are different
    claims. This script settles the first and leaves the second open until
    results accumulate.

HOW TO RUN
    From the project directory, with the venv active:
        python atm_check.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
import mc_pricing  # noqa: E402

MARKETS = {
    "receiving": dict(module="receiving", target="receiving_yards",
                      filt=("targets_roll", 3.0)),
    "receptions": dict(module="receptions", target="receptions",
                       filt=("targets_roll", 3.0)),
    "rushing": dict(module="rushing", target="rushing_yards",
                    filt=("carries_roll", 5.0)),
    "qb_passing": dict(module="qb_passing", target="passing_yards",
                       filt=("attempts_roll", 10.0)),
    "qb_rushing": dict(module="qb_rushing", target="rushing_yards", filt=None),
}

FIT_SEASONS = [2022, 2023, 2024]
TEST_SEASONS = [2025]


def load(name):
    cfg = MARKETS[name]
    mod = __import__(f"models.{cfg['module']}", fromlist=["*"])
    feats = list(getattr(mod, "LEAN_FEATS", None) or getattr(mod, "FEATS", None) or [])
    df = mod.build_dataset()
    need = feats + [cfg["target"], "season"]
    d = df.dropna(subset=need).copy()
    d["season"] = d["season"].astype(int)
    if cfg["filt"]:
        col, lo = cfg["filt"]
        d = d[pd.to_numeric(d[col], errors="coerce") >= lo]
    return d, feats, cfg


def fit_predict(d, feats, cfg):
    from sklearn.linear_model import LinearRegression
    tr = d[d["season"].isin(FIT_SEASONS)]
    te = d[d["season"].isin(TEST_SEASONS)]
    lm = LinearRegression().fit(tr[feats], tr[cfg["target"]])
    return (lm.predict(te[feats]), te[cfg["target"]].to_numpy(float),
            lm.predict(tr[feats]), tr[cfg["target"]].to_numpy(float))


def report(name):
    print("\n" + "=" * 74)
    print(f"MARKET: {name}")
    print("=" * 74)
    d, feats, cfg = load(name)
    p_te, y_te, p_tr, y_tr = fit_predict(d, feats, cfg)
    floor = mc_pricing.PARAMS[name]["floor"]

    for lbl, p, y in (("2022-2024 (in-sample)", p_tr, y_tr),
                      ("2025 (out-of-sample)", p_te, y_te)):
        m = p >= floor
        p, y = p[m], y[m]
        if len(p) < 50:
            print(f"  {lbl}: too few rows above floor")
            continue

        emp = float(np.mean(y > p))
        model = float(np.mean([mc_pricing.p_over(name, pv, pv) for pv in p]))
        print(f"\n  {lbl}   n={len(p):,} (above floor {floor})")
        print(f"    EMPIRICAL P(actual > projection): {emp:.4f}")
        print(f"    MODEL    P(over) at line=proj  : {model:.4f}")
        print(f"    gap: {100 * (model - emp):+.2f} pp")
        print(f"    mean resid {np.mean(y - p):+.2f} | median resid "
              f"{np.median(y - p):+.2f}")

        # where does the empirical 50/50 line actually sit?
        ratios = np.linspace(0.70, 1.15, 46)
        emp50 = None
        for r in ratios:
            if np.mean(y > p * r) <= 0.50:
                emp50 = r
                break
        mod50 = None
        for r in ratios:
            vals = [mc_pricing.p_over(name, pv, pv * r) for pv in p]
            vals = [v for v in vals if v is not None]
            if np.mean(vals) <= 0.50:
                mod50 = r
                break
        if emp50 and mod50:
            print(f"    line that is a true 50/50: empirical {emp50:.3f} x proj"
                  f" | model {mod50:.3f} x proj")
            print(f"    => the model's median is {'LOW' if mod50 < emp50 else 'HIGH'}"
                  f" by {abs(mod50 - emp50) * 100:.1f}% of projection")

        # by projection quintile, out-of-sample only
        if "2025" in lbl:
            q = pd.qcut(p, 5, labels=False, duplicates="drop")
            print("\n    quintile  n   med_proj   empirical  model   gap_pp")
            for qi in sorted(pd.unique(q)):
                mm = q == qi
                e = float(np.mean(y[mm] > p[mm]))
                mo = float(np.mean([mc_pricing.p_over(name, pv, pv)
                                    for pv in p[mm]]))
                print(f"    {int(qi) + 1:8d} {int(mm.sum()):4d} "
                      f"{np.median(p[mm]):9.1f} {e:11.4f} {mo:7.4f} "
                      f"{100 * (mo - e):+7.1f}")


def main():
    print("=" * 74)
    print("AT-THE-MONEY CALIBRATION CHECK")
    print("=" * 74)
    print("Does P(over) at line = projection match what actually happened?")
    for name in MARKETS:
        try:
            report(name)
        except Exception as exc:
            import traceback
            print(f"\n{name}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 74)
    print("HOW TO DECIDE")
    print("=" * 74)
    print("  gap within about +/-3 pp on 2025:")
    print("    the pricing is calibrated. An at-the-money line really is a")
    print("    better under than over, because the projection is a mean and the")
    print("    outcome distribution is right-skewed. The Max count is then a")
    print("    claim about the MARKET, not a bug - and that claim stays")
    print("    untested until results come in.")
    print("\n  model much BELOW empirical (gap negative, say -5 pp or worse):")
    print("    the gamma is over-skewed. It is manufacturing under edge and the")
    print("    Max rows are an artifact. Roll back and revisit the family.")
    print("\n  EITHER WAY, note what this does NOT show: nothing here proves the")
    print("  model beats FanDuel. Internal calibration against outcomes and")
    print("  edge against a sharp market are different claims. With 74 lines")
    print("  and no results, only the first is answerable today.")


if __name__ == "__main__":
    main()
