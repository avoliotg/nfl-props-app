"""
OpalScales - qb_rushing quintile 4: real defect or noise?

THE OBSERVATION
    atm_check on 2025 showed, by projection quintile:
        q1 med 4.3   empirical 0.331  model 0.280   -5.1 pp
        q2 med 8.1   empirical 0.289  model 0.362   +7.3
        q3 med 13.3  empirical 0.348  model 0.392   +4.4
        q4 med 20.7  empirical 0.548  model 0.391  -15.8   <-- the spike
        q5 med 28.9  empirical 0.430  model 0.389   -4.1

WHY SUSPECT NOISE FIRST
    The empirical series is NON-MONOTONIC. A genuine misspecification should be
    smooth; a single spike between two ordinary bins looks like sampling. And
    n=135 gives an SE near 4.3 pp, so 15.8 pp is about 3.7 SE - suggestive, but
    atm_check made 25 bin comparisons across five markets, and one at 3.7 SE is
    roughly what chance delivers.

    The decisive test is the same breakdown on 2022-2024 (1,937 rows), which
    atm_check never printed. Structural problems reproduce; noise does not.

THE HYPOTHESIS IF IT IS REAL
    qb_rushing has only two features. Around a 20-yard projection the pool
    mixes two player types:
      designed runners (Jackson, Hurts, Daniels) - consistent planned carries,
        so LESS skew and a higher chance of clearing a central line
      pocket QBs - yards come from rare scrambles, so MORE skew
    One gamma shape cannot serve both. carries_roll separates them, and the
    sigma rule currently ignores it. Stage 4 tests exactly this split.

HOW TO RUN
    python qb_rushing_defect.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import mc_pricing  # noqa: E402

MARKET = "qb_rushing"
TARGET = "rushing_yards"
FIT = [2022, 2023, 2024]
TEST = [2025]


def load():
    from sklearn.linear_model import LinearRegression
    mod = __import__("models.qb_rushing", fromlist=["*"])
    feats = list(getattr(mod, "LEAN_FEATS"))
    df = mod.build_dataset()
    d = df.dropna(subset=feats + [TARGET, "season", "week"]).copy()
    d["season"] = d["season"].astype(int)
    tr = d[d["season"].isin(FIT)]
    lm = LinearRegression().fit(tr[feats], tr[TARGET])
    d = d.copy()
    d["proj"] = lm.predict(d[feats])
    d["y"] = d[TARGET].astype(float)
    d["model_p"] = [mc_pricing.p_over(MARKET, p, p) for p in d["proj"]]
    d = d[d["proj"] >= mc_pricing.PARAMS[MARKET]["floor"]]
    d = d[d["model_p"].notna()]
    print(f"  features {feats}")
    print(f"  rows above floor: {len(d):,} "
          f"(fit {int(d['season'].isin(FIT).sum()):,}, "
          f"test {int(d['season'].isin(TEST).sum()):,})")
    return d, feats


def bins_report(d, label, nbins=5):
    print(f"\n  {label}  (n={len(d):,})")
    print("    bin   n   med_proj  empirical   model    gap_pp    SE      z")
    q = pd.qcut(d["proj"], nbins, labels=False, duplicates="drop")
    rows = []
    for qi in sorted(pd.unique(q)):
        m = q == qi
        sub = d[m]
        e = float(np.mean(sub["y"] > sub["proj"]))
        mo = float(sub["model_p"].mean())
        n = len(sub)
        se = float(np.sqrt(max(e * (1 - e), 1e-9) / n)) * 100
        gap = (mo - e) * 100
        z = gap / max(se, 1e-9)
        rows.append((int(qi) + 1, n, float(sub["proj"].median()), e, mo, gap, se, z))
        print(f"    {int(qi) + 1:3d} {n:4d} {sub['proj'].median():9.1f} "
              f"{e:10.4f} {mo:7.4f} {gap:+8.1f} {se:6.1f} {z:+6.2f}")
    return rows


def main():
    print("=" * 74)
    print("qb_rushing quintile-4 investigation")
    print("=" * 74)
    d, feats = load()

    # ---------------- STAGE 1: does it reproduce in-sample? ----------------
    print("\n" + "=" * 74)
    print("STAGE 1 - does the spike reproduce on 2022-2024?")
    print("=" * 74)
    tr = d[d["season"].isin(FIT)]
    te = d[d["season"].isin(TEST)]
    r_tr = bins_report(tr, "2022-2024 (in-sample, 3x the data)")
    r_te = bins_report(te, "2025 (out-of-sample)")

    print("\n  side by side, gap_pp by bin:")
    print("    bin   in-sample   2025")
    for a, b in zip(r_tr, r_te):
        print(f"    {a[0]:3d} {a[5]:+11.1f} {b[5]:+7.1f}")
    q4_tr = r_tr[3][5] if len(r_tr) > 3 else None
    if q4_tr is not None:
        if abs(q4_tr) < 5.0:
            print(f"\n  => in-sample q4 gap is only {q4_tr:+.1f} pp. The 2025 spike")
            print("     does NOT reproduce with 3x the data. This is NOISE.")
        else:
            print(f"\n  => in-sample q4 gap is {q4_tr:+.1f} pp too. STRUCTURAL.")

    # ---------------- STAGE 2: finer bins, all seasons ---------------------
    print("\n" + "=" * 74)
    print("STAGE 2 - all seasons pooled, finer bins (more power)")
    print("=" * 74)
    bins_report(d, "2022-2025 pooled", nbins=8)
    print("\n  A structural problem shows as a run of same-signed gaps in")
    print("  adjacent bins, not one isolated spike.")

    # ---------------- STAGE 3: is it concentrated somewhere? ---------------
    print("\n" + "=" * 74)
    print("STAGE 3 - is the 2025 q4 spike concentrated in a few players?")
    print("=" * 74)
    q = pd.qcut(te["proj"], 5, labels=False, duplicates="drop")
    q4 = te[q == 3]
    if len(q4):
        q4 = q4.copy()
        q4["beat"] = (q4["y"] > q4["proj"]).astype(int)
        g = (q4.groupby("player_display_name")
             .agg(n=("beat", "size"), beat=("beat", "sum"),
                  proj=("proj", "mean"), y=("y", "mean"))
             .sort_values("n", ascending=False).head(12))
        g["rate"] = g["beat"] / g["n"]
        print(f"\n  q4 rows: {len(q4)}, overall beat rate {q4['beat'].mean():.3f}")
        print(g.round(3).to_string())
        print("\n  If a couple of players account for most of it, the spike is")
        print("  a player-mix accident in one season, not a pricing flaw.")

    # ---------------- STAGE 4: the designed-runner hypothesis --------------
    print("\n" + "=" * 74)
    print("STAGE 4 - designed runners vs scramblers at the same projection")
    print("=" * 74)
    if "carries_roll" not in d.columns:
        print("  carries_roll not available, skipped")
    else:
        mid = d[(d["proj"] >= 12) & (d["proj"] <= 30)].copy()
        print(f"  rows with proj 12-30: {len(mid):,}")
        med_c = float(mid["carries_roll"].median())
        print(f"  carries_roll median in that band: {med_c:.2f}")
        for lbl, sub in (("HIGH carries (designed runner)",
                          mid[mid["carries_roll"] > med_c]),
                         ("LOW carries (scrambler)",
                          mid[mid["carries_roll"] <= med_c])):
            if len(sub) < 30:
                continue
            e = float(np.mean(sub["y"] > sub["proj"]))
            mo = float(sub["model_p"].mean())
            se = float(np.sqrt(max(e * (1 - e), 1e-9) / len(sub))) * 100
            # empirical skew and CV of residuals
            resid = sub["y"] - sub["proj"]
            zero = float(np.mean(sub["y"] == 0))
            print(f"\n    {lbl}  n={len(sub):,}")
            print(f"      med_proj {sub['proj'].median():.1f} | "
                  f"empirical P(beat) {e:.4f} | model {mo:.4f} | "
                  f"gap {(mo - e) * 100:+.1f} +- {se:.1f}")
            print(f"      outcome skew {float(sub['y'].skew()):.2f} | "
                  f"resid sd {float(resid.std()):.1f} | zero rate {zero:.3f}")
        print("\n  If HIGH-carries QBs beat their projection much more often than")
        print("  the model expects while LOW-carries QBs match, then one gamma")
        print("  shape is serving two populations and carries_roll belongs in the")
        print("  sigma or gate rule. If both groups look the same, the two-type")
        print("  story is wrong and there is nothing to fix.")

    print("\n" + "=" * 74)
    print("DECISION")
    print("=" * 74)
    print("  Fix ONLY if the in-sample q4 gap reproduces AND stage 4 separates")
    print("  the two groups. One 2025 bin at 3.7 SE, out of 25 comparisons made")
    print("  across five markets, is what chance produces. Adding a feature to")
    print("  the sigma rule to chase it would be fitting noise into live pricing.")


if __name__ == "__main__":
    main()
