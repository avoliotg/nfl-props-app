"""
OpalScales - is the model calibrated against REAL MARKET LINES?

THE QUESTION THIS ANSWERS
    Week 1 came in at 45-48% across the yardage markets and 50/152 (32.9%) on
    anytime TD. Below 50% consistently, across markets that are largely
    independent, points at something directional rather than noise.

    The leading suspect: the gamma pricing says P(over) at the money is about
    0.41 for receiving, because the median sits well below the mean. That made
    the board heavily under-weighted. If the truth is nearer 0.50, every under
    the model liked was a slightly bad bet and you would see exactly this.

    Crucially, atm_check.py validated P(actual > OUR PROJECTION) and got 0.41
    empirical vs 0.39 model - calibrated. It never tested P(actual > MARKET
    LINE), because there was no historical line data. Those are different
    claims, and the second one is what edge depends on. Now there is data.

WHAT IT REPORTS
  1. Reliability: bucket the stored p_over and compare each band's predicted
     rate to the actual over rate. A calibrated model tracks the diagonal.
  2. Expected vs actual wins on the side the model picked. This is the direct
     betting-relevant test: sum of p(picked side) vs how many actually won.
  3. Splits by market and by side, since the skew hypothesis predicts unders
     underperforming while overs hold up.
  4. Anytime TD separately, where projection IS the probability.
    Every figure carries a standard error, because 150 rows sounds like a lot
    and is not.

HOW TO RUN (Colab)
    !pip install -q supabase pandas
    %run calibration_check.py
    It will prompt for your Supabase URL, anon key, email and password. The
    password uses getpass so it is not stored in the notebook.
"""

import sys

import numpy as np
import pandas as pd


def connect():
    from getpass import getpass
    from supabase import create_client

    url = input("SUPABASE_URL: ").strip()
    key = getpass("SUPABASE_KEY (anon): ").strip()
    email = input("email: ").strip()
    pw = getpass("password: ")

    client = create_client(url, key)
    res = client.auth.sign_in_with_password({"email": email, "password": pw})
    if not res.user:
        raise SystemExit("login failed")
    print(f"  logged in as {res.user.email}")
    return client, res.user.id


def load_log(client, user_id):
    resp = client.table("bets").select("*").eq("user_id", user_id).execute()
    df = pd.DataFrame(resp.data or [])
    if len(df) == 0:
        raise SystemExit("bet log is empty")
    if "result" in df.columns:
        df = df.rename(columns={"result": "result_yards"})
    print(f"  {len(df):,} logged rows")
    return df


# ---------------------------------------------------------------------------
# Recomputing p_over
# ---------------------------------------------------------------------------
# The log turned out to hold p_over as null on essentially every yardage row
# (1 usable row out of 240), so sections 1 and 2 had nothing to work with and
# produced nonsense: expected wins of 0.7 across 237 rows, and +45 pp "gaps"
# that were division-by-nothing artifacts rather than findings.
#
# projection and line ARE stored, so p_over is recoverable. These parameters
# mirror mc_pricing.PARAMS exactly, inlined so this runs in Colab without the
# repo. Anything below a market's floor returns None, same as the app.
PARAMS = {
    "receiving":  dict(family="gamma",  sigma=("power",  3.2720, 0.6172),
                       gate=(-0.1440, -0.063440), floor=11.5),
    "rushing":    dict(family="gamma",  sigma=("power",  1.9220, 0.7149),
                       gate=None, floor=34.5),
    "qb_rushing": dict(family="gamma",  sigma=("affine", 4.6240, 0.7246),
                       gate=(0.6030, -0.139160), floor=1.5),
    "qb_passing": dict(family="gamma",  sigma=("prop",   0.3531),
                       gate=None, floor=159.0),
    "receptions": dict(family="nbinom", sigma=("affine", 1.1430, 0.2992),
                       gate=None, floor=1.7),
}
PI_CAP, SIGMA_MIN = 0.35, 0.05


def recompute_p_over(market, proj, line):
    """P(outcome > line), mirroring mc_pricing. None when unpriceable."""
    import math
    from scipy import stats
    cfg = PARAMS.get(market)
    if cfg is None:
        return None
    try:
        proj, line = float(proj), float(line)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(proj) and np.isfinite(line)) or proj < cfg["floor"]:
        return None

    rule = cfg["sigma"]
    x = abs(proj)
    if rule[0] == "power":
        sigma = rule[1] * (x ** rule[2]) if x > 0 else SIGMA_MIN
    elif rule[0] == "affine":
        sigma = rule[1] + rule[2] * x
    else:
        sigma = rule[1] * x
    sigma = max(sigma, SIGMA_MIN)

    pi = 0.0
    if cfg["gate"] is not None:
        g0, g1 = cfg["gate"]
        pi = min(1.0 / (1.0 + math.exp(-(g0 + g1 * proj))), PI_CAP)

    m = proj / (1.0 - pi)
    var_c = (sigma ** 2 + proj ** 2) / (1.0 - pi) - m ** 2
    if not np.isfinite(var_c) or var_c <= 0 or m <= 0:
        return None
    s = math.sqrt(var_c)

    if cfg["family"] == "gamma":
        sf = float(stats.gamma.sf(line, a=(m / s) ** 2, scale=s ** 2 / m))
    else:
        var = max(s ** 2, m * 1.0001)
        r = m ** 2 / (var - m)
        sf = float(1.0 - stats.nbinom.cdf(math.floor(line), n=r, p=r / (r + m)))
    val = (1.0 - pi) * sf
    return float(min(max(val, 0.0), 1.0)) if np.isfinite(val) else None


def se_prop(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))


def reliability(d, pred_col, hit_col, label, nbins=5):
    """Predicted probability vs realised rate, in bands."""
    d = d.dropna(subset=[pred_col, hit_col])
    if len(d) < 20:
        print(f"  {label}: only {len(d)} rows, skipped")
        return
    print(f"\n  {label}  (n={len(d):,})")
    print("    band          n    predicted   actual    gap_pp     SE     z")
    edges = np.linspace(0, 1, nbins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d[pred_col] >= lo) & (d[pred_col] < hi + (1e-9 if hi == 1 else 0))
        if m.sum() < 5:
            continue
        sub = d[m]
        pred = float(sub[pred_col].mean())
        act = float(sub[hit_col].mean())
        n = len(sub)
        se = se_prop(act, n)
        gap = (act - pred) * 100
        z = gap / max(se * 100, 1e-9)
        print(f"    {lo:.2f}-{hi:.2f} {n:6d} {pred:11.3f} {act:8.3f} "
              f"{gap:+8.1f} {se * 100:6.1f} {z:+6.2f}")
    pred = float(d[pred_col].mean())
    act = float(d[hit_col].mean())
    se = se_prop(act, len(d))
    print(f"    OVERALL {len(d):5d} {pred:11.3f} {act:8.3f} "
          f"{(act - pred) * 100:+8.1f} {se * 100:6.1f} "
          f"{(act - pred) / max(se, 1e-9):+6.2f}")


def main():
    print("=" * 74)
    print("OpalScales calibration against real market lines")
    print("=" * 74)
    client, user_id = connect()
    df = load_log(client, user_id)

    df["outcome"] = df["outcome"].astype(str).str.upper().str.strip()
    graded = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    print(f"  {len(graded):,} graded (WIN or LOSS)")
    if len(graded) == 0:
        raise SystemExit("nothing graded yet")

    print("\n  graded rows by market:")
    print("    " + graded.groupby("market").size().to_string().replace("\n", "\n    "))

    # ---------------- yardage / count markets ----------------
    yd = graded[graded["market"] != "anytime_td"].copy()
    for c in ("p_over", "line", "result_yards", "projection"):
        if c in yd.columns:
            yd[c] = pd.to_numeric(yd[c], errors="coerce")

    if len(yd) > 0:
        # Stored p_over first (percent -> 0-1), then recompute anything missing.
        stored = yd["p_over"] / 100.0 if yd["p_over"].max(skipna=True) > 1.5 \
            else yd["p_over"]
        recomputed = [recompute_p_over(m, p, l) for m, p, l
                      in zip(yd["market"], yd["projection"], yd["line"])]
        yd["p_hat"] = stored.where(stored.notna(), pd.Series(recomputed, index=yd.index))
        n_stored = int(stored.notna().sum())
        n_recomp = int(yd["p_hat"].notna().sum()) - n_stored
        print(f"\n  p_over: {n_stored} from the log, {n_recomp} recomputed, "
              f"{int(yd['p_hat'].isna().sum())} unavailable")
        yd["went_over"] = (yd["result_yards"] > yd["line"]).astype(float)
        yd = yd[yd["p_hat"].notna()].copy()

        print("\n" + "=" * 74)
        print("1. RELIABILITY OF P(OVER) AGAINST MARKET LINES")
        print("=" * 74)
        print("  gap_pp = actual minus predicted. Positive means the model")
        print("  UNDER-states the over, i.e. the skew is too strong.")
        reliability(yd, "p_hat", "went_over", "all yardage/count markets")
        for mkt, sub in yd.groupby("market"):
            reliability(sub, "p_hat", "went_over", f"market = {mkt}", nbins=4)

        # ---------------- expected vs actual wins ----------------
        print("\n" + "=" * 74)
        print("2. EXPECTED vs ACTUAL WINS ON THE SIDE THE MODEL PICKED")
        print("=" * 74)
        yd["side"] = yd["side"].astype(str).str.upper().str.strip()
        pick = yd[yd["side"].isin(["OVER", "UNDER"])].copy()
        pick["p_pick"] = np.where(pick["side"] == "OVER",
                                  pick["p_hat"], 1 - pick["p_hat"])
        pick["won"] = (pick["outcome"] == "WIN").astype(float)
        print("    group          n   exp_wins  act_wins   exp_rate  act_rate"
              "    gap_pp      z")
        rows = [("ALL", pick)]
        rows += [(f"side={s}", g) for s, g in pick.groupby("side")]
        rows += [(f"mkt={m}", g) for m, g in pick.groupby("market")]
        for label, g in rows:
            if len(g) < 10:
                continue
            exp = float(g["p_pick"].sum())
            act = float(g["won"].sum())
            er, ar = exp / len(g), act / len(g)
            se = se_prop(ar, len(g))
            print(f"    {label:12s} {len(g):5d} {exp:10.1f} {act:9.0f} "
                  f"{er:10.3f} {ar:9.3f} {(ar - er) * 100:+9.1f} "
                  f"{(ar - er) / max(se, 1e-9):+6.2f}")
        print("\n    A large negative gap on side=UNDER with side=OVER holding up")
        print("    is the signature of the gamma being over-skewed.")

    # ---------------- anytime TD ----------------
    td = graded[graded["market"] == "anytime_td"].copy()
    if len(td) >= 20:
        print("\n" + "=" * 74)
        print("3. ANYTIME TD (projection IS the probability)")
        print("=" * 74)
        td["projection"] = pd.to_numeric(td["projection"], errors="coerce")
        td["p_hat"] = td["projection"] / 100.0
        td["scored"] = (td["outcome"] == "WIN").astype(float)
        reliability(td, "p_hat", "scored", "anytime TD", nbins=5)
        pm = float(td["p_hat"].mean())
        ar = float(td["scored"].mean())
        se = se_prop(ar, len(td))
        print(f"\n    mean predicted {pm:.3f} vs actual {ar:.3f}  "
              f"gap {(ar - pm) * 100:+.1f} pp  (SE {se * 100:.1f}, "
              f"z {(ar - pm) / max(se, 1e-9):+.2f})")
        print("    If mean predicted is near the actual rate, the TD model is")
        print("    calibrated and a low hit rate just reflects that TDs are rare.")
        print("    If predicted is much higher, the TD model over-projects.")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("  The number that matters is the OVERALL gap in section 1 and the")
    print("  side split in section 2.")
    print("\n  gap near zero, |z| under 2:")
    print("    the pricing is calibrated against market lines. Week 1 was")
    print("    variance, and no change is warranted on this evidence.")
    print("\n  gap clearly POSITIVE (actual over-rate above predicted):")
    print("    the gamma is over-skewed. P(over) is too low, which manufactured")
    print("    under edges across the board. The sigma and gate would need")
    print("    refitting against outcomes at market lines, not at projections.")
    print("\n  Either way: one week is one week. Check the n on every line before")
    print("  concluding anything, and re-run this after week 3 and week 5.")


if __name__ == "__main__":
    main()
