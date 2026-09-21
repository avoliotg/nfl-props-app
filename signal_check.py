"""
OpalScales - does p_over rank outcomes at all?

WHY THIS COMES BEFORE ANY REFIT
    The reliability table on 240 graded rows was not merely miscalibrated, it
    was non-monotone and slightly inverted:

        predicted 0.331  ->  actual 0.565   (n=69,  z +3.92)
        predicted 0.491  ->  actual 0.451   (n=142, z -0.96)
        predicted 0.664  ->  actual 0.519   (n=27,  z -1.51)

    Higher predicted probability went with a LOWER actual over-rate. That
    matters because recalibration can only fix the mapping from a score to a
    probability. It cannot manufacture ranking power that is not there. So the
    question to answer first is not "what is the right sigma" but "does p_over
    separate overs from unders at all".

WHAT IT MEASURES
  1. AUC of p_over against the realised over/under. 0.50 is a coin flip.
  2. The same for the raw signal underneath it, proj - line, scaled by the
     market. If p_over has no power but the raw gap does, the pricing is
     destroying information rather than lacking it.
  3. A bootstrap CI on both, because 240 rows is thin and an AUC of 0.55
     sounds meaningful while being indistinguishable from noise at this size.
  4. Whether a two-parameter logistic recalibration would help, and by how
     much, scored by log loss and Brier against the current pricing.

HOW TO READ IT
    AUC clearly above 0.50  -> real signal. Recalibrate, and the fitted slope
                               tells you how much to shrink toward 0.5.
    AUC around 0.50         -> the edges are noise. Recalibration is the wrong
                               tool; the problem is upstream of the pricing.
    AUC below 0.50          -> the signal is inverted, which at this sample
                               size is far more likely to be chance than a
                               real anti-signal. Collect more before acting.

HOW TO RUN (Colab)
    !pip install -q supabase pandas scikit-learn
    %run signal_check.py
    Prompts for Supabase URL, anon key, email and password, same as
    calibration_check.py.
"""

import math

import numpy as np
import pandas as pd

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


def sigma_for(market, proj):
    rule = PARAMS[market]["sigma"]
    x = abs(float(proj))
    if rule[0] == "power":
        s = rule[1] * (x ** rule[2]) if x > 0 else SIGMA_MIN
    elif rule[0] == "affine":
        s = rule[1] + rule[2] * x
    else:
        s = rule[1] * x
    return max(s, SIGMA_MIN)


def recompute_p_over(market, proj, line):
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
    sigma = sigma_for(market, proj)
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


def auc(score, y):
    """Rank-based AUC, handling ties."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=float)
    pos, neg = y == 1, y == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties
    s = pd.Series(score)
    ranks = s.rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(score, y, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(auc(score[idx], y[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def main():
    print("=" * 74)
    print("Does p_over rank outcomes at all?")
    print("=" * 74)
    client, uid = connect()
    resp = client.table("bets").select("*").eq("user_id", uid).execute()
    df = pd.DataFrame(resp.data or [])
    if "result" in df.columns:
        df = df.rename(columns={"result": "result_yards"})
    df["outcome"] = df["outcome"].astype(str).str.upper().str.strip()
    d = df[(df["outcome"].isin(["WIN", "LOSS"])) &
           (df["market"] != "anytime_td")].copy()
    for c in ("line", "result_yards", "projection"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["line", "result_yards", "projection", "market"])

    d["p_hat"] = [recompute_p_over(m, p, l) for m, p, l
                  in zip(d["market"], d["projection"], d["line"])]
    d = d[d["p_hat"].notna()].copy()
    d["went_over"] = (d["result_yards"] > d["line"]).astype(float)
    # raw signal: how far the projection sits from the line, in sigmas, so the
    # markets are comparable
    d["z_gap"] = [(p - l) / sigma_for(m, p) for m, p, l
                  in zip(d["market"], d["projection"], d["line"])]

    n = len(d)
    y = d["went_over"].to_numpy()
    print(f"\n  {n} graded yardage/count rows, over-rate {y.mean():.3f}")
    if n < 50:
        raise SystemExit("too few rows")

    print("\n" + "=" * 74)
    print("1. DISCRIMINATION")
    print("=" * 74)
    print("    signal            AUC      95% CI          verdict")
    for label, col in (("p_over (pricing)", "p_hat"),
                       ("proj - line (raw, in sigmas)", "z_gap")):
        sc = d[col].to_numpy(dtype=float)
        a = auc(sc, y)
        lo, hi = boot_auc(sc, y)
        if lo > 0.5:
            v = "real signal"
        elif hi < 0.5:
            v = "INVERTED (suspect chance at this n)"
        else:
            v = "indistinguishable from noise"
        print(f"    {label:28s} {a:.3f}  [{lo:.3f}, {hi:.3f}]  {v}")

    print("\n    If p_over and the raw gap have similar AUC, the pricing is not")
    print("    destroying information. If the raw gap is clearly better, the")
    print("    distribution is the problem rather than the projection.")

    print("\n" + "=" * 74)
    print("2. WOULD RECALIBRATION HELP?")
    print("=" * 74)
    from sklearn.linear_model import LogisticRegression
    lg = np.log(np.clip(d["p_hat"], 1e-6, 1 - 1e-6) /
                (1 - np.clip(d["p_hat"], 1e-6, 1 - 1e-6))).to_numpy().reshape(-1, 1)
    lr = LogisticRegression().fit(lg, y)
    a_, b_ = float(lr.intercept_[0]), float(lr.coef_[0][0])
    p_cal = lr.predict_proba(lg)[:, 1]
    print(f"    fitted recalibration: logit(p_cal) = {a_:+.3f} "
          f"{b_:+.3f} * logit(p_raw)")
    print(f"    slope {b_:.3f}  (1.0 = already calibrated, "
          f"<1 = overconfident, <=0 = no usable signal)")
    print("\n    scoring         log_loss    Brier")
    print(f"    current pricing {logloss(d['p_hat'], y):9.4f} {brier(d['p_hat'], y):8.4f}")
    print(f"    recalibrated    {logloss(p_cal, y):9.4f} {brier(p_cal, y):8.4f}")
    print(f"    always 0.500    {logloss(np.full(n, 0.5), y):9.4f} "
          f"{brier(np.full(n, 0.5), y):8.4f}")
    print(f"    base rate {y.mean():.3f}   {logloss(np.full(n, y.mean()), y):9.4f} "
          f"{brier(np.full(n, y.mean()), y):8.4f}")
    print("\n    IMPORTANT: this recalibration is fitted and scored on the SAME")
    print("    240 rows, so its numbers are optimistic. If it cannot beat a")
    print("    flat 0.500 even in-sample, there is no signal to rescue.")

    print("\n" + "=" * 74)
    print("3. BY MARKET")
    print("=" * 74)
    print("    market        n   over_rate    AUC      95% CI")
    for mkt, g in d.groupby("market"):
        if len(g) < 25:
            print(f"    {mkt:12s} {len(g):4d}  (too few)")
            continue
        yy = g["went_over"].to_numpy()
        sc = g["p_hat"].to_numpy(dtype=float)
        a = auc(sc, yy)
        lo, hi = boot_auc(sc, yy, n_boot=1000)
        print(f"    {mkt:12s} {len(g):4d} {yy.mean():10.3f} {a:7.3f}  "
              f"[{lo:.3f}, {hi:.3f}]")

    print("\n" + "=" * 74)
    print("WHAT TO DO")
    print("=" * 74)
    print("  AUC CI excludes 0.5 on the low side:")
    print("    the pricing has real ranking power. Apply the fitted slope as a")
    print("    shrinkage layer, then re-validate on NEW weeks, never on these.")
    print("\n  AUC CI straddles 0.5:")
    print("    240 rows cannot tell whether there is signal. Do not refit on")
    print("    this data - a fit to noise looks like a fix and is not. Keep")
    print("    grading, treat tiers as ranks only, and re-run at week 5.")
    print("\n  Either way, do NOT bet the displayed edges as probabilities until")
    print("  the AUC CI clears 0.5 on data the recalibration never saw.")


if __name__ == "__main__":
    main()
