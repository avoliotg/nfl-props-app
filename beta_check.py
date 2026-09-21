"""
How much of the model's deviation from the market line is real?

The pricing layer currently assumes the projection IS the mean of the outcome
distribution. That is only valid if the projection is the best available
estimate of the mean. It is not: the line is also an estimate, made with
injury reports, beat reporting and real money at risk. When the two disagree,
the truth usually sits much nearer the line.

That single assumption produces the whole observed failure. The projection
deviates from the line by delta, the pricing treats all of delta as real, so
P(over) moves far from 0.5 while the outcome barely moves. The result is
symmetric overconfidence, which is exactly what the reliability bands show
(+20.1 pp at z +4.84 in the 0.20-0.40 band, -22.4 pp at z -3.85 in 0.60-0.80,
with the middle band clean and the overall gap near zero).

The fix is to estimate the credibility of the model's signal given the line:

    (actual - line) = alpha + beta * (projection - line) + error

  beta ~ 1  the projection carries the information and the line adds nothing
  beta ~ 0  the line is sufficient and the deviations are noise
  beta ~ 0.3  only 30 percent of the apparent edge is real

Then price off a blended mean rather than the raw projection:

    mean_for_pricing = line + beta * (projection - line)

This is far more statistically efficient than AUC, which discards the
magnitude of the outcome. AUC on 533 rows could not separate 0.46 from 0.53.
This regression can separate beta = 0 from beta = 0.5 decisively on the same
rows.

Two free extras fall out of the same fit. Alpha captures the mean-versus-median
offset, since books set lines near the balanced point while the gamma is
right-skewed. And the residual spread is the correct sigma for pricing around
the blended mean, where the current sigma was fitted as spread around the
model's own projection and therefore includes the projection error.

Run from the repo root with the venv active:
    python beta_check.py

Credentials are read from .streamlit/secrets.toml when present. Set
OPAL_EMAIL and OPAL_PASSWORD to skip the prompts entirely.
"""
import os
import sys
from getpass import getpass

import numpy as np
import pandas as pd

YARDAGE = ("receiving", "receptions", "rushing", "qb_passing", "qb_rushing")
SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
PAGE = 1000


# ----------------------------------------------------------------- connection

def _read_secrets():
    """Pull the Supabase URL and anon key out of the Streamlit secrets file."""
    if not os.path.exists(SECRETS_PATH):
        return {}
    try:
        import tomllib
        with open(SECRETS_PATH, "rb") as fh:
            return tomllib.load(fh)
    except Exception as e:
        print(f"  could not parse {SECRETS_PATH}: {e}")
        return {}


def connect():
    from supabase import create_client

    sec = _read_secrets()
    url = sec.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL") or ""
    key = sec.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY") or ""
    if url and key:
        print(f"  credentials from {SECRETS_PATH}")
    else:
        url = url or input("SUPABASE_URL: ").strip()
        key = key or getpass("SUPABASE_KEY (anon, hidden): ").strip()

    email = os.environ.get("OPAL_EMAIL") or sec.get("ADMIN_EMAIL") or ""
    if not email:
        email = input("email: ").strip()
    password = os.environ.get("OPAL_PASSWORD") or ""
    if not password:
        password = getpass("password (hidden): ")

    client = create_client(url, key)
    res = client.auth.sign_in_with_password({"email": email, "password": password})
    if not res.user:
        print("  login failed")
        sys.exit(1)
    print(f"  logged in as {res.user.email}")
    return client


def load_bets(client):
    """All graded rows, paged past the 1000-row PostgREST cap."""
    rows = []
    start = 0
    while True:
        resp = (client.table("bets").select("*")
                .order("id", desc=False)
                .range(start, start + PAGE - 1).execute())
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        start += PAGE
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- stats

def ols(x, y):
    """Fit y = a + b*x. Returns a dict with classical and HC1 robust SEs.

    Robust SEs matter here because residual spread grows with the projection
    level, so the classical SE understates uncertainty at the top end.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - 2
    xtx_inv = np.linalg.inv(X.T @ X)
    s2 = (resid @ resid) / dof
    se_cls = np.sqrt(np.diag(s2 * xtx_inv))
    # HC1 sandwich
    meat = (X * (resid ** 2)[:, None]).T @ X
    cov_hc = xtx_inv @ meat @ xtx_inv * (n / dof)
    se_hc = np.sqrt(np.diag(cov_hc))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {
        "n": n, "alpha": coef[0], "beta": coef[1],
        "se_alpha": se_cls[0], "se_beta": se_cls[1],
        "se_alpha_hc": se_hc[0], "se_beta_hc": se_hc[1],
        "resid_sd": np.sqrt(s2), "r2": r2,
    }


def _fmt_market(name, f):
    lo = f["beta"] - 1.96 * f["se_beta_hc"]
    hi = f["beta"] + 1.96 * f["se_beta_hc"]
    t_zero = f["beta"] / f["se_beta_hc"] if f["se_beta_hc"] > 0 else np.nan
    t_one = (f["beta"] - 1) / f["se_beta_hc"] if f["se_beta_hc"] > 0 else np.nan
    print(f"  {name:<13}{f['n']:>5}{f['beta']:>8.3f}{f['se_beta_hc']:>8.3f}"
          f"   [{lo:>6.3f},{hi:>7.3f}]{t_zero:>8.2f}{t_one:>9.2f}"
          f"{f['alpha']:>9.2f}{f['resid_sd']:>9.1f}")


# ---------------------------------------------------------------------- main

def main():
    print("=" * 78)
    print("HOW MUCH OF THE MODEL'S DEVIATION FROM THE LINE IS REAL?")
    print("=" * 78)

    client = connect()
    df = load_bets(client)
    if df.empty:
        print("  no rows returned from bets")
        return

    print(f"  {len(df)} logged rows")

    d = df[df["market"].isin(YARDAGE)].copy()
    for c in ("projection", "line", "result", "edge", "p_over"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["projection", "line", "result"])
    # a graded row is one with an outcome; result alone can be present on rows
    # that were never resolved
    if "outcome" in d.columns:
        d = d[d["outcome"].isin(["WIN", "LOSS"])]
    print(f"  {len(d)} graded yardage/count rows with projection, line and result")

    if len(d) < 50:
        print("  too few rows to fit")
        return

    d["dev"] = d["projection"] - d["line"]      # what the model claims
    d["out"] = d["result"] - d["line"]          # what actually happened

    print("\n  rows by market:")
    for m, g in d.groupby("market"):
        print(f"    {m:<13}{len(g):>5}   mean |dev| {g['dev'].abs().mean():>6.2f}")

    # ------------------------------------------------------------- section 1
    print("\n" + "=" * 78)
    print("1. BETA BY MARKET   (actual - line) = alpha + beta * (projection - line)")
    print("=" * 78)
    print("  beta is the weight the model's deviation deserves. HC1 robust SEs.")
    print("  t(b=0) tests whether the model adds anything to the line.")
    print("  t(b=1) tests the CURRENT pricing assumption, that the projection")
    print("  should be used as the mean unchanged.\n")
    print(f"  {'market':<13}{'n':>5}{'beta':>8}{'SE':>8}{'   95% CI':<17}"
          f"{'t(b=0)':>8}{'t(b=1)':>9}{'alpha':>9}{'residSD':>9}")

    fits = {}
    for m, g in d.groupby("market"):
        if len(g) < 25:
            print(f"  {m:<13}{len(g):>5}   too few rows to fit alone")
            continue
        f = ols(g["dev"], g["out"])
        fits[m] = f
        _fmt_market(m, f)

    # ------------------------------------------------------------- section 2
    print("\n" + "=" * 78)
    print("2. POOLED FIT, IN UNITS OF EACH MARKET'S OWN SPREAD")
    print("=" * 78)
    print("  Markets are scaled by their residual SD so receiving yards and")
    print("  receptions can be pooled. qb_passing at n=50 cannot support its")
    print("  own beta, so the pooled estimate is the one to trust for it.\n")

    scale = {}
    for m, g in d.groupby("market"):
        sd = fits[m]["resid_sd"] if m in fits else g["out"].std()
        scale[m] = sd if sd and sd > 0 else 1.0
    d["dev_z"] = d["dev"] / d["market"].map(scale)
    d["out_z"] = d["out"] / d["market"].map(scale)

    pooled = ols(d["dev_z"], d["out_z"])
    lo = pooled["beta"] - 1.96 * pooled["se_beta_hc"]
    hi = pooled["beta"] + 1.96 * pooled["se_beta_hc"]
    print(f"  pooled beta  {pooled['beta']:+.3f}   SE {pooled['se_beta_hc']:.3f}"
          f"   95% CI [{lo:+.3f}, {hi:+.3f}]   n={pooled['n']}")
    print(f"  pooled alpha {pooled['alpha']:+.3f} sigma   "
          f"(mean-versus-median offset in the line)")
    print(f"  R-squared    {pooled['r2']:.4f}")
    print(f"  t(beta=0) {pooled['beta'] / pooled['se_beta_hc']:+.2f}    "
          f"t(beta=1) {(pooled['beta'] - 1) / pooled['se_beta_hc']:+.2f}")

    # ------------------------------------------------------------- section 3
    print("\n" + "=" * 78)
    print("3. OUT OF SAMPLE: DOES THE MODEL BEAT THE LINE ALONE?")
    print("=" * 78)
    print("  Fit beta on one week, predict the other. Three predictors of the")
    print("  actual: the line by itself (beta=0), the projection by itself")
    print("  (beta=1, the current pricing), and the blend at the fitted beta.")
    print("  RMSE in each market's own sigma units, so they can be summed.\n")

    weeks = sorted(d["week"].dropna().unique().tolist())
    if len(weeks) < 2:
        print("  only one week of graded data, so no holdout is possible yet")
    else:
        print(f"  {'train':<8}{'test':<8}{'n_test':>7}{'beta_fit':>10}"
              f"{'RMSE line':>11}{'RMSE proj':>11}{'RMSE blend':>12}")
        for test_wk in weeks:
            tr = d[d["week"] != test_wk]
            te = d[d["week"] == test_wk]
            if len(tr) < 50 or len(te) < 20:
                continue
            fit = ols(tr["dev_z"], tr["out_z"])
            b = fit["beta"]
            a = fit["alpha"]
            y = te["out_z"].to_numpy()
            x = te["dev_z"].to_numpy()
            rmse_line = np.sqrt(np.mean((y - 0.0) ** 2))
            rmse_proj = np.sqrt(np.mean((y - x) ** 2))
            rmse_blend = np.sqrt(np.mean((y - (a + b * x)) ** 2))
            others = [w for w in weeks if w != test_wk]
            print(f"  {str(others):<8}{str(int(test_wk)):<8}{len(te):>7}"
                  f"{b:>10.3f}{rmse_line:>11.4f}{rmse_proj:>11.4f}"
                  f"{rmse_blend:>12.4f}")

    # ------------------------------------------------------------- section 4
    print("\n" + "=" * 78)
    print("4. WHAT THIS DOES TO THE DISPLAYED EDGES")
    print("=" * 78)
    b = pooled["beta"]
    b_use = max(0.0, min(1.0, b))
    print(f"  Using the pooled beta clipped to [0, 1]: {b_use:.3f}\n")
    print("  Every deviation from the line shrinks by that factor before")
    print("  pricing, so a projection 20 yards above the line is priced as")
    print(f"  {20 * b_use:.1f} yards above it instead.\n")
    print(f"  {'market':<13}{'mean |dev|':>12}{'shrunk':>10}"
          f"{'mean |edge|':>13}{'implied':>10}")
    for m, g in d.groupby("market"):
        mdev = g["dev"].abs().mean()
        medge = g["edge"].abs().mean() if "edge" in g else np.nan
        print(f"  {m:<13}{mdev:>12.2f}{mdev * b_use:>10.2f}"
              f"{medge:>13.2f}{medge * b_use:>10.2f}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  SECTION 1 AND 2, the estimate itself:")
    print("    beta CI excludes 1 -> the current pricing is provably too")
    print("    aggressive, and the blend is the fix.")
    print("    beta CI excludes 0 -> the model genuinely adds information to")
    print("    the line, which is the first positive evidence of an edge.")
    print("    beta CI contains 0 and excludes 1 -> no demonstrated signal, but")
    print("    a firm upper bound on how large any edge could be.")
    print("    A NEGATIVE beta means the deviations point the wrong way, which")
    print("    matches the fitted recalibration slopes of -0.21 and -0.107.")
    print("  SECTION 3 is the honest test, because it is out of sample:")
    print("    RMSE proj well above RMSE line -> the projection is actively")
    print("    worse than the line by itself, and the edge display is harmful.")
    print("    RMSE blend below both -> the blend is a real improvement and")
    print("    should be adopted.")
    print("    With two weeks this is a weak holdout. Re-run each week.")
    print("  CAVEAT: rows within one game share information, and this fit")
    print("  treats them as independent, so the true SEs are somewhat wider")
    print("  than shown. Clustering needs a game id on the bets table.")


if __name__ == "__main__":
    main()
