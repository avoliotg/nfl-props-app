"""
Is the anytime TD movement result real, or regression to the mean?

clv_check found, on 546 TD props:
    move = a + b * (projection - first_implied)
    b = +0.062, SE 0.011, t = +5.48, toward share 0.546 (z +1.95)

That is the only result in this project so far pointing in the model's favour.
But it has a mechanical explanation that must be ruled out first.

THE ARTIFACT. Write the first capture's implied probability as a true value
plus noise:  first_p = true_p + e.  Then

    dev  = proj_p - first_p = proj_p - true_p - e
    move = last_p - first_p = last_p - true_p - e

Both contain -e, so they correlate positively even when proj_p is pure noise
and carries no information whatsoever. The size of the spurious b is roughly
var(e) / var(dev), so a noisy first capture alone can produce b = +0.06.

FOUR TESTS, each removing the shared term a different way:

  1  BASELINE            reproduce the clv_check number, as a control.

  2  PLACEBO             replace the projection with a SHUFFLED projection
                         from another prop. Any b that survives here is pure
                         artifact, since a shuffled projection cannot carry
                         information. This is the single most decisive test.

  3  NO SHARED TERM      measure the move from the SECOND capture onward,
                         while keeping the deviation measured at the first.
                         dev uses first_p, move uses second_p to last_p, so
                         the -e term is no longer common to both.

  4  PURE MEAN REVERSION drop the projection entirely and regress the later
                         move on the earlier move. A negative b is direct
                         evidence that these odds series mean-revert, which
                         is the mechanism that creates the artifact.

Run from the repo root with the venv active:
    python td_reversion_check.py
"""
import os
import sys
from getpass import getpass

import numpy as np
import pandas as pd

SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
PAGE = 1000
N_SHUFFLES = 200
SEED = 11


# ----------------------------------------------------------------- connection

def _read_secrets():
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


def fetch_all(client, table, order_col="id"):
    rows = []
    start = 0
    while True:
        resp = (client.table(table).select("*")
                .order(order_col, desc=False)
                .range(start, start + PAGE - 1).execute())
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        start += PAGE
        if start > 200000:
            break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- stats

def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 10 or np.allclose(x, x[0]):
        return None
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - 2
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * (resid ** 2)[:, None]).T @ X
    cov = xtx_inv @ meat @ xtx_inv * (n / dof)
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"n": n, "alpha": coef[0], "beta": coef[1], "se_beta": se[1],
            "r2": 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan}


def show(label, f, note=""):
    if f is None:
        print(f"  {label:<34} not enough data")
        return
    t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
    lo = f["beta"] - 1.96 * f["se_beta"]
    hi = f["beta"] + 1.96 * f["se_beta"]
    print(f"  {label:<34}{f['n']:>5}{f['beta']:>9.4f}{f['se_beta']:>8.4f}"
          f"{t:>8.2f}   [{lo:>+7.4f},{hi:>+8.4f}]{f['r2']:>9.4f}")
    if note:
        print(f"      {note}")


def american_to_prob(odds):
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return np.nan
    if o != o:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def binom_z(k, n):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    return p, (p - 0.5) / np.sqrt(0.25 / n)


# ---------------------------------------------------------------------- main

def main():
    print("=" * 84)
    print("ANYTIME TD MOVEMENT: REAL SIGNAL, OR REGRESSION TO THE MEAN?")
    print("=" * 84)

    client = connect()
    lines = fetch_all(client, "lines")
    if lines.empty:
        print("  no rows in lines")
        return

    td = lines[lines["market"] == "anytime_td"].copy()
    td["over_odds"] = pd.to_numeric(td["over_odds"], errors="coerce")
    td["projection"] = pd.to_numeric(td["projection"], errors="coerce")
    td["captured_at"] = pd.to_datetime(td["captured_at"], errors="coerce", utc=True)
    td = td.dropna(subset=["over_odds", "captured_at"])
    td["p_imp"] = td["over_odds"].apply(american_to_prob)
    td = td.dropna(subset=["p_imp"]).sort_values("captured_at")

    keys = ["season", "week", "player"]
    recs = []
    for k, g in td.groupby(keys, sort=False):
        g = g.sort_values("captured_at")
        ps = g["p_imp"].tolist()
        projs = g["projection"].tolist()
        if len(ps) < 2:
            continue
        rec = {
            "season": k[0], "week": k[1], "player": k[2],
            "snaps": len(ps),
            "p1": ps[0], "p2": ps[1], "plast": ps[-1],
            "proj1": projs[0],
            "hours": (g["captured_at"].iloc[-1]
                      - g["captured_at"].iloc[0]).total_seconds() / 3600.0,
        }
        recs.append(rec)
    d = pd.DataFrame(recs)
    d = d.dropna(subset=["proj1"])
    # stored projection for this market is a percentage
    d["proj_p"] = d["proj1"] / 100.0
    print(f"  {len(d)} TD props with 2+ snapshots and a projection")
    print(f"  {int((d['snaps'] >= 3).sum())} of them have 3+ snapshots, "
          f"which tests 3 and 4 require")

    if len(d) < 50:
        print("  too few props to test")
        return

    d["dev1"] = d["proj_p"] - d["p1"]
    d["move_full"] = d["plast"] - d["p1"]
    d["move_early"] = d["p2"] - d["p1"]
    d["move_late"] = d["plast"] - d["p2"]

    print("\n" + "=" * 84)
    print("RESULTS")
    print("=" * 84)
    print(f"  {'test':<34}{'n':>5}{'b':>9}{'SE':>8}{'t':>8}"
          f"   {'95% CI':<19}{'R2':>9}")

    # 1 baseline
    f1 = ols(d["dev1"], d["move_full"])
    show("1 BASELINE  dev1 -> full move", f1,
         "reproduces the clv_check result as a control")

    # 2 placebo: shuffled projections
    rng = np.random.default_rng(SEED)
    betas = []
    dev_real = d["dev1"].to_numpy()
    mv = d["move_full"].to_numpy()
    proj = d["proj_p"].to_numpy()
    p1 = d["p1"].to_numpy()
    for _ in range(N_SHUFFLES):
        sp = rng.permutation(proj)
        f = ols(sp - p1, mv)
        if f:
            betas.append(f["beta"])
    if betas:
        betas = np.array(betas)
        print()
        print(f"  2 PLACEBO  shuffled projections, {len(betas)} draws")
        print(f"      mean b {betas.mean():+.4f}   SD {betas.std():.4f}   "
              f"2.5/97.5 pct [{np.percentile(betas, 2.5):+.4f}, "
              f"{np.percentile(betas, 97.5):+.4f}]")
        if f1:
            share = float((betas >= f1["beta"]).mean())
            print(f"      real b {f1['beta']:+.4f} sits above "
                  f"{100 * (1 - share):.1f}% of placebo draws")
            print("      A placebo mean near the real b means the effect is")
            print("      ENTIRELY the shared-noise artifact. A placebo mean")
            print("      near zero means the real b is not explained by it.")

    print()
    d3 = d[d["snaps"] >= 3]
    if len(d3) >= 30:
        # 3 no shared term: dev at capture 1, move measured from capture 2 on
        f3 = ols(d3["dev1"], d3["move_late"])
        show("3 NO SHARED TERM  dev1 -> late move", f3,
             "p1 no longer appears in the dependent variable")

        # 3b same, but deviation measured at capture 2 as well
        f3b = ols(d3["proj_p"] - d3["p2"], d3["move_late"])
        show("3b dev2 -> late move", f3b,
             "shares p2 with the move, so the artifact returns if present")

        # 4 pure mean reversion, projection removed entirely
        f4 = ols(d3["move_early"], d3["move_late"])
        show("4 PURE REVERSION  early -> late move", f4,
             "negative b is direct evidence these odds series mean-revert")
    else:
        print(f"  tests 3 and 4 need 3+ snapshots, only {len(d3)} props qualify")

    # direction counts on the cleanest available test
    print("\n" + "-" * 84)
    print("DIRECTION COUNTS")
    print("-" * 84)
    for label, dev_col, mv_col, frame in (
            ("baseline (dev1, full move)", "dev1", "move_full", d),
            ("no shared term (dev1, late move)", "dev1", "move_late", d3)):
        if len(frame) == 0:
            continue
        nz = frame[(frame[mv_col] != 0) & (frame[dev_col] != 0)]
        if len(nz) == 0:
            print(f"  {label:<36} no non-flat rows")
            continue
        tw = int((np.sign(nz[mv_col]) == np.sign(nz[dev_col])).sum())
        share, z = binom_z(tw, len(nz))
        print(f"  {label:<36}{tw:>5} of {len(nz):<5} = {share:.3f}   z {z:+.2f}")

    print("\n" + "=" * 84)
    print("HOW TO READ THIS")
    print("=" * 84)
    print("  TEST 2 IS THE DECISIVE ONE. A shuffled projection carries no")
    print("  information by construction, so any positive b it produces is the")
    print("  artifact, measured directly. Compare it to the baseline b:")
    print("    placebo mean close to baseline  -> the finding is an artifact")
    print("    placebo mean near zero          -> the finding survives")
    print("    placebo somewhere between       -> partly real, and the honest")
    print("                                       effect is baseline minus")
    print("                                       the placebo mean")
    print("  TEST 3 should agree with test 2. If b stays positive with p1")
    print("  removed from the dependent variable, that is independent support.")
    print("  If 3b is much larger than 3, the artifact is confirmed, because")
    print("  the only difference between them is the shared term.")
    print("  TEST 4 explains the mechanism rather than testing the model. A")
    print("  clearly negative b means these odds mean-revert, which is what")
    print("  generates a spurious positive result in the baseline.")
    print("  If the effect is real, note it still sits alongside a TD model")
    print("  that over-projects by 6.0 pp (z -2.65). The market agreeing with")
    print("  a projection that is too high needs an explanation of its own.")


if __name__ == "__main__":
    main()
