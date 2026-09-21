"""
Does the market move TOWARD the model?

beta_check answered one question: the model's disagreement with the line does
not predict the outcome (pooled beta -0.079, t(beta=1) = -8.81, and using the
projection as the mean is worse out of sample than using the line).

This asks a different and strictly easier question. A model can be early on
information without beating the closing line. If the model sees something the
market has not yet priced, the line should later drift toward the model's side
as the market absorbs the same information. That is closing line value, and it
converges in weeks rather than seasons, because line movement is far less
noisy than game outcomes.

The premise needs checking too, so section 4 tests whether movement in this
dataset carries any information about outcomes at all. If lines move at random
here, this whole test has no power and a null result would mean nothing.

Two confounds worth keeping in mind while reading:
  - Props within one game share information, and this treats rows as
    independent, so the true standard errors are wider than shown.
  - The stored projection changes between snapshots as the model refits, so
    the FIRST snapshot's projection is used throughout. That is what the model
    actually said before the movement happened.

Run from the repo root with the venv active:
    python clv_check.py

Credentials are read from .streamlit/secrets.toml when present. Set
OPAL_EMAIL and OPAL_PASSWORD to skip the prompts.
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
    """Fit y = a + b*x with HC1 robust standard errors."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 5 or np.allclose(x, x[0]):
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
    return {"n": n, "alpha": coef[0], "beta": coef[1],
            "se_beta": se[1], "se_alpha": se[0],
            "r2": 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan}


def binom(k, n):
    """Share, z against 0.5, and an exact two-sided p when scipy is present."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    z = (p - 0.5) / np.sqrt(0.25 / n)
    pv = np.nan
    try:
        from scipy.stats import binomtest
        pv = binomtest(k, n, 0.5).pvalue
    except Exception:
        pass
    return p, z, pv


def american_to_prob(odds):
    if odds is None or (isinstance(odds, float) and odds != odds):
        return np.nan
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


# ------------------------------------------------------------------ assembly

def summarise_props(lines, market_set, value_col="line"):
    """One row per (season, week, market, player): first and last snapshot."""
    d = lines[lines["market"].isin(market_set)].copy()
    d["captured_at"] = pd.to_datetime(d["captured_at"], errors="coerce", utc=True)
    d = d.dropna(subset=["captured_at", value_col])
    d = d.sort_values("captured_at")

    keys = ["season", "week", "market", "player"]
    grp = d.groupby(keys, sort=False)
    out = grp.agg(
        snapshots=("captured_at", "size"),
        first_cap=("captured_at", "first"),
        last_cap=("captured_at", "last"),
        first_val=(value_col, "first"),
        last_val=(value_col, "last"),
        first_proj=("projection", "first"),
        first_edge=("edge", "first"),
    ).reset_index()
    out = out[out["snapshots"] >= 2].copy()
    out["move"] = out["last_val"] - out["first_val"]
    out["hours"] = (out["last_cap"] - out["first_cap"]).dt.total_seconds() / 3600.0
    return out


def toward_share(sub):
    """Share of non-flat props whose line moved toward the model's side."""
    nz = sub[(sub["move"] != 0) & (sub["dev"] != 0)]
    if len(nz) == 0:
        return 0, 0, np.nan, np.nan, np.nan
    toward = (np.sign(nz["move"]) == np.sign(nz["dev"])).sum()
    p, z, pv = binom(int(toward), len(nz))
    return int(toward), len(nz), p, z, pv


# ---------------------------------------------------------------------- main

def main():
    print("=" * 78)
    print("DOES THE MARKET MOVE TOWARD THE MODEL?")
    print("=" * 78)

    client = connect()
    lines = fetch_all(client, "lines")
    if lines.empty:
        print("  no rows returned from lines")
        return
    for c in ("line", "over_odds", "under_odds", "projection", "edge"):
        if c in lines.columns:
            lines[c] = pd.to_numeric(lines[c], errors="coerce")
    print(f"  {len(lines)} line snapshots, "
          f"{lines['captured_at'].nunique()} distinct captures")

    props = summarise_props(lines, YARDAGE, "line")
    props = props.dropna(subset=["first_proj"])
    props["dev"] = props["first_proj"] - props["first_val"]
    print(f"  {len(props)} yardage/count props with 2+ snapshots and a projection")

    if len(props) < 40:
        print("  too few multi-snapshot props to test")
        return

    # ------------------------------------------------------------- section 1
    print("\n" + "=" * 78)
    print("1. HOW MUCH DO LINES ACTUALLY MOVE?")
    print("=" * 78)
    print("  A test of movement has no power if nothing moves. Flat props")
    print("  carry no information either way and are excluded from the")
    print("  toward/away counts below.\n")
    print(f"  {'market':<13}{'props':>7}{'flat':>7}{'moved':>7}"
          f"{'mean|move|':>12}{'mean snaps':>12}{'med hours':>11}")
    for m, g in props.groupby("market"):
        flat = int((g["move"] == 0).sum())
        nz = g[g["move"] != 0]
        print(f"  {m:<13}{len(g):>7}{flat:>7}{len(nz):>7}"
              f"{nz['move'].abs().mean() if len(nz) else 0:>12.2f}"
              f"{g['snapshots'].mean():>12.1f}{g['hours'].median():>11.1f}")

    # ------------------------------------------------------------- section 2
    print("\n" + "=" * 78)
    print("2. DOES THE MODEL'S DEVIATION PREDICT THE MOVE?")
    print("=" * 78)
    print("  Regression: (last line - first line) = a + b * (projection - first line)")
    print("  b > 0 means the line drifts toward the model, which is the")
    print("  signature of being early on real information.")
    print("  The binomial counts the same thing without using magnitudes.\n")
    print(f"  {'market':<13}{'n':>6}{'b':>8}{'SE':>7}{'t':>7}"
          f"{'R2':>8}   {'toward':>8}{'of':>6}{'share':>8}{'z':>7}{'p':>8}")

    for m, g in props.groupby("market"):
        f = ols(g["dev"], g["move"])
        tw, n_nz, share, z, pv = toward_share(g)
        if f is None:
            print(f"  {m:<13}{len(g):>6}   not enough variation to fit")
            continue
        t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
        pv_s = f"{pv:.3f}" if pv == pv else "   na"
        print(f"  {m:<13}{f['n']:>6}{f['beta']:>8.3f}{f['se_beta']:>7.3f}"
              f"{t:>7.2f}{f['r2']:>8.4f}   {tw:>8}{n_nz:>6}"
              f"{share:>8.3f}{z:>7.2f}{pv_s:>8}")

    # pooled, scaled by each market's own move spread
    sc = {}
    for m, g in props.groupby("market"):
        s = g["move"].std()
        sc[m] = s if s and s > 0 else 1.0
    props["dev_z"] = props["dev"] / props["market"].map(sc)
    props["move_z"] = props["move"] / props["market"].map(sc)
    pf = ols(props["dev_z"], props["move_z"])
    tw, n_nz, share, z, pv = toward_share(props)
    print("\n  POOLED (scaled by each market's move SD)")
    if pf:
        lo = pf["beta"] - 1.96 * pf["se_beta"]
        hi = pf["beta"] + 1.96 * pf["se_beta"]
        print(f"    b {pf['beta']:+.3f}  SE {pf['se_beta']:.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  t {pf['beta'] / pf['se_beta']:+.2f}  "
              f"R2 {pf['r2']:.4f}  n {pf['n']}")
    pv_s = f"{pv:.4f}" if pv == pv else "na"
    print(f"    toward {tw} of {n_nz} = {share:.3f}  z {z:+.2f}  p {pv_s}")

    # ------------------------------------------------------------- section 3
    print("\n" + "=" * 78)
    print("3. BY SIZE OF THE MODEL'S DEVIATION")
    print("=" * 78)
    print("  If the model is early, the biggest disagreements should move")
    print("  toward it most often. A flat or inverted pattern across buckets")
    print("  says the deviation carries no timing information either.\n")
    print(f"  {'bucket':<12}{'n':>6}{'mean|dev|':>11}{'toward':>8}"
          f"{'of':>6}{'share':>8}{'z':>7}")
    props["absdev_q"] = props.groupby("market")["dev"].transform(
        lambda s: pd.qcut(s.abs(), 4, labels=False, duplicates="drop"))
    labels = {0: "Q1 smallest", 1: "Q2", 2: "Q3", 3: "Q4 largest"}
    for q, g in props.groupby("absdev_q"):
        if q != q:
            continue
        tw, n_nz, share, z, pv = toward_share(g)
        if n_nz == 0:
            continue
        print(f"  {labels.get(int(q), str(q)):<12}{len(g):>6}"
              f"{g['dev'].abs().mean():>11.2f}{tw:>8}{n_nz:>6}"
              f"{share:>8.3f}{z:>7.2f}")

    # ------------------------------------------------------------- section 4
    print("\n" + "=" * 78)
    print("4. IS LINE MOVEMENT INFORMATIVE AT ALL?")
    print("=" * 78)
    print("  This validates the premise. Regression of (actual - first line)")
    print("  on (last line - first line). If b is clearly positive, movement")
    print("  tracks the truth and section 2 is a fair test. If b is near zero,")
    print("  movement here is mostly noise and section 2 had no power.\n")

    bets = fetch_all(client, "bets")
    if bets.empty:
        print("  no rows in bets, cannot join outcomes")
    else:
        for c in ("result", "line", "projection"):
            if c in bets.columns:
                bets[c] = pd.to_numeric(bets[c], errors="coerce")
        b = bets[bets["market"].isin(YARDAGE)].copy()
        if "outcome" in b.columns:
            b = b[b["outcome"].isin(["WIN", "LOSS"])]
        b = b.dropna(subset=["result"])
        b = b[["season", "week", "market", "player", "result"]].drop_duplicates(
            subset=["season", "week", "market", "player"])
        j = props.merge(b, on=["season", "week", "market", "player"], how="inner")
        print(f"  {len(j)} props joined to a graded result")
        if len(j) >= 40:
            j["out_first"] = j["result"] - j["first_val"]
            f = ols(j["move"], j["out_first"])
            if f:
                t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
                print(f"  movement -> outcome:  b {f['beta']:+.3f}  "
                      f"SE {f['se_beta']:.3f}  t {t:+.2f}  R2 {f['r2']:.4f}")
                print("  (b near 1 would mean the move was fully justified by")
                print("   the eventual outcome; b near 0 means it was noise)")
            # and the same rows scored the model's way, for a direct comparison
            f2 = ols(j["dev"], j["out_first"])
            if f2:
                t2 = f2["beta"] / f2["se_beta"] if f2["se_beta"] > 0 else np.nan
                print(f"  model dev -> outcome: b {f2['beta']:+.3f}  "
                      f"SE {f2['se_beta']:.3f}  t {t2:+.2f}  R2 {f2['r2']:.4f}")
        else:
            print("  too few joined rows to fit")

    # ------------------------------------------------------------- section 5
    print("\n" + "=" * 78)
    print("5. ANYTIME TD, IN IMPLIED PROBABILITY")
    print("=" * 78)
    print("  TD has no line, so movement is measured in implied probability")
    print("  from the over odds, and the projection is already a probability.\n")

    td = summarise_props(lines, ("anytime_td",), "over_odds")
    if len(td) < 40:
        print(f"  only {len(td)} TD props with 2+ snapshots, skipping")
    else:
        td["first_p"] = td["first_val"].apply(american_to_prob)
        td["last_p"] = td["last_val"].apply(american_to_prob)
        td = td.dropna(subset=["first_p", "last_p", "first_proj"])
        # stored projection is on a 0-100 scale for this market
        td["proj_p"] = td["first_proj"] / 100.0
        td["move"] = td["last_p"] - td["first_p"]
        td["dev"] = td["proj_p"] - td["first_p"]
        f = ols(td["dev"], td["move"])
        tw, n_nz, share, z, pv = toward_share(td)
        if f:
            t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
            print(f"  n {f['n']}   b {f['beta']:+.3f}  SE {f['se_beta']:.3f}  "
                  f"t {t:+.2f}  R2 {f['r2']:.4f}")
        pv_s = f"{pv:.4f}" if pv == pv else "na"
        print(f"  toward {tw} of {n_nz} = {share:.3f}  z {z:+.2f}  p {pv_s}")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  Read section 4 FIRST. It decides whether the test has any power.")
    print("  If movement does not predict outcomes in this dataset, a null in")
    print("  section 2 says nothing about the model.")
    print("  Then section 2. A positive b with a CI excluding zero would be the")
    print("  first positive evidence of real signal anywhere in this project,")
    print("  and it would mean the model is early rather than wrong. That is")
    print("  compatible with beta_check: being early and beating the closing")
    print("  line are different claims.")
    print("  Section 3 is the consistency check. Real timing signal should be")
    print("  strongest where the disagreement is largest.")
    print("  Captures taken minutes apart are mostly noise, so compare the")
    print("  median hours in section 1 across markets before trusting a")
    print("  difference between them.")


if __name__ == "__main__":
    main()
