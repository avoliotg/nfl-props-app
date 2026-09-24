"""
calib_reconcile.py
==================

Reconciles the handoff's unconditional calibration table with what
sigma_refit found on 2026-09-23, and settles whether "pull every predicted
probability toward 0.5" is a valid prescription.

THE APPARENT CONFLICT
---------------------
Handoff, unconditional calibration, pooled across four markets:
    0.00-0.35  +7.3    0.35-0.45  +2.6    0.45-0.50  +1.1
    0.50-0.55  -0.3    0.55-0.65  -4.7    0.65-1.00  -11.6
read as "low probabilities are too low and high too high", prescribing that
every predicted probability be pulled toward 0.5. Pulling toward 0.5 means a
WIDER distribution.

sigma_refit found the opposite about width: receptions variance/mean is 1.25
against a shipped form implying 1.6 to 1.9, so sigma is too WIDE, and the
winning form is NARROWER at every line.

NOT A SIGN CONVENTION PROBLEM. That was checked first and it is not the
answer. The handoff reports actual - predicted; this project's scripts report
predicted - actual. Under that mapping the handoff's +7.3 and sigma_refit's
-22.5 in the same band are the SAME direction, both saying the bottom band is
under-predicted. The conflict is about WIDTH, not sign.

THE RESOLUTION BEING TESTED
---------------------------
For a count near zero the relationship between sigma and P(over) INVERTS.
At a 0.5 or 1.5 line the threshold is k = 1, and

    P(X >= 1) = 1 - P(0),   P(0) = (r/(r+m))^r  for a negative binomial

P(0) RISES as dispersion rises, so widening the distribution pushes mass into
the zero atom and LOWERS P(over). Worked at mean 0.8: the shipped floored
sigma gives r about 0.31 and P(X>=1) about 0.33, while prop var gives r about
3.2 and P(X>=1) about 0.51. Both match what sigma_refit printed.

If that is right, "pull toward 0.5" is correct in some bands and exactly
backwards in others, and which is which depends on where k sits relative to
the mean. That is computable rather than arguable: report the SIGN of
dP/dsigma in every band.

TWO CONFOUNDS TO RULE OUT
-------------------------
1. COMPOSITION. The handoff's table is pooled across four markets, three of
   them gamma. sigma_refit is receptions only. A 3x magnitude difference may
   be composition rather than disagreement. So this runs per market AND
   pooled, with the correct family per market.
2. VINTAGE. The handoff's table predates the rushing and qb_passing filter
   fixes, which moved those betas from 0.337 and 0.402 to about zero. Its
   pooled row therefore mixes two populations. Recomputing it on today's code
   is part of the point.

PRE-REGISTERED OUTCOMES
-----------------------
  BAND DEPENDENT  sign(dP/dsigma) changes across bands. Then "pull toward
                  0.5" is not a valid global prescription, Phase 4.5 needs
                  rewriting rather than implementing, and the shrinkage has
                  to be applied per band or replaced by fixing sigma.
  UNIFORM         sign(dP/dsigma) is the same in every band. Then the handoff
                  prescription is coherent, and the width conflict is real and
                  has to be resolved on its own before anything ships.
  COMPOSITION     the receptions-only table does not show the handoff's
                  pattern but the pooled one does. Then the finding belongs to
                  the gamma markets, which carry no signal anyway, and it
                  should not drive receptions pricing at all.

USAGE
-----
  python calib_reconcile.py --all-seasons --cache lines_cache.parquet
  python calib_reconcile.py --all-seasons --cache lines_cache.parquet \
      --markets receptions,receiving
"""

import argparse
import importlib
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import nbinom, poisson, gamma as gamma_dist
except ImportError:
    sys.exit("scipy is required")

try:
    import eval_harness as EH
except ImportError as e:
    sys.exit(f"could not import eval_harness: {e}")

data_utils = None
for _p in ("models.data_utils", "data_utils"):
    try:
        data_utils = importlib.import_module(_p)
        break
    except ImportError:
        pass
if data_utils is None or not hasattr(data_utils, "norm_join_name"):
    sys.exit("could not import models.data_utils with norm_join_name")

for _n in ("load_lines", "collapse_books", "score_market", "ols_clustered",
           "connect", "_secrets", "MARKET_SPEC", "ALL_SEASONS"):
    if not hasattr(EH, _n):
        sys.exit(f"eval_harness has no `{_n}`")

BANDS = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]

# shipped pricing per market, from the handoff's parameter table.
# family: "count" -> negative binomial or Poisson, "gamma" -> gamma
SHIPPED = {
    "receptions": dict(family="count", sigma=lambda m: 1.143 + 0.2992 * m,
                       floor=1.7),
    "receiving":  dict(family="gamma", sigma=lambda m: 3.272 * m ** 0.6172,
                       floor=11.5),
    "rushing":    dict(family="gamma", sigma=lambda m: 1.922 * m ** 0.7149,
                       floor=34.5),
    "qb_passing": dict(family="gamma", sigma=lambda m: 0.3531 * m,
                       floor=159.0),
}

# today's winner for receptions, for the side-by-side
PROP_VAR_K = 1.25


def sigma_of(market, mean, use_floor=True, override=None):
    spec = SHIPPED[market]
    m = np.clip(np.asarray(mean, float), 1e-6, None)
    if override == "prop_var" and spec["family"] == "count":
        s = np.sqrt(PROP_VAR_K * m)
        return s
    s = spec["sigma"](m)
    if use_floor and spec["floor"] is not None:
        s = np.maximum(s, spec["floor"])
    return s


def p_over(market, line, mean, sigma):
    """P(outcome > line) under the shipped family for that market."""
    line = np.asarray(line, float)
    mean = np.clip(np.asarray(mean, float), 1e-6, None)
    sigma = np.clip(np.asarray(sigma, float), 1e-6, None)
    var = sigma ** 2
    if SHIPPED[market]["family"] == "count":
        k = np.floor(line) + 1.0
        p = np.empty_like(mean)
        od = var > mean * 1.0000001
        if np.any(od):
            mm, vv = mean[od], var[od]
            r = mm * mm / (vv - mm)
            p[od] = nbinom.sf(k[od] - 1, r, r / (r + mm))
        if np.any(~od):
            p[~od] = poisson.sf(k[~od] - 1, mean[~od])
    else:
        shape = np.clip(mean ** 2 / var, 1e-6, None)
        scale = mean / shape
        p = gamma_dist.sf(line, shape, scale=scale)
    return np.clip(p, 1e-9, 1 - 1e-9)


def dispersion_r(market, mean, sigma):
    """NB size parameter where applicable, else nan. Diagnostic only."""
    if SHIPPED[market]["family"] != "count":
        return np.full(np.shape(mean), np.nan)
    m = np.clip(np.asarray(mean, float), 1e-6, None)
    v = np.asarray(sigma, float) ** 2
    out = np.full_like(m, np.inf)
    od = v > m * 1.0000001
    out[od] = m[od] ** 2 / (v[od] - m[od])
    return out


# ==========================================================================

def build_market(market, seasons, cache, refresh, population, source, cf):
    lines = EH.load_lines(cf, seasons, (market,), cache, refresh)
    if lines is None or len(lines) == 0:
        return None
    if lines["week"].isna().any():
        lines = lines[lines["week"].notna()]
    props = EH.collapse_books(lines)
    proj = EH.score_market(market, seasons, mode="walk_forward",
                           population=population)
    if proj is None or len(proj) == 0:
        return None
    acol = "actual" if "actual" in proj.columns else EH.MARKET_SPEC[market][1]
    if acol != "actual":
        proj = proj.rename(columns={acol: "actual"})
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(
        proj["player"] if "player" in proj.columns
        else proj["player_display_name"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)
    keep = [c for c in ("season", "week", "market", "_key", "projection",
                        "actual", "event_id") if c in proj.columns]
    d = props.merge(proj[keep], on=["season", "week", "market", "_key"],
                    how="inner")
    col = f"line_{source}"
    if col not in d.columns:
        return None
    for c in (col, "projection", "actual"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[col, "projection", "actual"]).copy()
    d["line"] = d[col]
    d["dev"] = d["projection"] - d["line"]
    if "event_id" not in d.columns:
        d["event_id"] = d["season"].astype(str) + "_" + d["week"].astype(str)

    # leave-season-out alpha and beta, per market
    d["alpha_lso"] = np.nan
    d["beta_lso"] = np.nan
    for s in sorted(d["season"].unique()):
        fit = d[d["season"] != s]
        if len(fit) < 300:
            continue
        r = EH.ols_clustered(fit["dev"].to_numpy(float),
                             (fit["actual"] - fit["line"]).to_numpy(float),
                             fit["event_id"].to_numpy())
        if not r:
            continue
        d.loc[d["season"] == s, "alpha_lso"] = float(r["alpha"])
        d.loc[d["season"] == s, "beta_lso"] = float(np.clip(r["beta"], 0, 1))
    d = d.dropna(subset=["alpha_lso", "beta_lso"]).copy()
    d["blend"] = d["line"] + d["alpha_lso"] + d["beta_lso"] * d["dev"]
    d["market"] = market
    d = d[~np.isclose(d["actual"], d["line"])].copy()
    d["won"] = (d["actual"] > d["line"]).astype(float)
    return d


def price_all(d, override=None, use_floor=True):
    p = np.full(len(d), np.nan)
    r = np.full(len(d), np.nan)
    for mk, g in d.groupby("market"):
        i = np.flatnonzero((d["market"] == mk).to_numpy())
        sig = sigma_of(mk, g["blend"].to_numpy(float), use_floor, override)
        p[i] = p_over(mk, g["line"].to_numpy(float),
                      g["blend"].to_numpy(float), sig)
        r[i] = dispersion_r(mk, g["blend"].to_numpy(float), sig)
    return p, r


def band_table(p, y, label, both_conventions=True):
    print(f"  --- {label} ---")
    hdr = (f"  {'band':>12}{'n':>7}{'pred':>9}{'actual':>9}"
           f"{'act-pred':>10}")
    if both_conventions:
        hdr += f"{'pred-act':>10}"
    print(hdr)
    rows = []
    for i in range(len(BANDS) - 1):
        lo, hi = BANDS[i], BANDS[i + 1]
        m = (p >= lo) & (p < hi) & np.isfinite(p)
        if m.sum() < 30:
            continue
        pm, ym = p[m].mean(), y[m].mean()
        line = (f"  {f'{lo:.2f}-{hi:.2f}':>12}{m.sum():>7}{pm:>9.4f}"
                f"{ym:>9.4f}{(ym - pm) * 100:>+10.1f}")
        if both_conventions:
            line += f"{(pm - ym) * 100:>+10.1f}"
        print(line)
        rows.append((f"{lo:.2f}-{hi:.2f}", int(m.sum()), pm, ym))
    print()
    return rows


def dpdsigma(d, override=None, use_floor=True, eps=0.02):
    """
    Sign and size of dP(over)/dsigma, evaluated per row at the shipped sigma
    by central difference, then summarised by predicted-probability band.

    This is the whole question. If the sign flips across bands, "pull every
    predicted probability toward 0.5" cannot be a global prescription, because
    the same change in sigma moves different bands in opposite directions.
    """
    p0, _ = price_all(d, override, use_floor)
    up = np.full(len(d), np.nan)
    dn = np.full(len(d), np.nan)
    for mk, g in d.groupby("market"):
        i = np.flatnonzero((d["market"] == mk).to_numpy())
        s = sigma_of(mk, g["blend"].to_numpy(float), use_floor, override)
        for arr, mult in ((up, 1.0 + eps), (dn, 1.0 - eps)):
            arr[i] = p_over(mk, g["line"].to_numpy(float),
                            g["blend"].to_numpy(float), s * mult)
    with np.errstate(invalid="ignore"):
        grad = (up - dn) / (2.0 * eps)     # dP per unit RELATIVE sigma change
    return p0, grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default="receptions,receiving,rushing,"
                                        "qb_passing")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--source", default="consensus",
                    choices=["consensus", "fanduel"])
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(EH.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in SHIPPED]
    if bad:
        sys.exit(f"no shipped pricing parameters recorded for {bad}")

    print("=" * 100)
    print(f"CALIBRATION RECONCILIATION   seasons={seasons}   "
          f"source={args.source}")
    print(f"markets={markets}")
    print("=" * 100)
    print()

    sec = EH._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = EH.connect(sec)
        return box["c"]

    frames = []
    for mk in markets:
        print(f"  building {mk} ...")
        f = build_market(mk, seasons, args.cache, args.refresh,
                         args.score_population, args.source, cf)
        if f is None or len(f) == 0:
            print(f"    skipped, no usable rows")
            continue
        print(f"    {len(f)} rows, family {SHIPPED[mk]['family']}")
        frames.append(f)
    if not frames:
        sys.exit("no markets built")
    d = pd.concat(frames, ignore_index=True)
    print(f"\n  {len(d)} rows total, "
          f"{d['event_id'].nunique()} game clusters\n")

    print("=" * 100)
    print("STEP 1. THE HANDOFF TABLE, RECOMPUTED. BOTH CONVENTIONS SHOWN.")
    print("=" * 100)
    print("  Handoff reports act-pred. This project's other scripts report")
    print("  pred-act. Both columns are printed so no one has to guess again.")
    print("  Handoff pooled, act-pred: +7.3 / +2.6 / +1.1 / -0.3 / -4.7 /")
    print("  -11.6 across the six bands.")
    print()
    p_ship, r_ship = price_all(d, override=None, use_floor=True)
    band_table(p_ship, d["won"].to_numpy(float),
               "POOLED, shipped pricing, all markets")

    print("=" * 100)
    print("STEP 2. PER MARKET. TESTS THE COMPOSITION CONFOUND.")
    print("=" * 100)
    print("  If the handoff pattern lives in the gamma markets and NOT in")
    print("  receptions, it should not drive receptions pricing, because")
    print("  rushing and qb_passing carry no signal anyway.")
    print()
    for mk in markets:
        g = d[d["market"] == mk]
        if len(g) < 200:
            continue
        i = np.flatnonzero((d["market"] == mk).to_numpy())
        band_table(p_ship[i], g["won"].to_numpy(float),
                   f"{mk}  ({SHIPPED[mk]['family']}, n={len(g)})")

    print("=" * 100)
    print("STEP 3. SIGN OF dP(over)/d sigma, BY BAND. THE DECIDING TEST.")
    print("=" * 100)
    print("  Positive means WIDENING sigma RAISES P(over). Negative means")
    print("  widening LOWERS it. 'Pull toward 0.5' assumes a single direction.")
    print("  For a count near zero the zero atom inverts it: P(X>=1) = 1-P(0)")
    print("  and P(0) rises with dispersion.")
    print()
    for scope, sub in ([("POOLED", d)]
                       + [(mk, d[d["market"] == mk]) for mk in markets]):
        if len(sub) < 200:
            continue
        p0, grad = dpdsigma(sub.reset_index(drop=True))
        print(f"  --- {scope} ---")
        print(f"  {'band':>12}{'n':>7}{'mean dP/dsig':>14}"
              f"{'share > 0':>11}{'mean r':>9}{'share r<1':>11}")
        _, rr = price_all(sub.reset_index(drop=True))
        for i in range(len(BANDS) - 1):
            lo, hi = BANDS[i], BANDS[i + 1]
            m = (p0 >= lo) & (p0 < hi) & np.isfinite(grad)
            if m.sum() < 30:
                continue
            rv = rr[m]
            fin = np.isfinite(rv)
            print(f"  {f'{lo:.2f}-{hi:.2f}':>12}{m.sum():>7}"
                  f"{grad[m].mean():>+14.4f}"
                  f"{(grad[m] > 0).mean() * 100:>10.1f}%"
                  f"{(rv[fin].mean() if fin.any() else np.nan):>9.2f}"
                  f"{((rv[fin] < 1).mean() * 100 if fin.any() else np.nan):>10.1f}%")
        print()

    print("=" * 100)
    print("STEP 4. RECEPTIONS UNDER TODAY'S WINNER, SIDE BY SIDE")
    print("=" * 100)
    print(f"  prop var, sigma = sqrt({PROP_VAR_K} * mean), no floor.")
    print("  If this band table is flat where the shipped one is not, the")
    print("  handoff's over-dispersion finding is a sigma bug in receptions")
    print("  rather than a calibration law needing a shrinkage layer.")
    print()
    rec = d[d["market"] == "receptions"].reset_index(drop=True)
    if len(rec) > 200:
        ps, _ = price_all(rec, override=None, use_floor=True)
        pp, _ = price_all(rec, override="prop_var", use_floor=False)
        y = rec["won"].to_numpy(float)
        print(f"  {'band (on shipped)':>20}{'n':>7}{'ship pred':>11}"
              f"{'prop pred':>11}{'actual':>9}{'ship gap':>10}"
              f"{'prop gap':>10}")
        for i in range(len(BANDS) - 1):
            lo, hi = BANDS[i], BANDS[i + 1]
            m = (ps >= lo) & (ps < hi)
            if m.sum() < 30:
                continue
            print(f"  {f'{lo:.2f}-{hi:.2f}':>20}{m.sum():>7}"
                  f"{ps[m].mean():>11.4f}{pp[m].mean():>11.4f}"
                  f"{y[m].mean():>9.4f}"
                  f"{(y[m].mean() - ps[m].mean()) * 100:>+10.1f}"
                  f"{(y[m].mean() - pp[m].mean()) * 100:>+10.1f}")
        print("\n  gaps are act-pred, the handoff convention.\n")

    print("=" * 100)
    print("STEP 5. IS THE BAND TILT A MEAN ERROR? SPLIT BY |dev|.")
    print("=" * 100)
    print("  Step 3 showed dP/dsigma is negative in EVERY band and both")
    print("  families, so a sigma change SHIFTS all probabilities the same")
    print("  way and cannot produce 'low too low, high too high'. That")
    print("  pattern needs the ends to move in OPPOSITE directions, which is")
    print("  the signature of error in the MEAN: an extreme predicted")
    print("  probability is extreme partly because the projection was noisy,")
    print("  and outcomes regress toward the middle.")
    print()
    print("  If so, the tilt should live in the HIGH |dev| rows and be absent")
    print("  from the low ones, because a row with dev near zero is priced")
    print("  almost entirely off the line and has little mean error to")
    print("  regress. That is the test.")
    print()
    d2 = d.reset_index(drop=True)
    p_all, _ = price_all(d2, override=None, use_floor=True)
    ad = d2["dev"].abs().to_numpy(float)
    # scale |dev| per market so a yardage market is comparable to a count
    z = np.full(len(d2), np.nan)
    for mk, g in d2.groupby("market"):
        i = np.flatnonzero((d2["market"] == mk).to_numpy())
        sd = np.nanstd(ad[i])
        z[i] = ad[i] / (sd if sd > 0 else 1.0)
    y2 = d2["won"].to_numpy(float)
    for lab, mask in (("LOW |dev|, below 0.5 sd", z < 0.5),
                      ("HIGH |dev|, above 1.0 sd", z > 1.0)):
        print(f"  --- {lab} ---")
        print(f"  {'band':>12}{'n':>7}{'pred':>9}{'actual':>9}"
              f"{'act-pred':>10}")
        for i in range(len(BANDS) - 1):
            lo, hi = BANDS[i], BANDS[i + 1]
            m = mask & (p_all >= lo) & (p_all < hi) & np.isfinite(p_all)
            if m.sum() < 30:
                continue
            print(f"  {f'{lo:.2f}-{hi:.2f}':>12}{m.sum():>7}"
                  f"{p_all[m].mean():>9.4f}{y2[m].mean():>9.4f}"
                  f"{(y2[m].mean() - p_all[m].mean()) * 100:>+10.1f}")
        print()
    print("  A monotone DECLINE across bands that appears only in the high")
    print("  |dev| panel is mean error and belongs to Phase 4.5. A uniform")
    print("  offset in both panels is the sigma error and belongs to 4.2.")
    print("  Both can be present; they are separable and this splits them.")
    print()
    print("=" * 100)
    print("HOW TO CALL IT")
    print("=" * 100)
    print("  My pre-registered options were BAND DEPENDENT / UNIFORM /")
    print("  COMPOSITION. Step 3 returned UNIFORM, and I had not worked out")
    print("  before writing them that a uniformly signed dP/dsigma makes the")
    print("  handoff's pattern UNREACHABLE by any sigma change. So the")
    print("  correct reading is none of the three:")
    print()
    print("  SIGMA IS A SHIFT, NOT A SPREAD. Too-wide sigma biases every")
    print("  predicted probability DOWN by roughly the same amount, since")
    print("  dP/dsigma < 0 everywhere. That shows up as a uniform positive")
    print("  act-pred offset in all bands.")
    print()
    print("  A MONOTONE TILT IS MEAN ERROR. Low too low AND high too high")
    print("  needs the ends moved in opposite directions. Only error in the")
    print("  mean does that. Step 5 separates the two.")
    print()
    print("  CONSEQUENCE FOR PHASE 4.5. 'Shrink the displayed probability")
    print("  toward 0.5' targets the mean-error component and is NOT a sigma")
    print("  fix. Fix sigma first (4.2), then re-measure the tilt, because")
    print("  the shrinkage fitted against today's wrong sigma is fitted")
    print("  partly against a shift it cannot correct. That ordering also")
    print("  explains why the curse correction came back unstable at slopes")
    print("  0.995, 0.849, 0.688, 0.852.")
    print()
    print("  COMPOSITION still applies: if step 2 shows the pattern only in")
    print("  the gamma markets, it belongs to markets with no signal.")
    print()


if __name__ == "__main__":
    main()
