"""
sigma_refit.py
==============

Which sigma form prices receptions best OUT OF SAMPLE?

WHY THIS EXISTS
---------------
sigma_by_dev on 2026-09-23 returned LEVEL ONLY: C within line +0.0000, CI
[-0.048, +0.053]. No deviation term. But it also found the conditional
residual var/mean pinned at 1.20 to 1.23 across five consecutive line values,
against a shipped form that is linear in the mean and floored at 1.7. Constant
var/mean means variance PROPORTIONAL to the mean, which is one parameter, and
it explains why the doc's imposed sqrt beat a fitted exponent: sqrt is the
correct shape, not a lucky restriction.

Three independent sources agree on the value. Conditional residuals 1.22,
FanDuel alt ladders 1.38, the sigma_by_dev joint fit 1.21.

THE DELIVERABLE IS P(over) ACCURACY, NOT RESIDUAL FIT
-----------------------------------------------------
A form that fits residuals better can still price worse, because P(over) at a
half-integer line depends on the whole shape near k, not on the second moment.
So every candidate is scored on realized over/under outcomes:

  log loss   proper scoring rule, punishes confident errors
  Brier      proper, less sensitive to the tails
  worst gap  largest |predicted - actual| across the handoff's calibration
             bands. A form that wins on average while ruining one band is
             not an improvement; the handoff's bands are where the money is.
  zero-dev   the Phase 4.1 check, priced against measured at |dev| < 0.25

EVERY sigma parameter is fitted LEAVE-SEASON-OUT, as are alpha and beta. A
form fitted on the rows it scores wins by construction.

PRE-REGISTERED DECISION RULE (written before the first run)
-----------------------------------------------------------
A candidate replaces the shipped form only if it:
  1. beats shipped on log loss in EVERY season, not on the pooled average, and
  2. does not increase the worst-band gap.
A form that wins on average and loses in one season is the instability that
killed the winner's curse correction (slopes 0.995, 0.849, 0.688, 0.852).

THE FLOOR IS REPORTED, NOT FITTED
---------------------------------
The 1.7 floor is not a variance estimate, it is a guard against the negative
binomial moment match sending the dispersion parameter below 1 and piling mass
on zero. sigma_by_dev found the 0.00-0.35 band predicting 0.3091 against a
realized 0.5338, a 22.5 point miss on 281 props, which is that guard
misfiring. So each candidate reports how often the moment match produces
size < 1, and the floor is a separate decision made with that number in hand.

USAGE
-----
  python sigma_refit.py --all-seasons --cache lines_cache.parquet
  python sigma_refit.py --all-seasons --cache lines_cache.parquet --source fanduel
"""

import argparse
import importlib
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import nbinom, poisson
    from scipy.optimize import minimize
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

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

SIGMA_A, SIGMA_B, SIGMA_FLOOR = 1.143, 0.2992, 1.7
BANDS = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]


# ==========================================================================
# candidate sigma forms
# ==========================================================================

def _nll(sig, resid):
    s = np.clip(sig, 1e-3, None)
    return float(np.sum(np.log(s) + 0.5 * (resid / s) ** 2))


class Form:
    """A sigma form. fit() returns params, sigma() maps mean -> sd."""

    def __init__(self, name, n_par, sigma_fn, fit_fn, floor=None, desc=""):
        self.name = name
        self.n_par = n_par
        self._sigma = sigma_fn
        self._fit = fit_fn
        self.floor = floor
        self.desc = desc

    def fit(self, mean, resid):
        return self._fit(np.asarray(mean, float), np.asarray(resid, float))

    def sigma(self, mean, par):
        s = self._sigma(np.clip(np.asarray(mean, float), 0.05, None), par)
        if self.floor is not None:
            s = np.maximum(s, self.floor)
        return s


def _fit_none(mean, resid):
    return ()


def _fit_prop(mean, resid):
    # sigma = sqrt(k * mean). closed form under normal ML:
    # minimise sum log(sqrt(k m)) + r^2/(2 k m)  ->  k = mean(r^2 / m)
    return (float(np.mean(resid ** 2 / np.clip(mean, 0.05, None))),)


def _fit_linvar(mean, resid):
    def f(th):
        k0, k1 = th
        v = k0 * mean + k1
        if np.any(v <= 1e-6):
            return 1e12
        return _nll(np.sqrt(v), resid)
    best, bc = None, np.inf
    for k0 in (0.8, 1.2, 1.6):
        for k1 in (0.0, 0.5, 1.0):
            r = minimize(f, [k0, k1], method="Nelder-Mead",
                         options=dict(maxiter=3000, xatol=1e-7, fatol=1e-7))
            if r.fun < bc:
                bc, best = r.fun, r
    return tuple(best.x)


def _fit_power(mean, resid):
    def f(th):
        A, B = th
        if A <= 0:
            return 1e12
        return _nll(A * mean ** B, resid)
    best, bc = None, np.inf
    for a0 in (0.8, 1.2, 1.8):
        for b0 in (0.3, 0.5, 0.7):
            r = minimize(f, [a0, b0], method="Nelder-Mead",
                         options=dict(maxiter=3000, xatol=1e-7, fatol=1e-7))
            if r.fun < bc:
                bc, best = r.fun, r
    return tuple(best.x)


def _fit_scale(mean, resid):
    # sigma = c * (A + B*mean). Closed form under normal ML:
    # c^2 = mean(r^2 / g^2) where g is the shipped shape.
    g = np.clip(SIGMA_A + SIGMA_B * mean, 1e-6, None)
    return (float(np.sqrt(np.mean((resid / g) ** 2))),)


FORMS = [
    Form("shipped+floor", 0,
         lambda m, p: SIGMA_A + SIGMA_B * m, _fit_none, floor=SIGMA_FLOOR,
         desc="what ships today"),
    Form("shipped-nofloor", 0,
         lambda m, p: SIGMA_A + SIGMA_B * m, _fit_none, floor=None,
         desc="isolates the floor's contribution"),
    Form("shipped x c", 1,
         lambda m, p: p[0] * (SIGMA_A + SIGMA_B * m), _fit_scale, floor=None,
         desc="shipped SHAPE, corrected LEVEL. the minimal change"),
    Form("shipped x c+floor", 1,
         lambda m, p: p[0] * (SIGMA_A + SIGMA_B * m), _fit_scale,
         floor=SIGMA_FLOOR, desc="same, floor kept"),
    Form("prop var", 1,
         lambda m, p: np.sqrt(p[0] * m), _fit_prop, floor=None,
         desc="sigma=sqrt(k*mean), constant var/mean, ONE parameter"),
    Form("prop var+floor", 1,
         lambda m, p: np.sqrt(p[0] * m), _fit_prop, floor=SIGMA_FLOOR,
         desc="same, with today's floor kept"),
    Form("lin var", 2,
         lambda m, p: np.sqrt(np.clip(p[0] * m + p[1], 1e-6, None)),
         _fit_linvar, floor=None,
         desc="sigma=sqrt(k0*mean+k1), allows extra spread at the bottom"),
    Form("power", 2,
         lambda m, p: p[0] * m ** p[1], _fit_power, floor=None,
         desc="A*mean^B, free exponent"),
]


# ==========================================================================
# pricing and scoring
# ==========================================================================

def price(line, mean, sigma):
    """P(X > line) and the moment-matched dispersion parameter."""
    line = np.asarray(line, float)
    mean = np.clip(np.asarray(mean, float), 1e-6, None)
    var = np.asarray(sigma, float) ** 2
    k = np.floor(line) + 1.0
    p = np.empty_like(mean)
    size = np.full_like(mean, np.inf)
    od = var > mean * 1.0000001
    if np.any(od):
        m, v = mean[od], var[od]
        r = m * m / (v - m)
        size[od] = r
        p[od] = nbinom.sf(k[od] - 1, r, r / (r + m))
    if np.any(~od):
        p[~od] = poisson.sf(k[~od] - 1, mean[~od])
    return np.clip(p, 1e-6, 1 - 1e-6), size


def score(p, y):
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    br = float(np.mean((p - y) ** 2))
    return ll, br


def worst_band(p, y, min_n=50):
    worst, where = 0.0, ""
    for i in range(len(BANDS) - 1):
        m = (p >= BANDS[i]) & (p < BANDS[i + 1])
        if m.sum() < min_n:
            continue
        gap = abs(p[m].mean() - y[m].mean()) * 100
        if gap > worst:
            worst, where = gap, f"{BANDS[i]:.2f}-{BANDS[i+1]:.2f}"
    return worst, where


# ==========================================================================

def paired_boot_ll(p_new, p_base, y, games, n_boot=300, seed=0):
    """
    Cluster bootstrap on the PAIRED log-loss difference, new minus base.
    Paired because both forms score the same rows, so most of the sampling
    variation is common and differencing removes it. Unpaired season-by-season
    comparison has almost no power here: on synthetic data where the planted
    truth WAS one of the candidates, every form sat within 0.0005 of the same
    log loss, because P(over) at a half-integer line is dominated by the mean
    and barely moves with sigma.
    """
    rng = np.random.default_rng(seed)
    uniq = pd.unique(games)
    idx = {g: np.flatnonzero(games == g) for g in uniq}

    def ll(p, yy):
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

    per = ll(p_new, y) - ll(p_base, y)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        out.append(float(np.mean(per[rows])))
    a = np.array(out)
    return float(np.mean(per)), float(np.percentile(a, 2.5)), \
        float(np.percentile(a, 97.5))


def build(seasons, cache, refresh, market, population, source):
    sec = EH._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = EH.connect(sec)
        return box["c"]

    print(f"  loading lines ...")
    lines = EH.load_lines(cf, seasons, (market,), cache, refresh)
    if lines["week"].isna().any():
        lines = lines[lines["week"].notna()]
    props = EH.collapse_books(lines)
    print(f"  scoring {market} walk-forward ...")
    proj = EH.score_market(market, seasons, mode="walk_forward",
                           population=population)
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
    for c in (col, "projection", "actual"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[col, "projection", "actual"]).copy()
    d["line"] = d[col]
    d["dev"] = d["projection"] - d["line"]
    d["adev"] = d["dev"].abs()
    if "event_id" not in d.columns:
        d["event_id"] = d["season"].astype(str) + "_" + d["week"].astype(str)
    print(f"  {len(d)} rows on line_{source}, "
          f"{d['event_id'].nunique()} clusters\n")
    return d


def add_blend_lso(d):
    print("=" * 100)
    print("LEAVE-SEASON-OUT alpha AND beta")
    print("=" * 100)
    d = d.copy()
    d["alpha_lso"] = np.nan
    d["beta_lso"] = np.nan
    print(f"  {'held out':>10}{'n_fit':>8}{'alpha':>9}{'beta':>8}")
    for s in sorted(d["season"].unique()):
        fit = d[d["season"] != s]
        if len(fit) < 300:
            continue
        r = EH.ols_clustered(fit["dev"].to_numpy(float),
                             (fit["actual"] - fit["line"]).to_numpy(float),
                             fit["event_id"].to_numpy())
        if not r:
            continue
        a, b = float(r["alpha"]), float(np.clip(r["beta"], 0.0, 1.0))
        d.loc[d["season"] == s, "alpha_lso"] = a
        d.loc[d["season"] == s, "beta_lso"] = b
        print(f"  {int(s):>10}{len(fit):>8}{a:>9.3f}{b:>8.3f}")
    d = d.dropna(subset=["alpha_lso", "beta_lso"]).copy()
    d["blend"] = d["line"] + d["alpha_lso"] + d["beta_lso"] * d["dev"]
    d["resid"] = d["actual"] - d["blend"]
    d["won"] = (d["actual"] > d["line"]).astype(float)
    d = d[~np.isclose(d["actual"], d["line"])].copy()
    print(f"\n  {len(d)} rows after dropping pushes\n")
    return d


def run(d, zero_dev):
    seasons = sorted(d["season"].unique())
    print("=" * 100)
    print("CANDIDATE FORMS, EVERY PARAMETER FITTED LEAVE-SEASON-OUT")
    print("=" * 100)
    for f in FORMS:
        print(f"  {f.name:<18}{f.n_par} par   {f.desc}")
    print()

    # out-of-sample predictions per form
    preds = {f.name: np.full(len(d), np.nan) for f in FORMS}
    fitted = {f.name: {} for f in FORMS}
    sizes = {f.name: np.full(len(d), np.nan) for f in FORMS}
    pos = {s: np.flatnonzero((d["season"] == s).to_numpy()) for s in seasons}

    for f in FORMS:
        for s in seasons:
            tr = d[d["season"] != s]
            te_i = pos[s]
            if len(tr) < 300 or len(te_i) == 0:
                continue
            par = f.fit(tr["blend"], tr["resid"])
            fitted[f.name][s] = par
            te = d.iloc[te_i]
            sig = f.sigma(te["blend"].to_numpy(float), par)
            p, sz = price(te["line"].to_numpy(float),
                          te["blend"].to_numpy(float), sig)
            preds[f.name][te_i] = p
            sizes[f.name][te_i] = sz

    y = d["won"].to_numpy(float)

    print("=" * 100)
    print("POOLED, OUT OF SAMPLE")
    print("=" * 100)
    print(f"  {'form':<18}{'log loss':>10}{'vs ship':>9}{'Brier':>9}"
          f"{'worst band':>12}{'where':>12}{'size<1':>8}")
    base_ll = None
    pooled = {}
    for f in FORMS:
        p = preds[f.name]
        ok = np.isfinite(p)
        ll, br = score(p[ok], y[ok])
        wb, where = worst_band(p[ok], y[ok])
        sub1 = float(np.mean(sizes[f.name][ok] < 1.0)) * 100
        if base_ll is None:
            base_ll = ll
        pooled[f.name] = dict(ll=ll, br=br, wb=wb)
        print(f"  {f.name:<18}{ll:>10.5f}{ll - base_ll:>+9.5f}{br:>9.5f}"
              f"{wb:>11.2f}pp{where:>12}{sub1:>7.1f}%")
    print("\n  size<1 is the share of rows where the moment match puts the")
    print("  negative binomial dispersion below 1, which piles mass on zero.")
    print("  That is what the floor exists to prevent.\n")

    print("=" * 100)
    print("LOG LOSS BY SEASON. The decision rule needs EVERY season, not the "
          "average.")
    print("=" * 100)
    print(f"  {'form':<18}" + "".join(f"{int(s):>11}" for s in seasons)
          + f"{'seasons won':>13}")
    base_by_season = {}
    for f in FORMS:
        cells, wins = "", 0
        for s in seasons:
            i = pos[s]
            p = preds[f.name][i]
            ok = np.isfinite(p)
            if ok.sum() < 50:
                cells += f"{'.':>11}"
                continue
            ll, _ = score(p[ok], y[i][ok])
            if f.name == FORMS[0].name:
                base_by_season[s] = ll
                cells += f"{ll:>11.5f}"
            else:
                d_ll = ll - base_by_season.get(s, np.nan)
                cells += f"{d_ll:>+11.5f}"
                if d_ll < 0:
                    wins += 1
        tag = "" if f.name == FORMS[0].name else f"{wins} of {len(seasons)}"
        print(f"  {f.name:<18}{cells}{tag:>13}")
    print("\n  First row is the shipped log loss. Later rows are the "
          "DIFFERENCE,\n  so negative is better.\n")

    print("=" * 100)
    print("CALIBRATION BY BAND, SHIPPED AGAINST THE BEST ONE-PARAMETER FORM")
    print("=" * 100)
    cmp_forms = [FORMS[0].name, "shipped x c", "prop var"]
    print(f"  {'band':>12}{'n':>7}" + "".join(f"{c:>20}" for c in cmp_forms))
    for i in range(len(BANDS) - 1):
        lo, hi = BANDS[i], BANDS[i + 1]
        m = (preds[FORMS[0].name] >= lo) & (preds[FORMS[0].name] < hi)
        if m.sum() < 50:
            continue
        cells = ""
        for c in cmp_forms:
            p = preds[c][m]
            ok = np.isfinite(p)
            if ok.sum() < 20:
                cells += f"{'.':>20}"
                continue
            cells += (f"{p[ok].mean():.3f}v{y[m][ok].mean():.3f}"
                      f"={(p[ok].mean()-y[m][ok].mean())*100:+.1f}").rjust(20)
        print(f"  {f'{lo:.2f}-{hi:.2f}':>12}{m.sum():>7}{cells}")
    print("\n  Bands are fixed on the SHIPPED predictions so the same rows "
          "are compared\n  across forms. pred v actual = gap in points.\n")

    print("=" * 100
          )
    print(f"ZERO DEVIATION CHECK, |dev| < {zero_dev}. Phase 4.1.")
    print("=" * 100)
    zi = np.flatnonzero((d["adev"] < zero_dev).to_numpy())
    print(f"  {len(zi)} rows")
    print(f"  {'form':<18}{'priced':>9}{'actual':>9}{'gap pp':>9}")
    for f in FORMS:
        p = preds[f.name][zi]
        ok = np.isfinite(p)
        if ok.sum() < 50:
            continue
        print(f"  {f.name:<18}{p[ok].mean():>9.4f}{y[zi][ok].mean():>9.4f}"
              f"{(p[ok].mean() - y[zi][ok].mean()) * 100:>9.2f}")
    print()

    print("=" * 100)
    print("FITTED PARAMETERS BY HELD-OUT SEASON. Stability matters as much "
          "as fit.")
    print("=" * 100)
    for f in FORMS:
        if f.n_par == 0:
            continue
        cells = "  ".join(
            f"{int(s)}: " + ",".join(f"{v:.3f}" for v in fitted[f.name][s])
            for s in seasons if s in fitted[f.name])
        print(f"  {f.name:<18}{cells}")
    print()

    print("=" * 100)
    print("PRE-REGISTERED DECISION")
    print("=" * 100)
    print("  REVISED before the first real run, for a reason the synthetic")
    print("  test exposed: log loss is nearly BLIND to sigma here. Every form")
    print("  landed within 0.0005 of the same log loss on data where the")
    print("  planted truth was one of the candidates, because P(over) at a")
    print("  half-integer line is driven by the mean. So calibration is the")
    print("  primary criterion and log loss is only a guard against making")
    print("  things worse.")
    print()
    print("  REPLACE requires all three:")
    print("    1. worst calibration band gap smaller than shipped by at")
    print("       least 0.25 pp. A rounding-level win is not a win.")
    print("    2. |zero-deviation gap| smaller by at least 0.25 pp")
    print("    3. paired log-loss difference not significantly WORSE")
    print("       (cluster bootstrap 95% LOWER bound <= 0). Demanding a")
    print("       significant log-loss IMPROVEMENT is unachievable here and")
    print("       would reject the correct form, as the synthetic run showed.")
    print()
    MARGIN = 0.25   # pp. a rounding-level win is not a win.
    base = FORMS[0].name
    pb = preds[base]
    okb = np.isfinite(pb)
    zi_all = (d["adev"] < zero_dev).to_numpy()

    def zgap(name):
        p = preds[name][zi_all]
        m = np.isfinite(p)
        return abs(p[m].mean() - y[zi_all][m].mean()) * 100

    base_z = zgap(base)
    games = d["event_id"].to_numpy()
    print(f"  {'form':<19}{'worst band':>12}{'zero-dev':>10}"
          f"{'d logloss':>11}{'95% CI':>20}{'seasons':>9}{'verdict':>16}")
    print(f"  {base:<19}{pooled[base]['wb']:>10.2f}pp{base_z:>9.2f}pp"
          f"{'baseline':>11}{'':>20}{'':>9}{'':>16}")
    for f in FORMS[1:]:
        p = preds[f.name]
        ok = okb & np.isfinite(p)
        dll, lo, hi = paired_boot_ll(p[ok], pb[ok], y[ok], games[ok])
        wb = pooled[f.name]["wb"]
        zg = zgap(f.name)
        wins = 0
        for s in seasons:
            i = pos[s]
            pp = preds[f.name][i]
            o = np.isfinite(pp)
            if o.sum() < 50:
                continue
            l1, _ = score(pp[o], y[i][o])
            if l1 < base_by_season.get(s, np.inf):
                wins += 1
        c1 = wb < pooled[base]["wb"] - MARGIN
        c2 = zg < base_z - MARGIN
        c3 = lo <= 0   # not significantly WORSE, not 'significantly better'
        verdict = "REPLACE" if (c1 and c2 and c3) else "keep shipped"
        print(f"  {f.name:<19}{wb:>10.2f}pp{zg:>9.2f}pp{dll:>+11.5f}"
              f"   [{lo:+.5f},{hi:+.5f}]{f'{wins}/{len(seasons)}':>9}"
              f"{verdict:>16}")
    print()
    print("  seasons is descriptive only. The floor decision is separate:")
    print("  read the size<1 column in the pooled table before removing it.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--market", default="receptions")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--source", default="consensus",
                    choices=["consensus", "fanduel"])
    ap.add_argument("--zero-dev", type=float, default=0.25)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(EH.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 100)
    print(f"SIGMA REFIT   market={args.market}   seasons={seasons}   "
          f"source={args.source}")
    print(f"shipped: max({SIGMA_A} + {SIGMA_B}*mean, {SIGMA_FLOOR})")
    print("=" * 100)
    print()

    d = build(seasons, args.cache, args.refresh, args.market,
              args.score_population, args.source)
    d = add_blend_lso(d)
    if args.save_rows:
        d.to_csv(args.save_rows, index=False)
    run(d, args.zero_dev)


if __name__ == "__main__":
    main()
