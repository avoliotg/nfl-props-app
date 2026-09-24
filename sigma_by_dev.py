"""
sigma_by_dev.py
===============

Does the conditional sd of receptions depend on |projection - line| as well as
on level, and is the shipped sigma too wide?

WHY THIS EXISTS
---------------
alpha_by_line_v4 on 2026-09-23 found the empirical-to-shipped sd ratio below
1.00 at ALL SIX FanDuel line values (0.89, 0.92, 0.95, 0.94, 0.93, 0.90) with
empirical var/mean in a tight 1.20 to 1.25 band. FanDuel's own alt ladders the
same day implied 1.38. The shipped formula implies 1.58 at a mean of 1.83.

Two reasons that is not yet a licence to shrink sigma.

1. THAT MEASUREMENT WAS AN UPPER BOUND, NOT THE QUANTITY.
   It took var(actual) WITHIN a line bucket. By the law of total variance that
   equals the mean conditional variance PLUS the variance of true means across
   players sharing the line. Kraft and Dotson can both sit on 3.5 with
   different true means. So the real conditional sigma is SMALLER than what
   was printed, and the finding is stronger than it looked, but the number
   itself was wrong for the purpose.

2. IT APPEARS TO CONTRADICT THE HANDOFF.
   The handoff measures the pricing layer as OVER-dispersed: predicted
   probabilities too extreme, 0.00-0.35 running +7.3 pp and 0.65-1.00 running
   -11.6 pp. Too-extreme probabilities need a sigma that is too NARROW.
   Both can hold at once if sigma is too wide at small |dev|, where it barely
   moves a P(over) already near 0.5, and too narrow at large |dev|, which is
   what drives the extreme probabilities that populate those bands. That is
   the Phase 4.3 hypothesis and one test settles it.

WHAT IT MEASURES
----------------
The residual is taken around the BLENDED mean, not the projection:

    resid = actual - (line + alpha + beta * (projection - line))

with alpha and beta fitted LEAVE-SEASON-OUT. Using the projection as the
anchor folds beta error into what you then call sigma, which is the error the
handoff records as "recommended replacing the shipped level-dependent sigma
with a flat residual SD. Wrong." Fitting alpha and beta on the same rows whose
spread you then measure understates that spread.

PRE-REGISTERED OUTCOMES (written before the first run)
------------------------------------------------------
  ADD THE TERM   the deviation coefficient C is positive with a cluster
                 bootstrap 95% interval excluding zero, AND residual sd rises
                 with |dev| WITHIN line value. Then sigma gains a deviation
                 term and the level constant moves in the same pass.
  LEVEL ONLY     C's interval includes zero, or the |dev| effect disappears
                 once line value is held fixed. Then sigma is simply too wide
                 everywhere, the handoff's over-dispersion finding needs a
                 separate explanation, and NOTHING changes until it has one.
  INCOHERENT     C positive but the overconfident calibration bands are NOT
                 populated by high-|dev| rows. Then the two findings are
                 unrelated and neither is ready.

The within-line check is the falsification. If the deviation effect vanishes
when the line is held fixed, it was level in disguise.

USAGE
-----
  python sigma_by_dev.py --all-seasons --cache lines_cache.parquet
  python sigma_by_dev.py --all-seasons --cache lines_cache.parquet --source fanduel
  python sigma_by_dev.py --all-seasons --cache lines_cache.parquet --boot 400
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
    sys.exit(f"could not import eval_harness: {e}\n"
             "run this from the repo root, alongside eval_harness.py")

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
        sys.exit(f"eval_harness has no `{_n}`; check the harness API")

# shipped receptions pricing, literals not imports, so an app change cannot
# silently move a measurement
SIGMA_A = 1.143
SIGMA_B = 0.2992
SIGMA_FLOOR = 1.7

DEV_BINS = [0.0, 0.25, 0.50, 1.00, 1.50, 99.0]
DEV_LAB = ["0.00-0.25", "0.25-0.50", "0.50-1.00", "1.00-1.50", "1.50+"]


def sigma_shipped(anchor):
    return np.maximum(SIGMA_A + SIGMA_B * np.asarray(anchor, float),
                      SIGMA_FLOOR)


def p_over_count(line, mean, sigma):
    line = np.asarray(line, float)
    mean = np.clip(np.asarray(mean, float), 1e-6, None)
    var = np.asarray(sigma, float) ** 2
    k = np.floor(line) + 1.0
    out = np.empty_like(mean)
    od = var > mean * 1.0000001
    if np.any(od):
        m, v = mean[od], var[od]
        r = m * m / (v - m)
        out[od] = nbinom.sf(k[od] - 1, r, r / (r + m))
    if np.any(~od):
        out[~od] = poisson.sf(k[~od] - 1, mean[~od])
    return out


# ==========================================================================
# frame
# ==========================================================================

def build_frame(seasons, cache, refresh, market, population, source):
    sec = EH._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = EH.connect(sec)
        return box["c"]

    print(f"  loading lines, seasons {seasons} ...")
    lines = EH.load_lines(cf, seasons, (market,), cache, refresh)
    if len(lines) == 0:
        sys.exit("no line rows")
    if lines["week"].isna().any():
        lines = lines[lines["week"].notna()]
    props = EH.collapse_books(lines)

    print(f"  scoring {market} walk-forward, population={population} ...")
    proj = EH.score_market(market, seasons, mode="walk_forward",
                           population=population)
    if proj is None or len(proj) == 0:
        sys.exit("score_market returned nothing")
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
        sys.exit(f"{col} not in the joined frame; saw {list(d.columns)}")
    for c in (col, "projection", "actual"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[col, "projection", "actual"]).copy()
    d["line"] = d[col]
    d["dev"] = d["projection"] - d["line"]
    d["adev"] = d["dev"].abs()
    if "event_id" not in d.columns:
        d["event_id"] = (d["season"].astype(str) + "_"
                         + d["week"].astype(str))
    print(f"  {len(d)} rows on line_{source}, "
          f"{d['event_id'].nunique()} game clusters\n")
    return d


def add_blend_lso(d):
    """
    Leave-season-out alpha and beta, then the blended mean and its residual.
    Beta clipped to [0, 1]: a negative beta means the projection is
    anti-informative and should be ignored, not inverted.
    """
    print("=" * 96)
    print("LEAVE-SEASON-OUT BLEND PARAMETERS")
    print("=" * 96)
    print(f"  {'held out':>10}{'n_fit':>8}{'alpha':>9}{'beta':>8}"
          f"{'se_beta':>9}")
    d = d.copy()
    d["alpha_lso"] = np.nan
    d["beta_lso"] = np.nan
    for s in sorted(d["season"].unique()):
        fit = d[d["season"] != s]
        if len(fit) < 300:
            print(f"  {int(s):>10}{len(fit):>8}     too few rows to fit")
            continue
        r = EH.ols_clustered(fit["dev"].to_numpy(float),
                             (fit["actual"] - fit["line"]).to_numpy(float),
                             fit["event_id"].to_numpy())
        if not r:
            continue
        a = float(r["alpha"])
        b = float(np.clip(r["beta"], 0.0, 1.0))
        m = d["season"] == s
        d.loc[m, "alpha_lso"] = a
        d.loc[m, "beta_lso"] = b
        print(f"  {int(s):>10}{len(fit):>8}{a:>9.3f}{b:>8.3f}"
              f"{float(r['se_beta']):>9.3f}")
    d = d.dropna(subset=["alpha_lso", "beta_lso"]).copy()
    d["blend"] = d["line"] + d["alpha_lso"] + d["beta_lso"] * d["dev"]
    d["resid"] = d["actual"] - d["blend"]
    print(f"\n  {len(d)} rows with a blend. Residual is taken around the "
          f"BLEND, not the\n  projection, so beta error is not counted as "
          f"sigma.\n")
    return d


# ==========================================================================
# bootstrap
# ==========================================================================

def boot_stat(d, fn, n_boot, seed=0):
    """Cluster bootstrap on event_id. Resamples games, not rows."""
    rng = np.random.default_rng(seed)
    games = d["event_id"].to_numpy()
    uniq = pd.unique(games)
    idx = {g: np.flatnonzero(games == g) for g in uniq}
    out = []
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        try:
            out.append(fn(d.iloc[rows]))
        except Exception:
            out.append(np.nan)
        if (i + 1) % 50 == 0:
            print(f"\r    bootstrap {i + 1}/{n_boot}", end="", flush=True)
    print("\r" + " " * 40 + "\r", end="")
    a = np.array(out, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan, np.nan
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ==========================================================================
# tables
# ==========================================================================

def table_level(d, min_n, anchor):
    print("=" * 96)
    print("STEP 1. LEVEL. Corrected version of alpha_by_line step 4.")
    print("=" * 96)
    print("  resid sd is now around the LSO blend, so this is the conditional")
    print("  sd rather than the within-bucket spread that mixes in player")
    print("  heterogeneity. Expect it BELOW the earlier upper bound.")
    print()
    print(f"  {'line':>5}{'n':>7}{'mean blend':>12}{'resid sd':>10}"
          f"{'sd ship':>9}{'ratio':>8}{'prior bound':>13}")
    for L, g in d.groupby("line", sort=True):
        if len(g) < min_n:
            continue
        mb = g["blend"].mean()
        sd = g["resid"].std(ddof=1)
        ship = float(sigma_shipped(mb if anchor == "mean" else
                                   g["projection"].mean()))
        prior = g["actual"].std(ddof=1) / ship
        print(f"  {L:>5.1f}{len(g):>7}{mb:>12.2f}{sd:>10.2f}{ship:>9.2f}"
              f"{sd / ship:>8.2f}{prior:>13.2f}")
    print("\n  If ratio < prior bound at every line, the law of total "
          "variance is\n  behaving and the earlier figure was the upper bound "
          "it should be.\n")


def table_dev(d, min_n, anchor, n_boot):
    print("=" * 96)
    print("STEP 2. DEVIATION, POOLED")
    print("=" * 96)
    d = d.copy()
    d["dbin"] = pd.cut(d["adev"], DEV_BINS, labels=DEV_LAB, right=False)
    print(f"  {'|dev| bin':>12}{'n':>7}{'mean blend':>12}{'resid sd':>10}"
          f"{'sd ship':>9}{'ratio':>8}{'95% CI on ratio':>20}")
    for lab in DEV_LAB:
        g = d[d["dbin"] == lab]
        if len(g) < min_n:
            print(f"  {lab:>12}{len(g):>7}   thin")
            continue

        def f(h, _anchor=anchor):
            a = (h["blend"] if _anchor == "mean" else h["projection"])
            return float(h["resid"].std(ddof=1)
                         / sigma_shipped(a.mean()))

        lo, hi = boot_stat(g, f, n_boot)
        mb = g["blend"].mean()
        sd = g["resid"].std(ddof=1)
        ship = float(sigma_shipped(mb if anchor == "mean"
                                   else g["projection"].mean()))
        print(f"  {lab:>12}{len(g):>7}{mb:>12.2f}{sd:>10.2f}{ship:>9.2f}"
              f"{sd / ship:>8.2f}   [{lo:.2f}, {hi:.2f}]")
    print("\n  Pooled is CONFOUNDED with level: large |dev| happens more on")
    print("  high-volume players. Step 3 is the test that matters.\n")


def fit_within(d, n_boot, seed=1):
    """
    C estimated PURELY within line value, by giving every line value its own
    level constant and sharing one deviation coefficient:

        sigma_i = A_{line(i)} * (1 + C * |dev_i|)

    Level is then absorbed entirely by the per-line constants, so C cannot
    pick up a level effect. That is the falsification the sign-counting
    version only approximated: on synthetic data with C_TRUE = 0, counting
    signs reported 4 of 5 line values positive, which happens by chance about
    19 percent of the time with five lines.

    For a fixed C the A's have a closed form under the normal likelihood, so
    this profiles out to a one-dimensional search and is fast enough to
    bootstrap.
    """
    print("=" * 96)
    print("STEP 3b. C IDENTIFIED WITHIN LINE VALUE. THE FALSIFICATION.")
    print("=" * 96)
    print("  One level constant per line value, one shared deviation")
    print("  coefficient. Level is fully absorbed, so C here cannot be level")
    print("  in disguise.")

    def profile_nll(C, h):
        if C <= -0.9:
            return 1e12
        f = 1.0 + C * h["adev"].to_numpy(float)
        if np.any(f <= 1e-6):
            return 1e12
        z = h["resid"].to_numpy(float) / f
        tot = float(np.sum(np.log(f)))
        for _, g in h.groupby("line", sort=False):
            zz = z[h["line"].to_numpy() == g["line"].iloc[0]]
            v = float(np.mean(zz ** 2))
            if v <= 0:
                return 1e12
            tot += 0.5 * len(zz) * (1.0 + np.log(v))
        return tot

    def fit(h):
        grid = np.linspace(-0.4, 1.2, 65)
        vals = [profile_nll(c, h) for c in grid]
        c0 = grid[int(np.argmin(vals))]
        fine = np.linspace(max(c0 - 0.05, -0.85), c0 + 0.05, 41)
        vals2 = [profile_nll(c, h) for c in fine]
        return float(fine[int(np.argmin(vals2))])

    Cw = fit(d)
    lo, hi = boot_stat(d, fit, n_boot, seed=seed)
    print(f"\n  C within line   {Cw:+.4f}")
    print(f"  cluster bootstrap 95% CI   [{lo:+.4f}, {hi:+.4f}]")
    excl = np.isfinite(lo) and lo > 0
    print(f"  interval excludes zero: {'YES' if excl else 'no'}\n")
    return Cw, lo, hi, excl


def table_within(d, min_n):
    print("=" * 96)
    print("STEP 3a. RESIDUAL SD BY |dev| WITHIN LINE VALUE, DESCRIPTIVE")
    print("=" * 96)
    print("  Read the shape, not the signs. Step 3b is the test.")
    print()
    labs = DEV_LAB[:4]
    print(f"  {'line':>5}{'n':>7}" + "".join(f"{l:>11}" for l in labs)
          + f"{'lo->hi':>10}")
    d = d.copy()
    d["dbin"] = pd.cut(d["adev"], DEV_BINS, labels=DEV_LAB, right=False)
    for L, g in d.groupby("line", sort=True):
        if len(g) < min_n:
            continue
        cells, vals = "", []
        for lab in labs:
            h = g[g["dbin"] == lab]
            if len(h) < 60:
                cells += f"{'.':>11}"
                vals.append(np.nan)
                continue
            sd = h["resid"].std(ddof=1)
            cells += f"{sd:.2f}({len(h)})".rjust(11)
            vals.append(sd)
        ok = [v for v in vals if np.isfinite(v)]
        delta = (ok[-1] - ok[0]) if len(ok) >= 2 else np.nan
        print(f"  {L:>5.1f}{len(g):>7}{cells}{delta:>+10.2f}")
    print()


def fit_joint(d, n_boot, seed=0):
    """
    sigma_i = A * mean_i^B * (1 + C * |dev_i|)

    Fitted by normal maximum likelihood on the residuals. Normality is an
    approximation for a count, but it is a standard and unbiased way to
    estimate a variance function, and the deliverable is C's sign and
    interval rather than a distributional claim.

    C is the whole question. C > 0 with an interval excluding zero means
    sigma must carry a deviation term.
    """
    print("=" * 96)
    print("STEP 4. JOINT FIT  sigma = A * mean^B * (1 + C*|dev|)")
    print("=" * 96)

    def nll(th, h):
        A, B, C = th
        if A <= 0 or C <= -0.9:
            return 1e12
        m = np.clip(h["blend"].to_numpy(float), 0.05, None)
        s = A * m ** B * (1.0 + C * h["adev"].to_numpy(float))
        s = np.clip(s, 1e-3, None)
        r = h["resid"].to_numpy(float)
        return float(np.sum(np.log(s) + 0.5 * (r / s) ** 2))

    def fit(h):
        best, bc = None, np.inf
        for a0 in (0.8, 1.2, 1.8):
            for b0 in (0.3, 0.5, 0.7):
                for c0 in (0.0, 0.2):
                    r = minimize(nll, [a0, b0, c0], args=(h,),
                                 method="Nelder-Mead",
                                 options=dict(maxiter=4000, xatol=1e-6,
                                              fatol=1e-6))
                    if r.fun < bc:
                        bc, best = r.fun, r
        return best.x

    A, B, C = fit(d)
    print(f"  A {A:.4f}   B {B:.4f}   C {C:.4f}")
    lo, hi = boot_stat(d, lambda h: fit(h)[2], n_boot)
    print(f"  C cluster bootstrap 95% CI  [{lo:.4f}, {hi:.4f}]")
    excl = np.isfinite(lo) and np.isfinite(hi) and lo > 0
    print(f"  interval excludes zero: {'YES' if excl else 'no'}")

    print(f"\n  shipped is max({SIGMA_A} + {SIGMA_B}*mean, {SIGMA_FLOOR}), "
          f"fitted is above")
    print(f"  {'mean':>6}{'|dev|':>7}{'fitted':>9}{'shipped':>9}{'ratio':>8}")
    for m in (1.8, 2.6, 3.6, 4.5):
        for dv in (0.0, 1.0, 2.0):
            f = A * m ** B * (1.0 + C * dv)
            s = float(sigma_shipped(m))
            print(f"  {m:>6.1f}{dv:>7.1f}{f:>9.2f}{s:>9.2f}{f / s:>8.2f}")
    print()
    return A, B, C, lo, hi, excl


def table_bands(d, A, B, C, anchor):
    """
    Ties this back to the handoff's over-dispersion finding. If the
    overconfident bands are populated by high-|dev| rows, the two findings are
    one finding and a deviation term addresses both.
    """
    print("=" * 96)
    print("STEP 5. WHO POPULATES THE OVERCONFIDENT CALIBRATION BANDS?")
    print("=" * 96)
    print("  Handoff: unconditional calibration runs +7.3 pp at 0.00-0.35 and")
    print("  -11.6 pp at 0.65-1.00. If those bands are high-|dev| rows, the")
    print("  over-dispersion finding and the sigma finding are the same thing.")
    print()
    d = d.copy()
    anc = d["blend"] if anchor == "mean" else d["projection"]
    d["p_ship"] = p_over_count(d["line"].to_numpy(float),
                               d["blend"].to_numpy(float),
                               sigma_shipped(anc.to_numpy(float)))
    sig_fit = (A * np.clip(d["blend"].to_numpy(float), 0.05, None) ** B
               * (1.0 + C * d["adev"].to_numpy(float)))
    d["p_fit"] = p_over_count(d["line"].to_numpy(float),
                              d["blend"].to_numpy(float), sig_fit)
    d["won"] = (d["actual"] > d["line"]).astype(float)
    live = ~np.isclose(d["actual"], d["line"])
    d = d[live]

    edges = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]
    print(f"  {'band':>12}{'n':>7}{'mean |dev|':>12}{'share |dev|>1':>15}"
          f"{'pred ship':>11}{'pred fit':>10}{'actual':>9}"
          f"{'gap ship':>10}{'gap fit':>9}")
    for i in range(len(edges) - 1):
        lo_e, hi_e = edges[i], edges[i + 1]
        g = d[(d["p_ship"] >= lo_e) & (d["p_ship"] < hi_e)]
        if len(g) < 50:
            continue
        print(f"  {f'{lo_e:.2f}-{hi_e:.2f}':>12}{len(g):>7}"
              f"{g['adev'].mean():>12.2f}"
              f"{(g['adev'] > 1).mean() * 100:>14.1f}%"
              f"{g['p_ship'].mean():>11.4f}{g['p_fit'].mean():>10.4f}"
              f"{g['won'].mean():>9.4f}"
              f"{(g['p_ship'].mean() - g['won'].mean()) * 100:>10.2f}"
              f"{(g['p_fit'].mean() - g['won'].mean()) * 100:>9.2f}")
    print("\n  gap ship is today's layer, gap fit is the refitted sigma. If")
    print("  gap fit is closer to zero at the extreme bands, the deviation")
    print("  term is doing real work rather than moving an average.\n")


def verdict(C, lo, excl, Cw, lo_w, excl_w):
    print("=" * 96)
    print("PRE-REGISTERED VERDICT")
    print("=" * 96)
    print(f"  C pooled, interval excludes zero        "
          f"{'YES' if excl else 'no':<4}  (C {C:+.4f}, lo {lo:+.4f})")
    print(f"  C WITHIN line, interval excludes zero   "
          f"{'YES' if excl_w else 'no':<4}  (C {Cw:+.4f}, lo {lo_w:+.4f})")
    print()
    if excl and excl_w:
        print("  ==> ADD THE TERM. Sigma gains a deviation term and the level")
        print("      constant moves in the same pass. Re-run edge_threshold")
        print("      before believing any threshold computed under the old")
        print("      sigma, including the candidate rule's +0.1475.")
    elif not excl_w:
        print("  ==> LEVEL ONLY. Sigma is too wide but does not depend on")
        print("      |dev| once the line is held fixed. Do NOT ship a")
        print("      deviation term. The handoff's over-dispersion finding")
        print("      now needs its own explanation, and nothing changes")
        print("      until it has one.")
    else:
        print("  ==> INCOHERENT. The within-line estimate survives but the")
        print("      pooled one does not, which should not happen. Treat as")
        print("      a NULL and look for a bug before acting.")
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
                    choices=["consensus", "fanduel"],
                    help="consensus for power (10,022 rows), fanduel as the "
                         "secondary cut")
    ap.add_argument("--sigma-anchor", default="mean",
                    choices=["mean", "proj"])
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(EH.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 96)
    print(f"SIGMA BY LEVEL AND DEVIATION   market={args.market}   "
          f"seasons={seasons}")
    print(f"line source={args.source}   anchor={args.sigma_anchor}   "
          f"bootstrap={args.boot}")
    print("=" * 96)
    print()

    d = build_frame(seasons, args.cache, args.refresh, args.market,
                    args.score_population, args.source)
    d = add_blend_lso(d)
    if args.save_rows:
        d.to_csv(args.save_rows, index=False)
        print(f"  wrote {len(d)} rows to {args.save_rows}\n")

    table_level(d, args.min_n, args.sigma_anchor)
    table_dev(d, args.min_n, args.sigma_anchor, args.boot)
    table_within(d, args.min_n)
    Cw, lo_w, hi_w, excl_w = fit_within(d, args.boot)
    A, B, C, lo, hi, excl = fit_joint(d, args.boot)
    table_bands(d, A, B, C, args.sigma_anchor)
    verdict(C, lo, excl, Cw, lo_w, excl_w)


if __name__ == "__main__":
    main()
