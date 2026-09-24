"""
alpha_by_line.py
================

Measures receptions alpha, over rate, pricing gap and dispersion at EXACT line
values rather than pooled, and reports the alpha that would make the priced
P(over) match the measured over rate at each line.

WHY THIS EXISTS
---------------
The handoff establishes receptions alpha at +0.12 pooled, and Phase 4.1 calls
for choosing alpha so the priced P(over) at zero deviation matches the measured
over rate. Both are POOLED numbers. On 2026-09-23 a hand read of FanDuel's alt
reception ladders for ATL at GB implied a mean-minus-line of +0.52 for Kyle
Pitts and +0.61 for Tucker Kraft, both on a 3.5 line, against that pooled
+0.12. Twelve players on one game, so it is a lead and not a result.

It matters because the candidate rule trades the 2.5 and 3.5 lines and its
profitable tail is 82 percent UNDERS. A distribution mean sitting half a
reception below the truth in exactly that band manufactures under edges.

PRE-REGISTERED OUTCOMES (written before the first run)
------------------------------------------------------
  CONFIRM   alpha at 2.5 and 3.5 is >= +0.40, clustered t > 3, and the sign is
            consistent across all four seasons. The under tail must then be
            re-tested before any paper trading.
  REFUTE    alpha at 2.5 and 3.5 is within about 0.10 of the pooled +0.12. The
            two-player read was noise and the hypothesis is dropped.
  NULL      anything between, or a sign that flips across seasons. Treated the
            way the lead-time re-pull was treated: gated, not acted on.

Note that a positive alpha is expected BY CONSTRUCTION and is not a signal.
The line sits near the median, the mean of a right-skewed count sits above it.
The deliverable is therefore not alpha itself but ALPHA_TO_MATCH, the value
that aligns priced P(over) with the measured over rate at that line. That is
the Phase 4.1 parameter, stratified.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not re-implement anything in eval_harness. It imports load_lines,
collapse_books, score_market and cluster_mean and uses them as they stand.
On 2026-09-23 db.get_line_movement was found to be 8x wrong because it read
the `lines` table without sharing the harness book collapse. Nothing new
should read that table on its own.

It stratifies on line_fanduel, not line_consensus. A median across 8 to 11
books lands on values no book posted; settling against that produced the
+0.2991 artifact recorded as item 11 in NEW_DIRECTIONS.

USAGE
-----
  python alpha_by_line.py --all-seasons --cache lines_cache.parquet
  python alpha_by_line.py --all-seasons --cache lines_cache.parquet --by-season
  python alpha_by_line.py --seasons 2023,2024,2025 --cache lines_cache.parquet \
      --zero-dev 0.25 --min-n 150
  python alpha_by_line.py --all-seasons --cache lines_cache.parquet --audit-only
"""

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import nbinom, poisson, chi2 as chi2_dist
    from scipy.optimize import brentq
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

try:
    import eval_harness as EH
except ImportError as e:
    sys.exit(f"could not import eval_harness: {e}\n"
             "run this from the repo root, alongside eval_harness.py")

# eval_harness imports data_utils inside main(), so it is not on the module
# namespace. data_utils.py lives in models/, but try the repo root too in case
# it ever moves (plan 6.2 proposes consolidating helpers into it).
data_utils = None
_du_err = []
for _path in ("models.data_utils", "data_utils"):
    try:
        import importlib
        data_utils = importlib.import_module(_path)
        _du_path = _path
        break
    except ImportError as e:
        _du_err.append(f"{_path}: {e}")
if data_utils is None:
    sys.exit("could not import data_utils from any known location:\n  "
             + "\n  ".join(_du_err))

if not hasattr(data_utils, "norm_join_name"):
    sys.exit(f"{_du_path} has no `norm_join_name`. The harness joins on it, "
             f"so this script must use the same function or the join rules "
             f"drift.")

for _name in ("load_lines", "collapse_books", "score_market", "cluster_mean",
              "ols_clustered", "connect", "_secrets", "MARKET_SPEC",
              "ALL_SEASONS"):
    if not hasattr(EH, _name):
        sys.exit(f"eval_harness has no `{_name}`. This script was written "
                 f"against the 1,112 line rewrite of 2026-09-22. Check the "
                 f"harness API before trusting anything below.")

# --------------------------------------------------------------------------
# Shipped receptions pricing parameters, from the handoff. Kept here as
# literals rather than imported so a change in the app cannot silently change
# a measurement. If these drift from mc_pricing, the drift is the bug.
# --------------------------------------------------------------------------
SIGMA_A = 1.143
SIGMA_B = 0.2992
SIGMA_FLOOR = 1.7
POOLED_ALPHA = 0.12          # handoff value, used as the CURRENT pricing alpha

# Sanity targets. If the pooled run does not reproduce these, the script is
# wrong, not the handoff.
EXPECT_ALPHA = 0.12
EXPECT_OVER_RATE = 0.4799


# ==========================================================================
# pricing
# ==========================================================================

def sigma_shipped(mean):
    """Shipped receptions sigma. Level dependent, linear in the anchor."""
    return np.maximum(SIGMA_A + SIGMA_B * np.asarray(mean, float), SIGMA_FLOOR)


def p_over_count(line, mean, sigma):
    """
    P(X > line) for a count, moment matched to (mean, sigma).

    Negative binomial where overdispersed, Poisson where not, matching
    mc_pricing's documented behaviour. A half integer line means no push, so
    P(over) = P(X >= floor(line) + 1).
    """
    line = np.asarray(line, float)
    mean = np.asarray(mean, float)
    sigma = np.asarray(sigma, float)
    k = np.floor(line) + 1.0                    # 3.5 -> 4 or more
    mean = np.clip(mean, 1e-6, None)
    var = sigma ** 2

    out = np.empty_like(mean, dtype=float)
    od = var > mean * 1.0000001
    # overdispersed -> negative binomial
    if np.any(od):
        m, v = mean[od], var[od]
        r = m * m / (v - m)
        p = r / (r + m)
        out[od] = nbinom.sf(k[od] - 1, r, p)
    # underdispersed or equal -> Poisson
    if np.any(~od):
        out[~od] = poisson.sf(k[~od] - 1, mean[~od])
    return out


def alpha_to_match(line, target_rate, anchor="mean", lo=-2.0, hi=2.0):
    """
    Solve for the alpha that makes the priced P(over) at zero deviation equal
    the measured over rate, at a single line value.

    anchor="mean"  sigma evaluated at (line + alpha), the Phase 4.2 shape
    anchor="proj"  sigma evaluated at the line itself, i.e. what ships today
    """
    def f(a):
        mu = line + a
        s = sigma_shipped(mu if anchor == "mean" else line)
        return float(p_over_count(np.array([line]), np.array([mu]),
                                  np.array([s]))[0]) - target_rate

    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-6)
    except Exception:
        return np.nan


# ==========================================================================
# clustering
# ==========================================================================

def cluster_col(df):
    """
    Game cluster for standard errors. event_id where the harness carries it,
    otherwise season-week, which is coarser and gives WIDER SEs. Never fall
    back to treating rows as independent: reception props within a game share
    a quarterback.
    """
    if "event_id" in df.columns and df["event_id"].notna().any():
        return df["event_id"].astype(str).to_numpy(), "event_id"
    return ((df["season"].astype(str) + "_" + df["week"].astype(str))
            .to_numpy()), "season_week (event_id absent, SEs conservative)"


def clustered_mean(y, cl):
    """Thin wrapper so a harness signature change surfaces here."""
    res = EH.cluster_mean(np.asarray(y, float), np.asarray(cl))
    if not res:
        return np.nan, np.nan
    return float(res["mean"]), float(res["se"])


# ==========================================================================
# step 0: the null audit that has to pass first
# ==========================================================================

def audit_nulls(seasons, market="receptions"):
    """
    If a posted line joins to a row whose reception count is null and that row
    is dropped rather than filled with zero, alpha is biased UPWARD, which is
    the exact direction that would make the hypothesis under test look true.

    The handoff says nflverse appears to store 0.0 for active WR/TE/RB rows, so
    the expected impact is small. Expected is not measured.
    """
    print("=" * 96)
    print("STEP 0. NULL AUDIT. This must pass before any number below means "
          "anything.")
    print("=" * 96)

    mod_name, actual_col = EH.MARKET_SPEC[market]
    import importlib
    mod = importlib.import_module(f"models.{mod_name}")

    builder = getattr(mod, "build_all_rows", None)
    which = "build_all_rows (unfiltered)"
    if builder is None:
        builder = mod.build_dataset
        which = "build_dataset  <-- NO build_all_rows SIBLING, see plan 3.9"
    df = builder()

    n = len(df)
    nulls = int(df[actual_col].isna().sum())
    zeros = int((df[actual_col] == 0).sum())
    print(f"  builder            {which}")
    print(f"  rows               {n}")
    print(f"  null {actual_col:<14}{nulls}  ({100.0 * nulls / max(n, 1):.2f}%)")
    print(f"  zero {actual_col:<14}{zeros}  ({100.0 * zeros / max(n, 1):.2f}%)")

    if nulls == 0:
        print("\n  PASS. No nulls, so nothing is being dropped and alpha is "
              "not biased by\n  the grading path. Proceed.")
    else:
        print(f"\n  *** FAIL. {nulls} null rows will be dropped by the dropna "
              f"in score_market.\n"
              f"  *** If those are active players with zero catches, alpha is "
              f"biased UP and\n"
              f"  *** every alpha below is overstated. Fix models.{mod_name} "
              f"first (Phase 3.4,\n"
              f"  *** same fix as rushing and anytime_td) and re-run. Do not "
              f"read the tables.")
    print()
    return nulls == 0


# ==========================================================================
# frame construction
# ==========================================================================

def build_frame(seasons, cache, refresh, market="receptions",
                population="all"):
    sec = EH._secrets()
    box = {}

    def client_factory():
        if "c" not in box:
            box["c"] = EH.connect(sec)
        return box["c"]

    print(f"  loading lines for {market}, seasons {seasons} ...")
    lines = EH.load_lines(client_factory, seasons, (market,), cache, refresh)
    if len(lines) == 0:
        sys.exit("no line rows returned")

    nullw = int(lines["week"].isna().sum())
    if nullw:
        print(f"  dropping {nullw} rows with a null week")
        lines = lines[lines["week"].notna()]

    props = EH.collapse_books(lines)
    print(f"  {len(lines)} line rows -> {len(props)} player-weeks "
          f"after the book collapse")

    print(f"  scoring {market} walk-forward, population={population} ...")
    proj = EH.score_market(market, seasons, mode="walk_forward",
                           population=population)
    if proj is None or len(proj) == 0:
        sys.exit(f"score_market returned nothing for {market}")

    # actual column: the harness renames it, but detect rather than assume
    acol = "actual" if "actual" in proj.columns else EH.MARKET_SPEC[market][1]
    if acol not in proj.columns:
        sys.exit(f"cannot find the actual column in score_market output; "
                 f"saw {list(proj.columns)}")
    if acol != "actual":
        proj = proj.rename(columns={acol: "actual"})

    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(
        proj["player"] if "player" in proj.columns
        else proj["player_display_name"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)

    keep = [c for c in ("season", "week", "market", "_key", "projection",
                        "actual", "n_train", "train_seasons", "event_id")
            if c in proj.columns]
    joined = props.merge(proj[keep], on=["season", "week", "market", "_key"],
                         how="inner")
    print(f"  joined {len(joined)} rows "
          f"({100.0 * len(joined) / max(len(props), 1):.1f}% of props)")

    for c in ("line_fanduel", "line_consensus", "projection", "actual"):
        if c in joined.columns:
            joined[c] = pd.to_numeric(joined[c], errors="coerce")

    joined = joined.dropna(subset=["line_fanduel", "actual", "projection"])
    joined["dev_fd"] = joined["projection"] - joined["line_fanduel"]
    print(f"  {len(joined)} rows with a FanDuel line, an actual and a "
          f"projection\n")
    return joined


# ==========================================================================
# tables
# ==========================================================================

def pooled_sanity(df):
    """Reproduce the handoff before stratifying. If this fails, stop."""
    print("=" * 96)
    print("STEP 1. POOLED SANITY. Must reproduce the handoff.")
    print("=" * 96)
    cl, how = cluster_col(df)
    print(f"  clustering on {how}")

    a, a_se = clustered_mean(df["actual"] - df["line_fanduel"], cl)
    live = np.asarray(~np.isclose(df["actual"], df["line_fanduel"]), bool)
    over = (df["actual"] > df["line_fanduel"]).astype(float).to_numpy()[live]
    r, r_se = clustered_mean(over, cl[live])

    print(f"  {'':<22}{'measured':>12}{'SE':>9}{'handoff':>11}{'':>4}")
    da = abs(a - EXPECT_ALPHA)
    dr = abs(r - EXPECT_OVER_RATE)
    print(f"  {'alpha (FanDuel line)':<22}{a:>12.4f}{a_se:>9.4f}"
          f"{EXPECT_ALPHA:>11.4f}   {'ok' if da < 0.05 else 'MISMATCH'}")
    print(f"  {'over rate':<22}{r:>12.4f}{r_se:>9.4f}"
          f"{EXPECT_OVER_RATE:>11.4f}   {'ok' if dr < 0.02 else 'MISMATCH'}")
    print(f"  {'n rows':<22}{len(df):>12}")
    print(f"  {'n clusters':<22}{len(set(cl)):>12}")
    if da >= 0.05 or dr >= 0.02:
        print("\n  *** A mismatch here means this script disagrees with the "
              "instrument that\n"
              "  *** produced every number in the handoff. Resolve that before "
              "reading on.\n"
              "  *** Note the handoff figures are on CONSENSUS lines; a small "
              "gap is expected\n"
              "  *** because this stratifies on FanDuel. A large one is a bug.")
    print()


def by_line(df, zero_dev, min_n, anchor):
    print("=" * 96)
    print("STEP 2. BY EXACT FANDUEL LINE VALUE")
    print("=" * 96)
    print("  alpha        mean(actual - line), clustered SE, not a signal")
    print("  over         counted P(actual > line), pushes excluded")
    print("  priced       P(over) at zero deviation with the CURRENT pooled")
    print(f"               alpha of {POOLED_ALPHA:+.2f} and shipped sigma")
    print("  gap          priced minus measured, in points. Negative means "
          "the app")
    print("               UNDER-prices the over, which manufactures under "
          "edges.")
    print("  a_match      the alpha that closes the gap. THE DELIVERABLE.")
    print()

    for label, sub in (("ALL ROWS", df),
                       (f"ZERO DEVIATION  |proj - line| < {zero_dev}",
                        df[df["dev_fd"].abs() < zero_dev])):
        print(f"  --- {label} ---")
        print(f"  {'line':>5}{'n':>7}{'clus':>6}{'alpha':>8}{'SE':>7}{'t':>7}"
              f"{'over':>8}{'SE':>7}{'priced':>8}{'gap pp':>8}{'z':>7}"
              f"{'a_match':>9}")
        rows = []
        for L, g in sub.groupby("line_fanduel", sort=True):
            if len(g) < min_n:
                continue
            cl, _ = cluster_col(g)
            a, a_se = clustered_mean(g["actual"] - L, cl)
            t = a / a_se if a_se and a_se > 0 else np.nan

            live = np.asarray(~np.isclose(g["actual"], L), bool)
            if live.sum() < 30:
                continue
            ov = (g["actual"] > L).astype(float).to_numpy()[live]
            r, r_se = clustered_mean(ov, cl[live])

            mu = L + POOLED_ALPHA
            s = sigma_shipped(mu if anchor == "mean" else L)
            pr = float(p_over_count(np.array([L]), np.array([mu]),
                                    np.array([s]))[0])
            gap = (pr - r) * 100.0
            z = (pr - r) / r_se if r_se and r_se > 0 else np.nan
            am = alpha_to_match(L, r, anchor=anchor)

            print(f"  {L:>5.1f}{len(g):>7}{len(set(cl)):>6}{a:>8.3f}"
                  f"{a_se:>7.3f}{t:>7.2f}{r:>8.4f}{r_se:>7.4f}{pr:>8.4f}"
                  f"{gap:>8.2f}{z:>7.2f}{am:>9.3f}")
            rows.append(dict(line=L, n=len(g), alpha=a, se=a_se, t=t,
                             over=r, priced=pr, gap=gap, a_match=am))
        print()
        if label == "ALL ROWS":
            verdict(rows)
    return


def verdict(rows):
    """Score the pre-registered outcomes. No eyeballing."""
    band = [r for r in rows if r["line"] in (2.5, 3.5)]
    print("  --- PRE-REGISTERED VERDICT, 2.5 and 3.5 band ---")
    if not band:
        print("  no rows at 2.5 or 3.5 above the n floor. Inconclusive; "
              "receptions lines\n"
              "  cluster hard at 1.5, which is the qcut lesson from "
              "NEW_DIRECTIONS.")
        print()
        return
    for r in band:
        if r["alpha"] >= 0.40 and r["t"] > 3:
            v = "CONFIRM direction"
        elif abs(r["alpha"] - POOLED_ALPHA) <= 0.10:
            v = "REFUTE"
        else:
            v = "NULL, do not act"
        print(f"  line {r['line']:.1f}   alpha {r['alpha']:+.3f} "
              f"(t {r['t']:+.2f})   a_match {r['a_match']:+.3f}   -> {v}")
    print("  Season consistency is still required for a CONFIRM. Run "
          "--by-season.")
    print()


def by_line_season(df, min_n):
    """
    Season stability. A fixed spread threshold does NOT work here: at 300 rows
    per season-line the SE is about 0.14, so a spread of 0.30 is two standard
    errors and arises by chance. Verified on synthetic data with a flat truth,
    where a fixed 0.30 rule flagged two of four lines as unstable.

    So this runs a chi-square test of homogeneity across seasons instead,
    sum((a_i - abar)^2 / se_i^2) on k-1 degrees of freedom, where abar is the
    inverse-variance weighted mean. A small p means the season estimates
    genuinely disagree. A large p means one parameter explains all of them.
    """
    print("=" * 96)
    print("STEP 3. ALPHA BY LINE AND SEASON")
    print("=" * 96)
    print("  Homogeneity across seasons, not a fixed spread. p below 0.05 "
          "means the")
    print("  seasons genuinely disagree and the pooled value is not a "
          "parameter.")
    print()
    seasons = sorted(df["season"].unique())
    print(f"  {'line':>5}" + "".join(f"{int(s):>16}" for s in seasons)
          + f"{'chi2':>8}{'p':>8}{'verdict':>12}")
    for L, g in df.groupby("line_fanduel", sort=True):
        if len(g) < min_n:
            continue
        vals, ses, cells = [], [], ""
        for s in seasons:
            h = g[g["season"] == s]
            if len(h) < 40:
                cells += f"{'.':>16}"
                continue
            cl, _ = cluster_col(h)
            a, se = clustered_mean(h["actual"] - L, cl)
            cells += f"{a:+.3f}+-{se:.3f}".rjust(16)
            if se and se > 0:
                vals.append(a)
                ses.append(se)
        if len(vals) < 2:
            print(f"  {L:>5.1f}{cells}{'.':>8}{'.':>8}{'thin':>12}")
            continue
        v = np.array(vals)
        w = 1.0 / np.array(ses) ** 2
        abar = float((w * v).sum() / w.sum())
        chi2 = float((w * (v - abar) ** 2).sum())
        dof = len(v) - 1
        p = float(chi2_dist.sf(chi2, dof))
        verdict = "stable" if p >= 0.05 else "DISAGREES"
        print(f"  {L:>5.1f}{cells}{chi2:>8.2f}{p:>8.3f}{verdict:>12}")
    print("\n  Weighted mean is the right pooled value where the test says "
          "stable.\n")


def dispersion(df, min_n):
    """
    Tests the 1.143 sigma intercept. On 2026-09-23 FanDuel's alt ladders
    implied var/mean near 1.38 across the board, while the shipped formula
    implies 1.69 at a mean of 1.5, so the model looks about twenty percent
    too dispersed at the bottom of the board.
    """
    print("=" * 96)
    print("STEP 4. DISPERSION BY LINE. Tests the sigma intercept of "
          f"{SIGMA_A}.")
    print("=" * 96)
    breakeven = (SIGMA_FLOOR - SIGMA_A) / SIGMA_B
    print(f"  The floor of {SIGMA_FLOOR} BINDS below a mean of "
          f"{breakeven:.2f}, so for any line where")
    print(f"  mean(actual) < {breakeven:.2f} the level dependent formula is "
          f"not in use at all and")
    print(f"  sigma is flat at {SIGMA_FLOOR}. Reception lines cluster at 1.5, "
          f"so check how much")
    print(f"  of the board this covers before treating sigma as level "
          f"dependent here.")
    print()
    print(f"  {'line':>5}{'n':>7}{'mean':>8}{'var':>8}{'var/mean':>10}"
          f"{'sd emp':>9}{'sd ship':>9}{'ratio':>8}{'floor?':>8}")
    n_floored = 0
    for L, g in df.groupby("line_fanduel", sort=True):
        if len(g) < min_n:
            continue
        a = g["actual"].to_numpy(float)
        m, v = a.mean(), a.var(ddof=1)
        ship = float(sigma_shipped(m))
        floored = m < breakeven
        n_floored += len(g) if floored else 0
        print(f"  {L:>5.1f}{len(g):>7}{m:>8.2f}{v:>8.2f}{v / max(m, 1e-9):>10.2f}"
              f"{np.sqrt(v):>9.2f}{ship:>9.2f}{np.sqrt(v) / ship:>8.2f}"
              f"{('FLOOR' if floored else ''):>8}")
    print(f"\n  {n_floored} of {len(df)} rows "
          f"({100.0 * n_floored / max(len(df), 1):.1f}%) sit where the floor "
          f"binds.")
    print("\n  ratio below 1.00 means the shipped sigma is too WIDE at that "
          "line, which\n"
          "  puts too much mass at zero catches. That is the same defect "
          "ladder_calibration\n"
          "  found at k = 1 (+1.1 pp, z +3.62) from the other direction, and "
          "the same one\n"
          "  the FanDuel ladders implied on 2026-09-23 (book var/mean 1.38 "
          "against a\n"
          "  shipped 1.93 at a mean of 1.5, where the floor is what is "
          "binding).\n")


def fd_vs_consensus(df, min_n):
    """
    Bears on the twelve-bet concentration problem. The candidate rule's
    holdout falls from +0.1475 to +0.0742 when restricted to props where
    FanDuel matches consensus, and FanDuel differs on about 7 percent of them.
    """
    if "line_consensus" not in df.columns:
        print("  line_consensus absent, skipping step 5\n")
        return
    print("=" * 96)
    print("STEP 5. FANDUEL AGAINST CONSENSUS, BY LINE")
    print("=" * 96)
    d = df.dropna(subset=["line_consensus"]).copy()
    d["differs"] = ~np.isclose(d["line_fanduel"], d["line_consensus"])
    print(f"  {'line':>5}{'n':>7}{'differs':>9}{'mean gap':>10}"
          f"{'alpha same':>12}{'alpha diff':>12}")
    for L, g in d.groupby("line_fanduel", sort=True):
        if len(g) < min_n:
            continue
        share = g["differs"].mean()
        gap = (g["line_fanduel"] - g["line_consensus"]).mean()
        out = []
        for m in (False, True):
            h = g[g["differs"] == m]
            if len(h) < 30:
                out.append(np.nan)
                continue
            cl, _ = cluster_col(h)
            out.append(clustered_mean(h["actual"] - L, cl)[0])
        print(f"  {L:>5.1f}{len(g):>7}{share * 100:>8.1f}%{gap:>10.3f}"
              f"{out[0]:>12.3f}{out[1]:>12.3f}")
    print("\n  A large alpha difference between the same and differs columns "
          "means the\n"
          "  candidate rule's profit is coming from FanDuel being off "
          "consensus, which is\n"
          "  the same adverse selection that made line shopping negative "
          "(item 10).\n")


# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None, help="e.g. 2023,2024,2025")
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--market", default="receptions")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--zero-dev", type=float, default=0.25)
    ap.add_argument("--min-n", type=int, default=100,
                    help="minimum rows for a line value to be printed")
    ap.add_argument("--sigma-anchor", default="mean",
                    choices=["mean", "proj"],
                    help="mean = sigma at (line + alpha), Phase 4.2 shape; "
                         "proj = sigma at the line, what ships today")
    ap.add_argument("--by-season", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--save-rows", default=None)
    ap.add_argument("--skip-audit", action="store_true",
                    help="only after the audit has passed once")
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(EH.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 96)
    print(f"ALPHA BY LINE VALUE   market={args.market}   seasons={seasons}")
    print(f"sigma = max({SIGMA_A} + {SIGMA_B}*anchor, {SIGMA_FLOOR})   "
          f"anchor={args.sigma_anchor}   current pooled alpha={POOLED_ALPHA}")
    print("=" * 96)
    print()

    ok = True
    if not args.skip_audit:
        ok = audit_nulls(seasons, args.market)
    if args.audit_only:
        return
    if not ok:
        print("Audit failed. Stopping rather than printing numbers that are "
              "biased in the\n"
              "direction of the hypothesis under test. Pass --skip-audit to "
              "override, but\n"
              "do not believe the output if you do.")
        return

    df = build_frame(seasons, args.cache, args.refresh, args.market,
                     args.score_population)
    if args.save_rows:
        df.to_csv(args.save_rows, index=False)
        print(f"  wrote {len(df)} rows to {args.save_rows}\n")

    pooled_sanity(df)
    by_line(df, args.zero_dev, args.min_n, args.sigma_anchor)
    if args.by_season:
        by_line_season(df, args.min_n)
    dispersion(df, args.min_n)
    fd_vs_consensus(df, args.min_n)

    print("=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  The deliverable is a_match per line value, not alpha. Alpha is")
    print("  positive by construction on a right-skewed count and carries no")
    print("  betting information; the handoff closed that question.")
    print()
    print("  If a_match at 2.5 and 3.5 is materially above the shipped "
          f"{POOLED_ALPHA:+.2f},")
    print("  the app's reception distribution sits low in the band the")
    print("  candidate rule trades, and its 82 percent UNDER tail is at least")
    print("  partly a pricing artifact rather than an edge.")
    print()
    print("  A CONFIRM needs BOTH the pooled verdict and season stability.")
    print("  One without the other is a NULL under the gate-before-sweeping")
    print("  rule, which has already killed the alpha edge, the best-line")
    print("  beta and the lead-time re-pull.")


if __name__ == "__main__":
    main()
