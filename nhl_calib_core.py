"""nhl_calib_core.py - the estimator and null for the NHL calibration test.

SEPARATE FROM THE LOADER ON PURPOSE. Everything here is pure numpy and can
be executed against planted data, which is the only way to know the
estimator recovers a slope it is given. The handover's rule is that a
permutation null must be checked against planted data before it is trusted
on real data, because a null scheme can be degenerate: a within-cell shuffle
once gave sd 0.0000 and p 1.0000 against a true rho of -0.88.

PRE-REGISTERED IN PREREG_nhl_calibration.md. Nothing here chooses a
threshold, a bin edge or a direction.
"""
import numpy as np

# Fixed in the pre-registration. Not to be tuned here.
BIN_EDGES = (0.00, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 1.00)
HOLD_PILOT = 1.0651


def devig(over_odds, under_odds):
    """American odds to a two-sided devigged P(over), and the raw sum.

    The raw sum IS the hold and is returned because the pilot measured it at
    1.0651 with a range of only 1.0592 to 1.0718. That near-constancy is what
    makes devigging well defined here rather than approximate, so it is
    checked on the full sample rather than assumed.
    """
    o = np.asarray(over_odds, dtype=float)
    u = np.asarray(under_odds, dtype=float)

    def implied(a):
        # np.where evaluates BOTH branches, so odds of exactly -100 (even
        # money) make the positive branch divide by zero even though that
        # value is then discarded. Suppressed rather than left to print a
        # warning that looks like a data problem and is not.
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))

    io, iu = implied(o), implied(u)
    s = io + iu
    with np.errstate(invalid="ignore", divide="ignore"):
        p = io / s
    return p, s


def ols_clustered(x, y, cluster):
    """Slope and intercept of y on x with cluster-robust (CR0) errors.

    Returns dict with b, a, se_b, se_a, t_b (against H0: b = 1), n, n_clusters.

    The t statistic is against ONE, not zero. A calibration slope of zero
    would mean the price carries no information at all; the pre-registered
    hypothesis is that the price is informative but overconfident, so the
    null worth testing is b = 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cluster = np.asarray(cluster)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cluster = x[ok], y[ok], cluster[ok]
    n = x.size
    if n < 10 or np.std(x) == 0:
        return {"b": np.nan, "a": np.nan, "se_b": np.nan, "se_a": np.nan,
                "t_b": np.nan, "n": n, "n_clusters": 0}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    meat = np.zeros((2, 2))
    # Sum over clusters of (X_g' u_g)(X_g' u_g)', which is what a Bonferroni
    # cannot substitute for: props inside one game share a game script.
    order = np.argsort(cluster, kind="stable")
    cs, xs, rs = cluster[order], X[order], resid[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], cs.size]
    for s, e in zip(starts, ends):
        sg = xs[s:e].T @ rs[s:e]
        meat += np.outer(sg, sg)
    # CR1 small-cluster correction. Measured on planted data at 400 clusters,
    # uncorrected CR0 gave 0.930 interval coverage against a nominal 0.95,
    # which is anti-conservative in exactly the direction that would turn a
    # null into a finding. This is the standard Stata-style adjustment.
    G = int(starts.size)
    k = X.shape[1]
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - k)) if G > 1 else 1.0
    V = c * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))
    return {"b": float(beta[1]), "a": float(beta[0]),
            "se_b": float(se[1]), "se_a": float(se[0]),
            "t_b": float((beta[1] - 1.0) / se[1]) if se[1] > 0 else np.nan,
            "n": int(n), "n_clusters": int(starts.size)}


def intra_cluster_corr(y, cluster):
    """One-way ANOVA estimate of the intra-game correlation of outcomes.

    Reported because every MDE in the pre-registration is a function of it,
    and the pre-registered table assumed 0.05 without knowing.
    """
    y = np.asarray(y, dtype=float)
    cluster = np.asarray(cluster)
    ok = np.isfinite(y)
    y, cluster = y[ok], cluster[ok]
    _, inv, cnt = np.unique(cluster, return_inverse=True, return_counts=True)
    k, n = cnt.size, y.size
    if k < 2 or n <= k:
        return np.nan
    gmean = np.bincount(inv, weights=y) / cnt
    grand = y.mean()
    ssb = float(np.sum(cnt * (gmean - grand) ** 2))
    ssw = float(np.sum((y - gmean[inv]) ** 2))
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - np.sum(cnt ** 2) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan


def mde(se_b, power=0.80, alpha=0.05):
    """Minimum detectable deviation from b = 1, two-sided."""
    from math import sqrt
    z_a = 1.959963985
    z_b = {0.80: 0.841621234, 0.90: 1.281551566}.get(power, 0.841621234)
    return float((z_a + z_b) * se_b)


def bin_table(p, y, edges=BIN_EDGES):
    """Realised rate against mean price, per pre-specified bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) & np.isfinite(p) & np.isfinite(y)
        k = int(m.sum())
        out.append({
            "lo": lo, "hi": hi, "n": k,
            "mean_p": float(p[m].mean()) if k else np.nan,
            "realised": float(y[m].mean()) if k else np.nan,
            "miss": float(y[m].mean() - p[m].mean()) if k else np.nan,
        })
    return out


def max_bin_null(p, cluster, draws=2000, edges=BIN_EDGES, seed=20261009):
    """Max-statistic null for the EXPLORATORY per-bin test.

    Under H0 the price is calibrated, so y_i ~ Bernoulli(p_i). Simulating
    from that and taking the max absolute bin miss gives the bar a single
    bin must clear, which is the only honest way to read a best-of-seven.

    CHECKED AGAINST PLANTED DATA, because a degenerate null is the specific
    failure this project has already had. See the self-test at the bottom.

    LIMITATION, STATED: this draws independently within a game and so
    understates variance where outcomes are correlated inside a game. The
    measured intra-game correlation is printed beside it so the result can
    be discounted accordingly. The per-bin test is exploratory regardless.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    p = p[ok]
    maxima = np.empty(draws)
    for d in range(draws):
        ysim = (rng.random(p.size) < p).astype(float)
        misses = [abs(b["miss"]) for b in bin_table(p, ysim, edges)
                  if b["n"] > 0 and np.isfinite(b["miss"])]
        maxima[d] = max(misses) if misses else np.nan
    return maxima


def profitable_side(p, hold):
    """Which side, if any, a realised rate would have to beat.

    UNDER needs realised < hold*p - (hold-1); OVER needs realised > hold*p.
    Returned so the write-up cannot restate a significant slope as an edge
    without the arithmetic beside it.
    """
    v = hold - 1.0
    return hold * p - v, hold * p
