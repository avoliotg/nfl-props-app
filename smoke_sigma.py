"""
Synthetic smoke test for sigma_by_dev.

Runs TWICE:
  C_TRUE = 0.25  the deviation effect exists      -> must say ADD THE TERM
  C_TRUE = 0.00  no deviation effect at all       -> must say LEVEL ONLY

The second run is the one that matters. A tool that reports a deviation term
on data with no deviation effect is worse than no tool, and this project has
already been burned four times by instruments that found signal in nulls.

In both runs the shipped sigma is deliberately too WIDE by about 12 percent,
matching what alpha_by_line found, so the level ratio should land near 0.88.
"""
import sys
import types
import importlib
import numpy as np
import pandas as pd

C_TRUE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
SEASONS = (2023, 2024, 2025, 2026)
TRUE_BETA, TRUE_ALPHA = 0.28, 0.12
rng = np.random.default_rng(11)

rows = []
for L, n in {1.5: 1200, 2.5: 2400, 3.5: 1900, 4.5: 1300, 5.5: 600}.items():
    for i in range(n):
        s = SEASONS[i % 4]
        dev = rng.normal(0, 0.9)
        # true mean follows the blend the harness would fit
        mu = max(L + TRUE_ALPHA + TRUE_BETA * dev, 0.15)
        # true sd: level dependent, times the planted deviation factor,
        # scaled to sit about 12% BELOW the shipped formula
        ship = max(1.143 + 0.2992 * mu, 1.7)
        sd = 0.88 * ship * (1.0 + C_TRUE * abs(dev))
        var = sd ** 2
        if var > mu * 1.0001:
            r = mu * mu / (var - mu)
            y = rng.negative_binomial(r, r / (r + mu))
        else:
            y = rng.poisson(mu)
        rows.append(dict(
            season=s, week=1 + (i % 17), market="receptions",
            player=f"P{L}_{i}", event_id=f"{s}_{1 + (i % 17)}_{i % 16}",
            line_consensus=L, line_fanduel=L,
            actual=float(y), projection=L + dev))
df = pd.DataFrame(rows)


def _ols_clustered(x, y, cluster):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(cluster)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    order = np.argsort(c)
    Xs, rs, cs = X[order], resid[order], c[order]
    starts = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1, len(cs)]
    g = len(starts) - 1
    for i in range(g):
        sl = slice(starts[i], starts[i + 1])
        u = Xs[sl].T @ rs[sl]
        meat += np.outer(u, u)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - 2, 1))
    cov = xtx_inv @ meat @ xtx_inv * scale
    se = np.sqrt(np.diag(cov))
    return {"alpha": float(coef[0]), "beta": float(coef[1]),
            "se_beta": float(se[1])}


EH = types.ModuleType("eval_harness")
EH.MARKET_SPEC = {"receptions": ("receptions", "receptions")}
EH.ALL_SEASONS = SEASONS
EH.ols_clustered = _ols_clustered
EH.cluster_mean = lambda y, c: {"mean": float(np.mean(y)), "se": 0.01}
EH._secrets = lambda: {}
EH.connect = lambda sec: object()
EH.load_lines = lambda cf, s, m, c, r: df.assign(book="fanduel",
                                                 line=df["line_consensus"])
EH.collapse_books = lambda lines: df[[
    "season", "week", "market", "player", "event_id",
    "line_consensus", "line_fanduel"]].copy()
EH.score_market = lambda market, seasons, mode=None, fixed_train=None, \
    population=None: df[["season", "week", "market", "player",
                         "projection", "actual", "event_id"]].copy()
sys.modules["eval_harness"] = EH

models = types.ModuleType("models")
models.__path__ = []
du = types.ModuleType("models.data_utils")
du.norm_join_name = lambda s: pd.Series(s).astype(str).str.lower()
sys.modules["models"] = models
sys.modules["models.data_utils"] = du

print("#" * 96)
print(f"PLANTED:  C_TRUE = {C_TRUE:.2f}   beta {TRUE_BETA}   alpha "
      f"{TRUE_ALPHA}   sigma 12% below shipped")
print(f"EXPECTED VERDICT: "
      f"{'ADD THE TERM' if C_TRUE > 0.05 else 'LEVEL ONLY'}")
print("#" * 96)
print()

sys.argv = ["sigma_by_dev", "--all-seasons", "--boot", "120",
            "--min-n", "300"]
importlib.import_module("sigma_by_dev").main()
