"""
Smoke test for calib_reconcile.

Builds TWO markets with different families so the composition confound is
actually exercised:
  receptions  negative binomial, var/mean 1.25, so the SHIPPED floored sigma
              is too wide and the zero atom should misfire at low lines
  receiving   gamma, generated at the shipped sigma, so it should calibrate
              roughly and NOT show the handoff pattern

Expected: step 3 shows a POSITIVE dP/dsigma for low-probability count rows
(widening raises P(over) is FALSE for counts, so this should be NEGATIVE for
gamma rows and positive-signed for the count rows whose threshold sits below
the mean). The point of the run is that the sign is not uniform.
"""
import sys
import types
import importlib
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
SEASONS = (2023, 2024, 2025, 2026)
rows = []

# receptions: true var/mean 1.25
for L, n in {0.5: 400, 1.5: 1800, 2.5: 2000, 3.5: 1400, 4.5: 900}.items():
    for i in range(n):
        dev = rng.normal(0, 0.8)
        mu = max(L + 0.12 + 0.28 * dev, 0.15)
        var = 1.25 * mu
        if var > mu * 1.0001:
            r = mu * mu / (var - mu)
            y = rng.negative_binomial(r, r / (r + mu))
        else:
            y = rng.poisson(mu)
        rows.append(dict(season=SEASONS[i % 4], week=1 + (i % 17),
                         market="receptions", player=f"R{L}_{i}",
                         event_id=f"{SEASONS[i % 4]}_{1+(i%17)}_{i%16}",
                         line_consensus=L, line_fanduel=L,
                         actual=float(y), projection=L + dev))

# receiving: gamma at the shipped sigma
for L, n in {20.5: 900, 40.5: 1500, 60.5: 1200, 80.5: 600}.items():
    for i in range(n):
        dev = rng.normal(0, 12.0)
        mu = max(L + 4.0 + 0.09 * dev, 3.0)
        sd = max(3.272 * mu ** 0.6172, 11.5)
        shape = (mu / sd) ** 2
        y = rng.gamma(shape, mu / shape)
        rows.append(dict(season=SEASONS[i % 4], week=1 + (i % 17),
                         market="receiving", player=f"Y{L}_{i}",
                         event_id=f"{SEASONS[i % 4]}_{1+(i%17)}_{i%16}",
                         line_consensus=L, line_fanduel=L,
                         actual=float(y), projection=L + dev))

df = pd.DataFrame(rows)


def _ols(x, y, cluster):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "se_beta": 0.03}


EH = types.ModuleType("eval_harness")
EH.MARKET_SPEC = {"receptions": ("receptions", "receptions"),
                  "receiving": ("receiving", "receiving_yards")}
EH.ALL_SEASONS = SEASONS
EH.ols_clustered = _ols
EH._secrets = lambda: {}
EH.connect = lambda sec: object()
EH.load_lines = lambda cf, s, mk, c, r: (
    df[df["market"] == mk[0]].assign(book="fanduel",
                                     line=df["line_consensus"]))
EH.collapse_books = lambda lines: lines[[
    "season", "week", "market", "player", "event_id",
    "line_consensus", "line_fanduel"]].copy()
EH.score_market = lambda market, seasons, mode=None, fixed_train=None, \
    population=None: df[df["market"] == market][
        ["season", "week", "market", "player", "projection", "actual",
         "event_id"]].copy()
sys.modules["eval_harness"] = EH

models = types.ModuleType("models"); models.__path__ = []
du = types.ModuleType("models.data_utils")
du.norm_join_name = lambda s: pd.Series(s).astype(str).str.lower()
sys.modules["models"] = models
sys.modules["models.data_utils"] = du

print("#" * 100)
print("SYNTHETIC: receptions generated at var/mean 1.25 (shipped is too")
print("wide), receiving generated AT the shipped gamma sigma (should")
print("calibrate). Expect the sign of dP/dsigma to differ between them.")
print("#" * 100)
print()

sys.argv = ["calib_reconcile", "--all-seasons",
            "--markets", "receptions,receiving"]
importlib.import_module("calib_reconcile").main()
