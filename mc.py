"""
Monte Carlo humility layer — converts a point projection into P(over line)
and an edge-vs-vig, with a season-stage uncertainty multiplier.
Market-agnostic: the sigma rule per market is the only market-specific input.
"""
from scipy.stats import norm

import mc_pricing

# Calibrated pricing, validated 2025 OOS: P(over) error fell from ~5.0 pp to
# ~1.6 pp across markets. mc_pricing owns the distribution family, sigma rule
# and zero gate per market. The edge/tier interface below is unchanged.
#
# Season-stage multiplier is OFF for now. Under the old symmetric normal,
# widening sigma left P(over) at the money at 0.50. Under a mean-preserving
# gamma it lowers the median, so 1.5x adds ~6 pp of under-tilt that has never
# been scored against outcomes. The bake-off validated 1.0x only. Revisit once
# the early-season interaction is tested on weeks 1-4 of 2023-2025.
USE_STAGE_MULTIPLIER = False

# Per-market baseline sigma rules (from out-of-sample residual analysis).
# proportional: sigma = k * projection ; flat: fixed sigma.
# anytime_td is intentionally absent — it's already a calibrated probability.
SIGMA_RULES = {
    "receiving":   {"type": "proportional", "k": 0.66},
    "rushing":     {"type": "proportional", "k": 0.55},
    "receptions":  {"type": "flat", "sigma": 2.1},
    "qb_passing":  {"type": "flat", "sigma": 71.0},
    "qb_rushing":  {"type": "proportional", "k": 0.77},
}


def base_sigma(market, projection):
    """Calibrated sigma. Sub-proportional: scales like proj**0.76, not proj."""
    return mc_pricing.sigma_for(market, projection, 1.0)


def legacy_base_sigma(market, projection):
    """The old hand-set rule. Kept for comparison and rollback."""
    rule = SIGMA_RULES[market]
    if rule["type"] == "proportional":
        return rule["k"] * projection
    return rule["sigma"]


def stage_multiplier(games_played):
    """1.5x at 0 games, linear down to 1.0x at 4+ (early-season humility)."""
    if games_played >= 4:
        return 1.0
    return 1.5 - 0.125 * games_played


def prob_over(market, projection, line, games_played):
    """P(actual > line) via a normal curve, sigma inflated by season stage.
    Returns probability 0-1, or None if uncomputable."""
    mult = stage_multiplier(games_played) if USE_STAGE_MULTIPLIER else 1.0
    return mc_pricing.p_over(market, projection, line, stage_mult=mult)


def american_breakeven(odds):
    """Vig-adjusted breakeven probability implied by American odds (0-1)."""
    if odds is None:
        return None
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def edge_calc(market, projection, line, games_played,
              over_odds=None, under_odds=None):
    """Returns dict with p_over, p_under, edge_over, edge_under, best_side,
    best_edge, and whether odds were approximated (-110 fallback).
    All probabilities/edges in percentage points."""
    p_o = prob_over(market, projection, line, games_played)
    if p_o is None:
        return None
    p_u = 1 - p_o

    approx = False
    if over_odds is None or under_odds is None:
        over_odds = over_odds if over_odds is not None else -110
        under_odds = under_odds if under_odds is not None else -110
        approx = True

    be_o = american_breakeven(over_odds)
    be_u = american_breakeven(under_odds)
    edge_o = (p_o - be_o) * 100
    edge_u = (p_u - be_u) * 100

    if edge_o >= edge_u:
        best_side, best_edge = "OVER", edge_o
    else:
        best_side, best_edge = "UNDER", edge_u

    return {
        "p_over": round(p_o * 100, 1),
        "p_under": round(p_u * 100, 1),
        "edge_over": round(edge_o, 1),
        "edge_under": round(edge_u, 1),
        "best_side": best_side,
        "best_edge": round(best_edge, 1),
        "approx_odds": approx,
    }


def tier_for_edge(edge_points):
    """Market-agnostic tier from edge in percentage points.
    Thresholds mirror TD's existing prob-point tiers."""
    if edge_points is None:
        return ""
    if edge_points < 2:
        return "Pass"
    elif edge_points < 4:
        return "Lean"
    elif edge_points < 7:
        return "Strong"
    return "Max"