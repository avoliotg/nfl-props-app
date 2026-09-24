"""
Pricing layer: converts a point projection into P(over line) and an
edge-vs-vig.

September 24 2026, Phase 4.4. This layer now prices the BLENDED mean,
line + alpha + beta*(projection - line), rather than the raw projection.
Beta is clipped to zero in every market whose game-clustered CI includes
zero, which is all of them except receptions. See mc_pricing.BLEND for the
parameters, the evidence, and what was tested and found flat.

Markets with beta clipped to zero are REFERENCE ONLY. edge_calc returns
reference_only=True for them and call sites must blank the edge, side and
tier columns. The probability is still honest; the edge is not.

Sigma is unchanged and should stay that way unless blend_sigma_grid.py says
otherwise. Both a flat residual SD and a sqrt(k*mean) form were tested
against the blend and both were worse than the PARAMS forms.
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
    """P(actual > line), pricing the BLENDED mean, not the raw projection.

    Phase 4.4, September 24 2026. The mean is
    line + alpha + beta*(projection - line), with beta clipped to zero in
    every market whose game-clustered CI includes zero. See mc_pricing.BLEND
    for the parameters and the evidence behind them.

    This changed the distribution mean for every market. It did NOT change
    sigma: the PARAMS sigma forms were re-tested against the blend and beat
    both a flat residual SD and a sqrt(k*mean) form.

    Returns probability 0-1, or None if uncomputable.
    """
    mult = stage_multiplier(games_played) if USE_STAGE_MULTIPLIER else 1.0
    mean = mc_pricing.blended_mean(market, projection, line)
    if mean is None:
        return None
    return mc_pricing.p_over(market, mean, line, stage_mult=mult)


def prob_over_raw(market, projection, line, games_played):
    """The PRE-4.4 behaviour: price the raw projection. Kept for comparison
    and rollback. Not used by the app.

    If you are tempted to call this in production, the reason not to is that
    it puts the bottom calibration band 14 to 20 points too low and the top
    band 17 to 32 points too high in every market.
    """
    mult = stage_multiplier(games_played) if USE_STAGE_MULTIPLIER else 1.0
    return mc_pricing.p_over(market, projection, line, stage_mult=mult)


def american_breakeven(odds):
    """RAW implied probability from American odds (0-1), vig included.

    Not vig-adjusted despite the old docstring. The two sides sum to more
    than 1 by the hold. edge_calc removes it when both sides are real.
    """
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

    # Remove the hold. Raw implied probabilities sum to >1, so charging each
    # side its own raw number bills the full vig twice. Normalising so they
    # sum to 1 gives each side its fair breakeven: -114/-114 becomes 0.500
    # rather than 0.5327 each.
    #
    # ONLY when both sides are real. If one was defaulted to -110 (approx),
    # normalising would corrupt the side we do know: an anytime-TD over at
    # +250 would go from a correct 0.286 to a wrong 0.353.
    if not approx and be_o is not None and be_u is not None:
        total = be_o + be_u
        if total > 0:
            be_o, be_u = be_o / total, be_u / total

    edge_o = (p_o - be_o) * 100
    edge_u = (p_u - be_u) * 100

    if edge_o >= edge_u:
        best_side, best_edge = "OVER", edge_o
    else:
        best_side, best_edge = "UNDER", edge_u

    # REFERENCE ONLY markets: beta clipped to zero, so the projection does
    # not enter the price and the model makes no player-specific claim. The
    # probability is still meaningful (it is the line plus a measured skew
    # correction, priced through the right distribution) but an EDGE there is
    # a verdict the evidence does not support. Call sites must blank edge,
    # side and tier when this is True. Plan item J1.
    reference_only = not mc_pricing.has_model_signal(market)

    return {
        "p_over": round(p_o * 100, 1),
        "p_under": round(p_u * 100, 1),
        "edge_over": round(edge_o, 1),
        "edge_under": round(edge_u, 1),
        "best_side": best_side,
        "best_edge": round(best_edge, 1),
        "approx_odds": approx,
        "reference_only": reference_only,
        "blended_mean": mc_pricing.blended_mean(market, projection, line),
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