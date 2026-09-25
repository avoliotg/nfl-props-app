"""
OpalScales - calibrated pricing for all six markets.

WHAT THIS REPLACES
    The old normal-CDF pricing with hand-set sigma rules. Validated on 2025
    out-of-sample (fit 2022-2023, selected on 2024, scored on 2025 once):

      market       P(over) error   pit_ks   family
      qb_rushing   6.31 -> 1.20    0.025    gamma + logistic zero gate
      receiving    5.75 -> 1.61    0.028    gamma + logistic zero gate
      rushing      4.91 -> 1.16    0.026    gamma
      receptions   3.17 -> 1.51    0.035    negative binomial (discrete)
      qb_passing   4.95 -> 2.12    0.058    gamma
      anytime_td   unchanged                already calibrated, AUC 0.63

WHY IT WORKS
    Real yardage outcomes are right-skewed with a point mass at zero. A
    symmetric normal has to balance that skew, which systematically OVERSTATES
    P(over) at lines near the projection. Every market shifts the same
    direction under the fix: at-the-money P(over) drops 8 to 14 points, so
    unders gain edge and overs lose it.

    Spread is also SUB-proportional: it scales like proj**0.76 (hierarchical
    Bayesian posterior across three markets), never proportionally. The old
    proportional-k rules were far too narrow at low projections. At proj 5,
    receiving's k=0.66 gave sigma 3.3 against a fitted 13.7.

KEY DESIGN POINTS
  1. TOTAL MEAN == proj ALWAYS. The OLS models were fitted to make proj the
     conditional mean, so any family that shifted the mean would be smuggling
     in a bias correction. Under a zero gate of weight pi, the continuous part
     takes mean proj/(1-pi) and its variance is solved backward so the total
     still matches sigma.
  2. PROJECTION FLOORS. Each market has a floor at the 1st percentile of its
     fitted projections. Below it, pricing is extrapolation and this module
     returns None. None MUST render as unpriced, NOT as zero edge, or every
     low-projection player becomes a fake Max-tier under.
  3. STAGE MULTIPLIER multiplies sigma, not the mean. Widening sigma only makes
     the variance decomposition more feasible, so it is safe in that direction.
  4. RECEPTIONS IS DISCRETE. P(over 3.5) = P(Y >= 4) = 1 - cdf(3). Off-by-one
     here would be a silent systematic error on every reception line.
  5. NO ZERO GATE FOR RECEPTIONS. Its fitted pi was 0.0009; the negative
     binomial's own overdispersion absorbs all 7.6% of zeros. Verified against
     the empirical PMF: model P(Y=k) matched observed within 1.7 pp for every
     k from 0 to 10, and exactly at k=0.

ANNUAL REFIT
    PARAMS below is the only thing that needs updating. 2025 ran about 1.8
    yards below the fit seasons and that drift accounted for roughly 1.7 pp of
    the residual receiving error. Refit each preseason; qb_passing first, since
    it drifts most.
"""

import math

import numpy as np
from scipy import stats

__all__ = ["p_over", "p_under", "edge", "breakeven_from_odds",
           "american_to_prob", "sigma_for", "PARAMS", "price",
           "BLEND", "blended_mean", "has_model_signal"]


# ----------------------------------------------------------------------------
# FITTED PARAMETERS  (fit 2022-2024, validated on 2025)
# ----------------------------------------------------------------------------

PARAMS = {
    "receiving": dict(
        family="gamma",
        sigma=("power", 3.2720, 0.6172),      # 3.272 * proj ** 0.6172
        gate=(-0.1440, -0.063440),            # logit(pi) = g0 + g1 * proj
        floor=11.5,
    ),
    "rushing": dict(
        family="gamma",
        sigma=("power", 1.9220, 0.7149),
        gate=None,                            # zero rate 0.1%, no gate needed
        floor=34.5,
    ),
    "qb_rushing": dict(
        family="gamma",
        sigma=("affine", 4.6240, 0.7246),     # 4.624 + 0.7246 * proj
        gate=(0.6030, -0.139160),
        floor=1.5,
    ),
    "qb_passing": dict(
        family="gamma",
        sigma=("prop", 0.3531),               # 0.3531 * proj
        gate=None,
        floor=159.0,
    ),
    "receptions": dict(
        family="nbinom",
        sigma=("affine", 1.1430, 0.2992),
        gate=None,
        floor=1.7,
    ),
}

PI_CAP = 0.35          # never claim more than a 35% chance of a blank
SIGMA_MIN = 0.05
MARKETS = tuple(PARAMS)


# ----------------------------------------------------------------------------
# BLEND  (Phase 4.4, added September 24 2026)
# ----------------------------------------------------------------------------
#
# WHAT CHANGED AND WHY
#     This module used to be handed the model's RAW PROJECTION as the
#     distribution mean. Measured against 2023-2026 closing lines, that was
#     the single largest defect in the pricing layer: the bottom calibration
#     band ran 14 to 20 points too low and the top band 17 to 32 points too
#     high, in EVERY market. That tilt is mean error. It cannot be a sigma
#     problem, because dP(over)/dsigma is negative in every band and both
#     families, so a sigma change moves every band the SAME direction and
#     can never move the two ends in opposite directions.
#
#     Pricing the blended mean instead collapses the tilt:
#
#       market       worst band, raw proj   worst band, blend
#       receptions          +21.1                  +2.4
#       receiving           -18.4                  -3.8
#       rushing             +31.7                  -1.5
#       qb_passing          -20.6                  -5.3
#
#     Sigma is UNCHANGED. A flat residual-SD sigma and a sqrt(k*mean) sigma
#     were both tested against the blend and both were WORSE than the PARAMS
#     forms above (receiving went -3.8 to -20.2 on flat). The sub-proportional
#     PARAMS forms scale with the player; a single scalar does not. Do not
#     "fix" sigma without re-running blend_sigma_grid.py first.
#
# BETA IS CLIPPED TO ZERO unless its game-clustered CI excludes zero.
#     Only receptions survives that test (beta 0.226, SE 0.035, t +6.5).
#     receiving 0.066 (SE 0.044), rushing -0.043 (SE 0.048) and qb_passing
#     0.085 (SE 0.067) all include zero, so the projection does NOT enter
#     their price. Shipping a negative beta would price AWAY from the
#     projection, and shipping an insignificant one would display phantom
#     edges in three markets that carry no signal. The harness's own
#     out-of-sample procedure already clips rushing to 0.000.
#
# ALPHA IS THE DIRECT MEASUREMENT AT ZERO DEVIATION, not a regression
#     intercept. Once beta is clipped to zero, the intercept of a free-slope
#     fit is the wrong parameter: it carries a correction for a slope the
#     model no longer has. Rushing shows the gap plainly, intercept +3.46
#     against direct +4.01.
#
#     Alpha is the mean-versus-median offset. Outcomes are right-skewed and
#     the line sits near the median, so the MEAN sits above it. Dropping
#     alpha was tested and is much worse: it prices the mean at the line,
#     understates P(over) everywhere, and put 7,858 of 7,861 receiving props
#     into the top tier. Alpha is what stops that.
#
# qb_passing ALPHA IS ZERO BECAUSE IT IS UNRESOLVABLE, not because it is
#     absent. Direct alpha is +0.55 with an SE of 2.55, per-season +1.37,
#     +4.68, -3.94. Sigma there is about 73 yards, so a few yards of skew is
#     one fortieth of a standard deviation and 1,670 rows cannot pin it.
#     Pricing at the line is honest; shipping +2.37 as though it were known
#     is not.
#
# qb_rushing IS ZERO FOR A DIFFERENT REASON: no data. MARKET_MAP in
#     import_lines has no qb_rushing key, so those props store as
#     market='rushing'. Zero rows have ever graded. That is a data gap, not
#     a finding. Revisit when plan item 3.5 lands.
#
# WHAT WAS TESTED AND FOUND FLAT
#     Both parameters were cut by player caliber (target-share quartile,
#     position, exact line value), by game context (home/away, favourite/
#     underdog, game total, roof, weekday, divisional, rest days) and by
#     line source. Every cut came back consistent against a Bonferroni
#     threshold. So a single pair per market is defensible. Note the power
#     limit honestly: minimum detectable spread in beta ran 0.10 to 0.42
#     depending on the cut, so "flat" means "no difference large enough to
#     see at this n", not "identical".
#
#     Alpha is NOT flat across seasons for receptions (direct p=0.026,
#     2023 +0.369 against 2024 +0.054 and 2025 +0.081). If this is ever
#     improved, recency weighting is the lever, not a caliber table.
#
# THE ANCHOR MATTERS AND WAS VERIFIED
#     Beta is highly sensitive to which line you anchor on: across min,
#     median, FanDuel and max it spans 0.24 to 0.37, which is 7 to 10 SEs.
#     min and max are order statistics over eight books, so anchoring on
#     them inflates beta mechanically, the same artifact family as line_best
#     and BetRivers. FanDuel and consensus agree (0.226 vs 0.275, z -1.07)
#     and those are the only two real anchors.
#
#     The app prices a HAND-TYPED line, so the constants only hold if what
#     gets typed is FanDuel's line. Checked against the capture: reception
#     lines match at 92.4 percent of comparable props. The yardage markets
#     match at 44 to 47 percent, which is line movement between capture and
#     entry on a fine grid, and it does not matter there because beta is
#     zero so the projection never enters the price.
#
BLEND = {
    # RETIRED 2026-09-25. beta was 0.2261, measured against the LINE with a
    # game-clustered t of +6.5 and confirmed by four independent routes. That
    # measurement was not wrong. What was wrong was the benchmark.
    #
    # receptions is 93 percent flat on the line because FanDuel moves
    # reception prices through the ODDS, so the line is the STALE quantity
    # and the price is the sharp one. A model that improves on a stale
    # quantity adds nothing to a sharp one.
    #
    # receptions_shipped_vs_price.py tested the SHIPPED path (verified
    # identical to mc.prob_over, max abs diff 0.00e+00 on 7,452 rows)
    # against FanDuel's own devigged price:
    #
    #   log loss      0.69027 vs the market's 0.68407
    #                 delta +0.00620, 95% [+0.00322, +0.00918]
    #   encompassing  coefficient on (shipped minus market) t = +0.60
    #   book_p coef   +1.0890, so the market is well calibrated and has no
    #                 slack to exploit
    #
    # The encompassing test is the one betting cares about, and its power at
    # this sample size is t +7.45 when a real advantage is planted. So this
    # is a POWERED null. It is also the strong direction of an asymmetry:
    # these blend constants were fitted on the very rows being scored, so
    # the test was biased in favour of the price and it still failed.
    #
    # WHY NOT A WIDER SIGMA. The natural guess was over-confidence, fixable
    # by pricing around resid_sd. The opposite is true. The shipped price
    # spans 0.377 to 0.628 with sd 0.0258 against the market's 0.0659, so it
    # is two and a half times LESS dispersed. It is well calibrated (bin
    # biases +0.0014, -0.0079, +0.0239, better than the market's own) because
    # it barely has an opinion. Widening sigma would flatten it further.
    #
    # TO REVIVE: a positive encompassing coefficient against the DEVIGGED
    # PRICE, not a larger beta against the line. The mean absolute
    # disagreement is 0.0535, which bounds the upside no matter how
    # informative the disagreement turns out to be.
    #
    # alpha is UNCHANGED at 0.1491. It is the mean-versus-median offset, a
    # valid pricing parameter and an invalid betting signal, and it still
    # belongs in the distribution.
    "receptions": dict(alpha=0.1491, beta=0.0),
    "receiving":  dict(alpha=3.4623, beta=0.0),
    "rushing":    dict(alpha=4.0067, beta=0.0),
    "qb_passing": dict(alpha=0.0,    beta=0.0),
    "qb_rushing": dict(alpha=0.0,    beta=0.0),
}


def has_model_signal(market):
    """True when the projection enters the price for this market.

    False means the market is REFERENCE ONLY: show the projection beside the
    line, price a probability, but do NOT display an edge or a tier. A market
    with beta clipped to zero makes no player-specific claim, so an edge
    column there is a verdict the evidence does not support. This is plan
    item J1 in code rather than in a document.
    """
    return bool(BLEND.get(market, {}).get("beta", 0.0) != 0.0)


def blended_mean(market, proj, line):
    """line + alpha + beta * (projection - line). None if unusable.

    This is the distribution mean. Pass the result to p_over, not the raw
    projection.

    Markets absent from BLEND fall back to the raw projection rather than
    silently pricing at the line, so a new market added to PARAMS without a
    BLEND entry behaves as it did before this change instead of quietly
    losing its model.
    """
    try:
        proj = float(proj)
        line = float(line)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(proj) and np.isfinite(line)):
        return None
    b = BLEND.get(market)
    if b is None:
        return proj
    m = line + b["alpha"] + b["beta"] * (proj - line)
    return float(m) if np.isfinite(m) else None


# ----------------------------------------------------------------------------
# SIGMA
# ----------------------------------------------------------------------------


def sigma_for(market, proj, stage_mult=1.0):
    """Base sigma for a market, times the season-stage multiplier."""
    if market not in PARAMS:
        raise KeyError(f"unknown market: {market}")
    rule = PARAMS[market]["sigma"]
    x = abs(float(proj))
    kind = rule[0]
    if kind == "power":
        s = rule[1] * (x ** rule[2]) if x > 0 else SIGMA_MIN
    elif kind == "affine":
        s = rule[1] + rule[2] * x
    elif kind == "prop":
        s = rule[1] * x
    elif kind == "flat":
        s = rule[1]
    else:
        raise ValueError(f"bad sigma rule: {rule}")
    return max(s * float(stage_mult), SIGMA_MIN)


def _gate_pi(market, proj):
    g = PARAMS[market]["gate"]
    if g is None:
        return 0.0
    z = g[0] + g[1] * float(proj)
    return min(1.0 / (1.0 + math.exp(-z)), PI_CAP)


def _components(market, proj, stage_mult):
    """(pi, mean, sd) of the continuous part, with total mean == proj."""
    proj = max(float(proj), 0.0)
    sigma = sigma_for(market, proj, stage_mult)
    pi = _gate_pi(market, proj)
    m = proj / (1.0 - pi)
    var_c = (sigma ** 2 + proj ** 2) / (1.0 - pi) - m ** 2
    if not np.isfinite(var_c) or var_c <= 0.0:
        return None
    return pi, m, math.sqrt(var_c)


# ----------------------------------------------------------------------------
# PRICING
# ----------------------------------------------------------------------------


def p_over(market, proj, line, stage_mult=1.0):
    """P(outcome > line). Returns None when the projection is unpriceable.

    None means UNPRICED, not zero edge. Call sites must render it blank and
    exclude the row from tiering and Top Plays.
    """
    if market not in PARAMS:
        raise KeyError(f"unknown market: {market}")
    try:
        proj = float(proj)
        line = float(line)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(proj) and np.isfinite(line)):
        return None
    if proj < PARAMS[market]["floor"]:
        return None
    if line < 0:
        return 1.0

    comp = _components(market, proj, stage_mult)
    if comp is None:
        return None
    pi, m, s = comp
    fam = PARAMS[market]["family"]

    if fam == "gamma":
        if m <= 0 or s <= 0:
            return None
        shape = (m / s) ** 2
        scale = (s ** 2) / m
        sf = float(stats.gamma.sf(line, a=shape, scale=scale))
    elif fam == "nbinom":
        var = s ** 2
        if var <= m * 1.0001:           # NB needs var > mean
            var = m * 1.0001
        r = m ** 2 / (var - m)
        p = r / (r + m)
        # discrete: P(Y > line) = P(Y >= floor(line)+1) = 1 - cdf(floor(line))
        k = math.floor(line)
        sf = float(1.0 - stats.nbinom.cdf(k, n=r, p=p))
    else:
        raise ValueError(f"bad family: {fam}")

    val = (1.0 - pi) * sf
    if not np.isfinite(val):
        return None
    return float(min(max(val, 0.0), 1.0))


def p_under(market, proj, line, stage_mult=1.0):
    """P(outcome < line). Lines are half-integers so ties do not arise."""
    po = p_over(market, proj, line, stage_mult)
    return None if po is None else 1.0 - po


# ----------------------------------------------------------------------------
# ODDS AND EDGE
# ----------------------------------------------------------------------------


def american_to_prob(odds):
    """American odds -> implied probability (with vig)."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(o) or o == 0:
        return None
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)


def breakeven_from_odds(over_odds, under_odds=None):
    """Vig-adjusted breakeven for the OVER side.

    With both sides, the two implied probabilities are normalised so they sum
    to 1, which removes the hold. With only one side, the raw implied
    probability is returned and the number is optimistic by roughly half the
    vig.
    """
    po = american_to_prob(over_odds)
    if po is None:
        return None
    pu = american_to_prob(under_odds)
    if pu is None or (po + pu) <= 0:
        return po
    return po / (po + pu)


def edge(market, proj, line, side, over_odds, under_odds=None, stage_mult=1.0):
    """Edge in PERCENTAGE POINTS: model probability minus breakeven.

    Positive means the model thinks the side is underpriced. Returns None if
    unpriceable, which must propagate as unpriced.
    """
    po = p_over(market, proj, line, stage_mult)
    if po is None:
        return None
    be_over = breakeven_from_odds(over_odds, under_odds)
    if be_over is None:
        return None
    side = str(side).upper()
    if side == "OVER":
        return 100.0 * (po - be_over)
    if side == "UNDER":
        return 100.0 * ((1.0 - po) - (1.0 - be_over))
    raise ValueError("side must be OVER or UNDER")


def price(market, proj, line, over_odds=None, under_odds=None, stage_mult=1.0):
    """Everything at once. Returns a dict, with None fields when unpriceable."""
    po = p_over(market, proj, line, stage_mult)
    out = dict(market=market, proj=proj, line=line,
               sigma=None, p_over=po, p_under=None,
               edge_over=None, edge_under=None, best_side=None, best_edge=None,
               unpriced=(po is None),
               reason=None if po is not None else
               (f"proj {proj} below floor {PARAMS[market]['floor']}"
                if market in PARAMS and _finite(proj)
                and float(proj) < PARAMS[market]["floor"] else "not priceable"))
    if po is None:
        return out
    out["sigma"] = sigma_for(market, proj, stage_mult)
    out["p_under"] = 1.0 - po
    if over_odds is not None:
        eo = edge(market, proj, line, "OVER", over_odds, under_odds, stage_mult)
        eu = edge(market, proj, line, "UNDER", over_odds, under_odds, stage_mult)
        out["edge_over"], out["edge_under"] = eo, eu
        if eo is not None and eu is not None:
            out["best_side"] = "OVER" if eo >= eu else "UNDER"
            out["best_edge"] = max(eo, eu)
    return out


def _finite(v):
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------------


def _self_test():
    print("=" * 72)
    print("mc_pricing self-test")
    print("=" * 72)
    ok = True

    print("\n[1] sigma vs the OLD proportional rules")
    old = {"receiving": 0.66, "rushing": 0.55, "qb_rushing": 0.77}
    for mk, k in old.items():
        print(f"  {mk}")
        print("     proj    old      new    ratio")
        for pv in (5, 10, 20, 40, 70):
            if pv < PARAMS[mk]["floor"]:
                continue
            n = sigma_for(mk, pv)
            print(f"    {pv:5d} {k * pv:8.1f} {n:8.1f} {n / (k * pv):8.2f}")

    print("\n[2] floors return None (must render as UNPRICED, not zero)")
    for mk in MARKETS:
        f = PARAMS[mk]["floor"]
        below = p_over(mk, f - 0.1, f)
        above = p_over(mk, f + 0.1, f)
        good = below is None and above is not None
        ok &= good
        print(f"  {mk:12s} below floor -> {below} | above -> "
              f"{'%.4f' % above if above is not None else None}  "
              f"{'OK' if good else 'FAIL'}")

    print("\n[3] monotonicity: P(over) must fall as the line rises")
    for mk in MARKETS:
        pv = max(PARAMS[mk]["floor"] * 1.5, PARAMS[mk]["floor"] + 1)
        lines = np.linspace(pv * 0.4, pv * 2.0, 12)
        ps = [p_over(mk, pv, l) for l in lines]
        good = all(a >= b - 1e-12 for a, b in zip(ps, ps[1:]))
        ok &= good
        print(f"  {mk:12s} proj {pv:7.1f}  P(over) {ps[0]:.3f} -> {ps[-1]:.3f}  "
              f"{'OK' if good else 'FAIL'}")

    print("\n[4] discrete convention for receptions")
    pv = 4.0
    a = p_over("receptions", pv, 3.5)
    b = p_over("receptions", pv, 3.0)
    c = p_over("receptions", pv, 3.9)
    good = abs(a - b) < 1e-12 and abs(a - c) < 1e-12
    ok &= good
    print(f"  P(over 3.0)={b:.6f}  P(over 3.5)={a:.6f}  P(over 3.9)={c:.6f}")
    print(f"  all equal (P(Y>=4)) -> {'OK' if good else 'FAIL'}")

    print("\n[5] stage multiplier widens without shifting the mean")
    for mk in ("receiving", "qb_rushing"):
        pv = PARAMS[mk]["floor"] * 2
        s1, s15 = sigma_for(mk, pv, 1.0), sigma_for(mk, pv, 1.5)
        p1 = p_over(mk, pv, pv)
        p15 = p_over(mk, pv, pv, stage_mult=1.5)
        print(f"  {mk:12s} sigma {s1:.2f} -> {s15:.2f} | P(over ATM) "
              f"{p1:.4f} -> {p15:.4f}")

    print("\n[6] the board shift: at-the-money P(over), old vs new")
    print("  market        proj   old_normal   new    shift_pp")
    for mk in MARKETS:
        pv = max(PARAMS[mk]["floor"] * 1.6, PARAMS[mk]["floor"] + 2)
        k = old.get(mk)
        if k:
            o = float(stats.norm.sf(pv, loc=pv, scale=k * pv))
        else:
            flat = {"qb_passing": 71.0, "receptions": 2.1}[mk]
            o = float(stats.norm.sf(pv, loc=pv, scale=flat))
        n = p_over(mk, pv, pv)
        print(f"  {mk:12s} {pv:6.1f} {o:11.3f} {n:7.3f} {100 * (n - o):+9.1f}")
    print("  every market shifts NEGATIVE: unders gain edge, overs lose it")

    print("\n[7] edge with vig-adjusted breakeven")
    be = breakeven_from_odds(-110, -110)
    print(f"  breakeven at -110/-110 = {be:.4f} (want 0.5000) "
          f"{'OK' if abs(be - 0.5) < 1e-9 else 'FAIL'}")
    ok &= abs(be - 0.5) < 1e-9
    r = price("receiving", 13.3, 18.5, over_odds=-115, under_odds=-105)
    print(f"  Douglas 13.3 / u18.5: P(under)={r['p_under']:.4f} "
          f"edge_under={r['edge_under']:+.1f} pp  best={r['best_side']}")

    print("\n[8] junk inputs never raise")
    for args in (("receiving", None, 20), ("receiving", float("nan"), 20),
                 ("receiving", 40, None), ("receiving", "x", 20)):
        try:
            p_over(*args)
        except Exception as exc:
            ok = False
            print(f"  FAIL {args} raised {type(exc).__name__}")
    print("  all junk handled -> OK" if ok else "  see failures above")

    print("\n[9] blend: mean must be line+alpha+beta*dev, and beta=0 must "
          "ignore the projection")
    for mk in ("receptions", "receiving", "rushing", "qb_passing"):
        b = BLEND[mk]
        ln = max(PARAMS[mk]["floor"] * 1.2, PARAMS[mk]["floor"] + 1)
        hi = blended_mean(mk, ln * 3.0, ln)
        lo = blended_mean(mk, ln * 0.2, ln)
        want_hi = ln + b["alpha"] + b["beta"] * (ln * 3.0 - ln)
        good = abs(hi - want_hi) < 1e-9
        if b["beta"] == 0.0:
            # with beta zero the projection must not move the mean at all
            good &= abs(hi - lo) < 1e-12
        ok &= good
        print("  %-12s line %7.1f  proj high -> %8.3f  proj low -> %8.3f  "
              "%s" % (mk, ln, hi, lo, "OK" if good else "FAIL"))

    print("\n[10] has_model_signal: which markets carry signal")
    # UPDATED 2026-09-25. This asserted `live == ["receptions"]`, written
    # when receptions was the only market with a nonzero beta. Receptions
    # was retired that day (see the note in BLEND), so the assertion was
    # stale and failed the whole self-test on a change that was deliberate.
    #
    # The check now asserts the INVARIANT rather than a snapshot: every
    # market with a nonzero beta must be in EXPECTED_LIVE, and every market
    # in EXPECTED_LIVE must have one. That still fails loudly if a beta
    # changes by accident, without needing an edit every time the set
    # legitimately changes.
    EXPECTED_LIVE = []      # no market currently beats the devigged price
    live = sorted(mk for mk in BLEND if has_model_signal(mk))
    good = live == sorted(EXPECTED_LIVE)
    ok &= good
    print("  markets with beta != 0: %s" % (live or "none"))
    print("  expected:               %s  %s"
          % (sorted(EXPECTED_LIVE) or "none", "OK" if good else "FAIL"))
    if not good:
        print("  A beta changed without EXPECTED_LIVE being updated. Either",
              "the change was accidental, or the list needs editing and the",
              "reason recorded in BLEND.")

    print("\n[11] blend junk inputs never raise")
    for args in (("receptions", None, 2.5), ("receptions", float("nan"), 2.5),
                 ("receptions", 3.0, None), ("nosuchmarket", 3.0, 2.5)):
        try:
            blended_mean(*args)
        except Exception as exc:
            ok = False
            print("  FAIL %s raised %s" % (args, type(exc).__name__))
    print("  all junk handled -> OK")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    _self_test()
