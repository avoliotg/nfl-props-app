"""
devig.py - proper two-sided devigging. Plan item 5.8.

WHY THIS EXISTS

    A sportsbook posts two prices on a half-integer prop. Converting each to
    an implied probability and using it as-is DOUBLE COUNTS the hold: the two
    raw implied probabilities sum to more than 1, and the excess is the
    book's margin, not information. Every edge computed anywhere in this
    project against a raw implied probability inherits that error.

    The project's own note: multiplicative, additive and Shin differ by 1 to
    2 points on receptions. The candidate rule's whole claimed edge is
    +0.1475 over 176 bets, and its book component is +0.071. So a 1 to 2
    point devigging error is a material fraction of the thing being measured.

WHY FOUR METHODS AND NOT ONE

    There is no consensus answer to how a book distributes its margin across
    two sides, and the choice is an assumption about the book rather than a
    fact about arithmetic:

      multiplicative  the margin is PROPORTIONAL to each side's probability.
                      Simplest, preserves the ratio of implied odds, and
                      known to leave favourite-longshot bias in place.
      additive        the margin is an EQUAL ABSOLUTE amount per side.
                      Treats a 0.90 side and a 0.10 side identically, which
                      can push an extreme longshot below zero.
      shin            the margin exists because the book is defending against
                      informed money, with insider proportion z solved per
                      prop. Between the other two, and usually closest to
                      realized outcomes in the published literature.
      power           p_i proportional to q_i raised to a solved exponent.
                      No behavioural story, often fits best empirically.

    Picking one by assertion would be planting a number. This module computes
    all four and leaves the selection to measurement against realized
    outcomes. That is what market_calibration.py is for.

MEASURED HERE, AND IT REDUCES THE PROBLEM

    shin and additive return IDENTICAL probabilities for a two-outcome
    market. Verified to within 1e-12 across 20,000 random American price
    pairs spanning -5000 to +5000, and the closed forms agree to twelve
    decimal places on hand-checked cases.

    That is a mathematical property of the n = 2 case, not a coincidence of
    this implementation, and it matters for two reasons. There are THREE
    distinct methods here rather than four, so any comparison table that
    presents shin and additive as separate evidence is double counting. And
    the literature's general preference for Shin over multiplicative
    reduces, on a two-sided prop, to a preference for an EQUAL ABSOLUTE
    margin over a PROPORTIONAL one. Both are kept as separate entry points
    because the app may one day devig a three-way market, where they diverge.

WHERE THE METHODS ACTUALLY DISAGREE

    Not on even props. Measured spread in fair P(over) across methods:

        -130 / +104     0.30 points
        +102 / -122     0.18 points
        -154 / +124     0.61 points
        +280 / -360     1.76 points
        +920 / -1400    2.07 points

    The disagreement is negligible near a coin flip and grows with how
    lopsided the prop is. That brackets the 1 to 2 points the project's notes
    recorded, and it locates the problem: the choice of method is nearly
    irrelevant for receptions at 2.5 and 3.5 where prices sit near even, and
    it matters a great deal at the bottom of the board, which is exactly
    where the template-pricing lead lives.

WHAT THIS MODULE DOES NOT DO

    It does not decide which method is correct, it does not touch pricing,
    and it takes no view on whether any edge exists. It converts two American
    prices into a fair probability four ways, reports the hold, and flags
    every degenerate case rather than silently returning a plausible number.

    NOTE ON ONE-SIDED MARKETS. anytime_td carries a single price with no
    under, so two-sided devigging is IMPOSSIBLE there and this module says so
    rather than guessing. A one-sided price cannot be devigged without an
    assumption about the hold, and assuming the hold is the thing being
    measured.
"""
import numpy as np

__all__ = ["american_to_decimal", "american_to_raw_prob", "devig_two_sided",
           "METHODS", "hold_from_odds"]

METHODS = ("multiplicative", "additive", "shin", "power")

# Below this the two raw probabilities sum to essentially 1 and every method
# converges, so solving for z or k is numerically pointless.
_NO_VIG_TOL = 1e-9


def american_to_decimal(odds):
    """American odds to decimal. None for anything unusable.

    American odds have no valid value between -100 and +100 exclusive, and 0
    is not a price. Returning None rather than a number keeps a malformed
    scrape out of a probability.
    """
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(o) or o == 0:
        return None
    if -100.0 < o < 100.0:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def american_to_raw_prob(odds):
    """Raw (vig-inclusive) implied probability. None if the price is unusable.

    This is the quantity the project has been using directly. It is NOT a
    probability: two sides of the same prop sum to more than 1.
    """
    d = american_to_decimal(odds)
    return None if d is None else 1.0 / d


def hold_from_odds(over_odds, under_odds):
    """The book's margin. Returns (overround, hold) or (None, None).

    overround = q_over + q_under - 1, the excess above a fair book.
    hold      = overround / (1 + overround), the margin as a share of
                turnover, which is the number quoted as "the book holds 6
                percent".

    Both are reported because this project's notes use the word "hold" for
    figures like 0.0608 and 7.0 percent, and the two definitions differ by
    enough to matter when comparing books.
    """
    qo = american_to_raw_prob(over_odds)
    qu = american_to_raw_prob(under_odds)
    if qo is None or qu is None:
        return None, None
    s = qo + qu
    return s - 1.0, (s - 1.0) / s


def _bisect(f, lo, hi, tol=1e-12, iters=200):
    """Plain bisection. numpy only, no scipy, so this stays importable
    anywhere the project runs including Colab and the Actions runner."""
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return None
        if abs(fm) < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _multiplicative(qo, qu):
    s = qo + qu
    return qo / s, qu / s


def _additive(qo, qu):
    """Subtract an equal share of the overround from each side.

    Can drive an extreme longshot below zero, which is a real property of the
    method and not a bug. The caller is told via the `valid` flag rather than
    being handed a clipped number that looks fine.
    """
    s = qo + qu
    shift = (s - 1.0) / 2.0
    return qo - shift, qu - shift


def _shin(qo, qu):
    """Shin's insider-trading model, solving z per prop.

        p_i = ( sqrt(z^2 + 4(1-z) * q_i^2 / S) - z ) / (2(1-z))

    with z chosen so the two fair probabilities sum to 1. At z = 0 the sum is
    sqrt(S), which exceeds 1 whenever the book has a margin, and the sum
    falls as z rises, so a root exists and bisection finds it.
    """
    s = qo + qu

    def p_of(z, q):
        if z <= 0:
            return q / np.sqrt(s)
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / s) - z) / (2.0 * (1.0 - z))

    def gap(z):
        return p_of(z, qo) + p_of(z, qu) - 1.0

    z = _bisect(gap, 0.0, 0.999999)
    if z is None:
        return None, None, None
    return p_of(z, qo), p_of(z, qu), z


def _power(qo, qu):
    """p_i proportional to q_i ** k, with k solved so the sum is 1.

    k > 1 for a normal book, since raising numbers below 1 to a power above 1
    shrinks them. No behavioural story, but it often matches realized
    outcomes best, so it belongs in the comparison.
    """
    def gap(k):
        return qo ** k + qu ** k - 1.0

    k = _bisect(gap, 0.2, 20.0)
    if k is None:
        return None, None, None
    return qo ** k, qu ** k, k


def devig_two_sided(over_odds, under_odds, method="shin"):
    """Fair probabilities from a two-sided American price.

    Returns a dict, always, so a caller can branch on `valid` rather than on
    a None:

        valid        False when the input cannot be devigged at all
        reason       why, when valid is False
        method       the method actually applied
        p_over       fair P(over)
        p_under      fair P(under)
        q_over       raw vig-inclusive implied P(over), for comparison
        q_under      raw vig-inclusive implied P(under)
        overround    q_over + q_under - 1
        hold         overround / (1 + overround)
        param        z for shin, k for power, None otherwise
        all_methods  {method: p_over} for every method that succeeded

    p_over and p_under are guaranteed to sum to 1 when valid is True.
    """
    out = {"valid": False, "reason": None, "method": method,
           "p_over": None, "p_under": None, "q_over": None, "q_under": None,
           "overround": None, "hold": None, "param": None, "all_methods": {}}

    if method not in METHODS:
        out["reason"] = f"unknown method {method!r}, expected one of {METHODS}"
        return out

    qo = american_to_raw_prob(over_odds)
    qu = american_to_raw_prob(under_odds)
    if qo is None or qu is None:
        # The one-sided case lands here, and deliberately. A single price
        # cannot be devigged without assuming the hold.
        out["reason"] = ("need two valid American prices; one-sided markets "
                         "such as anytime_td cannot be devigged two-sided")
        return out

    s = qo + qu
    out.update(q_over=qo, q_under=qu, overround=s - 1.0, hold=(s - 1.0) / s)

    if s <= 1.0 + _NO_VIG_TOL:
        # No margin, or a genuine arbitrage. Every method agrees here, so
        # normalising is both correct and the only sensible thing to do, but
        # it is worth surfacing because a real book does not price this way.
        po, pu = _multiplicative(qo, qu)
        out.update(valid=True, p_over=po, p_under=pu,
                   reason=f"overround {s - 1.0:+.6f} is zero or negative; "
                          f"normalised without a method")
        out["all_methods"] = {m: po for m in METHODS}
        return out

    results = {}
    for m in METHODS:
        if m == "multiplicative":
            po, pu, prm = (*_multiplicative(qo, qu), None)
        elif m == "additive":
            po, pu, prm = (*_additive(qo, qu), None)
        elif m == "shin":
            po, pu, prm = _shin(qo, qu)
        else:
            po, pu, prm = _power(qo, qu)
        if po is None or not np.isfinite(po) or not (0.0 < po < 1.0):
            continue
        results[m] = (po, pu, prm)

    out["all_methods"] = {m: v[0] for m, v in results.items()}

    if method not in results:
        out["reason"] = (f"{method} produced no usable probability for these "
                         f"prices (over {over_odds}, under {under_odds}); "
                         f"methods that worked: {sorted(results)}")
        return out

    po, pu, prm = results[method]
    out.update(valid=True, p_over=po, p_under=pu, param=prm)
    return out
