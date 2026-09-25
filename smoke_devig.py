"""smoke_devig.py - planted-answer rig for devig.py.

No network, no database, no repo data. Devigging is pure arithmetic, so every
expected value here is derived by hand rather than read off the code.

Run from the repo root:
    python smoke_devig.py
"""
import sys

import numpy as np

import devig

TOL = 1e-9


def check(name, got, want, tol=TOL):
    if want is None:
        ok = got is None
    elif got is None:
        ok = False
    else:
        ok = abs(got - want) <= tol
    return ok, f"  {name}: got {got!r}, expected {want!r}"


def main():
    f = []

    # ---- conversions, hand-computed ----
    # -110 -> 1 + 100/110 = 1.909090..., so q = 0.5238095238
    # +150 -> 1 + 150/100 = 2.5,          so q = 0.4
    for odds, dec, q in ((-110, 1.0 + 100.0 / 110.0, 110.0 / 210.0),
                         (+150, 2.5, 0.4),
                         (-200, 1.5, 2.0 / 3.0),
                         (+100, 2.0, 0.5)):
        ok, msg = check(f"decimal({odds})", devig.american_to_decimal(odds), dec)
        if not ok:
            f.append(msg)
        ok, msg = check(f"raw_prob({odds})", devig.american_to_raw_prob(odds), q)
        if not ok:
            f.append(msg)

    # ---- prices that are not prices ----
    for bad in (0, 50, -50, 99, -99, None, "", "abc", float("nan"),
                float("inf")):
        if devig.american_to_decimal(bad) is not None:
            f.append(f"  decimal({bad!r}) should be None")

    # ---- the hold, hand-computed ----
    # -110 both sides: q sum = 220/210 = 1.047619..., overround 0.047619,
    # hold = 0.047619 / 1.047619 = 0.0454545...
    over, hold = devig.hold_from_odds(-110, -110)
    ok, msg = check("overround(-110,-110)", over, 2.0 * 110.0 / 210.0 - 1.0)
    if not ok:
        f.append(msg)
    ok, msg = check("hold(-110,-110)", hold, 1.0 / 22.0)
    if not ok:
        f.append(msg)

    # ---- symmetry: identical prices must give exactly 0.5 on every method ----
    for m in devig.METHODS:
        for o in (-110, -113, -120, -200, +100, +150):
            r = devig.devig_two_sided(o, o, m)
            if not r["valid"]:
                f.append(f"  {m} refused symmetric {o}: {r['reason']}")
                continue
            ok, msg = check(f"{m} p_over({o},{o})", r["p_over"], 0.5)
            if not ok:
                f.append(msg)

    # ---- multiplicative, hand-computed ----
    # over -130 -> q = 130/230 = 0.565217391
    # under +104 -> q = 100/204 = 0.490196078
    # sum = 1.055413469; p_over = 0.565217391 / 1.055413469
    qo, qu = 130.0 / 230.0, 100.0 / 204.0
    r = devig.devig_two_sided(-130, +104, "multiplicative")
    ok, msg = check("multiplicative p_over(-130,+104)", r["p_over"],
                    qo / (qo + qu))
    if not ok:
        f.append(msg)

    # ---- additive, hand-computed ----
    r = devig.devig_two_sided(-130, +104, "additive")
    ok, msg = check("additive p_over(-130,+104)", r["p_over"],
                    qo - (qo + qu - 1.0) / 2.0)
    if not ok:
        f.append(msg)

    # ---- shin EQUALS additive for two outcomes ----
    # Measured to 1e-12 over 20,000 random pairs. This is a property of the
    # n = 2 case. If a future edit breaks it, that is a real change and this
    # assertion should be the thing that catches it.
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(3000):
        oo = int(rng.choice([-1, 1]) * rng.integers(100, 5000))
        uo = int(rng.choice([-1, 1]) * rng.integers(100, 5000))
        ra = devig.devig_two_sided(oo, uo, "additive")
        rs = devig.devig_two_sided(oo, uo, "shin")
        if ra["valid"] and rs["valid"]:
            worst = max(worst, abs(ra["p_over"] - rs["p_over"]))
    if worst > 1e-9:
        f.append(f"  shin no longer equals additive for n=2 (max diff "
                 f"{worst:.3e}). Either a real improvement or a regression, "
                 f"but the module docstring claims they are identical.")

    # ---- every method must produce probabilities summing to exactly 1 ----
    bad_sum = 0
    for _ in range(3000):
        oo = int(rng.choice([-1, 1]) * rng.integers(100, 5000))
        uo = int(rng.choice([-1, 1]) * rng.integers(100, 5000))
        for m in devig.METHODS:
            r = devig.devig_two_sided(oo, uo, m)
            if r["valid"] and abs(r["p_over"] + r["p_under"] - 1.0) > 1e-8:
                bad_sum += 1
    if bad_sum:
        f.append(f"  {bad_sum} cases where p_over + p_under != 1")

    # ---- devigging must always REDUCE the favoured side's probability ----
    # The raw implied probability includes the hold, so a fair probability
    # must be lower. A method that raises it is broken.
    raised = 0
    for _ in range(2000):
        oo = int(rng.choice([-1, 1]) * rng.integers(100, 3000))
        uo = int(rng.choice([-1, 1]) * rng.integers(100, 3000))
        for m in devig.METHODS:
            r = devig.devig_two_sided(oo, uo, m)
            if not r["valid"] or r["overround"] is None or r["overround"] <= 0:
                continue
            if r["p_over"] > r["q_over"] + 1e-12:
                raised += 1
    if raised:
        f.append(f"  {raised} cases where the fair probability EXCEEDS the "
                 f"raw implied probability, which is impossible with a "
                 f"positive overround")

    # ---- one-sided markets must be refused, not guessed ----
    r = devig.devig_two_sided(+450, None)
    if r["valid"]:
        f.append("  a one-sided price was devigged; anytime_td must be refused")
    if r["p_over"] is not None:
        f.append("  a refused prop still returned a p_over")

    # ---- an unknown method must be refused ----
    r = devig.devig_two_sided(-110, -110, "vibes")
    if r["valid"]:
        f.append("  an unknown method was accepted")

    # ---- a no-vig book normalises rather than erroring ----
    r = devig.devig_two_sided(+100, +100)
    if not r["valid"]:
        f.append(f"  a zero-overround book was refused: {r['reason']}")
    elif abs(r["p_over"] - 0.5) > TOL:
        f.append(f"  zero-overround p_over {r['p_over']}, expected 0.5")

    # ---- the method spread grows with how lopsided the prop is ----
    def spread(oo, uo):
        am = devig.devig_two_sided(oo, uo)["all_methods"]
        return max(am.values()) - min(am.values()) if am else np.nan

    even, lopsided = spread(-110, -110), spread(+920, -1400)
    if not (even < lopsided):
        f.append(f"  method spread should widen on lopsided props: even "
                 f"{even:.4f}, lopsided {lopsided:.4f}")

    if f:
        print(f"SMOKE FAIL ({len(f)})")
        print("\n".join(f))
        return 1
    print("SMOKE PASS: conversions exact, symmetry exact, sums exact, "
          "shin==additive for n=2, fair < raw always, one-sided refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
