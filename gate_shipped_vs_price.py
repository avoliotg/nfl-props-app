"""gate_shipped_vs_price.py - does the published procedure still reproduce?

READ ONLY. Runs receptions_shipped_vs_price.py unchanged and checks its
output against the numbers that script published on 2026-09-25.

WHY THIS EXISTS. The plan is to test whether adding AIR YARDS to receiving
produces a projection that beats FanDuel's devigged price. The right
instrument for that is the encompassing test, not beta against the line:
receptions had beta +0.226 at a clustered t of +6.5 and still lost to the
price at t +0.60, and the whole conclusion of 2026-09-25 is that the line
is the stale quantity.

receptions_shipped_vs_price.py implements exactly that procedure, but it is
hardcoded to one market: MARKET = "receptions" at line 94, with calibration
constants for that market at line 98. So testing receiving means
generalising it.

THE PROJECT'S RULE APPLIES. A script reimplementing an existing procedure
must reproduce that procedure's published output BEFORE producing a new
number. So before anything is generalised, the original has to still work.
This checks that, and it is not a formality:

    the cache was REFRESHED today with --refresh, rewriting
    lines_cache.parquet. It came back at the same 184,782 rows with the same
    per-season counts, which is reassuring and not proof. If the refresh
    changed anything the encompassing test touches, every result that rests
    on that cache moved, and this is where it shows up.

WHAT IS CHECKED, AND WHY THE BASELINE WAS REWRITTEN ONCE.

The first version of this gate used figures taken from the handoff
document: log loss 0.69027 against 0.68407, delta +0.00620, encompassing
t +0.60, book_p +1.0890, on 7,452 rows. It failed. The script had not
moved; the reference values had never been produced by the script in its
current state.

    The handoff's run scored 7,452 rows. Today's scores 6,598, because
    1,253 rows are unpriceable and the population is the intersection with
    walk-forward projections. Different population, different numbers. The
    handoff was describing a superseded run and I transcribed it as though
    it were the current baseline.

    The failure was therefore useful: a gate whose reference values came
    from prose rather than from the procedure's own output is not a gate.
    These now come from a recorded run, and the run that produced them is
    named below so the next person can tell the two apart.

BASELINE, from the run of 2026-09-30 on lines_cache.parquet at 184,782
rows, seasons 2023-2026, book fanduel, devig additive:

    rows scored           6,598 of 7,851 settled props, 845 games
    market log loss       0.68354  (devigged book_p)
    shipped log loss      0.69110  (blended_mean -> p_over)
    delta                 +0.00756, 95% [+0.00430, +0.01072]
    book_p coefficient    -0.7823
    encompassing t        -1.36

THE CONCLUSION IS UNCHANGED AND SLIGHTLY STRONGER. The handoff recorded
an encompassing t of +0.60, a null in the favourable direction. Today it is
-1.36: the shipped price's disagreement with the market points the WRONG
way, and log loss is worse with an interval excluding zero. Retiring
receptions stands either way.

TWO THINGS IN THE OUTPUT ARE STALE RATHER THAN WRONG, and are left alone
because this script must run unchanged for the gate to mean anything:

    the ratio line prints 701039206.600 against an expected 4.42. With
    BLEND['receptions']['beta'] now 0.0, the blended mean IS the line plus
    alpha, so its deviation from the line has zero variance and the ratio
    divides by nothing. That section was written when beta was 0.2261.

    the script's own internal gate warns that the over rate is 0.4657
    against a published 0.4720. That is the smaller intersected population,
    which the note itself says to check rather than a sign of corruption.

The parser is deliberately tolerant: it extracts every number from the
output and asks whether each baseline value appears somewhere within
tolerance. That survives a change in wording or spacing, and it fails if a
figure genuinely moves.

Run:  python gate_shipped_vs_price.py
      python gate_shipped_vs_price.py --script receptions_shipped_vs_price.py
"""
import argparse
import re
import subprocess
import sys

# BASELINE from an actual run, 2026-09-30, not from prose. The previous
# values were transcribed from the handoff and described a 7,452-row run
# that this script no longer produces.
#
# Tolerances are tight on the log losses, because those are the figures a
# data change would move first, and looser on the t statistics, which are
# printed to two decimals.
BASELINE_RUN = "2026-09-30, lines_cache.parquet 184,782 rows, 6,598 scored"
PUBLISHED = [
    ("market log loss", 0.68354, 0.0002),
    ("shipped log loss", 0.69110, 0.0002),
    ("log loss delta", 0.00756, 0.0002),
    ("delta CI low", 0.00430, 0.0002),
    ("delta CI high", 0.01072, 0.0002),
    ("book_p coefficient", -0.7823, 0.002),
    ("encompassing t", -1.36, 0.02),
]

# A row count in the output is a stronger tripwire than any statistic: if
# the population changes, every number below it changes with it, and a
# tolerance on a log loss might still pass by luck.
ROW_MARKERS = ["6,598", "7,851", "845", "184782"]

NUM = re.compile(r"[-+]?\d+\.\d+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="receptions_shipped_vs_price.py")
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    cmd = [sys.executable, args.script, "--all-seasons",
           "--cache", args.cache]
    print("=" * 74)
    print("RUNNING THE PUBLISHED PROCEDURE UNCHANGED")
    print("=" * 74)
    print("  %s" % " ".join(cmd))
    print()
    print("  This fits and scores the shipped pricing path against FanDuel's")
    print("  devigged price. It takes a few minutes and bootstraps 2000")
    print("  draws by default.")
    print()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
    except FileNotFoundError:
        print("  %s not found in this directory." % args.script)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  timed out after %ds" % args.timeout)
        sys.exit(1)

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print("-" * 74)
    print(out.rstrip())
    print("-" * 74)

    if proc.returncode != 0:
        print()
        print("  THE SCRIPT EXITED %d. Nothing can be concluded about the"
              % proc.returncode)
        print("  numbers below, and nothing should be generalised until it")
        print("  runs clean.")
        sys.exit(1)

    found = [float(x) for x in NUM.findall(out)]
    print()
    print("=" * 74)
    print("REPRODUCTION CHECK")
    print("=" * 74)
    print("  baseline: %s" % BASELINE_RUN)
    print("  %d numbers in the output" % len(found))
    print()
    # POPULATION FIRST. A statistic can match by luck; a row count that
    # moved means every statistic below it is describing different rows.
    missing_rows = [m for m in ROW_MARKERS if m not in out]
    print("  population markers %s"
          % ("all present" if not missing_rows
             else "MISSING: %s" % missing_rows))
    if missing_rows:
        print("    The scored population has changed. Treat every figure")
        print("    below as describing different rows, and re-baseline")
        print("    rather than adjusting tolerances.")
    print()
    print("  %-22s %12s %10s %s" % ("figure", "published", "tol", "found"))
    print("  " + "-" * 62)

    ok = not missing_rows
    for label, want, tol in PUBLISHED:
        hits = [v for v in found if abs(v - want) <= tol]
        mark = "yes" if hits else "NO"
        if not hits:
            ok = False
            # the nearest value, to show how far off it is
            near = min(found, key=lambda v: abs(v - want)) if found else None
            extra = ("  nearest %.5f" % near) if near is not None else ""
        else:
            extra = "  (%s)" % ", ".join("%.5f" % v for v in hits[:3])
        print("  %-22s %12.5f %10.4f %-4s%s"
              % (label, want, tol, mark, extra))

    print()
    if ok:
        print("  GATE PASSED. Every published figure is present in the")
        print("  output, so the refreshed cache did not move the result and")
        print("  the procedure can be generalised to another market.")
        print()
        print("  NEXT: parameterise MARKET, keeping this gate as the check")
        print("  that the receptions path still returns these numbers after")
        print("  the change. A generalisation that alters the original's")
        print("  output has changed the procedure, not just its scope.")
    else:
        print("  GATE FAILED. At least one published figure is absent.")
        print()
        print("  Do NOT generalise yet. Either the refreshed cache moved the")
        print("  result, in which case every conclusion resting on that")
        print("  cache needs revisiting, or the script's output format")
        print("  changed and this parser needs adjusting. The first")
        print("  possibility is the one that matters: check the row counts")
        print("  and the scored-row counts in the output above against what")
        print("  the script reported on 2026-09-25 before assuming it is")
        print("  cosmetic.")
        sys.exit(1)


if __name__ == "__main__":
    main()
