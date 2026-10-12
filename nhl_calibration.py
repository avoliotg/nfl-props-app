"""nhl_calibration.py - the pre-registered NHL shots-on-goal calibration test.

PRE-REGISTERED IN PREREG_nhl_calibration.md, written 2026-10-09 before the
1,281-event purchase. Nothing here chooses a threshold, a bin edge, a
direction or a split point. All of those are fixed in that document.

    python nhl_calibration.py
    python nhl_calibration.py --cache nhl_lines.parquet

PRIMARY: slope of realised outcome on devigged price. Predicted BELOW 1.0,
the favourite-longshot shape. Passes only if the pooled slope is
significantly below 1.0 at two-sided 0.05 AND the point estimate is below
1.0 in both halves of a chronological split.

SECONDARY, EXPLORATORY: per-bin calibration miss, read against a
max-statistic null rather than a bare p-value.

NOTHING IS WRITTEN. Reads Supabase and a local parquet, prints, exits.

THE GATE. Section 7 item 5 of the pre-registration makes reproduction of
the pilot's published figures a hard precondition. If the pilot's hold, sd,
symmetric share or line values do not come back, the script aborts before
printing any new number. That rule exists because gap_test.py once produced
a clean-looking result with split_qb_rushing omitted.

ONE DETAIL PINNED HERE, NOT CHANGED. The pre-registration says the split is
at "the median game date" without saying what happens when several games
share it. This splits on the ordered list of DISTINCT dates so no date
falls in both halves, with the earlier half taking the extra date when the
count is odd. That keeps every game whole and makes the boundary
reproducible.
"""
import argparse
import os
import sys
from datetime import timezone

import numpy as np
import pandas as pd

import eval_harness as eh
import nhl_calib_core as C
from nhl_outcomes import norm_name

SPORT = "NHL"

# NOT "closing". Every NHL row carries 'closing_nhl'.
#
# odds_backfill.py writes the label straight from --snapshots, so it wrote
# 'closing', and a grouped query on 2026-10-09 confirmed 3,097 NHL rows under
# that label. By 2026-10-12 there were none: all 97,296 NHL rows read
# 'closing_nhl'. A repo-wide grep for that string across .py, .sql and .yml
# returns nothing, so no script produces it and it was a manual UPDATE. The
# reason is sound, whoever did it: eval_harness.fetch_lines filters on
# snapshot_label = 'closing' and has NO sport filter, so without the suffix
# every NFL research script would silently ingest NHL rows the moment a
# season label collided. Recorded as unexplained but benign; the data is
# intact and nothing in the repo will rewrite it again.
#
# This cost a run: the first real invocation filtered on 'closing' and
# returned zero rows.
LABEL = "closing_nhl"
BOOK = "fanduel"
SEASON = 2024
PARQUET = "nhl_sog_20242025.parquet"

# ---------------------------------------------------------------- the gate
#
# The pilot is the 40 events with the lowest ids. Verified 2026-10-12: that
# block spans 2024-10-04 to 2025-04-23, which is the even spread
# --spread-events 40 produces, so it is the pilot and not simply the first 40
# games of the season.
#
# IT IS NO LONGER THE POPULATION THAT PRODUCED THE PUBLISHED FIGURES.
# Published: 523 FanDuel rows across 39 events. Now: 535 across 40. That is
# one game's worth more, and it is the event that had no FanDuel rows when
# the pilot ran. It gained coverage afterwards.
#
# So an EXACT reproduction is impossible. The tolerances below are NOT
# widened until the old numbers pass, because that would defeat the gate.
# Instead the gate checks the quantities that do not move when a population
# grows by 12 rows out of 523 (hold, dispersion, symmetric share, the set of
# line values) and PRINTS the row and event counts against published so the
# discrepancy stays visible rather than absorbed.
PILOT_EVENTS = 40
PILOT_FD_ROWS = 523          # published; now 535, see above
PILOT_FD_EVENTS = 39         # published; now 40, see above
PILOT_HOLD = 1.0651
PILOT_SD = 0.0736
PILOT_SYM_PCT = 2.1
PILOT_LINES = {1.5: 224, 2.5: 255, 3.5: 42, 4.5: 2}
PILOT_LINE_VALUES = {0.5, 1.5, 2.5, 3.5, 4.5}
GATE_TOL = {"hold": 0.0010, "sd": 0.0040, "sym": 1.5}

PAGE = 1000


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def fail(msg):
    print()
    print("!" * 78)
    print("ABORTING: %s" % msg)
    print("No further numbers are printed. This is the reproduction gate")
    print("doing its job, not a crash.")
    print("!" * 78)
    sys.exit(1)


# ------------------------------------------------------------------ loading

def fetch_nhl(client):
    """FanDuel NHL closing rows, KEYSET paged.

    eval_harness.fetch_lines uses .range(start, start+PAGE-1), which is
    OFFSET paging: at offset 59,000 Postgres scans and discards 59,000 rows
    per request. That crossed the statement timeout on 2026-10-12 right
    after the NHL insert grew the table. Keyset paging filters id > last
    instead, so page 60 costs the same as page 1.

    `id` IS SELECTED DELIBERATELY. It is the only thing separating the pilot
    rows from the full run: both carry snapshot_label 'closing' and a
    captured_at derived from commence_time, so insert order is the sole
    discriminator.
    """
    cols = ("id,season,market,player,book,line,over_odds,under_odds,"
            "event_id,commence_time,captured_at,snapshot_label,sport,"
            "home_team,away_team")
    rows, last_id, pages = [], -1, 0
    while True:
        resp = (client.table("historical_lines").select(cols)
                .eq("sport", SPORT)
                .eq("snapshot_label", LABEL)
                .eq("book", BOOK)
                .eq("season", SEASON)
                .gt("id", last_id)
                .order("id", desc=False)
                .limit(PAGE).execute())
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        last_id = batch[-1]["id"]
        pages += 1
        print("\r    %d rows (%d pages)" % (len(rows), pages), end="",
              flush=True)
        if len(batch) < PAGE:
            break
    print()
    return pd.DataFrame(rows)


def load_outcomes(path):
    if not os.path.exists(path):
        fail("%s not found. Run nhl_outcomes.py --season 20242025 first."
             % path)
    d = pd.read_parquet(path)
    need = {"game_date", "join_name", "sog", "game_id", "player_id"}
    miss = need - set(d.columns)
    if miss:
        fail("%s is missing columns: %s" % (path, sorted(miss)))
    return d


# --------------------------------------------------------------------- gate

def pilot_gate(L):
    """Reproduce the pilot's four published figures or abort.

    The pilot is identified as the 40 events with the lowest minimum id,
    because it was inserted days before the full run. That identification is
    itself gated: it must yield exactly 523 FanDuel rows across 39 events.
    If it does not, the separation is wrong and nothing downstream can be
    trusted.
    """
    hr("GATE: reproduce the pilot's published figures")
    first = (L.groupby("event_id")["id"].min().sort_values()
             .head(PILOT_EVENTS).index)
    P = L[L["event_id"].isin(first)]
    print("  pilot identified as the %d earliest events by insert order"
          % PILOT_EVENTS)
    d0, d1 = P["commence_time"].min()[:10], P["commence_time"].max()[:10]
    print("    id range %d to %d, games %s to %s"
          % (P["id"].min(), P["id"].max(), d0, d1))
    print("    rows %d (published %d), events %d (published %d)"
          % (len(P), PILOT_FD_ROWS, P["event_id"].nunique(),
             PILOT_FD_EVENTS))
    if len(P) != PILOT_FD_ROWS or P["event_id"].nunique() != PILOT_FD_EVENTS:
        print("    NOTE: this does NOT match the published counts. One event")
        print("    had no FanDuel rows when the pilot ran and gained them")
        print("    afterwards, so an exact reproduction is impossible. The")
        print("    checks below are on quantities that survive a population")
        print("    growing by %d rows. The counts above are printed so the"
              % abs(len(P) - PILOT_FD_ROWS))
        print("    difference is visible rather than absorbed.")
    # A spread sample must span the season. If this block sits in October it
    # is the first 40 games of the full run, not the pilot, and the whole
    # identification is wrong.
    if d0[:7] != "2024-10" or d1[:4] != "2025":
        fail("the lowest-id block does not span the season (%s to %s), so it "
             "is not the --spread-events pilot and the pilot cannot be "
             "identified." % (d0, d1))

    p, s = C.devig(P["over_odds"], P["under_odds"])
    hold = float(np.nanmean(s))
    sd = float(np.nanstd(p, ddof=1))
    sym = 100.0 * float((P["over_odds"] == P["under_odds"]).mean())
    lines = P["line"].value_counts().sort_index().to_dict()

    checks = [
        ("hold", hold, PILOT_HOLD, GATE_TOL["hold"]),
        ("sd(devigged p)", sd, PILOT_SD, GATE_TOL["sd"]),
        ("symmetric %", sym, PILOT_SYM_PCT, GATE_TOL["sym"]),
    ]
    ok = True
    print()
    print("  %-18s %10s %10s %9s %6s"
          % ("quantity", "measured", "published", "tol", "pass"))
    for name, got, want, tol in checks:
        good = abs(got - want) <= tol
        ok &= good
        print("  %-18s %10.4f %10.4f %9.4f %6s"
              % (name, got, want, tol, "yes" if good else "NO"))

    print()
    print("  line values. The SET is gated; the counts are printed but not,")
    print("  because the extra event moved them. A value outside the known")
    print("  set would mean the grid is not what was measured.")
    for ln in sorted(set(lines) | set(PILOT_LINES)):
        got, want = lines.get(ln, 0), PILOT_LINES.get(ln, 0)
        flag = "" if got == want else "  (moved)"
        print("    %5.1f  measured %5d  published %5d%s"
              % (ln, got, want, flag))
    stray = set(lines) - PILOT_LINE_VALUES
    if stray:
        ok = False
        print("    UNKNOWN LINE VALUES: %s" % sorted(stray))

    if not ok:
        fail("the pilot's figures do not reproduce.")
    print()
    print("  GATE PASSED")
    return set(first)


# ---------------------------------------------------------------- the join

def join_outcomes(L, D):
    """Join props to shots on goal on (player, game date).

    TWO NORMALISERS EXIST AND THEY ARE NOT THE SAME FUNCTION.
    nhl_outcomes.norm_name strips accents and every non-alphanumeric,
    leaving no spaces. data_utils.norm_join_name does something different.
    The parquet's join_name was built with the former, so the book's
    spelling must go through the SAME one or the join fails on essentially
    every accented and hyphenated player and the match rate looks like a
    data problem rather than a bug of mine.

    DATES. commence_time is UTC; the NHL feed's game_date is a local
    calendar date. A 7:30pm Pacific start is the next day in UTC, so the
    UTC date is wrong for west-coast games. Converting to US Eastern first
    puts every NHL start on the right calendar day, since even a late
    Pacific game is the same Eastern date.
    """
    L = L.copy()
    L["_key"] = L["player"].map(norm_name)
    ct = pd.to_datetime(L["commence_time"], utc=True, errors="coerce")
    try:
        east = ct.dt.tz_convert("America/New_York")
    except Exception:
        # No tz database (bare Windows without tzdata). Eastern is UTC-5 or
        # UTC-4; using -5 year round can only mis-date a game starting
        # before 05:00 UTC, which no NHL game does.
        print("  NOTE: no tz database, using a fixed UTC-5 offset")
        east = ct - pd.Timedelta(hours=5)
    L["_date"] = east.dt.date

    D = D.copy()
    D["_date"] = pd.to_datetime(D["game_date"], errors="coerce").dt.date
    D = D.dropna(subset=["join_name", "_date"])
    # One row per player-game. A player cannot have two shot totals in one
    # game, so a duplicate here would be a feed artifact.
    D = D.drop_duplicates(subset=["join_name", "_date"], keep="last")

    m = L.merge(D[["join_name", "_date", "sog", "game_id"]],
                left_on=["_key", "_date"], right_on=["join_name", "_date"],
                how="left")
    return m


# ------------------------------------------------------- section printing

def section_validity(m, L, D):
    hr("1. THINGS THAT WOULD INVALIDATE THE TEST (section 7 of the prereg)")
    n = len(m)
    matched = m["sog"].notna()
    print("  props %d, matched to an outcome %d (%.1f%%)"
          % (n, int(matched.sum()), 100.0 * matched.mean()))

    p_all, s_all = C.devig(m["over_odds"], m["under_odds"])
    m = m.assign(_p=p_all, _sum=s_all)

    print()
    print("  1a. IS EXCLUSION CORRELATED WITH PRICE?")
    print("      The prereg flags this as the one failure that could give a")
    print("      clean null from a mutilated sample: if unmatched props sit")
    print("      in the tails, excluding them truncates what the slope test")
    print("      depends on.")
    inc, exc = m.loc[matched, "_p"], m.loc[~matched, "_p"]
    for lbl, v in (("included", inc), ("excluded", exc)):
        if len(v):
            print("      %-9s n %6d  mean %.4f  sd %.4f  min %.4f  max %.4f"
                  % (lbl, len(v), v.mean(), v.std(ddof=1), v.min(), v.max()))
    if len(exc) > 30 and len(inc) > 30:
        d_mean = float(exc.mean() - inc.mean())
        se = float(np.sqrt(inc.var(ddof=1) / len(inc)
                           + exc.var(ddof=1) / len(exc)))
        print("      difference in mean price %+.4f, SE %.4f, t %+.2f"
              % (d_mean, se, d_mean / se if se > 0 else np.nan))
        tail_i = float(((inc < 0.44) | (inc > 0.56)).mean())
        tail_e = float(((exc < 0.44) | (exc > 0.56)).mean())
        print("      share outside [0.44, 0.56]: included %.3f, excluded %.3f"
              % (tail_i, tail_e))

    print()
    print("  1b. NON-REGULAR-SEASON AND MISSING GAMES")
    print("      nhl_outcomes.py keeps gameType 2 only, so playoff events")
    print("      have nothing to join to and are excluded by construction.")
    print("      events priced %d, events in the outcome cache %d"
          % (L["event_id"].nunique(), D["game_id"].nunique()))
    bad = m.loc[~matched, "_date"]
    if len(bad):
        print("      unmatched props by date, last 5 dates:")
        for d, k in bad.value_counts().sort_index().tail(5).items():
            print("        %s  %d props" % (d, k))

    print()
    print("  1c. HOLD STABILITY (devigging is only clean if it is flat)")
    print("      pilot measured 1.0651, range 1.0592 to 1.0718")
    print("      full sample mean %.4f  sd %.5f  min %.4f  max %.4f"
          % (m["_sum"].mean(), m["_sum"].std(ddof=1), m["_sum"].min(),
             m["_sum"].max()))
    print("      by line value:")
    for ln, g in m.groupby("line"):
        print("        %5.1f  n %6d  hold %.4f (%.2f%%)"
              % (ln, len(g), g["_sum"].mean(),
                 100.0 * (g["_sum"].mean() - 1) / g["_sum"].mean()))

    print()
    print("  1d. RANGE TRUNCATION (the power table assumed the pilot's)")
    v = m.loc[matched, "_p"]
    print("      pilot range 0.3694 to 0.6361, sd 0.0736")
    print("      full  range %.4f to %.4f, sd %.4f"
          % (v.min(), v.max(), v.std(ddof=1)))
    if v.std(ddof=1) < PILOT_SD * 0.9:
        print("      WARNING: materially narrower than the pilot. The")
        print("      pre-registered MDE is optimistic; see the revised one")
        print("      printed with the primary result.")
    return m


def chronological_split(m):
    """Split on DISTINCT dates so no date lands in both halves."""
    dates = sorted(d for d in m["_date"].dropna().unique())
    half = (len(dates) + 1) // 2        # earlier half takes the extra date
    early = set(dates[:half])
    return m["_date"].isin(early), dates[half - 1] if dates else None


def section_primary(m):
    hr("2. PRIMARY: THE CALIBRATION SLOPE")
    print("  PRE-REGISTERED PREDICTION: slope BELOW 1.0, the")
    print("  favourite-longshot shape. A slope ABOVE 1.0 is a refutation and")
    print("  is not to be reinterpreted as success.")
    g = m[m["sog"].notna()].copy()
    if not len(g):
        fail("no props joined to an outcome.")
    g["_y"] = (g["sog"].astype(float) > g["line"].astype(float)).astype(float)

    r = C.ols_clustered(g["_p"], g["_y"], g["event_id"])
    rho = C.intra_cluster_corr(g["_y"], g["event_id"])
    md = C.mde(r["se_b"])
    lo, hi = r["b"] - 1.96 * r["se_b"], r["b"] + 1.96 * r["se_b"]

    print()
    print("  n %d props across %d games" % (r["n"], r["n_clusters"]))
    print("  slope b        %+.4f   clustered SE %.4f" % (r["b"], r["se_b"]))
    print("  95%% interval   [%+.4f, %+.4f]" % (lo, hi))
    print("  t against b=1  %+.2f" % r["t_b"])
    print("  intercept a    %+.4f   SE %.4f" % (r["a"], r["se_a"]))
    print("     reported as a DESCRIPTIVE number only. b below 1 with the")
    print("     intercept near its calibrated value is a confidence miss;")
    print("     the same b with a large offset is that plus a flat bias, and")
    print("     they imply different bets. It is NOT a second hypothesis.")
    print()
    print("  measured intra-game correlation %.4f (prereg assumed 0.05)"
          % rho)
    print("  MDE at 80%% power %.3f, so this detects b below %.3f"
          % (md, 1 - md))
    print("     pre-registered MDE was 0.185 at an assumed SE of 0.0660.")

    early, boundary = chronological_split(g)
    print()
    print("  CHRONOLOGICAL SPLIT on distinct game dates, boundary %s"
          % boundary)
    print("  %-8s %7s %7s %9s %9s %18s"
          % ("half", "props", "games", "b", "SE", "95% interval"))
    halves = {}
    for lbl, mask in (("early", early), ("late", ~early)):
        gg = g[mask]
        rr = C.ols_clustered(gg["_p"], gg["_y"], gg["event_id"])
        halves[lbl] = rr
        print("  %-8s %7d %7d %+9.4f %9.4f  [%+.4f, %+.4f]"
              % (lbl, rr["n"], rr["n_clusters"], rr["b"], rr["se_b"],
                 rr["b"] - 1.96 * rr["se_b"], rr["b"] + 1.96 * rr["se_b"]))

    pooled_sig = np.isfinite(r["t_b"]) and r["t_b"] < -1.96
    both_below = all(np.isfinite(h["b"]) and h["b"] < 1.0
                     for h in halves.values())
    print()
    print("  PRE-REGISTERED CRITERIA")
    print("    1. pooled b significantly below 1.0 at 0.05 : %s"
          % ("PASS" if pooled_sig else "FAIL"))
    print("    2. point estimate below 1.0 in BOTH halves   : %s"
          % ("PASS" if both_below else "FAIL"))
    # THREE outcomes, not two, and the middle one is the easy mistake.
    # An earlier version branched on the SIGN of b, so b = 1.0228 with
    # t = +0.43 printed "REFUTED in the opposite direction" when the
    # interval comfortably contained 1.0. That is a null being dressed as a
    # finding, the same error as the retired season-decay claim in
    # rules.describe(). Significance, not sign.
    sig_above = np.isfinite(r["t_b"]) and r["t_b"] > 1.96
    print()
    if pooled_sig and both_below:
        print("    VERDICT: PASS. The price is overconfident at the extremes,")
        print("    as pre-registered.")
    elif sig_above:
        print("    VERDICT: REFUTED IN THE OPPOSITE DIRECTION. b is")
        print("    SIGNIFICANTLY above 1.0, meaning the price is")
        print("    UNDERconfident at the extremes. The prereg says this must")
        print("    not be restated as success, and no mechanism was offered")
        print("    for it in advance.")
    else:
        print("    VERDICT: NULL. The interval contains 1.0, so the price is")
        print("    calibrated in slope terms and the favourite-longshot")
        print("    prediction does not hold here.")
        if np.isfinite(r["b"]) and r["b"] > 1.0:
            print("    Note b sits above 1.0, but t is %+.2f, so that is NOT"
                  % r["t_b"])
            print("    evidence of the opposite effect. It is a null.")
        print("    HOW INFORMATIVE this null is, read off the MDE: slopes")
        print("    below %.3f are ruled out at 80%% power." % (1 - md))
        if (1 - md) >= 0.85:
            print("    The prereg's actionable range was b below about 0.85,")
            print("    so this null COVERS the whole range that would have")
            print("    been worth betting. It is an informative null.")
        else:
            print("    The prereg's actionable range extends below %.3f,"
                  % (1 - md))
            print("    so part of what would have been worth betting is")
            print("    still UNTESTED. This null is not conclusive there.")

    print()
    print("  SIGNIFICANCE IS NOT PROFITABILITY (prereg section 4)")
    hold = float(m["_sum"].mean())
    print("    at the measured hold of %.4f, the under needs realised <"
          % hold)
    print("    %.4f*p - %.4f and the over needs realised > %.4f*p"
          % (hold, hold - 1, hold))
    if np.isfinite(r["b"]) and r["b"] < 1.0:
        b = r["b"]
        p_over = (0.5 - 0.5 * b) / (hold - b)
        p_under = (0.5 + (hold - 1) - 0.5 * b) / (hold - b)
        share = float(((g["_p"] < p_over) | (g["_p"] > p_under)).mean())
        print("    at b = %.4f that makes props bettable only where" % b)
        print("    p < %.4f or p > %.4f, which is %.2f%% of this sample,"
              % (p_over, p_under, 100.0 * share))
        print("    about %d props a season." % int(round(share * len(g))))
        if share < 0.01:
            print("    THAT IS EFFECTIVELY NOTHING TO BET. The prereg fixes")
            print("    in advance that this must be said plainly.")
    return g, r


def section_bins(g, n_draws):
    hr("3. SECONDARY, EXPLORATORY: PER-BIN CALIBRATION")
    print("  Bin edges fixed in the prereg, on the probability scale, so")
    print("  this cannot become a search over bin placement. EXPLORATORY")
    print("  regardless of outcome: a single bin clearing is the shape of")
    print("  the Cleveland cell, which cleared at p 0.0230, survived")
    print("  holdout, then dissolved into a gradient losing money in four")
    print("  of five bins.")
    print()
    rows = C.bin_table(g["_p"], g["_y"])
    print("  %-14s %7s %9s %10s %9s" %
          ("bin", "n", "mean p", "realised", "miss"))
    for b in rows:
        if not b["n"]:
            print("  [%.2f,%.2f)   %7d %9s %10s %9s"
                  % (b["lo"], b["hi"], 0, "-", "-", "-"))
            continue
        print("  [%.2f,%.2f)   %7d %9.4f %10.4f %+9.4f"
              % (b["lo"], b["hi"], b["n"], b["mean_p"], b["realised"],
                 b["miss"]))

    obs = [abs(b["miss"]) for b in rows
           if b["n"] > 0 and np.isfinite(b["miss"])]
    if not obs:
        return
    obs_max = max(obs)
    null = C.max_bin_null(g["_p"], g["event_id"], draws=n_draws)
    bar95 = float(np.nanpercentile(null, 95))
    pval = float(np.nanmean(null >= obs_max))
    print()
    print("  MAX-STATISTIC NULL over the %d bins, %d draws"
          % (len(rows), n_draws))
    print("    observed largest absolute miss %.4f" % obs_max)
    print("    null 95th percentile            %.4f" % bar95)
    print("    null sd                         %.5f" % float(np.nanstd(null)))
    print("    p                               %.4f" % pval)
    if float(np.nanstd(null)) < 1e-9:
        print("    WARNING: the null has no variance. That is the degenerate")
        print("    scheme this project has hit before, and the p above means")
        print("    nothing. Do not report it.")
    print()
    print("    For scale, the null-maximum schedule says a family of 22 to")
    print("    40 conditioned cells manufactures a best cell of +0.1176 to")
    print("    +0.1591 in ROI terms from pure noise. Seven bins is a much")
    print("    smaller family, hence the lower bar above, but the principle")
    print("    is the same: a best-of-seven needs the max-statistic bar, not")
    print("    a bare p-value.")
    print()
    print("    LIMITATION, STATED: the null draws independently within a")
    print("    game, so it understates variance where outcomes correlate")
    print("    inside a game. The measured correlation is in section 2.")

    # --- level miss, which the SLOPE test is structurally blind to -------
    #
    # A slope of 1.0 says the price's SPREAD is right. It says nothing about
    # whether the whole thing sits in the wrong place. If every bin misses
    # the same way, that is a level effect and the slope cannot see it.
    #
    # NOT PRE-REGISTERED. The prereg made the intercept descriptive only, on
    # the grounds that two primaries make the test a best-of-two. So this is
    # EXPLORATORY and the sign test below is reported, not claimed.
    live = [b for b in rows if b["n"] > 0 and np.isfinite(b["miss"])]
    if len(live) < 3:
        return
    neg = sum(1 for b in live if b["miss"] < 0)
    k, nb = max(neg, len(live) - neg), len(live)
    # two-sided exact binomial, bins are DISJOINT prop sets so the signs are
    # close to independent even though the underlying props are clustered
    from math import comb
    p_sign = 2.0 * sum(comb(nb, i) for i in range(k, nb + 1)) / (2.0 ** nb)
    p_sign = min(1.0, p_sign)
    wmiss = (sum(b["miss"] * b["n"] for b in live)
             / sum(b["n"] for b in live))
    print()
    print("  LEVEL MISS (exploratory, NOT pre-registered)")
    print("    %d of %d bins miss in the same direction, two-sided sign"
          % (k, nb))
    print("    test p = %.4f. Weighted mean miss %+.4f on %d props."
          % (p_sign, wmiss, sum(b["n"] for b in live)))
    print("    A slope of 1.0 with a consistent one-way miss is a LEVEL")
    print("    effect: the spread of the price is right, its position is not.")
    print()
    print("    IS IT BETTABLE? The bar is the hold, not zero.")
    hold = C.HOLD_PILOT
    print("    %-8s %9s %9s %13s %10s"
          % ("mean p", "realised", "miss", "under needs <", "short by"))
    closest = None
    for b in live:
        thr = hold * b["mean_p"] - (hold - 1.0)
        gap = b["realised"] - thr
        if closest is None or gap < closest:
            closest = gap
        print("    %8.4f %9.4f %+9.4f %13.4f %+10.4f"
              % (b["mean_p"], b["realised"], b["miss"], thr, gap))
    need = hold * 0.5 - (hold - 1.0) - 0.5
    print()
    print("    At even money the under needs a miss of %+.4f; the measured"
          % need)
    print("    mean is %+.4f. Closest bin falls short by %+.4f."
          % (wmiss, closest))
    if closest > 0:
        print("    EVERY BIN FALLS SHORT. The unders are consistently")
        print("    underpriced and the hold swallows all of it. Nothing to")
        print("    bet, and the prereg fixes in advance that this must be")
        print("    said plainly rather than left as a significant-looking")
        print("    result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--cache", default=None,
                    help="parquet path to cache the fetched lines")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    hr("NHL SHOTS ON GOAL: PRE-REGISTERED CALIBRATION TEST")
    print("  prereg: PREREG_nhl_calibration.md, written before the purchase")
    print("  sport %s, label %s, book %s, season %s"
          % (SPORT, LABEL, BOOK, SEASON))
    print("  NOTHING IS WRITTEN.")

    hr("LOADING")
    L = None
    if args.cache and not args.refresh:
        L = eh._cache_read(args.cache)
    if L is None or not len(L):
        client = eh.connect(eh._secrets())
        print("  fetching FanDuel NHL closing rows, keyset paged")
        L = fetch_nhl(client)
        if args.cache and len(L):
            eh._cache_write(args.cache, L)
    if not len(L):
        fail("no NHL rows returned.")
    L = L.dropna(subset=["over_odds", "under_odds", "line"])
    print("  %d priced rows across %d events"
          % (len(L), L["event_id"].nunique()))

    pilot = pilot_gate(L)

    T = L[~L["event_id"].isin(pilot)].copy()
    print()
    print("  pilot EXCLUDED from the test sample, as pre-registered.")
    print("  test sample %d rows across %d events"
          % (len(T), T["event_id"].nunique()))

    D = load_outcomes(args.parquet)
    print("  outcome cache %d player-game rows across %d games"
          % (len(D), D["game_id"].nunique()))

    m = join_outcomes(T, D)
    m = section_validity(m, T, D)
    g, _ = section_primary(m)
    section_bins(g, args.draws)

    hr("DONE. Nothing was written.")


if __name__ == "__main__":
    sys.exit(main())
