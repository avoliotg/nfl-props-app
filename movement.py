"""movement.py - per-book market movement, reported in BOTH channels.

WHAT A BOOK CAN DO, AND WHY ONE NUMBER WILL NOT DESCRIBE IT

    A book moves a prop two ways: change the number, or change the odds at
    the same number. Measured on the automated era, the markets split almost
    cleanly between the two habits:

        market       line moved   median |price move|
        receptions         8.7%              0.0178
        receiving         70.1%              0.0042
        rushing           68.9%              0.0042
        qb_passing        82.6%              0.0021

    receptions barely touches its line; the yardage markets move the line
    and reset the odds toward even. So a LINE measure is blind in receptions
    and a PRICE measure is blind in the other three.

    This module therefore reports BOTH, per book, and never combines them.

WHY NOT COMBINE THEM: A FAILED ATTEMPT WORTH RECORDING

    The obvious fix was to convert price into stat units and report one
    number, the change in the market's implied median:

        implied median = line + sigma * z(p_over) - alpha

    It does not work in the yardage markets, and the reason is a fact about
    the books rather than a tuning problem.

    First attempt used sigma from mc_pricing.sigma_for. That is the spread
    of ACTUAL OUTCOMES around a projection, and it gave a 0.50 -> 0.55
    reprice a worth of 4.0 yards in receiving and 10.2 in qb_passing. The
    odds channel then swamped the line channel: 181 of 191 receiving props
    "moved" and a third of them "disagreed". Noise, ranked.

    Second attempt measured the exchange rate from the market itself, as the
    cross-book slope of devigged price on line within one prop at one
    instant, restricted to books near consensus. That failed harder, and
    the raw board shows why. Dak Prescott, six books, one snapshot:

        fanatics     260.5   -115/-115   0.5000
        fanduel      261.5   -113/-113   0.5000
        betonlineag  263.5   -116/-111   0.5052
        betmgm       264.5   -115/-115   0.5000
        betrivers    268.5   -114/-117   0.4970
        draftkings   270.5   -111/-113   0.4979

    Ten yards of line spread at essentially ZERO price spread. The books are
    not disagreeing about the median at a shared price; each one parks at
    the middle of its own number and charges the vig. So there is no
    exchange rate to recover: dividing by a near-zero slope returned an
    implied dispersion of 1,460 yards and an implied-median shift of 183
    yards per five points of price.

    receptions escapes this because half a reception is a large fraction of
    the distribution, so a book cannot hold its own number at even money and
    the price has to move. That is exactly why receptions was the one market
    where the two independent estimates agreed, 2.26 against sigma_for's
    2.19.

    CONCLUSION, so nobody retries it: the yardage books do not reveal a
    line-to-price exchange rate, and any single-number movement measure
    built on one is fabricating the conversion.

THE DEFECT THIS MODULE MUST NOT REPRODUCE

    db.get_line_movement's docstring records that before 2026-09-24 it
    differenced the first ROW against the last ROW across interleaved books.
    Within one captured_at the book order is arbitrary, so the sign of the
    result was arbitrary: it measured BOOK SPREAD rather than movement.

    Comparing books is what this module does, so the protection is
    structural: every figure comes from a groupby on (player, book), each
    book is differenced against ITSELF, and nothing is derived from row
    order. n_books_window is returned per row for the same reason the
    original fix returned n_books.

THE WINDOW

    Per prop and maximal, to capture as much of the move as possible. The
    anchor is the time by which BOOK_QUORUM of that prop's books have a
    price. A book participates if it had a price at or before the anchor and
    a later one, so a book arriving late is excluded rather than shown as
    flat and cannot truncate the window for the others. Prices are STEP
    FUNCTIONS: a book's price at the anchor is its most recent capture at or
    before it.

SIGN

    Positive means toward the OVER in both channels. For the line that means
    the number went UP; for the price it means P(over) went up.
"""
import numpy as np
import pandas as pd

BOOK_QUORUM = 0.75
MIN_BOOKS_FOR_MINORITY = 4

# A line sits on a half-point grid, so any real change is at least 0.5 in
# the yardage markets and at least 0.5 in receptions. Anything smaller is a
# rounding artifact rather than a move.
LINE_DEADBAND = 0.25

# The price deadband scales to the market's own median absolute move, since
# books quantise odds and a four-four split at plus or minus 0.001 is
# rounding rather than disagreement.
PRICE_DEADBAND_FRAC = 0.25
PRICE_DEADBAND_FLOOR = 0.002


def american_to_decimal(o):
    o = pd.to_numeric(o, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 0, 1.0 + o / 100.0,
                        np.where(o < 0, 1.0 + 100.0 / (-o), np.nan))


def devig(over_odds, under_odds):
    """Two-sided devigged P(over). NaN unless BOTH sides are present.

    One-sided implied probability carries the hold, so it would move when a
    book widens its margin without changing its opinion. anytime_td has no
    under side and therefore no price channel here, which is correct rather
    than a gap.
    """
    do = american_to_decimal(over_odds)
    du = american_to_decimal(under_odds)
    with np.errstate(divide="ignore", invalid="ignore"):
        ro, ru = 1.0 / do, 1.0 / du
    tot = ro + ru
    return np.where(np.isfinite(tot) & (tot > 0), ro / tot, np.nan)


def _prep(df):
    d = df.copy()
    for c in ("player", "book", "captured_at", "line"):
        if c not in d.columns:
            raise KeyError("movement needs a %r column" % c)
    d["captured_at"] = pd.to_datetime(d["captured_at"], utc=True,
                                      errors="coerce", format="mixed")
    d["line"] = pd.to_numeric(d["line"], errors="coerce")
    d["p_over"] = devig(d.get("over_odds"), d.get("under_odds"))
    # A row needs a time and a line. The price may be missing: a market with
    # no under side still has a usable line channel.
    return d[d["captured_at"].notna() & d["line"].notna()].copy()


def per_book_movement(df, market=None, book_order=None):
    """One row per player, with both movement channels kept separate.

    Returns (frame, book_order, info).

    frame columns:
        player
        line_move       mean change in the line, in stat units
        price_move      mean change in devigged P(over)
        n_books_window  books that had a price at the anchor and a later one
        n_moved_line, n_up_line, n_down_line
        n_moved_price, n_up_price, n_down_price
        minority_share  computed on the DOMINANT channel for this market
        window_hours
        bars_line, bars_price   per-book, aligned to book_order, NaN absent

    info carries the dominant channel, both deadbands, both bar clips, and
    the share of books that moved in each channel, so the display can label
    its own scale and pick the bar group rather than assume either.
    """
    info = {"market": market, "dominant": None,
            "line_deadband": LINE_DEADBAND, "price_deadband": None,
            "clip_line": None, "clip_price": None,
            "frac_line": None, "frac_price": None, "n_props": 0}
    try:
        d = _prep(df)
    except KeyError:
        return pd.DataFrame(), [], info
    if d.empty:
        return pd.DataFrame(), [], info

    order = (list(book_order) if book_order is not None
             else sorted(d["book"].dropna().unique().tolist()))

    # Pass one: per-book endpoints in each channel.
    raw = {}
    for player, grp in d.groupby("player", sort=False):
        firsts = grp.groupby("book")["captured_at"].min().sort_values()
        if firsts.empty:
            continue
        k = min(max(int(np.ceil(BOOK_QUORUM * len(firsts))) - 1, 0),
                len(firsts) - 1)
        anchor = firsts.iloc[k]

        dl, dp, spans = {}, {}, []
        for bk, bg in grp.groupby("book"):
            bg = bg.sort_values("captured_at")
            before = bg[bg["captured_at"] <= anchor]
            after = bg[bg["captured_at"] > anchor]
            if before.empty or after.empty:
                continue
            s_row, e_row = before.iloc[-1], after.iloc[-1]
            dl[bk] = float(e_row["line"]) - float(s_row["line"])
            if np.isfinite(s_row["p_over"]) and np.isfinite(e_row["p_over"]):
                dp[bk] = float(e_row["p_over"]) - float(s_row["p_over"])
            spans.append((e_row["captured_at"]
                          - s_row["captured_at"]).total_seconds())
        if dl:
            raw[grp["player"].iloc[0]] = (dl, dp, spans)

    if not raw:
        return pd.DataFrame(), order, info

    # Pass two: deadbands, clips and the dominant channel are properties of
    # the whole market, so they need every prop before any row is finalised.
    all_dl = np.abs(np.array([v for dl, _, _ in raw.values()
                              for v in dl.values()], dtype=float))
    all_dp = np.abs(np.array([v for _, dp, _ in raw.values()
                              for v in dp.values()], dtype=float))

    p_band = PRICE_DEADBAND_FLOOR
    if len(all_dp):
        med = float(np.nanmedian(all_dp))
        if np.isfinite(med):
            p_band = max(PRICE_DEADBAND_FLOOR, PRICE_DEADBAND_FRAC * med)

    frac_line = float((all_dl > LINE_DEADBAND).mean()) if len(all_dl) else 0.0
    frac_price = float((all_dp > p_band).mean()) if len(all_dp) else 0.0
    # DOMINANT CHANNEL, chosen from the data rather than hardcoded per
    # market. Whichever channel more books actually use is the one the bar
    # group should show, because a bar group built on the idle channel would
    # read as unanimous stillness.
    dominant = "line" if frac_line >= frac_price else "price"

    clip_line = (max(float(np.nanpercentile(all_dl, 95)), LINE_DEADBAND * 2)
                 if len(all_dl) else LINE_DEADBAND * 2)
    clip_price = (max(float(np.nanpercentile(all_dp, 95)), p_band * 2)
                  if len(all_dp) else p_band * 2)

    info.update({"dominant": dominant, "price_deadband": p_band,
                 "clip_line": clip_line, "clip_price": clip_price,
                 "frac_line": frac_line, "frac_price": frac_price,
                 "n_props": len(raw)})

    def split(vals, band):
        v = np.array(list(vals), dtype=float)
        q = v[np.abs(v) > band]
        return int((q > 0).sum()), int((q < 0).sum())

    out = []
    for player, (dl, dp, spans) in raw.items():
        up_l, dn_l = split(dl.values(), LINE_DEADBAND)
        up_p, dn_p = split(dp.values(), p_band) if dp else (0, 0)
        moved_l, moved_p = up_l + dn_l, up_p + dn_p
        if dominant == "line":
            up, dn, moved = up_l, dn_l, moved_l
        else:
            up, dn, moved = up_p, dn_p, moved_p
        out.append({
            "player": player,
            "line_move": float(np.mean(list(dl.values()))) if dl else None,
            "price_move": float(np.mean(list(dp.values()))) if dp else None,
            "n_books_window": len(dl),
            "n_moved_line": moved_l, "n_up_line": up_l, "n_down_line": dn_l,
            "n_moved_price": moved_p, "n_up_price": up_p,
            "n_down_price": dn_p,
            "minority_share": (min(up, dn) / moved
                               if moved >= MIN_BOOKS_FOR_MINORITY
                               else np.nan),
            "window_hours": (max(spans) / 3600.0) if spans else np.nan,
            "bars_line": [float(np.clip(dl[b], -clip_line, clip_line))
                          if b in dl else np.nan for b in order],
            "bars_price": [float(np.clip(dp[b], -clip_price, clip_price))
                           if b in dp else np.nan for b in order],
        })
    return pd.DataFrame(out), order, info


def sort_key(res, dominant="line"):
    """Layered: book disagreement first, then the size of the move.

    Unanimous props tie at 0 on minority_share and then order by absolute
    move in the dominant channel. Props below MIN_BOOKS_FOR_MINORITY have
    NaN and sort last on that key rather than first, since a two-book split
    is not disagreement.
    """
    if res.empty:
        return res
    col = "line_move" if dominant == "line" else "price_move"
    r = res.copy()
    r["_m"] = r["minority_share"].fillna(-1.0)
    r["_a"] = pd.to_numeric(r[col], errors="coerce").abs().fillna(-1.0)
    return (r.sort_values(["_m", "_a"], ascending=[False, False])
             .drop(columns=["_m", "_a"]))
