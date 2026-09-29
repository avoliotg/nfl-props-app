"""movement.py - per-book price movement for the Line Movement tab.

WHY THIS IS SEPARATE FROM db.get_line_movement

    get_line_movement measures movement in the LINE at ONE book, chosen by
    _select_book. Two things are wrong with that as the only measure.

    1. It misses most of the movement in receptions. Measured on the
       automated era, receptions moves its LINE on 8.7 percent of paths and
       its PRICE on 82.5 percent, with a median absolute price move of
       0.0178 against 0.0042 for receiving and rushing. FanDuel moves
       reception prices through the ODDS, so a line-based measure is blind
       in the market where movement is largest.

    2. It uses one book when eight are captured.

    So this module measures movement in DEVIGGED TWO-SIDED PROBABILITY, per
    book, and returns one row per player for joining back.

THE DEFECT THIS MODULE MUST NOT REPRODUCE

    get_line_movement's docstring records that before 2026-09-24 it treated
    every ROW as a snapshot, so `Move` differenced the first row against the
    last row across interleaved books. Within one captured_at the book order
    is arbitrary, so the sign of Move was arbitrary: it measured BOOK SPREAD
    rather than movement over time.

    Comparing books is exactly what this module does, so the same failure is
    one mistake away. The difference is that each book is differenced
    AGAINST ITSELF over a common window, and only then are those differences
    compared. That is a real quantity where the old one was noise.

    The protection is structural, not careful reading: every movement here
    comes from a groupby on (player, book) and nothing is ever derived from
    row order. n_books_window is returned per row for the same reason the
    original fix returned n_books, so the defect cannot recur invisibly.

THE WINDOW

    Per prop, maximal rather than a fixed clock, because the goal is to
    capture as much of the move as possible. The anchor is the time by which
    BOOK_QUORUM of that prop's books have a price; each book's movement runs
    from its first capture at or after the anchor to its last. A book that
    arrives after the anchor is excluded from that prop rather than shown as
    flat, so a late book cannot truncate the window for the other seven.

    Consequence, accepted deliberately: windows differ between props, so one
    row's bars may span 70 hours and another's 14. window_hours is returned
    so the display can show it. The primary sort is on minority share, which
    is scale-free and unaffected.

THE DEADBAND

    A book counts as having MOVED only if its move exceeds a threshold,
    because books quantize prices and a four-four sign split at plus or
    minus 0.001 is rounding rather than disagreement. The threshold is
    scaled to each market's own median absolute move, since a fixed value
    would erase nearly all qb_passing movement (median 0.0021) while barely
    touching receptions (median 0.0178).

SIGN

    Positive means toward the OVER, everywhere, without exception.
"""
import numpy as np
import pandas as pd

# Fraction of a prop's books that must have a price at the window anchor.
BOOK_QUORUM = 0.75

# Retained for reference: superseded by the step-function window, which needs a
# price at or before the anchor and a later one, not two captures inside it.
MIN_CAPTURES = 2

# Minimum books with a qualifying move before minority share is meaningful.
# Below this a single book flips the ratio, so it is reported as NaN rather
# than as a large disagreement.
MIN_BOOKS_FOR_MINORITY = 4

# Deadband as a fraction of the market's median absolute move, with a floor
# so a market that barely moves at all does not get a near-zero threshold.
DEADBAND_FRAC = 0.25
DEADBAND_FLOOR = 0.002

# Display clip for the bar group, in probability points. Bars are comparable
# across rows only if the scale is shared, so it is fixed here rather than
# per row.
BAR_CLIP = 0.05


def american_to_decimal(o):
    o = pd.to_numeric(o, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 0, 1.0 + o / 100.0,
                        np.where(o < 0, 1.0 + 100.0 / (-o), np.nan))


def devig(over_odds, under_odds):
    """Two-sided devigged P(over). NaN unless BOTH sides are present.

    One-sided implied probability is not usable here: it carries the hold,
    so it would move when the book widens its margin without changing its
    opinion. anytime_td has no under side and therefore no value in this
    module at all, which is correct rather than a gap.
    """
    do = american_to_decimal(over_odds)
    du = american_to_decimal(under_odds)
    with np.errstate(divide="ignore", invalid="ignore"):
        ro = 1.0 / do
        ru = 1.0 / du
    tot = ro + ru
    return np.where(np.isfinite(tot) & (tot > 0), ro / tot, np.nan)


def _prep(df):
    d = df.copy()
    for c in ("player", "book", "captured_at"):
        if c not in d.columns:
            raise KeyError("movement needs a %r column" % c)
    d["captured_at"] = pd.to_datetime(d["captured_at"], utc=True,
                                      errors="coerce", format="mixed")
    d["p_over"] = devig(d.get("over_odds"), d.get("under_odds"))
    d = d[d["captured_at"].notna() & d["p_over"].notna()].copy()
    return d


def _deadband(d):
    """Market-scaled deadband, from the spread of this frame's own moves."""
    per = (d.sort_values("captured_at")
            .groupby(["player", "book"])["p_over"]
            .agg(lambda s: s.iloc[-1] - s.iloc[0]))
    if per.empty:
        return DEADBAND_FLOOR
    med = float(np.nanmedian(np.abs(per.to_numpy())))
    if not np.isfinite(med):
        return DEADBAND_FLOOR
    return max(DEADBAND_FLOOR, DEADBAND_FRAC * med)


def per_book_movement(df, book_order=None):
    """One row per player. Movement in devigged probability, per book.

    `df` must be the ALL-BOOKS frame, before _select_book. Returns columns:

        player              as given, first spelling seen
        price_move          mean move across participating books, pts of
                            probability, positive toward the over
        n_books_window      books with >= MIN_CAPTURES inside the window
        n_moved             of those, how many cleared the deadband
        n_up, n_down        direction split among those that moved
        minority_share      min(n_up, n_down) / n_moved, NaN below
                            MIN_BOOKS_FOR_MINORITY
        window_hours        span of the window for this prop
        bars                list aligned to book_order, NaN where a book did
                            not participate, clipped to +/- BAR_CLIP
        book_order          the order `bars` is aligned to, same every row
    """
    d = _prep(df)
    if d.empty:
        return pd.DataFrame(), []

    order = (list(book_order) if book_order is not None
             else sorted(d["book"].dropna().unique().tolist()))
    band = _deadband(d)

    out = []
    for player, grp in d.groupby("player", sort=False):
        # Window anchor: the time by which BOOK_QUORUM of this prop's books
        # have a price. Computed from per-book FIRST capture times, never
        # from row order.
        firsts = grp.groupby("book")["captured_at"].min().sort_values()
        if firsts.empty:
            continue
        k = int(np.ceil(BOOK_QUORUM * len(firsts))) - 1
        k = min(max(k, 0), len(firsts) - 1)
        anchor = firsts.iloc[k]

        moves, spans = {}, []
        for bk, bg in grp.groupby("book"):
            bg = bg.sort_values("captured_at")
            # A PRICE IS A STEP FUNCTION. A book's price at the anchor is its
            # most recent capture at or before the anchor, not a capture
            # exactly at it: the price posted at hour 24 is still the price at
            # hour 48 if nothing was recaptured in between. An earlier version
            # required two captures strictly inside the window and therefore
            # dropped a book priced at hour 24 and recaptured at hour 72 from
            # a window starting at 48, despite it plainly having a live price
            # throughout.
            at_or_before = bg[bg["captured_at"] <= anchor]
            after = bg[bg["captured_at"] > anchor]
            # Needs a price to start from and a later one to compare against.
            # A book with no capture at or before the anchor had no price in
            # this window at all and is excluded rather than shown as flat,
            # since a book that has had no time to move is not a holdout.
            if at_or_before.empty or after.empty:
                continue
            start = at_or_before.iloc[-1]
            end = after.iloc[-1]
            moves[bk] = float(end["p_over"] - start["p_over"])
            spans.append((end["captured_at"]
                          - start["captured_at"]).total_seconds())

        if not moves:
            continue

        vals = np.array(list(moves.values()), dtype=float)
        qual = vals[np.abs(vals) > band]
        n_up = int((qual > 0).sum())
        n_down = int((qual < 0).sum())
        n_moved = n_up + n_down
        minority = (min(n_up, n_down) / n_moved
                    if n_moved >= MIN_BOOKS_FOR_MINORITY else np.nan)

        bars = [float(np.clip(moves[b], -BAR_CLIP, BAR_CLIP))
                if b in moves else np.nan for b in order]

        out.append({
            "player": grp["player"].iloc[0],
            "price_move": float(vals.mean()),
            "n_books_window": len(moves),
            "n_moved": n_moved,
            "n_up": n_up,
            "n_down": n_down,
            "minority_share": minority,
            "window_hours": (max(spans) / 3600.0) if spans else np.nan,
            "bars": bars,
        })

    res = pd.DataFrame(out)
    return res, order


def sort_key(res):
    """Layered sort: book disagreement first, then size of the move.

    Unanimous props all tie at 0 on minority_share and then order by the
    absolute consensus move, which is the intended behaviour. Props below
    MIN_BOOKS_FOR_MINORITY have NaN minority_share and sort last on that
    key rather than first, since a two-book split is not disagreement.
    """
    if res.empty:
        return res
    r = res.copy()
    r["_m"] = r["minority_share"].fillna(-1.0)
    r["_a"] = r["price_move"].abs()
    return (r.sort_values(["_m", "_a"], ascending=[False, False])
             .drop(columns=["_m", "_a"]))
