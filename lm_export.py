"""
OpalScales - render the Line Movement table as PNG or PDF.

DESIGN NOTES
  1. NO EMOJI. Matplotlib's bundled fonts have no emoji glyphs, so a crystal
     ball or a tier icon renders as an empty box. Worse, the fonts that DO
     carry emoji differ between Windows and Streamlit Cloud's Linux, so it
     would look right locally and break on deploy. Cells are stripped to
     ASCII ("Max" out of the emoji tier label). The crystal ball in the header
     is DRAWN with shapes, which renders identically everywhere.
  2. THE TREND COLUMN is a list per row, for the sparkline. As table text it
     is noise, so it is dropped.
  3. HEADER AND FOOTER are positioned in INCHES from the figure edge, not axes
     fractions. Axes-fraction placement collided with the title whenever the
     row count changed, because a title's pad is in points while an axes
     offset scales with table height.
  4. ROW BANDING follows the Game column when present, so a multi-game export
     reads as blocks per matchup rather than stripes every other row.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")            # no display needed, must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

DROP_COLS = ("Trend", "Bet?")

BRAND = "#e0873a"                # the app's heading orange (wordmark)
ORB = "#8b5fc7"                  # crystal ball purple
ORB_EDGE = "#5f3d94"
ORB_HILITE = "#e2d2f5"
HEADER_BG = "#2b2b3d"
HEADER_FG = "white"
BAND_ROW = "#ececf4"
GRID = "#dcdce4"
MUTED = "#6b6b76"

TIER_COLORS = {"Pass": "#faf7ee", "Lean": "#fdf0c2",
               "Strong": "#f6d574", "Max": "#e8ab24"}
TOWARD_COLORS = {"toward": "#d6f5d6", "away": "#ffd6cc", "flat": "#f0f0f0"}
SIDE_FG = {"OVER": "#15723f", "UNDER": "#a82f24"}

DISCLAIMER = (
    "Projections from OpalScales, built on nflverse data 2022-2025. Edge = model "
    "probability minus the vig-adjusted breakeven, in points. Bigger edge means "
    "more model conviction, NOT a safer bet. Past results do not guarantee future "
    "results. One tool for research, not advice."
)


def _ascii(v):
    """Strip non-ASCII so emoji do not render as boxes. Keeps the word."""
    if v is None:
        return ""
    return "".join(ch for ch in str(v) if ord(ch) < 128).strip()


def _prep(df):
    d = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda v: "" if v != v else f"{v:g}")
        else:
            d[c] = d[c].map(_ascii)
    d.columns = [_ascii(c) for c in d.columns]
    return d


def _stamp():
    """Local time, Eastern when tz data is available."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        suffix = " ET"
    except Exception:
        now = datetime.now()
        suffix = ""
    s = now.strftime("%b %d, %I:%M %p")
    return s.replace(" 0", " ").replace("AM", "am").replace("PM", "pm") + suffix


def _crystal_ball(fig, x_in, y_in, size_in):
    """Crystal ball drawn from shapes, so no emoji font is needed. Ellipses in
    figure coords keep it circular regardless of the figure's aspect ratio."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    cx, cy = x_in / W, y_in / H
    rx, ry = (size_in / 2) / W, (size_in / 2) / H
    fig.patches.append(Polygon(
        [[cx - rx * 0.8, cy - ry * 1.35], [cx + rx * 0.8, cy - ry * 1.35],
         [cx + rx * 0.5, cy - ry * 0.7], [cx - rx * 0.5, cy - ry * 0.7]],
        closed=True, transform=fig.transFigure, facecolor="#4a3a63",
        edgecolor="none", zorder=3))
    fig.patches.append(Ellipse(
        (cx, cy), width=rx * 2, height=ry * 2, transform=fig.transFigure,
        facecolor=ORB, edgecolor=ORB_EDGE, linewidth=1.1, zorder=4))
    fig.patches.append(Ellipse(
        (cx - rx * 0.33, cy + ry * 0.35), width=rx * 0.55, height=ry * 0.55,
        transform=fig.transFigure, facecolor=ORB_HILITE, edgecolor="none",
        alpha=0.95, zorder=5))


def _legend(fig, y_in, x_in=0.10):
    """Tier legend chips, so a cold reader knows what Max means."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    x = x_in
    fig.text(x / W, y_in / H, "Tier:", fontsize=8, color=MUTED,
             ha="left", va="center")
    x += 0.40
    for tier in ("Pass", "Lean", "Strong", "Max"):
        w, h = 0.58, 0.17
        fig.patches.append(Rectangle(
            (x / W, (y_in - h / 2) / H), w / W, h / H,
            transform=fig.transFigure, facecolor=TIER_COLORS[tier],
            edgecolor=GRID, linewidth=0.5, zorder=3))
        fig.text((x + w / 2) / W, y_in / H, tier, fontsize=7.5,
                 ha="center", va="center", zorder=4)
        x += w + 0.07
    fig.text((x + 0.10) / W, y_in / H,
             "vs. Model: green = line moved toward the model, red = away",
             fontsize=7.5, color=MUTED, ha="left", va="center")


def _draw(fig, ax, d, market, season, week):
    ax.axis("off")
    W, H = fig.get_figwidth(), fig.get_figheight()
    head_in, foot_in = 1.45, 0.70
    fig.subplots_adjust(top=1.0 - head_in / H, bottom=foot_in / H,
                        left=0.004, right=0.996)

    _crystal_ball(fig, 0.34, H - 0.36, 0.36)
    fig.text(0.60 / W, (H - 0.33) / H, "OpalScales", fontsize=17,
             fontweight="bold", color=BRAND, ha="left", va="center")
    fig.text(2.22 / W, (H - 0.34) / H, f"{_ascii(market)} line movement",
             fontsize=13, color="#1c1c22", ha="left", va="center")

    tiers = list(d["Tier"]) if "Tier" in d.columns else []
    sub = (f"{season} Week {week}   |   {len(d)} rows, "
           f"{sum(1 for t in tiers if t == 'Max')} Max, "
           f"{sum(1 for t in tiers if t == 'Strong')} Strong"
           f"   |   generated {_stamp()}")
    fig.text(0.60 / W, (H - 0.68) / H, sub, fontsize=9, color=MUTED,
             ha="left", va="center")
    _legend(fig, H - 1.02)

    # content-proportional widths, or long player names truncate
    widths = [max(max([len(str(c))] + [len(str(v)) for v in d.iloc[:, i]]), 4)
              for i, c in enumerate(d.columns)]
    total = float(sum(widths))

    tbl = ax.table(cellText=d.values.tolist(), colLabels=list(d.columns),
                   cellLoc="center", loc="upper left",
                   colWidths=[w / total for w in widths], bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    cols = list(d.columns)
    tier_i = cols.index("Tier") if "Tier" in cols else None
    side_i = cols.index("Side") if "Side" in cols else None
    game_i = cols.index("Game") if "Game" in cols else None
    ta_i = next((i for i, c in enumerate(cols) if c.lower().startswith("vs.")), None)

    if game_i is not None:
        band, cur, flip = [], None, False
        for v in d.iloc[:, game_i]:
            if v != cur:
                cur, flip = v, not flip
            band.append(flip)
    else:
        band = [i % 2 == 0 for i in range(len(d))]

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold")
            continue
        val = d.iloc[r - 1, c] if c < len(cols) else ""
        if tier_i is not None and c == tier_i and val in TIER_COLORS:
            cell.set_facecolor(TIER_COLORS[val])
        elif ta_i is not None and c == ta_i and val in TOWARD_COLORS:
            cell.set_facecolor(TOWARD_COLORS[val])
        else:
            cell.set_facecolor(BAND_ROW if band[r - 1] else "white")
        if side_i is not None and c == side_i and val in SIDE_FG:
            cell.set_text_props(color=SIDE_FG[val], fontweight="bold")
        if c == 0:
            cell.set_text_props(ha="left", fontweight="bold")

    fig.text(0.006, 0.30 / H, DISCLAIMER, fontsize=6.6, color=MUTED,
             ha="left", va="center")
    return tbl


def _figsize(d):
    h = 0.30 * (len(d) + 2) + 2.2      # room for header and footer
    w = max(9.5, 1.05 * len(d.columns))
    return w, min(h, 200.0)


def to_png(df, market, season, week, dpi=170):
    """One market's table as PNG bytes."""
    d = _prep(df)
    w, h = _figsize(d)
    fig, ax = plt.subplots(figsize=(w, h))
    _draw(fig, ax, d, market, season, week)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def to_pdf(frames, season, week):
    """All markets, one page each."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for market, df in frames.items():
            d = _prep(df)
            if len(d) == 0:
                continue
            w, h = _figsize(d)
            fig, ax = plt.subplots(figsize=(w, h))
            _draw(fig, ax, d, market, season, week)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    return buf.getvalue()


def to_zip(frames, season, week, dpi=170):
    """Every market as a separate PNG inside one zip. Reddit takes these as a
    gallery, and each market stays readable on its own."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for market, df in frames.items():
            if df is None or len(df) == 0:
                continue
            safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                           for c in _ascii(market)).strip("_") or "market"
            z.writestr(f"opalscales_{safe}_{season}_wk{week}.png",
                       to_png(df, market, season, week, dpi=dpi))
    return buf.getvalue()
