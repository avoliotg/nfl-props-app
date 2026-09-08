"""
OpalScales - render the Line Movement table as PNG or PDF.

WHY A SEPARATE MODULE
    Keeps the app.py edit to a few lines, and makes the rendering testable
    without spinning up Streamlit.

TWO PROBLEMS THIS SOLVES
  1. EMOJI. Matplotlib's bundled fonts have no emoji glyphs, so the Tier and
     vs. Model columns would render as empty boxes. Every cell is stripped to
     ASCII, turning "🔥 Max" into "Max" and "🟢 toward" into "toward".
  2. THE TREND COLUMN. It holds a list per row to drive the sparkline. As table
     text that is noise, so it is dropped.

PNG is for posting (Reddit takes image posts, not spreadsheets). The PDF is one
page per market, for archiving or sending to someone.
"""

import io

import matplotlib
matplotlib.use("Agg")            # no display needed, must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

DROP_COLS = ("Trend", "Bet?")

# tier -> colour, applied to the Tier cell so the table stays readable at a glance
TIER_COLORS = {
    "Max": "#ffd6cc",
    "Strong": "#d6f5d6",
    "Lean": "#fff4cc",
    "Pass": "#f0f0f0",
}
TOWARD_COLORS = {
    "toward": "#d6f5d6",
    "away": "#ffd6cc",
    "flat": "#f0f0f0",
}

HEADER_BG = "#2b2b3d"
HEADER_FG = "white"
ALT_ROW = "#f7f7fa"


def _ascii(v):
    """Strip non-ASCII so emoji do not render as boxes. Keeps the word."""
    if v is None:
        return ""
    s = str(v)
    out = "".join(ch for ch in s if ord(ch) < 128)
    return out.strip()


def _prep(df):
    d = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda v: "" if v != v else f"{v:g}")
        else:
            d[c] = d[c].map(_ascii)
    d.columns = [_ascii(c) for c in d.columns]
    return d


def _draw(fig, ax, d, title, subtitle):
    ax.axis("off")

    # Title and subtitle are positioned in INCHES from the top of the figure,
    # not in axes fractions. Axes-fraction placement collided with the title
    # whenever the table height changed, because the title's pad is in points
    # while the subtitle's offset scaled with the axes.
    fig_h = fig.get_figheight()
    head_in = 0.85 if subtitle else 0.55
    fig.subplots_adjust(top=1.0 - head_in / fig_h, left=0.004, right=0.996,
                        bottom=0.004)
    fig.text(0.004, 1.0 - 0.22 / fig_h, title, fontsize=15,
             fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(0.004, 1.0 - 0.56 / fig_h, subtitle, fontsize=9,
                 color="#666666", ha="left", va="top")

    # Equal column widths truncate long player names ("Rashid Sha..."), so size
    # each column by the widest string it actually contains.
    widths = []
    for i, c in enumerate(d.columns):
        longest = max([len(str(c))] + [len(str(v)) for v in d.iloc[:, i]])
        widths.append(max(longest, 4))
    total = float(sum(widths))
    col_w = [w / total for w in widths]

    tbl = ax.table(cellText=d.values.tolist(), colLabels=list(d.columns),
                   cellLoc="center", loc="upper left", colWidths=col_w,
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    tier_i = list(d.columns).index("Tier") if "Tier" in d.columns else None
    ta_i = None
    for i, c in enumerate(d.columns):
        if c.lower().startswith("vs."):
            ta_i = i
            break

    ncols = len(d.columns)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold")
            continue
        val = d.iloc[r - 1, c] if c < ncols else ""
        if tier_i is not None and c == tier_i and val in TIER_COLORS:
            cell.set_facecolor(TIER_COLORS[val])
        elif ta_i is not None and c == ta_i and val in TOWARD_COLORS:
            cell.set_facecolor(TOWARD_COLORS[val])
        else:
            cell.set_facecolor(ALT_ROW if r % 2 == 0 else "white")
        if c == 0:
            cell.set_text_props(ha="left", fontweight="bold")
    return tbl


def _figsize(d):
    h = 0.30 * (len(d) + 2) + 1.0
    w = max(9.0, 1.05 * len(d.columns))
    return w, min(h, 200.0)


def to_png(df, market, season, week, dpi=170):
    """One market's table as PNG bytes."""
    d = _prep(df)
    w, h = _figsize(d)
    fig, ax = plt.subplots(figsize=(w, h))
    _draw(fig, ax, d, f"OpalScales - {_ascii(market)} line movement",
          f"{season} Week {week}")
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
            _draw(fig, ax, d, f"OpalScales - {_ascii(market)} line movement",
                  f"{season} Week {week}")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    return buf.getvalue()

def to_zip(frames, season, week, dpi=170):
    """Every market as a separate PNG inside one zip.

    Better than a single tall image for sharing: Reddit takes these as a
    gallery, and each market stays readable on its own.
    """
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for market, df in frames.items():
            if df is None or len(df) == 0:
                continue
            safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                           for c in _ascii(market)).strip("_") or "market"
            name = f"opalscales_{safe}_{season}_wk{week}.png"
            z.writestr(name, to_png(df, market, season, week, dpi=dpi))
    return buf.getvalue()
