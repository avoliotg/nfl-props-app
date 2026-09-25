"""receptions_classifier.py - can the model call the SIDE where beta is real?

Reports only. Writes nothing, places no bet.

WHY RECEPTIONS, AND WHY NOW

    qb_passing_followup.py established something the project had proposed and
    never tested: a side-picker can work where beta cannot. qb_passing's beta
    is 0.085 with a CI spanning zero, so the projection adds nothing to the
    line on a YARDS scale, and yet a walk-forward logistic had measurable
    skill at calling which side of the line the outcome landed. Calling the
    SIGN of a threshold crossing is an easier problem than improving the
    conditional mean.

    Receptions is the market where that should work best and has never been
    tried. Beta 0.226, game-clustered t +6.5, confirmed by four independent
    routes. It is the only market with a demonstrated projection signal, and
    it has no side-picker.

TWO THINGS THAT MAKE THIS A BETTER TEST THAN qb_passing WAS

    1. THE ODDS ACTUALLY MOVE HERE. FanDuel's qb_passing prices turned out to
       be degenerate: book_p standard deviation 0.00058, five distinct values,
       92 percent of props at -110 or -112. So "disagreeing with the book's
       price" was meaningless there because there was no price variation.
       Receptions is priced asymmetrically, with a breakeven range of roughly
       0.417 to 0.610 per the project's notes, so book_p carries real
       information and the disagreement framing means what it says. Section 2
       verifies that rather than assuming it.

    2. THE LINE IS A FEATURE HERE, AND IT SHOULD HAVE BEEN THERE ALL ALONG.
       The qb_passing side-picker predicted P(over) WITHOUT the line in the
       design. Since the target is P(Y > L), a model that cannot see L is
       inferring the threshold from correlated features. It still worked,
       because attempts_roll and total_line proxy the line closely, but it
       was solving a harder problem than necessary. That is a real gap in the
       earlier script and it is worth retesting there.

THE HEADLINE TEST IS SECTION 3, AND IT IS A PROPER SCORING RULE

    Log loss, out of sample, against the devigged market price. market
    _calibration.py measured the FanDuel receptions book at log loss 0.68374
    and Brier 0.24533 on 7,930 rows. If a model cannot beat the market's own
    probability on a proper scoring rule, it has no information the market
    lacks, and no threshold or filter downstream can manufacture some. If it
    can, that is the cleanest possible evidence and it does not depend on any
    betting rule at all.

    Five designs, so the contribution of each piece is visible:

        A  book_p only                    the market, as a baseline
        B  book_p + line
        C  book_p + line + LEAN_FEATS     the full model
        D  book_p + line + dev            dev = projection minus line
        E  line + LEAN_FEATS              no book_p, the qb_passing shape

    The difference in log loss is bootstrapped on game clusters, because two
    overlapping scores are not compared by eyeballing two intervals.

WHAT WOULD BE A FINDING

    Design C or D beating design A on out-of-sample log loss with a
    bootstrapped difference excluding zero. That would be the first direct
    demonstration in this project that the model knows something the market
    does not, measured without a betting rule in the way.

Run from the repo root:
    python receptions_classifier.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
import qb_passing_sidepicker as sp
import recency_test as rt
from models import data_utils
from sklearn.linear_model import LogisticRegression

MARKET = "receptions"
DEVIG_METHOD = "additive"
THRESHOLDS = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
EPS = 1e-9

# From market_calibration.py, FanDuel receptions, all seasons. Used as a
# reference point rather than a hard gate, because this script joins to model
# features and so scores a subset.
PUB_BOOK = {"n": 7930, "over_rate": 0.4720, "logloss": 0.68371,
            "brier": 0.24531}


def _rule(t):
    print()
    print("=" * 94)
    print(t)
    print("=" * 94)


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_delta(a, b, clusters, n_boot, seed=0):
    """Cluster bootstrap of mean(a) minus mean(b), paired on the same rows."""
    d = np.asarray(a, float) - np.asarray(b, float)
    df = pd.DataFrame({"d": d, "g": np.asarray(clusters)})
    agg = df.groupby("g")["d"].agg(["sum", "size"])
    if len(agg) < 15:
        return np.nan, np.nan
    s = agg["sum"].to_numpy()
    n = agg["size"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    return tuple(np.percentile(s[idx].sum(axis=1) / n[idx].sum(axis=1),
                               [2.5, 97.5]))


def load(args, seasons):
    _rule("LOADING")
    mod_name, actual_col = eh.MARKET_SPEC[MARKET]
    mod = importlib.import_module(f"models.{mod_name}")
    feats = list(mod.LEAN_FEATS)
    print(f"  module models.{mod_name}, target {actual_col}")
    print(f"  features from the module: {feats}")

    builder = getattr(mod, "build_all_rows", None) or mod.build_dataset
    df = builder()
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    keep = feats + [actual_col, "season", "week", "player_display_name"]
    df = df[[c for c in keep if c in df.columns]].dropna(
        subset=feats + [actual_col]).copy()
    df["_key"] = data_utils.norm_join_name(df["player_display_name"])
    df["week"] = df["week"].astype(int)
    df = df.drop_duplicates(subset=["season", "week", "_key"])
    print(f"  {len(df):,} model rows with complete features")

    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, [MARKET], args.cache, args.refresh)
    lines = lines[(lines["book"] == args.book) & lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    k = ["season", "week", "market", "player", "book"]
    if "captured_at" in lines.columns:
        lines = lines.sort_values("captured_at").drop_duplicates(k, keep="last")
    lines["_key"] = data_utils.norm_join_name(lines["player"])

    po, do, du = [], [], []
    for oo, uo in zip(lines["over_odds"], lines["under_odds"]):
        r = devig.devig_two_sided(oo, uo, DEVIG_METHOD)
        po.append(r["p_over"] if r["valid"] else np.nan)
        do.append(devig.american_to_decimal(oo))
        du.append(devig.american_to_decimal(uo))
    lines["book_p"] = po
    lines["dec_over"] = do
    lines["dec_under"] = du
    lines = lines.dropna(subset=["book_p", "dec_over", "dec_under"]).copy()
    print(f"  {len(lines):,} {args.book} {MARKET} rows devigged")

    # walk-forward projections, for design D
    try:
        proj = eh.score_market(MARKET, seasons, mode="walk_forward",
                               population="all")
        proj["_key"] = data_utils.norm_join_name(proj["player"])
        proj["week"] = proj["week"].astype(int)
        proj = proj.drop_duplicates(subset=["season", "week", "_key"])[
            ["season", "week", "_key", "projection"]]
    except Exception as e:
        print(f"  projections unavailable ({type(e).__name__}); design D "
              f"will be skipped")
        proj = None

    g = lines.merge(df, on=["season", "week", "_key"], how="inner")
    if proj is not None:
        g = g.merge(proj, on=["season", "week", "_key"], how="left")
        g["dev"] = g["projection"] - g["line"]
    g = g.rename(columns={actual_col: "actual"})
    g = g[g["actual"] != g["line"]].copy()
    g["over"] = (g["actual"] > g["line"]).astype(int)
    g["roi_over"] = np.where(g["over"] == 1, g["dec_over"] - 1.0, -1.0)
    g["roi_under"] = np.where(g["over"] == 0, g["dec_under"] - 1.0, -1.0)
    print(f"  {len(g):,} settled props joined to features")
    print(f"  realized over rate {g['over'].mean():.4f}   "
          f"published (larger population) {PUB_BOOK['over_rate']:.4f}")
    return g, feats


def section_variance(g):
    _rule("SECTION 2: DOES THE PRICE ACTUALLY MOVE IN THIS MARKET?")
    print("qb_passing failed this check: book_p SD 0.00058 across five")
    print("distinct values, which made its 'disagreement' meaningless. If")
    print("receptions fails it too, the disagreement framing is dead here as")
    print("well and only design E is interpretable.")
    print()
    bp = g["book_p"]
    print(f"  rows                  {len(bp):,}")
    print(f"  mean                  {bp.mean():.6f}")
    print(f"  standard deviation    {bp.std():.6f}   "
          f"(qb_passing: 0.000580)")
    print(f"  distinct values       {bp.nunique():,}   (qb_passing: 5)")
    print(f"  10th / 90th pct       {bp.quantile(.10):.4f} / "
          f"{bp.quantile(.90):.4f}")
    print(f"  min / max             {bp.min():.4f} / {bp.max():.4f}")
    print(f"  within 0.01 of 0.5    {(bp.sub(0.5).abs() < 0.01).mean():.2%}   "
          f"(qb_passing: 99.76%)")
    print()
    print("  most common odds pairs:")
    for (oo, uo), n in (g.groupby(["over_odds", "under_odds"]).size()
                        .sort_values(ascending=False).head(6).items()):
        print(f"    {oo:>7} / {uo:<7}  {n:>6,} rows ({n / len(g):.1%})")
    ok = bp.std() > 0.02
    print()
    print("  PRICE VARIES. The disagreement framing is meaningful here."
          if ok else
          "  DEGENERATE, like qb_passing. Treat designs A to D with caution.")
    return ok


def walk_forward(g, design, feats):
    """Fit one design walk-forward and return the scored rows."""
    cols = []
    for part in design:
        if part == "feats":
            cols += feats
        else:
            cols.append(part)
    if any(c not in g.columns for c in cols):
        return None
    sub = g.dropna(subset=cols).copy()
    out = []
    for season in sorted(sub["season"].unique()):
        tr = sub[sub["season"] < season]
        te = sub[sub["season"] == season]
        if len(tr) < 400 or len(te) < 50:
            continue
        m = LogisticRegression(max_iter=3000).fit(tr[cols], tr["over"])
        t = te.copy()
        t["p_model"] = m.predict_proba(te[cols])[:, 1]
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else None


DESIGNS = [
    ("A  book_p only", ["book_p"]),
    ("B  book_p + line", ["book_p", "line"]),
    ("C  book_p + line + feats", ["book_p", "line", "feats"]),
    ("D  book_p + line + dev", ["book_p", "line", "dev"]),
    ("E  line + feats (no book_p)", ["line", "feats"]),
]


def section_scoring(g, feats, n_boot):
    _rule("SECTION 3: PROPER SCORING RULES, OUT OF SAMPLE")
    print("The headline test, and it needs no betting rule. Every design is")
    print("scored against the SAME rows as the market's own devigged")
    print("probability, and the log-loss difference is bootstrapped on game")
    print("clusters. Lower is better. A NEGATIVE delta means the design beats")
    print("the market.")
    print()
    fits = {}
    for name, design in DESIGNS:
        r = walk_forward(g, design, feats)
        if r is None or not len(r):
            print(f"  {name:<30} skipped")
            continue
        fits[name] = r

    if not fits:
        print("  nothing fitted.")
        return {}

    # Score every design on the intersection of rows, so the comparison is
    # paired rather than each design being scored on its own population.
    keys = None
    for r in fits.values():
        k = set(zip(r["season"], r["week"], r["_key"]))
        keys = k if keys is None else (keys & k)
    print(f"  common rows across all designs: {len(keys):,}")
    print()
    print(f"  {'design':<30}{'n':>7}{'logloss':>10}{'delta':>9}"
          f"{'lo95':>9}{'hi95':>9}{'Brier':>9}")
    base = None
    trimmed = {}
    for name, r in fits.items():
        m = r[[(s, w, k) in keys for s, w, k in
               zip(r["season"], r["week"], r["_key"])]].copy()
        m = m.sort_values(["season", "week", "_key"]).reset_index(drop=True)
        trimmed[name] = m
        if base is None:
            base = m
    mkt = logloss(base["book_p"], base["over"])
    print(f"  {'market (devigged book_p)':<30}{len(base):>7,}"
          f"{mkt.mean():>10.5f}{'':>9}{'':>9}{'':>9}"
          f"{float(np.mean((base['book_p'] - base['over']) ** 2)):>9.5f}")
    res = {}
    for name, m in trimmed.items():
        ll = logloss(m["p_model"], m["over"])
        d = ll.mean() - mkt.mean()
        lo, hi = boot_delta(ll, mkt, m["event_id"], n_boot)
        star = "  <--" if np.isfinite(hi) and hi < 0 else ""
        br = float(np.mean((m["p_model"] - m["over"]) ** 2))
        print(f"  {name:<30}{len(m):>7,}{ll.mean():>10.5f}{d:>+9.5f}"
              f"{lo:>+9.5f}{hi:>+9.5f}{br:>9.5f}{star}")
        res[name] = m
    print()
    print("  A design whose interval excludes zero on the LOW side is BETTER")
    print("  CALIBRATED than the market overall. That is a stronger claim")
    print("  than the one betting needs, which is section 3b.")
    _rule("SECTION 3b: THE ENCOMPASSING TEST, AND THIS IS THE PRIMARY")
    print("Outcome regressed on book_p AND on (p_model minus book_p),")
    print("clustered on game. The coefficient on the disagreement answers the")
    print("question betting actually cares about: does the model carry")
    print("information the PRICE does not? A positive coefficient means")
    print("combining the two beats the price alone.")
    print()
    print("WHY THIS AND NOT THE SIMPLE SLOPE. Measured on synthetic data at")
    print("this project's sample size (860 games, 9 props each):")
    print()
    print("    test                        model better    model worse")
    print("    encompassing coefficient      t +7.45         t +4.01")
    print("    simple slope on disagree      t +0.62         t +0.40")
    print("    log loss delta                t -3.21         t +4.92")
    print()
    print("The simple slope has NO power once book_p varies, because it omits")
    print("the price. qb_passing's +0.7739 was interpretable only because")
    print("book_p was constant there, so the slope described the model alone.")
    print("Using it here would return roughly zero even with a real signal.")
    print()
    print("Note also that the encompassing coefficient was POSITIVE even when")
    print("the model was objectively worse. That is not a defect: a worse")
    print("forecast can still carry independent information. It does mean a")
    print("positive coefficient is NOT a claim that the model beats the")
    print("market, only that it adds to it. Read it with section 3.")
    print()
    print(f"  {'design':<30}{'n':>7}{'book_p':>10}{'disagree':>11}"
          f"{'SE':>9}{'t':>7}")
    for name, m in trimmed.items():
        dis = (m["p_model"] - m["book_p"]).to_numpy()
        X = np.column_stack([np.ones(len(m)), m["book_p"].to_numpy(), dis])
        f = rt.ols_cluster(X, m["over"].to_numpy(), m["event_id"].to_numpy())
        if f is None:
            continue
        t = f["beta"][2] / f["se"][2] if f["se"][2] else np.nan
        star = "  <--" if np.isfinite(t) and t > 2 else ""
        print(f"  {name:<30}{len(m):>7,}{f['beta'][1]:>+10.4f}"
              f"{f['beta'][2]:>+11.4f}{f['se'][2]:>9.4f}{t:>7.2f}{star}")
    print()
    print("  The book_p coefficient is a bonus diagnostic: near 1 means the")
    print("  market is well calibrated on these rows, which market_")
    print("  calibration.py already suggested it is.")
    return res


def section_rule(fits, n_boot):
    _rule("SECTION 4: THE THRESHOLD PROFILE, FOR WHICHEVER DESIGN WON")
    if not fits:
        return None
    # pick the design with the lowest log loss, stated explicitly
    best_name, best = None, None
    best_ll = np.inf
    for name, m in fits.items():
        ll = logloss(m["p_model"], m["over"]).mean()
        if ll < best_ll:
            best_ll, best_name, best = ll, name, m
    print(f"  design selected by log loss: {best_name}")
    print("  NOTE: selecting the design by its own score and then reporting")
    print("  its betting profile is a mild selection. Section 5's placebo")
    print("  corrects for the threshold grid but NOT for this choice.")
    print()
    b = best.copy()
    b["disagree"] = b["p_model"] - b["book_p"]
    print(f"  {'|disagree| >=':<16}{'bets':>7}{'per szn':>9}{'win':>8}"
          f"{'ROI':>9}{'lo95':>9}{'hi95':>9}")
    for thr in THRESHOLDS:
        sel = b[b["disagree"].abs() >= thr]
        if len(sel) < 100:
            print(f"  {thr:<16.2f}{len(sel):>7,}   too few")
            continue
        prof = np.where(sel["disagree"] > 0, sel["roi_over"],
                        sel["roi_under"])
        won = np.where(sel["disagree"] > 0, sel["over"], 1 - sel["over"])
        lo, hi = sp.boot_ci(prof, sel["event_id"], n_boot)
        star = "  <--" if np.isfinite(lo) and lo > 0 else ""
        print(f"  {thr:<16.2f}{len(sel):>7,}{len(sel) / 3.25:>9.0f}"
              f"{won.mean():>8.4f}{prof.mean():>+9.4f}{lo:>+9.4f}"
              f"{hi:>+9.4f}{star}")

    # Encompassing form, NOT the simple slope. The simple slope has no power
    # once book_p varies: measured t +0.62 on synthetic data where the model
    # was genuinely better. See section 3b.
    dis = b["disagree"].to_numpy()
    X = np.column_stack([np.ones(len(b)), b["book_p"].to_numpy(), dis])
    f = rt.ols_cluster(X, b["over"].to_numpy(), b["event_id"].to_numpy())
    Xs = np.column_stack([np.ones(len(b)), dis])
    fs = rt.ols_cluster(b_X := Xs, b["over"].to_numpy(),
                        b["event_id"].to_numpy())
    if f:
        t = f["beta"][2] / f["se"][2] if f["se"][2] else np.nan
        print()
        print(f"  encompassing coefficient on disagree {f['beta'][2]:+.4f} "
              f"(SE {f['se'][2]:.4f}, t {t:+.2f})")
        if fs:
            ts = fs["beta"][1] / fs["se"][1] if fs["se"][1] else np.nan
            print(f"  simple slope, for contrast only       "
                  f"{fs['beta'][1]:+.4f} (t {ts:+.2f})")
            print("  A large gap between those two is EXPECTED here and is")
            print("  not a contradiction. The simple slope omits the price")
            print("  and is uninformative whenever the price varies.")
    return b


def section_placebo(b, n_perm):
    _rule("SECTION 5: PLACEBO, WITH A FLOOR SWEEP")
    if b is None or not len(b):
        return
    print("Permute the disagreement within season and week, recompute the")
    print("grid, take the best cell. Floors swept so a result resting on one")
    print("narrow cell is visible, which is what qb_passing turned out to do")
    print("(ROI decayed +0.0839 to +0.0259 as the floor rose to 400).")
    print()
    print(f"  {'floor':<8}{'cells':>7}{'best ROI':>10}{'best t':>9}"
          f"{'null 95th t':>13}{'p':>8}")
    for floor in (150, 250, 400):
        usable = sum(1 for thr in THRESHOLDS
                     if len(b[b["disagree"].abs() >= thr]) >= floor)
        if not usable:
            print(f"  {floor:<8}{0:>7}   no cell survives")
            continue
        obs_roi, obs_t = sp._grid_best("disagree", b, min_bets=floor)
        rng = np.random.default_rng(0)
        gg = b.copy()
        nt = []
        for _ in range(n_perm):
            gg["_d"] = gg.groupby(["season", "week"])["disagree"].transform(
                lambda s: rng.permutation(s.to_numpy()))
            _, t = sp._grid_best("_d", gg, min_bets=floor)
            if t > -9.0:
                nt.append(t)
        if not nt:
            print(f"  {floor:<8}{usable:>7}   placebo empty")
            continue
        nt = np.array(nt)
        print(f"  {floor:<8}{usable:>7}{obs_roi:>+10.4f}{obs_t:>+9.2f}"
              f"{np.percentile(nt, 95):>+13.2f}"
              f"{float((nt >= obs_t).mean()):>8.3f}")


def section_seasons(b, n_boot):
    _rule("SECTION 6: SEASON SPLIT")
    if b is None or not len(b):
        return
    print("Receptions can score more seasons than qb_passing could, because")
    print("the first scorable season is still the second in the data. Expect")
    print("2024, 2025 and a thin 2026.")
    for thr in (0.04, 0.06, 0.08):
        sel = b[b["disagree"].abs() >= thr].copy()
        if len(sel) < 150:
            continue
        sel["prof"] = np.where(sel["disagree"] > 0, sel["roi_over"],
                               sel["roi_under"])
        print()
        print(f"  |disagree| >= {thr}  ({len(sel):,} bets, pooled "
              f"{sel['prof'].mean():+.4f})")
        print(f"    {'season':<8}{'bets':>7}{'ROI':>9}{'lo95':>9}{'hi95':>9}")
        signs = []
        for season, gg in sel.groupby("season"):
            if len(gg) < 40:
                print(f"    {season:<8}{len(gg):>7,}   too few")
                continue
            lo, hi = sp.boot_ci(gg["prof"], gg["event_id"], n_boot)
            print(f"    {season:<8}{len(gg):>7,}{gg['prof'].mean():>+9.4f}"
                  f"{lo:>+9.4f}{hi:>+9.4f}")
            signs.append(np.sign(gg["prof"].mean()))
        print("    SIGN CONSISTENT" if len(signs) >= 2 and all(s > 0 for s in signs)
              else "    SIGN FLIPS or too few seasons")


def section_summary():
    _rule("SECTION 7: WHAT THIS SETTLES")
    print("  Section 3 is the section that matters and it does not depend on")
    print("  any betting rule, any threshold, or any selection. Out-of-sample")
    print("  log loss against the market's own devigged probability is the")
    print("  cleanest question this project can ask:")
    print()
    print("    does the model know anything the market does not?")
    print()
    print("  If YES, receptions has a genuine informational edge and the")
    print("  remaining work is how to harvest it after vig.")
    print()
    print("  If NO, the answer is decisive in a way nothing else has been.")
    print("  Beta 0.226 with t +6.5 would then mean the model adds to the")
    print("  LINE while adding nothing to the PRICE, which is coherent given")
    print("  receptions is 93 percent flat on the line and FanDuel moves it")
    print("  through the odds. The line is the stale quantity; the price is")
    print("  the sharp one; and every beta in this project is measured")
    print("  against the stale one.")
    print()
    print("  ALSO WORTH RETESTING ELSEWHERE. The qb_passing side-picker")
    print("  omitted the LINE from its design, so it was inferring the")
    print("  threshold from correlated features instead of seeing it. Adding")
    print("  the line there is a one-word change and should only help.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--perm", type=int, default=300)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 94)
    print("RECEPTIONS THRESHOLD CLASSIFIER: does the model beat the PRICE?")
    print(f"  seasons {seasons}   book {args.book}   devig {DEVIG_METHOD}")
    print("=" * 94)

    g, feats = load(args, seasons)
    if len(g) < 1500:
        print("too few rows.")
        sys.exit(1)
    section_variance(g)
    fits = section_scoring(g, feats, args.boot)
    b = section_rule(fits, args.boot)
    section_placebo(b, args.perm)
    section_seasons(b, args.boot)
    section_summary()

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
