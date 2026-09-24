"""
HURDLE MODEL FOR RECEPTIONS: fixing the one rung that fails.

THE DEFECT

Ladder calibration found nine of ten rungs inside half a point and one clear
miss:

    k = 1   predicted 0.8911   realized 0.9029   gap +1.2 pp   z +3.72

The `caught_last` test then ruled out the obvious explanation. Both subgroups
miss in the SAME direction, caught-last-game at +1.0 (z +3.15) and
blanked-last-game at +3.1 (z +2.29), so this is not a single distribution
averaging two populations. Adding the feature also moved beta from 0.273 to
0.271 and MAE from 1.5723 to 1.5733, both marginally worse, which is the
signature of information the market already prices.

So the gap is DISTRIBUTIONAL. It is a property of the negative binomial
family: at a small mean an NB places more mass on exactly zero than reality
does, because a receiver who is active almost always draws at least one
target. The family has two parameters, they are spent on the mean and the
variance, and nothing is left to control the zero.

THE FIX, AND WHY A HURDLE RATHER THAN ZERO-INFLATION

Zero-INFLATION adds mass at zero. The defect runs the other way, so this needs
zero-DEFLATION. A hurdle model handles either, by modelling the zero
separately from the count:

    stage 1   P(X >= 1) = pi, fitted directly against realized frequency
    stage 2   given X >= 1, (X - 1) ~ NB with its own mean and variance

    P(X >= k) = pi * P(X - 1 >= k - 1 | X >= 1)      for k >= 1

The k = 1 rung then equals pi exactly, so it is calibrated by construction
rather than by luck. And the mean is preserved: pi * E[X | X >= 1] = m is
imposed, so E[X - 1 | X >= 1] = m / pi - 1, which pins stage 2.

WHAT HAS TO BE CHECKED, NOT ASSUMED

Fixing one rung is easy if you are allowed to break the others. Shifting mass
off zero has to go somewhere, so every rung moves. Three things therefore get
compared side by side against the plain NB:

  1. the FULL ladder, all ten rungs, not just k = 1
  2. the MAIN LINE rungs specifically, k = 3 and k = 4, since the candidate
     rule lives at 2.5 and 3.5 and those probabilities drive every edge
  3. the candidate rule's HOLDOUT ROI under both distributions

If the hurdle fixes k = 1 and degrades k = 3, it is a worse model for betting
even though it is a better model for describing zeros. That trade has to be
visible rather than inferred.

EVERYTHING IS FITTED LEAVE-SEASON-OUT. pi comes from a logistic regression on
log(mean) using the other seasons, and the stage-2 dispersion likewise, so no
season is scored with parameters it contributed to.

Run from the repo root:
    python hurdle_test.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

import eval_harness as eh
import edge_threshold_v5 as et

MAXK = 10
BREAKEVEN = 113.0 / 213.0


# --------------------------------------------------------------------- data

def load(args):
    sec = eh._secrets()
    box = {}

    def factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    seasons = (list(eh.ALL_SEASONS) if args.all_seasons
               else [int(x) for x in (args.seasons or "2025").split(",")])
    lines = eh.load_lines(factory, seasons, ["receptions"], args.cache,
                          args.refresh)
    if lines.empty:
        sys.exit("  no lines found")
    lines = lines[lines["week"].notna()]
    props = eh.collapse_books(lines)

    print("\nscoring receptions walk-forward")
    proj = eh.score_market("receptions", seasons, mode="walk_forward",
                           population=args.score_population)
    if not len(proj):
        sys.exit("  nothing scored")

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)
    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    j["dev_cons"] = j["projection"] - j["line_consensus"]
    j["out_cons"] = j["actual"] - j["line_consensus"]
    j = j.dropna(subset=["dev_cons", "out_cons"]).copy()
    print(f"  joined {len(j)} rows")
    return j, seasons


def attach_mean(j, params, sig):
    j = j.copy()
    key = list(zip(j["market"], j["season"]))
    j["alpha"] = [params.get(k, (np.nan,) * 4)[0] for k in key]
    j["beta"] = [params.get(k, (np.nan,) * 4)[1] for k in key]
    j["sig_a"] = [sig.get(k, (np.nan, 0.5))[0] for k in key]
    j = j.dropna(subset=["alpha", "beta", "sig_a"])
    j["blend"] = (j["line_consensus"] + j["alpha"]
                  + j["beta"] * j["dev_cons"])
    j["sigma"] = j["sig_a"] * np.sqrt(j["blend"].clip(lower=1e-6))
    return j[j["blend"] > 0.2].copy()


# --------------------------------------------------------------- the hurdle

def fit_hurdle(j, seasons, min_rows=500):
    """Leave-season-out: logistic pi(log mean), plus stage-2 dispersion."""
    from sklearn.linear_model import LogisticRegression
    out = {}
    print("\n  HURDLE PARAMETERS, leave-season-out")
    print(f"  {'season':>8}{'pi intercept':>14}{'pi slope':>10}"
          f"{'phi (stage 2)':>15}{'n_train':>9}{'pi at mean 3':>14}")
    for s in seasons:
        tr = j[j["season"] != s]
        if len(tr) < min_rows:
            continue
        x = np.log(tr["blend"].to_numpy()).reshape(-1, 1)
        y = (tr["actual"].to_numpy() >= 1).astype(int)
        if y.min() == y.max():
            continue
        lr = LogisticRegression(max_iter=1000).fit(x, y)
        b0, b1 = float(lr.intercept_[0]), float(lr.coef_[0][0])

        # stage 2: dispersion of (X-1) among X >= 1, relative to its mean
        pi_tr = 1.0 / (1.0 + np.exp(-(b0 + b1 * np.log(
            tr["blend"].to_numpy()))))
        mu_y = tr["blend"].to_numpy() / np.clip(pi_tr, 1e-6, None) - 1.0
        pos = tr["actual"].to_numpy() >= 1
        yv = tr["actual"].to_numpy()[pos] - 1.0
        mv = np.clip(mu_y[pos], 1e-6, None)
        # quasi-Poisson dispersion: Var(Y) = phi * E[Y]
        phi = float(np.mean((yv - mv) ** 2 / mv))
        out[s] = (b0, b1, phi)
        pi3 = 1.0 / (1.0 + np.exp(-(b0 + b1 * np.log(3.0))))
        print(f"  {int(s):>8}{b0:>14.4f}{b1:>10.4f}{phi:>15.4f}"
              f"{len(tr):>9}{pi3:>14.4f}")
    return out


def sf_nb(k, mu, phi):
    """P(Y >= k) for Y with mean mu and variance phi*mu."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    out = np.full(mu.shape, np.nan)
    if phi > 1.0001:
        p = 1.0 / phi
        r = mu * p / (1.0 - p)
        out = stats.nbinom.sf(k - 1, r, p)
    else:
        out = stats.poisson.sf(k - 1, mu)
    return out


def sf_hurdle(k, blend, hp):
    """P(X >= k) under the hurdle. k=1 returns pi exactly."""
    b0, b1, phi = hp
    pi = 1.0 / (1.0 + np.exp(-(b0 + b1 * np.log(np.clip(blend, 1e-6, None)))))
    if k <= 0:
        return np.ones_like(pi)
    if k == 1:
        return pi
    mu_y = np.clip(blend / np.clip(pi, 1e-6, None) - 1.0, 1e-6, None)
    return pi * sf_nb(k - 1, mu_y, phi)


def sf_plain(k, blend, sigma):
    """P(X >= k) under the plain moment-matched NB, for comparison."""
    m = np.clip(np.asarray(blend, float), 1e-6, None)
    v = np.asarray(sigma, float) ** 2
    out = np.empty(m.shape)
    nb = v > m * 1.0001
    if nb.any():
        p = m[nb] / v[nb]
        r = m[nb] * p / (1.0 - p)
        out[nb] = stats.nbinom.sf(k - 1, r, p)
    if (~nb).any():
        out[~nb] = stats.poisson.sf(k - 1, m[~nb])
    return out


def predict(j, hp_by_season, which):
    """P(X >= k) for every k, under 'plain' or 'hurdle'."""
    P = {}
    for k in range(1, MAXK + 1):
        v = np.full(len(j), np.nan)
        if which == "plain":
            v = sf_plain(k, j["blend"].to_numpy(), j["sigma"].to_numpy())
        else:
            for s, idx in j.groupby("season").indices.items():
                hp = hp_by_season.get(s)
                if hp is None:
                    continue
                v[idx] = sf_hurdle(k, j["blend"].to_numpy()[idx], hp)
        P[k] = v
    return P


# ------------------------------------------------------------ the comparison

def ladder_compare(j, Pp, Ph):
    print("\n" + "=" * 96)
    print("LADDER: plain NB against the hurdle, every rung")
    print("=" * 96)
    print(f"  {'k':>3}{'n':>7}{'realized':>10}"
          f"{'NB pred':>10}{'NB gap':>9}{'NB z':>8}"
          f"{'HUR pred':>10}{'HUR gap':>9}{'HUR z':>8}{'better':>9}")
    for k in range(1, MAXK + 1):
        hit = (j["actual"].to_numpy() >= k).astype(float)
        ok = np.isfinite(Pp[k]) & np.isfinite(Ph[k])
        if ok.sum() < 150:
            continue
        cm = eh.cluster_mean(hit[ok], j["event_id"].to_numpy()[ok])
        if cm is None:
            continue
        pn, phh = float(np.mean(Pp[k][ok])), float(np.mean(Ph[k][ok]))
        if max(pn, phh) < 0.01 or min(pn, phh) > 0.99:
            continue
        gn, gh = cm["mean"] - pn, cm["mean"] - phh
        zn = gn / cm["se"] if cm["se"] > 0 else np.nan
        zh = gh / cm["se"] if cm["se"] > 0 else np.nan
        win = "hurdle" if abs(gh) < abs(gn) else "NB"
        print(f"  {k:>3}{int(ok.sum()):>7}{cm['mean']:>10.4f}"
              f"{pn:>10.4f}{100 * gn:>+9.1f}{zn:>+8.2f}"
              f"{phh:>10.4f}{100 * gh:>+9.1f}{zh:>+8.2f}{win:>9}")
    print("\n    k = 1 is fixed by construction under the hurdle, so it is not")
    print("    evidence on its own. The rungs that matter are 3 and 4, where")
    print("    the candidate rule lives, and whether the rest hold up.")


def mainline_compare(j, Pp, Ph):
    print("\n" + "=" * 96)
    print("MAIN LINE ONLY: the rungs the candidate rule actually bets")
    print("=" * 96)
    for line, k in ((2.5, 3), (3.5, 4)):
        sel = np.isclose(j["line_consensus"], line)
        if sel.sum() < 200:
            continue
        hit = (j["actual"].to_numpy() >= k).astype(float)
        cm = eh.cluster_mean(hit[sel], j["event_id"].to_numpy()[sel])
        pn, phh = float(np.mean(Pp[k][sel])), float(np.mean(Ph[k][sel]))
        print(f"\n  line {line} (needs {k}+), n={int(sel.sum())}, "
              f"realized over rate {cm['mean']:.4f} (SE {cm['se']:.4f})")
        print(f"    plain NB  predicted {pn:.4f}   gap "
              f"{100 * (cm['mean'] - pn):+.1f} pp   "
              f"z {(cm['mean'] - pn) / cm['se']:+.2f}")
        print(f"    hurdle    predicted {phh:.4f}   gap "
              f"{100 * (cm['mean'] - phh):+.1f} pp   "
              f"z {(cm['mean'] - phh) / cm['se']:+.2f}")


def rule_compare(j, hp_by_season, min_edge, iters_note=True):
    """The candidate rule's holdout ROI under each distribution."""
    print("\n" + "=" * 96)
    print("THE CANDIDATE RULE UNDER EACH DISTRIBUTION")
    print("=" * 96)
    sub = j[np.isin(np.round(j["line_consensus"], 2), [2.5, 3.5])].copy()
    if len(sub) < 500:
        print("  too few rows at 2.5 and 3.5")
        return
    oo = pd.to_numeric(sub.get("over_odds"), errors="coerce").fillna(-113.0)
    uo = pd.to_numeric(sub.get("under_odds"), errors="coerce").fillna(-113.0)

    def be(o):
        o = np.asarray(o, float)
        r = np.where(o < 0, (-o) / ((-o) + 100.0), 100.0 / (o + 100.0))
        return r

    def pay(o):
        o = np.asarray(o, float)
        return np.where(o < 0, 100.0 / (-o), o / 100.0)

    beo, beu = be(oo), be(uo)
    payo, payu = pay(oo), pay(uo)

    print(f"  {'distribution':<16}{'bets':>7}{'per wk':>8}{'win rate':>10}"
          f"{'ROI':>9}{'SE':>8}{'t':>7}")
    weeks = sub.groupby("season")["week"].nunique().sum()
    results = {}
    for tag in ("plain", "hurdle"):
        # P(over) at the posted line: line 2.5 needs 3+, line 3.5 needs 4+
        k = np.where(np.isclose(sub["line_consensus"], 2.5), 3, 4)
        po = np.full(len(sub), np.nan)
        for kk in (3, 4):
            m = k == kk
            if not m.any():
                continue
            if tag == "plain":
                po[m] = sf_plain(kk, sub["blend"].to_numpy()[m],
                                 sub["sigma"].to_numpy()[m])
            else:
                for s, idx in sub.groupby("season").indices.items():
                    hp = hp_by_season.get(s)
                    if hp is None:
                        continue
                    sel = np.zeros(len(sub), bool)
                    sel[idx] = True
                    sel &= m
                    if sel.any():
                        po[sel] = sf_hurdle(kk,
                                            sub["blend"].to_numpy()[sel], hp)
        eo, eu = po - beo, (1.0 - po) - beu
        side_over = eo >= eu
        edge = np.maximum(eo, eu)
        won = np.where(side_over, sub["actual"] > sub["line_consensus"],
                       sub["actual"] < sub["line_consensus"])
        p_ = np.where(side_over, payo, payu)
        profit = np.where(won, p_, -1.0)
        q = np.isfinite(edge) & (edge >= min_edge)
        if q.sum() < 30:
            print(f"  {tag:<16}{int(q.sum()):>7}   too few bets")
            continue
        cm = eh.cluster_mean(profit[q], sub["event_id"].to_numpy()[q])
        t = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {tag:<16}{int(q.sum()):>7}{q.sum() / max(weeks, 1):>8.1f}"
              f"{won[q].mean():>10.4f}{cm['mean']:>+9.4f}{cm['se']:>8.4f}"
              f"{t:>+7.2f}")
        results[tag] = cm["mean"]
    if len(results) == 2:
        d = results["hurdle"] - results["plain"]
        print(f"\n    hurdle minus plain: {d:+.4f}")
        print("    This is in-sample across all four seasons, so it is a")
        print("    direction check rather than a holdout. If the hurdle is")
        print("    not clearly better here it will not be better out of")
        print("    sample either, and the plain NB stays.")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--min-edge", type=float, default=0.09)
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("HURDLE MODEL: does modelling the zero separately fix the ladder?")
    print("=" * 96)

    j, seasons = load(args)
    params = et.fit_params(j)
    if not params:
        sys.exit("  could not fit shrinkage parameters")
    et.report_params(params)
    sig = et.fit_sigma_sqrt(j, params)
    j = attach_mean(j, params, sig)
    print(f"\n  {len(j)} rows with a usable mean")

    hp = fit_hurdle(j, seasons)
    if not hp:
        sys.exit("  could not fit the hurdle")

    Pp = predict(j, hp, "plain")
    Ph = predict(j, hp, "hurdle")

    ladder_compare(j, Pp, Ph)
    mainline_compare(j, Pp, Ph)
    rule_compare(j, hp, args.min_edge)

    print("\n" + "=" * 96)
    print("HOW TO DECIDE")
    print("=" * 96)
    print("  The hurdle fixes k = 1 by construction, so ignore that row as")
    print("  evidence. Judge it on rungs 3 and 4, on whether the upper rungs")
    print("  hold, and on the rule comparison.")
    print("  If the hurdle wins at 3 and 4, adopt it: those drive every edge")
    print("  the candidate rule computes, and a better 0.5 rung also makes")
    print("  the alt-ladder fair odds usable at the bottom.")
    print("  If it fixes k = 1 and degrades 3 and 4, keep the plain NB for")
    print("  betting and use the hurdle only where the 0.5 rung is the bet.")
    print("  Shifting mass off zero has to go somewhere, and that trade is")
    print("  the whole question.")


if __name__ == "__main__":
    main()
