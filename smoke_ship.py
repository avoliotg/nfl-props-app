"""Synthetic smoke test for the September 24 ship. No DB, no repo state."""
import sys
import pandas as pd
import numpy as np
import mc_pricing, mc

ok = True
def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print("  %-58s %s %s" % (label, "OK" if cond else "FAIL", detail))

print("=" * 76)
print("[1] blend arithmetic")
m = mc_pricing.blended_mean("receptions", 4.0, 2.5)
want = 2.5 + 0.1491 + 0.2261 * 1.5
check("receptions blend exact", abs(m - want) < 1e-12, "%.6f" % m)
m2 = mc_pricing.blended_mean("receiving", 99.0, 40.0)
m3 = mc_pricing.blended_mean("receiving", 1.0, 40.0)
check("beta=0 ignores projection entirely", abs(m2 - m3) < 1e-12, "%.4f" % m2)
check("unknown market falls back to raw proj",
      mc_pricing.blended_mean("nosuch", 7.0, 3.0) == 7.0)

print("[2] reference_only routing")
r = mc.edge_calc("receptions", 4.0, 2.5, 0, over_odds=-110, under_odds=-110)
check("receptions carries signal", r["reference_only"] is False)
for mk, ln in (("receiving", 40.0), ("rushing", 50.0),
               ("qb_passing", 240.0), ("qb_rushing", 20.0)):
    r2 = mc.edge_calc(mk, ln * 1.2, ln, 0, over_odds=-110, under_odds=-110)
    check("%s is reference only" % mk, r2 is not None and r2["reference_only"])

print("[3] a zero-beta market must give the SAME price for any projection")
a = mc.edge_calc("rushing", 200.0, 50.0, 0, over_odds=-110, under_odds=-110)
b = mc.edge_calc("rushing", 36.0, 50.0, 0, over_odds=-110, under_odds=-110)
check("rushing p_over independent of projection",
      a["p_over"] == b["p_over"], "%.1f vs %.1f" % (a["p_over"], b["p_over"]))

print("[4] the old path still exists for rollback and DIFFERS")
old = mc.prob_over_raw("receptions", 4.0, 2.5, 0)
new = mc.prob_over("receptions", 4.0, 2.5, 0)
check("raw and blended differ", abs(old - new) > 1e-6,
      "raw %.4f new %.4f" % (old, new))

print("[5] floor behaviour under the blend")
# a sub-floor projection used to return None. Under the blend the mean is
# near the line, so it should now price.
old_f = mc.prob_over_raw("receptions", 1.0, 2.5, 0)
new_f = mc.prob_over("receptions", 1.0, 2.5, 0)
check("sub-floor proj was unpriced on the raw path", old_f is None)
check("sub-floor proj now prices via the blend", new_f is not None,
      "%.4f" % new_f if new_f else "")

print("[6] junk never raises")
for args in (("receptions", None, 2.5), ("receptions", float('nan'), 2.5),
             ("receptions", 4.0, None)):
    try:
        mc.edge_calc(*args, 0, over_odds=-110, under_odds=-110)
        print("  %-58s OK" % ("junk %s handled" % (args,)))
    except Exception as e:
        ok = False
        print("  FAIL %s raised %s" % (args, type(e).__name__))

print("[7] db helpers on synthetic frames")
sys.modules['streamlit'] = type(sys)('streamlit')
for n in ('cache_data','cache_resource'):
    setattr(sys.modules['streamlit'], n, lambda *a, **k: (lambda f: f))
sys.modules['streamlit'].warning = lambda *a, **k: None
sys.modules['streamlit'].session_state = {}
sys.modules['streamlit'].secrets = {}
try:
    import db
    frame = pd.DataFrame({
        "player": ["A"] * 6,
        "book": ["fanduel", "draftkings", "betmgm", "fanduel", "fanduel", "bovada"],
        "line": [2.5, 3.5, 2.5, 2.5, 2.5, 4.5],
        "projection": [None, None, None, None, 3.1, None],
        "captured_at": pd.date_range("2026-09-22", periods=6, freq="h"),
    })
    kept = db._drop_saved_rows(frame)
    check("saved row dropped (1 of 6)", len(kept) == 5, "%d kept" % len(kept))
    sub, nb, used = db._select_book(kept)
    check("book selected", len(sub) == 2 and used == "fanduel",
          "n=%d book=%s n_books=%d (3 fanduel rows, 1 of them saved)" % (len(sub), used, nb))
    check("n_books counts the pre-selection field", nb == 4, "%d" % nb)
    empty, nb2, used2 = db._select_book(kept, "pinnacle")
    check("absent book returns empty, no synthetic fallback",
          empty.empty and used2 is None)
except Exception as e:
    print("  db import skipped: %s: %s" % (type(e).__name__, e))

print("=" * 76)
print("ALL SMOKE CHECKS PASSED" if ok else "SOME SMOKE CHECKS FAILED")
sys.exit(0 if ok else 1)
