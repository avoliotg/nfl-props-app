import pandas as pd
import numpy as np

d = pd.read_csv("all_markets.csv")
d = d[d["line_fanduel"].notna() & d["projection"].notna() & d["actual"].notna()].copy()
d["dev"] = d["projection"] - d["line_fanduel"]
d["out"] = d["actual"] - d["line_fanduel"]

print("full-sample fit on line_fanduel")
print("%-12s %6s %9s %9s %9s" % ("market", "n", "beta", "alpha", "SE(ols)"))
for m, g in d.groupby("market"):
    b, a = np.polyfit(g["dev"], g["out"], 1)
    resid = g["out"] - (a + b * g["dev"])
    se = np.sqrt(np.sum(resid ** 2) / (len(g) - 2) / np.sum((g["dev"] - g["dev"].mean()) ** 2))
    print("%-12s %6d %+9.4f %+9.4f %9.4f" % (m, len(g), b, a, se))