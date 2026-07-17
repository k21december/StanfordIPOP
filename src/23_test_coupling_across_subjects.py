import pandas as pd
import numpy as np

PATH = "data/clean/bioage_estimates_2state.parquet"

df = pd.read_parquet(PATH).copy()
df = df.sort_values(["HashedID", "t_years"])

results = []

for sid, g in df.groupby("HashedID"):
    g = g.sort_values("t_years").copy()

    # need at least 3 points for differences
    if len(g) < 3:
        continue

    # first differences
    g["dMet"] = g["MetAge"].diff()
    g["dImm"] = g["ImmuneState"].diff()

    tmp = g[["dMet","dImm"]].dropna()

    if len(tmp) < 3:
        continue

    corr = tmp["dMet"].corr(tmp["dImm"])

    results.append({
        "HashedID": sid,
        "n_points": len(tmp),
        "corr_dMet_dImm": corr
    })

res = pd.DataFrame(results)

print("\nOverall summary:")
print(res["corr_dMet_dImm"].describe())

print("\nTop positive correlations:")
print(res.sort_values("corr_dMet_dImm", ascending=False).head(10).to_string(index=False))

print("\nTop negative correlations:")
print(res.sort_values("corr_dMet_dImm", ascending=True).head(10).to_string(index=False))

# overall pooled correlation (all data)
df["dMet"] = df.groupby("HashedID")["MetAge"].diff()
df["dImm"] = df.groupby("HashedID")["ImmuneState"].diff()

pooled = df[["dMet","dImm"]].dropna()

print("\nPooled correlation (all subjects combined):")
print(pooled["dMet"].corr(pooled["dImm"]))

# save results
res.to_csv("results/immune_met_coupling_by_subject.csv", index=False)
print("\nSaved: results/immune_met_coupling_by_subject.csv")
