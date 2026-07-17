import pandas as pd
import numpy as np

PATH = "data/clean/bioage_estimates_2state.parquet"
K = 3  # number of future steps to look at

df = pd.read_parquet(PATH).copy()
df = df.sort_values(["HashedID", "t_years"])

rows = []

for sid, g in df.groupby("HashedID"):
    g = g.sort_values("t_years").reset_index(drop=True)

    if len(g) < K + 2:
        continue

    # compute step changes
    g["dMet"] = g["MetAge"].diff()

    for i in range(len(g) - K):
        immune_now = g.loc[i, "ImmuneState"]

        future_deltas = g.loc[i+1:i+K, "dMet"].dropna()

        if len(future_deltas) < 2:
            continue

        var_future = np.var(future_deltas)

        rows.append({
            "HashedID": sid,
            "Immune_now": immune_now,
            "Future_MetAge_Var": var_future
        })

res = pd.DataFrame(rows)

print("\nOverall correlation:")
print(res["Immune_now"].corr(res["Future_MetAge_Var"]))

print("\nSummary stats:")
print(res.describe())

print("\nTop high immune samples:")
print(res.sort_values("Immune_now", ascending=False).head(10).to_string(index=False))

print("\nTop high volatility samples:")
print(res.sort_values("Future_MetAge_Var", ascending=False).head(10).to_string(index=False))

# also test only positive immune (activation)
res["Immune_pos"] = res["Immune_now"].clip(lower=0)

print("\nCorrelation (positive immune only):")
print(res["Immune_pos"].corr(res["Future_MetAge_Var"]))

res.to_csv("results/immune_volatility_test.csv", index=False)
print("\nSaved: results/immune_volatility_test.csv")
