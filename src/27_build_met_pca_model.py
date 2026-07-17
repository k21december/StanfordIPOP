import pandas as pd
import numpy as np

# load data
bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

# align IDs
met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")

print("Merged rows:", len(df))

# normalize met_pca_4 (important)
df["met_pca_4_z"] = (df["met_pca_4"] - df["met_pca_4"].mean()) / df["met_pca_4"].std()

# test multiple alpha values
alphas = np.linspace(-1.0, 1.0, 21)

results = []

# create future targets
df = df.sort_values(["HashedID", "t_years"]).copy()
df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["dMetAge"] = df["MetAge_next"] - df["MetAge"]

for a in alphas:
    df["BioAge_new"] = df["MetAge"] + a * df["met_pca_4_z"]

    # gap
    df["BioAge_new_gap"] = df["BioAge_new"] - df["t_years"]

    # future change
    corr_future = df["BioAge_new"].corr(df["dMetAge"])

    # gap vs gaptrue
    if "MetAge_gap" in df.columns:
        corr_gap = df["BioAge_new_gap"].corr(df["MetAge_gap"])
    else:
        corr_gap = np.nan

    results.append({
        "alpha": a,
        "corr_with_future_dMetAge": corr_future,
        "corr_with_gap": corr_gap
    })

res = pd.DataFrame(results)

# find best alpha
best = res.iloc[res["corr_with_future_dMetAge"].abs().idxmax()]

print("\nAlpha sweep results:")
print(res.to_string(index=False))

print("\nBest alpha (by future prediction):")
print(best)

# save best model
a = best["alpha"]
df["BioAge_best"] = df["MetAge"] + a * df["met_pca_4_z"]

df[[
    "HashedID",
    "VisitID",
    "t_years",
    "MetAge",
    "met_pca_4_z",
    "BioAge_best"
]].to_csv("data/clean/bioage_met_pca_model.csv", index=False)

print("\nSaved: data/clean/bioage_met_pca_model.csv")
