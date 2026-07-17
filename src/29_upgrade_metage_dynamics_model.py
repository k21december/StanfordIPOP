import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")

df = df.sort_values(["HashedID", "t_years"]).reset_index(drop=True)

pc_cols = ["met_pca_4", "met_pca_3", "met_pca_6", "met_pca_1"]

g = df.groupby("HashedID", sort=False)

df["MetAge_next"] = g["MetAge"].shift(-1)
df["t_next"] = g["t_years"].shift(-1)
df["dt_next"] = df["t_next"] - df["t_years"]
df["dMetAge_next"] = df["MetAge_next"] - df["MetAge"]

ana = df.replace([np.inf, -np.inf], np.nan)
ana = ana.dropna(subset=["MetAge", "MetAge_next", "dMetAge_next", "dt_next"] + pc_cols)
ana = ana[ana["dt_next"] > 0]

print("Analysis rows:", len(ana))

# baseline
ana["baseline_next_pred"] = ana["MetAge"]
ana["baseline_dMetAge_pred"] = 0.0

# model
X = ana[pc_cols].values
y = ana["dMetAge_next"].values

reg = LinearRegression()
reg.fit(X, y)

ana["model_dMetAge_pred"] = reg.predict(X)
ana["model_next_pred"] = ana["MetAge"] + ana["model_dMetAge_pred"]

# metrics
results = {
    "baseline_corr_dMetAge": ana["baseline_dMetAge_pred"].corr(ana["dMetAge_next"]),
    "model_corr_dMetAge": ana["model_dMetAge_pred"].corr(ana["dMetAge_next"]),
    "baseline_rmse_dMetAge": np.sqrt(np.mean((ana["baseline_dMetAge_pred"] - ana["dMetAge_next"])**2)),
    "model_rmse_dMetAge": np.sqrt(np.mean((ana["model_dMetAge_pred"] - ana["dMetAge_next"])**2)),
}

for col, coef in zip(pc_cols, reg.coef_):
    results[f"beta_{col}"] = coef

print("\nResults:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

os.makedirs("results", exist_ok=True)
pd.DataFrame([results]).to_csv("results/metage_dynamics_multipc_summary.csv", index=False)

print("\nSaved: results/metage_dynamics_multipc_summary.csv")
