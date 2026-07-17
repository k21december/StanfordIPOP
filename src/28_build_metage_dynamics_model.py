import pandas as pd
import numpy as np

# load data
bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

# align IDs
met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")

# normalize driver
df["met_pca_4_z"] = (df["met_pca_4"] - df["met_pca_4"].mean()) / df["met_pca_4"].std()

# sort + create future targets
df = df.sort_values(["HashedID", "t_years"]).copy()
df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["dMetAge"] = df["MetAge_next"] - df["MetAge"]

# filter valid rows
ana = df[df["dMetAge"].notna()].copy()

print("Analysis rows:", len(ana))

# --- BASELINE ---
ana["baseline_pred_next"] = ana["MetAge"]

# --- MODEL: MetAge + beta * met_pca_4 ---
# fit beta via simple regression
beta = np.cov(ana["met_pca_4_z"], ana["dMetAge"])[0,1] / np.var(ana["met_pca_4_z"])

ana["model_pred_dMetAge"] = beta * ana["met_pca_4_z"]
ana["model_pred_next"] = ana["MetAge"] + ana["model_pred_dMetAge"]

# --- EVALUATION ---
results = {
    "baseline_corr_next": ana["baseline_pred_next"].corr(ana["MetAge_next"]),
    "model_corr_next": ana["model_pred_next"].corr(ana["MetAge_next"]),
    "baseline_corr_dMetAge": ana["baseline_pred_next"].corr(ana["dMetAge"]),
    "model_corr_dMetAge": ana["model_pred_dMetAge"].corr(ana["dMetAge"]),
    "beta": beta
}

print("\nResults:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# save outputs
pd.DataFrame([results]).to_csv("results/metage_dynamics_summary.csv", index=False)
ana.to_csv("results/metage_dynamics_predictions.csv", index=False)

print("\nSaved:")
print(" - results/metage_dynamics_summary.csv")
print(" - results/metage_dynamics_predictions.csv")
