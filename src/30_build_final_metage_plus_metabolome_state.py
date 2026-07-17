import os
import numpy as np
import pandas as pd

bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")
df = df.sort_values(["HashedID", "t_years"]).reset_index(drop=True)

pc_cols = ["met_pca_4", "met_pca_3", "met_pca_6", "met_pca_1"]

df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["t_next"] = df.groupby("HashedID")["t_years"].shift(-1)
df["dt_next"] = df["t_next"] - df["t_years"]
df["dMetAge_next"] = df["MetAge_next"] - df["MetAge"]

ana = df.dropna(subset=["MetAge", "dMetAge_next", "dt_next"] + pc_cols).copy()
ana = ana[ana["dt_next"] > 0]

print("Analysis rows:", len(ana))

# standardize PCs
for c in pc_cols:
    mu = ana[c].mean()
    sd = ana[c].std()
    if sd == 0 or pd.isna(sd):
        sd = 1.0
    ana[c + "_z"] = (ana[c] - mu) / sd

z_cols = [c + "_z" for c in pc_cols]

# fit linear model: dMetAge = b0 + b_dt * dt + sum(b_i * PC_i)
X = np.column_stack(
    [np.ones(len(ana)), ana["dt_next"].values] + [ana[c].values for c in z_cols]
)
y = ana["dMetAge_next"].values

coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

intercept = coef[0]
beta_dt = coef[1]
beta_pcs = coef[2:]

# predictions
ana["pred_dMetAge_model"] = (
    intercept
    + beta_dt * ana["dt_next"]
    + sum(beta_pcs[i] * ana[z_cols[i]] for i in range(len(z_cols)))
)

ana["pred_MetAge_next_model"] = ana["MetAge"] + ana["pred_dMetAge_model"]

# final upgraded state
ana["FinalState"] = ana["pred_MetAge_next_model"]
ana["FinalState_gap"] = ana["FinalState"] - (ana["ChronAge"] + ana["dt_next"])

# metrics
def corr(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return a.corr(b)

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

results = {
    "corr_dMetAge": corr(ana["pred_dMetAge_model"], ana["dMetAge_next"]),
    "rmse_dMetAge": rmse(ana["pred_dMetAge_model"], ana["dMetAge_next"]),
    "corr_next": corr(ana["pred_MetAge_next_model"], ana["MetAge_next"]),
}

print("\nResults:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# save
os.makedirs("results", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)

pd.DataFrame([results]).to_csv("results/final_met_model_summary.csv", index=False)

coef_df = pd.DataFrame({
    "term": ["intercept", "beta_dt"] + pc_cols,
    "coef": list(coef[:2]) + list(beta_pcs)
})
coef_df.to_csv("results/final_met_model_coefficients.csv", index=False)

ana.to_csv("results/final_met_model_rows.csv", index=False)

print("\nSaved:")
print(" - results/final_met_model_summary.csv")
print(" - results/final_met_model_coefficients.csv")
print(" - results/final_met_model_rows.csv")
