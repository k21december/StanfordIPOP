import numpy as np
import pandas as pd

bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")
df = df.sort_values(["HashedID", "t_years"])

pc_cols = ["met_pca_4", "met_pca_3", "met_pca_6", "met_pca_1"]

# future values
df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["t_next"] = df.groupby("HashedID")["t_years"].shift(-1)
df["dt"] = df["t_next"] - df["t_years"]
df["dMetAge"] = df["MetAge_next"] - df["MetAge"]

# residual (THIS is key)
df["dMetAge_residual"] = df["dMetAge"] - df["dt"]

ana = df.dropna(subset=["dMetAge_residual", "dt"] + pc_cols)
ana = ana[ana["dt"] > 0]

print("Analysis rows:", len(ana))

# standardize PCs
for c in pc_cols:
    mu = ana[c].mean()
    sd = ana[c].std()
    if sd == 0 or pd.isna(sd):
        sd = 1.0
    ana[c + "_z"] = (ana[c] - mu) / sd

z_cols = [c + "_z" for c in pc_cols]

# fit model
X = np.column_stack([np.ones(len(ana))] + [ana[c].values for c in z_cols])
y = ana["dMetAge_residual"].values

coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

intercept = coef[0]
betas = coef[1:]

# predictions
ana["pred_residual"] = intercept + sum(betas[i] * ana[z_cols[i]] for i in range(len(z_cols)))

# reconstruct full prediction
ana["pred_dMetAge"] = ana["dt"] + ana["pred_residual"]

# metrics
corr = ana["pred_residual"].corr(ana["dMetAge_residual"])

rmse = np.sqrt(np.mean((ana["pred_residual"] - ana["dMetAge_residual"])**2))

print("\nResults (REAL signal):")
print(f"corr_residual: {corr:.4f}")
print(f"rmse_residual: {rmse:.4f}")

# save
pd.DataFrame({
    "term": ["intercept"] + pc_cols,
    "coef": list(coef)
}).to_csv("results/final_clean_coefficients.csv", index=False)

ana.to_csv("results/final_clean_predictions.csv", index=False)

print("\nSaved clean model outputs.")
