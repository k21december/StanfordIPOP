import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# Load
# -----------------------------
bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")

# -----------------------------
# Prep dynamics
# -----------------------------
df = df.sort_values(["HashedID", "t_years"]).copy()

df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["t_next"] = df.groupby("HashedID")["t_years"].shift(-1)

df["dt"] = df["t_next"] - df["t_years"]
df["dMetAge_next"] = df["MetAge_next"] - df["MetAge"]

df = df.dropna()

pc_cols = ["met_pca_4", "met_pca_3", "met_pca_6", "met_pca_1"]

# -----------------------------
# Model A: baseline (MetAge only)
# -----------------------------
X_A = df[["MetAge"]].values
y = df["dMetAge_next"].values

model_A = LinearRegression().fit(X_A, y)
pred_A = model_A.predict(X_A)

# -----------------------------
# Model B: MetAge + metabolome
# -----------------------------
X_B = df[["MetAge"] + pc_cols].values

model_B = LinearRegression().fit(X_B, y)
pred_B = model_B.predict(X_B)

# -----------------------------
# Metrics
# -----------------------------
r2_A = r2_score(y, pred_A)
r2_B = r2_score(y, pred_B)

corr_A = np.corrcoef(pred_A, y)[0,1]
corr_B = np.corrcoef(pred_B, y)[0,1]

print("\nDriver vs Proxy Test\n")

print("Model A (MetAge only):")
print("R2:", round(r2_A, 4))
print("Corr:", round(corr_A, 4))

print("\nModel B (MetAge + metabolome):")
print("R2:", round(r2_B, 4))
print("Corr:", round(corr_B, 4))

print("\nImprovement:")
print("ΔR2:", round(r2_B - r2_A, 4))
print("ΔCorr:", round(corr_B - corr_A, 4))
