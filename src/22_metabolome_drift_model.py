import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------- Load ----------
df = pd.read_parquet("data/clean/multimodal_channels_v1.parquet").copy()
old = pd.read_parquet("data/clean/bioage_estimates_2state.parquet")[["HashedID","VisitID","MetAge"]]

df = df.merge(old, on=["HashedID","VisitID"], how="inner")
df = df.sort_values(["HashedID","t_years"])

# ---------- Targets ----------
df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["dMetAge"] = df["MetAge_next"] - df["MetAge"]

met_cols = [c for c in df.columns if c.startswith("met_pca_")]

sub = df[["HashedID","VisitID","t_years","dMetAge"] + met_cols].dropna()

# ---------- Train ML drift model ----------
X = sub[met_cols]
y = sub["dMetAge"]

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
model.fit(X, y)

# ---------- Predict drift ----------
sub["drift_pred"] = model.predict(X)

# ---------- Save ----------
sub[["HashedID","VisitID","t_years","drift_pred"]].to_parquet(
    "data/clean/metabolome_drift.parquet", index=False
)

print("Saved metabolome-driven drift model.")
