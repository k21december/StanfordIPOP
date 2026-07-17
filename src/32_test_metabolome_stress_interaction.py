import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load data
# -----------------------------
bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")
met = pd.read_csv("data/clean/metabolome_pca.csv")

met["HashedID"] = met["SampleID"].str.split("-").str[0]
met["VisitID"] = met["SampleID"]

df = pd.merge(bio, met, on=["HashedID", "VisitID"], how="inner")

# -----------------------------
# Choose metabolome PCs
# -----------------------------
pc_cols = ["met_pca_4", "met_pca_3", "met_pca_6", "met_pca_1"]

# -----------------------------
# Build next-step dynamics table
# -----------------------------
df = df.sort_values(["HashedID", "t_years"]).copy()

df["MetAge_next"] = df.groupby("HashedID")["MetAge"].shift(-1)
df["t_next"] = df.groupby("HashedID")["t_years"].shift(-1)

df["dt"] = df["t_next"] - df["t_years"]
df["dMetAge_next"] = df["MetAge_next"] - df["MetAge"]

# baseline aging term
df["baseline_dMetAge"] = df["dt"]

# residual change beyond plain time
df["resid_dMetAge_next"] = df["dMetAge_next"] - df["baseline_dMetAge"]

keep_cols = [
    "HashedID",
    "VisitID",
    "Event",
    "MetAge",
    "MetAge_next",
    "t_years",
    "t_next",
    "dt",
    "dMetAge_next",
    "baseline_dMetAge",
    "resid_dMetAge_next",
] + pc_cols

ana = df[keep_cols].dropna().copy()

# -----------------------------
# Define stress groups
# -----------------------------
stress_events = {
    "Infection",
    "Infection_Early",
    "Infection_Middle",
    "Infection_Late",
    "Imz",
    "Imz_L",
    "Weight-loss",
    "Weight-gain",
}

ana["stress_flag"] = ana["Event"].isin(stress_events).astype(int)
ana["event_group"] = np.where(ana["stress_flag"] == 1, "stress", "baseline")

# -----------------------------
# Helper: evaluate linear model
# -----------------------------
def eval_model(subdf, label):
    out = {"group": label, "n": len(subdf)}

    if len(subdf) < 20:
        out.update({
            "corr_pred_vs_true": np.nan,
            "rmse": np.nan,
            "r2_in_sample": np.nan,
        })
        for c in ["intercept"] + pc_cols:
            out[c if c == "intercept" else f"beta_{c}"] = np.nan
        return out

    X = subdf[pc_cols].values
    y = subdf["resid_dMetAge_next"].values

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    corr = np.corrcoef(pred, y)[0, 1]
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = model.score(X, y)

    out.update({
        "corr_pred_vs_true": corr,
        "rmse": rmse,
        "r2_in_sample": r2,
        "intercept": model.intercept_,
    })

    for c, b in zip(pc_cols, model.coef_):
        out[f"beta_{c}"] = b

    return out

# -----------------------------
# Evaluate baseline vs stress
# -----------------------------
rows = []
rows.append(eval_model(ana, "all"))
rows.append(eval_model(ana[ana["event_group"] == "baseline"], "baseline"))
rows.append(eval_model(ana[ana["event_group"] == "stress"], "stress"))

for ev, sub in ana.groupby("Event"):
    rows.append(eval_model(sub, f"event::{ev}"))

res = pd.DataFrame(rows)

# -----------------------------
# Simple direct correlation table
# -----------------------------
corr_rows = []
for group_name, sub in {
    "all": ana,
    "baseline": ana[ana["event_group"] == "baseline"],
    "stress": ana[ana["event_group"] == "stress"],
}.items():
    row = {"group": group_name, "n": len(sub)}
    for pc in pc_cols:
        row[f"corr_{pc}_vs_resid"] = sub[pc].corr(sub["resid_dMetAge_next"])
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows)

# -----------------------------
# Save outputs
# -----------------------------
Path("results").mkdir(exist_ok=True)

res.to_csv("results/metabolome_stress_model_summary.csv", index=False)
corr_df.to_csv("results/metabolome_stress_corrs.csv", index=False)
ana.to_csv("results/metabolome_stress_analysis_rows.csv", index=False)

print("Analysis rows:", len(ana))
print("\nCounts by Event:")
print(ana["Event"].value_counts().to_string())

print("\nModel summary:")
print(
    res[["group", "n", "corr_pred_vs_true", "rmse", "r2_in_sample"]]
    .to_string(index=False)
)

print("\nDirect metabolome-vs-residual correlations:")
print(corr_df.to_string(index=False))

print("\nSaved:")
print(" - results/metabolome_stress_model_summary.csv")
print(" - results/metabolome_stress_corrs.csv")
print(" - results/metabolome_stress_analysis_rows.csv")
