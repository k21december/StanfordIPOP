import json
import numpy as np
import pandas as pd

COMP_PATH = "data/clean/bioage_estimates_2state_with_composites.parquet"
PRED_PATH = "data/clean/predictive_eval.parquet"
OUT_CSV = "data/clean/composite_bioage_lambda_summary.csv"
OUT_JSON = "data/clean/composite_bioage_lambda_summary.json"

def safe_corr(a, b):
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
        return np.nan
    return float(tmp["a"].corr(tmp["b"]))

# load files
df = pd.read_parquet(COMP_PATH).copy()
pred = pd.read_parquet(PRED_PATH).copy()

# rebuild transition rows from composite file
df = df.sort_values(["HashedID", "t_years"]).reset_index(drop=True)
g = df.groupby("HashedID", sort=False)

df["t_next"] = g["t_years"].shift(-1)
df["dt"] = df["t_next"] - df["t_years"]

# keep only rows that have a next visit
trans = df[df["dt"].notna() & (df["dt"] > 0)].copy()

# build merge keys matching predictive_eval structure
merge_keys = ["HashedID", "Sex", "RaceLabel", "dt"]

# round dt a bit to avoid floating-point nonsense
trans["dt"] = trans["dt"].round(6)
pred["dt"] = pred["dt"].round(6)

summary_rows = []

lambda_cols = [c for c in trans.columns if c.startswith("FinalBioAge_lam_") and not c.endswith("_gap")]
lambda_cols = sorted(lambda_cols)

for col in lambda_cols:
    gap_col = f"{col}_gap"
    keep = merge_keys + [col]
    if gap_col in trans.columns:
        keep.append(gap_col)

    merged = trans[keep].merge(pred, on=merge_keys, how="inner")

    row = {
        "lambda_col": col,
        "n_rows_merged": int(len(merged)),
        "corr_finalbioage_vs_xtrue": safe_corr(merged[col], merged["x_true"]),
        "corr_finalbioage_gap_vs_gaptrue": safe_corr(merged[gap_col], merged["gap_true"]) if gap_col in merged.columns else np.nan,
        "corr_finalbioage_gap_vs_ytrue": safe_corr(merged[gap_col], merged["y_true"]) if gap_col in merged.columns else np.nan,
        "corr_finalbioage_vs_ytrue": safe_corr(merged[col], merged["y_true"]),
    }

    # decode numeric lambda from column name
    lam_txt = col.replace("FinalBioAge_lam_", "").replace("p", ".")
    try:
        row["lambda"] = float(lam_txt)
    except Exception:
        row["lambda"] = np.nan

    summary_rows.append(row)

# baseline MetAge for comparison
base_keep = merge_keys + ["MetAge", "MetAge_gap"]
base = trans[base_keep].merge(pred, on=merge_keys, how="inner")

summary_rows.append({
    "lambda_col": "MetAge_baseline",
    "lambda": -1.0,
    "n_rows_merged": int(len(base)),
    "corr_finalbioage_vs_xtrue": safe_corr(base["MetAge"], base["x_true"]),
    "corr_finalbioage_gap_vs_gaptrue": safe_corr(base["MetAge_gap"], base["gap_true"]),
    "corr_finalbioage_gap_vs_ytrue": safe_corr(base["MetAge_gap"], base["y_true"]),
    "corr_finalbioage_vs_ytrue": safe_corr(base["MetAge"], base["y_true"]),
})

summary = pd.DataFrame(summary_rows)

# sort nicely: baseline last
summary_numeric = summary[summary["lambda"] >= 0].sort_values("lambda")
summary_base = summary[summary["lambda"] < 0]
summary = pd.concat([summary_numeric, summary_base], ignore_index=True)

summary.to_csv(OUT_CSV, index=False)
with open(OUT_JSON, "w") as f:
    json.dump(summary.to_dict(orient="records"), f, indent=2)

print("Saved:", OUT_CSV)
print("Saved:", OUT_JSON)
print("\nSummary:")
print(summary.to_string(index=False))

best = summary[summary["lambda"] >= 0].sort_values("corr_finalbioage_gap_vs_ytrue", ascending=False).iloc[-1:]
# ignore this accidental line by replacing with proper selection below
best = summary[summary["lambda"] >= 0].copy()
best = best.sort_values("corr_finalbioage_gap_vs_ytrue", ascending=False).iloc[0]

print("\nBest lambda by corr_finalbioage_gap_vs_ytrue:")
print(best.to_string())
