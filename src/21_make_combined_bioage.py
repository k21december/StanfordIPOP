import os
import json
import numpy as np
import pandas as pd

EST_PATH = "data/clean/bioage_estimates_2state.parquet"
PRED_PATH = "data/clean/predictive_eval.parquet"
OUT_PATH = "data/clean/bioage_estimates_2state_with_composites.parquet"
SUMMARY_JSON = "data/clean/composite_bioage_lambda_summary.json"
SUMMARY_CSV = "data/clean/composite_bioage_lambda_summary.csv"

LAMBDAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def safe_corr(a, b):
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
        return np.nan
    return float(tmp["a"].corr(tmp["b"]))

def find_target_column(df):
    candidates = [
        "future_mean_lab_z",
        "future_labs_z",
        "future_lab_z",
        "future_mean_z",
        "future_z",
        "future_outcome",
    ]
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # fallback: look for something with 'future' in it
    for c in df.columns:
        cl = c.lower()
        if "future" in cl and ("lab" in cl or "z" in cl or "mean" in cl):
            return c
    return None

def pick_merge_keys(est, pred):
    preferred_sets = [
        ["HashedID", "VisitID"],
        ["HashedID", "CollectionDate"],
        ["HashedID", "t_years"],
        ["HashedID", "ChronAge"],
    ]
    for keys in preferred_sets:
        if all(k in est.columns for k in keys) and all(k in pred.columns for k in keys):
            return keys
    common = [c for c in ["HashedID", "VisitID", "CollectionDate", "t_years", "ChronAge"] if c in est.columns and c in pred.columns]
    return common

def main():
    if not os.path.exists(EST_PATH):
        raise SystemExit(f"Missing estimates file: {EST_PATH}")

    df = pd.read_parquet(EST_PATH).copy()

    if "MetAge" not in df.columns or "ImmuneState" not in df.columns:
        raise SystemExit("MetAge and/or ImmuneState missing from estimates file.")

    # positive immune burden only
    df["ImmuneBurden_pos"] = df["ImmuneState"].clip(lower=0)

    # create composite columns
    for lam in LAMBDAS:
        col = f"FinalBioAge_lam_{lam:0.2f}".replace(".", "p")
        df[col] = df["MetAge"] + lam * df["ImmuneBurden_pos"]
        if "ChronAge" in df.columns:
            gap_col = f"{col}_gap"
            df[gap_col] = df[col] - df["ChronAge"]

    # default recommended version
    best_default_lambda = 0.20
    best_col = f"FinalBioAge_lam_{best_default_lambda:0.2f}".replace(".", "p")
    df["FinalBioAge"] = df[best_col]
    if "ChronAge" in df.columns:
        df["FinalBioAge_gap"] = df["FinalBioAge"] - df["ChronAge"]

    df.to_parquet(OUT_PATH, index=False)

    summary_rows = []

    # optional predictive evaluation against existing predictive file
    if os.path.exists(PRED_PATH):
        pred = pd.read_parquet(PRED_PATH).copy()
        target_col = find_target_column(pred)
        merge_keys = pick_merge_keys(df, pred)

        if target_col is not None and len(merge_keys) > 0:
            keep_cols = merge_keys + [target_col]
            pred_small = pred[keep_cols].copy()

            merged = df.merge(pred_small, on=merge_keys, how="inner")

            for lam in LAMBDAS:
                col = f"FinalBioAge_lam_{lam:0.2f}".replace(".", "p")
                gap_col = f"{col}_gap" if f"{col}_gap" in merged.columns else None

                row = {
                    "lambda": lam,
                    "merge_keys": ",".join(merge_keys),
                    "target_col": target_col,
                    "n_rows_merged": int(len(merged)),
                    "corr_finalbioage_vs_target": safe_corr(merged[col], merged[target_col]),
                }

                if gap_col is not None:
                    row["corr_finalbioage_gap_vs_target"] = safe_corr(merged[gap_col], merged[target_col])
                else:
                    row["corr_finalbioage_gap_vs_target"] = np.nan

                summary_rows.append(row)

            # include MetAge baseline for comparison
            base_row = {
                "lambda": "MetAge_baseline",
                "merge_keys": ",".join(merge_keys),
                "target_col": target_col,
                "n_rows_merged": int(len(merged)),
                "corr_finalbioage_vs_target": safe_corr(merged["MetAge"], merged[target_col]),
                "corr_finalbioage_gap_vs_target": safe_corr(merged["MetAge_gap"], merged[target_col]) if "MetAge_gap" in merged.columns else np.nan,
            }
            summary_rows.append(base_row)

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df.to_csv(SUMMARY_CSV, index=False)
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary_df.to_dict(orient="records"), f, indent=2)

        print("\nLambda evaluation summary:")
        print(summary_df.to_string(index=False))

        numeric = summary_df[summary_df["lambda"] != "MetAge_baseline"].copy()
        if len(numeric) > 0 and "corr_finalbioage_gap_vs_target" in numeric.columns:
            numeric["abs_rank_metric"] = numeric["corr_finalbioage_gap_vs_target"].abs()
            best = numeric.sort_values("abs_rank_metric", ascending=False).iloc[0]
            print("\nBest lambda by |corr(FinalBioAge_gap, target)|:")
            print(best.to_string())
    else:
        print("\nNo predictive evaluation summary created.")
        print("Composite columns were still written to:")
        print(OUT_PATH)

    print("\nDone.")
    print("Saved composite estimates to:", OUT_PATH)
    if len(summary_rows) > 0:
        print("Saved lambda summary CSV to:", SUMMARY_CSV)
        print("Saved lambda summary JSON to:", SUMMARY_JSON)

if __name__ == "__main__":
    main()
