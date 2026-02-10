import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")

def rmse(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan

def mae(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if len(x) else np.nan

def main():
    df = pd.read_parquet(CLEAN / "bioage_estimates.parquet").copy()
    keep = ["HashedID","VisitID","t_years","dt_years","ChronAge","BioAge","BioAge_gap","ybar_mean_z","Sex","RaceLabel"]
    df = df[keep].sort_values(["HashedID","t_years","VisitID"]).reset_index(drop=True)

    rows = []
    for sid, g in df.groupby("HashedID"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g)-1):
            dt = g.loc[i+1, "t_years"] - g.loc[i, "t_years"]
            if not np.isfinite(dt) or dt <= 0:
                continue

            x_pred = g.loc[i, "BioAge"] + dt
            x_true = g.loc[i+1, "BioAge"]

            ca_next = g.loc[i+1, "ChronAge"]
            gap_pred = x_pred - ca_next
            gap_true = g.loc[i+1, "BioAge_gap"]

            b = 0.25
            y_pred = b * gap_pred
            y_true = g.loc[i+1, "ybar_mean_z"]

            rows.append({
                "HashedID": sid,
                "Sex": g.loc[i+1, "Sex"],
                "RaceLabel": g.loc[i+1, "RaceLabel"],
                "dt": dt,
                "x_pred": x_pred,
                "x_true": x_true,
                "gap_pred": gap_pred,
                "gap_true": gap_true,
                "y_pred": y_pred,
                "y_true": y_true,
                "err_x": x_pred - x_true,
                "err_gap": gap_pred - gap_true,
                "err_y": y_pred - y_true,
            })

    ev = pd.DataFrame(rows)
    if ev.empty:
        raise SystemExit("No evaluation rows created (check time ordering / dt).")

    print("=== ONE-STEP AHEAD METRICS (overall) ===")
    print("N predictions:", len(ev))
    print("BioAge RMSE (years):", rmse(ev["err_x"]))
    print("BioAge MAE  (years):", mae(ev["err_x"]))
    print("Gap RMSE    (years):", rmse(ev["err_gap"]))
    print("Gap MAE     (years):", mae(ev["err_gap"]))
    print("ybar RMSE   (z):    ", rmse(ev["err_y"]))
    print("ybar MAE    (z):    ", mae(ev["err_y"]))

    print("\n=== BY SEX ===")
    for sex, g in ev.groupby("Sex"):
        print(f"\nSex={sex}  n_pred={len(g)}  n_sub={g['HashedID'].nunique()}")
        print("BioAge RMSE:", rmse(g["err_x"]), " MAE:", mae(g["err_x"]))
        print("Gap   RMSE:", rmse(g["err_gap"]), " MAE:", mae(g["err_gap"]))
        print("ybar  RMSE:", rmse(g["err_y"]), " MAE:", mae(g["err_y"]))

        print("Mean signed Gap error (years):", float(np.nanmean(g["err_gap"])))

    outp = CLEAN / "predictive_eval.parquet"
    ev.to_parquet(outp, index=False)
    print("\nSaved:", outp)

if __name__ == "__main__":
    main()
