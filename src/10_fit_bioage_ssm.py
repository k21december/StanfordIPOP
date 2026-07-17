import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")
OUT = CLEAN / "bioage_estimates.parquet"

def main():
    obs = pd.read_parquet(CLEAN / "observations_plus_cytokines.parquet")

    z_cols = [c for c in obs.columns if c.startswith("z_")]
    if not z_cols:
        raise SystemExit("No z-scored lab columns found (columns starting with 'z_').")

    results = []

    q0 = 0.30**2
    r0 = 0.80**2
    b  = 0.25

    for sid, g in obs.groupby("HashedID"):
        g = g.sort_values("t_years").reset_index(drop=True)

        t_years = g["t_years"].to_numpy()
        dt = g["dt_years"].to_numpy()
        chron = g["ChronAge"].to_numpy()

        Y = g[z_cols].to_numpy()
        y_bar = np.nanmean(Y, axis=1)

        n = len(g)
        if n < 2:
            continue

        if np.isnan(chron[0]):
            continue

        x = chron[0]
        P = 1.0

        xs = np.zeros(n)
        Ps = np.zeros(n)

        for i in range(n):
            if i > 0:
                x = x + dt[i]
                P = P + q0 * max(dt[i], 0.0)

            if not np.isnan(y_bar[i]) and not np.isnan(chron[i]):
                H = b
                z = y_bar[i]

                innov = z - H * (x - chron[i])

                S = H * P * H + r0
                K = (P * H) / S

                x = x + K * innov
                P = (1 - K * H) * P

            xs[i] = x
            Ps[i] = P

        g_out = g.copy()
        g_out["BioAge"] = xs
        g_out["BioAge_var"] = Ps
        g_out["BioAge_gap"] = g_out["BioAge"] - g_out["ChronAge"]
        g_out["ybar_mean_z"] = y_bar

        results.append(g_out)

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(OUT, index=False)

    print("✅ BioAge SSM fit complete")
    print("Saved:", OUT)
    print("Subjects:", out["HashedID"].nunique())
    print("Visits:", len(out))
    print("Mean BioAge gap:", float(out["BioAge_gap"].mean()))
    print("Std BioAge gap:", float(out["BioAge_gap"].std()))

if __name__ == "__main__":
    main()
