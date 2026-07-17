import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")
INP  = CLEAN / "observations_plus_cytokines.parquet"
OUT  = CLEAN / "bioage_estimates_2channel.parquet"

# your clinical labs (same as in observations.parquet)
LABS = ["BMI","CR","AG","ALB","ALKP","GLU","GLOB","NA.","K","BUN","CA","TP"]

def main():
    obs = pd.read_parquet(INP)

    # Build two observation channels
    z_labs = [f"z_{c}" for c in LABS if f"z_{c}" in obs.columns]
    z_cols = [c for c in obs.columns if c.startswith("z_")]
    z_cyt  = [c for c in z_cols if c not in set(z_labs)]

    if not z_labs:
        raise SystemExit("No lab z_ columns found.")
    if not z_cyt:
        raise SystemExit("No cytokine z_ columns found.")

    # Precompute channel means
    obs = obs.copy()
    obs["zmean_labs"] = np.nanmean(obs[z_labs].to_numpy(), axis=1)
    obs["zmean_cyt"]  = np.nanmean(obs[z_cyt].to_numpy(), axis=1)

    results = []

    # Process noise (state drift uncertainty)
    q0 = 0.30**2

    # Two measurement channels: separate loadings + noises
    b_labs = 0.25
    b_cyt  = 0.25

    r_labs = 0.80**2
    r_cyt  = 0.80**2

    for sid, g in obs.groupby("HashedID"):
        g = g.sort_values("t_years").reset_index(drop=True)

        dt = g["dt_years"].to_numpy()
        chron = g["ChronAge"].to_numpy()

        zL = g["zmean_labs"].to_numpy()
        zC = g["zmean_cyt"].to_numpy()

        n = len(g)
        if n < 2 or np.isnan(chron[0]):
            continue

        # state init
        x = chron[0]
        P = 1.0

        xs = np.zeros(n)
        Ps = np.zeros(n)

        for i in range(n):
            # Predict
            if i > 0:
                x = x + dt[i]
                P = P + q0 * max(dt[i], 0.0)

            # Update with labs channel (if present)
            if not np.isnan(zL[i]) and not np.isnan(chron[i]):
                H = b_labs
                innov = zL[i] - H * (x - chron[i])
                S = H * P * H + r_labs
                K = (P * H) / S
                x = x + K * innov
                P = (1 - K * H) * P

            # Update with cytokines channel (if present)
            if not np.isnan(zC[i]) and not np.isnan(chron[i]):
                H = b_cyt
                innov = zC[i] - H * (x - chron[i])
                S = H * P * H + r_cyt
                K = (P * H) / S
                x = x + K * innov
                P = (1 - K * H) * P

            xs[i] = x
            Ps[i] = P

        g_out = g.copy()
        g_out["BioAge"] = xs
        g_out["BioAge_var"] = Ps
        g_out["BioAge_gap"] = g_out["BioAge"] - g_out["ChronAge"]
        g_out["y_labs"] = zL
        g_out["y_cyt"]  = zC

        results.append(g_out)

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(OUT, index=False)

    print("✅ 2-channel BioAge SSM fit complete")
    print("Saved:", OUT)
    print("Subjects:", out["HashedID"].nunique())
    print("Visits:", len(out))
    print("Mean BioAge gap:", float(out["BioAge_gap"].mean()))
    print("Std BioAge gap:", float(out["BioAge_gap"].std()))

if __name__ == "__main__":
    main()
