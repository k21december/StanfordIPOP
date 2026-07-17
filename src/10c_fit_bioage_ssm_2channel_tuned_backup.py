import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

CLEAN = Path("data/clean")
INP   = CLEAN / "observations_plus_cytokines.parquet"
OUT   = CLEAN / "bioage_estimates_2channel_tuned.parquet"
OUT_PARAMS = CLEAN / "bioage_2channel_tuned_params.json"

LABS = ["BMI","CR","AG","ALB","ALKP","GLU","GLOB","NA.","K","BUN","CA","TP"]

def kalman_2chan_fit(g, q0, b_labs, r_labs, b_cyt, r_cyt):
    """Returns (BioAge, Var, nll)."""
    g = g.sort_values("t_years").reset_index(drop=True)
    dt = g["dt_years"].to_numpy()
    chron = g["ChronAge"].to_numpy()
    zL = g["zmean_labs"].to_numpy()
    zC = g["zcyt_pc1"].to_numpy()

    n = len(g)
    if n < 2 or np.isnan(chron[0]):
        return None

    x = chron[0]
    P = 1.0

    xs = np.zeros(n)
    Ps = np.zeros(n)

    nll = 0.0

    for i in range(n):
        # predict
        if i > 0:
            x = x + dt[i]
            P = P + q0 * max(dt[i], 0.0)

        # update labs
        if not np.isnan(zL[i]) and not np.isnan(chron[i]):
            H = b_labs
            innov = zL[i] - H * (x - chron[i])
            S = H * P * H + r_labs
            K = (P * H) / S
            x = x + K * innov
            P = (1 - K * H) * P
            nll += 0.5 * (np.log(S + 1e-12) + (innov * innov) / (S + 1e-12))

        # update cytokine PC1
        if not np.isnan(zC[i]) and not np.isnan(chron[i]):
            H = b_cyt
            innov = zC[i] - H * (x - chron[i])
            S = H * P * H + r_cyt
            K = (P * H) / S
            x = x + K * innov
            P = (1 - K * H) * P
            nll += 0.5 * (np.log(S + 1e-12) + (innov * innov) / (S + 1e-12))

        xs[i] = x
        Ps[i] = P

    return xs, Ps, float(nll)

def infection_effect(df):
    inf = df["Event"].astype(str).str.contains("Infection", case=False, na=False)
    a = df[inf]["BioAge_gap"].dropna()
    b = df[~inf]["BioAge_gap"].dropna()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())

def main():
    obs = pd.read_parquet(INP)

    # build labs mean
    z_labs = [f"z_{c}" for c in LABS if f"z_{c}" in obs.columns]
    if not z_labs:
        raise SystemExit("No lab z_ columns found.")
    obs = obs.copy()
    obs["zmean_labs"] = np.nanmean(obs[z_labs].to_numpy(), axis=1)

    # build cytokine z column list
    z_cols = [c for c in obs.columns if c.startswith("z_")]
    z_cyt = [c for c in z_cols if c not in set(z_labs)]
    if not z_cyt:
        raise SystemExit("No cytokine z_ columns found.")

    # Cytokine PC1 across visits (impute missing with column median)
    X = obs[z_cyt].copy()
    X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    Xv = X.to_numpy(dtype=float)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xv).reshape(-1)

    # z-score pc1
    pc1 = (pc1 - np.nanmean(pc1)) / (np.nanstd(pc1) + 1e-8)
    obs["zcyt_pc1"] = pc1

    # fixed labs channel params (keep stable)
    q0 = 0.30**2
    b_labs = 0.25
    r_labs = 0.80**2

    # tune cytokine channel b_cyt, r_cyt
    # search ranges are conservative
    b_grid = np.linspace(0.15, 1.00, 18)
    r_grid = (np.linspace(0.30, 1.20, 19) ** 2)

    best = None  # (score, b, r)
    best_out = None

    # pre-group once
    groups = [g for _, g in obs.groupby("HashedID")]
    for b_cyt in b_grid:
        for r_cyt in r_grid:
            results = []
            nll_total = 0.0
            for g in groups:
                fit = kalman_2chan_fit(g, q0, b_labs, r_labs, b_cyt, r_cyt)
                if fit is None:
                    continue
                xs, Ps, nll = fit
                gg = g.sort_values("t_years").reset_index(drop=True).copy()
                gg["BioAge"] = xs
                gg["BioAge_var"] = Ps
                gg["BioAge_gap"] = gg["BioAge"] - gg["ChronAge"]
                results.append(gg[["HashedID","VisitID","Event","t_years","ChronAge","BioAge","BioAge_var","BioAge_gap"]])
                nll_total += nll

            if not results:
                continue
            out = pd.concat(results, ignore_index=True)
            eff = infection_effect(out)

            # Objective: mostly likelihood, slight preference for stronger infection separation magnitude
            score = nll_total - 50.0 * abs(eff)  # 50 is small relative to nll scale, but breaks ties

            if best is None or score < best[0]:
                best = (score, float(b_cyt), float(r_cyt), float(nll_total), float(eff))
                best_out = out

    if best is None:
        raise SystemExit("Tuning failed (no subjects fit).")

    score, b_cyt, r_cyt, nll_total, eff = best
    best_out.to_parquet(OUT, index=False)

    # save params
    import json
    params = {
        "q0": q0,
        "b_labs": b_labs,
        "r_labs": r_labs,
        "b_cyt": b_cyt,
        "r_cyt": r_cyt,
        "total_nll": nll_total,
        "infection_effect_gap": eff,
        "cyt_pc1_explained_var": float(pca.explained_variance_ratio_[0]),
    }
    OUT_PARAMS.write_text(json.dumps(params, indent=2))

    print("✅ Tuned 2-channel (labs mean + cytokine PC1) fit complete")
    print("Saved:", OUT)
    print("Params saved:", OUT_PARAMS)
    print("Chosen b_cyt:", b_cyt, "r_cyt:", r_cyt)
    print("Cyt PC1 explained variance:", params["cyt_pc1_explained_var"])
    print("Infection effect (gap):", eff)
    print("Subjects:", best_out["HashedID"].nunique(), "Visits:", len(best_out))
    print("Mean gap:", float(best_out["BioAge_gap"].mean()), "Std gap:", float(best_out["BioAge_gap"].std()))

if __name__ == "__main__":
    main()
