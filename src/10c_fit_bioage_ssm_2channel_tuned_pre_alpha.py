import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

CLEAN = Path("data/clean")

if len(sys.argv) >= 3:
    INP = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
else:
    INP = CLEAN / "observations_plus_cytokines.parquet"
    OUT = CLEAN / "bioage_estimates_2channel_tuned.parquet"

if len(sys.argv) >= 4:
    OUT_PARAMS = Path(sys.argv[3])
else:
    OUT_PARAMS = OUT.with_name(OUT.stem + "_params.json")

LABS = ["BMI", "CR", "AG", "ALB", "ALKP", "GLU", "GLOB", "NA.", "K", "BUN", "CA", "TP"]


def kalman_2chan_fit(g, q0, b_labs, r_labs, b_cyt, r_cyt):
    """
    Fit a 1D latent biological age state using two observation channels:
      1) mean z-scored labs
      2) cytokine PC1

    Returns:
      xs, Ps, nll
    """
    g = g.sort_values("t_years").reset_index(drop=True)
    dt = g["dt_years"].to_numpy(dtype=float)
    chron = g["ChronAge"].to_numpy(dtype=float)
    zL = g["zmean_labs"].to_numpy(dtype=float)
    zC = g["zcyt_pc1"].to_numpy(dtype=float)

    n = len(g)
    if n < 2 or np.isnan(chron[0]):
        return None

    x = chron[0]
    P = 1.0

    xs = np.zeros(n, dtype=float)
    Ps = np.zeros(n, dtype=float)
    nll = 0.0

    for i in range(n):
        # Predict step
        if i > 0:
            dt_i = max(float(dt[i]), 0.0) if np.isfinite(dt[i]) else 0.0
            x = x + dt_i
            P = P + q0 * dt_i

        # Update with labs channel
        if np.isfinite(zL[i]) and np.isfinite(chron[i]):
            H = b_labs
            innov = zL[i] - H * (x - chron[i])
            S = H * P * H + r_labs
            if S > 0:
                K = (P * H) / S
                x = x + K * innov
                P = (1 - K * H) * P
                P = max(P, 1e-8)
                nll += 0.5 * (np.log(S + 1e-12) + (innov * innov) / (S + 1e-12))

        # Update with cytokine channel
        if np.isfinite(zC[i]) and np.isfinite(chron[i]):
            H = b_cyt
            innov = zC[i] - H * (x - chron[i])
            S = H * P * H + r_cyt
            if S > 0:
                K = (P * H) / S
                x = x + K * innov
                P = (1 - K * H) * P
                P = max(P, 1e-8)
                nll += 0.5 * (np.log(S + 1e-12) + (innov * innov) / (S + 1e-12))

        xs[i] = x
        Ps[i] = P

    return xs, Ps, float(nll)


def infection_effect(df):
    if "Event" not in df.columns:
        return 0.0
    inf = df["Event"].astype(str).str.contains("Infection", case=False, na=False)
    a = df.loc[inf, "BioAge_gap"].dropna()
    b = df.loc[~inf, "BioAge_gap"].dropna()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())


def build_cytokine_pc1(obs, z_labs):
    z_cols = [c for c in obs.columns if c.startswith("z_")]
    z_cyt = [c for c in z_cols if c not in set(z_labs)]
    if not z_cyt:
        raise SystemExit("No cytokine z_ columns found.")

    X = obs[z_cyt].copy()
    X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    Xv = X.to_numpy(dtype=float)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xv).reshape(-1)
    pc1 = (pc1 - np.nanmean(pc1)) / (np.nanstd(pc1) + 1e-8)

    return pc1, z_cyt, float(pca.explained_variance_ratio_[0])


def score_fit(df, nll_total):
    """
    Lower is better.
    Tune for:
      - good fit (low nll)
      - centered gap
      - reasonable spread
      - slight positive infection effect
    """
    mean_gap = float(df["BioAge_gap"].mean())
    std_gap = float(df["BioAge_gap"].std())
    inf_eff = infection_effect(df)

    score = float(nll_total)
    score += 200.0 * abs(mean_gap)                  # keep centered near zero
    score += 25.0 * abs(std_gap - 0.85)            # keep plausible spread
    if inf_eff < 0:
        score += 50.0 * abs(inf_eff)               # penalize wrong-direction infection effect
    else:
        score -= 10.0 * min(inf_eff, 0.10)         # mild reward for positive infection effect

    return score, mean_gap, std_gap, inf_eff


def main():
    obs = pd.read_parquet(INP).copy()

    required_cols = ["HashedID", "VisitID", "t_years", "dt_years", "ChronAge"]
    missing_required = [c for c in required_cols if c not in obs.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns: {missing_required}")

    # Build labs mean channel
    z_labs = [f"z_{c}" for c in LABS if f"z_{c}" in obs.columns]
    if not z_labs:
        raise SystemExit("No lab z_ columns found.")
    obs["zmean_labs"] = np.nanmean(obs[z_labs].to_numpy(dtype=float), axis=1)

    # Build cytokine PC1
    pc1, z_cyt, pc1_var = build_cytokine_pc1(obs, z_labs)
    obs["zcyt_pc1"] = pc1

    # Fixed labs channel params
    q0 = 0.30 ** 2
    b_labs = 0.25
    r_labs = 0.80 ** 2

    # Conservative cytokine tuning grid
    b_grid = np.linspace(0.15, 1.00, 18)
    r_grid = np.linspace(0.30, 1.20, 19) ** 2

    groups = [g.copy() for _, g in obs.groupby("HashedID")]
    best = None
    best_out = None

    for b_cyt in b_grid:
        for r_cyt in r_grid:
            results = []
            nll_total = 0.0

            for g in groups:
                fit = kalman_2chan_fit(g, q0, b_labs, r_labs, float(b_cyt), float(r_cyt))
                if fit is None:
                    continue

                xs, Ps, nll = fit
                tmp = g.sort_values("t_years").reset_index(drop=True).copy()
                tmp["BioAge"] = xs
                tmp["BioAge_var"] = Ps
                tmp["BioAge_sd"] = np.sqrt(np.maximum(Ps, 1e-12))
                tmp["BioAge_gap"] = tmp["BioAge"] - tmp["ChronAge"]

                results.append(tmp)
                nll_total += float(nll)

            if not results:
                continue

            out = pd.concat(results, ignore_index=True)
            score, mean_gap, std_gap, inf_eff = score_fit(out, nll_total)

            if best is None or score < best["score"]:
                best = {
                    "score": float(score),
                    "q0": float(q0),
                    "b_labs": float(b_labs),
                    "r_labs": float(r_labs),
                    "b_cyt": float(b_cyt),
                    "r_cyt": float(r_cyt),
                    "mean_gap": float(mean_gap),
                    "std_gap": float(std_gap),
                    "infection_effect_gap": float(inf_eff),
                    "pc1_explained_variance": float(pc1_var),
                    "n_subjects": int(out["HashedID"].nunique()),
                    "n_visits": int(len(out)),
                    "input_file": str(INP),
                    "output_file": str(OUT),
                    "lab_features": z_labs,
                    "cytokine_features": z_cyt,
                }
                best_out = out.copy()

    if best is None or best_out is None:
        raise SystemExit("Model fitting failed: no valid parameter combination produced output.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARAMS.parent.mkdir(parents=True, exist_ok=True)

    best_out.to_parquet(OUT, index=False)
    with open(OUT_PARAMS, "w") as f:
        json.dump(best, f, indent=2)

    print("✅ Tuned 2-channel (labs mean + cytokine PC1) fit complete")
    print(f"Saved: {OUT}")
    print(f"Params saved: {OUT_PARAMS}")
    print(f"Chosen b_cyt: {best['b_cyt']} r_cyt: {best['r_cyt']}")
    print(f"Cyt PC1 explained variance: {best['pc1_explained_variance']}")
    print(f"Infection effect (gap): {best['infection_effect_gap']}")
    print(f"Subjects: {best['n_subjects']} Visits: {best['n_visits']}")
    print(f"Mean gap: {best['mean_gap']} Std gap: {best['std_gap']}")


if __name__ == "__main__":
    main()
