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
    OUT = CLEAN / "bioage_estimates_2state.parquet"

if len(sys.argv) >= 4:
    OUT_PARAMS = Path(sys.argv[3])
else:
    OUT_PARAMS = OUT.with_name(OUT.stem + "_params.json")

LABS = ["BMI", "CR", "AG", "ALB", "ALKP", "GLU", "GLOB", "NA.", "K", "BUN", "CA", "TP"]


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


def kalman_2state_fit(g, q1, q2, alpha, rho, b_labs, r_labs, b_cyt, r_cyt):
    g = g.sort_values("t_years").reset_index(drop=True)
    dt = g["dt_years"].to_numpy(dtype=float)
    chron = g["ChronAge"].to_numpy(dtype=float)
    zL = g["zmean_labs"].to_numpy(dtype=float)
    zC = g["zcyt_pc1"].to_numpy(dtype=float)

    n = len(g)
    if n < 2 or np.isnan(chron[0]):
        return None

    # x = [metabolic_age, immune_state]
    x = np.array([chron[0], 0.0], dtype=float)
    P = np.eye(2, dtype=float)

    xs = np.zeros((n, 2), dtype=float)
    Ps = np.zeros((n, 2, 2), dtype=float)
    nll = 0.0

    I = np.eye(2, dtype=float)

    for i in range(n):
        if i > 0:
            dt_i = max(float(dt[i]), 0.0) if np.isfinite(dt[i]) else 0.0

            # Predict
            F = np.array([
                [1.0, 0.0],
                [0.0, rho]
            ], dtype=float)

            x = np.array([
                x[0] + alpha * dt_i,
                rho * x[1]
            ], dtype=float)

            Q = np.array([
                [q1 * dt_i, 0.0],
                [0.0, q2]
            ], dtype=float)

            P = F @ P @ F.T + Q

        # Labs observe metabolic age gap
        if np.isfinite(zL[i]) and np.isfinite(chron[i]):
            H = np.array([[b_labs, 0.0]], dtype=float)
            pred = H @ np.array([[x[0] - chron[i]], [x[1]]], dtype=float)
            innov = float(zL[i] - pred[0, 0])
            S = float((H @ P @ H.T)[0, 0] + r_labs)
            if S > 0:
                K = (P @ H.T) / S
                x = x + (K.flatten() * innov)
                P = (I - K @ H) @ P
                P = P + 1e-8 * I
                nll += 0.5 * (np.log(S + 1e-12) + (innov * innov) / (S + 1e-12))

        # Cytokines observe immune state
        if np.isfinite(zC[i]):
            H = np.array([[0.0, b_cyt]], dtype=float)
            pred = H @ x.reshape(-1, 1)
            innov = float(zC[i] - pred[0, 0])
            S = float((H @ P @ H.T)[0, 0] + r_cyt)
            if S > 0:
                K = (P @ H.T) / S
                x = x + (K.flatten() * innov)
                P = (I - K @ H) @ P
                P = P + 1e-8 * I
                nll += 0.5 * (np.log(S + 1e-12) + (innov * innov) / (S + 1e-12))

        xs[i] = x
        Ps[i] = P

    return xs, Ps, float(nll)


def infection_effect(df):
    if "Event" not in df.columns:
        return 0.0
    inf = df["Event"].astype(str).str.contains("Infection", case=False, na=False)
    a = df.loc[inf, "MetAge_gap"].dropna()
    b = df.loc[~inf, "MetAge_gap"].dropna()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())


def score_fit(df, nll_total):
    mean_gap = float(df["MetAge_gap"].mean())
    std_gap = float(df["MetAge_gap"].std())
    inf_eff = infection_effect(df)

    score = float(nll_total)
    score += 200.0 * abs(mean_gap)
    score += 25.0 * abs(std_gap - 0.85)

    return score, mean_gap, std_gap, inf_eff


def main():
    obs = pd.read_parquet(INP).copy()

    required_cols = ["HashedID", "VisitID", "t_years", "dt_years", "ChronAge"]
    missing_required = [c for c in required_cols if c not in obs.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns: {missing_required}")

    z_labs = [f"z_{c}" for c in LABS if f"z_{c}" in obs.columns]
    if not z_labs:
        raise SystemExit("No lab z_ columns found.")
    obs["zmean_labs"] = np.nanmean(obs[z_labs].to_numpy(dtype=float), axis=1)

    pc1, z_cyt, pc1_var = build_cytokine_pc1(obs, z_labs)
    obs["zcyt_pc1"] = pc1

    # keep search modest so your laptop doesn't age before the model does
    q1_grid = np.array([0.20, 0.30]) ** 2
    q2_grid = np.array([0.10, 0.20]) ** 2
    alpha_grid = np.array([0.90, 1.00])
    rho_grid = np.array([0.75, 0.90])
    b_cyt_grid = np.array([0.75, 1.00])
    r_cyt_grid = np.array([0.40, 0.60]) ** 2

    b_labs = 0.25
    r_labs = 0.80 ** 2

    groups = [g.copy() for _, g in obs.groupby("HashedID")]
    best = None
    best_out = None

    for q1 in q1_grid:
        for q2 in q2_grid:
            for alpha in alpha_grid:
                for rho in rho_grid:
                    for b_cyt in b_cyt_grid:
                        for r_cyt in r_cyt_grid:
                            results = []
                            nll_total = 0.0

                            for g in groups:
                                fit = kalman_2state_fit(
                                    g, float(q1), float(q2), float(alpha), float(rho),
                                    float(b_labs), float(r_labs), float(b_cyt), float(r_cyt)
                                )
                                if fit is None:
                                    continue

                                xs, Ps, nll = fit
                                tmp = g.sort_values("t_years").reset_index(drop=True).copy()
                                tmp["MetAge"] = xs[:, 0]
                                tmp["ImmuneState"] = xs[:, 1]
                                tmp["MetAge_var"] = Ps[:, 0, 0]
                                tmp["Immune_var"] = Ps[:, 1, 1]
                                tmp["MetAge_sd"] = np.sqrt(np.maximum(Ps[:, 0, 0], 1e-12))
                                tmp["Immune_sd"] = np.sqrt(np.maximum(Ps[:, 1, 1], 1e-12))
                                tmp["MetAge_gap"] = tmp["MetAge"] - tmp["ChronAge"]

                                results.append(tmp)
                                nll_total += float(nll)

                            if not results:
                                continue

                            out = pd.concat(results, ignore_index=True)
                            score, mean_gap, std_gap, inf_eff = score_fit(out, nll_total)

                            if best is None or score < best["score"]:
                                best = {
                                    "score": float(score),
                                    "q1": float(q1),
                                    "q2": float(q2),
                                    "alpha": float(alpha),
                                    "rho": float(rho),
                                    "b_labs": float(b_labs),
                                    "r_labs": float(r_labs),
                                    "b_cyt": float(b_cyt),
                                    "r_cyt": float(r_cyt),
                                    "mean_gap": float(mean_gap),
                                    "std_gap": float(std_gap),
                                    "infection_effect_met_gap": float(inf_eff),
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
        raise SystemExit("2-state model fitting failed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARAMS.parent.mkdir(parents=True, exist_ok=True)

    best_out.to_parquet(OUT, index=False)
    with open(OUT_PARAMS, "w") as f:
        json.dump(best, f, indent=2)

    print("✅ 2-state model fit complete")
    print(f"Saved: {OUT}")
    print(f"Params saved: {OUT_PARAMS}")
    print(f"Chosen q1: {best['q1']} q2: {best['q2']}")
    print(f"Chosen alpha: {best['alpha']} rho: {best['rho']}")
    print(f"Chosen b_cyt: {best['b_cyt']} r_cyt: {best['r_cyt']}")
    print(f"Cyt PC1 explained variance: {best['pc1_explained_variance']}")
    print(f"Infection effect (MetAge gap): {best['infection_effect_met_gap']}")
    print(f"Subjects: {best['n_subjects']} Visits: {best['n_visits']}")
    print(f"Mean MetAge gap: {best['mean_gap']} Std MetAge gap: {best['std_gap']}")


if __name__ == "__main__":
    main()
