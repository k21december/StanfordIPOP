import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

CLEAN = Path("data/clean")
RNG = np.random.default_rng(42)

if len(sys.argv) >= 3:
    INP = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
else:
    INP = CLEAN / "multimodal_channels_v1.parquet"
    OUT = CLEAN / "bioage_estimates_particle.parquet"

if len(sys.argv) >= 4:
    OUT_PARAMS = Path(sys.argv[3])
else:
    OUT_PARAMS = OUT.with_name(OUT.stem + "_params.json")

LABS = ["BMI", "CR", "AG", "ALB", "ALKP", "GLU", "GLOB", "NA.", "K", "BUN", "CA", "TP"]

def build_cytokine_pc1(obs, z_labs):
    if "CytPC1" in obs.columns:
        pc1 = obs["CytPC1"].to_numpy(dtype=float)
        return pc1, ["CytPC1"], np.nan

    z_cols = [c for c in obs.columns if c.startswith("z_")]
    z_cyt = [c for c in z_cols if c not in set(z_labs)]
    if not z_cyt:
        raise SystemExit("No cytokine features found.")

    X = obs[z_cyt].copy()
    X = X.apply(lambda c: c.fillna(c.median()), axis=0)
    Xv = X.to_numpy(dtype=float)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xv).reshape(-1)
    pc1 = (pc1 - np.nanmean(pc1)) / (np.nanstd(pc1) + 1e-8)
    return pc1, z_cyt, float(pca.explained_variance_ratio_[0])

def log_norm_pdf(y, mu, var):
    var = np.maximum(var, 1e-10)
    return -0.5 * (np.log(2.0 * np.pi * var) + ((y - mu) ** 2) / var)

def systematic_resample(weights):
    n = len(weights)
    positions = (RNG.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    idx = np.zeros(n, dtype=int)
    i = 0
    j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx

def particle_filter_subject(
    g,
    n_particles,
    alpha,
    rho,
    lam,
    q1,
    q2,
    b_labs,
    r_labs,
    b_cyt,
    r_cyt,
    beta_glob,
    beta_met,
    gamma_imm,
):
    g = g.sort_values("t_years").reset_index(drop=True)

    dt = g["dt_years"].to_numpy(dtype=float)
    chron = g["ChronAge"].to_numpy(dtype=float)
    zL = g["zmean_labs"].to_numpy(dtype=float)
    zC = g["zcyt_pc1"].to_numpy(dtype=float)
    zG = g["z_GLOB"].to_numpy(dtype=float)
    zM = g["z_met_pca_3"].to_numpy(dtype=float)

    n = len(g)
    if n < 2 or not np.isfinite(chron[0]):
        return None

    x_particles = np.zeros((n_particles, 2), dtype=float)
    x_particles[:, 0] = chron[0] + RNG.normal(0.0, 0.25, size=n_particles)
    x_particles[:, 1] = RNG.normal(0.0, 0.25, size=n_particles)
    w = np.full(n_particles, 1.0 / n_particles, dtype=float)

    xs = np.zeros((n, 2), dtype=float)
    xvars = np.zeros((n, 2), dtype=float)
    nll = 0.0

    for i in range(n):
        if i > 0:
            dt_i = max(float(dt[i]), 0.0) if np.isfinite(dt[i]) else 0.0
            glob_i = zG[i] if np.isfinite(zG[i]) else 0.0
            met_i = zM[i] if np.isfinite(zM[i]) else 0.0

            driver_term = beta_glob * glob_i + beta_met * met_i

            x_particles[:, 0] = (
                x_particles[:, 0]
                + alpha * dt_i
                + lam * driver_term * dt_i
                + gamma_imm * x_particles[:, 1] * dt_i
                + RNG.normal(0.0, np.sqrt(max(q1 * max(dt_i, 1e-6), 1e-10)), size=n_particles)
            )

            x_particles[:, 1] = (
                rho * x_particles[:, 1]
                + RNG.normal(0.0, np.sqrt(max(q2, 1e-10)), size=n_particles)
            )

        logw = np.log(w + 1e-300)

        if np.isfinite(zL[i]) and np.isfinite(chron[i]):
            mu_labs = b_labs * (x_particles[:, 0] - chron[i])
            logw += log_norm_pdf(zL[i], mu_labs, r_labs)

        if np.isfinite(zC[i]):
            mu_cyt = b_cyt * x_particles[:, 1]
            logw += log_norm_pdf(zC[i], mu_cyt, r_cyt)

        m = np.max(logw)
        w_unnorm = np.exp(logw - m)
        sw = np.sum(w_unnorm)
        if not np.isfinite(sw) or sw <= 0:
            return None

        w = w_unnorm / sw
        nll += -(m + np.log(sw) - np.log(n_particles))

        xs[i, 0] = np.sum(w * x_particles[:, 0])
        xs[i, 1] = np.sum(w * x_particles[:, 1])
        xvars[i, 0] = np.sum(w * (x_particles[:, 0] - xs[i, 0]) ** 2)
        xvars[i, 1] = np.sum(w * (x_particles[:, 1] - xs[i, 1]) ** 2)

        ess = 1.0 / np.sum(w ** 2)
        if ess < 0.5 * n_particles:
            idx = systematic_resample(w)
            x_particles = x_particles[idx]
            w = np.full(n_particles, 1.0 / n_particles, dtype=float)

    return xs, xvars, float(nll)

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

    required_cols = ["HashedID", "VisitID", "t_years", "dt_years", "ChronAge", "z_GLOB", "met_pca_3"]
    missing_required = [c for c in required_cols if c not in obs.columns]
    if missing_required:
        raise SystemExit(f"Missing required columns: {missing_required}")

    obs["z_met_pca_3"] = (
        obs["met_pca_3"] - obs["met_pca_3"].mean()
    ) / (obs["met_pca_3"].std() + 1e-8)

    z_labs = [f"z_{c}" for c in LABS if f"z_{c}" in obs.columns]
    if not z_labs:
        raise SystemExit("No lab z_ columns found.")
    obs["zmean_labs"] = np.nanmean(obs[z_labs].to_numpy(dtype=float), axis=1)

    pc1, z_cyt, pc1_var = build_cytokine_pc1(obs, z_labs)
    obs["zcyt_pc1"] = pc1

    groups = [g.copy() for _, g in obs.groupby("HashedID")]

    n_particles = 300
    q1_grid = np.array([0.04, 0.09, 0.16])
    q2_grid = np.array([0.01, 0.04])
    alpha_grid = np.array([0.9, 1.0])
    rho_grid = np.array([0.75, 0.9])
    lam_grid = np.array([0.0, 0.02, 0.05, 0.10])

    b_labs = 0.25
    r_labs = 0.80 ** 2
    b_cyt_grid = np.array([0.75, 1.0])
    r_cyt_grid = np.array([0.40, 0.60]) ** 2

    beta_glob = 0.03
    beta_met = 0.013
    gamma_imm = 0.017

    best = None
    best_out = None

    for q1 in q1_grid:
        for q2 in q2_grid:
            for alpha in alpha_grid:
                for rho in rho_grid:
                    for lam in lam_grid:
                        for b_cyt in b_cyt_grid:
                            for r_cyt in r_cyt_grid:
                                results = []
                                nll_total = 0.0

                                for g in groups:
                                    fit = particle_filter_subject(
                                        g=g,
                                        n_particles=n_particles,
                                        alpha=float(alpha),
                                        rho=float(rho),
                                        lam=float(lam),
                                        q1=float(q1),
                                        q2=float(q2),
                                        b_labs=float(b_labs),
                                        r_labs=float(r_labs),
                                        b_cyt=float(b_cyt),
                                        r_cyt=float(r_cyt),
                                        beta_glob=float(beta_glob),
                                        beta_met=float(beta_met),
                                        gamma_imm=float(gamma_imm),
                                    )
                                    if fit is None:
                                        continue

                                    xs, xvars, nll = fit
                                    tmp = g.sort_values("t_years").reset_index(drop=True).copy()
                                    tmp["MetAge"] = xs[:, 0]
                                    tmp["ImmuneState"] = xs[:, 1]
                                    tmp["MetAge_var"] = xvars[:, 0]
                                    tmp["Immune_var"] = xvars[:, 1]
                                    tmp["MetAge_sd"] = np.sqrt(np.maximum(xvars[:, 0], 1e-12))
                                    tmp["Immune_sd"] = np.sqrt(np.maximum(xvars[:, 1], 1e-12))
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
                                        "n_particles": int(n_particles),
                                        "q1": float(q1),
                                        "q2": float(q2),
                                        "alpha": float(alpha),
                                        "rho": float(rho),
                                        "lam": float(lam),
                                        "b_labs": float(b_labs),
                                        "r_labs": float(r_labs),
                                        "b_cyt": float(b_cyt),
                                        "r_cyt": float(r_cyt),
                                        "beta_glob": float(beta_glob),
                                        "beta_met": float(beta_met),
                                        "gamma_imm": float(gamma_imm),
                                        "mean_gap": float(mean_gap),
                                        "std_gap": float(std_gap),
                                        "infection_effect_met_gap": float(inf_eff),
                                        "pc1_explained_variance": None if np.isnan(pc1_var) else float(pc1_var),
                                        "n_subjects": int(out["HashedID"].nunique()),
                                        "n_visits": int(len(out)),
                                        "input_file": str(INP),
                                        "output_file": str(OUT),
                                        "lab_features": z_labs,
                                        "cytokine_features": z_cyt,
                                    }
                                    best_out = out.copy()

    if best is None or best_out is None:
        raise SystemExit("Particle model fitting failed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARAMS.parent.mkdir(parents=True, exist_ok=True)

    best_out.to_parquet(OUT, index=False)
    with open(OUT_PARAMS, "w") as f:
        json.dump(best, f, indent=2)

    print("✅ particle model fit complete")
    print(f"Saved: {OUT}")
    print(f"Params saved: {OUT_PARAMS}")
    print(f"Particles: {best['n_particles']}")
    print(f"Chosen q1: {best['q1']} q2: {best['q2']}")
    print(f"Chosen alpha: {best['alpha']} rho: {best['rho']} lam: {best['lam']}")
    print(f"Chosen b_cyt: {best['b_cyt']} r_cyt: {best['r_cyt']}")
    print(f"Mean MetAge gap: {best['mean_gap']} Std MetAge gap: {best['std_gap']}")
    print(f"Infection effect (MetAge gap): {best['infection_effect_met_gap']}")

if __name__ == "__main__":
    main()
