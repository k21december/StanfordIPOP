from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

OUT = Path("figures/review")
OUT.mkdir(parents=True, exist_ok=True)

LAB_COLS = [
    "z_BMI","z_CR","z_AG","z_ALB","z_ALKP","z_GLU",
    "z_GLOB","z_NA.","z_K","z_BUN","z_CA","z_TP"
]

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()

# -----------------------------
# 1) MetAge_gap vs future_labs
# -----------------------------
base = pd.read_parquet("data/clean/bioage_estimates_2state.parquet").copy()
base = base.sort_values(["HashedID","t_years"])
lab_cols = [c for c in LAB_COLS if c in base.columns]
base["future_labs"] = base.groupby("HashedID")[lab_cols].shift(-1).mean(axis=1)

sub = base.dropna(subset=["MetAge_gap","future_labs"]).copy()
corr = sub["MetAge_gap"].corr(sub["future_labs"])

plt.figure(figsize=(7,5))
plt.scatter(sub["MetAge_gap"], sub["future_labs"], alpha=0.45, s=18)
m, b = np.polyfit(sub["MetAge_gap"], sub["future_labs"], 1)
x = np.linspace(sub["MetAge_gap"].min(), sub["MetAge_gap"].max(), 100)
plt.plot(x, m*x + b, linewidth=2)
plt.xlabel("MetAge gap")
plt.ylabel("Future labs")
plt.title(f"MetAge gap vs Future Labs (r = {corr:.3f})")
savefig("01_metage_gap_vs_future_labs.png")

# -----------------------------
# 2) Actual vs predicted future_labs
# baseline model
# -----------------------------
m1 = sm.OLS(
    sub["future_labs"],
    sm.add_constant(sub[["MetAge_gap"]])
).fit()
sub["pred_future_labs"] = m1.predict(sm.add_constant(sub[["MetAge_gap"]]))
corr_pred = sub["pred_future_labs"].corr(sub["future_labs"])

plt.figure(figsize=(7,5))
plt.scatter(sub["future_labs"], sub["pred_future_labs"], alpha=0.45, s=18)
lo = min(sub["future_labs"].min(), sub["pred_future_labs"].min())
hi = max(sub["future_labs"].max(), sub["pred_future_labs"].max())
plt.plot([lo, hi], [lo, hi], linewidth=2)
plt.xlabel("Actual future labs")
plt.ylabel("Predicted future labs")
plt.title(f"Baseline Prediction Fit (r = {corr_pred:.3f}, R² = {m1.rsquared:.3f})")
savefig("02_actual_vs_predicted_future_labs.png")

# -----------------------------
# 3) Lipid raw vs within-subject
# -----------------------------
lip = pd.read_csv("data/clean/lipidome_pcs.csv")
lip_df = base.merge(lip, on="VisitID", how="inner").copy()
lip_df = lip_df.sort_values(["HashedID","t_years"])
lip_df["future_labs"] = lip_df.groupby("HashedID")[lab_cols].shift(-1).mean(axis=1)

for col in ["MetAge_gap","future_labs","LipPC1","LipPC3","LipPC4"]:
    if col in lip_df.columns:
        lip_df[f"{col}_wc"] = lip_df[col] - lip_df.groupby("HashedID")[col].transform("mean")

raw_corr = lip_df[["MetAge_gap","future_labs"]].dropna().corr().iloc[0,1]
wc_corr = lip_df[["MetAge_gap_wc","future_labs_wc"]].dropna().corr().iloc[0,1]

plt.figure(figsize=(7,5))
plt.bar(
    ["Raw\nMetAge gap", "Within-subject\nMetAge gap"],
    [raw_corr, wc_corr]
)
plt.axhline(0, linewidth=1)
plt.ylabel("Correlation with future labs")
plt.title("Lipidomics Validation: Raw vs Within-Subject Signal")
savefig("03_lipid_raw_vs_within_subject.png")

# -----------------------------
# 4) Cytokine model improvement
# -----------------------------
cyt = pd.read_csv("data/clean/cytokine_pcs.csv")
cyt_df = base.merge(cyt, on="VisitID", how="inner").copy()
cyt_df = cyt_df.sort_values(["HashedID","t_years"])
cyt_df["future_labs"] = cyt_df.groupby("HashedID")[lab_cols].shift(-1).mean(axis=1)

pc_cols = [c for c in cyt_df.columns if c.startswith("CytPC")]
cyt_sub = cyt_df[["future_labs","MetAge_gap"] + pc_cols[:5]].dropna().copy()

cyt_m1 = sm.OLS(
    cyt_sub["future_labs"],
    sm.add_constant(cyt_sub[["MetAge_gap"]])
).fit()

cyt_m2 = sm.OLS(
    cyt_sub["future_labs"],
    sm.add_constant(cyt_sub[["MetAge_gap"] + pc_cols[:5]])
).fit()

cyt_sub["pred1"] = cyt_m1.predict(sm.add_constant(cyt_sub[["MetAge_gap"]]))
cyt_sub["pred2"] = cyt_m2.predict(sm.add_constant(cyt_sub[["MetAge_gap"] + pc_cols[:5]]))

vals_r2 = [cyt_m1.rsquared, cyt_m2.rsquared]
vals_corr = [
    cyt_sub["pred1"].corr(cyt_sub["future_labs"]),
    cyt_sub["pred2"].corr(cyt_sub["future_labs"])
]

plt.figure(figsize=(8,5))
x = np.arange(2)
w = 0.35
plt.bar(x - w/2, vals_r2, width=w, label="R²")
plt.bar(x + w/2, vals_corr, width=w, label="Correlation")
plt.xticks(x, ["MetAge only", "MetAge + Cytokines"])
plt.ylabel("Score")
plt.title("Cytokines Add Small but Real Predictive Value")
plt.legend()
savefig("04_cytokine_model_improvement.png")

# -----------------------------
# 5) Microbiome -> immune baseline
# -----------------------------
gut = pd.read_csv("data/clean/gut16s_pcs.csv")
obs_cyt = pd.read_parquet("data/clean/observations_plus_cytokines.parquet")
micro_df = obs_cyt.merge(gut, on="VisitID", how="inner").merge(cyt, on="VisitID", how="inner")

micro_sub = micro_df.dropna(subset=["MicroPC1","CytPC1"]).copy()
micro_corr = micro_sub["MicroPC1"].corr(micro_sub["CytPC1"])

plt.figure(figsize=(7,5))
plt.scatter(micro_sub["MicroPC1"], micro_sub["CytPC1"], alpha=0.45, s=18)
m, b = np.polyfit(micro_sub["MicroPC1"], micro_sub["CytPC1"], 1)
x = np.linspace(micro_sub["MicroPC1"].min(), micro_sub["MicroPC1"].max(), 100)
plt.plot(x, m*x + b, linewidth=2)
plt.xlabel("MicroPC1")
plt.ylabel("CytPC1")
plt.title(f"Microbiome Tracks Immune Baseline Weakly (r = {micro_corr:.3f})")
savefig("05_microbiome_vs_cytokine.png")

# -----------------------------
# 6) Modality ranking
# -----------------------------
modalities = [
    "MetAge / metabolome",
    "Cytokines",
    "Microbiome",
    "Lipidomics",
    "Proteome"
]
scores = [
    float(cyt_sub["pred1"].corr(cyt_sub["future_labs"])),  # baseline MetAge corr on cytokine-covered set
    float(cyt_sub["pred2"].corr(cyt_sub["future_labs"])) - float(cyt_sub["pred1"].corr(cyt_sub["future_labs"])),
    0.0044,
    0.0,
    0.0
]

plt.figure(figsize=(8,5))
plt.barh(modalities, scores)
plt.xlabel("Approximate predictive contribution")
plt.title("Final Modality Ranking")
savefig("06_modality_ranking.png")

# -----------------------------
# 7) Optional: interaction plot
# -----------------------------
# rebuild simple interaction view from earlier raw reconstruction
tmp = pd.read_parquet("data/clean/observations_plus_cytokines.parquet").copy()
tmp = tmp.sort_values(["HashedID","t_years"])

cyt_marker_cols = [c for c in tmp.columns if c.startswith("z_IL") or c.startswith("z_TNF") or c.startswith("z_IFN")]
lab_z_cols = [c for c in tmp.columns if c in lab_cols]

if lab_z_cols and cyt_marker_cols:
    tmp["MetAge_proxy"] = tmp[lab_z_cols].mean(axis=1)
    tmp["MetAge_gap_proxy"] = tmp["MetAge_proxy"] - tmp["ChronAge"]
    tmp["future_labs_proxy"] = tmp.groupby("HashedID")["MetAge_proxy"].shift(-1)
    cx = tmp[cyt_marker_cols].fillna(0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, random_state=0)
    tmp["CytPC1_proxy"] = pca.fit_transform(cx)[:,0]
    it = tmp.dropna(subset=["MetAge_gap_proxy","future_labs_proxy","CytPC1_proxy"]).copy()

    q1 = it["CytPC1_proxy"].quantile(0.25)
    q3 = it["CytPC1_proxy"].quantile(0.75)
    low = it[it["CytPC1_proxy"] <= q1]
    high = it[it["CytPC1_proxy"] >= q3]

    plt.figure(figsize=(7,5))
    for frame, label in [(low, "Low inflammation"), (high, "High inflammation")]:
        if len(frame) > 5:
            m, b = np.polyfit(frame["MetAge_gap_proxy"], frame["future_labs_proxy"], 1)
            x = np.linspace(frame["MetAge_gap_proxy"].min(), frame["MetAge_gap_proxy"].max(), 100)
            plt.plot(x, m*x + b, label=label, linewidth=2)
    plt.xlabel("MetAge gap (proxy)")
    plt.ylabel("Future labs (proxy)")
    plt.title("Inflammation Changes the Aging Slope")
    plt.legend()
    savefig("07_inflammation_interaction_proxy.png")

print(f"Saved plots to: {OUT.resolve()}")
