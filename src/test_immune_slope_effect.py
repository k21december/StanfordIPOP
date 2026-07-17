import pandas as pd
import numpy as np

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise SystemExit("statsmodels not installed. Run: pip install statsmodels")

INFILE = "data/clean/bioage_estimates_2state.parquet"
OUTDIR = "results/immune_slope_test"

df = pd.read_parquet(INFILE).copy()

# keep only what we need
df = df[["HashedID", "t_years", "MetAge", "ImmuneState"]].copy()
df = df.sort_values(["HashedID", "t_years"]).reset_index(drop=True)

# per-subject differences and lags
g = df.groupby("HashedID", sort=False)

df["MetAge_prev"] = g["MetAge"].shift(1)
df["Immune_lag1"] = g["ImmuneState"].shift(1)
df["Immune_lag2"] = g["ImmuneState"].shift(2)
df["t_prev"] = g["t_years"].shift(1)

df["dMetAge"] = df["MetAge"] - df["MetAge_prev"]
df["dt"] = df["t_years"] - df["t_prev"]
df["MetAge_slope"] = df["dMetAge"] / df["dt"]

# valid rows for slope analysis
ana = df.dropna(subset=["Immune_lag1", "dMetAge", "dt", "MetAge_slope"]).copy()
ana = ana[ana["dt"] > 0].copy()

# save row table
ana.to_csv(f"{OUTDIR}/analysis_rows.csv", index=False)

# model 1: future change in MetAge from lagged immune
m1 = smf.ols("dMetAge ~ Immune_lag1", data=ana).fit()

# model 2: future slope in MetAge from lagged immune
m2 = smf.ols("MetAge_slope ~ Immune_lag1", data=ana).fit()

# model 3: add second lag if available
ana2 = ana.dropna(subset=["Immune_lag2"]).copy()
m3 = None
if len(ana2) > 10:
    m3 = smf.ols("MetAge_slope ~ Immune_lag1 + Immune_lag2", data=ana2).fit()

# model 4: subject fixed effects
m4 = smf.ols("MetAge_slope ~ Immune_lag1 + C(HashedID)", data=ana).fit()

with open(f"{OUTDIR}/model_summaries.txt", "w") as f:
    f.write("MODEL 1: dMetAge ~ Immune_lag1\n")
    f.write(m1.summary().as_text())
    f.write("\n\n")

    f.write("MODEL 2: MetAge_slope ~ Immune_lag1\n")
    f.write(m2.summary().as_text())
    f.write("\n\n")

    if m3 is not None:
        f.write("MODEL 3: MetAge_slope ~ Immune_lag1 + Immune_lag2\n")
        f.write(m3.summary().as_text())
        f.write("\n\n")

    f.write("MODEL 4: MetAge_slope ~ Immune_lag1 + C(HashedID)\n")
    f.write(m4.summary().as_text())
    f.write("\n")

coef_rows = []

for model_name, model in [
    ("m1_dMetAge_lag1", m1),
    ("m2_slope_lag1", m2),
    ("m3_slope_lag1_lag2", m3),
    ("m4_slope_lag1_FE", m4),
]:
    if model is None:
        continue
    for term in model.params.index:
        coef_rows.append({
            "model": model_name,
            "term": term,
            "coef": model.params[term],
            "stderr": model.bse[term],
            "pvalue": model.pvalues[term],
            "rsquared": model.rsquared,
            "nobs": model.nobs,
        })

pd.DataFrame(coef_rows).to_csv(f"{OUTDIR}/model_coefficients.csv", index=False)

print("done")
print("analysis rows:", len(ana))
print("saved:", f"{OUTDIR}/analysis_rows.csv")
print("saved:", f"{OUTDIR}/model_summaries.txt")
print("saved:", f"{OUTDIR}/model_coefficients.csv")
