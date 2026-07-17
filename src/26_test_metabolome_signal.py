import pandas as pd

# load metabolome PCA
met = pd.read_csv("data/clean/metabolome_pca.csv")

# load your bioage model (use your best one)
bio = pd.read_parquet("data/clean/bioage_estimates_2state_with_composites.parquet")

# --- key step: align IDs ---
# metabolome SampleID looks like: ZOZOW1T-01
# extract VisitID

met["VisitID"] = met["SampleID"].str.split("-").str[1]
met["SubjectID"] = met["SampleID"].str.split("-").str[0]

# merge
df = pd.merge(bio, met, on=["SubjectID", "VisitID"], how="inner")

print("Merged shape:", df.shape)

# compute correlations
results = {}

def safe_corr(a, b):
    return pd.Series(a).corr(pd.Series(b))

targets = [
    ("MetAge", "met_pca_1"),
    ("MetAge_gap", "met_pca_1"),
    ("ImmuneState", "met_pca_1"),
]

# optional if exists
if "FinalBioAge_lam_0p05" in df.columns:
    targets.append(("FinalBioAge_lam_0p05", "met_pca_1"))

for x, y in targets:
    if x in df.columns:
        corr = safe_corr(df[x], df[y])
        results[f"{x} vs {y}"] = corr

print("\nCorrelations:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# save
pd.DataFrame(list(results.items()), columns=["comparison", "correlation"]) \
    .to_csv("results/metabolome_validation.csv", index=False)

print("\nSaved: results/metabolome_validation.csv")
