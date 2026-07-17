import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/metabolome_abundance.txt", sep="\t")

id_col = "SampleID"
meta_cols = [c for c in ["SubjectID", "CL1", "CL2", "CL3", "CL4"] if c in df.columns]
feature_cols = [c for c in df.columns if c not in [id_col] + meta_cols]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

keep_cols = X.columns[X.isna().mean() < 0.30]
X = X[keep_cols]
X = X.fillna(X.median())

X = X.clip(lower=0)
X_log = np.log10(X + 1.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

pca = PCA(n_components=10)
scores = pca.fit_transform(X_scaled)

out = pd.DataFrame(scores, columns=[f"met_pca_{i+1}" for i in range(10)])
out.insert(0, "SampleID", df[id_col])

for c in meta_cols:
    out[c] = df[c]

out.to_csv("data/clean/metabolome_pca.csv", index=False)

ev = pd.DataFrame({
    "pc": [f"PC{i+1}" for i in range(10)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained": pca.explained_variance_ratio_.cumsum(),
})
ev.to_csv("data/clean/metabolome_pca_variance.csv", index=False)

print("Saved: data/clean/metabolome_pca.csv", out.shape)
print("Saved: data/clean/metabolome_pca_variance.csv")
print("\nTop explained variance:")
print(ev.to_string(index=False))
print("\nMetabolite columns kept:", len(keep_cols))
