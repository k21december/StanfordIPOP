from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

IN_WIDE = Path("data/clean/proteome_protein_matrix.csv")
OUT_PCS = Path("data/clean/proteome_pcs.csv")

N_PCS = 5
MIN_NONMISSING_FRAC = 0.6

def main():
    df = pd.read_csv(IN_WIDE)
    visit = df["VisitID"].astype(str)
    X = df.drop(columns=["VisitID"])

    # Drop proteins with too much missingness
    keep = X.notna().mean(axis=0) >= MIN_NONMISSING_FRAC
    X = X.loc[:, keep]

    # Impute remaining missing with column median (simple + stable)
    X = X.apply(lambda c: c.fillna(c.median()), axis=0)

    # Center/scale
    Xv = X.values
    Xv = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + 1e-8)

    pca = PCA(n_components=min(N_PCS, Xv.shape[0], Xv.shape[1]))
    Z = pca.fit_transform(Xv)

    out = pd.DataFrame(Z, columns=[f"ProtPC{i+1}" for i in range(Z.shape[1])])
    out.insert(0, "VisitID", visit)

    out.to_csv(OUT_PCS, index=False)
    print("Wrote:", OUT_PCS, "| shape:", out.shape)
    print("Explained variance:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()
