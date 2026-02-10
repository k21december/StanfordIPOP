import numpy as np
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")
OUT = CLEAN / "observations.parquet"

NON_LAB_COLS = {
    "SubjectID", "VisitID", "HashedID",
    "CollectionDate", "Event", "SubStudy",
    "Age", "Sex", "Race", "RaceLabel",
    "has_age", "has_sex", "has_race",
    "t_days", "t_years", "dt_years", "ChronAge",
}

def coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s.astype(float)
    out = s.astype(str).str.strip()
    out = out.replace({"NA": np.nan, "nan": np.nan, "": np.nan})
    out = out.str.replace(r"^[<>]=?", "", regex=True)
    return pd.to_numeric(out, errors="coerce")

def main():
    panel = pd.read_parquet(CLEAN / "panel.parquet")

    candidates = [c for c in panel.columns if c not in NON_LAB_COLS]

    miss = {}
    numeric = {}
    for c in candidates:
        s = coerce_numeric_series(panel[c])
        numeric[c] = s
        miss[c] = s.isna().mean()

    miss_s = pd.Series(miss).sort_values()

    top_n = 12
    chosen = miss_s.head(top_n).index.tolist()

    obs = panel[[
        "HashedID","VisitID","CollectionDate","t_years","dt_years",
        "Age","ChronAge","Sex","RaceLabel","Event","SubStudy"
    ]].copy()

    for c in chosen:
        obs[c] = numeric[c]

    z_cols = []
    for c in chosen:
        mu = obs[c].mean(skipna=True)
        sd = obs[c].std(skipna=True)
        if sd == 0 or np.isnan(sd):
            continue
        zname = f"z_{c}"
        obs[zname] = (obs[c] - mu) / sd
        z_cols.append(zname)

    rep = pd.DataFrame({"lab": miss_s.index, "missing_frac": miss_s.values})
    rep.to_csv(CLEAN / "lab_missingness.csv", index=False)

    meta = {
        "chosen_labs": chosen,
        "chosen_z": z_cols,
        "n_rows": len(obs),
        "n_subjects": obs["HashedID"].nunique()
    }
    pd.Series(meta, dtype="object").to_json(CLEAN / "observations_meta.json", indent=2)

    obs.to_parquet(OUT, index=False)

    print("✅ Observations built")
    print("Saved:", OUT)
    print("Chosen labs (lowest missingness):")
    for c in chosen:
        print(f"  {c:>10}  missing={miss[c]:.3f}")
    print("\nZ-scored columns:", len(z_cols))

if __name__ == "__main__":
    main()
