import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

RACE_MAP = {
    "A": "Asian",
    "B": "Black",
    "C": "Caucasian",
    "H": "Hispanic",
}

def coerce_numeric(s):
    if s.dtype.kind in "biufc":
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.replace({"NA": np.nan, "nan": np.nan, "": np.nan})
    s = s.str.replace(r"^[<>]=?", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def main():
    clinical = pd.read_csv(RAW / "clinical_tests.csv")
    visits   = pd.read_csv(RAW / "visits.csv")
    classes  = pd.read_csv(RAW / "classifications.csv")

    clinical["VisitID"] = clinical["VisitID"].astype(str)
    visits["VisitID"]   = visits["VisitID"].astype(str)
    visits["SubjectID"] = visits["SubjectID"].astype(str)
    classes["SubjectID"] = classes["SubjectID"].astype(str)

    clinical["HashedID"] = clinical["VisitID"].str.split("-").str[0]

    visits["CollectionDate"] = coerce_numeric(visits["CollectionDate"])
    visits = visits.dropna(subset=["CollectionDate"])

    panel = clinical.merge(
        visits[["VisitID", "CollectionDate", "Event", "SubStudy"]],
        on="VisitID",
        how="left"
    )

    panel = panel.dropna(subset=["CollectionDate"])

    classes["Age"] = coerce_numeric(classes["Age"])
    classes["Sex"] = classes["Sex"].astype(str).str.strip()
    classes["Race"] = classes["Race"].astype(str).str.strip()
    classes["RaceLabel"] = classes["Race"].map(RACE_MAP).fillna("Unknown")

    panel = panel.merge(
        classes.rename(columns={"SubjectID": "HashedID"}),
        on="HashedID",
        how="left",
        suffixes=("", "_demo")
    )

    panel["has_age"]  = panel["Age"].notna()
    panel["has_sex"]  = panel["Sex"].notna() & (panel["Sex"] != "")
    panel["has_race"] = panel["RaceLabel"] != "Unknown"

    panel = panel.sort_values(["HashedID", "CollectionDate", "VisitID"]).reset_index(drop=True)

    first_date = panel.groupby("HashedID")["CollectionDate"].transform("min")
    panel["t_days"]  = panel["CollectionDate"] - first_date
    panel["t_years"] = panel["t_days"] / 365.25
    panel["dt_years"] = panel.groupby("HashedID")["t_years"].diff().fillna(0.0)

    panel["ChronAge"] = np.where(
        panel["has_age"],
        panel["Age"] + panel["t_years"],
        np.nan
    )

    panel.to_parquet(CLEAN / "panel.parquet", index=False)

    print("\n=== PANEL SUMMARY ===")
    print("Rows:", len(panel))
    print("Subjects (hashed):", panel["HashedID"].nunique())
    print("Avg visits / subject:", len(panel) / panel["HashedID"].nunique())

    print("\nAge available subjects:", panel.loc[panel["has_age"], "HashedID"].nunique())
    print("Sex available subjects:", panel.loc[panel["has_sex"], "HashedID"].nunique())
    print("Race available subjects:", panel.loc[panel["has_race"], "HashedID"].nunique())

    print("\nRace counts:")
    print(panel.loc[panel["has_race"], ["HashedID","RaceLabel"]]
          .drop_duplicates()["RaceLabel"].value_counts())

    print("\n✅ Panel built correctly with demographics.")

if __name__ == "__main__":
    main()
