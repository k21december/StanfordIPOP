import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

def req_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[{name}] missing required columns: {missing}")

def main():
    clinical_p = RAW / "clinical_tests.csv"
    visits_p   = RAW / "visits.csv"
    class_p    = RAW / "classifications.csv"

    for p in (clinical_p, visits_p, class_p):
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    clinical = pd.read_csv(clinical_p)
    visits   = pd.read_csv(visits_p)
    classes  = pd.read_csv(class_p)

    req_cols(clinical, ["SubjectID", "VisitID"], "clinical_tests")
    req_cols(visits, ["SubjectID", "VisitID", "CollectionDate"], "visits")
    req_cols(classes, ["SubjectID", "Age", "Sex", "Race"], "classifications")

    print("=== BASIC SHAPES ===")
    print("clinical:", clinical.shape)
    print("visits:  ", visits.shape)
    print("classes: ", classes.shape)

    print("\n=== DUPLICATES ===")
    print("clinical duplicated VisitID:", clinical["VisitID"].duplicated().sum())
    print("visits duplicated VisitID:  ", visits["VisitID"].duplicated().sum())
    print("classes duplicated SubjectID:", classes["SubjectID"].duplicated().sum())

    print("\n=== MERGE COVERAGE ===")
    v_ids = set(visits["VisitID"].astype(str))
    c_ids = set(clinical["VisitID"].astype(str))
    print("VisitID overlap:", len(v_ids & c_ids))
    print("Visits-only VisitIDs:", len(v_ids - c_ids))
    print("Clinical-only VisitIDs:", len(c_ids - v_ids))

    sub_v = set(visits["SubjectID"].astype(str))
    sub_c = set(clinical["SubjectID"].astype(str))
    sub_k = set(classes["SubjectID"].astype(str))
    print("Subjects overlap (clinical ∩ visits):", len(sub_v & sub_c))
    print("Subjects missing demographics:", len((sub_v | sub_c) - sub_k))

    print("\n=== DEMOGRAPHICS ===")
    print("Sex counts:\n", classes["Sex"].value_counts(dropna=False))
    print("Race counts:\n", classes["Race"].value_counts(dropna=False))

    print("\nOK ✅ Inputs look structurally usable.")

if __name__ == "__main__":
    main()
