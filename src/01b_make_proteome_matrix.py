from pathlib import Path
import pandas as pd
import numpy as np

FILES_DIR = Path("data/raw/proteome/processed_files")
OUT_WIDE = Path("data/clean/proteome_protein_matrix.csv")
OUT_QC = Path("data/clean/proteome_qc_summary.csv")

M_SCORE_THRESH = 0.01
TOP_K = 3

def quantify_one(file_path: Path):
    df = pd.read_csv(file_path)

    required = {"ProteinName", "Intensity", "decoy", "m_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} missing columns: {missing}")

    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["m_score"] = pd.to_numeric(df["m_score"], errors="coerce")
    df["decoy"] = pd.to_numeric(df["decoy"], errors="coerce")
    df = df.dropna(subset=["ProteinName", "Intensity", "m_score", "decoy"])

    # Filter confident, non-decoy
    pre_n = len(df)
    df = df[(df["decoy"] == 0) & (df["m_score"] <= M_SCORE_THRESH)].copy()
    post_n = len(df)

    # Optional: favor best peak group if present
    if "peak_group_rank" in df.columns:
        df["peak_group_rank"] = pd.to_numeric(df["peak_group_rank"], errors="coerce")
        # keep top few ranks only (rank 1 is usually best)
        df = df[df["peak_group_rank"].fillna(999) <= 3]

    if df.empty:
        prot = pd.Series(dtype=float)
    else:
        # top-k peptide/transition intensities per protein
        df = df.sort_values("Intensity", ascending=False)
        df["rank_in_prot"] = df.groupby("ProteinName").cumcount() + 1
        df = df[df["rank_in_prot"] <= TOP_K]
        prot = df.groupby("ProteinName")["Intensity"].sum()
        prot = np.log1p(prot)

    qc = {
        "file": file_path.name,
        "rows_total": pre_n,
        "rows_pass": post_n,
        "proteins_nonzero": int((prot > 0).sum()) if len(prot) else 0,
    }
    return prot, qc

def main():
    files = sorted(FILES_DIR.glob("*__*Prot*.csv"))
    if not files:
        print("No proteome processed CSVs found yet.")
        return

    rows = []
    visit_ids = []
    qcs = []

    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name}")
        visit_id = fp.name.split("__", 1)[0]
        prot, qc = quantify_one(fp)
        rows.append(prot)
        visit_ids.append(visit_id)
        qcs.append(qc)

    wide = pd.DataFrame(rows)
    wide.insert(0, "VisitID", visit_ids)
    wide.to_csv(OUT_WIDE, index=False)

    qc_df = pd.DataFrame(qcs)
    qc_df.insert(0, "VisitID", visit_ids)
    qc_df.to_csv(OUT_QC, index=False)

    print("Wrote:", OUT_WIDE, "| shape:", wide.shape)
    print("Wrote:", OUT_QC, "| shape:", qc_df.shape)
    print("Median proteins_nonzero:", float(qc_df["proteins_nonzero"].median()))

if __name__ == "__main__":
    main()
