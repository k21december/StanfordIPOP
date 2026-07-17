from pathlib import Path
import pandas as pd

FILES_DIR = Path("data/raw/proteome/processed_files")

def main():
    files = sorted(FILES_DIR.glob("*.csv"))
    if not files:
        print("No proteome processed CSVs found yet.")
        return

    print("Found", len(files), "files. Showing first 5.\n")
    for fp in files[:5]:
        df = pd.read_csv(fp, nrows=50)
        print("==", fp.name, "==")
        print("cols:", list(df.columns))
        print("nrows (sample):", len(df))
        print()

if __name__ == "__main__":
    main()
