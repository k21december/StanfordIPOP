import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

INDEX_URL = "http://hmp2-data.stanford.edu/script.php?table=targeted_assays"
BASE_URL  = "http://hmp2-data.stanford.edu/"

OUT_DIR = Path("data/raw/targeted_assays")
LINKS_CSV = OUT_DIR / "targeted_assays_links.csv"
FILES_DIR = OUT_DIR / "files"

def scrape_links(index_url: str) -> pd.DataFrame:
    r = requests.get(index_url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    table = soup.find("table")
    if table is None:
        raise RuntimeError("No <table> found on the targeted_assays page. Site layout may have changed.")

    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        sample_id = tds[0].get_text(strip=True)
        substudy  = tds[1].get_text(strip=True)
        assay     = tds[2].get_text(strip=True)

        a = tds[3].find("a")
        href = a["href"].strip() if a and a.has_attr("href") else ""
        url = urljoin(BASE_URL, href) if href else ""

        rows.append({"SampleID": sample_id, "SubStudy": substudy, "Assay": assay, "URL": url})

    df = pd.DataFrame(rows)
    df = df[df["URL"].astype(bool)].reset_index(drop=True)
    return df

def safe_filename(sample_id: str, url: str) -> str:
    # Prefer a stable name: SampleID + original filename (if present)
    path = urlparse(url).path
    tail = os.path.basename(path) or "file.tsv"
    if not tail.lower().endswith((".tsv", ".txt")):
        tail = tail + ".tsv"
    return f"{sample_id}__{tail}"

def download_one(url: str, out_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"[WARN] Failed: {url} -> {out_path.name} ({e})")
        return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

    print("Scraping targeted_assays index…")
    df = scrape_links(INDEX_URL)
    df.to_csv(LINKS_CSV, index=False)
    print(f"Saved {len(df)} links -> {LINKS_CSV}")

    print("Downloading TSVs…")
    ok = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["URL"]
        sample_id = row["SampleID"]
        fname = safe_filename(sample_id, url)
        out_path = FILES_DIR / fname

        # Skip already-downloaded files (saves time)
        if out_path.exists() and out_path.stat().st_size > 0:
            ok += 1
            continue

        if download_one(url, out_path):
            ok += 1

    print(f"Done. Downloaded/exists: {ok}/{len(df)} files")
    print(f"TSVs are in: {FILES_DIR}")

if __name__ == "__main__":
    main()
