import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TABLES = [
    "targeted_assays",
    "metabolome",
    "lipidome",
]

BASE_URL = "http://hmp2-data.stanford.edu/"

def scrape_table(table_name):
    index_url = f"{BASE_URL}script.php?table={table_name}"
    r = requests.get(index_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")

    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        sample_id = tds[0].get_text(strip=True)
        substudy  = tds[1].get_text(strip=True)
        assay     = tds[2].get_text(strip=True)

        a = tds[3].find("a")
        href = a["href"].strip() if a and a.has_attr("href") else ""
        url = urljoin(BASE_URL, href)

        rows.append({
            "SampleID": sample_id,
            "SubStudy": substudy,
            "Assay": assay,
            "URL": url
        })

    return pd.DataFrame(rows)

def download_files(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["URL"]
        sample_id = row["SampleID"]

        filename = os.path.basename(urlparse(url).path)
        out_path = out_dir / f"{sample_id}__{filename}"

        if out_path.exists():
            continue

        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print("FAILED:", url)

def main():
    for table in TABLES:
        print(f"\n=== Processing {table} ===")

        df = scrape_table(table)

        out_root = Path(f"data/raw/{table}")
        out_root.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_root / f"{table}_links.csv", index=False)

        download_files(df, out_root / "files")

        print(f"{table}: {len(df)} files processed")

if __name__ == "__main__":
    main()
