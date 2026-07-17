import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "http://hmp2-data.stanford.edu/"
TABLE_URL = f"{BASE_URL}script.php?table=proteome"

OUT_ROOT = Path("data/raw/proteome")
LINKS_CSV = OUT_ROOT / "proteome_links.csv"
FILES_DIR = OUT_ROOT / "processed_files"

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

    r = requests.get(TABLE_URL, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("No table found on proteome page (layout/access issue).")

    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue

        sample_id = tds[0].get_text(strip=True)   # e.g. ZJBOZ2X-E11-Prot
        visit_id  = tds[1].get_text(strip=True)   # e.g. ZJBOZ2X-E11
        substudy  = tds[2].get_text(strip=True)

        # columns: URL_RAW, URL_Processed
        a_raw = tds[3].find("a")
        a_prc = tds[4].find("a")

        href_raw = a_raw["href"].strip() if a_raw and a_raw.has_attr("href") else ""
        href_prc = a_prc["href"].strip() if a_prc and a_prc.has_attr("href") else ""

        url_raw = urljoin(BASE_URL, href_raw) if href_raw else ""
        url_prc = urljoin(BASE_URL, href_prc) if href_prc else ""

        rows.append({
            "SampleID": sample_id,
            "VisitID": visit_id,
            "SubStudy": substudy,
            "URL_RAW": url_raw,
            "URL_Processed": url_prc,
        })

    df = pd.DataFrame(rows)
    df.to_csv(LINKS_CSV, index=False)
    print(f"Saved {len(df)} rows -> {LINKS_CSV}")

    # Download processed only
    ok = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["URL_Processed"]
        if not isinstance(url, str) or not url:
            continue

        tail = os.path.basename(urlparse(url).path) or "prot.csv"
        out_path = FILES_DIR / f'{row["VisitID"]}__{tail}'

        if out_path.exists() and out_path.stat().st_size > 0:
            ok += 1
            continue

        try:
            with requests.get(url, stream=True, timeout=180) as rr:
                rr.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in rr.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            ok += 1
        except Exception as e:
            print(f"[WARN] failed {url} ({e})")

    print(f"Processed downloaded/exists: {ok}")

if __name__ == "__main__":
    main()
