# file: scraper_FSC.py
"""
Scrape all judgments for query "Zürich" on the Swiss Federal Supreme Court (BGer) site
in one-year increments starting 2000. Handles pagination via "page=N" (Vorwärts).
Outputs:
  - ./bger_out/*.txt  (UTF-8)
  - ./bger_out/metadata.csv (URL, filename, year, page, title)
  - ./bger_out/manifest.json (saved URLs for dedupe)
Usage:
  python scraper_FSC.py
  # optional:
  python scraper_FSC.py --start-year 2000 --end-year 2025 --delay 0.8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse, quote

import requests
from bs4 import BeautifulSoup

BASE = "https://search.bger.ch"
LIST_PATH = "/ext/eurospider/live/de/php/aza/http/index.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
}

OUT_DIR = Path("./bger_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.json"
META_CSV_PATH = OUT_DIR / "metadata.csv"

# --------- helpers ---------

def load_manifest() -> Set[str]:
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            return set(data.get("saved_urls", []))
        except Exception:
            return set()
    return set()

def save_manifest(urls: Set[str]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps({"saved_urls": sorted(urls)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def append_metadata_row(row: dict, header: List[str]) -> None:
    file_exists = META_CSV_PATH.exists()
    with META_CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def year_dates(y: int) -> Tuple[str, str]:
    return f"01.01.{y}", f"31.12.{y}"

def build_list_url(query: str, from_date: str, to_date: str, page: int) -> str:
    # Note: BGer accepts UTF-8; we’ll pass raw "Zürich" (requests will encode).
    params = {
        "lang": "de",
        "type": "simple_query",
        "query_words": query,
        "from_date": from_date,
        "to_date": to_date,
        "subcollection_mI12": "on",
        "sort": "relevance",
        "page": str(page),
        "insertion_date": "",           # keep present to mirror site URLs
        "top_subcollection_aza": "any", # ditto
    }
    return f"{BASE}{LIST_PATH}?{urlencode(params, doseq=True)}"

def get_soup(sess: requests.Session, url: str, tries: int = 5, timeout: int = 30) -> BeautifulSoup:
    last = None
    for k in range(tries):
        try:
            r = sess.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code >= 500:
                time.sleep(1.5 * (k + 1))
                continue
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    raise last or RuntimeError(f"Failed to GET {url}")

def find_result_links(soup: BeautifulSoup) -> List[str]:
    """
    Try to find document detail links.
    Common patterns:
      - type=show_document&highlight_docid=...
      - highlight_docid=... (with or without type)
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "highlight_docid=" in href or "type=show_document" in href:
            # Exclude navigation and javascript
            if "javascript:" in href:
                continue
            full = urljoin(BASE, href)
            links.append(full)
    # Heuristic: also accept links inside result list items with title/strong
    # but above usually suffices.
    # Dedup preserve order
    seen = set()
    uniq = []
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def has_forward_link(soup: BeautifulSoup) -> bool:
    # The site shows "Vorwärts" anchor with ?page=N+1; detect by text or rel.
    a = soup.find("a", string=lambda t: t and "Vorwärts" in t)
    if a and a.get("href", "").strip():
        return True
    # Fallback: any link with "page=" and title mentioning Ränge / Vorwärts
    for x in soup.find_all("a", href=True, title=True):
        if "page=" in x["href"] and ("Vorwärts" in x.get_text() or "Ränge" in x["title"]):
            return True
    return False

CASE_PATTERNS = [
    re.compile(r"\b[A-Z]?\d{1,2}[A-Z]?[_\-]\d{1,4}/\d{4}\b"),  # 1C_123/2020, 5A-100/2019
    re.compile(r"\b\d{1,3}\s?[A-Z]{1,2}\s?\d{1,4}/\d{4}\b"),   # 2 C 123/2020 (rare)
]
DATE_PATTERNS = [
    re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b"),                    # 31.12.2020
]

def extract_text_and_title(soup: BeautifulSoup) -> Tuple[str, str]:
    # Try specific containers first
    main = soup.find(id="content") or soup.find(id="aza-content") or soup.find("main")
    text = ""
    if main:
        text = main.get_text("\n", strip=True)
    if not text:
        text = soup.get_text("\n", strip=True)
    title = ""
    h = soup.find("h1") or soup.find("h2") or soup.find("title")
    if h:
        title = h.get_text(" ", strip=True)
    return text, title

def derive_filename(text: str, title: str, seq: int) -> str:
    case_no = ""
    for pat in CASE_PATTERNS:
        m = pat.search(text) or pat.search(title)
        if m:
            case_no = m.group(0)
            break
    dt = ""
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if m:
            dt = m.group(0)
            break
    parts = [p for p in [dt, case_no, (title[:80] if title else None)] if p]
    base = " - ".join(parts) if parts else f"judgment_{seq:06d}"
    base = re.sub(r'[\\/:*?"<>|]+', "_", base).strip()
    return (base or f"judgment_{seq:06d}") + ".txt"

# --------- main scraping ---------

def scrape_year(sess: requests.Session, year: int, saved: Set[str], delay: float, seq_start: int) -> int:
    from_d, to_d = year_dates(year)
    page = 1
    saved_this_year = 0
    while True:
        list_url = build_list_url("Zürich", from_d, to_d, page)
        soup = get_soup(sess, list_url)
        links = find_result_links(soup)

        # Filter to show_document or highlight_docid detail pages
        doc_links = []
        for u in links:
            # Most detail pages include highlight_docid= and often type=show_document
            if "highlight_docid=" in u:
                doc_links.append(u)
        # If regex too strict and nothing found, fall back to all links we saw
        if not doc_links:
            doc_links = links

        if not doc_links:
            # No results on this page; finish year
            break

        for u in doc_links:
            if u in saved:
                continue
            detail = get_soup(sess, u)
            text, title = extract_text_and_title(detail)
            if len(text) < 80:
                # probably a stub; still save to inspect
                pass
            fname = derive_filename(text, title, seq_start + saved_this_year)
            out_path = OUT_DIR / fname
            out_path.write_text(text, encoding="utf-8")
            saved.add(u)
            save_manifest(saved)
            row = {
                "url": u,
                "filename": fname,
                "year": year,
                "page": page,
                "title": title,
                "from_date": from_d,
                "to_date": to_d,
            }
            append_metadata_row(row, ["url", "filename", "year", "page", "title", "from_date", "to_date"])
            saved_this_year += 1
            time.sleep(delay)

        # Pagination: stop if there is no "Vorwärts"
        if not has_forward_link(soup):
            break
        page += 1
        time.sleep(delay * 0.5)

    return saved_this_year

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=date.today().year)
    ap.add_argument("--delay", type=float, default=0.6, help="Seconds between requests")
    args = ap.parse_args()

    saved = load_manifest()
    seq = 0
    with requests.Session() as sess:
        sess.headers.update(HEADERS)
        total = 0
        for y in range(args.start_year, args.end_year + 1):
            print(f"[*] Year {y}: from {y}-01-01 to {y}-12-31")
            added = scrape_year(sess, y, saved, args.delay, seq)
            seq += added
            total += added
            print(f"    Saved this year: {added} (total {total})")

    print(f"[DONE] Total saved: {len(saved)}")
    print(f"TXT: {OUT_DIR}")
    print(f"CSV: {META_CSV_PATH}")

if __name__ == "__main__":
    main()

