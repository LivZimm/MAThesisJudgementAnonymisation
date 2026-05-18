import argparse
import csv
import re
import time
from pathlib import Path
from typing import List, Tuple

from playwright.sync_api import sync_playwright, Page, BrowserContext, TimeoutError as PWTimeoutError

VOLLTEXT_URL = "https://www.baurekursgericht-zh.ch/rechtsprechung/entscheiddatenbank/volltextsuche/"

# Centralized selectors
SEL_BTN_Suchen = "button:has-text('Suchen'), input[type=submit][value='Suchen'], input[type=button][value='Suchen']"
SEL_DOWNLOAD_ANCHORS = "a:has-text('Entscheidauszug')"
SEL_NEXT = "a:has-text('nächste Seite')"
SEL_SEARCH_INPUTS = "input[type=text], textarea"
COOKIE_ACCEPT = [
    "button:has-text('Alle akzeptieren')",
    "button:has-text('Akzeptieren')",
    "button:has-text('Einverstanden')",
    "button:has-text('OK')",
    "text=/Cookies akzeptieren/i",
]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def accept_cookies_if_present(page: Page) -> None:
    for sel in COOKIE_ACCEPT:
        try:
            loc = page.locator(sel)
            if loc.count() and loc.first.is_visible():
                loc.first.click(timeout=1000)
                time.sleep(0.3)
                return
        except Exception:
            continue

def sanitize_filename_from_text(text: str) -> str:
    """
    Transform link text like 'Download: Entscheidauszug aus BRGE IV Nr. 0025/2025'
    into a filesystem-safe 'BRGE_IV_Nr_0025_2025.pdf'.
    """
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"^Download:\s*", "", t, flags=re.IGNORECASE)
    m = re.search(r"\baus\s+(.+)", t, flags=re.IGNORECASE)
    core = m.group(1) if m else t
    core = core.replace("/", "_")
    core = re.sub(r"[^\w\s\.-]", "_", core)
    core = re.sub(r"\s+", "_", core)
    core = re.sub(r"_+", "_", core).strip("_")
    return core + ("" if core.lower().endswith(".pdf") else ".pdf")

def make_absolute(current_url: str, href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"https://www.baurekursgericht-zh.ch{href}"
    base = current_url.rsplit("/", 1)[0]
    return f"{base}/{href}"

def submit_search(page: Page, query: str | None) -> None:
    """
    Click Suchen after optionally filling the search field.
    """
    if query:
        inputs = page.locator(SEL_SEARCH_INPUTS)
        if inputs.count():
            box = inputs.first
            box.fill("")
            box.type(query)
    try:
        page.locator(SEL_BTN_Suchen).first.click(timeout=4000)
    except PWTimeoutError:
        # Fallback: hit Enter in the first input
        inputs = page.locator(SEL_SEARCH_INPUTS)
        if inputs.count():
            inputs.first.press("Enter")

def wait_for_results(page: Page, timeout_ms: int = 20000) -> None:
    # We wait for at least one download anchor
    page.wait_for_selector(SEL_DOWNLOAD_ANCHORS, timeout=timeout_ms)

def collect_pdf_links(page: Page) -> List[Tuple[str, str]]:
    anchors = page.locator(SEL_DOWNLOAD_ANCHORS)
    links: List[Tuple[str, str]] = []
    n = anchors.count()
    for i in range(n):
        a = anchors.nth(i)
        try:
            href = (a.get_attribute("href") or "").strip()
            text = a.inner_text().strip()
            if href.lower().endswith(".pdf"):
                links.append((href, text))
        except Exception:
            continue
    return links

def click_next_if_any(page: Page) -> bool:
    try:
        nxt = page.locator(SEL_NEXT)
        if nxt.count() and nxt.first.is_enabled():
            nxt.first.scroll_into_view_if_needed()
            nxt.first.click()
            page.wait_for_load_state("load")
            return True
    except Exception:
        pass
    return False

def http_download(context: BrowserContext, url: str, out_path: Path, retries: int = 3, backoff: float = 0.8) -> None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = context.request.get(
                url,
                timeout=30000,
                headers={"User-Agent": "Mozilla/5.0 (compatible; BRG-Scraper/1.0)"}
            )
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status}")
            data = resp.body()
            if not data:
                raise RuntimeError("Empty body")
            out_path.write_bytes(data)
            return
        except Exception as e:
            last_err = e
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")

def run(outdir: Path, headful: bool, max_pages: int, start_page: int, rate_sec: float, dry_run: bool, query: str | None) -> None:
    ensure_dir(outdir)
    manifest = outdir / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "filename", "url", "status", "error"])
        writer.writeheader()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        context = browser.new_context(accept_downloads=False)
        page = context.new_page()

        page.goto(VOLLTEXT_URL, wait_until="load")
        accept_cookies_if_present(page)
        submit_search(page, query)
        wait_for_results(page)

        # Fast-forward to start_page if requested
        page_no = 1
        while page_no < start_page and click_next_if_any(page):
            page_no += 1
            wait_for_results(page)

        total_downloaded = total_skipped = total_errors = 0

        while True:
            links = collect_pdf_links(page)
            if not links:
                time.sleep(1.0)
                links = collect_pdf_links(page)

            with manifest.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["page", "filename", "url", "status", "error"])

                for href, text in links:
                    url = make_absolute(page.url, href)
                    fname = sanitize_filename_from_text(text)
                    out_path = outdir / fname

                    if out_path.exists():
                        total_skipped += 1
                        writer.writerow({"page": page_no, "filename": fname, "url": url, "status": "skipped", "error": ""})
                        continue

                    try:
                        if not dry_run:
                            http_download(context, url, out_path)
                        total_downloaded += 1
                        writer.writerow({"page": page_no, "filename": fname, "url": url, "status": "ok", "error": ""})
                    except Exception as e:
                        total_errors += 1
                        writer.writerow({"page": page_no, "filename": fname, "url": url, "status": "error", "error": str(e)})
                    time.sleep(max(0.0, rate_sec))

            if max_pages and (page_no - start_page + 1) >= max_pages:
                break
            if not click_next_if_any(page):
                break
            wait_for_results(page)
            page_no += 1

        browser.close()
        print("\n=== Summary ===")
        print(f"Downloaded: {total_downloaded}")
        print(f"Skipped   : {total_skipped}")
        print(f"Errors    : {total_errors}")
        print(f"Manifest  : {manifest}")

def main():
    ap = argparse.ArgumentParser(description="Download BRG Zürich decision PDFs via browser automation.")
    ap.add_argument("--out", type=Path, default=Path.cwd() / "baurekursgericht_pdfs", help="Output directory.")
    ap.add_argument("--headful", action="store_true", help="Show Chromium window.")
    ap.add_argument("--max-pages", type=int, default=0, help="Stop after N pages (0 = all).")
    ap.add_argument("--start-page", type=int, default=1, help="Start pagination at this 1-based page number.")
    ap.add_argument("--rate", type=float, default=0.25, help="Seconds to wait between downloads.")
    ap.add_argument("--dry-run", action="store_true", help="Do not download, only list and write manifest.")
    ap.add_argument("--query", type=str, default=None, help="Optional search term to fill before clicking Suchen.")
    args = ap.parse_args()

    run(
        outdir=args.out,
        headful=args.headful,
        max_pages=max(0, args.max_pages),
        start_page=max(1, args.start_page),
        rate_sec=max(0.0, args.rate),
        dry_run=args.dry_run,
        query=(args.query or None),
    )

if __name__ == "__main__":
    main()
