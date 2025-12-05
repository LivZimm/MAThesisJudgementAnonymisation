r"""
VG Zürich scraper for query: "Baurekursgericht" with robust resume by clicking pagination.
Usage:
  python .\scraper_AC_ZH.py --from-page 74
  python .\scraper_AC_ZH.py --from-page 74 --from-link 3 --headless
"""

import argparse
import os
import re
import json
import time
import random
import socket
import urllib.request
from pathlib import Path
from typing import List, Tuple, Set, Optional

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

SEARCH_URL = (
    "https://vgrzh.djiktzh.ch/cgi-bin/nph-omniscgi.exe"
    "?OmnisPlatform=WINDOWS"
    "&WebServerUrl=https://vgrzh.djiktzh.ch"
    "&WebServerScript=/cgi-bin/nph-omniscgi.exe"
    "&OmnisLibrary=JURISWEB"
    "&OmnisClass=rtFindinfoWebHtmlService"
    "&OmnisServer=JURISWEB,127.0.0.1:7000"
    "&Aufruf=loadTemplate"
    "&cTemplate=standard/search.fiw"
    "&Schema=ZH_VG_WEB"
    "&cSprache=GER"
    "&Parametername=WWW"
)
QUERY = "Baurekursgericht"
MAX_RESULTS_EXPECTED = 1396

BASE_DIR = Path(r"C:\Users\Livia\Dropbox\hsg_dropbox\hsg_notizen_master_sem3\hsg_sem3_MA\97_Python\Python_Alle Urteile\Verwaltungsgericht_scraper")
OUT_DIR = BASE_DIR / "out"
DEBUG_DIR = BASE_DIR / "debug"
MANIFEST_PATH = BASE_DIR / "manifest.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Locators
LOCATORS_SEARCH_INPUT = [
    (By.CSS_SELECTOR, "input[type='text'][name*='such' i]"),
    (By.CSS_SELECTOR, "input[type='text'][id*='such' i]"),
    (By.XPATH, "//input[@type='text' and (contains(translate(@name,'SUCH','such'),'such') or contains(translate(@id,'SUCH','such'),'such'))]"),
    (By.XPATH, "//form//input[@type='text']"),
]
LOCATORS_SEARCH_SUBMIT = [
    (By.XPATH, "//input[@type='submit' and contains(translate(@value,'SUCHE','suche'),'suche')]"),
    (By.XPATH, "//button[contains(translate(.,'SUCHE','suche'),'suche')]"),
    (By.XPATH, "//button[contains(., 'Suchen') or contains(., 'SUCHEN')]"),
]
LOCATORS_RESULT_LINKS = [
    (By.CSS_SELECTOR, "table a[href*='nph-omniscgi.exe']"),
    (By.CSS_SELECTOR, "div a[href*='nph-omniscgi.exe']"),
    (By.XPATH, "//a[contains(@href,'nph-omniscgi.exe')]"),
]
LOCATORS_NEXT = [
    (By.XPATH, "//a[normalize-space(text())='>']"),
    (By.XPATH, "//a[@title='Weiter' or contains(@title,'Weiter')]"),
    (By.XPATH, "//a[contains(@href,'nSeite=') and (normalize-space(text())='>' or contains(., '>'))]"),
]
LOCATORS_DETAIL_READY = [
    (By.XPATH, "//h1|//h2"),
    (By.XPATH, "//*[contains(@class,'content') or contains(@class,'detail')]"),
]

# Filename hints
CASE_NO_PATTERNS = [
    re.compile(r"\bVB\.?\s?\d{1,4}/\d{2,4}\b", re.I),
    re.compile(r"\b[A-ZÄÖÜ]{1,4}\.\d{4}\.\d{1,4}\b"),
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),
]

# ---------- utils ----------

def init_browser(headless: bool) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.page_load_strategy = "eager"
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--lang=de-DE,de")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)  # Selenium Manager
    driver.set_page_load_timeout(60)
    return driver

def wait_any(driver, locators: List[Tuple[By, str]], timeout: int):
    last = None
    for by, sel in locators:
        try:
            return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, sel)))
        except Exception as e:
            last = e
    raise last or TimeoutException("Element not found")

def get_candidate_result_links(driver) -> List[Tuple[str, str]]:
    anchors = []
    for by, sel in LOCATORS_RESULT_LINKS:
        anchors.extend(driver.find_elements(by, sel))
    out, seen = [], set()
    for a in anchors:
        href = a.get_attribute("href") or ""
        txt = (a.text or "").strip()
        if not href or href in seen:
            continue
        if any(s in href for s in ["resultpage.fiw", "Aufruf=search", "loadTemplate", "nSeite=", "javascript:", "#"]):
            continue
        seen.add(href)
        out.append((href, txt))
    return out

def save_debug(driver, tag: str):
    ts = int(time.time())
    html_path = DEBUG_DIR / f"{ts}_{tag}.html"
    png_path = DEBUG_DIR / f"{ts}_{tag}.png"
    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
    except Exception:
        pass
    try:
        driver.save_screenshot(str(png_path))
    except Exception:
        pass
    print(f"[DEBUG] Saved {html_path.name} and {png_path.name}")

def clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [re.sub(r"\s+"," ", ln).strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def derive_filename(page_text: str, idx: int) -> str:
    first = re.search(r"^(.*)$", page_text, flags=re.M)
    title = first.group(1) if first else ""
    case_no = ""
    for pat in CASE_NO_PATTERNS:
        m = pat.search(page_text)
        if m:
            case_no = m.group(0).strip(". ")
            break
    parts = [p for p in [case_no, (title[:80] if title else None)] if p]
    base = " - ".join(parts) if parts else f"judgment_{idx:05d}"
    base = re.sub(r"[\\/:*?\"<>|]+","_", base).strip()
    return (base if base else f"judgment_{idx:05d}") + ".txt"

def load_manifest() -> Set[str]:
    if MANIFEST_PATH.exists():
        try:
            return set(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("saved_urls", []))
        except Exception:
            return set()
    return set()

def save_manifest(saved: Set[str]) -> None:
    MANIFEST_PATH.write_text(json.dumps({"saved_urls": sorted(saved)}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- connectivity ----------

def check_internet() -> bool:
    try:
        socket.gethostbyname("vgrzh.djiktzh.ch")
        req = urllib.request.Request("https://www.google.com/generate_204", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False

def wait_for_internet():
    print("[!] Internet disconnected. Waiting…")
    while not check_internet():
        time.sleep(5)
    print("[*] Internet is back.")

def get_with_retry(driver, url: str, max_tries=8, base_delay=2.0):
    for attempt in range(1, max_tries + 1):
        try:
            driver.get(url)
            return True
        except WebDriverException as e:
            msg = str(e)
            if "ERR_INTERNET_DISCONNECTED" in msg or "ERR_NAME_NOT_RESOLVED" in msg:
                wait_for_internet()
            else:
                print(f"[!] Nav error (try {attempt}/{max_tries}): {e}")
                time.sleep(base_delay * attempt)
    return False

# ---------- flow ----------

def submit_search(driver, query: str) -> None:
    print("[*] Opening search page…")
    if not get_with_retry(driver, SEARCH_URL):
        raise RuntimeError("Failed to open search page.")
    inp = wait_any(driver, LOCATORS_SEARCH_INPUT, timeout=30)
    print("[*] Submitting query…")
    inp.clear()
    inp.send_keys(query)
    try:
        btn = wait_any(driver, LOCATORS_SEARCH_SUBMIT, timeout=5)
        btn.click()
    except Exception:
        inp.send_keys(Keys.ENTER)
    try:
        WebDriverWait(driver, 40).until(
            lambda d: len(get_candidate_result_links(d)) > 0 or any(d.find_elements(*loc) for loc in LOCATORS_NEXT)
        )
    except Exception:
        print("[!] No results visible after submit.")
        save_debug(driver, "no_results_after_submit")
    print("[*] Results page loaded.")

def click_next(driver) -> bool:
    for by, sel in LOCATORS_NEXT:
        for a in driver.find_elements(by, sel):
            try:
                href = a.get_attribute("href") or ""
                if "nSeite=" in href or a.text.strip() == ">" or (a.get_attribute("title") or "").lower().startswith("weiter"):
                    a.click()
                    WebDriverWait(driver, 20).until(EC.staleness_of(a))
                    WebDriverWait(driver, 20).until(lambda d: True)
                    return True
            except Exception:
                continue
    return False

def jump_by_clicking(driver, target_page: int):
    print(f"[*] Jumping to page {target_page} by clicking '>' {target_page-1} times…")
    current = 1
    failures = 0
    while current < target_page:
        if click_next(driver):
            current += 1
            failures = 0
            # ensure results present
            WebDriverWait(driver, 30).until(lambda d: len(get_candidate_result_links(d)) >= 0)
            time.sleep(0.4 + random.random() * 0.3)
        else:
            failures += 1
            print(f"[!] Could not click next (attempt {failures}). Re-running search to restore state…")
            submit_search(driver, QUERY)
            current = 1
            if failures >= 5:
                raise RuntimeError("Pagination failed repeatedly.")

def process_detail_same_tab(driver, href: str, idx: int, saved_urls: Set[str]) -> bool:
    if href in saved_urls:
        return True
    if not get_with_retry(driver, href):
        print("[x] Could not open detail (after retries).")
        save_debug(driver, f"nav_error_{idx}")
        return False
    try:
        try:
            wait_any(driver, LOCATORS_DETAIL_READY, timeout=15)
        except Exception:
            pass
        html = driver.page_source
        text = clean_text_from_html(html)
        if len(text) < 200:
            time.sleep(1.0)
            text = clean_text_from_html(driver.page_source)
        if len(text) < 80:
            print("[!] Very short text.")
            save_debug(driver, f"short_detail_{idx}")
        filename = derive_filename(text, idx)
        out_path = OUT_DIR / filename
        stem, k = out_path.stem, 1
        while out_path.exists():
            out_path = OUT_DIR / f"{stem} ({k}).txt"
            k += 1
        out_path.write_text(text, encoding="utf-8")
        saved_urls.add(href)
        save_manifest(saved_urls)
        print(f"[+] Saved: {out_path.name}")
        return True
    except Exception as e:
        print(f"[x] Error saving detail: {e}")
        save_debug(driver, f"error_detail_{idx}")
        return False

def go_back_to_results(driver) -> None:
    try:
        driver.back()
        WebDriverWait(driver, 20).until(lambda d: True)
    except Exception:
        print("[!] back() failed; restoring results via fresh search…")
        submit_search(driver, QUERY)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--max-pages", dest="max_pages", type=int, default=None, help="Stop after N result pages.")
    parser.add_argument("--delay", type=float, default=1.0, help="Base delay between detail pages.")
    parser.add_argument("--from-page", dest="from_page", type=int, default=None, help="Start at results page N (1-based).")
    parser.add_argument("--from-link", dest="from_link", type=int, default=0, help="Start at link M on that page (1-based).")
    args = parser.parse_args()

    saved_urls = load_manifest()

    print(f"[*] Headless={args.headless} | Output={OUT_DIR}")
    if not check_internet():
        wait_for_internet()

    driver = init_browser(headless=args.headless)

    total_saved = 0
    total_skipped = 0

    try:
        submit_search(driver, QUERY)

        # Manual resume by clicking pagination
        if args.from_page and args.from_page > 1:
            jump_by_clicking(driver, args.from_page)

        page_idx = args.from_page or 1

        while True:
            print(f"[*] On results page {page_idx}… collecting links.")
            links = get_candidate_result_links(driver)
            if not links:
                print("[!] No links on this page.")
                save_debug(driver, f"no_links_p{page_idx}")
                break

            start_i = max(0, args.from_link) if page_idx == (args.from_page or 1) else 0

            for i, (href, label) in enumerate(links, start=1):
                if i <= start_i:
                    continue
                if href in saved_urls:
                    total_skipped += 1
                    print(f"[-] Skip (already saved): {label or href}")
                    continue
                print(f"[*] Open detail {i}/{len(links)}: {label or href}")
                ok = process_detail_same_tab(driver, href, i + total_saved, saved_urls)
                go_back_to_results(driver)
                if ok:
                    total_saved += 1
                time.sleep(args.delay + random.random() * 0.5)

            if args.max_pages and (page_idx - (args.from_page or 1) + 1) >= args.max_pages:
                print(f"[*] Reached --max-pages={args.max_pages}. Stopping.")
                break

            print("[*] Going to next page…")
            moved = click_next(driver)
            if not moved:
                print("[*] No next arrow found. Done.")
                break

            page_idx += 1
            time.sleep(0.8 + random.random() * 0.6)

            if total_saved > MAX_RESULTS_EXPECTED + 200:
                print("[!] Safety stop: unexpectedly high number of results.")
                break

        print(f"[DONE] Pages processed up to: {page_idx} | Saved: {total_saved} | Skipped: {total_skipped}")
        print(f"Files in: {OUT_DIR}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()

