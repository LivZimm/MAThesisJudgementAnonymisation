import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# =========================================================
# PATHS
# =========================================================

EXCEL_PATH = Path(
    "X.xlsx"
)

SOURCE_FOLDERS = [
    Path("X/baurekursgericht_scraper"),
    Path("X/Bundesgericht_scraper"),
    Path("X/Verwaltungsgericht_scraper"),
]

OUTPUT_PATH = EXCEL_PATH.with_name(EXCEL_PATH.stem + "_text_corrected_full_filename_match.xlsx")


# =========================================================
# HELPERS
# =========================================================

def normalize_full_filename(value) -> Optional[str]:
    """
    Normalize a full filename string for exact matching while accounting
    for Unicode differences such as 'ü' vs 'ü'.

    Keeps the FULL basename intact, e.g.
    '06.10.2022 - Verwaltungsgericht des Kantons Zürich_ VB.2022.00051.txt'
    """
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    # Normalize Unicode so visually identical strings compare equal
    s = unicodedata.normalize("NFC", s)

    # In case Excel contains a full path, reduce to basename only
    s = os.path.basename(s)

    # Normalize whitespace around the filename
    s = s.strip()

    return s


def build_full_filename_index(source_folders: List[Path]) -> Dict[str, List[Path]]:
    """
    Build index:
        normalized full basename -> list of matching file paths

    We keep all matches so duplicates are explicit and never silently overwritten.
    """
    index: Dict[str, List[Path]] = {}

    for folder in source_folders:
        if not folder.exists():
            print(f"[WARNING] Folder does not exist: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for file in files:
                normalized_name = normalize_full_filename(file)
                if normalized_name is None:
                    continue

                full_path = Path(root) / file
                index.setdefault(normalized_name, []).append(full_path)

    return index


def read_text_file(file_path: Path) -> str:
    """
    Read text file with encoding fallbacks.
    """
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue

    return file_path.read_text(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    """
    Clean raw text and merge hard line breaks into paragraphs.
    """
    if text is None:
        return text

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")

    lines = [line.strip() for line in text.split("\n")]

    paragraphs = []
    current = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    text = "\n\n".join(paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    print("Loading Excel...")
    df = pd.read_excel(EXCEL_PATH)

    required_columns = {"filename", "text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Indexing source files by FULL filename...")
    file_index = build_full_filename_index(SOURCE_FOLDERS)
    print(f"Indexed {len(file_index)} unique full filenames.")

    duplicate_keys = {k: v for k, v in file_index.items() if len(v) > 1}
    if duplicate_keys:
        print(f"[WARNING] {len(duplicate_keys)} full filenames occur multiple times.")
        for k, paths in list(duplicate_keys.items())[:20]:
            print(f"  DUPLICATE: {k}")
            for p in paths:
                print(f"    - {p}")

    new_texts = []
    matched_paths = []
    match_status = []

    updated_count = 0
    not_found_count = 0
    duplicate_count = 0
    error_count = 0

    for i, row in df.iterrows():
        excel_filename = normalize_full_filename(row["filename"])

        if excel_filename is None:
            new_texts.append(row["text"])
            matched_paths.append(None)
            match_status.append("invalid_filename")
            print(f"[{i}] Invalid filename")
            continue

        matches = file_index.get(excel_filename, [])

        if len(matches) == 0:
            new_texts.append(row["text"])
            matched_paths.append(None)
            match_status.append("not_found")
            not_found_count += 1
            print(f"[{i}] Not found: {excel_filename}")
            continue

        if len(matches) > 1:
            new_texts.append(row["text"])
            matched_paths.append(" | ".join(str(p) for p in matches))
            match_status.append("duplicate_match")
            duplicate_count += 1
            print(f"[{i}] Duplicate match for: {excel_filename}")
            for p in matches:
                print(f"      {p}")
            continue

        source_path = matches[0]

        # Final safety check: full basename must match exactly after normalization
        source_full_name = normalize_full_filename(source_path.name)
        if source_full_name != excel_filename:
            new_texts.append(row["text"])
            matched_paths.append(str(source_path))
            match_status.append("full_filename_mismatch")
            error_count += 1
            print(
                f"[{i}] FULL FILENAME MISMATCH: row={excel_filename} | source={source_full_name}"
            )
            continue

        try:
            raw_text = read_text_file(source_path)

            # NEW: remove YAML metadata
            no_yaml = remove_yaml_front_matter(raw_text)

            # THEN clean formatting
            cleaned = clean_text(no_yaml)

            new_texts.append(cleaned)
            matched_paths.append(str(source_path))
            match_status.append("matched_exact_full_filename")
            updated_count += 1

            print(
                f"[{i}] Matched exactly: {excel_filename} | raw chars: {len(raw_text)} | cleaned chars: {len(cleaned)}"
            )

        except Exception as e:
            new_texts.append(row["text"])
            matched_paths.append(str(source_path))
            match_status.append(f"read_error: {e}")
            error_count += 1
            print(f"[{i}] Error reading {source_path}: {e}")

    df["text"] = new_texts
    df["matched_source_path"] = matched_paths
    df["match_status"] = match_status

    print("Saving output...")
    df.to_excel(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Exact full-filename matches updated: {updated_count}")
    print(f"Not found: {not_found_count}")
    print(f"Duplicate full-filename matches skipped: {duplicate_count}")
    print(f"Other errors: {error_count}")
    print(f"Saved to: {OUTPUT_PATH}")

def remove_yaml_front_matter(text: str) -> str:
    """
    Remove YAML front matter at the beginning of a file:
    --- 
    ...
    ---
    """
    if not text:
        return text

    # Normalize line endings first
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Match YAML front matter only at the beginning
    pattern = r"^---\n.*?\n---\n"
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned.lstrip()

if __name__ == "__main__":
    main()
