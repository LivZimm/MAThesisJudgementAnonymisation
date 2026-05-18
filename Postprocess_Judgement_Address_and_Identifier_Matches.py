# tools/postprocess_zh_judgments.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, DefaultDict, Optional
from collections import defaultdict

import pandas as pd
import csv


# ----------------- helpers -----------------

PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
MULTI_SPLIT_RE = re.compile(r"[|;,/ ]+")
WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

def split_pipe(s: object) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in PIPE_SPLIT_RE.split(s) if x.strip()]


def norm_strict(s: str) -> str:
    """Umlauts -> ae/oe/ue; lowercase; collapse whitespace."""
    s = (s or "").lower()
    s = s.replace("ß", "ss")
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = WS_RE.sub(" ", s)
    return s.strip()


def norm_loose(s: str) -> str:
    """Umlauts -> a/o/u; lowercase; collapse whitespace."""
    s = (s or "").lower()
    s = s.replace("ß", "ss")
    s = s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    s = WS_RE.sub(" ", s)
    return s.strip()


def key_norm_strict(s: str) -> str:
    """Umlauts->ae/oe/ue + remove ALL spaces/punct."""
    return NON_ALNUM_RE.sub("", norm_strict(s))


def key_norm_loose(s: str) -> str:
    """Umlauts->a/o/u + remove ALL spaces/punct."""
    return NON_ALNUM_RE.sub("", norm_loose(s))


def uniq_preserve(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_addresses_csv(path: Path, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Address file is usually ';'. sep can be forced to avoid parser warnings.
    """
    use_sep = sep or ";"
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, sep=use_sep, dtype=str, encoding=enc, engine="python", on_bad_lines="warn")
            sample = " ".join(df.head(200).astype(str).fillna("").values.ravel())
            if "�" not in sample:
                return df
        except Exception:
            pass
    return pd.read_csv(
        path,
        sep=use_sep,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="warn",
    )


def safe_read_any(path: Path, sep: Optional[str] = None, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    Read assessed judgments:
    - xlsx/xls: read_excel
    - csv: read_csv using sep if provided; else tries ';' then ','
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if sep:
        return pd.read_csv(path, sep=sep, dtype=str, encoding=encoding, encoding_errors="replace", engine="python", on_bad_lines="warn")

    try:
        return pd.read_csv(path, sep=";", dtype=str, encoding=encoding, encoding_errors="replace", engine="python", on_bad_lines="warn")
    except Exception:
        return pd.read_csv(path, sep=",", dtype=str, encoding=encoding, encoding_errors="replace", engine="python", on_bad_lines="warn")


# ----------------- OCR spacing fix -----------------

AN_DER_FORMS = {"ander": "an der", "anden": "an den", "andem": "an dem"}

RE_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")
RE_DOTTED_ID = re.compile(r"\b[A-Za-z0-9]+\.(?:[A-Za-z0-9]+\s*)+(?:\.[A-Za-z0-9]+\s*)+\b")
RE_ID_BEFORE_WORD = re.compile(r"\b([A-Za-z]\d{2,})(?=[a-zäöü])")
RE_JOIN_SHORT = re.compile(r"\b([A-Za-zÄÖÜäöü]{2,})\s+([a-zäöü]{1,2})\b")
RE_WS_2PLUS = re.compile(r"[ \t]{2,}")
RE_WS_BEFORE_NL = re.compile(r"\s+\n")
RE_NL_3PLUS = re.compile(r"\n{3,}")

AN_DER_PATTERNS: List[Tuple[re.Pattern, str]] = []
for glued, spaced in AN_DER_FORMS.items():
    AN_DER_PATTERNS.append(
        (
            re.compile(rf"(\b[A-Za-z]\s*\d{{2,}})\s*{glued}\s*([A-Za-zÄÖÜ])\b", re.IGNORECASE),
            spaced,
        )
    )

def fix_ocr_spacing(text: str, *, no_dotted_id_fix: bool = False) -> str:
    if not text:
        return text
    t = text

    t = RE_HYPHEN_BREAK.sub(r"\1\2", t)

    def _tighten(m: re.Match) -> str:
        return WS_RE.sub("", m.group(0))

    if not no_dotted_id_fix:
        t = RE_DOTTED_ID.sub(_tighten, t)

    for pat, spaced in AN_DER_PATTERNS:
        def _repl(m: re.Match) -> str:
            left = WS_RE.sub("", m.group(1))
            return f"{left} {spaced} {m.group(2)}"
        t = pat.sub(_repl, t)

    t = RE_ID_BEFORE_WORD.sub(r"\1 ", t)
    t = RE_JOIN_SHORT.sub(r"\1\2", t)

    t = RE_WS_2PLUS.sub(" ", t)
    t = RE_WS_BEFORE_NL.sub("\n", t)
    t = RE_NL_3PLUS.sub("\n\n", t)
    return t


# ----------------- ID extraction from text -----------------

VERS_RE = re.compile(r"\bvers\.?\s*[-.]?\s*nrn?\.?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s\/\.\-]{0,40})", re.IGNORECASE)
KAT_RE  = re.compile(r"\bkat\.?\s*[-.]?\s*nrn?\.?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s\/\.\-]{0,40})",  re.IGNORECASE)
OBJ_RE  = re.compile(r"\b(?:objekt|obj)\.?\s*[-.]?\s*nrn?\.?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\s\/\.\-]{0,40})", re.IGNORECASE)

RE_NON_DIGITS = re.compile(r"\D+")

def _clean_id_token(tok: str) -> str:
    tok = (tok or "").strip()
    tok = WS_RE.sub("", tok)
    return tok.strip(".,;:()[]{}")

def _digits_count(tok: str) -> int:
    return len(RE_NON_DIGITS.sub("", tok or ""))

def extract_ids_from_text(text: str) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    text = text or ""
    vers_found: List[str] = []
    vers_ph: List[str] = []
    kat_found: List[str] = []
    kat_ph: List[str] = []
    obj_found: List[str] = []
    obj_ph: List[str] = []

    def add_bucket(tok: str, found: List[str], ph: List[str]) -> None:
        t = _clean_id_token(tok)
        if not t:
            return
        d = _digits_count(t)
        if d >= 3:
            found.append(t)
        elif d in (1, 2):
            ph.append(t)

    for m in VERS_RE.finditer(text):
        add_bucket(m.group(1), vers_found, vers_ph)
    for m in KAT_RE.finditer(text):
        add_bucket(m.group(1), kat_found, kat_ph)
    for m in OBJ_RE.finditer(text):
        add_bucket(m.group(1), obj_found, obj_ph)

    return (
        uniq_preserve(vers_found), uniq_preserve(vers_ph),
        uniq_preserve(kat_found),  uniq_preserve(kat_ph),
        uniq_preserve(obj_found),  uniq_preserve(obj_ph),
    )


# ----------------- Party / non-anonymity detection -----------------

PARTY_HEADER_TERMS = [
    "verfahrensbeteiligte","verfahrensbeteiligter","verfahrensbeteiligten","parteien",
    "rekurrent","rekurrentin","rekurrenten","rekursgegner","rekursgegnerin",
    "beschwerdefuehrer","beschwerdefuehrerin","beschwerdegegner","beschwerdegegnerin",
    "mitbeteiligte","mitbeteiligter","mitbeteiligten","vorinstanz","gegenpartei",
    "gesuchsteller","gesuchstellerin","einsprecher","einsprecherin",
]
PARTY_HEADER_RE = re.compile(r"^\s*(?:" + "|".join(re.escape(x) for x in PARTY_HEADER_TERMS) + r")\s*[:\-]?\s*$", re.IGNORECASE)
REPRESENTED_BY_RE = re.compile(r"\bvertreten\s+durch\b", re.IGNORECASE)
LAWYER_RE = re.compile(r"\b(rechtsanwalt|rechtsanwältin|advokat|fürsprecher)\b", re.IGNORECASE)

AUTHORITY_TERMS = [
    "baudirektion","direktion","kanton","stadt","gemeinde","gemeinderat","stadtrat",
    "hochbauamt","tiefbauamt","bauamt","amt","grundbuchamt","steueramt","sozialamt",
    "polizei","staatsanwaltschaft","verwaltungsgericht","obergericht","bezirksgericht",
    "bundesgericht","baurekursgericht","regierungsrat","departement","sekretariat",
    "abteilung","bausektion","bauausschuss","baukommission","planungskommission",
    "bau- und planungskommission","bau und planungskommission","kommission",
]
AUTHORITY_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in AUTHORITY_TERMS) + r")\b", re.IGNORECASE)

ANON_NAME_RE = re.compile(r"_{2,}")
INITIALS_ONLY_RE = re.compile(r"^\s*(?:[A-ZÄÖÜ]\.\s*){1,6}[A-ZÄÖÜ]\.?\s*$")

def is_anonymized_party_name(s: str) -> bool:
    s = (s or "").strip()
    return bool(s and (ANON_NAME_RE.search(s) or INITIALS_ONLY_RE.match(s)))

def looks_like_real_person_or_entity_name(s: str) -> bool:
    s = WS_RE.sub(" ", (s or "").strip())
    if len(s) < 3:
        return False
    if LAWYER_RE.search(s) or AUTHORITY_RE.search(s) or is_anonymized_party_name(s):
        return False
    if not re.search(r"[A-Za-zÄÖÜäöü]", s):
        return False
    if re.search(r"\b(ag|gmbh|verein|stiftung|genossenschaft|erben(?:gemeinschaft)?)\b", s.lower()):
        return True
    if re.search(r"\b[A-ZÄÖÜ][a-zäöü]{1,}\b", s):
        return True
    return False

def extract_non_anonymous_parties_from_header(text: str, max_lines: int = 120) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()[:max_lines]
    non_anon: List[str] = []
    in_party_block = False

    for ln in lines:
        raw = (ln or "").strip()
        if not raw:
            in_party_block = False
            continue
        if PARTY_HEADER_RE.match(raw):
            in_party_block = True
            continue
        if not in_party_block:
            continue

        if AUTHORITY_RE.search(raw) or LAWYER_RE.search(raw):
            continue

        cut = raw
        m = REPRESENTED_BY_RE.search(cut)
        if m:
            cut = cut[:m.start()].strip()

        if looks_like_real_person_or_entity_name(cut):
            non_anon.append(cut)

    return uniq_preserve(non_anon)


# ----------------- Exclude institutional/lawyer/header addresses -----------------

SECTION_START_RE = re.compile(
    r"(^\s*sachverhalt\s*$|^\s*erwaegungen\s*$|^\s*erwägungen\s*$|^\s*begruendung\s*$|^\s*begründung\s*$|^\s*entscheid\s*$|^\s*urteil\s*$|^\s*dispositiv\s*$|^\s*in\s+sachen\b)",
    re.IGNORECASE | re.MULTILINE,
)
POSTCODE_RE = re.compile(r"\b\d{4}\b")

def build_line_index(text: str) -> Tuple[List[str], List[int], List[bool]]:
    text = text or ""
    lines = text.splitlines() if text else [""]
    char_to_line = [0] * (len(text) + 1)

    header_end = min(len(lines), 120)
    for i, ln in enumerate(lines[:200]):
        if SECTION_START_RE.search(ln):
            header_end = i
            break

    is_header_line = [False] * len(lines)
    for i in range(min(header_end, len(lines))):
        is_header_line[i] = True

    pos = 0
    for i, ln in enumerate(lines):
        end = pos + len(ln)
        for j in range(pos, min(end + 1, len(char_to_line))):
            char_to_line[j] = i
        pos = end + 1

    return lines, char_to_line, is_header_line

def is_service_address_line(line: str, in_header: bool) -> bool:
    if not in_header or not line:
        return False
    ln = norm_strict(line)
    if not POSTCODE_RE.search(ln):
        return False
    if AUTHORITY_RE.search(ln):
        return True
    if LAWYER_RE.search(ln):
        return True
    if "vertreten durch" in ln:
        return True
    return False


# ----------------- Address extraction -----------------

ADDRESS_CAND_RE = re.compile(r"\b([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\.\- ]{1,80}?)\s+(\d{1,4}\s*[A-Za-z]?)\b")
NOISE_LEADS = re.compile(r"^(?:art|abs|ziff|lit|vgl|bge|urteil|entscheid|beschluss|nr|nrn|rn|vb|brke|bez|fr|s|e)$", re.IGNORECASE)

ANON_STREET_RE = re.compile(
    r"\b[A-Z]\.\_{2,}\s*(?:strasse|straße|str\.|weg|gasse|platz|allee|ring|quai|kai|hof|rain|promenade|steig|ufer|halde|buehl|bühl|tal|matte|feld|park)\b",
    re.IGNORECASE,
)
ANON_STREET_COMPACT_RE = re.compile(r"\b[A-Z]\.\_{2,}[A-Za-zÄÖÜäöüß]*?(?:strasse|straße|str)\b", re.IGNORECASE)

def extract_address_candidates_preindexed(
    text: str,
    lines: List[str],
    char_to_line: List[int],
    is_header_line: List[bool],
) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for m in ADDRESS_CAND_RE.finditer(text):
        street = (m.group(1) or "").strip()
        house = (m.group(2) or "").strip()
        if not street or not house:
            continue

        li = char_to_line[m.start()] if m.start() < len(char_to_line) else 0
        line = lines[li] if 0 <= li < len(lines) else ""
        in_header = is_header_line[li] if 0 <= li < len(is_header_line) else False
        if is_service_address_line(line, in_header=in_header):
            continue

        street_clean = re.sub(r"[^\wÄÖÜäöüß]+", " ", street).strip()
        first_word = street_clean.split(" ")[0] if street_clean else ""
        if NOISE_LEADS.match(first_word):
            continue
        if len(street_clean) < 4:
            continue

        out.append((WS_RE.sub(" ", street).strip(), WS_RE.sub("", house).strip(), m.start()))

    seen: Set[Tuple[str, str]] = set()
    res: List[Tuple[str, str, int]] = []
    for st, hn, pos in out:
        k = (st, hn)
        if k not in seen:
            seen.add(k)
            res.append((st, hn, pos))
    return res


# ----------------- FAST indices -----------------

def tokenize_addr_cell(v: object) -> List[str]:
    if v is None:
        return []
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[|;,/]+", s)
    out: List[str] = []
    for p in parts:
        p = WS_RE.sub("", p.strip())
        if not p:
            continue
        out.append(p.upper() if re.search(r"[A-Za-z]", p) else p)
    return out


FA_LEFT_RE = re.compile(r"^(.*\D)\s+(\d{1,4}\s*[A-Za-z]?)$")

def build_fa_map(addr: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    fa_map: Dict[Tuple[str, str, str], str] = {}
    for row in addr.itertuples(index=False):
        muni = str(getattr(row, "municipality", "") or "").strip()
        fa = str(getattr(row, "full_address", "") or "").strip()
        if not muni or not fa:
            continue
        left = fa.split(",")[0].strip()
        m = FA_LEFT_RE.match(left)
        if not m:
            continue
        st = m.group(1).strip()
        hn = WS_RE.sub("", m.group(2).strip())

        st_s = key_norm_strict(st)
        st_l = key_norm_loose(st)
        muni_s = norm_strict(muni)
        muni_l = norm_loose(muni)

        for mk in (muni_s, muni_l):
            for sk in (st_s, st_l):
                fa_map.setdefault((mk, sk, hn), fa)
    return fa_map


def build_fast_indices(
    addr: pd.DataFrame,
    addr_id_cols: List[str],
) -> Tuple[
    Dict[str, List[Tuple[str, str, str]]],   # token -> [(full_address, muni_strict, post_code)]
    Dict[str, Dict[str, Set[str]]],          # muni_key -> street_key -> set(house)
    Dict[str, Dict[str, str]],               # muni_key -> street_key -> display street
]:
    token_index: DefaultDict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    street_hn_index: Dict[str, Dict[str, Set[str]]] = {}
    street_display_index: Dict[str, Dict[str, str]] = {}

    for row in addr.itertuples(index=False):
        muni = str(getattr(row, "municipality", "") or "").strip()
        pc = str(getattr(row, "post_code", "") or "").strip()
        fa = str(getattr(row, "full_address", "") or "").strip()
        if not muni or not fa:
            continue

        muni_s = norm_strict(muni)
        muni_l = norm_loose(muni)

        left = fa.split(",")[0].strip()
        m = FA_LEFT_RE.match(left)
        if m:
            st = m.group(1).strip()
            hn = WS_RE.sub("", m.group(2).strip())
            st_s = key_norm_strict(st)
            st_l = key_norm_loose(st)

            for mk in (muni_s, muni_l):
                sb = street_hn_index.setdefault(mk, {})
                sd = street_display_index.setdefault(mk, {})
                for sk in (st_s, st_l):
                    sb.setdefault(sk, set()).add(hn)
                    sd.setdefault(sk, st)

        for c in addr_id_cols:
            if not hasattr(row, c):
                continue
            for tok in tokenize_addr_cell(getattr(row, c)):
                token_index[tok].append((fa, muni_s, pc))

    for tok, lst in list(token_index.items()):
        seen = set()
        new = []
        for fa, ms, pc in lst:
            k = (fa, ms, pc)
            if k not in seen:
                seen.add(k)
                new.append(k)
        token_index[tok] = new

    return dict(token_index), street_hn_index, street_display_index


# ----------------- FAST street-only detection (bounded n-grams) -----------------

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-]{3,}")

def iter_street_ngrams_keys(
    text: str,
    max_words: int = 1200,
    max_ngrams: int = 2500,
) -> Iterable[Tuple[str, str]]:
    words = WORD_RE.findall(text or "")
    if not words:
        return
    words = words[:max_words]

    emitted = 0
    for n in (1, 2, 3, 4):
        for i in range(0, len(words) - n + 1):
            gram = " ".join(words[i:i+n])
            yield key_norm_strict(gram), key_norm_loose(gram)
            emitted += 1
            if emitted >= max_ngrams:
                return


# ----------------- main -----------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # accept BOTH naming styles (so you stop fighting flags)
    ap.add_argument("--judgments", "--assessed", dest="judgments", type=Path, required=True)
    ap.add_argument("--dataset", "--addresses", dest="dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)

    ap.add_argument("--judgments-encoding", default="utf-8-sig")
    ap.add_argument("--dataset-encoding", default="cp1252")

    ap.add_argument("--judgments-sep", default=None)  # "," for judgments_utf8.csv
    ap.add_argument("--dataset-sep", default=None)    # ";" for ZH dataset

    ap.add_argument("--text-col", default="text")
    ap.add_argument("--text-col-muni", default="municipalities_found_norm")
    ap.add_argument("--text-col-plz", default="postcodes_found")

    ap.add_argument("--no-street-only", action="store_true")
    ap.add_argument("--no-dotted-id-fix", action="store_true")

    ap.add_argument(
        "--source-cols",
        default="kat_nr_found,kat_nr_placeholder_found,kat_nr_addresses_or_plots,vers_nr_found,objekt_nr_found",
    )
    ap.add_argument("--addr-id-cols", default="plot_number,plot_numbers_all,egrid_all,egid_all,egaid_all")

    args = ap.parse_args()

    df = safe_read_any(args.judgments, sep=args.judgments_sep, encoding=args.judgments_encoding)
    addr = read_addresses_csv(args.dataset, sep=args.dataset_sep)
    for need in ["full_address", "municipality", "post_code"]:
        if need not in addr.columns:
            raise SystemExit(f"Address CSV missing required column: {need}")

    if args.text_col not in df.columns:
        raise SystemExit(f"Judgment file missing text column '{args.text_col}'. Available: {list(df.columns)}")
    for need in (args.text_col_muni, args.text_col_plz):
        if need not in df.columns:
            df[need] = ""

    addr = addr.copy()
    for col in ["full_address", "municipality", "post_code"]:
        addr[col] = addr[col].fillna("").astype(str)

    addr_id_cols = [c.strip() for c in args.addr_id_cols.split(",") if c.strip() and c.strip() in addr.columns]
    if not addr_id_cols:
        raise SystemExit("None of the --addr-id-cols exist in the address CSV. Check column names.")

    token_index, street_hn_index, street_display_index = build_fast_indices(addr, addr_id_cols)
    fa_map = build_fa_map(addr)

    source_cols = [c.strip() for c in args.source_cols.split(",") if c.strip()]

    best_out: List[str] = []
    alt_out: List[str] = []
    token_out: List[str] = []
    token_source_out: List[str] = []

    vers_found_out: List[str] = []
    vers_ph_out: List[str] = []
    kat_found_out: List[str] = []
    kat_ph_out: List[str] = []
    obj_found_out: List[str] = []
    obj_ph_out: List[str] = []

    address_found_out: List[str] = []
    address_placeholder_out: List[str] = []

    party_non_anon_flag_out: List[str] = []
    party_non_anon_out: List[str] = []

    for r in df.itertuples(index=False):
        raw_text = getattr(r, args.text_col, "") or ""
        text = fix_ocr_spacing(str(raw_text), no_dotted_id_fix=args.no_dotted_id_fix)

        # line index ONCE per judgment
        lines, char_to_line, is_header_line = build_line_index(text)

        non_anon_parties = extract_non_anonymous_parties_from_header(text)
        party_non_anon_flag_out.append("1" if non_anon_parties else "0")
        party_non_anon_out.append("|".join(non_anon_parties))

        v_found, v_ph, k_found, k_ph, o_found, o_ph = extract_ids_from_text(text)
        vers_found_out.append("|".join(v_found))
        vers_ph_out.append("|".join(v_ph))
        kat_found_out.append("|".join(k_found))
        kat_ph_out.append("|".join(k_ph))
        obj_found_out.append("|".join(o_found))
        obj_ph_out.append("|".join(o_ph))

        muni_tokens = set(split_pipe(getattr(r, args.text_col_muni, "")))
        plz_tokens = set(split_pipe(getattr(r, args.text_col_plz, "")))

        muni_norms: Set[str] = set()
        for m in muni_tokens:
            muni_norms.add(norm_strict(m))
            muni_norms.add(norm_loose(m))
        plz_norms: Set[str] = set(p.strip() for p in plz_tokens if p and p.strip())

        found_addrs: List[str] = []
        placeholder_addrs: List[str] = []

        for mm in ANON_STREET_RE.finditer(text):
            li = char_to_line[mm.start()] if mm.start() < len(char_to_line) else 0
            line = lines[li] if 0 <= li < len(lines) else ""
            in_header = is_header_line[li] if 0 <= li < len(is_header_line) else False
            if not is_service_address_line(line, in_header=in_header):
                placeholder_addrs.append(mm.group(0).strip())

        for mm in ANON_STREET_COMPACT_RE.finditer(text):
            li = char_to_line[mm.start()] if mm.start() < len(char_to_line) else 0
            line = lines[li] if 0 <= li < len(lines) else ""
            in_header = is_header_line[li] if 0 <= li < len(is_header_line) else False
            if not is_service_address_line(line, in_header=in_header):
                placeholder_addrs.append(mm.group(0).strip())

        addr_cands = extract_address_candidates_preindexed(text, lines, char_to_line, is_header_line)
        matched_street_keys: Set[Tuple[str, str]] = set()

        if muni_norms and addr_cands:
            for street_raw, house_raw, _pos in addr_cands:
                st_s = key_norm_strict(street_raw)
                st_l = key_norm_loose(street_raw)
                hn = WS_RE.sub("", house_raw)

                street_exists = False
                house_exists = False
                matched_addresses: Set[str] = set()

                for mk in muni_norms:
                    sb = street_hn_index.get(mk)
                    if not sb:
                        continue
                    if st_s in sb or st_l in sb:
                        street_exists = True
                        hset = sb.get(st_s) or sb.get(st_l) or set()
                        if hn in hset:
                            house_exists = True

                for mk in muni_norms:
                    fa = fa_map.get((mk, st_s, hn)) or fa_map.get((mk, st_l, hn))
                    if fa:
                        matched_addresses.add(fa)
                        matched_street_keys.add((mk, st_s if (mk, st_s, hn) in fa_map else st_l))

                if matched_addresses:
                    found_addrs.extend(sorted(matched_addresses))
                elif street_exists and not house_exists:
                    muni_show = sorted(muni_tokens)[0] if muni_tokens else ""
                    placeholder_addrs.append(f"{street_raw} {house_raw} [{muni_show}]")

        if muni_norms and not args.no_street_only:
            body_text = "\n".join(
                ln for i, ln in enumerate(lines)
                if not (is_header_line[i] and is_service_address_line(ln, True))
            )

            seen_street_keys: Set[Tuple[str, str]] = set()
            for g_s, g_l in iter_street_ngrams_keys(body_text):
                for mk in muni_norms:
                    sb = street_hn_index.get(mk)
                    if not sb:
                        continue
                    if g_s in sb:
                        seen_street_keys.add((mk, g_s))
                    elif g_l in sb:
                        seen_street_keys.add((mk, g_l))

                if len(seen_street_keys) > 150:
                    break

            for mk, sk in sorted(seen_street_keys):
                if (mk, sk) in matched_street_keys:
                    continue
                muni_show = sorted(muni_tokens)[0] if muni_tokens else ""
                st_disp = street_display_index.get(mk, {}).get(sk, sk)
                placeholder_addrs.append(f"{st_disp} [street only, {muni_show}]")

        address_found_out.append(" | ".join(uniq_preserve(found_addrs)))
        address_placeholder_out.append(" | ".join(uniq_preserve(placeholder_addrs)))

        collected_sources: List[Tuple[str, str]] = []

        for c in source_cols:
            if not hasattr(r, c):
                continue
            for v in split_pipe(getattr(r, c)):
                for p in MULTI_SPLIT_RE.split(str(v)):
                    p = WS_RE.sub("", p.strip())
                    if not p:
                        continue
                    tok = p.upper() if re.search(r"[A-Za-z]", p) else p
                    if tok.isdigit() and len(tok) < 3:
                        continue
                    if tok.isdigit() or re.search(r"\d", tok):
                        collected_sources.append((tok, c))

        if not collected_sources:
            best_out.append("")
            alt_out.append("")
            token_out.append("")
            token_source_out.append("")
            continue

        best_parts: List[str] = []
        alt_parts: List[str] = []
        tok_parts: List[str] = []
        tok_src_parts: List[str] = []

        for token, src_col in collected_sources:
            hits = token_index.get(token, [])
            tok_parts.append(token)
            tok_src_parts.append(src_col)

            if not hits:
                best_parts.append(f"{token}=>")
                continue

            filtered: List[Tuple[str, str, str]] = []
            if muni_norms or plz_norms:
                for fa, muni_s, pc in hits:
                    if muni_norms and muni_s not in muni_norms:
                        continue
                    if plz_norms and pc not in plz_norms:
                        continue
                    filtered.append((fa, muni_s, pc))
            else:
                filtered = hits

            if not filtered and plz_norms:
                for fa, muni_s, pc in hits:
                    if pc in plz_norms:
                        filtered.append((fa, muni_s, pc))

            if not filtered:
                filtered = hits

            scored: List[Tuple[int, Tuple[str, str, str]]] = []
            for fa, muni_s, pc in filtered:
                s = 0
                if muni_norms and muni_s in muni_norms:
                    s += 6
                if plz_norms and pc in plz_norms:
                    s += 4
                scored.append((s, (fa, muni_s, pc)))
            scored.sort(key=lambda x: x[0], reverse=True)

            best_fa = scored[0][1][0]
            best_parts.append(f"{token}=>{best_fa}")

            alts: List[str] = []
            for _s, (fa, _muni_s, _pc) in scored[1:]:
                if fa != best_fa:
                    alts.append(fa)
            alts2 = uniq_preserve(alts)[:10]
            if alts2:
                alt_parts.append(f"{token}=>{' || '.join(alts2)}")

        best_out.append(" | ".join(best_parts))
        alt_out.append(" | ".join([x for x in alt_parts if x.strip()]))
        token_out.append("|".join(tok_parts))
        token_source_out.append("|".join(tok_src_parts))

    df_out = df.copy()
    df_out["vers_nr_found"] = vers_found_out
    df_out["vers_nr_placeholder_found"] = vers_ph_out
    df_out["kat_nr_found"] = kat_found_out
    df_out["kat_nr_placeholder_found"] = kat_ph_out
    df_out["obj_nr_found"] = obj_found_out
    df_out["obj_nr_placeholder_found"] = obj_ph_out

    df_out["address_found"] = address_found_out
    df_out["address_placeholder_found"] = address_placeholder_out

    df_out["party_names_has_non_anonymous"] = party_non_anon_flag_out
    df_out["party_names_non_anonymous"] = party_non_anon_out

    df_out["matched_addresses_best"] = best_out
    df_out["matched_addresses_alternatives"] = alt_out
    df_out["matched_tokens"] = token_out
    df_out["matched_token_sources"] = token_source_out

    if "anonymity_assessment" in df_out.columns:
        df_out.loc[df_out["party_names_has_non_anonymous"] == "1", "anonymity_assessment"] = "non-anonymous"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() in {".xlsx", ".xls"}:
        with pd.ExcelWriter(args.out, engine="openpyxl") as w:
            df_out.to_excel(w, index=False, sheet_name="postprocessed")
    else:
        df_out.to_csv(
            args.out,
            sep=";",
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,     # ensures newlines stay inside the cell
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            lineterminator="\n",       # consistent row endings
        )

    print("Wrote:", str(args.out))


if __name__ == "__main__":
    main()
