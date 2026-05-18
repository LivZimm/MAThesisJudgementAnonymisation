from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

import pandas as pd


# ---------------- Normalization ----------------
def norm_text(s: str) -> str:
    """Stricter normalisation: umlauts -> ae/oe/ue (keeps distinctions)."""
    s = (s or "").lower()
    s = s.replace("ß", "ss")
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_text_loose(s: str) -> str:
    """Looser normalisation: umlauts -> a/o/u (so Höri can match Hori)."""
    s = (s or "").lower()
    s = s.replace("ß", "ss")
    s = s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_street(s: str) -> str:
    s = norm_text(s)
    s = re.sub(r"[^\w ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_house(s: str) -> str:
    s = norm_text(s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w]+", "", s)
    return s.strip()


def normalize_spacing_artifacts(s: str) -> str:
    s = re.sub(r"\b([A-Za-zÄÖÜ])\s+([a-zäöü]{2,})\b", r"\1\2", s)  # A ntissa -> Antissa
    s = re.sub(r"\b([A-Za-zÄÖÜ])\s+([a-zäöü])\s+([a-zäöü])\b", r"\1\2\3", s)  # A r x -> Arx
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


# ---------------- Patterns ----------------
PARTY_HEADERS = {
    "rekurrent",
    "rekurrentin",
    "rekurrenten",
    "rekursgegner",
    "rekursgegnerin",
    "beschwerdefuehrer",
    "beschwerdefuehrerin",
    "beschwerdegegner",
    "beschwerdegegnerin",
    "mitbeteiligte",
    "mitbeteiligter",
    "mitbeteiligten",
    "vorinstanz",
    "gegenpartei",
    "gesuchsteller",
    "gesuchstellerin",
    "einsprecher",
    "einsprecherin",
    "parteien",
    "verfahrensbeteiligte",
    "verfahrensbeteiligter",
    "verfahrensbeteiligten",
}
party_header_pattern = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(h) for h in sorted(PARTY_HEADERS, key=len, reverse=True))
    + r")\s*$",
    re.IGNORECASE,
)

AUTHORITY_TERMS = [
    "baudirektion",
    "direktion",
    "kanton",
    "stadt",
    "gemeinde",
    "gemeinderat",
    "stadtrat",
    "hochbauamt",
    "tiefbauamt",
    "bauamt",
    "amt",
    "steueramt",
    "sozialamt",
    "grundbuchamt",
    "polizei",
    "staatsanwaltschaft",
    "bezirksgericht",
    "obergericht",
    "verwaltungsgericht",
    "bundesgericht",
    "regierungsrat",
    "departement",
    "sekretariat",
    "abteilung",
    "bausektion",
    "bauausschuss",
]
authority_pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in AUTHORITY_TERMS) + r")\b", re.IGNORECASE)

SIGNATURE_MARKERS = [
    "vertreten durch",
    "rechtsanwalt",
    "rechtsanwältin",
    "postfach",
    "walchetor",
    "kaspar escher-haus",
    "abteilung",
]
signature_marker_pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in SIGNATURE_MARKERS) + r")\b", re.IGNORECASE)

SECTION_START_PATTERNS = [
    r"^\s*sachverhalt\s*$",
    r"^\s*erwaegungen\s*$",
    r"^\s*erwägungen\s*$",
    r"^\s*begruendung\s*$",
    r"^\s*begründung\s*$",
    r"^\s*entscheid\s*$",
    r"^\s*urteil\s*$",
    r"^\s*dispositiv\s*$",
    r"^\s*in\s+sachen\b",
]
section_start_pattern = re.compile("|".join(SECTION_START_PATTERNS), re.IGNORECASE)

postcode_pattern = re.compile(r"\b(\d{4})\b")

street_suffixes = (
    r"(?:strasse|straße|str\.|weg|gasse|platz|allee|ring|quai|kai|hof|rain|"
    r"promenade|steig|ufer|halde|buehl|bühl|tal|matte|feld|park|str)"
)
street_no_pattern = re.compile(
    rf"\b([A-Za-zÄÖÜäöüß\.\- ]{{2,80}}?{street_suffixes})\s+(\d{{1,4}}[A-Za-z]?)\b",
    flags=re.IGNORECASE,
)

# Kat/parcel extraction (label-gated in text)
kat_nr_pattern = re.compile(
    r"\b(?:kat\.?\s*(?:[-.\u2010\u2011\u2012\u2013\u2014\u2015]?\s*)?nr\.?|kat\s*nr\.?|"
    r"katasternr\.?|parzelle|grundstueck|grundstück)\s*[:\-]?\s*(\d{1,10})\b",
    re.IGNORECASE,
)

digits_token_pattern = re.compile(r"\b\d{1,10}\b")

# extraction-only
vers_id_pat = re.compile(r"\bvers\.?\s*[-.]?\s*nr\.?\s*[:\-]?\s*([0-9]{1,10}[a-zA-Z]?)", re.IGNORECASE)
obj_id_pat = re.compile(
    r"\bobjekt\s*[-.]?\s*nr\.?\s*[:\-]?\s*([0-9]{1,5}\s*(?:[\/\-\.]\s*[0-9]{1,6})?[a-zA-Z]?)",
    re.IGNORECASE,
)

# Placeholder detection
placeholder_token_pattern = re.compile(r"\b[A-Z]\.\_{2,}\b")
placeholder_underscore_pattern = re.compile(r"_{2,}")
placeholder_single_letter_street_pattern = re.compile(r"\b[A-Z]\s*[-–]\s*strasse\b", re.IGNORECASE)

# Party extraction helpers
represented_by_pattern = re.compile(r"\bvertreten\s+durch\b", re.IGNORECASE)
party_list_prefix_pattern = re.compile(r"^\s*\d{1,3}\s*[\.\)]\s*")
role_lead_pattern = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(h) for h in sorted(PARTY_HEADERS, key=len, reverse=True))
    + r")\s*[:\-]?\s*",
    re.IGNORECASE,
)

anon_underscore_name = re.compile(r"_{2,}")  # X.________
anon_initials = re.compile(r"^\s*(?:[A-ZÄÖÜ]\.\s*){1,6}[A-ZÄÖÜ]\.?\s*$")  # E. S.
anon_single_letter_entity = re.compile(
    r"\b(?:bausektion|gemeinderat|stadtrat|bauausschuss|gemeinde|stadt)\s+[A-Z]\b",
    re.IGNORECASE,
)

possibly_identifiable_pattern = re.compile(
    r"\b("
    r"oestlich|östlich|westlich|noerdlich|nördlich|suedlich|südlich|"
    r"nordwestlich|nordöstlich|südwestlich|südöstlich|"
    r"kommunal(?:en)?\s+inventar|inventar\s+der\s+schuetzenswerten|inventar\s+der\s+schützenswerten|"
    r"quartierplan|perimeter|kernzone|bauzone|schutz(?:stellung)?|unterschutz|"
    r"objekt\s*[-.]?\s*nr\.?|vers\.?\s*[-.]?\s*nr\.?"
    r")\b",
    re.IGNORECASE,
)

geo_indicator_pattern = re.compile(
    r"\b(norden|nördlich|nord|osten|östlich|ost|süden|südlich|sued|westen|westlich|west|"
    r"nordost|nordöstlich|nordwest|nordwestlich|südost|südöstlich|suedost|"
    r"südwest|südwestlich|suedwest)\b",
    re.IGNORECASE,
)

connected_public_sources_pattern = re.compile(
    r"\b(kommunal(?:en)?\s+inventar|inventar\s+der\s+schützenswerten|inventar\s+der\s+schuetzenswerten|"
    r"quartierplan|gestaltungsplan|zonenplan|planauflage|perimeter|"
    r"objekt\s*[-.]?\s*nr\.?|vers\.?\s*[-.]?\s*nr\.?|"
    r"schutz(?:stellung)?|unterschutz|denkmalpflege)\b",
    re.IGNORECASE,
)

dossier_pattern = re.compile(
    r"\b(aktenzeichen|geschaefts(?:-| )?nr\.?|geschäfts(?:-| )?nr\.?|dossier|dossier(?:-| )?nr\.?|"
    r"referenz|ref\.|projekt(?:-| )?nr\.?|baugesuch(?:-| )?nr\.?|baubewilligung(?:-| )?nr\.?)\b",
    re.IGNORECASE,
)

dossier_number_pattern = re.compile(
    r"\b(?:aktenzeichen|geschaefts(?:-| )?nr\.?|geschäfts(?:-| )?nr\.?|dossier(?:-| )?nr\.?|"
    r"projekt(?:-| )?nr\.?|baugesuch(?:-| )?nr\.?|baubewilligung(?:-| )?nr\.?)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\/\.\-]{2,30})\b",
    re.IGNORECASE,
)

coord_pair_pattern = re.compile(r"\b(\d{6,7})\s*[,/ ]\s*(\d{6,7})\b")

egid_token_pattern = re.compile(r"\bEGID\b", re.IGNORECASE)
edid_token_pattern = re.compile(r"\bEDID\b", re.IGNORECASE)
egrid_token_pattern = re.compile(r"\bE-?GRID\b", re.IGNORECASE)

case_citation_noise_pattern = re.compile(
    r"\b("
    r"brke|vb\.\d{2,}|bez\s*\d{4}|bge|bgr|urteil|entscheid|beschluss|nr\.|nrn\.|"
    r"art\.|abs\.|lit\.|§\s*\d+|pbg|rpg|ggg|bvv|i\.v\.m\.|vgl\."
    r")\b",
    re.IGNORECASE,
)
many_digits_pattern = re.compile(r"(?:\D*\d){6,}")  # >=6 digits anywhere


# ---------------- Helpers ----------------
def looks_placeholder_street(s: str) -> bool:
    s = s or ""
    return bool(
        placeholder_token_pattern.search(s)
        or placeholder_underscore_pattern.search(s)
        or placeholder_single_letter_street_pattern.search(s)
    )


def is_placeholder_kat(k: str) -> bool:
    """
    Placeholder rule-of-thumb:
    - < 3 digits (after stripping leading zeros) => placeholder (01, 4, 99)
    - >= 3 digits => not anonymised (310)
    """
    k = (k or "").strip()
    if not k:
        return False
    k2 = k.lstrip("0")
    if k2 == "":
        k2 = "0"
    if not k2.isdigit():
        return False
    if k2 == "0":
        return True
    return len(k2) < 3


def is_public_institution_name(name: str) -> bool:
    n = norm_text(name)
    if not n:
        return False
    if authority_pattern.search(n):
        return True
    if re.search(r"\b(stadt|gemeinde)\s+[a-zäöü].{1,40}\b", n):
        return True
    if re.search(r"\b(gericht|direktion|departement|regierungsrat)\b", n):
        return True
    return False


def is_anonymized_name(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if anon_underscore_name.search(n):
        return True
    if anon_initials.match(n):
        return True
    if anon_single_letter_entity.search(n):
        return True
    return False


def clean_party_name_fragment(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = role_lead_pattern.sub("", s)
    s = party_list_prefix_pattern.sub("", s)

    m = represented_by_pattern.search(s)
    if m:
        s = s[: m.start()]

    m = re.search(r",\s*\d", s)
    if m:
        s = s[: m.start()]

    s = s.strip(" ,;:-\t")
    s = re.sub(r"\s{2,}", " ", s)
    s = normalize_spacing_artifacts(s)
    return s.strip()


def split_pipe(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def add_many(target: Set[str], values: Iterable[str]) -> None:
    for v in values:
        if v:
            target.add(v)


def add_match_groups(target: Set[str], regex: re.Pattern, text: str, group: int = 0) -> None:
    for m in regex.finditer(text or ""):
        val = (m.group(group) or "").strip()
        if val:
            target.add(re.sub(r"\s+", " ", val))


def read_addresses(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, sep=";", dtype=str, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", encoding_errors="replace")


def looks_like_party_entity(line: str) -> bool:
    s = normalize_spacing_artifacts((line or "").strip())
    if not s or len(s) < 3:
        return False
    n = norm_text(s)
    if case_citation_noise_pattern.search(n):
        return False
    if many_digits_pattern.search(s):
        return False
    if not re.search(r"[A-Za-zÄÖÜäöü]", s):
        return False
    if re.search(r"\b(ag|gmbh|verein|stiftung|genossenschaft|sarl|sa)\b", n):
        return True
    if re.search(r"\b[A-ZÄÖÜ][a-zäöü]{1,}\b.*\b[A-ZÄÖÜ][a-zäöü]{1,}\b", s):
        return True
    if re.search(r"\b(pro\s+natura|erben(?:gemeinschaft)?|verwaltung|sekretariat)\b", n):
        return True
    return False


def build_line_index(raw: str) -> Tuple[List[str], List[int], List[bool], List[bool]]:
    raw = raw or ""
    lines = raw.splitlines() if raw else [""]
    char_to_line = [0] * (len(raw) + 1)
    party_line = [False] * len(lines)
    sig_line = [False] * len(lines)

    sig_end = len(lines)
    for i, ln in enumerate(lines[:120]):
        if section_start_pattern.search(norm_text(ln)):
            sig_end = i
            break
    for i in range(min(sig_end, len(lines))):
        sig_line[i] = True

    in_party = False
    for i, ln in enumerate(lines):
        ln_norm = norm_text(ln)
        if party_header_pattern.match(ln_norm):
            in_party = True
            party_line[i] = True
            continue
        if in_party:
            party_line[i] = True
            if not ln_norm:
                in_party = False

    pos = 0
    for i, ln in enumerate(lines):
        end = pos + len(ln)
        for j in range(pos, min(end + 1, len(char_to_line))):
            char_to_line[j] = i
        pos = end + 1
    return lines, char_to_line, party_line, sig_line


def should_exclude_signature(line: str, in_sig: bool) -> bool:
    if not in_sig:
        return False
    ln = norm_text(line)
    if not (authority_pattern.search(ln) or signature_marker_pattern.search(ln)):
        return False
    if not postcode_pattern.search(ln):
        return False
    return "," in line


def should_exclude_party_authority(line: str, in_party: bool) -> bool:
    if not in_party:
        return False
    ln = norm_text(line)
    if not authority_pattern.search(ln):
        return False
    if not postcode_pattern.search(ln):
        return False
    return "," in line


def extract_party_names_from_header(lines: List[str], sig_line: List[bool]) -> Tuple[Set[str], Set[str], List[str]]:
    non_anon: Set[str] = set()
    anon: Set[str] = set()
    evidence_lines: List[str] = []

    for i, ln in enumerate(lines):
        if i >= len(sig_line) or not sig_line[i]:
            break

        raw = ln.strip()
        if not raw:
            continue
        if party_header_pattern.match(norm_text(raw)):
            continue

        cut = raw
        m = represented_by_pattern.search(cut)
        if m:
            cut = cut[: m.start()]

        if not looks_like_party_entity(cut):
            continue

        name = clean_party_name_fragment(cut)
        if not name or len(name) < 3:
            continue
        if "rechtsanw" in norm_text(name):
            continue
        if is_public_institution_name(name):
            continue

        evidence_lines.append(raw[:500])
        if is_anonymized_name(name):
            anon.add(name)
        else:
            non_anon.add(name)

    return non_anon, anon, evidence_lines


def unique_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def collect_snippets(text: str, spans: List[Tuple[int, int]], pad: int = 50) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    n = len(text)
    for a, b in spans:
        a2 = max(0, a - pad)
        b2 = min(n, b + pad)
        snippet = text[a2:b2].replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if snippet:
            out.append(snippet)
    return unique_preserve(out)


def decide_anonymity_assessment(
    *,
    has_address_or_plot_match: bool,
    has_non_anonymous_parties: bool,
    has_anonymized_parties: bool,
    has_placeholders: bool,
    has_possibly_identifiable_markers: bool,
) -> Tuple[str, str]:
    reasons: List[str] = []
    if has_address_or_plot_match:
        reasons.append("address_or_plot_matched")
    if has_non_anonymous_parties:
        reasons.append("non_anonymous_parties")
    if has_anonymized_parties:
        reasons.append("anonymized_parties_present")
    if has_placeholders:
        reasons.append("placeholders_present")
    if has_possibly_identifiable_markers:
        reasons.append("possibly_identifiable_markers")

    if has_address_or_plot_match or has_non_anonymous_parties:
        return "non-anonymous", "|".join(reasons) or "non-anonymous"
    if has_possibly_identifiable_markers:
        return "possibly identifyable with further information", "|".join(reasons) or "possibly_identifiable"
    if has_placeholders or has_anonymized_parties:
        return "anonymised", "|".join(reasons) or "anonymised"
    return "de-personalised", "|".join(reasons) or "de-personalised"


def extract_parcel_tokens_from_row(row: pd.Series, parcel_cols: List[str]) -> Set[str]:
    """
    Return normalized parcel tokens (digits) from various dataset columns.
    We filter using is_placeholder_kat: keep only >=3 digits (rule of thumb).
    """
    toks: Set[str] = set()
    for col in parcel_cols:
        if col not in row.index:
            continue
        val = row.get(col)
        if val is None:
            continue
        for t in digits_token_pattern.findall(str(val)):
            if not is_placeholder_kat(t):
                toks.add(t.lstrip("0") or "0")
    return toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--addresses", type=Path, required=True)
    ap.add_argument("--output-xlsx", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--text-col", type=str, default="text")
    ap.add_argument("--kat-fallback", choices=["if_no_address", "always", "off"], default="if_no_address")
    ap.add_argument(
        "--parcel-cols",
        type=str,
        default="plot_numbers_all,lgbkr_all,egrid_all",
        help="Comma-separated address columns to use as parcel token sources for Kat.-Nr. matching.",
    )
    # important: avoid pulling '*' candidates when multiple municipalities might exist
    ap.add_argument(
        "--kat-require-plz-if-multi",
        action="store_true",
        help="If more than one municipality is detected in a judgment, only allow Kat matching with explicit PLZ.",
    )
    args = ap.parse_args()

    judg = pd.read_excel(args.judgments)
    if args.text_col not in judg.columns:
        raise SystemExit(f"Missing text column '{args.text_col}'. Available: {list(judg.columns)}")

    addr = read_addresses(args.addresses)
    required = {"street_name", "house_number", "post_code", "municipality", "full_address"}
    missing = required - set(addr.columns)
    if missing:
        raise SystemExit(f"Address file missing columns: {sorted(missing)}")

    # ensure optional cols exist
    for c in ("plot_numbers_all", "egrid_all", "lgbkr_all", "Gemeindename"):
        if c not in addr.columns:
            addr[c] = ""

    parcel_cols = [c.strip() for c in (args.parcel_cols or "").split(",") if c.strip()]
    if not parcel_cols:
        parcel_cols = ["plot_numbers_all", "lgbkr_all", "egrid_all"]

    addr = addr.copy()
    addr["street_norm"] = addr["street_name"].fillna("").astype(str).map(norm_street)
    addr["house_norm"] = addr["house_number"].fillna("").astype(str).map(norm_house)
    addr["post_code"] = addr["post_code"].fillna("").astype(str).str.strip()
    addr["municipality"] = addr["municipality"].fillna("").astype(str).str.strip()
    addr["Gemeindename"] = addr["Gemeindename"].fillna("").astype(str).str.strip()
    addr["full_address"] = addr["full_address"].fillna("").astype(str).str.strip()

    def muni_variants(row: pd.Series) -> List[str]:
        """
        Store BOTH strict+loose variants in the dataset index
        so Höri can match either "hoeri" or "hori" in the text.
        """
        vals: Set[str] = set()

        def add_muni(x: str) -> None:
            x = (x or "").strip()
            if not x:
                return
            vals.add(norm_text(x))        # Höri -> hoeri
            vals.add(norm_text_loose(x))  # Höri -> hori

        add_muni(str(row.get("municipality", "")))
        for x in str(row.get("Gemeindename", "")).split("|"):
            add_muni(x)

        return sorted(vals)

    addr["muni_variants"] = addr.apply(muni_variants, axis=1)

    # (street_norm, house_norm) -> candidates
    key_to_candidates: Dict[Tuple[str, str], List[Tuple[str, str, List[str], str, str, str]]] = {}
    for st, hn, fa, pc, mv, pl, eg, lg in addr[
        ["street_norm", "house_norm", "full_address", "post_code", "muni_variants", "plot_numbers_all", "egrid_all", "lgbkr_all"]
    ].itertuples(index=False, name=None):
        if st and hn and fa:
            key_to_candidates.setdefault((st, hn), []).append((fa, pc, mv, pl, eg, lg))

    postcodes: Set[str] = set(addr["post_code"].dropna().astype(str).tolist())

    # Build muni pattern lists (split strict vs loose) and patterns
    all_muni_strict: List[str] = []
    all_muni_loose: List[str] = []
    for mv in addr["muni_variants"].tolist():
        for token in mv:
            # heuristic: strict-normalized tokens usually contain 'ae/oe/ue' when umlauts exist
            # but we just build both sets explicitly:
            all_muni_strict.append(token)
            all_muni_loose.append(token)

    # de-dup, longest first to reduce partial matching risk
    all_muni_strict = sorted(set(all_muni_strict), key=lambda x: (-len(x), x))
    all_muni_loose = sorted(set(all_muni_loose), key=lambda x: (-len(x), x))

    muni_pattern_strict = re.compile(r"\b(" + "|".join(re.escape(m) for m in all_muni_strict) + r")\b") if all_muni_strict else None
    muni_pattern_loose = re.compile(r"\b(" + "|".join(re.escape(m) for m in all_muni_loose) + r")\b") if all_muni_loose else None

    # muni_variant -> kat -> plz -> candidates
    muni_kat_plz_index: DefaultDict[str, DefaultDict[str, DefaultDict[str, List[Tuple[str, str, List[str], str, str, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    if args.kat_fallback != "off":
        # Build index from parcel token sources
        for _, row in addr.iterrows():
            fa = str(row.get("full_address") or "").strip()
            pc = str(row.get("post_code") or "").strip()
            mv = row.get("muni_variants") or []
            pl = str(row.get("plot_numbers_all") or "")
            eg = str(row.get("egrid_all") or "")
            lg = str(row.get("lgbkr_all") or "")
            if not fa or not pc or not mv:
                continue

            parcel_tokens = extract_parcel_tokens_from_row(row, parcel_cols)
            if not parcel_tokens:
                continue

            muni_orig = str(row.get("municipality") or "").strip()
            cand_tuple = (fa, pc, mv, pl, eg, lg, muni_orig)
            for mvar in mv:
                for kat in parcel_tokens:
                    muni_kat_plz_index[mvar][kat][pc].append(cand_tuple)
                    muni_kat_plz_index[mvar][kat]["*"].append(cand_tuple)

    def analyze_text(raw: str) -> dict:
        raw = raw or ""
        t_norm = norm_text(raw)
        t_norm_loose = norm_text_loose(raw)

        spans_for_evidence: List[Tuple[int, int]] = []

        # ---------------- municipality detection (FIX to avoid wrong municipality matches) ----------------
        strict_found: Set[str] = set(muni_pattern_strict.findall(t_norm)) if muni_pattern_strict else set()
        loose_found: Set[str] = set(muni_pattern_loose.findall(t_norm_loose)) if muni_pattern_loose else set()

        # IMPORTANT FIX:
        # Prefer strict matches; only if none exist, fall back to loose matches
        mun_found: Set[str] = strict_found if strict_found else loose_found

        # Evidence spans for municipality tokens (indices must come from raw)
        if mun_found:
            mun_re = re.compile(
                r"\b(" + "|".join(re.escape(x) for x in sorted(mun_found, key=len, reverse=True)) + r")\b",
                flags=re.IGNORECASE,
            )
            for m in mun_re.finditer(raw):
                spans_for_evidence.append((m.start(), m.end()))

        # ---------------- postcode ----------------
        pcs_found = sorted({m.group(1) for m in postcode_pattern.finditer(raw) if m.group(1) in postcodes})
        for m in postcode_pattern.finditer(raw):
            if m.group(1) in postcodes:
                spans_for_evidence.append((m.start(), m.end()))

        # ---------------- kat in text ----------------
        kat_iters = list(kat_nr_pattern.finditer(raw))
        kat_all = sorted({m.group(1) for m in kat_iters})
        spans_for_evidence.extend([(m.start(), m.end()) for m in kat_iters])

        kat_placeholders = sorted({k for k in kat_all if is_placeholder_kat(k)})
        kat_found = sorted({(k.lstrip("0") or "0") for k in kat_all if not is_placeholder_kat(k)})

        # If we have Kat numbers, restrict municipality candidates to those near any Kat mention (reduces false muni hits)
        if kat_iters and mun_found:
            kat_spans = [(m.start(), m.end()) for m in kat_iters]
            # find muni occurrences in raw (for distance filtering)
            muni_occurrences: List[Tuple[str, int, int]] = []
            mun_re = re.compile(
                r"\b(" + "|".join(re.escape(x) for x in sorted(mun_found, key=len, reverse=True)) + r")\b",
                flags=re.IGNORECASE,
            )
            for mm in mun_re.finditer(raw):
                muni_occurrences.append((mm.group(1), mm.start(), mm.end()))

            # keep municipalities that appear within 800 chars of any Kat mention
            near: Set[str] = set()
            for token, a, b in muni_occurrences:
                for ka, kb in kat_spans:
                    if abs(a - ka) <= 800 or abs(ka - a) <= 800:
                        near.add(token)
                        break
            if near:
                mun_found = near

        # ---------------- vers/obj ----------------
        vers_iters = list(vers_id_pat.finditer(raw))
        vers_found = sorted({re.sub(r"\s+", "", m.group(1)) for m in vers_iters})
        spans_for_evidence.extend([(m.start(), m.end()) for m in vers_iters])

        obj_iters = list(obj_id_pat.finditer(raw))
        obj_found = sorted({re.sub(r"\s+", "", m.group(1)) for m in obj_iters})
        spans_for_evidence.extend([(m.start(), m.end()) for m in obj_iters])

        spans_for_evidence.extend([(m.start(), m.end()) for m in possibly_identifiable_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in geo_indicator_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in connected_public_sources_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in dossier_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in dossier_number_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in coord_pair_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in egid_token_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in edid_token_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in egrid_token_pattern.finditer(raw)])

        # header party extraction
        lines, char_to_line, party_line, sig_line = build_line_index(raw)
        party_non_anon, party_anon, party_evidence_lines = extract_party_names_from_header(lines, sig_line)

        # placeholder streets
        placeholder_streets: Set[str] = set()
        for m in street_no_pattern.finditer(raw):
            if looks_placeholder_street(m.group(1)) or placeholder_underscore_pattern.search(m.group(1)):
                placeholder_streets.add(re.sub(r"\s+", " ", m.group(1)).strip())
                spans_for_evidence.append((m.start(1), m.end(2)))

        spans_for_evidence.extend([(m.start(), m.end()) for m in placeholder_token_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in placeholder_single_letter_street_pattern.finditer(raw)])
        spans_for_evidence.extend([(m.start(), m.end()) for m in placeholder_underscore_pattern.finditer(raw)])

        # matches
        matched_full: Set[str] = set()
        matched_plots: Set[str] = set()
        matched_egrid: Set[str] = set()
        matched_lgbkr: Set[str] = set()

        addr_matches = 0
        excl_sig = 0
        excl_party = 0

        # 1) street+no matches
        for m in street_no_pattern.finditer(raw):
            if looks_placeholder_street(m.group(1)) or placeholder_underscore_pattern.search(m.group(1)):
                continue

            li = char_to_line[m.start()]
            line = lines[li] if 0 <= li < len(lines) else ""

            if should_exclude_signature(line, sig_line[li] if 0 <= li < len(sig_line) else False):
                excl_sig += 1
                continue
            if should_exclude_party_authority(line, party_line[li] if 0 <= li < len(party_line) else False):
                excl_party += 1
                continue

            st = norm_street(m.group(1))
            hn = norm_house(m.group(2))
            cand = key_to_candidates.get((st, hn))
            if not cand:
                continue

            if len(cand) > 1 and pcs_found:
                cand2 = [c for c in cand if c[1] in pcs_found]
                if cand2:
                    cand = cand2
            if len(cand) > 1 and mun_found:
                cand2 = [c for c in cand if any(v in mun_found for v in c[2])]
                if cand2:
                    cand = cand2

            addr_matches += 1
            spans_for_evidence.append((m.start(), m.end()))
            for fa, pc, mv, pl, eg, lg in cand:
                matched_full.add(fa)
                add_many(matched_plots, split_pipe(pl))
                add_many(matched_egrid, split_pipe(eg))
                add_many(matched_lgbkr, split_pipe(lg))

                # 2) Kat fallback: choose BEST match by proximity (municipality mention closest to Kat.-Nr. in text)
        kat_match_best_full: Set[str] = set()
        kat_match_count = 0
        kat_other_candidate_municipalities: Set[str] = set()

        # --- positions of Kat mentions in raw text (for distance scoring) ---
        kat_positions = [m.start() for m in kat_iters]  # kat_iters computed above

        # --- positions of municipality mentions in raw text ---
        muni_pos: Dict[str, List[int]] = {}
        if mun_found:
            mun_re_for_pos = re.compile(
                r"\b(" + "|".join(re.escape(x) for x in sorted(mun_found, key=len, reverse=True)) + r")\b",
                flags=re.IGNORECASE,
            )
            for mm in mun_re_for_pos.finditer(raw):
                # store both strict+loose normalized keys so we can score using mun_found tokens
                muni_pos.setdefault(norm_text(mm.group(1)), []).append(mm.start())
                muni_pos.setdefault(norm_text_loose(mm.group(1)), []).append(mm.start())

        def best_score_for_mvar(mvar: str) -> int:
            """Minimal absolute distance between any Kat mention and any municipality mention."""
            poss = muni_pos.get(mvar) or []
            if not poss or not kat_positions:
                return 10**9
            return min(abs(kp - mp) for kp in kat_positions for mp in poss)

        kat_should_run = args.kat_fallback == "always" or (args.kat_fallback == "if_no_address" and addr_matches == 0)
        if kat_should_run and kat_found and mun_found and args.kat_fallback != "off":
            # if more than one muni token and user enabled flag -> require explicit PLZ (no '*' fallback)
            if args.kat_require_plz_if_multi and len(mun_found) > 1:
                plz_candidates = pcs_found[:]  # may be empty -> then nothing matches
            else:
                plz_candidates = pcs_found[:] if pcs_found else ["*"]

            scored: List[Tuple[int, str, str]] = []  # (score, full_address, muni_orig)

            for mvar in mun_found:
                score = best_score_for_mvar(mvar)
                if score >= 10**9:
                    continue

                for kat in kat_found:
                    by_plz = muni_kat_plz_index.get(mvar, {}).get(kat)
                    if not by_plz:
                        continue

                    pulled: List[Tuple[str, str, List[str], str, str, str, str]] = []
                    for plz in plz_candidates:
                        pulled.extend(by_plz.get(plz, []))

                    if not pulled and ("*" in plz_candidates):
                        pulled = by_plz.get("*", [])

                    if not pulled:
                        continue

                    kat_match_count += len(pulled)

                    for fa, pc, mv, pl, eg, lg, muni_orig in pulled:
                        bonus = -5 if (pcs_found and pc in pcs_found) else 0
                        scored.append((score + bonus, fa, muni_orig))

                        # keep side-matches
                        add_many(matched_plots, split_pipe(pl))
                        add_many(matched_egrid, split_pipe(eg))
                        add_many(matched_lgbkr, split_pipe(lg))

            if scored:
                scored.sort(key=lambda x: (x[0], x[1]))
                best_score, best_fa, best_muni = scored[0]
                kat_match_best_full.add(best_fa)
                matched_full.add(best_fa)

                for _, _, muni_orig in scored[1:]:
                    if muni_orig and muni_orig != best_muni:
                        kat_other_candidate_municipalities.add(muni_orig)

        # Kat output:
        # - if best address found: output that ONE
        # - else: output ONLY non-placeholder Kat.-Nr. (>=3 digits)
        kat_addr_or_plot: List[str] = []
        if kat_match_best_full:
            kat_addr_or_plot.extend(sorted(kat_match_best_full))
        else:
            for k in kat_all:
                if not is_placeholder_kat(k):
                    kat_addr_or_plot.append(f"Kat.-Nr. {k.lstrip('0') or '0'}")

        kat_nr_addresses_or_plots = "|".join(unique_preserve(kat_addr_or_plot))
        kat_other_candidate_municipalities_str = "|".join(sorted(unique_preserve(kat_other_candidate_municipalities)))


        # match level
        has_addr = addr_matches > 0
        has_plot = bool(kat_match_best_full)
        has_muni = bool(mun_found)
        has_pc = bool(pcs_found)

        if has_addr:
            if has_muni and has_pc:
                level = "street+number+postcode+municipality"
            elif has_pc:
                level = "street+number+postcode"
            elif has_muni:
                level = "street+number+municipality"
            else:
                level = "street+number"
        elif has_plot:
            level = "parcel+municipality"
        elif has_muni:
            level = "municipality_only"
        else:
            level = "none"

        # municipality-only counts as possibly identifiable (also if street anonymised)
        has_municipality_only = (bool(mun_found) or bool(pcs_found)) and not bool(matched_full) and not bool(party_non_anon)

        has_dossier = bool(dossier_pattern.search(t_norm)) or bool(dossier_number_pattern.search(t_norm))
        has_coords = bool(coord_pair_pattern.search(raw))
        has_building_ids = bool(egid_token_pattern.search(raw)) or bool(edid_token_pattern.search(raw)) or bool(egrid_token_pattern.search(raw))

        has_possibly_identifiable_markers = (
            bool(possibly_identifiable_pattern.search(t_norm))
            or bool(geo_indicator_pattern.search(t_norm))
            or bool(connected_public_sources_pattern.search(t_norm))
            or bool(vers_found)
            or bool(obj_found)
            or has_dossier
            or has_coords
            or has_building_ids
            or has_municipality_only
            or (bool(placeholder_streets) and (bool(mun_found) or bool(pcs_found)))
        )

        has_placeholders = (
            bool(placeholder_streets)
            or bool(kat_placeholders)
            or bool(party_anon)
            or bool(placeholder_underscore_pattern.search(raw))
        )

        category, reasons = decide_anonymity_assessment(
            has_address_or_plot_match=bool(matched_full),
            has_non_anonymous_parties=bool(party_non_anon),
            has_anonymized_parties=bool(party_anon),
            has_placeholders=has_placeholders,
            has_possibly_identifiable_markers=has_possibly_identifiable_markers,
        )

        # evidence snippets
        evidence_snips = collect_snippets(raw, spans_for_evidence, pad=50)
        if party_evidence_lines:
            evidence_snips.extend([re.sub(r"\s+", " ", x).strip() for x in party_evidence_lines[:10]])
        evidence_snips = unique_preserve(evidence_snips)
        evidence_snips_str = " || ".join(evidence_snips)[:30000]

        # identifiable info extracted (NO context)
        identifiable_tokens: Set[str] = set()
        add_many(identifiable_tokens, sorted(matched_full))
        if kat_nr_addresses_or_plots:
            add_many(identifiable_tokens, split_pipe(kat_nr_addresses_or_plots))
        add_many(identifiable_tokens, sorted(matched_plots))
        add_many(identifiable_tokens, sorted(matched_egrid))
        add_many(identifiable_tokens, sorted(matched_lgbkr))
        add_many(identifiable_tokens, vers_found)
        add_many(identifiable_tokens, obj_found)
        add_many(identifiable_tokens, sorted(mun_found))
        add_many(identifiable_tokens, pcs_found)
        add_match_groups(identifiable_tokens, dossier_number_pattern, raw, group=1)
        add_match_groups(identifiable_tokens, coord_pair_pattern, raw, group=0)
        if egid_token_pattern.search(raw):
            identifiable_tokens.add("EGID")
        if edid_token_pattern.search(raw):
            identifiable_tokens.add("EDID")
        if egrid_token_pattern.search(raw):
            identifiable_tokens.add("EGRID")
        add_match_groups(identifiable_tokens, geo_indicator_pattern, raw, group=0)
        add_match_groups(identifiable_tokens, connected_public_sources_pattern, raw, group=0)

        identifiable_info_extracted = "|".join(unique_preserve(sorted(identifiable_tokens)))

        # possibly-identifiable subcategories
        primary = ""
        other: List[str] = []
        if category == "possibly identifyable with further information":
            subs: List[str] = []
            if has_coords:
                subs.append("coordinates / spatial reference")
            if has_building_ids:
                subs.append("building or registry identifiers (EGID/EDID/EGRID)")
            if has_dossier:
                subs.append("administrative file / dossier identifiers")
            if geo_indicator_pattern.search(raw):
                subs.append("geographic descriptions")
            if connected_public_sources_pattern.search(raw) or vers_found or obj_found:
                subs.append("connected to other public sources")
            if (bool(mun_found) or bool(pcs_found)) and not matched_full and not party_non_anon:
                if mun_found and pcs_found:
                    subs.append("municipality name and post code")
                elif mun_found:
                    subs.append("municipality name")
                else:
                    subs.append("municipality name and post code")

            subs = unique_preserve(subs)
            if subs:
                primary = subs[0]
                other = subs[1:]

        return {
            "addr_info_found": bool(matched_full or has_muni),
            "match_level": level,
            "municipalities_found_norm": "|".join(sorted(mun_found)),
            "postcodes_found": "|".join(pcs_found),
            "kat_nr_found": "|".join(kat_found),
            "kat_nr_placeholder_found": "|".join(kat_placeholders),
            "kat_nr_addresses_or_plots": kat_nr_addresses_or_plots,
            "kat_other_candidate_municipalities": kat_other_candidate_municipalities_str,
            "vers_nr_found": "|".join(vers_found),
            "objekt_nr_found": "|".join(obj_found),
            "address_matches_count": int(addr_matches),
            "kat_match_count": int(kat_match_count),
            "full_addresses_matched": "|".join(sorted(matched_full)),
            "plot_numbers_matched": "|".join(sorted(matched_plots)),
            "egrid_matched": "|".join(sorted(matched_egrid)),
            "lgbkr_matched": "|".join(sorted(matched_lgbkr)),
            "placeholder_street_found": "1" if placeholder_streets else "0",
            "placeholder_street_examples": "|".join(sorted(placeholder_streets))[:3000],
            "party_names_non_anonymous": "|".join(sorted(party_non_anon)),
            "party_names_anonymized": "|".join(sorted(party_anon)),
            "party_names_has_non_anonymous": "1" if party_non_anon else "0",
            "party_names_has_anonymized": "1" if party_anon else "0",
            "anonymity_assessment": category,
            "anonymity_assessment_reasons": reasons,
            "assessment_evidence_snippets": evidence_snips_str,
            "identifiable_info_extracted": identifiable_info_extracted,
            "possibly_identifiable_primary": primary,
            "possibly_identifiable_other": "|".join(other),
            "excluded_signature_addresses_count": int(excl_sig),
            "excluded_authority_party_addresses_count": int(excl_party),
        }

    results = pd.DataFrame([analyze_text(t) for t in judg[args.text_col].fillna("").astype(str).tolist()])
    out = pd.concat([judg, results], axis=1)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="judgments_addresses")

    out.to_csv(args.output_csv, sep=";", index=False, encoding="utf-8")
    print("Wrote:", args.output_xlsx)
    print("Wrote:", args.output_csv)


if __name__ == "__main__":
    main()
