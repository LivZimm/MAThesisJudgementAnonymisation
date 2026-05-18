from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import pandas as pd
import pdfplumber
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================
# CONFIG
# =========================

MODEL_NAME = "gpt-5"
MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.5

EXCEL_CELL_LIMIT = 32767
EXCEL_TRUNCATION_SUFFIX = "... [TRUNCATED]"
EMPTY_EXAMPLES_PLACEHOLDER = "-"


# =========================
# STRUCTURED OUTPUT SCHEMA
# =========================

Category = Literal[
    "Anonymised",
    "Possibly identifiable with further information",
    "Non-anonymous",
]


class Section(BaseModel):
    value: bool
    examples: list[str] = Field(default_factory=list)


class DateSection(BaseModel):
    value: bool
    examples: list[str] = Field(default_factory=list)
    reasoning: str = ""


class AnalysisResult(BaseModel):
    category: Category

    private_person_party_full_names_present: Section
    private_person_party_anonymised_names_present: Section

    company_association_party_full_names_present: Section
    company_association_party_anonymised_names_present: Section

    third_party_professional_names: Section

    addresses_present: Section
    addresses_anonymised: Section

    parcel_numbers_present: Section
    parcel_numbers_anonymised: Section

    dates_case_related: DateSection

    municipality_present: Section
    municipality_anonymised: Section

    location_descriptions_present: Section
    other_potentially_identifying_information: Section

    name_assessment: str
    justification: str


# =========================
# PROMPTS
# =========================

SYSTEM_PROMPT = """
Du bist ein präziser wissenschaftlicher Kodierassistent.
Du analysierst Gerichtsurteile im Hinblick auf ihren Grad der Anonymisierung.
Beurteile streng nach den vorgegebenen Kriterien.
Wenn ein Feld den Wert true hat, musst du mindestens ein konkretes Beispiel angeben.
""".strip()


USER_PROMPT_TEMPLATE = """
Analysiere den folgenden Urteilstext und bewerte, wie stark er anonymisiert ist.

Hinweis zum Eingabetext:
Der Text stammt aus einer PDF-Extraktion und kann Artefakte enthalten
(verlorene Umlaute wie "Zuerich" statt "Zürich", auseinandergerissene Komposita
wie "Rechts anwalt", non-breaking spaces, Soft-Hyphens). Behandle solche Formen
semantisch korrekt, ohne dich davon irritieren zu lassen.

Ziel:
Es geht darum, ob die eigentlichen Fallparteien direkt identifizierbar sind
oder mit Zusatzwissen re-identifiziert werden könnten.

Ordne den Text genau einer der folgenden Kategorien zu:

1. "Anonymised"
- Keine direkt identifizierenden Angaben zu den eigentlichen Fallparteien vorhanden.
- Weiterhin als anonym gelten insbesondere Texte, die nur folgende Angaben enthalten:
  - Gemeindenamen
  - verfahrensinterne Daten
  - Namen von Berufsvertretern, Richtern, Gerichtsschreibern, Amtspersonen oder Behördenvertretern
  - anonymisierte Namen / Initialen / Platzhalter
- Gemeindenamen allein machen ein Urteil NICHT non-anonymous und auch nicht automatisch re-identifizierbar.

2. "Possibly identifiable with further information"
- Die eigentlichen Parteien sind nicht direkt namentlich offengelegt,
  aber es bestehen konkrete Re-Identifikationsrisiken durch zusätzliche Informationen.
- Dazu zählen insbesondere:
  - echte, nicht anonymisierte Adressen
  - echte, nicht anonymisierte Grundstücknummern
  - echte sonstige objekt- oder fallbezogene Identifikatoren
  - Gemeindenamen in Verbindung mit weiteren konkreten Lage- oder Objektbeschreibungen
  - sonstige charakteristische geografische oder sachliche Informationen

3. "Non-anonymous"
- Echte Namen der eigentlichen Parteien sind vollständig sichtbar
- Oder andere unmittelbare eindeutige Identifikatoren der eigentlichen Parteien sind direkt sichtbar

Wichtige Abgrenzung:
- Gemeindenamen allein genügen NICHT für "Possibly identifiable with further information".
- Verfahrensinterne Daten genügen NICHT für "Possibly identifiable with further information".
- Namen von Anwälten, Richtern, Gerichtsschreibern, Amtspersonen, Behördenvertretern oder anderen Berufsparteien genügen NICHT für "Possibly identifiable with further information".
- Diese Angaben sind separat zu erfassen, haben aber für die Hauptkategorie grundsätzlich kein Gewicht.

A. Parteiennamen
Unterscheide die eigentlichen Fallparteien strikt in:

1. private_person_party_full_names_present
- echte, nicht anonymisierte volle Namen privater Personen, die eigentliche Fallparteien sind

2. private_person_party_anonymised_names_present
- anonymisierte Namen privater Personen, die eigentliche Fallparteien sind

3. company_association_party_full_names_present
- echte, nicht anonymisierte Namen von Firmen, Vereinen, Stiftungen, Verbänden,
  Genossenschaften oder anderen privaten Organisationen, die eigentliche Fallparteien sind

4. company_association_party_anonymised_names_present
- anonymisierte Namen solcher Firmen, Vereine, Stiftungen, Verbände,
  Genossenschaften oder anderer privater Organisationen, die eigentliche Fallparteien sind

Wichtige Regeln:
- anonymisierte Namen dürfen NICHT zugleich in den nicht anonymisierten Feldern gezählt werden
- Gemeinden sind KEINE company/association parties
- Gemeinden dürfen NICHT in company_association_party_* Feldern erscheinen
- Gemeinden gehören ausschliesslich in municipality_present oder municipality_anonymised
- Behörden und öffentliche Institutionen sind ebenfalls nicht automatisch company/association parties

Die folgenden Formen gelten als anonymisiert und NICHT als voll sichtbare Namen:
- reine Initialen wie "A.", "B.", "C."
- Kombinationen wie "A. B.", "A.A.", "B.B."
- Buchstaben-Platzhalter wie "X", "Y", "Z"
- Formen wie "A.________", "B.________", "A.___"
- abstrahierte Firmen-/Organisationsbezeichnungen wie "L AG", "M GmbH", sofern nur ein Buchstabe oder offensichtlicher Platzhalter verwendet wird
- Rollenbezeichnungen mit Platzhalter wie "Gemeinderat X"

B. Professionelle Dritte
- Namen von Richtern, Gerichtsschreibern, Anwälten, Professoren, Gutachtern,
  Amtspersonen und Behördenvertretern gehören ausschliesslich in "third_party_professional_names"
- sie dürfen NICHT in die Parteienfelder eingeordnet werden

C. Adressen
- "addresses_present" erfasst nur echte, nicht anonymisierte Adressen.
- "addresses_anonymised" erfasst nur anonymisierte Adressen.
- anonymisierte Adressen dürfen NICHT zugleich in "addresses_present" gezählt werden.

Die folgenden Formen gelten als anonymisiert und NICHT als echte Adressen:
- maskierte Strassennamen wie "P-/Q-strasse", "P-strasse", "X-strasse"
- sonstige klar abstrahierte Strassenangaben mit Buchstaben-Platzhaltern
- maskierte Hausnummern

D. Grundstücknummern
- "parcel_numbers_present" erfasst nur echte, nicht anonymisierte Grundstücknummern.
- "parcel_numbers_anonymised" erfasst nur anonymisierte Grundstücknummern.
- anonymisierte Grundstücknummern dürfen NICHT zugleich in "parcel_numbers_present" gezählt werden.

Die folgenden Formen gelten als anonymisiert und NICHT als echte Grundstücknummern:
- "Kat.Nr. 4"
- "Kat.Nrn. 1 und 2-3"
- vergleichbare Platzhalter-Nummern, wenn klar ist, dass sie nicht die echte Parzellennummer wiedergeben, sondern anonymisierte Ersatznummern sind

E. Gemeinden / Orte / Lageangaben
- "municipality_present" erfasst nur echte, nicht anonymisierte Gemeindenamen.
- "municipality_anonymised" erfasst nur anonymisierte Gemeindeangaben.
- anonymisierte Gemeindeangaben dürfen NICHT zugleich in "municipality_present" gezählt werden.
- Gemeindenamen allein sind für die Hauptkategorie nicht ausschlaggebend.
- Erst wenn ein Gemeindename zusammen mit zusätzlichen konkreten Lage- oder Objektbeschreibungen erscheint,
  kann "Possibly identifiable with further information" vorliegen.
- "location_descriptions_present" erfasst konkrete Lage-, Objekt- oder Umgebungsbeschreibungen.

F. Nicht als identifizierend für die Hauptkategorie werten
- Gerichtsname
- Spruchkörper
- Namen von Richtern, Gerichtsschreibern, Anwälten, Professoren, Gutachtern
- Namen von Amtspersonen und Behördenvertretern
- Behörden und öffentliche Institutionen
- Verfahrenshandlungen wie Einreichung einer Klageschrift, Beschwerde, Eingabe, Stellungnahme oder Verfügung,
  sofern daraus keine öffentlich recherchierbare Identifikation der eigentlichen Parteien folgt

G. Fallbezogene Datumsangaben
- Nur echte, konkret fallbezogene und öffentlich recherchierbare Datumsangaben zählen als identifizierend.
- Allgemeine Verfahrensdaten, Beschwerdeeingaben, Verfügungsdaten, Gesetzesdaten oder historische Hintergrunddaten
  sind nicht automatisch identifizierend.
- Wenn "dates_case_related.value" true ist, müssen unter "dates_case_related.examples" die konkreten Datumsangaben
  wörtlich oder nahezu wörtlich aufgeführt werden, z.B. "24. August 2011".
- Wenn keine solchen konkreten Datumsangaben sicher extrahiert werden können, setze "dates_case_related.value" auf false.

H. Beispiele sind verpflichtend
- Für JEDES Feld mit "value": true musst du mindestens ein konkretes Beispiel angeben.
- "examples" darf NUR dann leer sein, wenn "value": false ist.
- Gib möglichst kurze, wörtliche oder nahezu wörtliche Textausschnitte an.

I. Bedeutung von Parteinamen für die Hauptkategorie
- Wenn der Name einer privaten Person als eigentliche Partei vollständig sichtbar ist:
  → Kategorie = "Non-anonymous"
- Wenn der Name einer Firma, eines Vereins oder einer Organisation als eigentliche Partei vollständig sichtbar ist:
  → Dies stellt ebenfalls eine nicht anonymisierte Partei dar.
- In diesem Fall:
  → Kategorie = "Non-anonymous"
- Es ist NICHT zulässig, ein Urteil als "Anonymised" zu klassifizieren,
  wenn eine Partei (egal ob private Person oder Unternehmen/Organisation)
  mit ihrem echten, nicht anonymisierten Namen genannt wird.

K. Sonstige potenziell identifizierende Informationen
- Erfasse unter "other_potentially_identifying_information" alle sonstigen Informationen,
  die nicht bereits durch die anderen Felder abgedeckt sind,
  aber potenziell zur Identifikation oder Re-Identifikation der eigentlichen Fallparteien beitragen können.
- Nutze hierfür dein eigenes begründetes Urteil.
- Dazu können insbesondere gehören:
  - ungewöhnlich spezifische Sachverhaltsdetails
  - wörtliche oder nahezu wörtliche Zitate aus öffentlich zugänglichen Listen, Registern oder Publikationen
  - sehr spezifische Objekt-, Nutzungs-, Projekt- oder Verfahrenskonstellationen
  - Kombinationen mehrerer Informationen, die zusammen identifizierend wirken können
  - andere charakteristische Merkmale, die in den übrigen Feldern nicht passend erfasst werden

- NICHT aufnehmen:
  - blosse Gerichtsnamen oder Urteilsnummern
  - allgemeine Behördenbezeichnungen
  - Namen von Richtern, Anwälten, Gerichtsschreibern, Gutachtern oder Amtspersonen
  - Informationen, die bereits passend in anderen Feldern erfasst wurden
  - rein generische oder offenkundig nicht identifizierende Angaben
  - Adressen, die nichts mit den Parteien zu tun haben, wie Adressen von Gerichten oder öffentlichen Institutionen

- Wenn ein Eintrag hier aufgenommen wird, gib möglichst kurze, konkrete Textausschnitte als Beispiele an.
- Verwende dieses Feld zurückhaltend, aber nicht zu eng: Es soll echte Restfälle auffangen.

L. Gehe in dieser Reihenfolge vor
1. Identifiziere nur die eigentlichen Fallparteien
2. Trenne diese strikt von professionellen Dritten und öffentlichen Institutionen
3. Ordne Parteinamen nach privater Person vs. Firma/Verein/Organisation
4. Ordne Gemeinden nur den municipality-Feldern zu
5. Prüfe danach, ob Angaben echt oder anonymisiert sind
6. Bestimme erst danach die Hauptkategorie

Falls etwas nicht vorhanden ist:
- "value": false
- "examples": []

Hier ist der Urteilstext:

<<<TEXT>>>
{judgment_text}
<<<END>>>
""".strip()


# =========================
# I/O HELPERS
# =========================

def truncate_for_excel(value: Any, limit: int = EXCEL_CELL_LIMIT) -> Any:
    if value is None:
        return value
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - len(EXCEL_TRUNCATION_SUFFIX)] + EXCEL_TRUNCATION_SUFFIX
    return value


def truncate_dataframe_for_excel(df: pd.DataFrame, limit: int = EXCEL_CELL_LIMIT) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: truncate_for_excel(x, limit))
    return df


def join_examples_or_dash(values: list[str]) -> str:
    cleaned = [str(x).strip() for x in values if str(x).strip()]
    return " | ".join(cleaned) if cleaned else EMPTY_EXAMPLES_PLACEHOLDER


def flatten_analysis_json(parsed: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(value, dict) and "value" in value and "examples" in value:
            flat[f"{key}_examples"] = join_examples_or_dash(value.get("examples", []))
            if "reasoning" in value:
                flat[f"{key}_reasoning"] = value.get("reasoning", "")
        else:
            flat[key] = value if value not in ("", None) else EMPTY_EXAMPLES_PLACEHOLDER
    return flat


def extract_pdf_text(pdf_path: str | Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                parts.append(page_text)
    return "\n".join(parts).strip()


# =========================
# OUTPUT VALIDATION
# =========================

SECTION_FIELDS = [
    "private_person_party_full_names_present",
    "private_person_party_anonymised_names_present",
    "company_association_party_full_names_present",
    "company_association_party_anonymised_names_present",
    "third_party_professional_names",
    "addresses_present",
    "addresses_anonymised",
    "parcel_numbers_present",
    "parcel_numbers_anonymised",
    "municipality_present",
    "municipality_anonymised",
    "location_descriptions_present",
    "other_potentially_identifying_information",
]

DATE_SECTION_FIELDS: list[str] = []


def validate_examples(parsed: AnalysisResult) -> None:
    data = parsed.model_dump()
    missing_examples: list[str] = []

    for field in SECTION_FIELDS:
        section = data.get(field, {})
        if section.get("value") is True:
            examples = [str(x).strip() for x in section.get("examples", []) if str(x).strip()]
            if not examples:
                missing_examples.append(field)

    for field in DATE_SECTION_FIELDS:
        section = data.get(field, {})
        if section.get("value") is True:
            examples = [str(x).strip() for x in section.get("examples", []) if str(x).strip()]
            if not examples:
                missing_examples.append(field)

    if missing_examples:
        raise ValueError(
            "Modell lieferte true ohne Beispiele für: " + ", ".join(missing_examples)
        )


# =========================
# ANALYZER
# =========================

class JudgmentAnonymizationAnalyzer:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = MODEL_NAME,
        max_retries: int = MAX_RETRIES,
        sleep_between_requests: float = SLEEP_BETWEEN_REQUESTS,
    ) -> None:
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY ist nicht gesetzt.")
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.sleep_between_requests = sleep_between_requests

    def analyze(self, text: str) -> AnalysisResult:
        prompt = USER_PROMPT_TEMPLATE.format(judgment_text=text)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=AnalysisResult,
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise ValueError("Modell lieferte kein parsbares Ergebnis.")

                validate_examples(parsed)
                return parsed

            except Exception as e:
                last_error = e
                error_text = str(e)
                if "insufficient_quota" in error_text or "exceeded your current quota" in error_text:
                    raise RuntimeError(
                        "OpenAI API quota exhausted. Check billing and usage limits."
                    ) from e
                wait_time = min(2 ** attempt, 20)
                print(f"[WARN] Versuch {attempt}/{self.max_retries} fehlgeschlagen: {e}")
                if attempt < self.max_retries:
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Analyse nach {self.max_retries} Versuchen fehlgeschlagen: {last_error}"
        )

    def analyze_to_row(self, text: str) -> dict[str, Any]:
        try:
            result = self.analyze(text)
        except Exception as e:
            error_msg = str(e)

            if "Modell lieferte true ohne Beispiele" in error_msg:
                return {
                    "category": "",
                    "raw_json": "",
                    "analysis_status": "validation_failed",
                    "analysis_error": error_msg,
                    "needs_review": 1,
                    "geographic_search_prompt": EMPTY_EXAMPLES_PLACEHOLDER,
                }

            return {
                "category": "",
                "raw_json": "",
                "analysis_status": "error",
                "analysis_error": error_msg,
                "needs_review": 1,
                "geographic_search_prompt": EMPTY_EXAMPLES_PLACEHOLDER,
            }

        parsed_dict = result.model_dump()
        flat = flatten_analysis_json(parsed_dict)

        flat["geographic_search_prompt"] = EMPTY_EXAMPLES_PLACEHOLDER
        flat["raw_json"] = truncate_for_excel(json.dumps(parsed_dict, ensure_ascii=False))
        flat["analysis_status"] = "ok"
        flat["analysis_error"] = ""
        flat["needs_review"] = 0

        return flat

    def analyze_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        text = extract_pdf_text(pdf_path)
        if not text:
            return {
                "category": "",
                "raw_json": "",
                "analysis_status": "error",
                "analysis_error": f"Konnte keinen Text aus {pdf_path} extrahieren.",
                "needs_review": 1,
            }
        return self.analyze_to_row(text)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        save_path: Optional[str] = None,
        save_every: int = 5,
        row_limit: Optional[int] = None,
        skip_already_processed: bool = True,
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise KeyError(
                f"Spalte '{text_column}' nicht gefunden. Vorhandene: {list(df.columns)}"
            )

        df = df.copy()
        for col in ("category", "raw_json", "analysis_status", "analysis_error", "needs_review"):
            if col not in df.columns:
                df[col] = None

        indices = []

        for idx, row in df.iterrows():
            text_ok = isinstance(row.get(text_column), str) and row.get(text_column).strip()
            if not text_ok:
                continue

            if skip_already_processed:
                status = str(row.get("analysis_status", "")).strip()
                if status == "ok":
                    continue

            indices.append(idx)

        if row_limit is not None:
            indices = indices[:row_limit]

        print(f"[INFO] Verarbeite {len(indices)} Zeilen.")
        processed_since_save = 0

        for idx in indices:
            text = str(df.at[idx, text_column]).strip()
            row_label = f"Zeile {idx + 2}"
            print(f"[INFO] Analysiere {row_label}")

            result = self.analyze_to_row(text)
            for key, value in result.items():
                if key not in df.columns:
                    df[key] = None
                df.at[idx, key] = value

            processed_since_save += 1
            time.sleep(self.sleep_between_requests)

            if save_path and processed_since_save >= save_every:
                truncate_dataframe_for_excel(df).to_excel(save_path, index=False)
                print(f"[INFO] Zwischengespeichert: {save_path}")
                processed_since_save = 0

        if save_path:
            truncate_dataframe_for_excel(df).to_excel(save_path, index=False)
            print(f"[DONE] Datei gespeichert: {save_path}")

        return df

    def analyze_pdfs(
        self,
        pdf_paths: Iterable[str | Path],
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            print(f"[INFO] Analysiere {pdf_path.name}")
            text = extract_pdf_text(pdf_path)
            row: dict[str, Any] = {
                "source_pdf": pdf_path.name,
                "text_excerpt": truncate_for_excel(text[:2000]),
                "text_length": len(text),
            }
            if not text:
                row.update({
                    "category": "",
                    "raw_json": "",
                    "analysis_status": "error",
                    "analysis_error": "Konnte keinen Text aus der PDF extrahieren.",
                    "needs_review": 1,
                })
            else:
                row.update(self.analyze_to_row(text))
            rows.append(row)
            time.sleep(self.sleep_between_requests)

        df = pd.DataFrame(rows)
        if save_path:
            truncate_dataframe_for_excel(df).to_excel(save_path, index=False)
            print(f"[DONE] Datei gespeichert: {save_path}")
        return df


# =========================
# CLI ENTRY
# =========================

if __name__ == "__main__":
    INPUT_PATH = "/Users/LZN/Desktop/LIVIA/260218_China/hsg_sem3_MA/97_Python/Python_Analysing_Judgements/260414_judgments_linked_addresses_joined_for_GPT_BAC_ZH_text_replaced_input.xlsx"
    OUTPUT_PATH = "/Users/LZN/Desktop/LIVIA/260218_China/hsg_sem3_MA/97_Python/Python_Analysing_Judgements/260415_judgments_linked_addresses_joined_for_GPT_BAC_ZH_text_replaced_output.xlsx"

    df = pd.read_excel(INPUT_PATH)

    analyzer = JudgmentAnonymizationAnalyzer()

    df = analyzer.analyze_dataframe(
        df,
        text_column="text",
        save_path=OUTPUT_PATH,
        row_limit=None,
        skip_already_processed=True,
    )
