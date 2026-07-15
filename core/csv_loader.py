"""
APTIVA AI — CSV / Excel Data Ingestion Engine
=============================================
Sprint 3B: Load CSV and Excel candidate files and normalize them
into the internal Candidate schema used by the ranking engine.

Pipeline:
    raw file bytes
        -> CSVLoader / ExcelLoader   (parse into DataFrame)
        -> CandidateNormalizer       (map columns -> internal schema)
        -> List[Dict]                (same shape as Demo Dataset)

No AI, no resume parsing, no Gemini.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Column alias mapping — single source of truth
# ---------------------------------------------------------------------------
# Each entry maps an INTERNAL field name -> ordered list of accepted headers.
# Resolution is case-insensitive and strips surrounding whitespace.
# First matching alias wins.

COLUMN_ALIASES: Dict[str, List[str]] = {
    # ── Profile ─────────────────────────────────────────────────────────────
    "candidate_id": [
        "candidate_id", "id", "candidate id", "cand_id", "cand id",
        "applicant_id", "applicant id",
    ],
    "name": [
        "anonymized_name", "name", "full name", "full_name",
        "candidate name", "candidate_name",
    ],
    "headline": [
        "headline", "professional headline", "tagline",
    ],
    "summary": [
        "summary", "bio", "about", "profile summary", "profile_summary",
        "professional summary", "description", "about me",
    ],
    "location": [
        "location", "city", "city/region", "city, state",
        "current location", "current_location",
    ],
    "country": [
        "country", "country code", "country_code", "nation",
    ],
    "years_of_experience": [
        "years_of_experience", "yoe", "experience", "years", "exp",
        "years of experience", "total experience", "total_experience",
        "work experience", "work_experience", "experience (years)",
        "experience_years", "years_exp",
    ],
    "current_title": [
        "current_title", "current title", "current role", "role",
        "job title", "job_title", "position", "title", "designation",
    ],
    "current_company": [
        "current_company", "current company", "company", "employer",
        "organization", "organisation", "current employer",
    ],
    "current_company_size": [
        "current_company_size", "company size", "company_size",
        "org size", "headcount",
    ],
    "current_industry": [
        "current_industry", "industry", "sector", "vertical",
        "current industry",
    ],
    # ── Skills ───────────────────────────────────────────────────────────────
    "skills": [
        "skills", "technical skills", "technical_skills",
        "skill set", "skill_set", "key skills", "key_skills",
        "technologies", "tech stack", "tech_stack",
        "tools", "programming languages",
    ],
    # ── Education ────────────────────────────────────────────────────────────
    "degree": [
        "degree", "qualification", "highest qualification",
        "highest_qualification", "education level",
    ],
    "institution": [
        "institution", "university", "college", "school",
        "educational institution",
    ],
    "field_of_study": [
        "field_of_study", "field of study", "major", "specialization",
        "course", "branch",
    ],
    # ── Signals ──────────────────────────────────────────────────────────────
    "open_to_work_flag": [
        "open_to_work", "open to work", "available",
        "actively looking", "job seeking", "open_to_work_flag",
    ],
    "notice_period_days": [
        "notice_period_days", "notice period", "notice_period",
        "notice (days)", "notice_days",
    ],
    "willing_to_relocate": [
        "willing_to_relocate", "relocation", "can relocate",
        "willing to relocate", "open to relocation",
    ],
    "recruiter_response_rate": [
        "recruiter_response_rate", "response_rate", "response rate",
    ],
    "email": [
        "email", "email address", "email_address", "e-mail",
    ],
}


# ---------------------------------------------------------------------------
# Ingestion Report
# ---------------------------------------------------------------------------

@dataclass
class IngestionReport:
    """Summary produced by each ingestion run."""
    total_rows: int = 0
    candidates_loaded: int = 0
    rows_skipped: int = 0
    successful_imports: int = 0
    validation_errors: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.validation_errors)

    def add_error(self, msg: str) -> None:
        """Append an error message, capped at 20 to avoid noise."""
        if len(self.validation_errors) < 20:
            self.validation_errors.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (safe to store on HiringProject)."""
        return {
            "total_rows":        self.total_rows,
            "candidates_loaded": self.candidates_loaded,
            "rows_skipped":      self.rows_skipped,
            "successful_imports":self.successful_imports,
            "validation_errors": list(self.validation_errors),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_col(df, aliases: List[str]) -> Optional[str]:
    """
    Return the first column in *df* that matches any alias
    (case-insensitive, stripped). Returns None if no match.
    """
    lower_map = {c.strip().lower(): c for c in df.columns}
    for alias in aliases:
        match = lower_map.get(alias.strip().lower())
        if match is not None:
            return match
    return None


def _parse_bool(val: Any) -> bool:
    """Coerce flexible truthy values to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in ("yes", "true", "1", "y", "t", "x",
                                         "open", "available")


def _parse_float(val: Any, default: float = 0.0) -> Tuple[float, bool]:
    """
    Return (float_value, parsed_ok).
    parsed_ok=False means the value was non-numeric; default was used.

    NaN arising from a genuinely empty cell (None or pure float NaN passed
    through from pandas when the cell was blank) returns (default, True)
    silently — no warning needed for missing data.

    Any other non-numeric value (including 'N/A' strings that pandas
    converts to NaN) returns (default, False) so the caller can emit a
    validation warning.
    """
    if val is None:
        return default, True
    # Genuinely blank cell — pandas produces float NaN for cells that were
    # empty in the CSV.  We check for NaN *only* when the original dtype
    # would have been numeric; since we read with dtype=str, a blank cell
    # becomes the string 'nan' or pandas keeps it as float NaN depending on
    # the column. We treat pure float NaN as a missing value (silent).
    if isinstance(val, float) and val != val:
        return default, True   # true blank — silent default
    s = str(val).strip()
    if not s or s.lower() in ("", "nan", "none"):
        return default, True   # empty string / None / 'nan' string — silent
    try:
        return float(s.replace(",", "")), True
    except (ValueError, AttributeError):
        return default, False   # bad string such as 'N/A', 'TBD', etc.


def _parse_skills(val: Any) -> List[Dict[str, Any]]:
    """
    Convert a comma-separated skills string (or list) into skill dicts.
    Each skill: name, proficiency='intermediate', endorsements=0, duration_months=0.
    """
    if not val or (isinstance(val, float) and val != val):
        return []
    if isinstance(val, (list, tuple)):
        names = [str(s).strip() for s in val if str(s).strip()]
    else:
        names = [s.strip() for s in str(val).split(",") if s.strip()]
    return [
        {"name": n, "proficiency": "intermediate",
         "endorsements": 0, "duration_months": 0}
        for n in names if n
    ]


def _row_hash(row_dict: Dict[str, Any]) -> str:
    """Content hash for duplicate row detection."""
    key = str(sorted(str(v) for v in row_dict.values()))
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


def _gen_candidate_id(index: int) -> str:
    """Synthetic CAND_ id for rows that lack an id column."""
    return f"CSV_{index:07d}"


def _default_signals() -> Dict[str, Any]:
    """Safe default redrob_signals — covers every required schema field."""
    return {
        "profile_completeness_score":  50,
        "signup_date":                 "2020-01-01",
        "last_active_date":            "2024-01-01",
        "open_to_work_flag":           False,
        "profile_views_received_30d":  0,
        "applications_submitted_30d":  0,
        "recruiter_response_rate":     0.5,
        "avg_response_time_hours":     24,
        "skill_assessment_scores":     {},
        "connection_count":            0,
        "endorsements_received":       0,
        "notice_period_days":          30,
        "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
        "preferred_work_mode":         "hybrid",
        "willing_to_relocate":         False,
        "github_activity_score":       -1,
        "search_appearance_30d":       0,
        "saved_by_recruiters_30d":     0,
        "interview_completion_rate":   0.5,
        "offer_acceptance_rate":       -1,
        "verified_email":              False,
        "verified_phone":              False,
        "linkedin_connected":          False,
    }


# ---------------------------------------------------------------------------
# Candidate Normalizer
# ---------------------------------------------------------------------------

class CandidateNormalizer:
    """
    Converts a raw pandas DataFrame into the internal ranking engine schema.

    One normalizer shared by both CSVLoader and ExcelLoader — no duplicate
    mapping logic.

    Output shape per candidate:
        {
          "candidate_id": str,
          "profile":      {anonymized_name, headline, summary, location,
                           country, years_of_experience, current_title,
                           current_company, current_company_size,
                           current_industry},
          "career_history": [...],
          "education":      [...],
          "skills":         [{name, proficiency, endorsements, duration_months}],
          "certifications": [],
          "languages":      [],
          "redrob_signals": {...},
        }
    """

    # File is usable if at least one of these resolves
    _REQUIRED_ANY = ["candidate_id", "name", "current_title", "years_of_experience"]

    _VALID_SIZES = {
        "1-10", "11-50", "51-200", "201-500",
        "501-1000", "1001-5000", "5001-10000", "10001+",
    }

    def normalize(
        self,
        df,
        source_filename: str = "upload",
    ) -> Tuple[List[Dict[str, Any]], IngestionReport]:
        """
        Normalise *df* into a list of candidate dicts.

        Returns:
            (candidates, report)
        """
        import pandas as pd  # lazy import — keeps module importable without pandas

        report = IngestionReport(total_rows=len(df))

        # ── Minimum viability ────────────────────────────────────────────────
        has_useful = any(
            _find_col(df, COLUMN_ALIASES[k]) is not None
            for k in self._REQUIRED_ANY
            if k in COLUMN_ALIASES
        )
        if not has_useful:
            report.add_error(
                "File is missing all required columns. "
                "Expected at least one of: "
                + ", ".join(self._REQUIRED_ANY)
                + "."
            )
            return [], report

        # ── Resolve column names once ────────────────────────────────────────
        col: Dict[str, Optional[str]] = {
            k: _find_col(df, aliases)
            for k, aliases in COLUMN_ALIASES.items()
        }

        # ── Per-row processing ───────────────────────────────────────────────
        seen_ids:     set = set()
        seen_hashes:  set = set()
        candidates:   List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            row_label = f"Row {idx + 2}"   # 1-indexed + skip header

            # ── candidate_id ─────────────────────────────────────────────────
            cid_col = col.get("candidate_id")
            if cid_col and pd.notna(row.get(cid_col, None)):
                cid = str(row[cid_col]).strip()
            else:
                cid = _gen_candidate_id(idx)

            # ── Duplicate check by id ────────────────────────────────────────
            if cid in seen_ids:
                report.rows_skipped += 1
                report.add_error(
                    f"{row_label}: Duplicate candidate_id '{cid}' — skipped."
                )
                continue

            # ── Duplicate check by content ───────────────────────────────────
            row_h = _row_hash(row.to_dict())
            if row_h in seen_hashes:
                report.rows_skipped += 1
                report.add_error(f"{row_label}: Duplicate row content — skipped.")
                continue

            seen_ids.add(cid)
            seen_hashes.add(row_h)

            # ── years_of_experience ──────────────────────────────────────────
            yoe_col = col.get("years_of_experience")
            yoe_raw = row.get(yoe_col) if yoe_col else None
            yoe, yoe_ok = _parse_float(yoe_raw, default=0.0)
            if not yoe_ok:
                report.add_error(
                    f"{row_label}: years_of_experience '{yoe_raw}' "
                    "is not numeric — defaulted to 0."
                )
            yoe = max(0.0, min(yoe, 50.0))   # clamp to schema range

            # ── Skills ───────────────────────────────────────────────────────
            skills_col = col.get("skills")
            skills_raw = row.get(skills_col) if skills_col else None
            skills     = _parse_skills(skills_raw)

            # ── String helper ────────────────────────────────────────────────
            def _str(key: str, default: str = "") -> str:
                c = col.get(key)
                if c is None:
                    return default
                v = row.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    return default
                return str(v).strip()

            name          = _str("name",             f"Candidate {cid}")
            headline      = _str("headline",         _str("current_title", ""))
            summary       = _str("summary",          "")
            location      = _str("location",         "")
            country       = _str("country",          "")
            current_title = _str("current_title",    "")
            current_co    = _str("current_company",  "")
            industry      = _str("current_industry", "")
            co_size_raw   = _str("current_company_size", "")
            co_size       = co_size_raw if co_size_raw in self._VALID_SIZES else "51-200"

            # ── Education ────────────────────────────────────────────────────
            edu_degree = _str("degree",        "")
            edu_inst   = _str("institution",   "")
            edu_field  = _str("field_of_study","")
            education: List[Dict] = []
            if edu_degree or edu_inst:
                education.append({
                    "institution":    edu_inst or "Unknown",
                    "degree":         edu_degree or "",
                    "field_of_study": edu_field,
                    "start_year":     2010,
                    "end_year":       2014,
                    "grade":          None,
                    "tier":           "unknown",
                })

            # ── Career history (synthesised from current role) ───────────────
            career_history: List[Dict] = []
            if current_title or current_co:
                career_history.append({
                    "company":         current_co or "Unknown",
                    "title":           current_title or "Professional",
                    "start_date":      "2020-01-01",
                    "end_date":        None,
                    "duration_months": max(1, int(yoe * 12)),
                    "is_current":      True,
                    "industry":        industry or "Technology",
                    "company_size":    co_size,
                    "description":     summary or f"{current_title} at {current_co}.",
                })

            # ── Signals ──────────────────────────────────────────────────────
            signals = _default_signals()

            otw_col = col.get("open_to_work_flag")
            if otw_col:
                signals["open_to_work_flag"] = _parse_bool(row.get(otw_col))

            np_col = col.get("notice_period_days")
            if np_col:
                np_val, _ = _parse_float(row.get(np_col), default=30.0)
                signals["notice_period_days"] = int(max(0, min(np_val, 180)))

            rel_col = col.get("willing_to_relocate")
            if rel_col:
                signals["willing_to_relocate"] = _parse_bool(row.get(rel_col))

            rr_col = col.get("recruiter_response_rate")
            if rr_col:
                rr_val, _ = _parse_float(row.get(rr_col), default=0.5)
                signals["recruiter_response_rate"] = max(0.0, min(1.0, rr_val))

            # ── Assemble ─────────────────────────────────────────────────────
            candidates.append({
                "candidate_id": cid,
                "profile": {
                    "anonymized_name":      name,
                    "headline":             headline,
                    "summary":              summary,
                    "location":             location,
                    "country":              country,
                    "years_of_experience":  yoe,
                    "current_title":        current_title,
                    "current_company":      current_co,
                    "current_company_size": co_size,
                    "current_industry":     industry,
                },
                "career_history":  career_history,
                "education":       education,
                "skills":          skills,
                "certifications":  [],
                "languages":       [],
                "redrob_signals":  signals,
            })

        report.candidates_loaded  = len(candidates)
        report.successful_imports = len(candidates)
        return candidates, report


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

class CSVLoader:
    """Load and validate a CSV file from raw bytes."""

    def load(
        self,
        file_bytes: bytes,
        filename: str = "upload.csv",
    ) -> Tuple[List[Dict[str, Any]], IngestionReport]:
        """
        Parse CSV bytes into (candidates, report).

        Validates:
        - File not empty
        - Parseable as CSV
        - At least one useful column present
        - No duplicate rows
        - years_of_experience is numeric
        """
        import pandas as pd

        report = IngestionReport()

        if not file_bytes:
            report.add_error("File is empty.")
            return [], report

        try:
            df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
        except Exception as exc:
            report.add_error(f"Failed to parse CSV: {exc}")
            return [], report

        # Normalise column names
        df.columns = [str(c).strip() for c in df.columns]

        if len(df) == 0:
            report.add_error(
                "CSV file has no data rows (file has a header but is otherwise empty)."
            )
            return [], report

        normalizer = CandidateNormalizer()
        candidates, norm_report = normalizer.normalize(df, source_filename=filename)
        return candidates, norm_report


# ---------------------------------------------------------------------------
# Excel Loader
# ---------------------------------------------------------------------------

class ExcelLoader:
    """Load and validate an Excel file (.xlsx or .xls) from raw bytes."""

    def load(
        self,
        file_bytes: bytes,
        filename: str = "upload.xlsx",
    ) -> Tuple[List[Dict[str, Any]], IngestionReport]:
        """
        Parse Excel bytes into (candidates, report).
        Same validation contract as CSVLoader.
        """
        import pandas as pd

        report = IngestionReport()

        if not file_bytes:
            report.add_error("File is empty.")
            return [], report

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "xlsx"

        try:
            engine = "xlrd" if ext == "xls" else "openpyxl"
            df = pd.read_excel(io.BytesIO(file_bytes), engine=engine, dtype=str, keep_default_na=False)
        except ImportError:
            # Retry without specifying engine (pandas will try installed engines)
            try:
                df = pd.read_excel(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
            except Exception as exc2:
                report.add_error(
                    f"Excel engine not available. Install openpyxl (for .xlsx) "
                    f"or xlrd (for .xls): {exc2}"
                )
                return [], report
        except Exception as exc:
            report.add_error(f"Failed to parse Excel file: {exc}")
            return [], report

        # Normalise column names
        df.columns = [str(c).strip() for c in df.columns]

        if len(df) == 0:
            report.add_error(
                "Excel file has no data rows (file has a header but is otherwise empty)."
            )
            return [], report

        normalizer = CandidateNormalizer()
        candidates, norm_report = normalizer.normalize(df, source_filename=filename)
        return candidates, norm_report
