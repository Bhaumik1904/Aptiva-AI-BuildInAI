"""
APTIVA AI — Resume Intelligence Agent
======================================
Sprint 5A: Convert raw PDF/DOCX resume bytes into the existing internal
Candidate schema used by the ranking engine.

Design constraints (identical to JDIntelligenceAgent):
- Completely stateless — zero Streamlit imports, zero session state.
- Model is configured entirely via config.yaml.
  Switching model only requires a config change, never a code change.
- response_mime_type="application/json" forces JSON-only output.
- 5-layer validation with resilient defaults:
    Layer 1 — JSON decode (retry after stripping markdown fences)
    Layer 2 — Required fields (name, skills)
    Layer 3 — Optional fields → safe defaults
    Layer 4 — Type normalization (clamp years, dedupe skills)
    Layer 5 — Assemble into internal Candidate dict

Extraction metadata (approved improvement):
  Every candidate produced by this agent carries a top-level
  "_extraction_meta" key:
    source        = "resume"
    source_file   = original uploaded filename
    processed_at  = ISO-8601 timestamp
    parser        = "Resume Intelligence Agent"
  This key is invisible to the ranking engine but available for
  reporting, explainability, and debugging.

Candidate IDs (approved improvement):
  Every resume that does not supply its own id receives a unique
  UUID-based id with a "RESUME_" prefix, e.g. "RESUME_3f4a8b2c".
"""

from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set

import google.generativeai as genai
import pypdf
import docx

from core.secrets_utils import resolve_api_key
from core.csv_loader import _parse_skills, _default_signals


# ---------------------------------------------------------------------------
# Gemini prompt
# ---------------------------------------------------------------------------

def _build_prompt(resume_text: str) -> str:
    """
    Build the Gemini extraction prompt for a single resume text.
    Uses concatenated strings (not .format()) to avoid escaping curly braces.
    """
    return (
        "You are a Resume extraction engine for an AI recruitment platform.\n\n"
        "INSTRUCTIONS\n"
        "- Carefully read the Resume text provided below.\n"
        "- Output a single valid JSON object matching the schema exactly.\n"
        "- Do NOT output markdown, code fences, backticks, comments, or any text "
        "other than the JSON object.\n"
        "- If a field cannot be determined from the text, use the safe default shown.\n\n"
        "OUTPUT SCHEMA (all keys must be present)\n"
        "{\n"
        '  "name": "<full name of the candidate, REQUIRED>",\n'
        '  "email": "<email address, default empty string>",\n'
        '  "phone": "<phone number, default empty string>",\n'
        '  "current_role": "<most recent job title, default empty string>",\n'
        '  "current_company": "<most recent employer, default empty string>",\n'
        '  "years_of_experience": <total years of professional experience, number 0-50, REQUIRED>,\n'
        '  "skills": ["<technical or professional skill>"],\n'
        '  "education": [{"institution": "<name>", "degree": "<degree>", '
        '"field_of_study": "<field>", "end_year": <year or null>}],\n'
        '  "certifications": ["<certification name>"],\n'
        '  "projects": ["<key project or achievement, 1 sentence>"],\n'
        '  "summary": "<2-3 sentence professional summary, default empty string>",\n'
        '  "location": "<city, region, or country, default empty string>",\n'
        '  "industry": "<primary industry the candidate works in, default empty string>",\n'
        '  "keywords": ["<key phrase useful for candidate-JD matching>"]\n'
        "}\n\n"
        "FIELD RULES\n"
        "- name: REQUIRED. Extract the full name as stated.\n"
        "- years_of_experience: REQUIRED. Infer from career history if not explicit. "
        "Integer or float, 0-50.\n"
        "- skills: 3 to 25 items. Technical and professional skills only. "
        "Include programming languages, frameworks, tools, methodologies. "
        "Do NOT include soft skills like 'communication' or 'teamwork'.\n"
        "- education: list of education entries. Use empty list [] if not found.\n"
        "- certifications: list of certification names. Use [] if none.\n"
        "- projects: up to 5 key projects or achievements. Use [] if none.\n"
        "- keywords: 5-15 phrases most useful for matching this candidate to job descriptions.\n"
        "- Do NOT invent information not present in the resume.\n"
        "- All list values must be plain strings with no nested objects "
        "(except education which is a list of objects).\n\n"
        "RESUME TEXT\n"
        f"{resume_text}\n"
    )


# ---------------------------------------------------------------------------
# Resume Intelligence Agent
# ---------------------------------------------------------------------------

class ResumeIntelligenceAgent:
    """
    Stateless agent that extracts a structured Candidate dict from a raw
    PDF or DOCX resume.

    Model resolution (identical to JDIntelligenceAgent, 3-tier):
      1. resume_agent.model   (config.yaml — change to switch models)
      2. gemini_model         (global fallback)
      3. "gemini-2.5-flash"   (hard-coded fallback)

    Usage
    -----
        agent = ResumeIntelligenceAgent(config)
        candidate, steps = agent.analyze(file_bytes, "john_doe.pdf")

    Batch usage
    -----------
        results = agent.analyze_batch([(bytes1, "a.pdf"), (bytes2, "b.docx")])
        # results: List[{"candidate": dict, "steps": list, "error": str|None, "filename": str}]
    """

    _SUPPORTED_EXTENSIONS = frozenset({"pdf", "docx", "doc"})

    def __init__(self, config: dict):
        self._config     = config
        agent_cfg        = config.get("resume_agent", {})

        # Model resolution — fully config-driven
        self._model_name = (
            agent_cfg.get("model")
            or config.get("gemini_model")
            or "gemini-2.5-flash"
        )
        self._temperature  = float(agent_cfg.get("temperature", 0.1))
        self._max_tokens   = int(agent_cfg.get("max_output_tokens", 2048))

        # API key — Streamlit Secrets > Env Var > Config
        self._api_key = resolve_api_key(config, "gemini_api_key", "GEMINI_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if a Gemini API key is available."""
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def analyze(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract a structured Candidate dict from a single resume file.

        Parameters
        ----------
        file_bytes : bytes
            Raw file content (PDF or DOCX).
        filename : str
            Original filename — used for metadata and extension detection.

        Returns
        -------
        (candidate_dict, steps_log)
            candidate_dict : dict
                Internal schema dict ready for the ranking engine.
                Includes "_extraction_meta" key (invisible to scorer).
            steps_log : List[str]
                Human-readable completed-step strings for the UI summary.

        Raises
        ------
        ValueError
            Empty bytes, unsupported format, empty text (scanned PDF),
            or unrecoverable Gemini response.
        RuntimeError
            API key not configured, or Gemini SDK/network failure.
        """
        if not file_bytes:
            raise ValueError("File bytes are empty — nothing to parse.")

        if not self.is_configured():
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set 'gemini_api_key' in config.yaml or export GEMINI_API_KEY."
            )

        ext = _get_ext(filename)
        if ext not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                "Only PDF (.pdf) and DOCX (.docx) are supported."
            )

        # Extract text
        if ext == "pdf":
            resume_text = extract_text_pdf(file_bytes)
        else:  # docx / doc
            resume_text = extract_text_docx(file_bytes)

        if not resume_text or len(resume_text.strip()) < 30:
            raise ValueError(
                f"Could not extract readable text from '{filename}'. "
                "The file may be a scanned image PDF with no embedded text, "
                "a password-protected file, or corrupt. "
                "Please upload a text-based PDF or DOCX."
            )

        # Call Gemini
        try:
            raw_response = self._call_gemini(resume_text.strip())
        except Exception as exc:
            raise RuntimeError("AI service is temporarily unavailable. Please try again.") from exc

        # Parse with retry
        try:
            validated = self._validate_and_parse(raw_response)
        except json.JSONDecodeError:
            cleaned = _strip_markdown_fences(raw_response)
            try:
                validated = self._validate_and_parse(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned output that could not be parsed as JSON. "
                    "This may be a transient error — please try again."
                ) from exc

        # Assemble internal candidate dict
        candidate = _to_candidate_dict(validated, filename)
        steps_log = _build_steps_log(validated, filename)
        return candidate, steps_log

    def analyze_batch(
        self,
        files: List[Tuple[bytes, str]],
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple resume files.

        Parameters
        ----------
        files : List[(bytes, filename)]

        Returns
        -------
        List of result dicts:
            {
                "filename":  str,
                "candidate": dict | None,   # None if error
                "steps":     List[str],     # empty if error
                "error":     str | None,    # None if success
            }
        """
        results = []
        for file_bytes, filename in files:
            try:
                candidate, steps = self.analyze(file_bytes, filename)
                results.append({
                    "filename":  filename,
                    "candidate": candidate,
                    "steps":     steps,
                    "error":     None,
                })
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "filename":  filename,
                    "candidate": None,
                    "steps":     [],
                    "error":     str(exc),
                })
        return results

    # ------------------------------------------------------------------
    # Private: Gemini call
    # ------------------------------------------------------------------

    def _call_gemini(self, resume_text: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self._api_key)

        generation_config = {
            "temperature":        self._temperature,
            "max_output_tokens":  self._max_tokens,
            "response_mime_type": "application/json",
        }

        model    = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
        )
        prompt   = _build_prompt(resume_text)
        response = model.generate_content(prompt)
        return response.text

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    def _validate_and_parse(self, raw: str) -> dict:
        """
        5-layer validation with resilient defaults.

        Layer 1 — JSON decode.
        Layer 2 — Required fields: name, years_of_experience, skills.
        Layer 3 — Optional fields → safe defaults.
        Layer 4 — Type normalization (clamp yoe 0-50, dedupe skills).
        Layer 5 — Assemble validated dict ready for _to_candidate_dict().

        Raises json.JSONDecodeError if not valid JSON.
        Raises ValueError if required fields are unrecoverable.
        """
        # Layer 1: JSON decode
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Gemini returned a {type(data).__name__}, expected a JSON object."
            )

        # Layer 2: Required fields
        errors: List[str] = []

        name = str(data.get("name", "")).strip()
        if not name:
            errors.append("'name' is missing or empty")

        yoe = _safe_float(data.get("years_of_experience"))
        if yoe is None:
            errors.append("'years_of_experience' is missing or non-numeric")

        raw_skills = data.get("skills", [])
        if not isinstance(raw_skills, list):
            raw_skills = []
        skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        # Not a hard failure if skills is empty — some PM resumes are skill-light

        if errors:
            raise ValueError(
                "AI response is missing required fields that cannot be recovered: "
                + "; ".join(errors)
            )

        # Layer 3 & 4: Optional fields with safe defaults + type normalization
        yoe_clamped = max(0.0, min(float(yoe), 50.0))

        email   = str(data.get("email",           "")).strip()
        phone   = str(data.get("phone",           "")).strip()
        role    = str(data.get("current_role",    "")).strip()
        company = str(data.get("current_company", "")).strip()
        summary = str(data.get("summary",         "")).strip()
        location= str(data.get("location",        "")).strip()
        industry= str(data.get("industry",        "")).strip()

        certs    = _safe_str_list(data.get("certifications"))
        projects = _safe_str_list(data.get("projects"))
        keywords = _safe_str_list(data.get("keywords"))

        # Deduplicate skills case-insensitively, preserve first occurrence order
        seen_lower: set = set()
        deduped_skills: List[str] = []
        for s in skills:
            sl = s.lower()
            if sl not in seen_lower:
                seen_lower.add(sl)
                deduped_skills.append(s)

        # If keywords missing, derive from skills
        if not keywords:
            keywords = deduped_skills[:10]

        # Education: list of dicts — validate structure
        raw_edu = data.get("education", [])
        education: List[Dict[str, Any]] = []
        if isinstance(raw_edu, list):
            for edu_item in raw_edu:
                if not isinstance(edu_item, dict):
                    continue
                education.append({
                    "institution":    str(edu_item.get("institution", "Unknown")).strip(),
                    "degree":         str(edu_item.get("degree", "")).strip(),
                    "field_of_study": str(edu_item.get("field_of_study", "")).strip(),
                    "start_year":     None,
                    "end_year":       _safe_int_or_none(edu_item.get("end_year")),
                    "grade":          None,
                    "tier":           "unknown",
                })

        return {
            "name":                 name,
            "email":                email,
            "phone":                phone,
            "current_role":         role,
            "current_company":      company,
            "years_of_experience":  yoe_clamped,
            "skills":               deduped_skills[:25],
            "education":            education,
            "certifications":       certs[:10],
            "projects":             projects[:5],
            "summary":              summary,
            "location":             location,
            "industry":             industry,
            "keywords":             keywords[:20],
        }


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from a PDF using pypdf.
    Returns concatenated page text. Empty string if no text found.

    Raises RuntimeError if pypdf is not installed.
    Raises ValueError if bytes cannot be parsed as a PDF.
    """
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed. Run: pip install pypdf"
        ) from exc

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}") from exc

    parts: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
            parts.append(text)
        except Exception:
            continue  # Skip unreadable pages silently

    return "\n".join(parts)


def extract_text_docx(file_bytes: bytes) -> str:
    """
    Extract plain text from a DOCX file using python-docx.
    Returns all paragraph text joined by newlines.

    Raises RuntimeError if python-docx is not installed.
    Raises ValueError if bytes cannot be parsed as a DOCX.
    """
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        ) from exc

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse DOCX: {exc}") from exc

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Candidate dict assembly
# ---------------------------------------------------------------------------

def _to_candidate_dict(
    validated: dict,
    source_filename: str,
) -> Dict[str, Any]:
    """
    Assemble the internal Candidate dict from a validated extraction result.

    Output shape is identical to what CandidateNormalizer produces:
        {
          "candidate_id":     str,
          "profile":          { ... },
          "career_history":   [ ... ],
          "education":        [ ... ],
          "skills":           [ {name, proficiency, endorsements, duration_months} ],
          "certifications":   [ str, ... ],
          "languages":        [],
          "redrob_signals":   { ... },
          "_extraction_meta": { source, source_file, processed_at, parser },
        }

    The "_extraction_meta" key is ignored by the scoring engine (it only
    reads the keys listed in the schema) but is available for reporting.

    Approved improvements implemented here:
      1. UUID-based candidate_id: "RESUME_<8-char uuid>"
      2. _extraction_meta dict with source, source_file, processed_at, parser
    """
    # Improvement 1: Unique candidate_id
    candidate_id = f"RESUME_{uuid.uuid4().hex[:8].upper()}"

    name           = validated["name"]
    yoe            = float(validated["years_of_experience"])
    role           = validated["current_role"]
    company        = validated["current_company"]
    summary        = validated["summary"]
    location       = validated["location"]
    industry       = validated["industry"]
    education      = validated["education"]
    certifications = validated["certifications"]

    # Skills — reuse _parse_skills for consistent shape
    # _parse_skills expects a comma-joined string OR a list; pass the list.
    skill_dicts = _parse_skills(", ".join(validated["skills"]))

    # Career history — synthesised from current role
    career_history: List[Dict[str, Any]] = []
    if role or company:
        career_history.append({
            "company":         company or "Unknown",
            "title":           role or "Professional",
            "start_date":      "2020-01-01",
            "end_date":        None,
            "duration_months": max(1, int(yoe * 12)),
            "is_current":      True,
            "industry":        industry or "Technology",
            "company_size":    "51-200",
            "description":     summary or f"{role} at {company}.",
        })

    signals = _default_signals()

    # Improvement 2: Extraction metadata
    extraction_meta = {
        "source":       "resume",
        "source_file":  source_filename,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "parser":       "Resume Intelligence Agent",
    }

    return {
        "candidate_id":    candidate_id,
        "profile": {
            "anonymized_name":      name,
            "headline":             role,
            "summary":              summary,
            "location":             location,
            "country":              "",
            "years_of_experience":  yoe,
            "current_title":        role,
            "current_company":      company,
            "current_company_size": "51-200",
            "current_industry":     industry,
        },
        "career_history":   career_history,
        "education":        education,
        "skills":           skill_dicts,
        "certifications":   certifications,
        "languages":        [],
        "redrob_signals":   signals,
        # Approved improvement 2 — invisible to ranking engine
        "_extraction_meta": extraction_meta,
    }


# ---------------------------------------------------------------------------
# UI steps log
# ---------------------------------------------------------------------------

def _build_steps_log(validated: dict, filename: str) -> List[str]:
    """Build the ✓ step strings displayed in the UI summary card."""
    steps: List[str] = []

    ext = _get_ext(filename).upper()
    steps.append(f"Reading **{ext}** file: `{filename}`")

    name = validated.get("name", "")
    if name:
        steps.append(f"Candidate identified: **{name}**")

    yoe = validated.get("years_of_experience", 0)
    steps.append(f"**{yoe:.0f} years** of experience extracted")

    n_skills = len(validated.get("skills", []))
    if n_skills:
        steps.append(f"**{n_skills} skills** identified")

    n_certs = len(validated.get("certifications", []))
    if n_certs:
        steps.append(f"**{n_certs} certifications** found")

    n_edu = len(validated.get("education", []))
    if n_edu:
        edu_word = "entry" if n_edu == 1 else "entries"
        steps.append(f"**{n_edu} education** {edu_word} extracted")

    role = validated.get("current_role", "")
    if role:
        steps.append(f"Current role: **{role}**")

    steps.append("Candidate profile **ready for ranking**")
    return steps


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _get_ext(filename: str) -> str:
    """Return lowercased file extension without the dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def _safe_int_or_none(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return None


def _safe_str_list(val: Any) -> List[str]:
    if not isinstance(val, list):
        return []
    return [str(s).strip() for s in val if str(s).strip()]


def _strip_markdown_fences(text: str) -> str:
    """Extract JSON object from text using robust incremental parsing."""
    import json
    
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in ('{', '['):
            try:
                obj, end_idx = decoder.raw_decode(text[i:])
                return text[i:i+end_idx]
            except json.JSONDecodeError:
                continue
                
    raise ValueError("No valid JSON object or array could be found in the AI response.")