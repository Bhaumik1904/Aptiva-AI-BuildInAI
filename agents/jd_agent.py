"""
APTIVA AI — JD Intelligence Agent
==================================
Sprint 4: Transform an unstructured Job Description into the
structured JobDescription model using Gemini.

Design constraints
- No Streamlit imports — this module is stateless.
- No session state access — one call in, one result out.
- Gemini model is configured entirely via config.yaml.
  Switching from gemini-2.5-flash to gemini-2.5-pro (or any
  future model) requires only a config change, never a code change.
- response_mime_type="application/json" forces JSON-only output,
  eliminating markdown-fence wrapping in nearly all cases.
- 4-layer validation with resilient defaults:
    Layer 1 — JSON decode (retry once after stripping markdown)
    Layer 2 — Required fields (title, core_skills, experience range)
    Layer 3 — Optional fields populated with safe defaults
    Layer 4 — Experience ordering clamp
  Only truly unrecoverable failures raise ValueError.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import google.generativeai as genai

from core.models import JobDescription
from core.secrets_utils import resolve_api_key


# ---------------------------------------------------------------------------
# Gemini prompt (module-level constant — model-agnostic)
# ---------------------------------------------------------------------------

def _build_prompt(jd_text: str) -> str:
    """
    Build the full Gemini prompt for a given job description text.

    Uses concatenated strings rather than .format() to avoid escaping
    curly braces in the JSON schema example.
    """
    return (
        "You are a Job Description extraction engine for an AI recruitment platform.\n\n"
        "INSTRUCTIONS\n"
        "- Carefully read the Job Description text provided below.\n"
        "- Output a single valid JSON object matching the schema exactly.\n"
        "- Do NOT output markdown, code fences, backticks, comments, explanations,\n"
        "  or any text other than the JSON object itself.\n"
        "- If a field cannot be determined from the text, use the safe default shown.\n\n"
        "OUTPUT SCHEMA (all keys must be present)\n"
        "{\n"
        '  "title": "<concise job title, REQUIRED>",\n'
        '  "description": "<2-3 sentence role summary, default empty string>",\n'
        '  "responsibilities": "<key responsibilities joined by newline, default empty string>",\n'
        '  "core_skills": ["<must-have technical skill>"],\n'
        '  "bonus_skills": ["<nice-to-have or implied skill>"],\n'
        '  "experience_target_min": <minimum acceptable years, integer 0-30, REQUIRED>,\n'
        '  "experience_target_max": <maximum acceptable years, integer 0-30, REQUIRED>,\n'
        '  "experience_sweet_spot_min": <ideal lower bound, integer 0-30>,\n'
        '  "experience_sweet_spot_max": <ideal upper bound, integer 0-30>,\n'
        '  "preferred_locations": ["<city or region, only if explicitly mentioned>"],\n'
        '  "preferred_industries": ["<industry name>"],\n'
        '  "jd_career_keywords": ["<key phrase for candidate-JD text matching>"],\n'
        '  "risk_keywords": [],\n'
        '  "title_scores": {}\n'
        "}\n\n"
        "FIELD RULES\n"
        "- title: required. Extract the exact job title stated, or infer the best match.\n"
        "- core_skills: 3 to 12 items. Must-have technical skills only.\n"
        "- bonus_skills: 0 to 8 items. Nice-to-have or strongly implied skills.\n"
        "- experience ordering must hold: "
        "experience_target_min <= experience_sweet_spot_min <= "
        "experience_sweet_spot_max <= experience_target_max.\n"
        "- jd_career_keywords: 5 to 15 phrases most useful for matching candidates "
        "to this role. Include tech stack, domain, methodologies, tools.\n"
        "- preferred_locations: only include if explicitly stated. Default to empty list.\n"
        "- risk_keywords: default to empty list unless the JD implies disqualifying traits.\n"
        "- title_scores: default to empty object.\n"
        "- Do NOT invent qualifications not mentioned or strongly implied by the text.\n"
        "- All list values must be plain strings with no nested objects.\n\n"
        "JOB DESCRIPTION TEXT\n"
        f"{jd_text}\n"
    )


# ---------------------------------------------------------------------------
# JD Intelligence Agent
# ---------------------------------------------------------------------------

class JDIntelligenceAgent:
    """
    Stateless agent that extracts a structured JobDescription from raw text.

    Model is resolved (in priority order) from:
      1. config["jd_agent"]["model"]    (Sprint 4 agent section)
      2. config["gemini_model"]          (global Gemini setting)
      3. Hard-coded fallback: "gemini-2.5-flash"

    This means switching to gemini-2.5-pro only requires changing config.yaml:

      jd_agent:
        model: "gemini-2.5-pro"

    No code changes required.

    Usage
    -----
        agent = JDIntelligenceAgent(config)
        jd, steps = agent.analyze(jd_text)
    """

    # Required keys — if missing/invalid AND unrecoverable, raise ValueError
    _REQUIRED_KEYS: frozenset = frozenset({
        "title",
        "core_skills",
        "experience_target_min",
        "experience_target_max",
    })

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Application config dict (from config.yaml loaded via load_config()).
            API key resolved from config["gemini_api_key"] or GEMINI_API_KEY env var.
        """
        self._config   = config
        agent_cfg      = config.get("jd_agent", {})

        # Model resolution — fully config-driven
        self._model_name = (
            agent_cfg.get("model")
            or config.get("gemini_model")
            or "gemini-2.5-flash"
        )
        self._temperature   = float(agent_cfg.get("temperature", 0.1))
        self._max_tokens    = int(agent_cfg.get("max_output_tokens", 2048))

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
        """The Gemini model this agent will use."""
        return self._model_name

    def analyze(self, jd_text: str) -> Tuple[JobDescription, List[str]]:
        """
        Extract a structured JobDescription from raw job description text.

        Parameters
        ----------
        jd_text : str
            Raw, unstructured job description in any format.

        Returns
        -------
        (job_description, steps_log)
            job_description : JobDescription
                Fully populated instance ready for the ranking engine.
            steps_log : List[str]
                Human-readable completed-step strings for the UI summary card.

        Raises
        ------
        ValueError
            jd_text is empty, or Gemini returns unrecoverable output.
        RuntimeError
            API key not configured, or Gemini network/SDK failure.
        """
        if not jd_text or not jd_text.strip():
            raise ValueError("Job Description text cannot be empty.")

        if not self.is_configured():
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set 'gemini_api_key' in config.yaml or export GEMINI_API_KEY."
            )

        # Layer 0: Call Gemini
        try:
            raw_response = self._call_gemini(jd_text.strip())
        except Exception as exc:
            raise RuntimeError("AI service is temporarily unavailable. Please try again.") from exc

        # Layer 1: JSON decode with one retry after stripping markdown fences
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

        # Construct JobDescription from validated dict
        jd = JobDescription.from_dict(validated)
        # Fields not in legacy JD_CONFIG dict — set explicitly
        jd.title            = validated["title"]
        jd.description      = validated.get("description", "")
        jd.responsibilities = validated.get("responsibilities", "")

        steps_log = _build_steps_log(validated)
        return jd, steps_log

    # ------------------------------------------------------------------
    # Private: Gemini call
    # ------------------------------------------------------------------

    def _call_gemini(self, jd_text: str) -> str:
        """
        Call the Gemini API with the structured prompt.
        Uses response_mime_type='application/json' to force JSON-only output.
        Model name comes from self._model_name — fully config-driven.
        """
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self._api_key)

        generation_config = {
            "temperature":         self._temperature,
            "max_output_tokens":   self._max_tokens,
            "response_mime_type":  "application/json",
        }

        model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
        )

        prompt   = _build_prompt(jd_text)
        response = model.generate_content(prompt)
        return response.text

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    def _validate_and_parse(self, raw: str) -> dict:
        """
        4-layer validation with resilient defaults.

        Layer 1 — JSON decode (raises json.JSONDecodeError on failure).
        Layer 2 — Required field presence and correct types.
        Layer 3 — Optional fields filled with safe defaults when missing.
        Layer 4 — Experience ordering enforced (clamp, no ValueError).

        Returns a clean dict compatible with JobDescription.from_dict().
        Raises ValueError only when required fields are unrecoverable.
        """
        # Layer 1: JSON decode
        data = json.loads(raw)   # raises json.JSONDecodeError if invalid

        if not isinstance(data, dict):
            raise ValueError(
                f"Gemini returned a {type(data).__name__}, expected a JSON object."
            )

        # Layer 2: Required fields
        errors: List[str] = []

        # title
        title = str(data.get("title", "")).strip()
        if not title:
            errors.append("'title' is missing or empty")

        # core_skills
        raw_core = data.get("core_skills", [])
        if not isinstance(raw_core, list):
            raw_core = []
        core_skills = [str(s).strip() for s in raw_core if str(s).strip()]
        if not core_skills:
            errors.append("'core_skills' is missing or contains no valid entries")

        # experience range
        exp_min = _safe_int(data.get("experience_target_min"))
        exp_max = _safe_int(data.get("experience_target_max"))
        if exp_min is None:
            errors.append("'experience_target_min' is missing or non-numeric")
        if exp_max is None:
            errors.append("'experience_target_max' is missing or non-numeric")

        if errors:
            raise ValueError(
                "AI response is missing required fields that cannot be recovered: "
                + "; ".join(errors)
            )

        # Layer 3: Optional fields with safe defaults
        exp_min = max(0, min(int(exp_min), 30))
        exp_max = max(exp_min, min(int(exp_max), 30))

        # Sweet spot — derive intelligently if missing
        raw_sweet_min = _safe_int(data.get("experience_sweet_spot_min"))
        raw_sweet_max = _safe_int(data.get("experience_sweet_spot_max"))
        span = max(1, exp_max - exp_min)

        sweet_min = int(raw_sweet_min) if raw_sweet_min is not None else exp_min + max(1, span // 3)
        sweet_max = int(raw_sweet_max) if raw_sweet_max is not None else exp_max - max(1, span // 3)

        # Layer 4: Clamp experience ordering — never raises, always fixes silently
        sweet_min = max(exp_min, min(sweet_min, exp_max))
        sweet_max = max(sweet_min, min(sweet_max, exp_max))

        bonus_skills         = _safe_str_list(data.get("bonus_skills"))
        preferred_locations  = _safe_str_list(data.get("preferred_locations"))
        preferred_industries = _safe_str_list(data.get("preferred_industries"))
        consulting_firms     = _safe_str_list(data.get("consulting_firms"))
        jd_career_keywords   = _safe_str_list(data.get("jd_career_keywords"))
        risk_keywords        = _safe_str_list(data.get("risk_keywords"))

        # If keywords missing, derive from core skills (ensures non-empty baseline)
        if not jd_career_keywords:
            jd_career_keywords = core_skills[:10]

        # title_scores: must be dict[str, float]; silently default if malformed
        raw_title_scores = data.get("title_scores", {})
        if not isinstance(raw_title_scores, dict):
            raw_title_scores = {}
        title_scores: Dict[str, float] = {}
        for k, v in raw_title_scores.items():
            f = _safe_float(v)
            if f is not None:
                title_scores[str(k)] = f

        return {
            "title":                     title,
            "description":               str(data.get("description", "")).strip(),
            "responsibilities":          str(data.get("responsibilities", "")).strip(),
            "core_skills":               core_skills[:20],
            "bonus_skills":              bonus_skills[:10],
            "experience_target_min":     exp_min,
            "experience_target_max":     exp_max,
            "experience_sweet_spot_min": sweet_min,
            "experience_sweet_spot_max": sweet_max,
            "preferred_locations":       preferred_locations,
            "preferred_industries":      preferred_industries,
            "consulting_firms":          consulting_firms,
            "jd_career_keywords":        jd_career_keywords[:30],
            "risk_keywords":             risk_keywords,
            "title_scores":              title_scores,
        }


# ---------------------------------------------------------------------------
# Module-level pure helpers (no class state needed)
# ---------------------------------------------------------------------------

def _safe_int(val: Any) -> Optional[int]:
    """Return int(val) or None if unparseable."""
    if val is None:
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> Optional[float]:
    """Return float(val) or None if unparseable."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_str_list(val: Any) -> List[str]:
    """Convert a value to a list of non-empty strings."""
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


def _build_steps_log(validated: dict) -> List[str]:
    """
    Build human-readable completed-step strings from a validated JD dict.
    These are displayed as ✓ items in the UI summary card.
    """
    steps: List[str] = []

    title = validated.get("title", "")
    if title:
        steps.append(f"Job Title: **{title}**")

    n_core  = len(validated.get("core_skills", []))
    n_bonus = len(validated.get("bonus_skills", []))
    if n_core:
        bonus_str = f" + {n_bonus} bonus" if n_bonus else ""
        steps.append(f"{n_core} Core Skills identified{bonus_str}")

    exp_min   = validated.get("experience_target_min", 0)
    exp_max   = validated.get("experience_target_max", 0)
    sweet_min = validated.get("experience_sweet_spot_min", exp_min)
    sweet_max = validated.get("experience_sweet_spot_max", exp_max)
    steps.append(
        f"Experience: {exp_min}–{exp_max} years "
        f"(sweet spot: {sweet_min}–{sweet_max} yrs)"
    )

    n_kw = len(validated.get("jd_career_keywords", []))
    if n_kw:
        steps.append(f"{n_kw} Career Keywords mapped")

    locs = validated.get("preferred_locations", [])
    if locs:
        steps.append(f"Locations: {', '.join(locs[:3])}")

    industries = validated.get("preferred_industries", [])
    if industries:
        steps.append(f"Industries: {', '.join(industries[:3])}")

    steps.append("Review the fields below and click **Save & Activate**")
    return steps