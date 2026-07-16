"""
APTIVA AI — Comparative Intelligence Agent
===========================================
Sprint 6B: Generate structured AI comparison of two shortlisted candidates
against the active Job Description.

Design constraints (identical to CandidateIntelligenceAgent):
- Completely stateless — zero Streamlit imports, zero session state.
- Model resolved 3-tier: comparison_agent.model → gemini_model → "gemini-2.5-flash"
- response_mime_type="application/json" forces structured JSON output.
- 5-layer validation with resilient defaults (mirrors matching_agent.py).
- Ranking score NEVER modified. This agent only reads scores.
- Lazy-loaded: called only when a recruiter opens the comparison page and
  clicks "Generate AI Comparison". Never called automatically.
- Results cached in session_state keyed by (cid_a, cid_b, jd_title) to
  avoid redundant Gemini calls.

Output — ComparisonPayload (validated dict):
  1. overall_comparison       — 2-3 sentence executive summary
  2. recommended_candidate    — "A" | "B" | "Equal"
  3. recommendation_reason    — 2-3 sentences with evidence
  4. shared_strengths         — list[str] (skills/qualities both have)
  5. unique_strengths_a       — list[str] (Candidate A only)
  6. unique_strengths_b       — list[str] (Candidate B only)
  7. skill_gaps_a             — list[{skill, severity, evidence}]
  8. skill_gaps_b             — list[{skill, severity, evidence}]
  9. experience_comparison    — {verdict, reasoning} comparing both vs JD
  10. education_comparison    — {verdict_a, verdict_b, reasoning}
  11. hiring_recommendation   — "Strong Hire A" | "Hire A" | "Strong Hire B" |
                                "Hire B" | "Consider Both" | "Neither Recommended"
  12. evidence_summary        — 2-3 sentences of evidence-grounded conclusion
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core.models import JobDescription


# ---------------------------------------------------------------------------
# Valid enum values
# ---------------------------------------------------------------------------

VALID_RECOMMENDED = frozenset({"A", "B", "Equal"})

VALID_HIRING_RECOMMENDATIONS = frozenset({
    "Strong Hire A",
    "Hire A",
    "Strong Hire B",
    "Hire B",
    "Consider Both",
    "Neither Recommended",
})

VALID_SEVERITIES = frozenset({"Critical", "Important", "Optional"})

VALID_EXP_VERDICTS = frozenset({
    "A Stronger",
    "B Stronger",
    "Both Meet Expectation",
    "Both Below Expectation",
    "Both Exceed Expectation",
    "A Exceeds, B Meets",
    "B Exceeds, A Meets",
})

VALID_EDU_VERDICTS = frozenset({
    "Aligned",
    "Partially Aligned",
    "Not Aligned",
    "Not Specified",
})


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    jd:              JobDescription,
    candidate_a:     Dict[str, Any],
    candidate_b:     Dict[str, Any],
    components_a:    Dict[str, Any],
    components_b:    Dict[str, Any],
) -> str:
    """
    Build a grounded comparison prompt for Gemini.

    Injects:
      1. Job Description — skills, experience range, keywords
      2. Candidate A — profile, skills, education, score signals
      3. Candidate B — profile, skills, education, score signals

    Uses string concatenation (not .format()) to avoid brace-escaping issues
    in the JSON schema section.
    """
    # --- JD context ---
    jd_core_str    = ", ".join(jd.core_skills[:15])         if jd.core_skills        else "Not specified"
    jd_bonus_str   = ", ".join(jd.bonus_skills[:8])         if jd.bonus_skills       else "None"
    jd_kw_str      = ", ".join(jd.jd_career_keywords[:15])  if jd.jd_career_keywords else "Not specified"
    exp_min        = jd.experience_target_min
    exp_max        = jd.experience_target_max
    sweet_min      = jd.experience_sweet_spot_min
    sweet_max      = jd.experience_sweet_spot_max

    # --- Candidate context builder ---
    def _cand_block(label: str, cand: Dict[str, Any], comp: Dict[str, Any]) -> str:
        p       = cand.get("profile", {})
        skills  = [s.get("name", "") for s in cand.get("skills", []) if s.get("name")]
        edu_raw = cand.get("education", [])
        edu_parts = []
        for e in edu_raw[:3]:
            parts = [e.get("degree", ""), e.get("field_of_study", ""), e.get("institution", "")]
            edu_parts.append(" · ".join(x for x in parts if x))
        edu_str  = "; ".join(edu_parts) if edu_parts else "Not specified"
        certs    = cand.get("certifications", [])
        cert_str = ", ".join(str(c) for c in certs[:5]) if certs else "None"

        hi       = comp.get("hireability_index", {}) or {}

        return (
            f"CANDIDATE {label}\n"
            f"  Role:            {p.get('current_title', 'Not specified')}\n"
            f"  Company:         {p.get('current_company', 'Not specified')}\n"
            f"  Years of Exp:    {p.get('years_of_experience', 0)}\n"
            f"  Location:        {p.get('location', 'Not specified')}\n"
            f"  Industry:        {p.get('current_industry', 'Not specified')}\n"
            f"  Skills:          {', '.join(skills[:25]) if skills else 'Not specified'}\n"
            f"  Education:       {edu_str}\n"
            f"  Certifications:  {cert_str}\n"
            f"  Summary:         {(p.get('summary', '') or 'Not provided')[:400]}\n"
            "  PRE-COMPUTED SCORES (use to ground analysis — do not re-derive):\n"
            f"    Hireability Index:  {hi.get('overall', 0):.0f}/100\n"
            f"    Technical Fit:      {hi.get('technical_fit', 0):.0f}/100\n"
            f"    Career Relevance:   {hi.get('career_relevance', 0):.0f}/100\n"
            f"    Behavior Signals:   {hi.get('behavior_signals', 0):.0f}/100\n"
            f"    Availability:       {hi.get('availability', 0):.0f}/100\n"
            f"    Skill Match Score:  {comp.get('skill_score', 0):.2f} (0-1)\n"
            f"    Experience Score:   {comp.get('experience_score', 0):.2f} (0-1)\n"
            f"    Education Score:    {comp.get('education_score', 0):.2f} (0-1)\n"
        )

    cand_a_block = _cand_block("A", candidate_a, components_a)
    cand_b_block = _cand_block("B", candidate_b, components_b)

    return (
        "You are an AI Recruitment Intelligence engine for an enterprise hiring platform.\n\n"
        "TASK\n"
        "Compare Candidate A and Candidate B for the role described below. "
        "Generate a structured comparison report to help a recruiter choose the better candidate.\n"
        "Every conclusion MUST be grounded in evidence from the candidate data and score signals.\n"
        "Do NOT invent qualifications not present in the data.\n\n"
        "INSTRUCTIONS\n"
        "- Output a single valid JSON object matching the schema below exactly.\n"
        "- Do NOT output markdown, code fences, backticks, or explanatory text.\n"
        "- If evidence is absent for a field, use the safe default shown.\n\n"
        "JOB DESCRIPTION\n"
        f"  Title:              {jd.title}\n"
        f"  Core Skills:        {jd_core_str}\n"
        f"  Bonus Skills:       {jd_bonus_str}\n"
        f"  Experience Window:  {exp_min}–{exp_max} years (sweet spot: {sweet_min}–{sweet_max} yrs)\n"
        f"  Career Keywords:    {jd_kw_str}\n\n"
        + cand_a_block + "\n"
        + cand_b_block + "\n"
        "OUTPUT SCHEMA (return this exact JSON, all keys required)\n"
        "{\n"
        '  "overall_comparison": "<2-3 sentence executive summary comparing both candidates for this role>",\n'
        '  "recommended_candidate": "<exactly one of: A | B | Equal>",\n'
        '  "recommendation_reason": "<2-3 sentences explaining why A/B is recommended with specific evidence>",\n'
        '  "shared_strengths": ["<skill or quality both candidates share that is relevant to the role>"],\n'
        '  "unique_strengths_a": ["<skill or quality that only Candidate A has, with evidence>"],\n'
        '  "unique_strengths_b": ["<skill or quality that only Candidate B has, with evidence>"],\n'
        '  "skill_gaps_a": [\n'
        '    {"skill": "<name>", "severity": "<Critical|Important|Optional>", '
        '"evidence": "<why it is missing or weak for Candidate A>"}\n'
        "  ],\n"
        '  "skill_gaps_b": [\n'
        '    {"skill": "<name>", "severity": "<Critical|Important|Optional>", '
        '"evidence": "<why it is missing or weak for Candidate B>"}\n'
        "  ],\n"
        '  "experience_comparison": {\n'
        '    "verdict": "<one of: A Stronger | B Stronger | Both Meet Expectation | Both Below Expectation | '
        'Both Exceed Expectation | A Exceeds, B Meets | B Exceeds, A Meets>",\n'
        '    "reasoning": "<2 sentences comparing both candidates years of experience vs JD window>"\n'
        "  },\n"
        '  "education_comparison": {\n'
        '    "verdict_a": "<one of: Aligned | Partially Aligned | Not Aligned | Not Specified>",\n'
        '    "verdict_b": "<one of: Aligned | Partially Aligned | Not Aligned | Not Specified>",\n'
        '    "reasoning": "<1-2 sentences comparing both candidates education relevance to the role>"\n'
        "  },\n"
        '  "hiring_recommendation": "<exactly one of: Strong Hire A | Hire A | Strong Hire B | Hire B | '
        'Consider Both | Neither Recommended>",\n'
        '  "evidence_summary": "<2-3 sentences of the most compelling evidence supporting your recommendation>"\n'
        "}\n\n"
        "FIELD RULES\n"
        "- recommended_candidate: MUST be exactly A, B, or Equal.\n"
        "- hiring_recommendation: MUST be exactly one of the 6 allowed strings.\n"
        "- shared_strengths: 1–5 items. Only include skills/qualities BOTH candidates clearly demonstrate.\n"
        "- unique_strengths_a / unique_strengths_b: 1–5 items each. Evidence-grounded.\n"
        "- skill_gaps: list all JD core skills missing or weak for each candidate.\n"
        f"  Core skills to check: {jd_core_str}.\n"
        "- Do NOT invent information not present in the candidate data above.\n"
        "- Base experience_comparison.verdict ONLY on the years-of-experience values and JD window.\n"
    )


# ---------------------------------------------------------------------------
# ComparisonIntelligenceAgent
# ---------------------------------------------------------------------------

class ComparisonIntelligenceAgent:
    """
    Stateless agent that generates structured AI comparisons of two candidates.

    Model resolution (3-tier, identical to other agents):
      1. comparison_agent.model  (config.yaml — per-agent override)
      2. gemini_model            (global fallback)
      3. "gemini-2.5-flash"      (hard-coded fallback)

    Usage
    -----
        agent = ComparisonIntelligenceAgent(config)
        payload = agent.compare(jd, candidate_a, candidate_b, components_a, components_b)

    The returned payload is a validated dict (ComparisonPayload).
    It contains NO Streamlit types. Rendering is entirely the UI layer's concern.

    Caching
    -------
    This agent is stateless and does NOT cache internally.
    The UI layer (comparison.py) caches the result in session_state keyed by
    (cid_a, cid_b, jd_title) to avoid redundant Gemini calls.
    """

    def __init__(self, config: dict):
        self._config = config
        agent_cfg    = config.get("comparison_agent", {})

        self._model_name = (
            agent_cfg.get("model")
            or config.get("gemini_model")
            or "gemini-2.5-flash"
        )
        self._temperature  = float(agent_cfg.get("temperature", 0.3))
        self._max_tokens   = int(agent_cfg.get("max_output_tokens", 3072))

        self._api_key = (
            str(config.get("gemini_api_key", "")).strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if a Gemini API key is available."""
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def compare(
        self,
        jd:           JobDescription,
        candidate_a:  Dict[str, Any],
        candidate_b:  Dict[str, Any],
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a structured comparison of Candidate A vs Candidate B.

        Parameters
        ----------
        jd           : JobDescription — active project JD
        candidate_a  : dict — internal candidate schema (A)
        candidate_b  : dict — internal candidate schema (B)
        components_a : dict — pre-computed score components for A
        components_b : dict — pre-computed score components for B

        Returns
        -------
        dict
            Validated ComparisonPayload (12-section schema).

        Raises
        ------
        ValueError
            Missing/empty input, or unrecoverable Gemini JSON response.
        RuntimeError
            API key not configured, or Gemini SDK/network failure.
        """
        if not candidate_a or not candidate_b:
            raise ValueError("Both candidates must be provided for comparison.")
        if not jd:
            raise ValueError("JobDescription is required for comparison.")
        if not self.is_configured():
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set 'gemini_api_key' in config.yaml or export GEMINI_API_KEY."
            )

        raw = self._call_gemini(jd, candidate_a, candidate_b, components_a, components_b)

        try:
            payload = self._validate_and_parse(raw)
        except json.JSONDecodeError:
            cleaned = _strip_markdown_fences(raw)
            try:
                payload = self._validate_and_parse(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned output that could not be parsed as JSON. "
                    "This may be a transient error — please try again."
                ) from exc

        return payload

    # ------------------------------------------------------------------
    # Private: Gemini call
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        jd:           JobDescription,
        candidate_a:  Dict[str, Any],
        candidate_b:  Dict[str, Any],
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
    ) -> str:
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
        prompt   = _build_prompt(jd, candidate_a, candidate_b, components_a, components_b)
        response = model.generate_content(prompt)
        return response.text

    # ------------------------------------------------------------------
    # Private: Validation (5-layer — mirrors matching_agent.py)
    # ------------------------------------------------------------------

    def _validate_and_parse(self, raw: str) -> Dict[str, Any]:
        """
        5-layer validation producing a clean, fully-populated ComparisonPayload.

        Layer 1 — JSON decode (raises json.JSONDecodeError if invalid).
        Layer 2 — Required fields: recommended_candidate, hiring_recommendation.
        Layer 3 — Optional fields populated with safe defaults.
        Layer 4 — Type normalization (clamp strings to valid enums, strip lists).
        Layer 5 — Ensure list fields are non-empty with safe defaults.
        """
        # Layer 1
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Gemini returned {type(data).__name__}, expected a JSON object."
            )

        # Layer 2: Required fields
        errors: List[str] = []

        rec_cand = str(data.get("recommended_candidate", "")).strip()
        if not rec_cand:
            errors.append("'recommended_candidate' is missing")

        hiring_rec = str(data.get("hiring_recommendation", "")).strip()
        if not hiring_rec:
            errors.append("'hiring_recommendation' is missing")

        if errors:
            raise ValueError(
                "Gemini response is missing required fields: "
                + "; ".join(errors)
            )

        # Layer 3: Optional fields with safe defaults
        overall      = str(data.get("overall_comparison", "")).strip()
        rec_reason   = str(data.get("recommendation_reason", "")).strip()
        ev_summary   = str(data.get("evidence_summary", "")).strip()

        shared_s     = _to_str_list(data.get("shared_strengths", []))
        unique_a     = _to_str_list(data.get("unique_strengths_a", []))
        unique_b     = _to_str_list(data.get("unique_strengths_b", []))
        gaps_a       = _to_gap_list(data.get("skill_gaps_a", []))
        gaps_b       = _to_gap_list(data.get("skill_gaps_b", []))

        exp_cmp_raw  = data.get("experience_comparison", {}) or {}
        exp_verdict  = str(exp_cmp_raw.get("verdict", "")).strip()
        exp_reason   = str(exp_cmp_raw.get("reasoning", "")).strip()

        edu_cmp_raw  = data.get("education_comparison", {}) or {}
        edu_va       = str(edu_cmp_raw.get("verdict_a", "")).strip()
        edu_vb       = str(edu_cmp_raw.get("verdict_b", "")).strip()
        edu_reason   = str(edu_cmp_raw.get("reasoning", "")).strip()

        # Layer 4: Normalize enums
        rec_cand    = _normalize_enum(rec_cand,    VALID_RECOMMENDED,             "Equal")
        hiring_rec  = _normalize_enum(hiring_rec,  VALID_HIRING_RECOMMENDATIONS,  "Consider Both")
        exp_verdict = _normalize_enum(exp_verdict, VALID_EXP_VERDICTS,            "Both Meet Expectation")
        edu_va      = _normalize_enum(edu_va,      VALID_EDU_VERDICTS,            "Not Specified")
        edu_vb      = _normalize_enum(edu_vb,      VALID_EDU_VERDICTS,            "Not Specified")

        # Layer 5: Ensure non-empty text fields
        overall    = overall    or "Both candidates were evaluated against the role."
        rec_reason = rec_reason or "See the score signals and shared strengths for details."
        ev_summary = ev_summary or "Review the full comparison above for evidence."

        return {
            "overall_comparison":    overall,
            "recommended_candidate": rec_cand,
            "recommendation_reason": rec_reason,
            "shared_strengths":      shared_s,
            "unique_strengths_a":    unique_a,
            "unique_strengths_b":    unique_b,
            "skill_gaps_a":          gaps_a,
            "skill_gaps_b":          gaps_b,
            "experience_comparison": {
                "verdict":   exp_verdict,
                "reasoning": exp_reason or "Experience levels compared against JD requirements.",
            },
            "education_comparison": {
                "verdict_a": edu_va,
                "verdict_b": edu_vb,
                "reasoning": edu_reason or "Education credentials evaluated against role requirements.",
            },
            "hiring_recommendation": hiring_rec,
            "evidence_summary":      ev_summary,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from Gemini output."""
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _to_str_list(raw) -> List[str]:
    """Convert a raw list to a clean list of non-empty strings."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        s = str(item).strip() if item is not None else ""
        if s:
            result.append(s)
    return result[:8]


def _to_gap_list(raw) -> List[Dict[str, str]]:
    """Convert raw skill_gaps list to validated [{skill, severity, evidence}] list."""
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        skill    = str(item.get("skill", "")).strip()
        severity = str(item.get("severity", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not skill:
            continue
        severity = severity if severity in VALID_SEVERITIES else "Optional"
        result.append({
            "skill":    skill,
            "severity": severity,
            "evidence": evidence or "Not specified",
        })
    return result[:10]


def _normalize_enum(value: str, valid_set: frozenset, default: str) -> str:
    """Return value if in valid_set, else the default."""
    if value in valid_set:
        return value
    # Case-insensitive fuzzy match
    for v in valid_set:
        if v.lower() == value.lower():
            return v
    return default
