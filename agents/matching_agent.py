"""
APTIVA AI — Candidate Intelligence Engine
==========================================
Sprint 5B: Explain WHY a candidate ranked the way they did, and what
a recruiter should do about it.

The agent accepts:
  - Active JobDescription  (from the active HiringProject)
  - Candidate dict         (internal schema, same as ranking engine input)
  - Score components       (already-computed by scorer.py — never re-computed here)

It produces a structured InsightsPayload containing:
  1. Overall match percentage + one-line summary
  2. Strengths (evidence-grounded)
  3. Skill Gaps (Critical / Important / Optional)
  4. Experience Analysis (verdict + reasoning vs active JD)
  5. Education Analysis (verdict + reasoning)
  6. Hiring Recommendation (exactly one of 4 values)
  7. Interview Questions (3 technical + 2 behavioural, candidate-specific)
  8. Skill Evidence (per-skill matched/missing evidence)
  9. Hiring Confidence (how certain the AI is — based on resume evidence quality)

Design constraints (identical to JD and Resume agents):
- Completely stateless — zero Streamlit imports, zero session state.
- Model is resolved 3-tier: matching_agent.model → gemini_model → "gemini-2.5-flash"
- response_mime_type="application/json" forces structured JSON output.
- 5-layer validation with resilient defaults:
    Layer 1 — JSON decode (retry after stripping markdown fences)
    Layer 2 — Required fields (hiring_recommendation, strengths, skill_gaps)
    Layer 3 — Optional fields → safe defaults
    Layer 4 — Type normalization (clamp pct, map recommendation enum)
    Layer 5 — Interview questions padded to exactly 3+2
- The ranking score is NEVER modified. This agent only reads it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from core.models import JobDescription
from core.secrets_utils import resolve_api_key


# ---------------------------------------------------------------------------
# Valid hiring recommendation values
# ---------------------------------------------------------------------------

VALID_RECOMMENDATIONS = frozenset({
    "Strong Hire",
    "Hire",
    "Consider",
    "Not Recommended",
})

VALID_SEVERITIES = frozenset({"Critical", "Important", "Optional"})

VALID_EXP_VERDICTS = frozenset({
    "Below Expectation",
    "Meets Expectation",
    "Exceeds Expectation",
})

VALID_EDU_VERDICTS = frozenset({
    "Aligned",
    "Partially Aligned",
    "Not Aligned",
    "Not Specified",
})

VALID_CONFIDENCE_LEVELS = frozenset({
    "High",
    "Medium",
    "Low",
})


# ---------------------------------------------------------------------------
# Gemini prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    jd: JobDescription,
    candidate: Dict[str, Any],
    score_components: Dict[str, Any],
) -> str:
    """
    Build a rich, grounded Gemini prompt for candidate intelligence analysis.

    Context fed in:
      1. Job Description — title, skills, experience range, keywords
      2. Candidate profile — name, role, years, skills, education, summary
      3. Score components — pre-computed numerical signals for grounding

    The prompt uses string concatenation (not .format()) to avoid escaping
    curly braces in the JSON schema.
    """
    # --- JD context ---
    jd_skills_str  = ", ".join(jd.core_skills[:15])   if jd.core_skills  else "Not specified"
    jd_bonus_str   = ", ".join(jd.bonus_skills[:8])   if jd.bonus_skills else "None"
    jd_keywords_str= ", ".join(jd.jd_career_keywords[:15]) if jd.jd_career_keywords else "Not specified"

    # --- Candidate context ---
    profile     = candidate.get("profile", {})
    cand_name   = profile.get("anonymized_name", "Candidate")
    cand_role   = profile.get("current_title", "")
    cand_co     = profile.get("current_company", "")
    cand_yoe    = profile.get("years_of_experience", 0)
    cand_loc    = profile.get("location", "")
    cand_ind    = profile.get("current_industry", "")
    cand_summary= profile.get("summary", "")

    skills_raw  = candidate.get("skills", [])
    skill_names = [s.get("name", "") for s in skills_raw if s.get("name")]
    skills_str  = ", ".join(skill_names[:25]) if skill_names else "Not specified"

    edu_raw     = candidate.get("education", [])
    edu_parts   = []
    for e in edu_raw[:3]:
        parts = [e.get("degree", ""), e.get("field_of_study", ""), e.get("institution", "")]
        edu_parts.append(" · ".join(p for p in parts if p))
    edu_str = "; ".join(edu_parts) if edu_parts else "Not specified"

    certs_raw  = candidate.get("certifications", [])
    certs_str  = ", ".join(str(c) for c in certs_raw[:5]) if certs_raw else "None"

    projects_raw = candidate.get("_extraction_meta", {})  # check for resume-sourced
    # Projects may be stored as certifications or in career desc for CSV candidates
    is_resume_source = candidate.get("_extraction_meta", {}).get("source") == "resume"

    # --- Score component context (grounds the explanation numerically) ---
    hi          = score_components.get("hireability_index", {})
    hi_overall  = hi.get("overall", 0)
    hi_tech     = hi.get("technical_fit", 0)
    hi_career   = hi.get("career_relevance", 0)
    hi_behav    = hi.get("behavior_signals", 0)
    hi_avail    = hi.get("availability", 0)
    title_s     = score_components.get("title_score", 0)
    skill_s     = score_components.get("skill_score", 0)
    exp_s       = score_components.get("experience_score", 0)
    edu_s       = score_components.get("education_score", 0)

    # Pre-compute experience verdict for the prompt
    exp_target_min = jd.experience_target_min
    exp_target_max = jd.experience_target_max
    exp_sweet_min  = jd.experience_sweet_spot_min
    exp_sweet_max  = jd.experience_sweet_spot_max

    return (
        "You are an AI Recruitment Intelligence engine for an enterprise hiring platform.\n\n"
        "TASK\n"
        "Analyze the candidate below against the job description and produce a structured "
        "intelligence report that helps a recruiter make a confident hiring decision.\n"
        "Every claim MUST be grounded in evidence from the resume or score data provided.\n"
        "Do NOT invent qualifications not present in the candidate data.\n\n"
        "INSTRUCTIONS\n"
        "- Output a single valid JSON object matching the schema below exactly.\n"
        "- Do NOT output markdown, code fences, backticks, or explanatory text.\n"
        "- If evidence for a field is absent, use the safe default shown.\n\n"
        "JOB DESCRIPTION\n"
        f"  Title:                {jd.title}\n"
        f"  Core Skills Required: {jd_skills_str}\n"
        f"  Bonus Skills:         {jd_bonus_str}\n"
        f"  Experience Window:    {exp_target_min}–{exp_target_max} years "
        f"(sweet spot: {exp_sweet_min}–{exp_sweet_max} yrs)\n"
        f"  Career Keywords:      {jd_keywords_str}\n\n"
        "CANDIDATE\n"
        f"  Name:            {cand_name}\n"
        f"  Current Role:    {cand_role}\n"
        f"  Current Company: {cand_co}\n"
        f"  Years of Exp:    {cand_yoe}\n"
        f"  Location:        {cand_loc}\n"
        f"  Industry:        {cand_ind}\n"
        f"  Skills:          {skills_str}\n"
        f"  Education:       {edu_str}\n"
        f"  Certifications:  {certs_str}\n"
        f"  Summary:         {cand_summary[:500] if cand_summary else 'Not provided'}\n\n"
        "PRE-COMPUTED SCORE SIGNALS (use these to ground your analysis)\n"
        f"  Hireability Index (overall):  {hi_overall:.0f}/100\n"
        f"  Technical Fit:               {hi_tech:.0f}/100\n"
        f"  Career Relevance:            {hi_career:.0f}/100\n"
        f"  Behavior Signals:            {hi_behav:.0f}/100\n"
        f"  Availability:                {hi_avail:.0f}/100\n"
        f"  Title Match Score:           {title_s:.2f} (0.0–1.0)\n"
        f"  Skill Match Score:           {skill_s:.2f} (0.0–1.0)\n"
        f"  Experience Score:            {exp_s:.2f} (0.0–1.0)\n"
        f"  Education Score:             {edu_s:.2f} (0.0–1.0)\n\n"
        "OUTPUT SCHEMA (return this exact JSON, all keys required)\n"
        "{\n"
        '  "overall_match_pct": <integer 0-100, derived from Hireability Index and score signals>,\n'
        '  "match_summary": "<2-3 sentences. Evidence-grounded. Reference specific skills and experience.>",\n'
        '  "hiring_recommendation": "<exactly one of: Strong Hire | Hire | Consider | Not Recommended>",\n'
        '  "recommendation_reason": "<2-3 sentences explaining the recommendation with evidence>",\n'
        '  "strengths": ["<matched skill or quality with brief evidence>"],\n'
        '  "skill_gaps": [\n'
        '    {"skill": "<name>", "severity": "<Critical|Important|Optional>", '
        '"evidence": "<why it is missing or weak>"}\n'
        "  ],\n"
        '  "experience_analysis": {\n'
        '    "verdict": "<exactly one of: Below Expectation | Meets Expectation | Exceeds Expectation>",\n'
        '    "reasoning": "<1-2 sentences comparing candidate YOE vs JD range>"\n'
        "  },\n"
        '  "education_analysis": {\n'
        '    "verdict": "<exactly one of: Aligned | Partially Aligned | Not Aligned | Not Specified>",\n'
        '    "reasoning": "<1-2 sentences about education relevance to the role>"\n'
        "  },\n"
        '  "hiring_confidence": {\n'
        '    "level": "<exactly one of: High | Medium | Low>",\n'
        '    "reasoning": "<1-2 sentences explaining how complete and consistent '
        'the resume evidence is. High = rich, verifiable evidence. '
        'Medium = some gaps in evidence. Low = sparse or inconsistent data.>"\n'
        "  },\n"
        '  "technical_questions": [\n'
        '    "<specific technical interview question based on this candidate and JD>",\n'
        '    "<specific technical interview question>",\n'
        '    "<specific technical interview question>"\n'
        "  ],\n"
        '  "behavioral_questions": [\n'
        '    "<specific behavioural interview question based on this candidate and JD>",\n'
        '    "<specific behavioural interview question>"\n'
        "  ],\n"
        '  "skill_evidence": [\n'
        '    {"skill": "<name>", "matched": <true|false>, '
        '"evidence": "<what resume says about this skill>"}\n'
        "  ]\n"
        "}\n\n"
        "FIELD RULES\n"
        f"- overall_match_pct: derive from the Hireability Index ({hi_overall:.0f}/100) "
        "adjusted by skill and experience signals. Round to nearest integer.\n"
        "- hiring_recommendation: MUST be exactly one of the 4 allowed strings.\n"
        "- strengths: 3 to 8 items. Each item must cite specific evidence from the resume.\n"
        "- skill_gaps: list ALL core skills from the JD that are absent or weak. "
        "Empty list only if ALL core skills are matched. "
        f"Core skills to check: {jd_skills_str}.\n"
        "- technical_questions: MUST be exactly 3 items. "
        "Questions must be specific to THIS candidate and THIS job — not generic.\n"
        "- behavioral_questions: MUST be exactly 2 items. "
        "Base them on candidate's career history and the role's challenges.\n"
        "- skill_evidence: include evidence for ALL core JD skills (matched or missing).\n"
        "- hiring_confidence: assess the richness of the resume data. "
        "A resume-sourced candidate with detailed sections = High. "
        "A CSV row with minimal fields = Low or Medium.\n"
        "- Do NOT invent information not present in the candidate data above.\n"
    )


# ---------------------------------------------------------------------------
# CandidateIntelligenceAgent
# ---------------------------------------------------------------------------

class CandidateIntelligenceAgent:
    """
    Stateless agent that explains a candidate's ranking in recruiter-friendly terms.

    Model resolution (identical to JD and Resume agents, 3-tier):
      1. matching_agent.model  (config.yaml — change to switch models)
      2. gemini_model          (global fallback)
      3. "gemini-2.5-flash"    (hard-coded fallback)

    Usage
    -----
        agent = CandidateIntelligenceAgent(config)
        payload = agent.analyze(jd, candidate, score_components)

    The returned payload is a validated dict matching InsightsPayload schema.
    It contains NO Streamlit-specific types. Rendering is entirely the UI layer's concern.
    """

    def __init__(self, config: dict):
        self._config    = config
        agent_cfg       = config.get("matching_agent", {})

        self._model_name = (
            agent_cfg.get("model")
            or config.get("gemini_model")
            or "gemini-2.5-flash"
        )
        # Slightly higher temperature than extraction agents — allows creative questions
        self._temperature  = float(agent_cfg.get("temperature", 0.2))
        self._max_tokens   = int(agent_cfg.get("max_output_tokens", 3072))

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
        jd: JobDescription,
        candidate: Dict[str, Any],
        score_components: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate the full InsightsPayload for one candidate.

        Parameters
        ----------
        jd : JobDescription
            The active project's JobDescription (already structured).
        candidate : dict
            Internal candidate dict (from ranking engine / normalizer).
        score_components : dict
            Pre-computed score components from compute_final_score().
            This agent reads them for grounding — never re-computes.

        Returns
        -------
        dict
            Validated InsightsPayload matching the 9-section schema.

        Raises
        ------
        ValueError
            Empty or missing candidate/JD, or unrecoverable Gemini response.
        RuntimeError
            API key not configured, or Gemini SDK/network failure.
        """
        if not candidate:
            raise ValueError("Candidate dict is empty — nothing to analyze.")
        if not jd:
            raise ValueError("JobDescription is required for matching analysis.")
        if not self.is_configured():
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set 'gemini_api_key' in config.yaml or export GEMINI_API_KEY."
            )

        raw = self._call_gemini(jd, candidate, score_components)

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
        jd: JobDescription,
        candidate: Dict[str, Any],
        score_components: Dict[str, Any],
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
        prompt   = _build_prompt(jd, candidate, score_components)
        response = model.generate_content(prompt)
        return response.text

    # ------------------------------------------------------------------
    # Private: Validation
    # ------------------------------------------------------------------

    def _validate_and_parse(self, raw: str) -> Dict[str, Any]:
        """
        5-layer validation that mirrors JD and Resume agent philosophy.

        Layer 1 — JSON decode (raises json.JSONDecodeError if invalid).
        Layer 2 — Required fields: hiring_recommendation, strengths, skill_gaps.
        Layer 3 — Optional fields populated with safe defaults.
        Layer 4 — Type normalization (clamp pct, map enums to valid values).
        Layer 5 — Interview questions padded to exactly 3 technical + 2 behavioural.

        Returns a clean, fully-populated InsightsPayload dict.
        """
        # Layer 1
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(
                f"Gemini returned {type(data).__name__}, expected a JSON object."
            )

        # Layer 2: Required fields
        errors: List[str] = []

        rec = str(data.get("hiring_recommendation", "")).strip()
        if not rec:
            errors.append("'hiring_recommendation' is missing")

        strengths_raw = data.get("strengths", [])
        if not isinstance(strengths_raw, list):
            strengths_raw = []

        skill_gaps_raw = data.get("skill_gaps", [])
        if not isinstance(skill_gaps_raw, list):
            skill_gaps_raw = []

        if errors:
            raise ValueError(
                "AI response is missing required fields: " + "; ".join(errors)
            )

        # Layer 3 & 4: Optional fields + type normalization

        # Clamp match percentage
        raw_pct = data.get("overall_match_pct", 0)
        try:
            match_pct = max(0, min(100, int(float(str(raw_pct)))))
        except (ValueError, TypeError):
            match_pct = 0

        # Map recommendation to enum — default to "Consider" on unknown
        rec_normalized = rec if rec in VALID_RECOMMENDATIONS else "Consider"

        # Experience verdict
        exp_data    = data.get("experience_analysis", {}) or {}
        exp_verdict = str(exp_data.get("verdict", "")).strip()
        if exp_verdict not in VALID_EXP_VERDICTS:
            exp_verdict = "Meets Expectation"
        exp_reason  = str(exp_data.get("reasoning", "")).strip() or "Experience analysis not available."

        # Education verdict
        edu_data    = data.get("education_analysis", {}) or {}
        edu_verdict = str(edu_data.get("verdict", "")).strip()
        if edu_verdict not in VALID_EDU_VERDICTS:
            edu_verdict = "Not Specified"
        edu_reason  = str(edu_data.get("reasoning", "")).strip() or "Education analysis not available."

        # Hiring confidence (new section — approved addition)
        conf_data   = data.get("hiring_confidence", {}) or {}
        conf_level  = str(conf_data.get("level", "")).strip()
        if conf_level not in VALID_CONFIDENCE_LEVELS:
            conf_level = "Medium"
        conf_reason = str(conf_data.get("reasoning", "")).strip() or "Confidence assessment not available."

        # Strengths — list of non-empty strings
        strengths = [str(s).strip() for s in strengths_raw if str(s).strip()]
        if not strengths:
            strengths = ["Candidate data insufficient for strength analysis"]

        # Skill gaps — normalize each entry
        skill_gaps: List[Dict[str, str]] = []
        for g in skill_gaps_raw:
            if not isinstance(g, dict):
                continue
            sev = str(g.get("severity", "")).strip()
            skill_gaps.append({
                "skill":    str(g.get("skill", "")).strip() or "Unknown",
                "severity": sev if sev in VALID_SEVERITIES else "Important",
                "evidence": str(g.get("evidence", "")).strip() or "No evidence found.",
            })

        # Skill evidence
        evidence_raw = data.get("skill_evidence", [])
        if not isinstance(evidence_raw, list):
            evidence_raw = []
        skill_evidence: List[Dict[str, Any]] = []
        for e in evidence_raw:
            if not isinstance(e, dict):
                continue
            skill_evidence.append({
                "skill":    str(e.get("skill", "")).strip() or "Unknown",
                "matched":  bool(e.get("matched", False)),
                "evidence": str(e.get("evidence", "")).strip() or "No detail available.",
            })

        # Layer 5: Interview questions — pad to exactly 3 + 2
        tech_q_raw  = data.get("technical_questions", [])
        behav_q_raw = data.get("behavioral_questions", [])
        if not isinstance(tech_q_raw, list):
            tech_q_raw = []
        if not isinstance(behav_q_raw, list):
            behav_q_raw = []

        tech_qs  = [str(q).strip() for q in tech_q_raw  if str(q).strip()]
        behav_qs = [str(q).strip() for q in behav_q_raw if str(q).strip()]

        # Pad to minimums with informative placeholders
        while len(tech_qs) < 3:
            tech_qs.append(
                "Describe your approach to a technical challenge in your most recent role."
            )
        while len(behav_qs) < 2:
            behav_qs.append(
                "Tell me about a time you overcame a significant professional challenge."
            )

        return {
            "overall_match_pct":     match_pct,
            "match_summary":         str(data.get("match_summary", "")).strip()
                                     or "Match summary not available.",
            "hiring_recommendation": rec_normalized,
            "recommendation_reason": str(data.get("recommendation_reason", "")).strip()
                                     or "Recommendation reasoning not available.",
            "strengths":             strengths[:10],
            "skill_gaps":            skill_gaps,
            "experience_analysis": {
                "verdict":   exp_verdict,
                "reasoning": exp_reason,
            },
            "education_analysis": {
                "verdict":   edu_verdict,
                "reasoning": edu_reason,
            },
            "hiring_confidence": {
                "level":     conf_level,
                "reasoning": conf_reason,
            },
            "technical_questions":   tech_qs[:3],
            "behavioral_questions":  behav_qs[:2],
            "skill_evidence":        skill_evidence,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from Gemini output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text