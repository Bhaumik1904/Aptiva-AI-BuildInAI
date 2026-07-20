"""
APTIVA AI — Interview Intelligence Agent
========================================
Generates an AI Interview Kit for recruiters.
Helps recruiters conduct a structured interview based on:
- Candidate Resume
- Job Description
- Matching Analysis

This agent does NOT influence candidate ranking. It is a completely
independent intelligence layer.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import google.generativeai as genai

from core.models import JobDescription
from core.secrets_utils import resolve_api_key
from agents.matching_agent import _strip_markdown_fences

VALID_DIFFICULTIES = frozenset({"Junior", "Mid", "Senior"})

def _build_prompt(
    jd: JobDescription,
    candidate: Dict[str, Any],
    score_components: Dict[str, Any],
) -> str:
    """Build a rich, grounded Gemini prompt for interview kit generation."""
    jd_skills_str = ", ".join(jd.core_skills[:15]) if jd.core_skills else "Not specified"
    jd_bonus_str = ", ".join(jd.bonus_skills[:8]) if jd.bonus_skills else "None"
    
    profile = candidate.get("profile", {})
    cand_name = profile.get("anonymized_name", "Candidate")
    cand_role = profile.get("current_title", "")
    cand_co = profile.get("current_company", "")
    cand_yoe = profile.get("years_of_experience", 0)
    
    skills_raw = candidate.get("skills", [])
    skill_names = [s.get("name", "") for s in skills_raw if s.get("name")]
    skills_str = ", ".join(skill_names[:25]) if skill_names else "Not specified"
    
    # Reuse deterministic fields if available, otherwise compute manually
    matched_skills = score_components.get("matched_skills")
    if matched_skills is None:
        cand_skills_lower = {s.lower() for s in skill_names}
        matched_skills = [s for s in jd.core_skills if s.lower() in cand_skills_lower]

    missing_skills = score_components.get("missing_skills")
    if missing_skills is None:
        cand_skills_lower = {s.lower() for s in skill_names}
        missing_skills = [s for s in jd.core_skills if s.lower() not in cand_skills_lower]
        
    missing_skills_str = ", ".join(missing_skills) if missing_skills else "None (All matched)"
    matched_skills_str = ", ".join(matched_skills) if matched_skills else "None"
    
    exp_raw = candidate.get("career", [])
    exp_parts = []
    for e in exp_raw[:3]:
        parts = [e.get("title", ""), e.get("company", ""), e.get("description", "")]
        exp_parts.append(" | ".join(p for p in parts if p))
    exp_str = "\n".join(f"- {p}" for p in exp_parts) if exp_parts else "Not specified"
    
    hi = score_components.get("hireability_index", {})
    hi_overall = hi.get("overall", 0)
    hi_tech = hi.get("technical_fit", 0)
    hi_career = hi.get("career_relevance", 0)
    
    skill_score = score_components.get("skills", 0)
    exp_score = score_components.get("experience", 0)
    edu_score = score_components.get("education", 0)
    trust_score = score_components.get("trust_score", 0)
    confidence_score = score_components.get("confidence_score", "Unknown")
    
    return (
        "You are an AI Recruitment Intelligence engine for an enterprise hiring platform.\n\n"
        "TASK\n"
        "Generate a structured, recruiter-grade Interview Kit to help a recruiter conduct a highly "
        "personalized, evidence-backed interview for the candidate below.\n\n"
        "CRITICAL RULES\n"
        "1. NEVER invent information, projects, skills, or experience not present in the candidate data.\n"
        "2. NEVER ask generic interview questions (e.g., 'What is OOP?').\n"
        "3. ONLY use evidence present in the candidate profile to formulate questions.\n"
        "4. ASK deeper architectural questions for STRONG evidence-backed skills.\n"
        "5. ASK verification questions for weak confidence areas (e.g., trust_score or confidence_score is low).\n"
        "6. ASK learning and adaptability questions for missing required skills.\n"
        "7. PRIORITIZE questions using deterministic scoring provided below instead of random importance.\n"
        "8. Output a single valid JSON object matching the exact schema below.\n"
        "9. Do NOT output markdown, code fences, backticks, or explanatory text. STRICT JSON ONLY.\n\n"
        "JOB DESCRIPTION\n"
        f"  Title:                {jd.title}\n"
        f"  Core Skills Required: {jd_skills_str}\n"
        f"  Bonus Skills:         {jd_bonus_str}\n\n"
        "CANDIDATE\n"
        f"  Name:            {cand_name}\n"
        f"  Current Role:    {cand_role}\n"
        f"  Current Company: {cand_co}\n"
        f"  Years of Exp:    {cand_yoe}\n"
        f"  Skills:          {skills_str}\n"
        f"  Matched JD Skills: {matched_skills_str}\n"
        f"  Missing JD Skills: {missing_skills_str}\n"
        f"  Experience:\n{exp_str}\n\n"
        "PRE-COMPUTED SCORE SIGNALS (Use to inform focus areas and question priority)\n"
        f"  Hireability Index (overall): {hi_overall:.0f}/100\n"
        f"  Technical Fit:               {hi_tech:.0f}/100\n"
        f"  Career Relevance:            {hi_career:.0f}/100\n"
        f"  Skill Score:                 {skill_score:.2f} (0.0-1.0)\n"
        f"  Experience Score:            {exp_score:.2f} (0.0-1.0)\n"
        f"  Education Score:             {edu_score:.2f} (0.0-1.0)\n"
        f"  Trust Score:                 {trust_score:.2f} (0.0-1.0)\n"
        f"  Confidence Score:            {confidence_score}\n\n"
        "OUTPUT SCHEMA (Return exact JSON, all keys required)\n"
        "{\n"
        '  "candidate_summary": "<1-2 sentences summarizing candidate readiness>",\n'
        '  "interview_strategy": "<1-2 paragraphs on overall interview strategy>",\n'
        '  "estimated_duration_minutes": 45,\n'
        '  "difficulty": "<exactly one of: Junior | Mid | Senior>",\n'
        '  "candidate_strengths": ["<strength 1 with evidence>", "<strength 2 with evidence>"],\n'
        '  "risk_signals": ["<risk 1 with evidence>"],\n'
        '  "hire_signals": ["<strong positive signal 1>"],\n'
        '  "interviewer_checklist": ["<item to verify 1>"],\n'
        '  "technical_questions": [\n'
        '    {\n'
        '      "question": "<technical question citing resume evidence>",\n'
        '      "why_this_question": "<why this matters>",\n'
        '      "expected_strong_answer": "<what to listen for>",\n'
        '      "red_flags": ["<warning sign 1>", "<warning sign 2>"],\n'
        '      "suggested_follow_up": "<natural continuation question>"\n'
        '    }\n'
        '  ],\n'
        '  "project_questions": [\n'
        '    {\n'
        '      "question": "<project question citing resume evidence>",\n'
        '      "why_this_question": "<why this matters>",\n'
        '      "expected_strong_answer": "<what to listen for>",\n'
        '      "red_flags": ["<warning sign 1>"],\n'
        '      "suggested_follow_up": "<natural continuation question>"\n'
        '    }\n'
        '  ],\n'
        '  "behavioral_questions": [\n'
        '    {\n'
        '      "question": "<behavioral question citing resume evidence>",\n'
        '      "why_this_question": "<why this matters>",\n'
        '      "expected_strong_answer": "<what to listen for>",\n'
        '      "red_flags": ["<warning sign 1>"],\n'
        '      "suggested_follow_up": "<natural continuation question>"\n'
        '    }\n'
        '  ],\n'
        '  "areas_to_probe": [\n'
        '    "<area 1>"\n'
        '  ],\n'
        '  "final_interviewer_notes": "<closing advice>"\n'
        "}\n"
    )

class InterviewIntelligenceAgent:
    """
    Stateless agent that generates a structured Interview Kit for recruiters.
    """

    def __init__(self, config: dict):
        self._config = config
        
        # Reuse existing Gemini configuration block
        agent_cfg = config.get("matching_agent", {})
        
        self._model_name = (
            agent_cfg.get("model")
            or config.get("gemini_model")
            or "gemini-2.5-flash"
        )
        self._temperature = float(agent_cfg.get("temperature", 0.4))
        self._max_tokens = int(agent_cfg.get("max_output_tokens", 3072))
        self._api_key = resolve_api_key(config, "gemini_api_key", "GEMINI_API_KEY")

    def is_configured(self) -> bool:
        """
        Check if the Gemini API key is configured.

        Returns
        -------
        bool
            True if configured, False otherwise.
        """
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        """
        Get the current model name used by the agent.

        Returns
        -------
        str
            The Gemini model name (e.g., 'gemini-2.5-flash').
        """
        return self._model_name

    def generate_interview_kit(
        self,
        jd: JobDescription,
        candidate: Dict[str, Any],
        score_components: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a structured Interview Kit tailored to the candidate and job description.
        
        Usage
        -----
            agent = InterviewIntelligenceAgent(config)
            kit = agent.generate_interview_kit(jd, candidate, score_components)
        
        Inputs
        ------
        jd : JobDescription
            The active job description object for the role.
        candidate : Dict[str, Any]
            The normalized candidate data dictionary.
        score_components : Dict[str, Any]
            The deterministic scoring outputs computed by the scoring engine.
            
        Outputs
        -------
        Dict[str, Any]
            A strictly validated dictionary matching the Interview Kit schema.
            
        Exceptions
        ----------
        ValueError
            If candidate/JD is missing, or if Gemini returns unparseable JSON 
            or omits required fields.
        RuntimeError
            If API key is missing, or if Gemini encounters network/API failures
            that exhaust all retry attempts, or returns an empty response.
        """
        if not candidate:
            raise ValueError("Candidate dict is empty.")
        if not jd:
            raise ValueError("JobDescription is required.")
        if not self.is_configured():
            raise RuntimeError(
                "Gemini API key is not configured. "
                "Set 'gemini_api_key' in config.yaml or export GEMINI_API_KEY."
            )

        raw = self._call_gemini_with_retries(jd, candidate, score_components)

        try:
            payload = self._validate_and_parse(raw)
        except json.JSONDecodeError:
            cleaned = _strip_markdown_fences(raw)
            try:
                payload = self._validate_and_parse(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned output that could not be parsed as JSON."
                ) from exc

        return payload

    def _call_gemini_with_retries(
        self,
        jd: JobDescription,
        candidate: Dict[str, Any],
        score_components: Dict[str, Any],
        max_attempts: int = 3
    ) -> str:
        """Call Gemini API with exponential backoff for transient failures."""
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError("google-generativeai is not installed.") from exc

        genai.configure(api_key=self._api_key)
        generation_config = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_tokens,
            "response_mime_type": "application/json",
        }
        model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
        )
        prompt = _build_prompt(jd, candidate, score_components)
        
        last_exception = None
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(prompt)
                
                # --- TEMPORARY DEBUGGING ---
                print("\n" + "="*60)
                print("DEBUG: GEMINI API RESPONSE")
                print("="*60)
                print(f"1. Model name: {self._model_name}")
                print(f"2. Response object type: {type(response)}")
                
                try:
                    has_text = hasattr(response, "text") and getattr(response, "text", None) is not None
                except Exception:
                    has_text = False
                print(f"3. Whether response.text exists: {has_text}")
                
                if has_text:
                    try:
                        text_val = response.text
                        print(f"4. Length of response.text: {len(text_val)}")
                        print("5. Full response.text (use repr()):")
                        print("RAW RESPONSE:")
                        print(repr(text_val))
                        print("-" * 60)
                        print(f"9. Whether response.text is empty: {len(text_val.strip()) == 0}")
                        print(f"10. Whether response contains ```json fences: {'```json' in text_val}")
                    except Exception as e:
                        print(f"Error accessing response.text: {e}")
                else:
                    print("4. Length of response.text: N/A")
                    print("5. Full response.text: N/A")
                    print("9. Whether response.text is empty: N/A")
                    print("10. Whether response contains ```json fences: N/A")
                
                try:
                    candidates_list = getattr(response, "candidates", [])
                    if candidates_list:
                        c_obj = candidates_list[0]
                        print(f"6. Finish Reason: {getattr(c_obj, 'finish_reason', 'N/A')}")
                        print(f"7. Safety Ratings: {getattr(c_obj, 'safety_ratings', 'N/A')}")
                    else:
                        print("6. Finish Reason: N/A")
                        print("7. Safety Ratings: N/A")
                    print(f"8. Prompt Feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                except Exception as exc:
                    print(f"   Error reading metadata: {exc}")
                print("="*60 + "\n")
                # ---------------------------
                
                try:
                    text = response.text
                except Exception as e:
                    raise RuntimeError(f"Failed to access response text (possible safety block): {e}")
                
                if not text or not text.strip():
                    raise RuntimeError("Empty response from Gemini API.")
                    
                return text
                
            except RuntimeError as e:
                # If it's our own RuntimeError (e.g. empty text or safety block), we may retry
                last_exception = e
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                # Catch transient network / Google API errors
                last_exception = e
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Gemini API request failed after {max_attempts} attempts. Last error: {last_exception}")

    def _parse_question_list(self, q_list: Any) -> List[Dict[str, Any]]:
        """Safely parse a list of question dictionaries."""
        if not isinstance(q_list, list):
            return []
        parsed = []
        for q in q_list:
            if not isinstance(q, dict):
                continue
            red_flags = q.get("red_flags", [])
            if not isinstance(red_flags, list):
                red_flags = []
            parsed.append({
                "question": str(q.get("question", "")).strip(),
                "why_this_question": str(q.get("why_this_question", "")).strip(),
                "expected_strong_answer": str(q.get("expected_strong_answer", "")).strip(),
                "red_flags": [str(rf).strip() for rf in red_flags if str(rf).strip()],
                "suggested_follow_up": str(q.get("suggested_follow_up", "")).strip(),
            })
        return parsed

    def _validate_and_parse(self, raw: str) -> Dict[str, Any]:
        """
        Validate schema, ensure required fields exist, and apply defaults.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")

        # Enforce required fields instead of failing silently
        required_keys = [
            "candidate_summary", "interview_strategy", "candidate_strengths",
            "risk_signals", "hire_signals", "interviewer_checklist",
            "technical_questions", "project_questions", "behavioral_questions"
        ]
        
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"AI response is missing required fields: {', '.join(missing)}")

        tech_qs = self._parse_question_list(data.get("technical_questions", []))
        proj_qs = self._parse_question_list(data.get("project_questions", []))
        behav_qs = self._parse_question_list(data.get("behavioral_questions", []))

        # Difficulty normalization
        diff = str(data.get("difficulty", "")).strip()
        if diff not in VALID_DIFFICULTIES:
            diff = "Mid"

        # Integer duration
        try:
            dur = int(data.get("estimated_duration_minutes", 45))
        except (ValueError, TypeError):
            dur = 45

        # String list parsing helper
        def _parse_str_list(lst: Any) -> List[str]:
            if not isinstance(lst, list):
                return []
            return [str(x).strip() for x in lst if str(x).strip()]

        return {
            "candidate_summary": str(data.get("candidate_summary", "")).strip(),
            "interview_strategy": str(data.get("interview_strategy", "")).strip(),
            "estimated_duration_minutes": max(15, min(120, dur)),
            "difficulty": diff,
            "candidate_strengths": _parse_str_list(data.get("candidate_strengths", [])),
            "risk_signals": _parse_str_list(data.get("risk_signals", [])),
            "hire_signals": _parse_str_list(data.get("hire_signals", [])),
            "interviewer_checklist": _parse_str_list(data.get("interviewer_checklist", [])),
            "technical_questions": tech_qs,
            "project_questions": proj_qs,
            "behavioral_questions": behav_qs,
            "areas_to_probe": _parse_str_list(data.get("areas_to_probe", [])),
            "final_interviewer_notes": str(data.get("final_interviewer_notes", "")).strip()
        }
