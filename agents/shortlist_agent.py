"""
APTIVA AI — AI Shortlist Agent
================================
Sprint 6A: Generate an intelligent shortlist from already-ranked candidates.

Design constraints:
- The ranking score is NEVER modified. The ranking engine is the single
  source of truth. Shortlist candidates are taken in ranking order.
- Recruiter memories are used ONLY to enrich textual explanations,
  e.g.: "This candidate also aligns with your previous hiring preferences."
- Completely deterministic — no Gemini calls, no network calls.
  Zero-latency: works offline and without any API keys.
- Completely stateless — no Streamlit imports, no session state.

Architecture:
  results (already ranked)
    ↓
  ShortlistAgent.generate(results, jd, memories, top_n)
    ↓  (takes top-N in ranking order — no score modification)
  List[ShortlistEntry]  →  rendered in home.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.models import JobDescription


# ---------------------------------------------------------------------------
# ShortlistEntry — structured output per shortlisted candidate
# ---------------------------------------------------------------------------

def make_shortlist_entry(
    candidate:        Dict[str, Any],
    rank:             int,
    hi_score:         float,
    recommendation:   str,
    match_summary:    str,
    strengths:        List[str],
    risks:            List[str],
    memory_note:      str = "",
) -> Dict[str, Any]:
    """
    Create a ShortlistEntry dict.

    Fields:
      candidate        — full internal candidate dict (unchanged)
      rank             — original ranking position (never modified)
      hi_score         — Hireability Index™ overall (0–100)
      recommendation   — STRONG_YES / YES / MAYBE / NO (from scorer)
      match_summary    — one-line auto-generated summary
      strengths        — top 1-3 evidence-grounded strengths
      risks            — 0-3 risk flags (may be empty)
      memory_note      — optional Mem0-enriched context line
    """
    return {
        "candidate":      candidate,
        "rank":           rank,
        "hi_score":       hi_score,
        "recommendation": recommendation,
        "match_summary":  match_summary,
        "strengths":      strengths,
        "risks":          risks,
        "memory_note":    memory_note,
    }


# ---------------------------------------------------------------------------
# ShortlistAgent
# ---------------------------------------------------------------------------

class ShortlistAgent:
    """
    Deterministic shortlist generator.

    Takes the top-N candidates from the existing ranked results and
    enriches them with recruiter-facing summaries and optional Mem0
    memory context. The ranking order and scores are NEVER changed.

    Usage
    -----
        agent = ShortlistAgent()
        shortlist = agent.generate(
            results   = state["results"],
            jd        = project.job_description,
            memories  = ["Recruiter prefers Python, FastAPI..."],
            top_n     = 5,
        )
    """

    # Maximum candidates to read from results (performance guard)
    _MAX_INPUT = 20

    def generate(
        self,
        results:  List[Dict[str, Any]],
        jd:       Optional[JobDescription],
        memories: Optional[List[str]] = None,
        top_n:    int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate a shortlist from already-ranked results.

        Parameters
        ----------
        results  : list — ranked result dicts from the ranking engine
                   Each has keys: rank, score, candidate, components
        jd       : JobDescription — active project JD (for context)
        memories : list[str] — plain-text recruiter memories from Mem0
        top_n    : int — number of candidates to shortlist (5 or 10)

        Returns
        -------
        List[ShortlistEntry dicts] — in ranking order, max top_n items.
        """
        if not results:
            return []

        top_n     = max(1, min(top_n, self._MAX_INPUT))
        memories  = memories or []
        input_set = results[:self._MAX_INPUT]     # never scan the full list
        shortlist = []

        for result in input_set[:top_n]:
            candidate  = result.get("candidate", {})
            components = result.get("components", {})
            rank       = result.get("rank", 0)

            hi         = components.get("hireability_index", {}) or {}
            hi_score   = float(hi.get("overall", 0))
            rec        = components.get("recommendation", "MAYBE")

            summary    = _generate_summary(candidate, components, jd)
            strengths  = _identify_strengths(candidate, components, jd)
            risks      = _identify_risks(candidate, components)
            mem_note   = _build_memory_note(candidate, memories, jd)

            shortlist.append(make_shortlist_entry(
                candidate      = candidate,
                rank           = rank,
                hi_score       = hi_score,
                recommendation = rec,
                match_summary  = summary,
                strengths      = strengths,
                risks          = risks,
                memory_note    = mem_note,
            ))

        return shortlist


# ---------------------------------------------------------------------------
# Private helpers — deterministic, no external calls
# ---------------------------------------------------------------------------

def _generate_summary(
    candidate:  Dict[str, Any],
    components: Dict[str, Any],
    jd:         Optional[JobDescription],
) -> str:
    """
    Generate a one-line recruiter-facing match summary from score data.
    Template-based — no Gemini, zero latency.
    """
    profile  = candidate.get("profile", {})
    role     = profile.get("current_title", "")
    yoe      = profile.get("years_of_experience", 0)
    loc      = profile.get("location", "")
    hi       = components.get("hireability_index", {}) or {}
    hi_score = float(hi.get("overall", 0))
    rec      = components.get("recommendation", "MAYBE")

    jd_title = jd.title if jd else "this role"

    rec_labels = {
        "STRONG_YES": "Strong Hire",
        "YES":        "Hire",
        "MAYBE":      "Consider",
        "NO":         "Not Recommended",
    }
    rec_label = rec_labels.get(rec, "Consider")

    # Core sentence
    parts = []
    if role:
        parts.append(role)
    if yoe:
        yoe_str = f"{yoe:.0f}yr"
        parts.append(yoe_str)
    if loc:
        parts.append(loc)

    if parts:
        profile_str = " · ".join(parts)
        return (
            f"{profile_str}. "
            f"Hireability™ {hi_score:.0f}/100 — {rec_label} for {jd_title}."
        )

    return f"Hireability™ {hi_score:.0f}/100 — {rec_label} for {jd_title}."


def _identify_strengths(
    candidate:  Dict[str, Any],
    components: Dict[str, Any],
    jd:         Optional[JobDescription],
) -> List[str]:
    """
    Identify the top 1-3 candidate strengths from score components.
    Uses existing computed signals — no Gemini.
    """
    strengths: List[str] = []

    hi     = components.get("hireability_index", {}) or {}
    hi_tech  = float(hi.get("technical_fit", 0))
    hi_car   = float(hi.get("career_relevance", 0))
    hi_avail = float(hi.get("availability", 0))
    hi_trust = float(hi.get("trust_score", 0))

    profile = candidate.get("profile", {})
    yoe     = float(profile.get("years_of_experience", 0))
    skills  = candidate.get("skills", [])

    jd_exp_max  = jd.experience_target_max  if jd else 99
    jd_skills   = set(s.lower() for s in (jd.core_skills or [])) if jd else set()

    # Strength 1: Technical fit
    if hi_tech >= 70:
        matched_skills = [
            s.get("name", "") for s in skills
            if s.get("name", "").lower() in jd_skills
        ][:3]
        if matched_skills:
            strengths.append(f"Strong skill match: {', '.join(matched_skills)}")
        else:
            strengths.append(f"High technical fit ({hi_tech:.0f}/100)")

    # Strength 2: Experience
    exp_s = float(components.get("experience_score", 0))
    if exp_s >= 0.7:
        if yoe > jd_exp_max:
            strengths.append(f"Senior-level experience ({yoe:.0f} yrs — exceeds window)")
        else:
            strengths.append(f"Well-matched experience ({yoe:.0f} years)")

    # Strength 3: Availability / Career relevance
    if hi_avail >= 75:
        notice = candidate.get("redrob_signals", {}).get("notice_period_days", 0)
        open_w = candidate.get("redrob_signals", {}).get("open_to_work_flag", False)
        if open_w and notice <= 30:
            strengths.append("Immediately available and open to work")
        elif notice <= 15:
            strengths.append(f"Short notice period ({notice} days)")
        else:
            strengths.append("High availability score")
    elif hi_car >= 75:
        strengths.append(f"Strong career relevance ({hi_car:.0f}/100)")

    # Strength 4: Trust/Profile completeness (only if no other strengths yet)
    if not strengths and hi_trust >= 70:
        strengths.append(f"Strong trust signals ({hi_trust:.0f}/100)")

    return strengths[:3]


def _identify_risks(
    candidate:  Dict[str, Any],
    components: Dict[str, Any],
) -> List[str]:
    """
    Identify 0-3 risk flags from score components and signals.
    Uses existing computed data — no Gemini.
    """
    risks: List[str] = []

    hi       = components.get("hireability_index", {}) or {}
    hi_tech  = float(hi.get("technical_fit", 0))
    hi_avail = float(hi.get("availability", 0))
    hi_trust = float(hi.get("trust_score", 0))
    exp_s    = float(components.get("experience_score", 0))

    honeypot = components.get("honeypot_flags", [])
    if honeypot:
        risks.append(f"Honeypot flag: {honeypot[0]}")

    if hi_tech < 50:
        risks.append(f"Low technical fit ({hi_tech:.0f}/100) — review skill gaps")

    if exp_s < 0.4:
        profile = candidate.get("profile", {})
        yoe     = float(profile.get("years_of_experience", 0))
        risks.append(f"Experience mismatch ({yoe:.0f} yrs vs JD range)")

    if hi_avail < 40:
        notice = candidate.get("redrob_signals", {}).get("notice_period_days", 0)
        if notice and notice > 60:
            risks.append(f"Long notice period ({notice} days)")
        else:
            risks.append("Low availability score")

    if hi_trust < 50:
        risks.append("Low trust / incomplete profile signals")

    return risks[:3]


def _build_memory_note(
    candidate: Dict[str, Any],
    memories:  List[str],
    jd:        Optional[JobDescription],
) -> str:
    """
    Check if this candidate aligns with stored recruiter memories.
    Returns a plain-text enrichment note, or empty string if no match.

    This note NEVER modifies the ranking — it is purely informational.
    """
    if not memories:
        return ""

    profile     = candidate.get("profile", {})
    cand_ind    = (profile.get("current_industry", "") or "").lower()
    cand_loc    = (profile.get("location", "") or "").lower()
    skills_raw  = candidate.get("skills", [])
    cand_skills = {s.get("name", "").lower() for s in skills_raw if s.get("name")}

    # Look for overlap between candidate attributes and any memory sentence
    matched_signals: List[str] = []

    for mem in memories:
        mem_lower = mem.lower()

        # Check if any candidate skill appears in the memory
        for skill in cand_skills:
            if skill in mem_lower and skill not in ("", "nan"):
                matched_signals.append(f"{skill.title()} skill")
                break

        # Check industry
        if cand_ind and cand_ind in mem_lower:
            matched_signals.append(f"{profile.get('current_industry', '')} industry")

        # Check location
        if cand_loc and cand_loc in mem_lower:
            matched_signals.append(f"{profile.get('location', '')} location")

        if len(matched_signals) >= 2:
            break   # enough evidence

    if not matched_signals:
        return ""

    # Deduplicate
    seen:   set = set()
    unique: List[str] = []
    for s in matched_signals:
        sl = s.lower()
        if sl not in seen:
            seen.add(sl)
            unique.append(s)

    signal_str = " and ".join(unique[:2])
    return (
        f"\U0001f9e0 Aligns with your previous hiring preferences "
        f"({signal_str})."
    )
