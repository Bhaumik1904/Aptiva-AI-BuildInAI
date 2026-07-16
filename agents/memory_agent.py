"""
APTIVA AI — Recruiter Memory Agent
====================================
Sprint 6A: Store and retrieve recruiter preferences using Mem0.

Mem0 SDK compatibility: mem0ai v2.0.12
  - Import: from mem0 import MemoryClient  (module name is `mem0`, not `mem0ai`)
  - add(messages, **kwargs): user_id must go in filters={"user_id": ...}
    Top-level user_id= is rejected by the v2 API and raises ValueError.
  - search(query, **kwargs): same; limit renamed to top_k.
  - get_all(**kwargs): same; limit renamed to page_size.
  - Return values from search() and get_all() are Dict with a "results" key,
    not bare lists (v1 behaviour).

Design constraints:
- Completely stateless — no Streamlit imports, no session state.
- Every public method returns gracefully if Mem0 is unavailable.
  The application MUST continue working with zero Mem0 configuration.
- Memories are stored as plain English sentences for maximum readability
  and to leverage Mem0's semantic deduplication.
- Memories are NEVER used to modify the ranking score.
  They are used ONLY to enrich textual explanations in the shortlist.

Mem0 docs: https://docs.mem0.ai/
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.models import HiringProject, JobDescription


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_USER_ID   = "aptiva_recruiter"
_DEFAULT_MEM_LIMIT = 10        # max memories returned per recall()


# ---------------------------------------------------------------------------
# RecruiterMemoryAgent
# ---------------------------------------------------------------------------

class RecruiterMemoryAgent:
    """
    Stateless agent that stores and retrieves recruiter preferences via Mem0.

    All public methods silently return safe defaults when Mem0 is
    not configured or unavailable — the app is never broken by a
    missing API key or network error.

    Supported trigger events:
      - Hiring Project created
      - Job Description saved / activated
      - AI Insights generated for a candidate
      - Candidate shortlisted

    Memory is stored as plain English sentences, e.g.:
      "Recruiter prefers Python, FastAPI and Docker skills"
      "Recruiter targets 5–8 years of experience for backend roles"

    Usage
    -----
        agent = RecruiterMemoryAgent(config)
        agent.store_project_created(project, user_id="recruiter_001")
        memories = agent.recall("preferred skills", user_id="recruiter_001")
    """

    def __init__(self, config: dict):
        self._config   = config
        agent_cfg      = config.get("mem0_agent", {})

        self._api_key  = (
            str(config.get("mem0_api_key", "")).strip()
            or os.environ.get("MEM0_API_KEY", "").strip()
        )
        self._user_id  = str(agent_cfg.get("default_user_id", _DEFAULT_USER_ID)).strip()
        self._limit    = int(agent_cfg.get("memory_limit", _DEFAULT_MEM_LIMIT))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if a Mem0 API key is available."""
        return bool(self._api_key)

    @property
    def default_user_id(self) -> str:
        return self._user_id

    def store_project_created(
        self,
        project: HiringProject,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Store memory when a recruiter creates or activates a Hiring Project.

        Memories stored:
          - Project name and JD title
          - Required skills (core_skills)
          - Experience range
          - Preferred industries / locations (if set)

        Returns True on success, False on any failure.
        """
        if not self.is_configured():
            return False

        jd    = project.job_description
        uid   = user_id or self._user_id
        texts = _build_project_memories(project, jd)
        return self._add_memories(texts, uid)

    def store_jd_saved(
        self,
        jd: JobDescription,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Store memory when a recruiter saves a Job Description.

        Memories stored:
          - Core and bonus skills the recruiter is seeking
          - Experience range preferences
          - Industry and location preferences
        """
        if not self.is_configured():
            return False

        uid   = user_id or self._user_id
        texts = _build_jd_memories(jd)
        return self._add_memories(texts, uid)

    def store_candidate_shortlisted(
        self,
        candidate: Dict[str, Any],
        jd_title:  str,
        user_id:   Optional[str] = None,
    ) -> bool:
        """
        Store memory when a recruiter clicks into a shortlisted candidate.

        Memories stored:
          - Candidate role and seniority level
          - Skills that made the candidate stand out
          - Industry and experience level
        """
        if not self.is_configured():
            return False

        uid   = user_id or self._user_id
        texts = _build_shortlist_memories(candidate, jd_title)
        return self._add_memories(texts, uid)

    def store_ai_insights_generated(
        self,
        jd:        JobDescription,
        candidate: Dict[str, Any],
        user_id:   Optional[str] = None,
    ) -> bool:
        """
        Store memory when a recruiter generates AI Insights for a candidate.

        This captures which roles the recruiter is actively evaluating,
        helping Mem0 build a pattern of preferred candidate profiles.
        """
        if not self.is_configured():
            return False

        uid   = user_id or self._user_id
        texts = _build_insights_memories(jd, candidate)
        return self._add_memories(texts, uid)

    def recall(
        self,
        query:   str = "recruiter hiring preferences",
        user_id: Optional[str] = None,
        limit:   Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve relevant recruiter memories for a given query.

        Parameters
        ----------
        query   : str  — search query (semantic search via Mem0)
        user_id : str  — recruiter identifier
        limit   : int  — max memories to return (default: self._limit)

        Returns
        -------
        List[str]
            Plain-text memory strings, most relevant first.
            Returns [] on any error or if Mem0 is not configured.
        """
        if not self.is_configured():
            return []

        uid = user_id or self._user_id
        n   = limit   or self._limit

        try:
            client = self._get_client()
            # v2 API: user_id must be inside filters={}; limit renamed to top_k
            result = client.search(
                query,
                filters={"user_id": uid},
                top_k=n,
            )
            # v2 returns Dict{"results": [...]} not a bare list
            items = result.get("results", []) if isinstance(result, dict) else result
            memories = []
            for item in (items if isinstance(items, list) else []):
                if isinstance(item, dict):
                    mem = item.get("memory") or item.get("text") or ""
                else:
                    mem = str(item)
                if mem and str(mem).strip():
                    memories.append(str(mem).strip())
            return memories
        except Exception:   # noqa: BLE001
            return []

    def recall_all(
        self,
        user_id: Optional[str] = None,
        limit:   Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve all stored memories for a user (no semantic filter).

        Returns [] on any error or if Mem0 is not configured.
        """
        if not self.is_configured():
            return []

        uid = user_id or self._user_id
        n   = limit   or self._limit

        try:
            client = self._get_client()
            # v2 API: user_id must be inside filters={}; limit renamed to page_size
            result = client.get_all(
                filters={"user_id": uid},
                page_size=n,
            )
            # v2 returns Dict{"count": N, "results": [...]} not a bare list
            items = result.get("results", []) if isinstance(result, dict) else result
            memories = []
            for item in (items if isinstance(items, list) else []):
                if isinstance(item, dict):
                    mem = item.get("memory") or item.get("text") or ""
                else:
                    mem = str(item)
                if mem and str(mem).strip():
                    memories.append(str(mem).strip())
            return memories
        except Exception:   # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """
        Lazily import and initialise the Mem0 client.
        Raises ImportError if mem0ai is not installed.
        Raises RuntimeError if API key is missing.
        """
        try:
            from mem0 import MemoryClient            # mem0ai>=0.1.0
        except ImportError as exc:
            raise ImportError(
                "mem0ai is not installed. Run: pip install mem0ai"
            ) from exc

        return MemoryClient(api_key=self._api_key)

    def _add_memories(self, texts: List[str], user_id: str) -> bool:
        """
        Write a list of plain-text memory sentences to Mem0.

        Each sentence is stored individually so Mem0 can deduplicate
        semantically similar memories across sessions.

        v2 API: user_id must be passed inside filters={"user_id": ...}.
        Top-level user_id= raises ValueError in mem0ai>=2.0.

        Returns True if all writes succeeded, False on any error.
        """
        if not texts:
            return True

        try:
            client = self._get_client()
            for text in texts:
                if text and text.strip():
                    # v2 API: identity fields go in filters, not at top level
                    client.add(
                        [{"role": "user", "content": text.strip()}],
                        filters={"user_id": user_id},
                    )
            return True
        except Exception:   # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# Memory sentence builders (pure functions — no Mem0 dependency)
# ---------------------------------------------------------------------------

def _build_project_memories(
    project: HiringProject,
    jd: JobDescription,
) -> List[str]:
    """Build memory sentences when a project is created/activated."""
    texts: List[str] = []

    if project.project_name and jd.title:
        texts.append(
            f"Recruiter created a Hiring Project called '{project.project_name}' "
            f"for the role '{jd.title}'."
        )

    if jd.core_skills:
        skills_str = ", ".join(jd.core_skills[:10])
        texts.append(
            f"Recruiter is seeking candidates with these core skills: {skills_str}."
        )

    exp_min = jd.experience_target_min
    exp_max = jd.experience_target_max
    if exp_min is not None and exp_max is not None and exp_max > 0:
        texts.append(
            f"Recruiter targets {exp_min}–{exp_max} years of experience "
            f"for the '{jd.title}' role."
        )

    if jd.preferred_industries:
        ind_str = ", ".join(jd.preferred_industries[:5])
        texts.append(
            f"Recruiter prefers candidates from these industries: {ind_str}."
        )

    if jd.preferred_locations:
        loc_str = ", ".join(jd.preferred_locations[:5])
        texts.append(
            f"Recruiter prefers candidates located in: {loc_str}."
        )

    return texts


def _build_jd_memories(jd: JobDescription) -> List[str]:
    """Build memory sentences when a JD is saved."""
    texts: List[str] = []

    if jd.core_skills:
        skills_str = ", ".join(jd.core_skills[:10])
        texts.append(
            f"Recruiter frequently hires candidates with these skills: {skills_str}."
        )

    if jd.bonus_skills:
        bonus_str = ", ".join(jd.bonus_skills[:6])
        texts.append(
            f"Recruiter values bonus skills such as: {bonus_str}."
        )

    exp_min = jd.experience_target_min
    exp_max = jd.experience_target_max
    if exp_min is not None and exp_max is not None and exp_max > 0:
        texts.append(
            f"Recruiter prefers {exp_min}–{exp_max} years of experience "
            f"for {jd.title} roles."
        )

    if jd.preferred_industries:
        texts.append(
            f"Recruiter frequently hires from: "
            f"{', '.join(jd.preferred_industries[:5])} industries."
        )

    if jd.preferred_locations:
        texts.append(
            f"Recruiter prefers candidates in: "
            f"{', '.join(jd.preferred_locations[:5])}."
        )

    return texts


def _build_shortlist_memories(
    candidate: Dict[str, Any],
    jd_title: str,
) -> List[str]:
    """Build memory sentences when a candidate is shortlisted."""
    texts: List[str] = []

    profile = candidate.get("profile", {})
    role    = profile.get("current_title", "")
    yoe     = profile.get("years_of_experience", 0)
    ind     = profile.get("current_industry", "")
    loc     = profile.get("location", "")

    if role and jd_title:
        texts.append(
            f"Recruiter shortlisted a '{role}' candidate for the '{jd_title}' role."
        )

    if yoe and jd_title:
        texts.append(
            f"Recruiter shortlisted a candidate with {yoe:.0f} years of experience "
            f"for '{jd_title}'."
        )

    skills_raw = candidate.get("skills", [])
    top_skills = [s.get("name", "") for s in skills_raw[:5] if s.get("name")]
    if top_skills and jd_title:
        texts.append(
            f"Recruiter shortlisted a candidate with skills: "
            f"{', '.join(top_skills)} for '{jd_title}'."
        )

    if ind:
        texts.append(
            f"Recruiter shortlisted a candidate from the {ind} industry."
        )

    return texts


def _build_insights_memories(
    jd: JobDescription,
    candidate: Dict[str, Any],
) -> List[str]:
    """Build memory sentences when AI Insights are generated."""
    texts: List[str] = []

    profile = candidate.get("profile", {})
    role    = profile.get("current_title", "")

    if jd.title:
        texts.append(
            f"Recruiter generated AI insights for a '{jd.title}' candidate."
        )

    if role and jd.title:
        texts.append(
            f"Recruiter evaluated a '{role}' for the '{jd.title}' position."
        )

    if jd.core_skills:
        texts.append(
            f"Recruiter is actively evaluating candidates for skills: "
            f"{', '.join(jd.core_skills[:8])}."
        )

    return texts