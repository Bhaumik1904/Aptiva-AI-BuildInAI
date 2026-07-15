"""
APTIVA AI — Domain Models
=========================
JobDescription and HiringProject dataclasses.
These are the core data structures that replace the hardcoded jd_config.py
and enable the dynamic Hiring Project architecture.

No AI logic lives here. This is a pure data layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.jd_config import JD_CONFIG


# ---------------------------------------------------------------------------
# JobDescription
# ---------------------------------------------------------------------------

@dataclass
class JobDescription:
    """
    Structured representation of a Job Description.
    Contains all fields required by the scoring engine.
    Replaces the static JD_CONFIG dict imported globally by scorer.py.
    """
    title: str = "Senior AI Engineer"
    description: str = ""
    responsibilities: str = ""

    # Skills
    core_skills: List[str] = field(default_factory=list)
    bonus_skills: List[str] = field(default_factory=list)

    # Experience window
    experience_target_min: int = 5
    experience_target_max: int = 9
    experience_sweet_spot_min: int = 6
    experience_sweet_spot_max: int = 8

    # Location
    preferred_locations: List[str] = field(default_factory=list)

    # Industry / company filters
    consulting_firms: List[str] = field(default_factory=list)
    preferred_industries: List[str] = field(default_factory=list)

    # Title scoring map  {title_string: 0.0-1.0}
    title_scores: Dict[str, float] = field(default_factory=dict)

    # TF-IDF career keywords
    jd_career_keywords: List[str] = field(default_factory=list)

    # Risk keywords
    risk_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the legacy JD_CONFIG dict format expected by scorer.py."""
        return {
            "title_scores":            self.title_scores,
            "core_skills":             self.core_skills,
            "bonus_skills":            self.bonus_skills,
            "experience_target_min":   self.experience_target_min,
            "experience_target_max":   self.experience_target_max,
            "experience_sweet_spot_min": self.experience_sweet_spot_min,
            "experience_sweet_spot_max": self.experience_sweet_spot_max,
            "preferred_locations":     self.preferred_locations,
            "consulting_firms":        self.consulting_firms,
            "preferred_industries":    self.preferred_industries,
            "jd_career_keywords":      self.jd_career_keywords,
            "risk_keywords":           self.risk_keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobDescription":
        """Construct a JobDescription from a legacy JD_CONFIG dict."""
        return cls(
            title_scores              = data.get("title_scores", {}),
            core_skills               = data.get("core_skills", []),
            bonus_skills              = data.get("bonus_skills", []),
            experience_target_min     = data.get("experience_target_min", 5),
            experience_target_max     = data.get("experience_target_max", 9),
            experience_sweet_spot_min = data.get("experience_sweet_spot_min", 6),
            experience_sweet_spot_max = data.get("experience_sweet_spot_max", 8),
            preferred_locations       = data.get("preferred_locations", []),
            consulting_firms          = data.get("consulting_firms", []),
            preferred_industries      = data.get("preferred_industries", []),
            jd_career_keywords        = data.get("jd_career_keywords", []),
            risk_keywords             = data.get("risk_keywords", []),
        )


# ---------------------------------------------------------------------------
# Default JD (wraps existing hardcoded config — zero regression)
# ---------------------------------------------------------------------------

DEFAULT_JD: JobDescription = JobDescription.from_dict(JD_CONFIG)
DEFAULT_JD.title = "Senior AI Engineer"
DEFAULT_JD.description = (
    "Looking for a Senior AI Engineer with expertise in retrieval systems, "
    "embeddings, and production ML pipelines."
)


# ---------------------------------------------------------------------------
# HiringProject
# ---------------------------------------------------------------------------

@dataclass
class HiringProject:
    """
    A hiring project groups a JD, a candidate source, and ranking results.
    Stored entirely in st.session_state — no database required.
    """
    project_name: str
    recruiter_name: str = "Recruiter"
    job_description: JobDescription = field(default_factory=lambda: JobDescription.from_dict(JD_CONFIG))
    status: str = "draft"          # draft | active | archived
    candidate_source: str = "demo" # demo | csv | zip | resume
    ranking_results: List[Any] = field(default_factory=list)
    total_candidates: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    project_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # Sprint 3A: CSV file metadata (filename, size_str, extension). Never stores bytes.
    csv_file_info: Optional[Dict] = None
    # Sprint 3B: raw file bytes — persisted so ranking can read after rerun.
    csv_file_bytes: Optional[bytes] = None
    # Sprint 3B: plain-dict version of the last IngestionReport (avoids circular import).
    last_ingestion_report: Optional[Dict] = None
    # Sprint 5A: list of internal candidate dicts produced by ResumeIntelligenceAgent.
    # Shape identical to CandidateNormalizer output — feeds run_ranking_from_candidates().
    resume_candidates: Optional[List[Any]] = None
    # Sprint 5A: metadata list, one entry per processed resume file.
    # Each entry: {filename, size_str, name, yoe, n_skills, n_certs, error}
    resume_file_infos: Optional[List[Dict]] = None

    @property
    def display_name(self) -> str:
        return f"{self.project_name} ({self.job_description.title})"

    def is_ranked(self) -> bool:
        return len(self.ranking_results) > 0
