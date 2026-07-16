"""
APTIVA AI — Streamlit Application
==================================
AI Recruitment Intelligence Platform.

Run: streamlit run app.py
"""

from core.reasoning import generate_reasoning
import csv
import io
import time
from pathlib import Path

import streamlit as st
import yaml

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="APTIVA AI — Intelligent Candidate Discovery",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Core imports ──────────────────────────────────────────────────────────────
from core.data_ingestion import DatasetLoader
from core.models import DEFAULT_JD, HiringProject, JobDescription
from core.scorer import compute_final_score
from core.jd_config import JD_CONFIG
from core.similarity import build_tfidf_index
from core.csv_loader import CSVLoader, ExcelLoader
from ui.styles import inject_styles
from ui.pages import home, ai_analysis, candidate_profile, comparison, judge_mode_page, analytics
from ui.pages import projects as projects_page
from ui.pages import candidate_sources as candidate_sources_page
# ── Sprint 6A: Memory + Shortlist agents ───────────────────────────────────────
from agents.memory_agent import RecruiterMemoryAgent
from agents.shortlist_agent import ShortlistAgent


# ── Load Config ───────────────────────────────────────────────────────────────
@st.cache_data
def load_config() -> dict:
    try:
        with open("config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


# ── Dataset Auto-Setup ────────────────────────────────────────────────────────
@st.cache_resource
def setup_dataset() -> DatasetLoader:
    loader = DatasetLoader(data_dir="./data")
    loader.auto_setup()
    return loader


# ── Ranking Pipeline ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_ranking(candidates_path_str: str, jd_dict_key: str = "default", top_n: int = 100) -> dict:
    """
    Run the full ranking pipeline. Cached per (candidates_path, jd_dict_key).
    Returns: {"results": [...], "total": int, "submission_csv": str}

    jd_dict_key is a hashable cache key derived from the active JD so that
    changing the JD triggers a fresh ranking run without clearing the entire cache.
    The actual JD dict is pulled from session state inside the function.
    """
    import heapq
    from pathlib import Path

    loader = DatasetLoader(data_dir="./data")
    target_path = Path(candidates_path_str)

    # Resolve active JD from session state (not passed directly — keeps cache key small)
    active_project: HiringProject = st.session_state.get("active_project")
    jd_dict = active_project.job_description.to_dict() if active_project else JD_CONFIG

    candidates = loader.load_all_candidates(target_path)
    if not candidates:
        return {"results": [], "total": 0, "submission_csv": ""}

    # Build TF-IDF using the active JD's career keywords
    _, _, _, similarities = build_tfidf_index(candidates, jd=jd_dict)

    # Score all — pass active JD so every scoring function uses the project JD
    scored = []
    for i, candidate in enumerate(candidates):
        tfidf_sim = float(similarities[i])
        final_score, components = compute_final_score(candidate, tfidf_sim, jd=jd_dict)
        scored.append((final_score, candidate, components))

    # Top-N
    import heapq
    top_results = heapq.nlargest(top_n, scored, key=lambda x: x[0])
    top_results.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))

    # Build results list
    results = []
    for rank, (score, candidate, components) in enumerate(top_results, start=1):
        reasoning = generate_reasoning(candidate, rank, components)
        results.append({
            "rank":        rank,
            "score":       score,
            "candidate":   candidate,
            "components":  components,
            "reasoning":   reasoning,
        })

    # Build CSV string
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([r["candidate"]["candidate_id"], r["rank"], round(r["score"], 4), r["reasoning"]])
    csv_str = csv_buffer.getvalue()

    return {
        "results":        results,
        "total":          len(candidates),
        "submission_csv": csv_str,
    }


# ── Sprint 3B: Ranking from pre-loaded candidate list ─────────────────────────
def run_ranking_from_candidates(
    candidates: list,
    jd_dict: dict,
    top_n: int = 100,
) -> dict:
    """
    Run the full ranking pipeline on a pre-loaded List[Dict] of candidates.
    Used by the CSV/Excel ingestion path. Not cached (data already in memory).
    Logic is identical to run_ranking() except file loading is skipped.

    Returns: {"results": [...], "total": int, "submission_csv": str}
    """
    import heapq
    from core.reasoning import generate_reasoning

    if not candidates:
        return {"results": [], "total": 0, "submission_csv": ""}

    _, _, _, similarities = build_tfidf_index(candidates, jd=jd_dict)

    scored = []
    for i, candidate in enumerate(candidates):
        tfidf_sim = float(similarities[i])
        final_score, components = compute_final_score(candidate, tfidf_sim, jd=jd_dict)
        scored.append((final_score, candidate, components))

    top_results = heapq.nlargest(top_n, scored, key=lambda x: x[0])
    top_results.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))

    results = []
    for rank, (score, candidate, components) in enumerate(top_results, start=1):
        reasoning = generate_reasoning(candidate, rank, components)
        results.append({
            "rank":       rank,
            "score":      score,
            "candidate":  candidate,
            "components": components,
            "reasoning":  reasoning,
        })

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([
            r["candidate"]["candidate_id"],
            r["rank"],
            round(r["score"], 4),
            r["reasoning"],
        ])

    return {
        "results":        results,
        "total":          len(candidates),
        "submission_csv": csv_buffer.getvalue(),
    }


def _run_csv_ranking(project: HiringProject) -> None:
    """
    Sprint 3B: Load a CSV/Excel file from project.csv_file_bytes,
    normalise it, and run the ranking pipeline.
    Writes results directly to session state.
    """
    file_bytes = getattr(project, "csv_file_bytes", None)
    if not file_bytes:
        st.session_state["ranking_error"] = (
            "No CSV/Excel file found. "
            "Go to Candidate Sources and upload a file first."
        )
        return

    file_info = getattr(project, "csv_file_info", {}) or {}
    filename  = file_info.get("filename", "upload.csv")
    ext       = file_info.get("extension", "CSV").lower()

    st.session_state["ranking_running"] = True

    # Simple loading placeholder for the CSV path
    loading_slot = st.empty()
    loading_slot.markdown(
        """
<div style="position:fixed;top:0;left:0;width:100vw;height:100vh;
            background:#FFFFFF;z-index:9999999;
            display:flex;flex-direction:column;
            align-items:center;justify-content:center;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="font-size:1.625rem;font-weight:800;color:#1D1D1F;
              letter-spacing:-0.035em;margin-bottom:0.375rem">APTIVA AI</div>
  <div style="font-size:0.8125rem;color:#86868B;margin-bottom:2rem">
    Processing uploaded candidates&hellip;
  </div>
  <div style="font-size:0.875rem;color:#6E6E73">Normalising &rarr; Ranking &rarr; Scoring</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Load and normalise
    try:
        if ext == "csv":
            file_loader = CSVLoader()
        else:
            file_loader = ExcelLoader()
        candidates, report = file_loader.load(file_bytes, filename=filename)
        # Persist the final report back to the project
        project.last_ingestion_report = report.to_dict()
    except Exception as exc:
        loading_slot.empty()
        st.session_state["ranking_running"] = False
        st.session_state["ranking_error"]   = f"Failed to load file: {exc}"
        return

    if not candidates:
        loading_slot.empty()
        st.session_state["ranking_running"] = False
        errs = "; ".join((report.validation_errors or ["Unknown error"])[:3])
        st.session_state["ranking_error"] = (
            f"No valid candidates found in '{filename}'. {errs}"
        )
        return

    # Run ranking
    jd_dict = project.job_description.to_dict()
    try:
        result_data = run_ranking_from_candidates(candidates, jd_dict)
    except Exception as exc:
        loading_slot.empty()
        st.session_state["ranking_running"] = False
        st.session_state["ranking_error"]   = f"Ranking failed: {exc}"
        return

    loading_slot.empty()
    st.session_state["ranking_running"]   = False
    st.session_state["results"]           = result_data.get("results", [])
    st.session_state["total_candidates"]  = result_data.get("total", 0)
    st.session_state["submission_csv"]    = result_data.get("submission_csv", "")
    st.session_state["ranking_done"]      = True
    if st.session_state["results"] and not st.session_state.get("selected_candidate_id"):
        st.session_state["selected_candidate_id"] = (
            st.session_state["results"][0]["candidate"]["candidate_id"]
        )
    # ── Sprint 6A: generate shortlist after CSV/Excel ranking ─────────────────
    _generate_shortlist_from_state(st.session_state.get("app_config", {}))


def _run_resume_ranking(project: HiringProject) -> None:
    """
    Sprint 5A: Run the ranking pipeline on resume_candidates.
    Mirrors _run_csv_ranking() — feeds project.resume_candidates through
    run_ranking_from_candidates(). The ranking engine sees zero difference.
    Writes results directly to session state.
    """
    candidates = getattr(project, "resume_candidates", None) or []
    if not candidates:
        st.session_state["ranking_error"] = (
            "No resume candidates found. "
            "Go to Candidate Sources, upload resumes, and click ✨ Analyze Resumes first."
        )
        return

    st.session_state["ranking_running"] = True

    loading_slot = st.empty()
    loading_slot.markdown(
        """
<div style="position:fixed;top:0;left:0;width:100vw;height:100vh;
            background:#FFFFFF;z-index:9999999;
            display:flex;flex-direction:column;
            align-items:center;justify-content:center;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="font-size:1.625rem;font-weight:800;color:#1D1D1F;
              letter-spacing:-0.035em;margin-bottom:0.375rem">APTIVA AI</div>
  <div style="font-size:0.8125rem;color:#86868B;margin-bottom:2rem">
    Processing resume candidates&hellip;
  </div>
  <div style="font-size:0.875rem;color:#6E6E73">Normalising &rarr; Ranking &rarr; Scoring</div>
</div>""",
        unsafe_allow_html=True,
    )

    jd_dict = project.job_description.to_dict()
    try:
        result_data = run_ranking_from_candidates(candidates, jd_dict)
    except Exception as exc:
        loading_slot.empty()
        st.session_state["ranking_running"] = False
        st.session_state["ranking_error"]   = f"Ranking failed: {exc}"
        return

    loading_slot.empty()
    st.session_state["ranking_running"]   = False
    st.session_state["results"]           = result_data.get("results", [])
    st.session_state["total_candidates"]  = result_data.get("total", 0)
    st.session_state["submission_csv"]    = result_data.get("submission_csv", "")
    st.session_state["ranking_done"]      = True
    if st.session_state["results"] and not st.session_state.get("selected_candidate_id"):
        st.session_state["selected_candidate_id"] = (
            st.session_state["results"][0]["candidate"]["candidate_id"]
        )
    # ── Sprint 6A: generate shortlist after resume ranking ────────────────────
    _generate_shortlist_from_state(st.session_state.get("app_config", {}))


def init_state():
    defaults = {
        # Navigation
        "page":                  "home",
        # Candidate selection
        "selected_candidate_id": None,
        "compare_list":          [],
        # Ranking results
        "results":               [],
        "total_candidates":      0,
        "submission_csv":        "",
        "ranking_done":          False,
        "ranking_running":       False,
        "ranking_error":         None,
        "dataset_status":        None,
        # ── Sprint 2: Hiring Projects ──────────────────────────────────────
        # projects: dict[project_id -> HiringProject]
        "projects":              {},
        # active_project: the currently selected HiringProject (or None)
        "active_project":        None,
        # ── Sprint 6A: Recruiter Memory + Shortlist ─────────────────────────────
        # recruiter_memories: plain-text list from Mem0 for the active project
        "recruiter_memories":    [],
        # shortlist: List[ShortlistEntry] — top-N from ShortlistAgent
        "shortlist":             [],
        # ── Sprint 6B: AI Comparison cache ──────────────────────────────────────
        # ai_comparison_cache: Dict[cache_key -> ComparisonPayload]
        # Cached by (cid_a, cid_b, jd_title) — cleared when project changes.
        "ai_comparison_cache":   {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(config: dict, loader: DatasetLoader):
    with st.sidebar:
        # Logo / Brand
        _logo_hex = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-3px"><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/></svg>'
        st.markdown(
            f"""
<div style="padding:0.5rem 0 1.5rem">
  <div style="font-size:1.375rem;font-weight:800;color:#1D1D1F;letter-spacing:-0.03em;display:flex;align-items:center;gap:0.4rem">{_logo_hex} APTIVA AI</div>
  <div style="font-size:0.75rem;color:#86868B;margin-top:0.125rem;letter-spacing:0.01em">Intelligent Candidate Discovery</div>
</div>""",
            unsafe_allow_html=True,
        )

        # Navigation
        # SVG icons — Lucide-style, 16×16, stroke-based
        _ICONS = {
            "home": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7c0 3.31 2.69 6 6 6s6-2.69 6-6V2z"/></svg>',
            "projects": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 7a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2Z"/></svg>',
            "ai_analysis": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>',
            "candidate_profile": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
            "comparison": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>',
            "judge_mode": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m14 13-7.5 7.5c-.83.83-2.17.83-3 0 0 0 0 0 0 0a2.12 2.12 0 0 1 0-3L11 10"/><path d="m16 16 6-6"/><path d="m8 8 6-6"/><path d="m9 7 8 8"/><path d="m21 11-8-8"/></svg>',
            "analytics": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="18" y="3" width="4" height="18"/><rect x="10" y="8" width="4" height="13"/><rect x="2" y="13" width="4" height="8"/></svg>',
            "candidate_sources": '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
        }

        pages = [
            ("projects",           "Hiring Projects"),
            ("candidate_sources",  "Candidate Sources"),
            ("home",               "Rankings"),
            ("ai_analysis",        "AI Analysis"),
            ("candidate_profile",  "Candidate View"),
            ("comparison",         "Compare"),
            ("judge_mode",         "Judge Mode"),
            ("analytics",          "Analytics"),
        ]

        st.markdown('<div style="font-size:0.6875rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Navigation</div>', unsafe_allow_html=True)

        for page_key, label in pages:
            is_active = st.session_state["page"] == page_key
            icon_svg = _ICONS.get(page_key, "")
            active_style = (
                "background:#E8F2FF;color:#0071E3;border:1px solid #C8DEFF;"
                if is_active else
                "background:transparent;color:#1D1D1F;border:1px solid transparent;"
            )
            # Render as an HTML nav item so the SVG icon shows cleanly
            st.markdown(
                f'<div style="{active_style}display:flex;align-items:center;gap:0.625rem;'
                f'padding:0.5rem 0.75rem;border-radius:6px;margin:0.125rem 0;'
                f'font-size:0.875rem;font-weight:500;cursor:pointer">'
                f'{icon_svg}<span>{label}</span></div>',
                unsafe_allow_html=True,
            )
            # Invisible button that captures the click
            if st.button(label, key=f"nav_{page_key}", use_container_width=True,
                         type="secondary" if not is_active else "primary"):
                st.session_state["page"] = page_key
                st.rerun()

        st.markdown("---")

        # Sprint 3B / 5A: compute can_rank
        _ap_sb   = st.session_state.get("active_project")
        _src_sb  = _ap_sb.candidate_source if _ap_sb else "demo"
        _csv_rdy = (
            _src_sb in ("csv", "excel")
            and _ap_sb is not None
            and bool(getattr(_ap_sb, "csv_file_bytes", None))
        )
        _resume_rdy = (
            _src_sb == "resume"
            and _ap_sb is not None
            and bool(getattr(_ap_sb, "resume_candidates", None))
        )
        candidates_path = loader.get_candidates_path()
        can_rank = bool(candidates_path) or _csv_rdy or _resume_rdy

        # Dataset status
        st.markdown('<div style="font-size:0.6875rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Dataset</div>', unsafe_allow_html=True)

        if _csv_rdy:
            _fi   = getattr(_ap_sb, "csv_file_info", {}) or {}
            _rpt  = getattr(_ap_sb, "last_ingestion_report", {}) or {}
            fname = _fi.get("filename", "Uploaded file")
            _n    = _rpt.get("successful_imports", "?")
            _ck   = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1A8917" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><polyline points="20 6 9 17 4 12"/></svg>'
            st.markdown(
                f'<div style="font-size:0.8125rem;color:#1A8917;display:flex;align-items:center;gap:0.35rem">{_ck} {fname}</div>'
                f'<div style="font-size:0.75rem;color:#6E6E73;margin-top:0.125rem">{_n} candidates ready</div>',
                unsafe_allow_html=True,
            )
        elif _resume_rdy:
            _res_cands = getattr(_ap_sb, "resume_candidates", []) or []
            _n_res     = len(_res_cands)
            _ck = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1A8917" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><polyline points="20 6 9 17 4 12"/></svg>'
            st.markdown(
                f'<div style="font-size:0.8125rem;color:#1A8917;display:flex;align-items:center;gap:0.35rem">{_ck} {_n_res} resume(s) analyzed</div>'
                f'<div style="font-size:0.75rem;color:#6E6E73;margin-top:0.125rem">{_n_res} candidates ready</div>',
                unsafe_allow_html=True,
            )
        elif candidates_path:
            fname = candidates_path.name
            _ck = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1A8917" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><polyline points="20 6 9 17 4 12"/></svg>'
            st.markdown(
                f'<div style="font-size:0.8125rem;color:#1A8917;display:flex;align-items:center;gap:0.35rem">{_ck} {fname}</div>',
                unsafe_allow_html=True,
            )
        else:
            _warn = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#CC0000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            st.markdown(
                f'<div style="font-size:0.8125rem;color:#CC0000;display:flex;align-items:center;gap:0.35rem">{_warn} No dataset found</div>',
                unsafe_allow_html=True,
            )
            st.caption("Place ZIP file in data/ or upload CSV in Candidate Sources")

        st.markdown("---")
        
        # ── Deployment Health Check ──────────────────────────────────────
        st.markdown('<div style="font-size:0.6875rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Cloud Readiness</div>', unsafe_allow_html=True)
        
        from core.secrets_utils import resolve_api_key
        has_gemini = bool(resolve_api_key(config, "gemini_api_key", "GEMINI_API_KEY"))
        has_mem0 = bool(resolve_api_key(config, "mem0_api_key", "MEM0_API_KEY"))
        
        def _render_health_item(label: str, is_ready: bool):
            _icon = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1A8917" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><polyline points="20 6 9 17 4 12"/></svg>' if is_ready else '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#86868B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
            _color = "#1A8917" if is_ready else "#86868B"
            _text = "Ready" if is_ready else "Missing"
            st.markdown(f'<div style="font-size:0.8125rem;display:flex;justify-content:space-between;margin-bottom:0.25rem"><span style="color:#1D1D1F">{label}</span><span style="color:{_color};display:flex;align-items:center;gap:0.2rem">{_icon} {_text}</span></div>', unsafe_allow_html=True)

        _render_health_item("Gemini API", has_gemini)
        _render_health_item("Mem0 Config", has_mem0)
        _render_health_item("Demo Dataset", bool(candidates_path) or _csv_rdy or _resume_rdy)
        _render_health_item("File Uploads", True) # openpyxl+docx present

        st.markdown("---")

        # Run Ranking Button
        if can_rank:
            if st.session_state.get("ranking_running"):
                # Disabled state shown while analysis is running
                _ldr = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#86868B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-1px"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>'
                st.markdown(
                    f'<div style="background:#F5F5F7;border-radius:6px;padding:0.5rem 0.75rem;'
                    f'font-size:0.8125rem;color:#86868B;text-align:center;cursor:not-allowed;'
                    f'display:flex;align-items:center;justify-content:center;gap:0.4rem">'
                    f'{_ldr} Running Analysis…</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button("▶ Run Ranking Analysis", use_container_width=True, type="primary"):
                    st.session_state["ranking_done"]    = False
                    st.session_state["ranking_error"]   = None
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.markdown(
                '<div style="font-size:0.8125rem;color:#86868B">Upload candidates to run ranking</div>',
                unsafe_allow_html=True,
            )

        # Stats if ranked
        if st.session_state.get("ranking_done"):
            results = st.session_state.get("results", [])
            total = st.session_state.get("total_candidates", 0)
            st.markdown("---")
            st.markdown('<div style="font-size:0.6875rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Ranking Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.8125rem;color:#1D1D1F">Analyzed: <strong>{total:,}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.8125rem;color:#1D1D1F">Top ranked: <strong>{len(results)}</strong></div>', unsafe_allow_html=True)
            if results:
                top_hi = results[0].get("components", {}).get("hireability_index", {}).get("overall", 0)
                st.markdown(f'<div style="font-size:0.8125rem;color:#1D1D1F">Best Hireability™: <strong>{top_hi:.0f}/100</strong></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Active Project widget
        active_project: HiringProject = st.session_state.get("active_project")
        st.markdown('<div style="font-size:0.6875rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Active Project</div>', unsafe_allow_html=True)
        if active_project:
            st.markdown(
                f'<div style="font-size:0.8125rem;color:#0071E3;font-weight:600">{active_project.project_name}</div>'
                f'<div style="font-size:0.75rem;color:#86868B">{active_project.job_description.title}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.8125rem;color:#C7C7CC;font-style:italic">No active project</div>',
                unsafe_allow_html=True,
            )
            if st.button("+ Create Project", use_container_width=True, key="sidebar_create_project"):
                st.session_state["page"] = "projects"
                st.rerun()

        st.markdown("---")

        # Compare list
        compare_list = st.session_state.get("compare_list", [])
        if compare_list:
            st.markdown('<div style="font-size:0.6875rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Compare List</div>', unsafe_allow_html=True)
            for cid in compare_list:
                st.markdown(f'<div style="font-size:0.8125rem;color:#1D1D1F">• {cid}</div>', unsafe_allow_html=True)
            if st.button("Clear Compare List", use_container_width=True):
                st.session_state["compare_list"] = []
                st.rerun()


# ── Auto-Run Ranking ──────────────────────────────────────────────────────────
def auto_run_ranking(loader: DatasetLoader):
    """Run ranking automatically when ranking_done is False."""
    if st.session_state.get("ranking_done"):
        return  # Already ranked — nothing to do

    # ── Sprint 5A: dispatch to Resume path ───────────────────────────────
    active_project: HiringProject = st.session_state.get("active_project")
    source = active_project.candidate_source if active_project else "demo"
    if source == "resume":
        _run_resume_ranking(active_project)
        return

    # ── Sprint 3B: dispatch to CSV/Excel path ─────────────────────────────
    if source in ("csv", "excel"):
        _run_csv_ranking(active_project)
        return

    # ── Demo dataset path (unchanged) ─────────────────────────────────────
    candidates_path = loader.get_candidates_path()
    if not candidates_path:
        return

    # ── Premium Loading Screen ────────────────────────────────────────────
    # All UI lives in a single HTML slot so updates feel like transitions,
    # not Streamlit widget flashes. Only the slot content is swapped.
    loading_slot = st.empty()

    _CSS = """
<style>
@keyframes aptiva-fadein {
  from { opacity:0; }
  to   { opacity:1; }
}
@keyframes aptiva-fadeout {
  from { opacity:1; }
  to   { opacity:0; }
}
@keyframes aptiva-label-pulse {
  0%,100% { opacity:1; }
  50%      { opacity:0.6; }
}
.aptiva-exiting {
  animation: aptiva-fadeout 0.4s ease forwards !important;
  pointer-events: none !important;
}
.aptiva-loader {
  /* True full-screen: covers sidebar, main area, and everything else */
  position: fixed; top: 0; left: 0;
  width: 100vw; height: 100vh;
  z-index: 9999999;
  background: #FFFFFF;
  display: flex; flex-direction: column; align-items: center;
  justify-content: center;
  padding: 2rem 1rem;
  animation: aptiva-fadein 0.3s ease;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  /* Capture all pointer events so nothing behind can be clicked */
  pointer-events: all;
  cursor: wait;
  /* Scroll lock via overflow */
  overflow: hidden;
}
.aptiva-logo {
  font-size:1.625rem; font-weight:800; color:#1D1D1F;
  letter-spacing:-0.035em; margin-bottom:0.375rem;
}
.aptiva-sub {
  font-size:0.8125rem; color:#86868B; letter-spacing:0.02em;
  margin-bottom:2.5rem;
}
/* Progress bar */
.aptiva-bar-track {
  width:min(480px,90vw); height:3px;
  background:#E5E5EA; border-radius:2px;
  overflow:hidden; margin-bottom:2rem;
}
.aptiva-bar-fill {
  height:100%; background:#0071E3;
  border-radius:2px;
  transition:width 0.8s cubic-bezier(0.4,0,0.2,1);
}
/* Pipeline list */
.aptiva-pipeline {
  width:min(400px,88vw); margin-bottom:1.5rem;
}
.aptiva-step {
  display:flex; align-items:baseline; gap:0.75rem;
  padding:0.3rem 0; font-size:0.875rem;
}
.aptiva-step-icon {
  width:1.125rem; text-align:center; flex-shrink:0;
  font-size:0.875rem; font-weight:600; line-height:1.4;
}
.aptiva-step-done  { color:#1A8917; }
.aptiva-step-active { color:#0071E3; }
.aptiva-step-idle  { color:#C7C7CC; }
.aptiva-step-label-done   { color:#1D1D1F; }
.aptiva-step-label-active {
  color:#0071E3; font-weight:600;
  animation:aptiva-label-pulse 2s ease-in-out infinite;
}
.aptiva-step-label-idle   { color:#C7C7CC; }
/* Active stage detail */
.aptiva-detail {
  font-size:0.8125rem; color:#6E6E73;
  text-align:center; min-height:1.25rem;
  margin-bottom:2rem; max-width:400px;
}
/* Stats grid */
.aptiva-stats {
  display:grid; grid-template-columns:repeat(5,1fr);
  gap:0.75rem; width:min(520px,92vw); margin-bottom:2rem;
}
.aptiva-stat {
  display:flex; flex-direction:column; align-items:center;
  background:#F5F5F7; border-radius:8px; padding:0.625rem 0.375rem;
}
.aptiva-stat-val {
  font-size:0.9375rem; font-weight:700; color:#1D1D1F;
  letter-spacing:-0.02em; line-height:1.2;
}
.aptiva-stat-label {
  font-size:0.625rem; color:#86868B; text-align:center;
  text-transform:uppercase; letter-spacing:0.07em;
  margin-top:0.25rem; line-height:1.3;
}
/* ETA block */
.aptiva-eta {
  text-align:center; margin-bottom:1.75rem; min-height:3.5rem;
}
.aptiva-eta-label {
  font-size:0.6875rem; color:#86868B; text-transform:uppercase;
  letter-spacing:0.1em; margin-bottom:0.25rem;
}
.aptiva-eta-value {
  font-size:1.5rem; font-weight:700; color:#1D1D1F;
  letter-spacing:-0.03em;
}
/* Footer */
.aptiva-footer {
  font-size:0.6875rem; color:#C7C7CC; text-align:center;
  letter-spacing:0.03em; padding-top:0.5rem;
  border-top:1px solid #F0F0F5; width:min(480px,90vw);
}
</style>
"""

    _PIPELINE = [
        ("Load Dataset",                  "Reading and validating 100,000 candidate profiles."),
        ("Build TF-IDF Index",            "Creating an 8,000-feature career similarity index."),
        ("Score Candidates",              "Running the 7-component ranking engine across all profiles."),
        ("Generate Ranking Explanations", "Creating transparent, fact-grounded ranking explanations."),
        ("Select Top 100",                "Selecting the highest-ranked AI/ML candidates by Final Score."),
    ]

    # stage_idx = which step is currently active (0-based).
    def _render(stage_idx: int, progress: float, eta: str, complete: bool = False):
        steps_html = ""
        for i, (label, _) in enumerate(_PIPELINE):
            if complete or i < stage_idx:
                icon_cls = "aptiva-step-done"
                lbl_cls  = "aptiva-step-label-done"
                # SVG check mark
                icon_ch  = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
            elif i == stage_idx:
                icon_cls = "aptiva-step-active"
                lbl_cls  = "aptiva-step-label-active"
                # SVG right-pointing triangle (play)
                icon_ch  = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3"/></svg>'
            else:
                icon_cls = "aptiva-step-idle"
                lbl_cls  = "aptiva-step-label-idle"
                # Small circle (pending)
                icon_ch  = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/></svg>'
            steps_html += (
                f'<div class="aptiva-step">'
                f'<span class="aptiva-step-icon {icon_cls}">{icon_ch}</span>'
                f'<span class="{lbl_cls}">{label}</span>'
                f'</div>'
            )

        if complete:
            detail = ""
        elif 0 <= stage_idx < len(_PIPELINE):
            detail = _PIPELINE[stage_idx][1]
        else:
            detail = ""

        bar_pct = min(100, int(progress * 100))

        if complete:
            _ck_green = '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1A8917" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
            eta_block = (
                f'<div class="aptiva-eta-label" style="color:#1A8917;letter-spacing:0.06em;display:flex;align-items:center;gap:0.3rem">'
                f'{_ck_green} Initialization Complete</div>'
                f'<div class="aptiva-eta-value" style="color:#1A8917;font-size:1.0625rem;font-weight:600">'
                f'Launching APTIVA AI…</div>'
            )
        elif eta:
            eta_block = (
                f'<div class="aptiva-eta-label">Estimated Remaining Time</div>'
                f'<div class="aptiva-eta-value">{eta}</div>'
            )
        else:
            eta_block = f'<div class="aptiva-eta-label">&nbsp;</div><div class="aptiva-eta-value">&nbsp;</div>'

        # Inject scroll lock JS in every frame (persists on DOM as long as overlay is visible)
        scroll_lock_js = """
<script>
(function(){
  document.documentElement.style.overflow='hidden';
  document.body.style.overflow='hidden';
  // Also lock Streamlit's own scroll container
  var main=document.querySelector('[data-testid="stAppViewContainer"]');
  if(main) main.style.overflow='hidden';
})();
</script>"""

        html = f"""{_CSS}
{scroll_lock_js}
<div class="aptiva-loader" id="aptiva-overlay">
  <div class="aptiva-logo"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-4px;margin-right:0.3rem"><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/></svg> APTIVA AI</div>
  <div class="aptiva-sub">Intelligent Candidate Ranking &middot; Redrob AI Hackathon</div>

  <div class="aptiva-bar-track">
    <div class="aptiva-bar-fill" style="width:{bar_pct}%"></div>
  </div>

  <div class="aptiva-pipeline">{steps_html}</div>

  <div class="aptiva-detail">{detail}</div>

  <div class="aptiva-stats">
    <div class="aptiva-stat">
      <div class="aptiva-stat-val">100,000</div>
      <div class="aptiva-stat-label">Dataset<br>Candidates</div>
    </div>
    <div class="aptiva-stat">
      <div class="aptiva-stat-val">7</div>
      <div class="aptiva-stat-label">Ranking Engine<br>Components</div>
    </div>
    <div class="aptiva-stat">
      <div class="aptiva-stat-val">8,000</div>
      <div class="aptiva-stat-label">Career Index<br>Features</div>
    </div>
    <div class="aptiva-stat">
      <div class="aptiva-stat-val">CPU Only</div>
      <div class="aptiva-stat-label">Execution<br>Mode</div>
    </div>
    <div class="aptiva-stat">
      <div class="aptiva-stat-val">&lt;5 min</div>
      <div class="aptiva-stat-label">Target<br>Runtime</div>
    </div>
  </div>

  <div class="aptiva-eta">{eta_block}</div>

  <div class="aptiva-footer">
    Deterministic Ranking &bull; Explainable AI &bull; CPU Only &bull; Fully Reproducible
  </div>
</div>"""
        loading_slot.markdown(html, unsafe_allow_html=True)

    # ── Mark as running (disables sidebar button) ─────────────────────────
    st.session_state["ranking_running"] = True

    # Stage 0 — dataset loading (renders before blocking call)
    _render(0, 0.05, "~190 seconds")
    time.sleep(0.08)
    # Stage 1 — TF-IDF indexing (bulk of wall-clock time)
    _render(1, 0.10, "~180 seconds")

    # ── blocking ranking call ─────────────────────────────────────────────
    result_data = None
    try:
        result_data = run_ranking(str(candidates_path))
    except Exception as exc:
        # ── Error state: show clean failure overlay ───────────────────────
        error_html = f"""{_CSS}
<script>
document.documentElement.style.overflow='hidden';
document.body.style.overflow='hidden';
</script>
<div class="aptiva-loader" id="aptiva-overlay">
  <div class="aptiva-logo">&#x2B21; APTIVA AI</div>
  <div class="aptiva-sub">Intelligent Candidate Ranking &middot; Redrob AI Hackathon</div>
  <div style="margin:2rem 0;text-align:center">
    <div style="font-size:2rem;margin-bottom:0.75rem">&#9888;</div>
    <div style="font-size:1rem;font-weight:600;color:#CC0000;margin-bottom:0.5rem">Initialization Failed</div>
    <div style="font-size:0.8125rem;color:#6E6E73;max-width:360px">{exc}</div>
  </div>
  <div style="margin-top:1rem;font-size:0.8125rem;color:#86868B;text-align:center">
    Click <strong>&#9654; Run Ranking Analysis</strong> in the sidebar to retry.
  </div>
  <div class="aptiva-footer" style="margin-top:2rem">
    Deterministic Ranking &bull; Explainable AI &bull; CPU Only &bull; Fully Reproducible
  </div>
</div>"""
        loading_slot.markdown(error_html, unsafe_allow_html=True)
        time.sleep(3.0)   # Show error for 3 s, then unlock UI
        # Unlock scroll before clearing
        loading_slot.markdown(
            '<script>document.documentElement.style.overflow="";'
            'document.body.style.overflow="";</script>',
            unsafe_allow_html=True,
        )
        time.sleep(0.1)
        loading_slot.empty()
        st.session_state["ranking_running"] = False
        st.session_state["ranking_error"]   = str(exc)
        return
    # ─────────────────────────────────────────────────────────────────────

    # Smooth sweep through remaining stages with interpolated progress.
    _render(2, 0.80, "~4 seconds");  time.sleep(0.12)
    _render(2, 0.85, "~3 seconds");  time.sleep(0.12)
    _render(3, 0.88, "~2 seconds");  time.sleep(0.12)
    _render(3, 0.92, "~1 second");   time.sleep(0.12)
    _render(4, 0.95, "~1 second");   time.sleep(0.12)
    _render(4, 0.98, "<1 second");   time.sleep(0.12)

    # Completion state -- shows check mark + Initialization Complete / Launching APTIVA AI
    _render(4, 1.00, "", complete=True)
    time.sleep(0.75)   # 700-800 ms dwell before clearing (per spec)

    # Fade-out: switch to aptiva-exiting class + unlock scroll simultaneously
    fadeout_html = f"""{_CSS}
<script>
document.documentElement.style.overflow='';
document.body.style.overflow='';
var main=document.querySelector('[data-testid="stAppViewContainer"]');
if(main) main.style.overflow='';
</script>
<div class="aptiva-loader aptiva-exiting" id="aptiva-overlay">
  <div class="aptiva-logo"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-4px;margin-right:0.3rem"><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/></svg> APTIVA AI</div>
  <div class="aptiva-sub">Intelligent Candidate Ranking &middot; Redrob AI Hackathon</div>
  <div class="aptiva-bar-track"><div class="aptiva-bar-fill" style="width:100%"></div></div>
  <div style="text-align:center;margin-top:2rem">
    <div style="font-size:0.6875rem;color:#1A8917;text-transform:uppercase;letter-spacing:0.06em;display:flex;align-items:center;justify-content:center;gap:0.3rem">
      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#1A8917" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Initialization Complete</div>
    <div style="font-size:1.0625rem;font-weight:600;color:#1A8917;margin-top:0.25rem">
      Launching APTIVA AI&hellip;</div>
  </div>
</div>"""
    loading_slot.markdown(fadeout_html, unsafe_allow_html=True)
    time.sleep(0.45)   # Match aptiva-fadeout animation duration (0.4s)
    loading_slot.empty()

    # ── Commit results to session state ───────────────────────────────────
    st.session_state["ranking_running"] = False
    if result_data is not None:
        st.session_state["results"]           = result_data.get("results", [])
        st.session_state["total_candidates"]  = result_data.get("total", 0)
        st.session_state["submission_csv"]    = result_data.get("submission_csv", "")
        st.session_state["ranking_done"]      = True
        # Auto-select top candidate if available
        if st.session_state["results"] and not st.session_state.get("selected_candidate_id"):
            st.session_state["selected_candidate_id"] = st.session_state["results"][0]["candidate"]["candidate_id"]



# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    inject_styles()
    init_state()
    config = load_config()
    loader = setup_dataset()

    # Sprint 4: Make config available to pages via session state.
    st.session_state["app_config"] = config

    # Auto-run ranking silently on first load
    auto_run_ranking(loader)

    # Sprint 6A: Retrieve recruiter memories once per active project load.
    # Fire-and-forget — Mem0 unavailability never blocks the UI.
    _refresh_recruiter_memories(config)

    # Render sidebar
    render_sidebar(config, loader)

    # Page Routing
    # Pass session state directly — pages read/write the live store.
    # (A copy dict would silently discard writes, breaking navigation context.)
    state = st.session_state

    # Auto-create a default project on first load if no projects exist
    if not st.session_state.get("projects"):
        default_project = HiringProject(
            project_name="Default Project",
            job_description=DEFAULT_JD,
            candidate_source="demo",
        )
        st.session_state["projects"][default_project.project_id] = default_project
        st.session_state["active_project"] = default_project

    # Route to page
    page = st.session_state["page"]

    if page == "home":
        home.render(state)
    elif page == "projects":
        projects_page.render(state)
    elif page == "candidate_sources":
        candidate_sources_page.render(state)
    elif page == "ai_analysis":
        ai_analysis.render(state)
    elif page == "candidate_profile":
        candidate_profile.render(state)
    elif page == "comparison":
        comparison.render(state)
    elif page == "judge_mode":
        judge_mode_page.render(state)
    elif page == "analytics":
        analytics.render(state)
    else:
        home.render(state)

    # Sync state back to session (for navigation mutations from pages)
    if state.get("page") != st.session_state["page"]:
        st.session_state["page"] = state["page"]
    if state.get("selected_candidate_id") != st.session_state["selected_candidate_id"]:
        st.session_state["selected_candidate_id"] = state["selected_candidate_id"]
    if state.get("compare_list") != st.session_state["compare_list"]:
        st.session_state["compare_list"] = state["compare_list"]


if __name__ == "__main__":
    main()


# ── Sprint 6A: Memory + Shortlist helpers ────────────────────────────────────────────────

def _refresh_recruiter_memories(config: dict) -> None:
    """
    Retrieve recruiter memories from Mem0 for the active project.

    Runs once per active project (keyed by project_id in session state).
    Results cached in st.session_state["recruiter_memories"].
    Fire-and-forget: silently returns on any error.
    """
    active_project = st.session_state.get("active_project")
    if active_project is None:
        return

    cache_key = f"_mem_fetched_{active_project.project_id}"
    if st.session_state.get(cache_key):
        return  # Already fetched for this project

    try:
        agent = RecruiterMemoryAgent(config)
        if not agent.is_configured():
            st.session_state["recruiter_memories"] = []
            st.session_state[cache_key] = True
            return

        jd    = active_project.job_description
        query = f"recruiter preferences for {jd.title} role with {' '.join((jd.core_skills or [])[:5])}"
        memories = agent.recall(query=query)
        if not memories:
            memories = agent.recall_all(limit=10)

        st.session_state["recruiter_memories"] = memories
        st.session_state[cache_key] = True

        # Also fire-and-forget: store project-created memory (idempotent via Mem0 dedup)
        agent.store_project_created(active_project)
    except Exception:   # noqa: BLE001
        st.session_state["recruiter_memories"] = []
        st.session_state[cache_key] = True


def _generate_shortlist_from_state(config: dict) -> None:
    """
    Generate the AI Shortlist from the current ranked results.

    Called immediately after any ranking pipeline completes.
    The shortlist is stored in st.session_state["shortlist"].
    The ranking order and scores are NEVER modified.
    """
    results  = st.session_state.get("results", [])
    memories = st.session_state.get("recruiter_memories", [])
    active_project = st.session_state.get("active_project")
    jd = active_project.job_description if active_project else None

    try:
        agent = ShortlistAgent()
        shortlist = agent.generate(
            results  = results,
            jd       = jd,
            memories = memories,
            top_n    = 5,
        )
        st.session_state["shortlist"] = shortlist
    except Exception:   # noqa: BLE001
        st.session_state["shortlist"] = []
