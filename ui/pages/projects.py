"""
APTIVA AI -- Projects Page
===========================
Create, manage, select, rename, and delete Hiring Projects.
Each project holds a JobDescription and all ranking state.

This page is the entry point for every new hiring flow.
"""

import streamlit as st

from agents.jd_agent import JDIntelligenceAgent
from agents.memory_agent import RecruiterMemoryAgent
from core.models import DEFAULT_JD, HiringProject, JobDescription
from ui.icons import icon
from ui.styles import page_header, section_label


# ---------------------------------------------------------------------------
# AI Panel
# ---------------------------------------------------------------------------

_AI_PANEL_STYLE = """
<style>
.ai-panel {
  background: linear-gradient(135deg, #F0F7FF 0%, #F5F5F7 100%);
  border: 1px solid #C0D9F5;
  border-radius: 12px;
  padding: 1.125rem 1.25rem 0.875rem;
  margin-bottom: 1.25rem;
}
.ai-panel-header {
  font-size: 0.9375rem;
  font-weight: 700;
  color: #0071E3;
  letter-spacing: -0.015em;
  margin-bottom: 0.25rem;
}
.ai-panel-sub {
  font-size: 0.8125rem;
  color: #6E6E73;
  margin-bottom: 0.875rem;
}
.ai-success-card {
  background: #F0FBF0;
  border: 1px solid #C3EAC3;
  border-radius: 8px;
  padding: 0.875rem 1rem;
  margin-top: 0.75rem;
  margin-bottom: 0.5rem;
}
.ai-success-title {
  font-size: 0.8rem;
  font-weight: 700;
  color: #1A8917;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 0.5rem;
}
.ai-step {
  font-size: 0.8125rem;
  color: #1D1D1F;
  display: flex;
  align-items: flex-start;
  gap: 0.4rem;
  margin-bottom: 0.275rem;
  line-height: 1.4;
}
.ai-step-icon { color: #1A8917; font-weight: 700; flex-shrink: 0; }
.ai-error-card {
  background: #FFF5F5;
  border: 1px solid #F5C0C0;
  border-radius: 8px;
  padding: 0.875rem 1rem;
  margin-top: 0.75rem;
}
.ai-error-title {
  font-size: 0.8rem;
  font-weight: 700;
  color: #CC0000;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 0.375rem;
}
.ai-error-msg {
  font-size: 0.8125rem;
  color: #4A1515;
  word-break: break-word;
}
.ai-model-badge {
  display: inline-block;
  font-size: 0.6875rem;
  background: #E8F2FF;
  color: #0071E3;
  border-radius: 4px;
  padding: 0.1rem 0.4rem;
  font-weight: 600;
  margin-left: 0.4rem;
  vertical-align: middle;
}
</style>
"""


def _render_ai_panel(project: HiringProject, state: dict) -> None:
    """
    AI-powered JD extraction panel.
    Rendered above the manual JD form — adds AI without touching the form.

    Flow:
      1. API key check — show warning and return early if not configured.
      2. Text area for raw JD paste (persisted via session state key per project).
      3. ✨ Analyze button — calls JDIntelligenceAgent, pre-fills form widget
         state keys so the form below shows AI values on the same render pass.
      4. ✕ Clear button — removes summary/error; preserves recruiter form edits.
      5. Compact summary card (✓ steps) shown after successful analysis.
      6. Error card shown if analysis failed; text preserved; retry available.
    """
    pid        = project.project_id
    steps_key  = f"jd_ai_steps_{pid}"
    error_key  = f"jd_ai_error_{pid}"
    raw_key    = f"jd_raw_text_{pid}"

    config = state.get("app_config", {})
    agent  = JDIntelligenceAgent(config)

    # Inject panel CSS (idempotent)
    st.markdown(_AI_PANEL_STYLE, unsafe_allow_html=True)

    # ── Panel header ─────────────────────────────────────────────────────────
    model_badge = f'<span class="ai-model-badge">{agent.model_name}</span>'
    st.markdown(
        f'<div class="ai-panel">'
        f'<div class="ai-panel-header">✨ AI-Powered JD Analysis{model_badge}</div>'
        f'<div class="ai-panel-sub">'
        f'Paste your full job description — Gemini extracts all fields automatically.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── API key warning — exit early if not configured ────────────────────────
    if not agent.is_configured():
        st.warning(
            "**Gemini API key not configured.**  "
            "Set `gemini_api_key` in `config.yaml` or export `GEMINI_API_KEY`.  "
            "The manual form below is still available.",
            icon="⚠️",
        )
        st.markdown("---")
        return

    # ── Raw JD text area ─────────────────────────────────────────────────────
    jd_text = st.text_area(
        "Paste Job Description",
        key=raw_key,
        height=180,
        placeholder=(
            "Paste any job description here — any format, any length.\n\n"
            'Example: "We are looking for a Senior Backend Engineer with 5+ years '
            'of Python experience, strong knowledge of FastAPI, Docker, '
            'PostgreSQL and AWS..."'
        ),
        label_visibility="collapsed",
    )

    # ── Button row ───────────────────────────────────────────────────────────
    col_btn1, col_btn2, col_spacer = st.columns([2, 1, 3])
    with col_btn1:
        analyze_clicked = st.button(
            "✨ Analyze with AI",
            type="primary",
            use_container_width=True,
            key=f"ai_analyze_{pid}",
        )
    with col_btn2:
        clear_clicked = st.button(
            "✕ Clear",
            type="secondary",
            use_container_width=True,
            key=f"ai_clear_{pid}",
        )

    # ── Handle clear ─────────────────────────────────────────────────────────
    if clear_clicked:
        state.pop(steps_key, None)
        state.pop(error_key, None)
        # Remove the text area value so it resets to empty
        st.session_state.pop(raw_key, None)
        st.rerun()

    # ── Handle analyze ───────────────────────────────────────────────────────
    if analyze_clicked:
        raw_input = (jd_text or "").strip()
        if not raw_input:
            st.error("Please paste a Job Description before clicking Analyze.")
        else:
            with st.spinner(
                f"Analyzing with {agent.model_name}…  "
                "Extracting skills · Experience · Keywords"
            ):
                try:
                    jd_result, steps = agent.analyze(raw_input)
                    # Store steps for summary card
                    state[steps_key] = steps
                    state[error_key] = None
                    # ── Pre-fill form widget state ────────────────────────────
                    # Set each session_state key that the existing _render_jd_form
                    # widgets use. Since the form renders BELOW this panel in the
                    # same script run, the widgets pick up these values immediately.
                    st.session_state[f"jd_title_{pid}"]    = jd_result.title
                    st.session_state[f"jd_desc_{pid}"]     = jd_result.description
                    st.session_state[f"jd_resp_{pid}"]     = jd_result.responsibilities
                    st.session_state[f"jd_core_{pid}"]     = "\n".join(jd_result.core_skills)
                    st.session_state[f"jd_bonus_{pid}"]    = "\n".join(jd_result.bonus_skills)
                    st.session_state[f"exp_min_{pid}"]     = int(jd_result.experience_target_min)
                    st.session_state[f"exp_max_{pid}"]     = int(jd_result.experience_target_max)
                    st.session_state[f"sweet_min_{pid}"]   = int(jd_result.experience_sweet_spot_min)
                    st.session_state[f"sweet_max_{pid}"]   = int(jd_result.experience_sweet_spot_max)
                    st.session_state[f"jd_locations_{pid}"] = ", ".join(
                        jd_result.preferred_locations
                    )
                except (ValueError, RuntimeError) as exc:
                    state[steps_key] = None
                    state[error_key] = str(exc)
                except Exception as exc:  # noqa: BLE001
                    state[steps_key] = None
                    state[error_key] = (
                        f"Unexpected error during analysis: {exc}. "
                        "Please try again."
                    )

    # ── Compact success summary card ──────────────────────────────────────────
    steps = state.get(steps_key)
    if steps:
        step_html = "".join(
            f'<div class="ai-step">'
            f'<span class="ai-step-icon">✓</span>'
            f'<span>{_md_bold_to_html(step)}</span>'
            f'</div>'
            for step in steps
        )
        st.markdown(
            f'<div class="ai-success-card">'
            f'<div class="ai-success-title">✓ Analysis Complete</div>'
            f'{step_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Error card ───────────────────────────────────────────────────────────
    error = state.get(error_key)
    if error:
        st.markdown(
            f'<div class="ai-error-card">'
            f'<div class="ai-error-title">⚠ Analysis Failed</div>'
            f'<div class="ai-error-msg">{error}</div>'
            f'<div class="ai-error-msg" style="margin-top:0.5rem;color:#6E6E73">'
            f'Your pasted text is preserved. Edit the form manually or click '
            f'<strong>✨ Analyze with AI</strong> to retry.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")


def _md_bold_to_html(text: str) -> str:
    """Convert **bold** markdown to <strong> tags for inline HTML rendering."""
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_project(state) -> HiringProject:
    return state.get("active_project")


def _set_active(state, project: HiringProject):
    state["active_project"] = project
    # Switching projects invalidates previous ranking results
    state["ranking_done"] = False
    state["results"] = []
    state["total_candidates"] = 0
    state["submission_csv"] = ""
    state["selected_candidate_id"] = None
    st.cache_data.clear()


def _create_project(state, name: str) -> HiringProject:
    """Create a new project with the default JD and add it to session state."""
    from core.jd_config import JD_CONFIG
    jd = JobDescription.from_dict(JD_CONFIG)
    p = HiringProject(project_name=name, job_description=jd)
    state["projects"][p.project_id] = p
    return p


# ---------------------------------------------------------------------------
# JD Form
# ---------------------------------------------------------------------------

def _render_jd_form(project: HiringProject, state):
    """Render an editable form for the active project's Job Description."""
    jd = project.job_description

    st.markdown("---")
    section_label("Job Description")

    new_title = st.text_input(
        "Job Title",
        value=jd.title,
        key=f"jd_title_{project.project_id}",
        placeholder="e.g. Senior AI Engineer",
    )

    new_description = st.text_area(
        "Job Description (optional)",
        value=jd.description,
        height=100,
        key=f"jd_desc_{project.project_id}",
        placeholder="Paste or type a brief job description...",
    )

    new_responsibilities = st.text_area(
        "Responsibilities (optional)",
        value=jd.responsibilities,
        height=80,
        key=f"jd_resp_{project.project_id}",
        placeholder="Key responsibilities...",
    )

    # Skills
    st.markdown("---")
    section_label("Skills Configuration")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        new_core = st.text_area(
            "Core Required Skills (one per line)",
            value="\n".join(jd.core_skills),
            height=200,
            key=f"jd_core_{project.project_id}",
        )
    with col_s2:
        new_bonus = st.text_area(
            "Bonus / Nice-to-Have Skills (one per line)",
            value="\n".join(jd.bonus_skills),
            height=200,
            key=f"jd_bonus_{project.project_id}",
        )

    # Experience
    st.markdown("---")
    section_label("Experience & Location")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        new_exp_min = st.number_input("Min Experience (yrs)", value=jd.experience_target_min, min_value=0, max_value=30, key=f"exp_min_{project.project_id}")
    with col_e2:
        new_exp_max = st.number_input("Max Experience (yrs)", value=jd.experience_target_max, min_value=0, max_value=30, key=f"exp_max_{project.project_id}")
    with col_e3:
        new_sweet_min = st.number_input("Sweet Spot Min (yrs)", value=jd.experience_sweet_spot_min, min_value=0, max_value=30, key=f"sweet_min_{project.project_id}")
    with col_e4:
        new_sweet_max = st.number_input("Sweet Spot Max (yrs)", value=jd.experience_sweet_spot_max, min_value=0, max_value=30, key=f"sweet_max_{project.project_id}")

    new_locations = st.text_input(
        "Preferred Locations (comma-separated)",
        value=", ".join(jd.preferred_locations),
        key=f"jd_locations_{project.project_id}",
        placeholder="bangalore, mumbai, pune, delhi",
    )

    # Save button
    st.markdown("---")
    col_save, col_cancel = st.columns([1, 5])
    with col_save:
        if st.button("Save & Activate", type="primary", use_container_width=True, key=f"save_jd_{project.project_id}"):
            # Apply changes to JD
            project.job_description.title           = new_title.strip() or jd.title
            project.job_description.description     = new_description.strip()
            project.job_description.responsibilities = new_responsibilities.strip()
            project.job_description.core_skills     = [s.strip() for s in new_core.splitlines() if s.strip()]
            project.job_description.bonus_skills    = [s.strip() for s in new_bonus.splitlines() if s.strip()]
            project.job_description.experience_target_min     = int(new_exp_min)
            project.job_description.experience_target_max     = int(new_exp_max)
            project.job_description.experience_sweet_spot_min = int(new_sweet_min)
            project.job_description.experience_sweet_spot_max = int(new_sweet_max)
            project.job_description.preferred_locations = [l.strip() for l in new_locations.split(",") if l.strip()]

            _set_active(state, project)
            # Sprint 6A: fire-and-forget memory storage (never blocks UI)
            try:
                _mem_agent = RecruiterMemoryAgent(state.get("app_config", {}))
                _mem_agent.store_jd_saved(project.job_description)
                # Clear the per-project memory cache so recall refreshes on next load
                _mem_cache_key = f"_mem_fetched_{project.project_id}"
                if _mem_cache_key in st.session_state:
                    del st.session_state[_mem_cache_key]
            except Exception:   # noqa: BLE001
                pass
            st.success(f"Project saved. Ranking will re-run with the updated JD for '{project.job_description.title}'.")
            st.rerun()


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(state):
    """Render the Hiring Projects management page."""
    page_header(
        "Hiring Projects",
        "Create and manage hiring searches. Each project has its own Job Description and candidate ranking.",
        icon("folder", 26),
    )

    projects: dict = state.get("projects", {})
    active_project = _active_project(state)

    # ── Empty State ────────────────────────────────────────────────────────
    if not projects:
        st.markdown(
            """
            <div style="text-align:center;padding:4rem 2rem;color:#86868B">
              <div style="font-size:3rem;margin-bottom:1rem">📂</div>
              <div style="font-size:1.25rem;font-weight:600;color:#1D1D1F;margin-bottom:0.5rem">No Hiring Projects Yet</div>
              <div style="font-size:0.9375rem">Create your first Hiring Project to start ranking candidates.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c, _, _ = st.columns([1, 2, 1])
        with col_c:
            if st.button("+ Create First Project", type="primary", use_container_width=True, key="empty_create_btn"):
                new_p = _create_project(state, "My First Project")
                _set_active(state, new_p)
                st.rerun()
        return

    # ── Project List ───────────────────────────────────────────────────────
    col_list, col_detail = st.columns([1, 2], gap="large")

    with col_list:
        section_label("Your Projects")

        # New project form
        with st.expander("+ New Project", expanded=False):
            new_name = st.text_input(
                "Project Name",
                placeholder="e.g. Q3 AI Hiring",
                key="new_project_name_input",
            )
            if st.button("Create", type="primary", key="create_project_btn"):
                if new_name.strip():
                    new_p = _create_project(state, new_name.strip())
                    _set_active(state, new_p)
                    st.success(f"Project '{new_p.project_name}' created and activated.")
                    st.rerun()
                else:
                    st.error("Please enter a project name.")

        st.markdown("---")

        # Project cards
        for pid, project in list(projects.items()):
            is_active = (active_project is not None and active_project.project_id == pid)
            bg = "#E8F2FF" if is_active else "#F5F5F7"
            border = "2px solid #0071E3" if is_active else "1px solid transparent"
            badge = '<span style="background:#0071E3;color:#fff;font-size:0.625rem;padding:0.125rem 0.4rem;border-radius:4px;margin-left:0.4rem;font-weight:600">ACTIVE</span>' if is_active else ""

            st.markdown(
                f'<div style="background:{bg};border:{border};border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem">'
                f'  <div style="font-size:0.875rem;font-weight:600;color:#1D1D1F">{project.project_name}{badge}</div>'
                f'  <div style="font-size:0.75rem;color:#86868B">{project.job_description.title}</div>'
                f'  <div style="font-size:0.6875rem;color:#C7C7CC;margin-top:0.25rem">Created {project.created_at} · {project.candidate_source.upper()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("Select", key=f"select_{pid}", use_container_width=True, disabled=is_active):
                    _set_active(state, project)
                    st.rerun()
            with btn_col2:
                rename_key = f"renaming_{pid}"
                if st.button("Rename", key=f"rename_btn_{pid}", use_container_width=True):
                    state[rename_key] = True
                    st.rerun()
                if state.get(rename_key):
                    new_n = st.text_input("New name", key=f"rename_input_{pid}", value=project.project_name)
                    if st.button("OK", key=f"rename_ok_{pid}"):
                        project.project_name = new_n.strip() or project.project_name
                        state[rename_key] = False
                        st.rerun()
            with btn_col3:
                if st.button("Delete", key=f"delete_{pid}", use_container_width=True, type="secondary"):
                    del state["projects"][pid]
                    if active_project and active_project.project_id == pid:
                        remaining = list(state["projects"].values())
                        state["active_project"] = remaining[0] if remaining else None
                        state["ranking_done"] = False
                        state["results"] = []
                    st.rerun()

    # ── Project Detail (JD Editor) ─────────────────────────────────────────
    with col_detail:
        if active_project:
            section_label(f"Configure JD — {active_project.project_name}")
            _render_ai_panel(active_project, state)
            _render_jd_form(active_project, state)
        else:
            st.info("Select a project on the left to configure its Job Description.")