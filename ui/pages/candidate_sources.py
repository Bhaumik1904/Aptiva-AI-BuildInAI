"""
APTIVA AI — Candidate Sources Page
====================================
Select the candidate data source for the active Hiring Project.

Four source cards:
  1. Demo Dataset   — Available (immediate ranking via Demo Dataset path)
  2. CSV / Excel    — Available (upload, validate, normalise, rank)
  3. Resume Upload  — Disabled  (Coming in Sprint 4)
  4. ZIP Upload     — Disabled  (Coming in Sprint 4)

Sprint 3A:  Source selection + file acceptance.
Sprint 3B:  Full CSV/Excel ingestion, normalisation, validation UI,
            and ranking pipeline integration.
Sprint 4:   Resume / ZIP parsing.
"""

import streamlit as st

from core.csv_loader import CSVLoader, ExcelLoader
from core.models import HiringProject
from ui.icons import icon
from ui.styles import page_header, section_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_project(state) -> HiringProject:
    return state.get("active_project")


def _set_source(project: HiringProject, source: str, state: dict):
    """
    Store the chosen source on the project and reset ranking state so the
    next run picks up the new source cleanly.
    """
    project.candidate_source = source
    state["ranking_done"]          = False
    state["results"]               = []
    state["total_candidates"]      = 0
    state["submission_csv"]        = ""
    state["selected_candidate_id"] = None
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Card renderer
# ---------------------------------------------------------------------------

def _source_card(
    emoji: str,
    title: str,
    description: str,
    status: str,
    status_color: str,   # "green" | "grey"
    card_key: str,
    is_selected: bool,
    disabled: bool,
) -> bool:
    """
    Renders one source selection card and an action button below it.

    Returns True if the user clicked the Select button this frame.
    Disabled cards always return False.
    """
    # -- Visual tokens -------------------------------------------------------
    if disabled:
        bg        = "#F5F5F7"
        border    = "1px solid #E8E8ED"
        opacity   = "0.55"
        title_clr = "#86868B"
        desc_clr  = "#C7C7CC"
    elif is_selected:
        bg        = "#E8F2FF"
        border    = "2px solid #0071E3"
        opacity   = "1"
        title_clr = "#0071E3"
        desc_clr  = "#6E6E73"
    else:
        bg        = "#FFFFFF"
        border    = "1px solid #E8E8ED"
        opacity   = "1"
        title_clr = "#1D1D1F"
        desc_clr  = "#6E6E73"

    pill_bg  = "#EBF5EA" if status_color == "green" else "#EBEBED"
    pill_clr = "#1A8917" if status_color == "green" else "#86868B"

    active_badge = (
        '<span style="background:#0071E3;color:#fff;font-size:0.625rem;'
        'padding:0.125rem 0.5rem;border-radius:10px;font-weight:700;'
        'letter-spacing:0.03em;margin-left:0.5rem">ACTIVE</span>'
        if is_selected else ""
    )

    st.markdown(
        f"""
<div style="background:{bg};border:{border};border-radius:12px;
            padding:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.07);
            opacity:{opacity};min-height:168px;
            display:flex;flex-direction:column;gap:0.625rem">
  <div style="font-size:2rem;line-height:1">{emoji}</div>
  <div style="display:flex;align-items:center;gap:0.25rem;flex-wrap:wrap">
    <span style="font-size:1rem;font-weight:700;color:{title_clr};
                 letter-spacing:-0.01em">{title}</span>
    {active_badge}
  </div>
  <div style="font-size:0.8125rem;color:{desc_clr};line-height:1.55;
              flex:1">{description}</div>
  <div>
    <span style="background:{pill_bg};color:{pill_clr};font-size:0.6875rem;
                 font-weight:600;padding:0.2rem 0.6rem;border-radius:10px;
                 letter-spacing:0.03em">{status}</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    if disabled:
        st.markdown(
            '<div style="font-size:0.75rem;color:#C7C7CC;text-align:center;'
            'padding:0.3rem 0 0.6rem;font-style:italic">Coming in Sprint 4</div>',
            unsafe_allow_html=True,
        )
        return False

    label   = "Selected" if is_selected else "Select"
    clicked = st.button(
        label,
        key=f"src_card_{card_key}",
        use_container_width=True,
        disabled=is_selected,
    )
    return clicked


# ---------------------------------------------------------------------------
# CSV / Excel detail panel
# ---------------------------------------------------------------------------

def _render_csv_panel(project: HiringProject):
    """
    Shown beneath the CSV/Excel card when that source is active.
    Sprint 3B:
      - Stores raw file bytes on project.csv_file_bytes for the ranking pipeline.
      - Runs a preview parse (normalisation) to produce an IngestionReport.
      - Displays validation stats: Candidates Loaded, Rows Skipped,
        Successful Imports, Validation Errors.
    """
    st.markdown(
        """
<div style="background:#F0F7FF;border:1px solid #C8DEFF;border-radius:8px;
            padding:0.875rem 1.25rem;margin-top:0.625rem">
  <div style="font-size:0.8125rem;font-weight:600;color:#0071E3;
              margin-bottom:0.375rem">Upload Candidate Data File</div>
  <div style="font-size:0.8125rem;color:#6E6E73">
    Supported formats: <strong>CSV</strong>, <strong>Excel (.xlsx, .xls)</strong>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        key="csv_uploader",
        label_visibility="collapsed",
        help="Upload a CSV or Excel file containing candidate data.",
    )

    if uploaded is not None:
        ext      = uploaded.name.rsplit(".", 1)[-1].upper() if "." in uploaded.name else "UNKNOWN"
        size_kb  = uploaded.size / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"

        # ── Store metadata and bytes on the project ──────────────────────────
        project.csv_file_info = {
            "filename":  uploaded.name,
            "size_str":  size_str,
            "extension": ext,
        }
        project.csv_file_bytes = uploaded.getvalue()

        # ── Preview parse to get validation report ───────────────────────────
        with st.spinner("Validating file…"):
            try:
                if ext.lower() == "csv":
                    loader_inst = CSVLoader()
                else:
                    loader_inst = ExcelLoader()
                candidates, report = loader_inst.load(
                    project.csv_file_bytes, filename=uploaded.name
                )
                project.last_ingestion_report = report.to_dict()
            except Exception as exc:
                project.last_ingestion_report = {
                    "total_rows": 0,
                    "candidates_loaded": 0,
                    "rows_skipped": 0,
                    "successful_imports": 0,
                    "validation_errors": [str(exc)],
                }
                candidates = []

        rpt = project.last_ingestion_report

        # ── File metadata bar ────────────────────────────────────────────────
        st.markdown(
            f"""
<div style="background:#EBF5EA;border:1px solid #A8D5A2;border-radius:8px;
            padding:0.875rem 1.25rem;margin-top:0.75rem">
  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.625rem">
    <span style="color:#1A8917;font-size:1rem">&#10003;</span>
    <span style="font-size:0.875rem;font-weight:600;color:#1A8917">
      File Accepted &mdash; Validation Complete
    </span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.625rem;margin-bottom:0.625rem">
    <div style="background:#fff;border-radius:6px;padding:0.5rem 0.75rem">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:0.2rem">Filename</div>
      <div style="font-size:0.8125rem;font-weight:600;color:#1D1D1F;
                  word-break:break-all">{uploaded.name}</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:0.5rem 0.75rem">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:0.2rem">File Size</div>
      <div style="font-size:0.8125rem;font-weight:600;color:#1D1D1F">{size_str}</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:0.5rem 0.75rem">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:0.2rem">Extension</div>
      <div style="font-size:0.8125rem;font-weight:600;color:#0071E3">
        .{ext.lower()}
      </div>
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        # ── Ingestion report ─────────────────────────────────────────────────
        _render_ingestion_report(rpt)

    elif project.csv_file_bytes is not None:
        # Bytes already stored from a previous upload in this session
        info = project.csv_file_info or {}
        rpt  = project.last_ingestion_report or {}

        st.markdown(
            f"""
<div style="background:#F0F7FF;border:1px solid #C8DEFF;border-radius:8px;
            padding:0.875rem 1.25rem;margin-top:0.625rem">
  <div style="font-size:0.8125rem;font-weight:600;color:#0071E3;margin-bottom:0.25rem">
    File loaded and ready for ranking
  </div>
  <div style="font-size:0.8125rem;color:#6E6E73">
    <strong style="color:#1D1D1F">{info.get('filename','(unknown)')}</strong>
    &nbsp;&middot;&nbsp;{info.get('size_str','')}&nbsp;&middot;&nbsp;
    <span style="color:#0071E3">.{(info.get('extension','') or '').lower()}</span>
    &nbsp;&middot;&nbsp;
    <strong style="color:#1A8917">{rpt.get('successful_imports', '?')} candidates ready</strong>
  </div>
  <div style="font-size:0.75rem;color:#86868B;margin-top:0.25rem">
    Click <strong>&#9654; Run Ranking Analysis</strong> in the sidebar to rank these candidates.
    Upload a new file to replace.
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        if rpt:
            _render_ingestion_report(rpt)


def _render_ingestion_report(rpt: dict):
    """Render the 4-stat Ingestion Report panel."""
    if not rpt:
        return

    total_rows        = rpt.get("total_rows", 0)
    candidates_loaded = rpt.get("candidates_loaded", 0)
    rows_skipped      = rpt.get("rows_skipped", 0)
    successful        = rpt.get("successful_imports", 0)
    errors            = rpt.get("validation_errors", [])
    error_count       = len(errors)

    # Stat card colours
    skip_clr  = "#CC3300" if rows_skipped   > 0 else "#1D1D1F"
    err_clr   = "#CC3300" if error_count    > 0 else "#1D1D1F"

    st.markdown(
        f"""
<div style="border:1px solid #E8E8ED;border-radius:8px;
            padding:1rem 1.25rem;margin-top:0.75rem;background:#FAFAFA">
  <div style="font-size:0.6875rem;font-weight:700;color:#86868B;
              text-transform:uppercase;letter-spacing:0.08em;
              margin-bottom:0.75rem">Ingestion Report</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.625rem">
    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;
                border:1px solid #E8E8ED">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:0.25rem">Candidates Loaded</div>
      <div style="font-size:1.25rem;font-weight:700;color:#1D1D1F">{candidates_loaded:,}</div>
      <div style="font-size:0.6875rem;color:#86868B">of {total_rows:,} rows</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;
                border:1px solid #E8E8ED">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:0.25rem">Rows Skipped</div>
      <div style="font-size:1.25rem;font-weight:700;color:{skip_clr}">{rows_skipped:,}</div>
      <div style="font-size:0.6875rem;color:#86868B">duplicates removed</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;
                border:1px solid #E8E8ED">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:0.25rem">Successful Imports</div>
      <div style="font-size:1.25rem;font-weight:700;color:#1A8917">{successful:,}</div>
      <div style="font-size:0.6875rem;color:#86868B">ready for ranking</div>
    </div>
    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;
                border:1px solid #E8E8ED">
      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;
                  letter-spacing:0.06em;margin-bottom:0.25rem">Validation Errors</div>
      <div style="font-size:1.25rem;font-weight:700;color:{err_clr}">{error_count}</div>
      <div style="font-size:0.6875rem;color:#86868B">warnings</div>
    </div>
  </div>
  {_error_list_html(errors)}
</div>""",
        unsafe_allow_html=True,
    )


def _error_list_html(errors: list) -> str:
    """Return HTML for up to 10 validation error messages."""
    if not errors:
        return ""
    shown = errors[:10]
    items = "".join(
        f'<div style="font-size:0.75rem;color:#86868B;padding:0.2rem 0;'
        f'border-top:1px solid #F0F0F0">'
        f'<span style="color:#CC3300;margin-right:0.35rem">&#9888;</span>{e}</div>'
        for e in shown
    )
    more = (
        f'<div style="font-size:0.75rem;color:#C7C7CC;padding:0.2rem 0">'
        f'  … and {len(errors) - 10} more</div>'
        if len(errors) > 10 else ""
    )
    return (
        f'<div style="margin-top:0.75rem;border-top:1px solid #E8E8ED;'
        f'padding-top:0.5rem">{items}{more}</div>'
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(state: dict):
    """Render the Candidate Sources management page."""
    page_header(
        "Candidate Sources",
        "Choose how candidates are provided to the active Hiring Project.",
        icon("upload", 26),
    )

    project: HiringProject = _active_project(state)

    # -- Guard: no active project --------------------------------------------
    if project is None:
        st.markdown(
            """
<div style="text-align:center;padding:4rem 2rem;color:#86868B">
  <div style="font-size:3rem;margin-bottom:1rem">&#128194;</div>
  <div style="font-size:1.25rem;font-weight:600;color:#1D1D1F;margin-bottom:0.5rem">
    No Active Project
  </div>
  <div style="font-size:0.9375rem">
    Create or activate a Hiring Project before selecting a candidate source.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
        if st.button("Go to Hiring Projects", key="cs_goto_projects"):
            state["page"] = "projects"
            st.rerun()
        return

    current_source = project.candidate_source

    # -- Active project context banner ---------------------------------------
    source_labels = {
        "demo":   "Demo Dataset",
        "csv":    "CSV / Excel",
        "zip":    "ZIP Upload",
        "resume": "Resume Upload",
    }
    source_label = source_labels.get(current_source, current_source.upper())

    st.markdown(
        f"""
<div style="background:#F5F5F7;border:1px solid #E8E8ED;border-radius:8px;
            padding:0.75rem 1.25rem;margin-bottom:1.5rem;
            display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
  <div style="flex:1;min-width:200px">
    <div style="font-size:0.6875rem;color:#86868B;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.125rem">Active Project</div>
    <div style="font-size:0.9375rem;font-weight:600;color:#1D1D1F">
      {project.project_name}
    </div>
    <div style="font-size:0.8125rem;color:#6E6E73">{project.job_description.title}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:0.6875rem;color:#86868B;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.125rem">Current Source</div>
    <div style="font-size:0.875rem;font-weight:700;color:#0071E3">{source_label}</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    section_label("Select Candidate Source")

    # -- Row 1: Demo Dataset + CSV / Excel -----------------------------------
    col1, col2 = st.columns(2, gap="large")

    with col1:
        demo_clicked = _source_card(
            emoji="&#128640;",
            title="Demo Dataset",
            description=(
                "Use the built-in sample candidate dataset for instant analysis. "
                "No upload required &mdash; 100,000 profiles ready to rank immediately."
            ),
            status="Available",
            status_color="green",
            card_key="demo",
            is_selected=(current_source == "demo"),
            disabled=False,
        )
        if demo_clicked:
            _set_source(project, "demo", state)
            st.rerun()

        if current_source == "demo":
            st.markdown(
                """
<div style="background:#EBF5EA;border:1px solid #A8D5A2;border-radius:8px;
            padding:0.875rem 1.25rem;margin-top:0.5rem">
  <div style="font-size:0.8125rem;font-weight:600;color:#1A8917;margin-bottom:0.25rem">
    Demo Dataset Active
  </div>
  <div style="font-size:0.8125rem;color:#6E6E73;line-height:1.5">
    100,000 candidate profiles loaded and ready.<br>
    Head to <strong>Rankings</strong> and click
    <strong>&#9654; Run Ranking Analysis</strong> to start.
  </div>
</div>""",
                unsafe_allow_html=True,
            )

    with col2:
        csv_clicked = _source_card(
            emoji="&#128202;",
            title="CSV / Excel Upload",
            description=(
                "Upload candidate data using a CSV or Excel file. "
                "Supports .csv, .xlsx, and .xls formats."
            ),
            status="Available",
            status_color="green",
            card_key="csv",
            is_selected=(current_source == "csv"),
            disabled=False,
        )
        if csv_clicked:
            _set_source(project, "csv", state)
            st.rerun()

        if current_source == "csv":
            _render_csv_panel(project)

    # Spacer between rows
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # -- Row 2: Resume Upload + ZIP Upload (disabled) ------------------------
    col3, col4 = st.columns(2, gap="large")

    with col3:
        _source_card(
            emoji="&#128196;",
            title="Resume Upload",
            description=(
                "Upload multiple PDF or DOCX resumes. Each resume will be parsed "
                "and ranked automatically using AI extraction."
            ),
            status="Coming in Sprint 4",
            status_color="grey",
            card_key="resume",
            is_selected=False,
            disabled=True,
        )

    with col4:
        _source_card(
            emoji="&#128230;",
            title="ZIP Upload",
            description=(
                "Upload a ZIP archive containing multiple resumes. "
                "Bulk processing with automatic extraction and parsing."
            ),
            status="Coming in Sprint 4",
            status_color="grey",
            card_key="zip",
            is_selected=False,
            disabled=True,
        )

    # -- Contextual hint when demo is active ---------------------------------
    if current_source == "demo":
        st.markdown("---")
        st.markdown(
            """
<div style="background:#F0F7FF;border:1px solid #C8DEFF;border-radius:8px;
            padding:0.875rem 1.25rem;display:flex;align-items:flex-start;gap:1rem">
  <div style="font-size:1.25rem;flex-shrink:0">&#128161;</div>
  <div style="font-size:0.8125rem;color:#1D1D1F;line-height:1.5">
    <strong>Demo Dataset selected.</strong> Navigate to
    <strong>Rankings</strong> in the sidebar and click
    <strong>&#9654; Run Ranking Analysis</strong> to score all 100,000 candidates.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
