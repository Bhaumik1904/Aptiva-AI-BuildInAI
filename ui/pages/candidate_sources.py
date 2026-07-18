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

from agents.resume_agent import ResumeIntelligenceAgent
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
# Resume Upload detail panel
# ---------------------------------------------------------------------------

_RESUME_CARD_CSS = """
<style>
.resume-step-card {
  background:#F0FBF0;border:1px solid #C3EAC3;border-radius:8px;
  padding:0.75rem 1rem;margin-top:0.5rem;
}
.resume-step-title {
  font-size:0.75rem;font-weight:700;color:#1A8917;
  text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem;
}
.resume-step-row {
  font-size:0.8rem;color:#1D1D1F;
  display:flex;align-items:flex-start;gap:0.35rem;
  margin-bottom:0.2rem;line-height:1.4;
}
.resume-step-icon { color:#1A8917;font-weight:700;flex-shrink:0; }
.resume-err-card {
  background:#FFF5F5;border:1px solid #F5C0C0;border-radius:8px;
  padding:0.75rem 1rem;margin-top:0.5rem;
}
.resume-err-title {
  font-size:0.75rem;font-weight:700;color:#CC0000;
  text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.3rem;
}
.resume-err-msg { font-size:0.8rem;color:#4A1515;word-break:break-word; }
.resume-meta-bar {
  background:#F5F5F7;border:1px solid #E8E8ED;border-radius:6px;
  padding:0.5rem 0.875rem;font-size:0.8rem;color:#6E6E73;
  margin-top:0.375rem;
}
</style>
"""


def _render_resume_panel(project: HiringProject, state: dict) -> None:
    """
    Resume Upload pipeline panel.

    Flow:
      1. API key check — warn and return early if not configured.
      2. Multi-file uploader (PDF/DOCX).
      3. Per-file Analyze button that calls ResumeIntelligenceAgent.
      4. ✓ step card per processed file.
      5. Error card per failed file (file preserved, retry available).
      6. Persistent results bar showing total accumulated candidates.
      7. Clear button to reset all resume candidates for this project.
    """
    import re

    pid       = project.project_id
    cfg       = state.get("app_config", {})
    agent     = ResumeIntelligenceAgent(cfg)

    st.markdown(_RESUME_CARD_CSS, unsafe_allow_html=True)

    # -- Header ----------------------------------------------------------------
    model_badge = (
        f'<span style="display:inline-block;font-size:0.6875rem;'
        f'background:#E8F2FF;color:#0071E3;border-radius:4px;'
        f'padding:0.1rem 0.4rem;font-weight:600;margin-left:0.4rem;'
        f'vertical-align:middle">{agent.model_name}</span>'
    )
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#F0F7FF 0%,#F5F5F7 100%);'
        f'border:1px solid #C0D9F5;border-radius:10px;padding:1rem 1.25rem 0.875rem;'
        f'margin-bottom:0.75rem">'
        f'<div style="font-size:0.9375rem;font-weight:700;color:#0071E3;'
        f'letter-spacing:-0.015em">'
        f'\U0001f4c4 AI Resume Analysis{model_badge}</div>'
        f'<div style="font-size:0.8125rem;color:#6E6E73;margin-top:0.2rem">'
        f'Upload PDF or DOCX resumes — Gemini extracts candidate profiles automatically.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # -- API key guard ---------------------------------------------------------
    if not agent.is_configured():
        st.warning(
            "**Gemini API key not configured.** "
            "Set `gemini_api_key` in `config.yaml` or export `GEMINI_API_KEY`.",
            icon="\u26a0\ufe0f",
        )
        return

    # -- Multi-file uploader ---------------------------------------------------
    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"resume_uploader_{pid}",
        label_visibility="collapsed",
        help="Upload one or more PDF or DOCX resume files.",
    )

    # -- Analyze button --------------------------------------------------------
    col_btn, col_clear, col_space = st.columns([2, 1, 3])
    with col_btn:
        analyze_clicked = st.button(
            "\u2728 Analyze Resumes",
            type="primary",
            use_container_width=True,
            key=f"resume_analyze_{pid}",
            disabled=not bool(uploaded_files),
        )
    with col_clear:
        clear_clicked = st.button(
            "\u2715 Clear All",
            type="secondary",
            use_container_width=True,
            key=f"resume_clear_{pid}",
        )

    if clear_clicked:
        project.resume_candidates = None
        project.resume_file_infos = None
        state.pop(f"resume_results_{pid}", None)
        st.rerun()

    # -- Run analysis on click -------------------------------------------------
    if analyze_clicked and uploaded_files:
        new_candidates = list(project.resume_candidates or [])
        new_file_infos = list(project.resume_file_infos or [])

        for uf in uploaded_files:
            file_bytes = uf.getvalue()
            filename   = uf.name
            size_kb    = uf.size / 1024
            size_str   = (
                f"{size_kb:.1f} KB" if size_kb < 1024
                else f"{size_kb / 1024:.2f} MB"
            )

            with st.spinner(f"Analyzing `{filename}`\u2026"):
                try:
                    candidate, steps = agent.analyze(file_bytes, filename)
                    new_candidates.append(candidate)
                    new_file_infos.append({
                        "filename": filename,
                        "size_str": size_str,
                        "steps":    steps,
                        "error":    None,
                        "candidate_id": candidate.get("candidate_id"),
                        "name":     candidate.get("profile", {}).get("anonymized_name"),
                        "yoe":      candidate.get("profile", {}).get("years_of_experience"),
                        "n_skills": len(candidate.get("skills", [])),
                        "n_certs":  len(candidate.get("certifications", [])),
                    })
                except Exception as exc:  # noqa: BLE001
                    new_file_infos.append({
                        "filename": filename,
                        "size_str": size_str,
                        "steps":    [],
                        "error":    str(exc),
                        "candidate_id": None,
                        "name":     None,
                        "yoe":      None,
                        "n_skills": 0,
                        "n_certs":  0,
                    })

        project.resume_candidates = new_candidates
        project.resume_file_infos = new_file_infos
        # Reset ranking so new candidates will be re-ranked
        state["ranking_done"] = False
        state["results"]      = []

    # -- Display results -------------------------------------------------------
    file_infos = project.resume_file_infos or []
    if file_infos:
        st.markdown(
            f'<div style="font-size:0.6875rem;font-weight:700;color:#86868B;'
            f'text-transform:uppercase;letter-spacing:0.08em;'
            f'margin:0.75rem 0 0.4rem">Results — {len(file_infos)} file(s) processed</div>',
            unsafe_allow_html=True,
        )
        for fi in file_infos:
            fname = fi.get("filename", "")
            ssize = fi.get("size_str", "")
            error = fi.get("error")

            if error:
                # Error card
                st.markdown(
                    f'<div class="resume-err-card">'
                    f'<div class="resume-err-title">\u26a0 Failed: {fname}</div>'
                    f'<div class="resume-err-msg">{error}</div>'
                    f'<div class="resume-err-msg" style="margin-top:0.35rem;color:#6E6E73">'
                    f'The file is preserved. Fix the issue and click '
                    f'<strong>\u2728 Analyze Resumes</strong> to retry.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Success step card
                steps = fi.get("steps", [])

                def _bold(t: str) -> str:
                    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)

                step_html = "".join(
                    f'<div class="resume-step-row">'
                    f'<span class="resume-step-icon">\u2713</span>'
                    f'<span>{_bold(s)}</span></div>'
                    for s in steps
                )
                meta_bar = (
                    f'<div class="resume-meta-bar">'
                    f'ID: <code>{fi.get("candidate_id","?")}</code>'
                    f' &nbsp;&middot;&nbsp; {ssize}'
                    f' &nbsp;&middot;&nbsp; source: <strong>resume</strong>'
                    f'</div>'
                )
                st.markdown(
                    f'<div class="resume-step-card">'
                    f'<div class="resume-step-title">\u2713 {fname}</div>'
                    f'{step_html}'
                    f'{meta_bar}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # -- Total candidates bar --------------------------------------------------
    total = len(project.resume_candidates or [])
    if total > 0:
        st.markdown(
            f'<div style="background:#EBF5EA;border:1px solid #A8D5A2;'
            f'border-radius:8px;padding:0.75rem 1.25rem;margin-top:0.75rem;'
            f'font-size:0.8125rem;color:#1A8917;font-weight:600">'
            f'\u2713 {total} candidate profile{"s" if total != 1 else ""} ready for ranking.'
            f'<span style="color:#6E6E73;font-weight:400"> '
            f'Click <strong>&#9654; Run Ranking Analysis</strong> in the sidebar.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def _render_zip_panel(project: HiringProject, state: dict) -> None:
    st.markdown("### ZIP Archive Upload")
    st.markdown("Upload a ZIP file containing PDF or DOCX resumes.")
    
    config = state.get("app_config", {})
    agent = ResumeIntelligenceAgent(config)
    
    if not agent.is_configured():
        st.error("⚠️ **Gemini API Key missing.** Configure it in Settings to enable resume parsing.")
        return
        
    uploaded_file = st.file_uploader(
        "Upload a ZIP archive",
        type=["zip"],
        accept_multiple_files=False,
        key="zip_upload"
    )
    
    if uploaded_file:
        from core.zip_loader import extract_resumes_from_zip
        from collections import Counter
        
        st.markdown("#### Processing Archive...")
        
        with st.spinner("Extracting files..."):
            zip_bytes = uploaded_file.getvalue()
            successful_files, skipped_files = extract_resumes_from_zip(zip_bytes)
            
        total_discovered = len(successful_files) + len(skipped_files)
        
        if total_discovered == 0:
            st.warning("No supported files (PDF, DOCX) found in the archive.")
            return
            
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        new_candidates = []
        new_file_infos = []
        successful_count = 0
        all_skills = []
        all_yoe = []
        
        for idx, (file_bytes, filename, relative_path) in enumerate(successful_files):
            progress = (idx) / len(successful_files)
            progress_bar.progress(progress)
            progress_text.text(f"Analyzing {idx+1}/{len(successful_files)}: {filename}")
            
            size_kb = len(file_bytes) / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
            
            try:
                candidate, steps = agent.analyze(file_bytes, filename)
                new_candidates.append(candidate)
                new_file_infos.append({
                    "filename": relative_path,
                    "size_str": size_str,
                    "steps":    steps,
                    "error":    None,
                    "candidate_id": candidate.get("candidate_id"),
                    "name":     candidate.get("profile", {}).get("anonymized_name"),
                    "yoe":      candidate.get("profile", {}).get("years_of_experience"),
                    "n_skills": len(candidate.get("skills", [])),
                    "n_certs":  len(candidate.get("certifications", [])),
                })
                successful_count += 1
                all_skills.extend(candidate.get("skills", []))
                yoe = candidate.get("profile", {}).get("years_of_experience")
                if yoe is not None and isinstance(yoe, (int, float)):
                    all_yoe.append(yoe)
            except Exception as exc:
                new_file_infos.append({
                    "filename": relative_path,
                    "size_str": size_str,
                    "steps":    [],
                    "error":    str(exc),
                    "candidate_id": None,
                    "name":     None,
                    "yoe":      None,
                    "n_skills": 0,
                    "n_certs":  0,
                })
                skipped_files.append((relative_path, str(exc)))
                
        progress_text.empty()
        progress_bar.empty()

        project.resume_candidates = new_candidates
        project.resume_file_infos = new_file_infos
        
        top_skills = [s for s, c in Counter(all_skills).most_common(5)]
        avg_yoe = round(sum(all_yoe) / len(all_yoe), 1) if all_yoe else 0
        
        setattr(project, "zip_ingestion_summary", {
            "total_discovered": total_discovered,
            "processed_successfully": successful_count,
            "skipped": len(skipped_files),
            "skipped_details": skipped_files,
            "top_skills": top_skills,
            "avg_experience": avg_yoe
        })

        # Reset ranking so new candidates will be re-ranked
        state["ranking_done"] = False
        state["results"]      = []

    # -- Display ZIP Summary ---------------------------------------------------
    summary = getattr(project, "zip_ingestion_summary", None)
    if summary:
        skipped_html = ""
        if summary["skipped"] > 0:
            reasons_html = "".join([f"<li><code>{f[0]}</code>: {f[1]}</li>" for f in summary["skipped_details"][:10]])
            more_html = f"<li>... and {summary['skipped'] - 10} more</li>" if summary["skipped"] > 10 else ""
            skipped_html = f'''
            <div style="margin-top:0.75rem;font-size:0.75rem;color:#CC0000;background:#FFF5F5;border:1px solid #F5C0C0;border-radius:6px;padding:0.625rem">
              <strong style="display:block;margin-bottom:0.25rem">⚠ {summary["skipped"]} file(s) skipped:</strong>
              <ul style="margin:0 0 0 1rem;padding:0">{reasons_html}{more_html}</ul>
            </div>
            '''
        
        st.markdown(
            f'''
            <div style="background:#F0F7FF;border:1px solid #C8DEFF;border-radius:8px;padding:1rem 1.25rem;margin-top:0.75rem;">
                <div style="font-size:0.875rem;font-weight:700;color:#0071E3;margin-bottom:0.75rem;display:flex;align-items:center;gap:0.3rem">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                  ZIP Upload Summary
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.625rem;margin-bottom:0.625rem">
                    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;border:1px solid #E8E8ED">
                      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem">Files Discovered</div>
                      <div style="font-size:1.1rem;font-weight:700;color:#1D1D1F">{summary["total_discovered"]}</div>
                    </div>
                    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;border:1px solid #E8E8ED">
                      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem">Processed</div>
                      <div style="font-size:1.1rem;font-weight:700;color:#1A8917">{summary["processed_successfully"]}</div>
                    </div>
                    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;border:1px solid #E8E8ED">
                      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem">Skipped</div>
                      <div style="font-size:1.1rem;font-weight:700;color:{'#CC0000' if summary['skipped'] > 0 else '#1D1D1F'}">{summary["skipped"]}</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.625rem">
                    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;border:1px solid #E8E8ED">
                      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem">Top Detected Skills</div>
                      <div style="font-size:0.8125rem;color:#1D1D1F;font-weight:500">{', '.join(summary["top_skills"]) if summary["top_skills"] else 'None'}</div>
                    </div>
                    <div style="background:#fff;border-radius:6px;padding:0.625rem 0.75rem;border:1px solid #E8E8ED">
                      <div style="font-size:0.625rem;color:#86868B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem">Average Experience</div>
                      <div style="font-size:0.8125rem;color:#1D1D1F;font-weight:500">{summary["avg_experience"]} years</div>
                    </div>
                </div>
                {skipped_html}
            </div>
            ''',
            unsafe_allow_html=True
        )

    # -- Total candidates bar --------------------------------------------------
    total = len(project.resume_candidates or [])
    if total > 0:
        st.markdown(
            f'<div style="background:#EBF5EA;border:1px solid #A8D5A2;'
            f'border-radius:8px;padding:0.75rem 1.25rem;margin-top:0.75rem;'
            f'font-size:0.8125rem;color:#1A8917;font-weight:600">'
            f'✓ {total} candidate profile{"s" if total != 1 else ""} ready for ranking.'
            f'<span style="color:#6E6E73;font-weight:400"> '
            f'Click <strong>&#9654; Run Ranking Analysis</strong> in the sidebar.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )



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
        resume_clicked = _source_card(
            emoji="&#128196;",
            title="Resume Upload",
            description=(
                "Upload individual PDF or DOCX resumes. Each resume is analyzed "
                "by the AI Resume Intelligence Agent and ready for ranking."
            ),
            status="Available",
            status_color="green",
            card_key="resume",
            is_selected=(current_source == "resume"),
            disabled=False,
        )
        if resume_clicked:
            _set_source(project, "resume", state)
            st.rerun()

        if current_source == "resume":
            _render_resume_panel(project, state)

    with col4:
        zip_clicked = _source_card(
            emoji="&#128230;",
            title="ZIP Upload",
            description=(
                "Upload a ZIP archive containing multiple resumes. "
                "Bulk processing with automatic extraction and parsing."
            ),
            status="Available",
            status_color="green",
            card_key="zip",
            is_selected=(current_source == "zip"),
            disabled=False,
        )
        if zip_clicked:
            _set_source(project, "zip", state)
            st.rerun()

        if current_source == "zip":
            _render_zip_panel(project, state)

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
