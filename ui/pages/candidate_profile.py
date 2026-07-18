"""
APTIVA AI — Candidate Profile Deep Dive
Full profile card, career timeline, skill list, education, and all signals.
"""

import streamlit as st

from agents.matching_agent import CandidateIntelligenceAgent
from ui.voice_card import render_voice_card
from core.reasoning import generate_ai_insights, generate_reasoning
from core.skill_gap import analyze_skill_gap
from ui.charts import behavioral_radar, skill_match_chart
from ui.components import (
    render_ai_insights,
    render_hireability_index,
    render_honeypot_warning,
    render_profile_header,
    render_score_breakdown,
)
from ui.icons import icon
from ui.styles import page_header, section_label


def render(state: dict):
    """Render the full Candidate Profile page."""
    page_header("Candidate Profile", "Full profile deep dive")

    # -- Candidate Selector ------------------------------------------------
    results = state.get("results", [])
    if not results:
        st.info("No candidates loaded. Run the ranker first.")
        return

    candidate_options = {
        f"#{r['rank']} · {r['candidate']['candidate_id']} · {r['candidate'].get('profile', {}).get('current_title', '')[:30]}": r["candidate"]["candidate_id"]
        for r in results
    }

    _, col_sel = st.columns([2, 1])
    with col_sel:
        new_selection = st.selectbox(
            "Select Candidate",
            list(candidate_options.keys()),
            index=next((i for i, v in enumerate(candidate_options.values()) if v == state.get("selected_candidate_id")), 0),
            key="profile_select",
        )
        if new_selection:
            new_cid = candidate_options[new_selection]
            if new_cid != state.get("selected_candidate_id"):
                st.session_state["selected_candidate_id"] = new_cid
                st.rerun()

    result = _get_selected_result(state)
    if not result:
        st.info("Select a candidate from the Rankings page to view their profile.")
        return

    candidate = result["candidate"]
    components = result.get("components", {})
    rank = result["rank"]

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    certifications = candidate.get("certifications", [])

    honeypot_flags = components.get("honeypot_flags", [])
    if honeypot_flags:
        render_honeypot_warning(honeypot_flags)

    render_profile_header(candidate, components, rank)

    # ── Voice AI Panel — first-class, always visible ───────────────────────
    _render_voice_ai_panel(candidate, components, state)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Tabs --------------------------------------------------------------
    tab_overview, tab_career, tab_skills, tab_signals, tab_reasoning, tab_ai = st.tabs([
        "Overview", "Career History", "Skills & Gap",
        "Behavioral Signals", "AI Reasoning", "✨ AI Insights"
    ])

    # -- TAB 1: Overview ---------------------------------------------------
    with tab_overview:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            section_label("HIREABILITY INDEX")
            hi = components.get("hireability_index", {})
            render_hireability_index(hi)

            section_label("SCORE BREAKDOWN")
            render_score_breakdown(components)

        with col2:
            section_label("PROFILE SUMMARY")
            summary = profile.get("summary", "No summary available.")
            st.markdown(
                f'<div style="font-size:0.9375rem;color:#1D1D1F;line-height:1.65;background:#F5F5F7;border-radius:10px;padding:1.25rem">{summary}</div>',
                unsafe_allow_html=True,
            )

            # Quick stats
            st.markdown("<br>", unsafe_allow_html=True)
            section_label("QUICK FACTS")
            qc1, qc2 = st.columns(2)
            with qc1:
                st.markdown(_stat("Current Company", profile.get("current_company", "—")), unsafe_allow_html=True)
                st.markdown(_stat("Industry", profile.get("current_industry", "—")), unsafe_allow_html=True)
                st.markdown(_stat("Company Size", profile.get("current_company_size", "—")), unsafe_allow_html=True)
                st.markdown(_stat("Location", profile.get("location", "—")), unsafe_allow_html=True)
            with qc2:
                st.markdown(_stat("Notice Period", f"{signals.get('notice_period_days', '—')} days"), unsafe_allow_html=True)
                st.markdown(_stat("Open to Work", "Yes" if signals.get("open_to_work_flag") else "No"), unsafe_allow_html=True)
                st.markdown(_stat("Work Mode Preference", signals.get("preferred_work_mode", "—")), unsafe_allow_html=True)
                st.markdown(_stat("Willing to Relocate", "Yes" if signals.get("willing_to_relocate") else "No"), unsafe_allow_html=True)

    # -- TAB 2: Career History ---------------------------------------------
    with tab_career:
        if not career:
            st.info("No career history available.")
        else:
            section_label(f"CAREER HISTORY ({len(career)} roles)")
            for i, job in enumerate(sorted(career, key=lambda j: j.get("start_date", ""), reverse=True)):
                _render_job_card(job, i)

    # -- TAB 3: Skills & Gap -----------------------------------------------
    with tab_skills:
        skill_gap = analyze_skill_gap(candidate)

        sc1, sc2 = st.columns([1.2, 1.8])
        with sc1:
            skill_fig = skill_match_chart(skill_gap)
            st.plotly_chart(skill_fig, use_container_width=True, config={"displayModeBar": False})

            st.metric("Core Match", f"{skill_gap.get('core_match_pct', 0):.0f}%")
            st.metric("Bonus Match", f"{skill_gap.get('bonus_match_pct', 0):.0f}%")

        with sc2:
            # Skill gap detail table
            section_label("REQUIRED SKILLS ANALYSIS")
            required = skill_gap.get("required_skills", [])
            for skill_row in required:
                _render_skill_row(skill_row)

            st.markdown("<br>", unsafe_allow_html=True)
            section_label("BONUS SKILLS MATCHED")
            bonus = skill_gap.get("bonus_skills_matched", [])
            if bonus:
                st.markdown(
                    " ".join(f'<span class="skill-tag skill-tag-bonus">{s}</span>' for s in bonus),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span style="color:#86868B;font-size:0.875rem">None matched</span>', unsafe_allow_html=True)

        # All candidate skills
        st.markdown("---")
        section_label("ALL CANDIDATE SKILLS")
        prof_colors = {"expert": "#1A8917", "advanced": "#0071E3", "intermediate": "#C47000", "beginner": "#CC0000"}
        cols = st.columns(3)
        for i, skill in enumerate(sorted(skills, key=lambda s: s.get("duration_months", 0), reverse=True)):
            with cols[i % 3]:
                pc = prof_colors.get(skill.get("proficiency", "intermediate"), "#6E6E73")
                st.markdown(
                    f"""
<div style="background:#F5F5F7;border-radius:8px;padding:0.625rem 0.875rem;margin:0.25rem 0;display:flex;justify-content:space-between;align-items:center">
  <span style="font-size:0.875rem;font-weight:500;color:#1D1D1F">{skill.get('name','')}</span>
  <span style="font-size:0.75rem;font-weight:600;color:{pc}">{skill.get('proficiency','').capitalize()}</span>
</div>""",
                    unsafe_allow_html=True,
                )

    # -- TAB 4: Behavioral Signals ------------------------------------------
    with tab_signals:
        bc1, bc2 = st.columns([1.5, 1.5])
        with bc1:
            radar_fig = behavioral_radar(signals)
            st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

        with bc2:
            section_label("ALL 23 SIGNALS")
            _render_signals_table(signals)

        # Education
        st.markdown("---")
        section_label("EDUCATION")
        for edu in education:
            tier = edu.get("tier", "unknown")
            tier_color = {"tier_1": "#1A8917", "tier_2": "#0071E3", "tier_3": "#C47000", "tier_4": "#86868B"}.get(tier, "#86868B")
            st.markdown(
                f"""
<div class="aptiva-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-weight:600;color:#1D1D1F">{edu.get('degree','')} in {edu.get('field_of_study','')}</div>
      <div style="color:#6E6E73;font-size:0.875rem">{edu.get('institution','')} · {edu.get('start_year','?')} – {edu.get('end_year','?')}</div>
    </div>
    <span style="font-size:0.75rem;font-weight:600;color:{tier_color};background:{'#EBF5EA' if tier=='tier_1' else '#F5F5F7'};padding:0.2rem 0.6rem;border-radius:4px">{tier.upper()}</span>
  </div>
</div>""",
                unsafe_allow_html=True,
            )

    # -- TAB 5: AI Reasoning ------------------------------------------------
    with tab_reasoning:
        # AI Insights
        insights = generate_ai_insights(candidate, components)
        section_label("AI INSIGHTS")
        render_ai_insights(insights)

        st.markdown("---")

        section_label("APTIVA AI REASONING")
        reasoning = generate_reasoning(candidate, rank, components)
        st.markdown(
            f'<div style="background:#F5F5F7;border-radius:10px;padding:1.25rem;font-size:0.9375rem;color:#1D1D1F;line-height:1.6">{reasoning}</div>',
            unsafe_allow_html=True,
        )

        # Navigate to Judge Mode
        st.markdown("---")
        if st.button("Open in Judge Mode", use_container_width=False, icon=":material/gavel:"):
            state["page"] = "judge_mode"
            st.rerun()

    # -- TAB 6: ✨ AI Insights -----------------------------------------------
    with tab_ai:
        _render_ai_insights_tab(candidate, components, state)


# -- Voice AI Panel — first-class feature, always rendered above tabs ----------

def _render_voice_ai_panel(
    candidate:        dict,
    score_components: dict,
    state:            dict,
) -> None:
    """
    Premium Voice AI card placed immediately after the profile header.
    Always visible. Auto-generates AI Insights if not cached before narrating.
    Uses render_voice_card() for status, progress, audio player, and cache UX.
    """
    from services.gnani_service import GnaniService
    config = state.get("app_config", {})
    gnani  = GnaniService(config)
    cid    = candidate.get("candidate_id", "unknown")

    def _generate() -> tuple:
        """Generate AI Insights (if needed) then synthesize speech."""
        cache_key     = f"ai_insights_{cid}"
        ai_was_cached = (
            cache_key in st.session_state
            and st.session_state[cache_key].get("ok")
        )

        with st.status("🧠 Preparing AI summary...", expanded=True) as status:
            cached = st.session_state.get(cache_key, {})
            if not cached.get("ok"):
                active_project = state.get("active_project")
                if active_project is None:
                    raise RuntimeError(
                        "No active Hiring Project. Set one in the Hiring Projects page."
                    )
                jd    = active_project.job_description
                agent = CandidateIntelligenceAgent(config)
                if not agent.is_configured():
                    raise RuntimeError("Gemini API key not configured.")
                payload = agent.analyze(jd, candidate, score_components)
                st.session_state[cache_key] = {"ok": True, "payload": payload}
                cached = st.session_state[cache_key]
            else:
                st.write("⚡ Using cached AI Insights.")

            status.update(label="🎤 Synthesizing speech...", state="running")
            payload     = cached.get("payload", {})
            audio_bytes = gnani.candidate_brief(candidate, payload)
            status.update(label="✅ Done", state="complete", expanded=False)

        return audio_bytes, ai_was_cached

    render_voice_card(
        button_label  = "🎧 Recruiter Brief",
        button_key    = f"voice_brief_{cid}",
        subtitle      = "Generate a spoken candidate summary for this profile.",
        gnani_enabled = gnani.enabled,
        on_generate   = _generate,
    )


# -- AI Insights Tab Logic -------------------------------------------------------

_REC_CONFIG = {
    "Strong Hire":      ("#EBF5EA", "#1A8917", "1px solid #A8D5A2", "\U0001f7e2"),
    "Hire":             ("#F0F7FF", "#0071E3", "1px solid #C8DEFF", "\U0001f535"),
    "Consider":         ("#FFF8E6", "#C47000", "1px solid #F5DDA0", "\U0001f7e1"),
    "Not Recommended":  ("#FFF5F5", "#CC0000", "1px solid #F5C0C0", "\U0001f534"),
}

_CONF_CONFIG = {
    "High":   ("#EBF5EA", "#1A8917", "High Confidence"),
    "Medium": ("#FFF8E6", "#C47000", "Medium Confidence"),
    "Low":    ("#FFF5F5", "#CC0000", "Low Confidence"),
}

_SEV_CONFIG = {
    "Critical": ("#FFF5F5", "#CC0000", "#F5C0C0"),
    "Important":("#FFF8E6", "#C47000", "#F5DDA0"),
    "Optional": ("#F5F5F7", "#6E6E73", "#E8E8ED"),
}


def _render_ai_insights_tab(
    candidate: dict,
    score_components: dict,
    state: dict,
) -> None:
    """
    Renders the \u2728 AI Insights tab.

    On first open: calls CandidateIntelligenceAgent and caches the result
    in st.session_state under a per-candidate key so Gemini is not re-called
    on every Streamlit rerun.
    """
    cid      = candidate.get("candidate_id", "unknown")
    cache_key= f"ai_insights_{cid}"
    cfg      = state.get("app_config", {})
    agent    = CandidateIntelligenceAgent(cfg)

    # Retrieve the active project's JD
    active_project = state.get("active_project")
    if active_project is None:
        st.warning("No active Hiring Project. Set one in the Hiring Projects page.")
        return
    jd = active_project.job_description

    # ── API key guard ────────────────────────────────────────────────────────
    if not agent.is_configured():
        st.warning(
            "**Gemini API key not configured.** "
            "Set `gemini_api_key` in `config.yaml` or export `GEMINI_API_KEY`.",
            icon="\u26a0\ufe0f",
        )
        return

    # ── Load or generate ─────────────────────────────────────────────────────
    if cache_key not in st.session_state:
        progress_steps = [
            "Analyzing candidate profile\u2026",
            "Comparing skills against Job Description\u2026",
            "Measuring experience against requirements\u2026",
            "Identifying skill gaps\u2026",
            "Building hiring recommendation\u2026",
            "Generating tailored interview questions\u2026",
            "Assessing hiring confidence\u2026",
        ]

        status_slot = st.empty()

        for step in progress_steps:
            status_slot.markdown(
                f'<div style="background:#F0F7FF;border:1px solid #C8DEFF;'
                f'border-radius:8px;padding:0.875rem 1.25rem;'
                f'font-size:0.8125rem;color:#0071E3;">'
                f'\u23f3 {step}</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Generating AI insights\u2026"):
            try:
                payload = agent.analyze(jd, candidate, score_components)
                st.session_state[cache_key] = {"ok": True, "payload": payload}
            except Exception as exc:  # noqa: BLE001
                st.session_state[cache_key] = {"ok": False, "error": str(exc)}

        status_slot.empty()

    result = st.session_state[cache_key]

    if not result.get("ok"):
        st.error(
            f"\u26a0\ufe0f AI analysis failed: {result.get('error', 'Unknown error')}\n\n"
            "Please check your API key and try again. "
            "You can clear the cache by switching to another candidate and back.",
        )
        if st.button("\U0001f504 Retry Analysis", key=f"retry_{cid}"):
            st.session_state.pop(cache_key, None)
            st.rerun()
        return

    payload = result["payload"]
    _render_ai_insights_panel(payload, candidate)


def _render_ai_insights_panel(payload: dict, candidate: dict) -> None:
    """Render the full 9-section AI Insights panel."""

    rec  = payload.get("hiring_recommendation", "Consider")
    pct  = payload.get("overall_match_pct", 0)
    rec_bg, rec_clr, rec_border, rec_dot = _REC_CONFIG.get(
        rec, ("#F5F5F7", "#6E6E73", "1px solid #E8E8ED", "\u26aa")
    )

    name = candidate.get("profile", {}).get("anonymized_name", "Candidate")

    # ── Section 1: Hero bar ──────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{rec_bg};border:{rec_border};border-radius:12px;'
        f'padding:1.25rem 1.5rem;margin-bottom:1.25rem;'
        f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem">'
        f'<div>'
        f'<div style="font-size:0.6875rem;font-weight:700;color:{rec_clr};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.25rem">Overall Match</div>'
        f'<div style="font-size:2.5rem;font-weight:800;color:{rec_clr};'
        f'letter-spacing:-0.04em;line-height:1">{pct}%</div>'
        f'<div style="font-size:0.8125rem;color:#6E6E73;margin-top:0.25rem">{name}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div style="font-size:0.6875rem;font-weight:700;color:{rec_clr};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.35rem">Recommendation</div>'
        f'<div style="font-size:1.375rem;font-weight:700;color:{rec_clr}">{rec_dot} {rec}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Voice brief is now handled by _render_voice_ai_panel() above the tabs.

    # Match summary
    summary = payload.get("match_summary", "")
    if summary:
        st.markdown(
            f'<div style="background:#F5F5F7;border-radius:10px;padding:1rem 1.25rem;'
            f'font-size:0.9375rem;color:#1D1D1F;line-height:1.65;margin-bottom:1rem">'
            f'{summary}</div>',
            unsafe_allow_html=True,
        )

    # ── Sections 2 & 3: Strengths + Skill Gaps side by side ─────────────────
    col_str, col_gap = st.columns(2, gap="large")

    with col_str:
        section_label("STRENGTHS")
        strengths = payload.get("strengths", [])
        if strengths:
            items_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:0.5rem;'
                f'margin:0.3rem 0;font-size:0.875rem;color:#1D1D1F">'
                f'<span style="color:#1A8917;font-weight:700;flex-shrink:0">\u2713</span>'
                f'<span>{s}</span></div>'
                for s in strengths
            )
            st.markdown(
                f'<div style="background:#EBF5EA;border:1px solid #C3EAC3;border-radius:8px;'
                f'padding:1rem 1.125rem">{items_html}</div>',
                unsafe_allow_html=True,
            )

    with col_gap:
        section_label("SKILL GAPS")
        skill_gaps = payload.get("skill_gaps", [])
        if skill_gaps:
            for sg in skill_gaps:
                sev  = sg.get("severity", "Important")
                skl  = sg.get("skill", "")
                evid = sg.get("evidence", "")
                bg, clr, brd = _SEV_CONFIG.get(sev, ("#F5F5F7", "#6E6E73", "#E8E8ED"))
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {brd};'
                    f'border-radius:6px;padding:0.5rem 0.75rem;margin:0.25rem 0">'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.15rem">'
                    f'<span style="font-size:0.6875rem;font-weight:700;color:{clr};'
                    f'text-transform:uppercase;letter-spacing:0.06em;background:white;'
                    f'padding:0.1rem 0.35rem;border-radius:3px">{sev}</span>'
                    f'<span style="font-size:0.875rem;font-weight:600;color:#1D1D1F">{skl}</span>'
                    f'</div>'
                    f'<div style="font-size:0.8rem;color:#6E6E73">{evid}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#1A8917;font-size:0.875rem">'
                '\u2713 All core skills matched. No critical gaps identified.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sections 4 & 5: Experience + Education analysis side by side ─────────
    col_exp, col_edu = st.columns(2, gap="large")

    with col_exp:
        section_label("EXPERIENCE ANALYSIS")
        exp = payload.get("experience_analysis", {})
        ev  = exp.get("verdict", "Meets Expectation")
        er  = exp.get("reasoning", "")
        exp_clr = {"Below Expectation": "#CC0000", "Meets Expectation": "#0071E3",
                   "Exceeds Expectation": "#1A8917"}.get(ev, "#6E6E73")
        st.markdown(
            f'<div style="background:#F5F5F7;border-radius:8px;padding:0.875rem 1rem">'
            f'<div style="font-weight:700;color:{exp_clr};font-size:0.9375rem;'
            f'margin-bottom:0.4rem">{ev}</div>'
            f'<div style="font-size:0.8125rem;color:#6E6E73;line-height:1.55">{er}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_edu:
        section_label("EDUCATION ANALYSIS")
        edu = payload.get("education_analysis", {})
        edv = edu.get("verdict", "Not Specified")
        edr = edu.get("reasoning", "")
        edu_clr = {"Aligned": "#1A8917", "Partially Aligned": "#C47000",
                   "Not Aligned": "#CC0000", "Not Specified": "#86868B"}.get(edv, "#86868B")
        st.markdown(
            f'<div style="background:#F5F5F7;border-radius:8px;padding:0.875rem 1rem">'
            f'<div style="font-weight:700;color:{edu_clr};font-size:0.9375rem;'
            f'margin-bottom:0.4rem">{edv}</div>'
            f'<div style="font-size:0.8125rem;color:#6E6E73;line-height:1.55">{edr}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 6: Hiring Recommendation ─────────────────────────────────────
    section_label("HIRING RECOMMENDATION")
    rec_reason = payload.get("recommendation_reason", "")
    st.markdown(
        f'<div style="background:{rec_bg};border:{rec_border};border-radius:10px;'
        f'padding:1rem 1.25rem">'
        f'<div style="font-size:1rem;font-weight:700;color:{rec_clr};margin-bottom:0.4rem">'
        f'{rec_dot} {rec}</div>'
        f'<div style="font-size:0.875rem;color:#1D1D1F;line-height:1.6">{rec_reason}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 9: Hiring Confidence (approved addition) ─────────────────────
    section_label("HIRING CONFIDENCE")
    conf      = payload.get("hiring_confidence", {})
    conf_lvl  = conf.get("level", "Medium")
    conf_reas = conf.get("reasoning", "")
    conf_bg, conf_clr, conf_label = _CONF_CONFIG.get(
        conf_lvl, ("#F5F5F7", "#6E6E73", "Medium Confidence")
    )
    conf_icons = {"High": "\U0001f7e2 \U0001f7e2 \U0001f7e2",
                  "Medium": "\U0001f7e1 \U0001f7e1 \u26aa",
                  "Low": "\U0001f534 \u26aa \u26aa"}
    st.markdown(
        f'<div style="background:{conf_bg};border:1px solid {conf_clr}33;'
        f'border-radius:10px;padding:1rem 1.25rem;'
        f'display:flex;align-items:flex-start;gap:1rem">'
        f'<div style="flex-shrink:0">'
        f'<div style="font-size:1.375rem">{conf_icons.get(conf_lvl, "")}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-weight:700;color:{conf_clr};font-size:0.9375rem;margin-bottom:0.3rem">'
        f'{conf_label}</div>'
        f'<div style="font-size:0.8125rem;color:#1D1D1F;line-height:1.55">{conf_reas}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 7: Interview Questions ───────────────────────────────────────
    section_label("INTERVIEW QUESTIONS")
    tech_qs  = payload.get("technical_questions", [])
    behav_qs = payload.get("behavioral_questions", [])

    iq_col1, iq_col2 = st.columns([3, 2], gap="large")

    with iq_col1:
        st.markdown(
            '<div style="font-size:0.75rem;font-weight:700;color:#0071E3;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem">'
            '\U0001f4bb Technical (3)</div>',
            unsafe_allow_html=True,
        )
        for i, q in enumerate(tech_qs[:3], start=1):
            st.markdown(
                f'<div style="background:#F0F7FF;border:1px solid #C8DEFF;'
                f'border-radius:8px;padding:0.75rem 1rem;margin:0.375rem 0;'
                f'font-size:0.875rem;color:#1D1D1F;line-height:1.55">'
                f'<strong style="color:#0071E3">{i}.</strong> {q}</div>',
                unsafe_allow_html=True,
            )

    with iq_col2:
        st.markdown(
            '<div style="font-size:0.75rem;font-weight:700;color:#6E3FA3;'
            'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem">'
            '\U0001f9e0 Behavioural (2)</div>',
            unsafe_allow_html=True,
        )
        for i, q in enumerate(behav_qs[:2], start=1):
            st.markdown(
                f'<div style="background:#F5F0FF;border:1px solid #D5C8F0;'
                f'border-radius:8px;padding:0.75rem 1rem;margin:0.375rem 0;'
                f'font-size:0.875rem;color:#1D1D1F;line-height:1.55">'
                f'<strong style="color:#6E3FA3">{i}.</strong> {q}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 8: Skill Evidence (expandable) ────────────────────────────────
    skill_evidence = payload.get("skill_evidence", [])
    if skill_evidence:
        with st.expander(f"\U0001f50d Skill Evidence ({len(skill_evidence)} skills)", expanded=False):
            for ev in skill_evidence:
                matched = ev.get("matched", False)
                skl     = ev.get("skill", "")
                evid    = ev.get("evidence", "")
                tick    = "\u2713" if matched else "\u2717"
                clr     = "#1A8917" if matched else "#CC0000"
                bg      = "#F5FFF5" if matched else "#FFF5F5"
                st.markdown(
                    f'<div style="background:{bg};border-radius:6px;'
                    f'padding:0.5rem 0.75rem;margin:0.2rem 0;'
                    f'display:flex;align-items:flex-start;gap:0.5rem">'
                    f'<span style="color:{clr};font-weight:700;flex-shrink:0">{tick}</span>'
                    f'<div>'
                    f'<span style="font-size:0.875rem;font-weight:600;color:#1D1D1F">{skl}</span>'
                    f'<div style="font-size:0.8rem;color:#6E6E73;margin-top:0.1rem">{evid}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )


# -- Helper Functions ----------------------------------------------------------

def _get_selected_result(state: dict):
    results = state.get("results", [])
    selected_id = state.get("selected_candidate_id")
    if not selected_id and results:
        return results[0]
    for r in results:
        if r["candidate"].get("candidate_id") == selected_id:
            return r
    return results[0] if results else None


def _stat(label: str, value: str) -> str:
    return f'<div style="margin:0.375rem 0"><span style="font-size:0.75rem;color:#86868B;font-weight:500">{label}</span><div style="font-size:0.875rem;color:#1D1D1F;font-weight:500">{value}</div></div>'


def _render_job_card(job: dict, index: int):
    is_current = job.get("is_current", False)
    current_badge = '<span style="background:#EBF5EA;color:#1A8917;font-size:0.7rem;font-weight:600;padding:0.15rem 0.5rem;border-radius:4px;margin-left:0.5rem">Current</span>' if is_current else ""
    duration = job.get("duration_months", 0)
    duration_str = f"{duration // 12}yr {duration % 12}mo" if duration >= 12 else f"{duration}mo"

    desc = job.get("description", "")
    with st.expander(f"**{job.get('title','')}** at {job.get('company','')} · {duration_str}", expanded=(index == 0)):
        st.markdown(
            f'<div style="display:flex;gap:1.5rem;font-size:0.8125rem;color:#86868B;margin-bottom:0.75rem">'
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem">{icon("calendar",13)} {job.get("start_date","?")[:7]} → {job.get("end_date","Present")[:7] if job.get("end_date") else "Present"}</span>'
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem">{icon("building",13)} {job.get("company_size","?")}</span>'
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem">{icon("factory",13)} {job.get("industry","?")}</span>'
            f'</div><div style="font-size:0.9rem;color:#1D1D1F;line-height:1.65">{desc}</div>',
            unsafe_allow_html=True,
        )


def _render_skill_row(skill_row: dict):
    status = skill_row["status"]
    name = skill_row["name"]
    prof = skill_row.get("proficiency") or ""
    dur = skill_row.get("duration_months") or 0
    assessment = skill_row.get("assessment_score")
    mq = skill_row.get("match_quality", 0)

    if status == "present":
        _icon_svg = icon("check", 13, "#1A8917")
        icon_color = "#1A8917"
        bg = "#F5FFF5"
        prof_str = f"{prof.capitalize()} · {dur}mo"
        if assessment is not None:
            prof_str += f" · Assessed: {assessment:.0f}"
    else:
        _icon_svg = icon("x", 13, "#CC0000")
        icon_color = "#CC0000"
        bg = "#FFF5F5"
        prof_str = "Not found"

    st.markdown(
        f'<div style="background:{bg};border-radius:6px;padding:0.5rem 0.75rem;margin:0.25rem 0;display:flex;justify-content:space-between;align-items:center">'
        f'<div style="display:flex;align-items:center;gap:0.5rem">{_icon_svg}<span style="font-size:0.875rem;font-weight:500;color:#1D1D1F">{name}</span></div>'
        f'<span style="font-size:0.75rem;color:#6E6E73">{prof_str}</span></div>',
        unsafe_allow_html=True,
    )


def _render_signals_table(signals: dict):
    """Render key behavioral signals as a clean table."""
    items = [
        ("Profile Completeness", f"{signals.get('profile_completeness_score', 0)}/100"),
        ("Open to Work", "Yes" if signals.get("open_to_work_flag") else "No"),
        ("Last Active", signals.get("last_active_date", "—")),
        ("Notice Period", f"{signals.get('notice_period_days', 0)} days"),
        ("Recruiter Response Rate", f"{signals.get('recruiter_response_rate', 0):.0%}"),
        ("Avg Response Time", f"{signals.get('avg_response_time_hours', 0):.0f} hours"),
        ("Interview Completion", f"{signals.get('interview_completion_rate', 0):.0%}"),
        ("GitHub Activity", str(signals.get("github_activity_score", -1))),
        ("Saved by Recruiters (30d)", str(signals.get("saved_by_recruiters_30d", 0))),
        ("Profile Views (30d)", str(signals.get("profile_views_received_30d", 0))),
        ("Applications (30d)", str(signals.get("applications_submitted_30d", 0))),
        ("Email Verified", "Yes" if signals.get("verified_email") else "No"),
        ("Phone Verified", "Yes" if signals.get("verified_phone") else "No"),
        ("LinkedIn", "Connected" if signals.get("linkedin_connected") else "Not connected"),
        ("Connection Count", str(signals.get("connection_count", 0))),
        ("Willing to Relocate", "Yes" if signals.get("willing_to_relocate") else "No"),
        ("Work Mode Preference", signals.get("preferred_work_mode", "—")),
        ("Offer Acceptance Rate", f"{signals.get('offer_acceptance_rate', -1):.0%}" if signals.get("offer_acceptance_rate", -1) != -1 else "No history"),
    ]
    for label, value in items:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:0.375rem 0;border-bottom:1px solid #F0F0F0;font-size:0.875rem"><span style="color:#6E6E73">{label}</span><span style="font-weight:500;color:#1D1D1F">{value}</span></div>',
            unsafe_allow_html=True,
        )
