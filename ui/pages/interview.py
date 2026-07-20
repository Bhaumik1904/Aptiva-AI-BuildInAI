"""
APTIVA AI — Interview Kit UI
============================
Displays the pre-generated AI Interview Kit.
This module is strictly a view layer. It does not generate insights or invoke Gemini.
"""

import streamlit as st

def _render_question_section(title: str, questions: list, emoji: str):
    """Helper to render a list of questions inside numbered expanders."""
    st.markdown(f"### {emoji} {title}")
    
    if not questions:
        st.info(f"No {title.lower()} generated.")
        return

    for idx, q in enumerate(questions, start=1):
        with st.expander(f"{title} #{idx}", expanded=(idx == 1)):
            st.markdown(f"**Question:** {q.get('question', 'Not specified')}")
            st.markdown(f"*{q.get('why_this_question', 'Not specified')}*")
            
            st.divider()
            
            st.markdown(f"**✅ Expected strong answer:** {q.get('expected_strong_answer', 'Not specified')}")
            
            red_flags = q.get("red_flags", [])
            if red_flags:
                st.markdown("**⚠ Red flags to watch for:**")
                for rf in red_flags:
                    st.markdown(f"- {rf}")
                    
            follow_up = q.get("suggested_follow_up")
            if follow_up:
                st.markdown(f"**🔁 Suggested follow-up:** {follow_up}")

def _render_list_section(title: str, emoji: str, items: list, empty_message: str, expanded: bool = False):
    """Helper to render a simple list of strings inside an expander."""
    with st.expander(f"{emoji} {title}", expanded=expanded):
        if not items:
            st.info(empty_message)
            return
            
        for item in items:
            st.markdown(f"- {item}")

def render_interview_page():
    # --- Header ---
    st.title("🎙️ Interview Kit")
    
    # Change 1: Personalized Header
    cand = st.session_state.get("current_candidate") or {}
    if not isinstance(cand, dict):
        cand = {}
    profile = cand.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
        
    name = profile.get("anonymized_name") or "Unknown Candidate"
    role = profile.get("current_title") or "Unknown Role"
    
    with st.container(border=True):
        st.markdown(f"### {name}")
        st.caption(role)
    
    kit = st.session_state.get("current_interview_kit")
    
    # Change 3: Better Empty State
    if not kit or not isinstance(kit, dict):
        with st.container(border=True):
            st.warning("⚠️ No Interview Kit Available")

            st.write(
                "This candidate does not have a generated Interview Kit yet."
            )

            st.info(
                "Generate Candidate Insights first, then click **Generate Interview Kit**."
    )
        return

    # --- Summaries in Bordered Containers ---
    summary = kit.get("candidate_summary", "")
    strategy = kit.get("interview_strategy", "")
    
    if summary:
        with st.container(border=True):
            st.markdown("### 🧑‍💼 Candidate Summary")
            st.write(summary)
            
    if strategy:
        with st.container(border=True):
            st.markdown("### 🗺️ Interview Strategy")
            st.write(strategy)
            
    st.divider()

    # --- Metrics Row ---
    diff = kit.get("difficulty", "Unknown")
    dur = kit.get("estimated_duration_minutes", 0)
    tech_qs = kit.get("technical_questions", [])
    behav_qs = kit.get("behavioral_questions", [])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Difficulty", diff)
    
    # Change 2: Improve Duration Metric
    col2.metric("Duration", f"🕒 {dur} mins" if dur else "Unknown")
    
    col3.metric("Technical Questions", len(tech_qs) if isinstance(tech_qs, list) else 0)
    col4.metric("Behavioral Questions", len(behav_qs) if isinstance(behav_qs, list) else 0)
    
    st.divider()

    # --- Expandable Analysis Sections ---
    strengths = kit.get("candidate_strengths", [])
    risks = kit.get("risk_signals", [])
    hires = kit.get("hire_signals", [])
    probes = kit.get("areas_to_probe", [])
    checklist = kit.get("interviewer_checklist", [])

    colA, colB = st.columns(2)
    with colA:
        _render_list_section("Strengths", "✅", strengths, "No specific strengths identified.", expanded=True)
        _render_list_section("Hire Signals", "⭐", hires, "No strong hire signals identified.")
        _render_list_section("Checklist", "📋", checklist, "No checklist items identified.")
        
    with colB:
        _render_list_section("Risks", "⚠", risks, "No specific risk signals identified.", expanded=True)
        _render_list_section("Areas To Probe", "🎯", probes, "No specific areas to probe identified.")
        
    st.divider()

    # --- Questions Sections ---
    _render_question_section("Technical", tech_qs, "💻")
    
    st.divider()
    
    proj_qs = kit.get("project_questions", [])
    _render_question_section("Projects", proj_qs, "📦")
    
    st.divider()
    
    _render_question_section("Behavioral", behav_qs, "🧠")

    # --- Final Notes ---
    notes = kit.get("final_interviewer_notes", "")
    if notes:
        st.divider()
        st.markdown("### 📝 Final Interviewer Notes")
        st.success(notes)

if __name__ == "__main__":
    # If this page is executed directly by Streamlit's multi-page router
    render_interview_page()
