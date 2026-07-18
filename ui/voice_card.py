"""
APTIVA AI -- Voice AI Card
==========================
Premium, reusable Voice AI UX component.
Used by: home.py, candidate_profile.py, comparison.py
"""

from __future__ import annotations
import streamlit as st

# ---- card CSS fragments ------------------------------------------------

_CARD_OPEN = (
    '<div style="' 
    'background:linear-gradient(135deg,#F8FAFF 0%,#F4F0FF 100%);' 
    'border:1px solid #D0DDFF;' 
    'border-radius:14px;' 
    'padding:1.125rem 1.375rem 1rem;' 
    'margin-bottom:1rem;">' 
)

_CARD_CLOSE = "</div>"

_DIVIDER = (
    "<hr style=\"border:0;border-top:1px solid #E0E8FF;"
    "margin:0.75rem 0 0.5rem\">"
)


def _card_header(subtitle: str) -> str:
    return (
        '<div style="display:flex;align-items:flex-start;' 
        'justify-content:space-between;margin-bottom:0.25rem">' 
        '<div style="display:flex;align-items:center;gap:0.5rem">' 
        '<span style="font-size:1.125rem">🎤</span>' 
        '<span style="font-size:0.9375rem;font-weight:700;color:#1D1D1F;' 
        'letter-spacing:-0.01em">Voice AI</span>' 
        '</div>' 
        '<span style="font-size:0.6875rem;color:#86868B;font-weight:500;' 
        'padding-top:0.1rem">Powered by Gnani.ai</span>' 
        '</div>' 
        f'<div style="font-size:0.8125rem;color:#6E6E73;margin-bottom:0.875rem">' 
        f'{subtitle}</div>'
    )


def _status_badge(text: str, color: str, bg: str) -> str:
    return (
        f'<div style="display:inline-flex;align-items:center;gap:0.35rem;' 
        f'background:{bg};border-radius:6px;padding:0.25rem 0.625rem;' 
        f'font-size:0.8125rem;font-weight:600;color:{color};' 
        f'margin-bottom:0.75rem">{text}</div>'
    )


def render_voice_card(
    *,
    button_label:  str,
    button_key:    str,
    subtitle:      str,
    gnani_enabled: bool,
    on_generate,
) -> None:
    """
    Render a premium Voice AI card.

    Parameters
    ----------
    button_label  : CTA button label e.g. "Recruiter Brief"
    button_key    : Unique Streamlit widget key
    subtitle      : One-line description shown below the title
    gnani_enabled : Whether GnaniService is configured
    on_generate   : Zero-arg callable -> (bytes | None, was_cached: bool)
                    Performs all AI generation and TTS.
                    Should use st.status() for step-by-step progress.
    """
    _audio_key  = f"_vc_audio_{button_key}"
    _cached_key = f"_vc_cached_{button_key}"
    _error_key  = f"_vc_error_{button_key}"

    audio_bytes = st.session_state.get(_audio_key)
    error_msg   = st.session_state.get(_error_key)

    # Card shell + header
    st.markdown(_CARD_OPEN + _card_header(subtitle), unsafe_allow_html=True)

    # Status indicator
    if error_msg:
        st.markdown(
            _status_badge(f"\u26a0\ufe0f {error_msg}", "#CC0000", "#FFF5F5"),
            unsafe_allow_html=True,
        )
    elif audio_bytes:
        st.markdown(
            _status_badge("\u2705 Voice summary ready.", "#1A8917", "#EBF5EA"),
            unsafe_allow_html=True,
        )
    elif gnani_enabled:
        st.markdown(
            _status_badge("\U0001f7e2 Voice Ready", "#1A8917", "#EBF5EA"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _status_badge(
                "\U0001f7e1 Configure your Gnani API key to enable Voice AI.",
                "#C47000", "#FFF8E6",
            ),
            unsafe_allow_html=True,
        )

    # Action button
    if gnani_enabled:
        if st.button(button_label, key=button_key,
                     help="Click to generate a spoken summary."):
            st.session_state.pop(_error_key, None)
            st.session_state.pop(_audio_key, None)
            st.session_state.pop(_cached_key, None)
            try:
                result_audio, result_cached = on_generate()
                if result_audio:
                    st.session_state[_audio_key]  = result_audio
                    st.session_state[_cached_key] = result_cached
                else:
                    st.session_state[_error_key] = (
                        "Voice service temporarily unavailable. Please try again."
                    )
            except Exception as exc:
                st.exception(exc)
            #st.rerun()
    else:
        st.button(button_label, key=button_key, disabled=True)

    # Audio player -- inside the card
    audio_bytes = st.session_state.get(_audio_key)
    if audio_bytes:
        st.markdown(_DIVIDER, unsafe_allow_html=True)
        if st.session_state.get(_cached_key):
            st.caption("\u26a1 Using cached narration")
        st.audio(audio_bytes, format="audio/wav")

    # Card close
    st.markdown(_CARD_CLOSE, unsafe_allow_html=True)
