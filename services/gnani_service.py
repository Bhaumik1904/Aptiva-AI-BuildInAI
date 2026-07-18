"""
APTIVA AI — Gnani.ai Voice Intelligence Service
================================================
A centralized service acting as a Voice Experience Layer.
Encapsulates all STT and TTS logic. Implements caching and resilient
error handling so the application never crashes if voice is unavailable.
"""

import hashlib
import streamlit as st
from typing import List, Dict, Any, Optional

from core.secrets_utils import resolve_api_key

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    _GNANI_AVAILABLE = True
except ImportError:
    _GNANI_AVAILABLE = False


class GnaniService:
    def __init__(self, config: dict):
        self._config = config
        self._voice_cfg = config.get("voice", {})
        
        self._api_key = resolve_api_key(config, "gnani_api_key", "GNANI_API_KEY")
        self.enabled = self._voice_cfg.get("enabled", True) and bool(self._api_key) and _GNANI_AVAILABLE
        
        self.default_voice = self._voice_cfg.get("default_voice", "Kaveri")
        self.cache_enabled = self._voice_cfg.get("cache_enabled", True)
        
        self.tts = None
        self.stt = None
        
        if self.enabled:
            try:
                self.tts = GnaniTTSClient(api_key=self._api_key)
                self.stt = GnaniSTTClient(api_key=self._api_key)
            except Exception:
                self.enabled = False

    def generate_audio(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Reusable helper to generate and cache audio.
        Uses SHA256 hashing for cache keys as per revisions.
        """
        if not self.enabled or not text.strip():
            return None
            
        voice_to_use = voice or self.default_voice
        
        # SHA256 caching
        if self.cache_enabled:
            hash_input = f"{text}_{voice_to_use}".encode("utf-8")
            cache_key = hashlib.sha256(hash_input).hexdigest()
            
            # Using st.session_state as the stable caching mechanism 
            # to prevent re-generation during Streamlit reruns.
            # St.cache_data on a class method can be tricky with unhashable self,
            # so we store the cache inside st.session_state securely.
            if "gnani_audio_cache" not in st.session_state:
                st.session_state["gnani_audio_cache"] = {}
                
            if cache_key in st.session_state["gnani_audio_cache"]:
                return st.session_state["gnani_audio_cache"][cache_key]

        try:
            audio_bytes = self.tts.synthesize(text, voice=voice_to_use)
            
            if self.cache_enabled:
                st.session_state["gnani_audio_cache"][cache_key] = audio_bytes
                
            return audio_bytes
        except Exception as e:
            import traceback
            print(f"[Gnani TTS Error] {e}")
            traceback.print_exc()
            return None

    def transcribe(self, audio_file: bytes) -> Optional[str]:
        if not self.enabled or not audio_file:
            return None
            
        try:
            import io
            file_obj = io.BytesIO(audio_file)
            file_obj.name = "audio.wav"
            res = self.stt.transcribe(file_obj)
            return res.get("transcript", "")
        except Exception as e:
            import traceback
            print(f"[Gnani STT Error] {e}")
            traceback.print_exc()
            return None

    def synthesize(self, text: str) -> Optional[bytes]:
        return self.generate_audio(text)

    def candidate_brief(self, candidate: dict, payload: dict) -> Optional[bytes]:
        """
        Synthesize a comprehensive candidate brief containing the AI score, 
        strengths, and gaps from the cached insights payload.
        """
        if not payload:
            return None
            
        name = candidate.get("profile", {}).get("anonymized_name", "The candidate")
        rec = payload.get("hiring_recommendation", "Consider")
        summary = payload.get("match_summary", "")
        
        strengths = payload.get("strengths", [])
        strengths_str = ("Their key strengths include " + ", ".join(strengths[:3]) + ".") if strengths else ""
        
        gaps = payload.get("skill_gaps", [])
        gaps_str = ""
        if gaps:
            gap_names = [g.get("skill", "") for g in gaps[:2]]
            gaps_str = "Some gaps to note are in " + ", ".join(gap_names) + "."
            
        text = f"{name} is recommended as {rec}. {summary} {strengths_str} {gaps_str}"
        return self.generate_audio(text)

    def shortlist_brief(self, shortlisted_candidates: List[dict]) -> Optional[bytes]:
        """
        Summarize the top-ranked candidates from the shortlist.
        """
        if not shortlisted_candidates:
            return None
            
        # Narrate up to the top 3
        top_candidates = shortlisted_candidates[:3]
        
        brief_parts = [f"Here is your AI Shortlist. We have identified {len(shortlisted_candidates)} top candidates."]
        
        for idx, cand in enumerate(top_candidates):
            cid = cand.get("candidate", {}).get("candidate_id", f"Candidate {idx+1}")
            rec = cand.get("recommendation", "Consider")
            brief_parts.append(f"Rank {idx+1} is {cid}, marked as {rec}. {cand.get('match_summary', '')}")
            
        return self.generate_audio(" ".join(brief_parts))

    def comparison_brief(self, candidate_a: dict, candidate_b: dict, comparison_payload: dict) -> Optional[bytes]:
        """
        Narrate the existing AI comparison without regenerating.
        Uses 'overall_comparison' (primary) and 'evidence_summary' (secondary)
        which are the actual keys returned by ComparisonIntelligenceAgent.
        """
        if not comparison_payload:
            return None

        # 'executive_summary' does not exist in the payload — use the correct keys.
        overall = comparison_payload.get("overall_comparison", "").strip()
        evidence = comparison_payload.get("evidence_summary", "").strip()
        rec_reason = comparison_payload.get("recommendation_reason", "").strip()

        # Build the narration from available fields
        parts = [p for p in [overall, rec_reason, evidence] if p]
        summary = " ".join(parts)

        if not summary:
            return None

        return self.generate_audio(summary)
