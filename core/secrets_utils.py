"""
APTIVA AI — Secrets Utility
Provides unified API key resolution for Streamlit Community Cloud and local environments.
Priority: st.secrets > os.environ > config.yaml
"""

import os
from typing import Dict, Any

def resolve_api_key(config: Dict[str, Any], config_key: str, env_key: str) -> str:
    """
    Resolve an API key using the following priority order:
    1. Streamlit Community Cloud Secrets (st.secrets)
    2. Local Environment Variables (os.environ)
    3. Fallback to config.yaml value
    """
    # 1. Streamlit Secrets (Safely check without failing if Streamlit isn't running)
    try:
        import streamlit as st
        # Check config key (e.g., "gemini_api_key")
        if config_key in st.secrets:
            return str(st.secrets[config_key]).strip()
        # Check env key (e.g., "GEMINI_API_KEY")
        if env_key in st.secrets:
            return str(st.secrets[env_key]).strip()
    except Exception:
        # Streamlit may not be initialized during CLI execution
        pass
        
    # 2. Environment Variables
    env_val = os.environ.get(env_key)
    if env_val:
        return str(env_val).strip()
        
    # 3. Config Fallback
    if config and config_key in config:
        return str(config[config_key]).strip()
        
    return ""
