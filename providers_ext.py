#!/usr/bin/env python3
"""
providers_ext.py — non-Databricks provider call paths for the sensitivity
and cross-domain scripts, so both can cover all five models in one run.

The OpenAI and Google call contracts below are lifted verbatim from the main
experiment script (run_experiments_patched.py), so the GPT-4o-mini and Gemini
cells produced here are apples-to-apples with the main-benchmark cells.

Databricks models (gpt-oss-120b, claude, llama) are NOT routed through this
module; each script keeps its own proven Databricks path. This module only
adds the two providers those scripts did not have.

Gemini note: the main experiments called Gemini through the generative-language
REST endpoint with an API key (GOOGLE_API_KEY), not the Vertex SDK. This module
matches that exact contract so the numbers line up. Set GOOGLE_API_KEY.

Env vars expected:
  OPENAI_API_KEY   for gpt-4o-mini
  GOOGLE_API_KEY   for gemini-flash
  GEMINI_MIN_INTERVAL_S  optional pacing for Gemini (seconds, default 1.5)
"""

import os
import time
import requests


# Unified model registry. provider is one of: databricks, openai, google.
# For databricks rows, the id is the serving-endpoint name; each script may
# also have its own DATABRICKS_ENDPOINTS map, which takes precedence if present.
MODEL_PROVIDER = {
    "dbrx":         ("databricks", "databricks-gpt-oss-120b"),
    "gpt-oss-120b": ("databricks", "databricks-gpt-oss-120b"),
    "claude":       ("databricks", "databricks-claude-sonnet-4-6"),
    "llama":        ("databricks", "databricks-llama-4-maverick"),
    "gpt-4o-mini":  ("openai",     "gpt-4o-mini"),
    "gemini-flash": ("google",     "gemini-2.5-flash"),
}


def provider_of(model_key):
    """Return (provider_name, model_id) for a model key. Unknown keys default
    to databricks with the key used as the endpoint name."""
    return MODEL_PROVIDER.get(model_key, ("databricks", model_key))


# =====================================================================
# OPENAI (verbatim contract from main script)
# =====================================================================

_OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def openai_chat(model_id, system_prompt, user_input,
                max_tokens=512, temperature=0.3):
    """Single chat call to the OpenAI API. Returns the content string."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    resp = requests.post(
        f"{_OPENAI_BASE}/chat/completions",
        json={"model": model_id, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# =====================================================================
# GOOGLE GEMINI (verbatim contract from main script, generative-language REST)
# =====================================================================

def google_chat(model_id, system_prompt, user_input,
                max_tokens=512, temperature=0.3):
    """Single chat call to the Gemini generative-language REST endpoint.
    Returns the content string. Includes the same pacing the main script used,
    since this endpoint is frequently overloaded."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    contents = [{"role": "user", "parts": [{"text": user_input}]}]
    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:generateContent?key={api_key}")

    # Pace calls to a frequently-overloaded endpoint so retries do not all land
    # in the same congested window. Tunable via GEMINI_MIN_INTERVAL_S.
    _min_interval = float(os.environ.get("GEMINI_MIN_INTERVAL_S", "1.5"))
    _last = getattr(google_chat, "_last_call_ts", 0.0)
    _wait = _min_interval - (time.time() - _last)
    if _wait > 0:
        time.sleep(_wait)
    google_chat._last_call_ts = time.time()

    resp = requests.post(url, json=body,
                         headers={"Content-Type": "application/json"},
                         timeout=180)
    resp.raise_for_status()
    data = resp.json()
    content = ""
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            content += part.get("text", "")
    return content


def validate_key_for(model_key):
    """Assert the right credential is present for a model key's provider."""
    prov, _ = provider_of(model_key)
    if prov == "openai":
        assert os.environ.get("OPENAI_API_KEY"), \
            "Set OPENAI_API_KEY env var for gpt-4o-mini"
    elif prov == "google":
        assert os.environ.get("GOOGLE_API_KEY"), \
            "Set GOOGLE_API_KEY env var for gemini-flash"
