"""Shared LLM client for the local OpenAI-compatible endpoint (LLM-as-judge + extend-phase
auto-fill).

Used by both the LLM-as-judge workflow (main_train_judge.py) and the extend-phase auto-fill
(iwf/fetchers/llm_fill.py). Supports structured JSON output (for impact predictions) and
plain text (for notes/abstracts). All calls are best-effort: network errors, malformed
responses, and a missing/unconfigured endpoint all return None gracefully so the pipeline
never crashes on an LLM fault.

Configured entirely from workflow/.env.workflow (NOT the repo-root .env — this is the one
deliberate exception to the "secrets live in root .env" rule, since this workflow module is
meant to be configurable standalone from .env.workflow alone):
    LLM_JUDGE_PROVIDER   documentation label (anthropic | openai | local); this client only
                         talks to an OpenAI-compatible chat-completions endpoint, so only
                         LOCAL_LLM_URL is actually read as the base URL regardless of label
    LLM_JUDGE_MODEL      model name/tag sent in the request body
    LLM_JUDGE_API_KEY    bearer token (secret — set only in YOUR .env.workflow, never commit)
    LOCAL_LLM_URL        base URL of the OpenAI-compatible endpoint, e.g.
                         https://genai.example.edu/openai or http://localhost:11434

The exact chat-completions path varies by server: standard OpenAI-compatible servers
(vLLM, LM Studio, Ollama's compat layer, ...) expose ``/v1/chat/completions``; some gateways
(observed with an Open WebUI-style university proxy) instead expose a bare
``/chat/completions`` and return 403 on the ``/v1/...`` passthrough route. This client tries
``/v1/chat/completions`` first and falls back to ``/chat/completions`` on any failure,
remembering whichever one worked for the rest of the process so later calls skip straight to
it (matters for the judge workflow, which can make thousands of calls in one run).
"""

from __future__ import annotations

import json

import requests

from workflow.src.iwf import config

_CHAT_SUFFIXES = ("/v1/chat/completions", "/chat/completions")
_working_suffix: dict[str, str] = {}   # base_url -> suffix that last worked (process-local)


def _post_chat(base_url: str, api_key: str, payload: dict) -> requests.Response:
    """POST ``payload`` to base_url + a chat-completions suffix, trying the remembered
    suffix first (or /v1/... then bare /chat/completions on a fresh base_url)."""
    remembered = _working_suffix.get(base_url)
    order = [remembered, *(s for s in _CHAT_SUFFIXES if s != remembered)] if remembered \
        else list(_CHAT_SUFFIXES)
    headers = {"Authorization": f"Bearer {api_key}"}
    last_resp = None
    for suffix in order:
        try:
            resp = requests.post(f"{base_url}{suffix}", json=payload, headers=headers,
                                 timeout=60)
        except requests.RequestException:
            continue
        if resp.ok:
            _working_suffix[base_url] = suffix
            return resp
        last_resp = resp
    if last_resp is not None:
        return last_resp
    raise requests.RequestException(f"no chat-completions endpoint reachable at {base_url}")


def complete(prompt: str, system: str | None = None, as_json: bool = False) -> str | dict | None:
    """Call the configured OpenAI-compatible LLM endpoint and return the response.

    Args:
        prompt: the user message (e.g. "Predict the impact score for this paper...")
        system: optional system prompt (e.g. "You are an expert in scientific impact...")
        as_json: if True, request JSON mode and parse the response as JSON (or None on error)

    Returns:
        str or dict if successful; None on any error (network, timeout, malformed, missing
        config). When as_json=True and parsing succeeds, returns a dict; otherwise a string.
    """
    base_url = config.get_str("LOCAL_LLM_URL", "").strip().rstrip("/")
    api_key = config.api_key("LLM_JUDGE_API_KEY")
    model_name = config.get_str("LLM_JUDGE_MODEL", "").strip()

    if not (base_url and api_key and model_name):
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
    }
    if as_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = _post_chat(base_url, api_key, payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return None
        if as_json:
            stripped = content
            if stripped.startswith("```json"):
                stripped = stripped[7:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()
            return json.loads(stripped)
        return content
    except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
