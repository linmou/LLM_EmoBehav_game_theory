# File: tests/test_openai_settings.py
# Purpose: Sanity-check OpenAI settings (base_url/api_key) and connectivity using LangChain ChatOpenAI.
#
# We try the same client stack the code uses:
# - Instantiate ChatOpenAI with api_configs.OAI_CONFIG (base_url/api_key)
# - Call llm.invoke(\"ping\") and accept any non-connection error (e.g., 401/403/404/400).
# - Fail only on DNS/connection/socket/timeout errors.

from __future__ import annotations

import os
from urllib.parse import urljoin

import httpx
import pytest


def test_openai_connectivity_chat_completions():
    # Load endpoint config
    from api_configs import OAI_CONFIG  # type: ignore

    base_url = (OAI_CONFIG or {}).get("base_url")
    api_key = (OAI_CONFIG or {}).get("api_key")

    assert isinstance(base_url, str) and base_url.startswith(
        "http"
    ), f"Invalid or missing base_url in OAI_CONFIG: {base_url!r}"
    assert isinstance(api_key, str) and len(api_key) > 0, "Missing api_key in OAI_CONFIG"

    # First: try LangChain ChatOpenAI to simulate runtime
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        # Instantiate with explicit creds and base; small timeout, no retries
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0.0,
            api_key=api_key,
            base_url=base_url,
            timeout=8.0,
            max_retries=0,
        )
        # Minimal invoke; we accept non-connection failures
        try:
            _ = llm.invoke("ping")
            return  # success
        except Exception as e:
            msg = str(e).lower()
            # Connection/DNS layer failures should fail the test
            connection_markers = [
                "name or service not known",
                "connecterror",
                "connection error",
                "connection refused",
                "timed out",
                "dns",
            ]
            if any(m in msg for m in connection_markers):
                pytest.fail(f"LangChain ChatOpenAI connectivity failed: {e}")
            # Otherwise, endpoint reachable but request rejected (auth/format) -> skip
            pytest.skip(f"LangChain reachable; non-connection error: {e}")
    except ImportError:
        # Fallback: use httpx reachability check against /chat/completions
        url = urljoin(base_url.rstrip("/") + "/", "chat/completions")
        headers = {"Authorization": f"Bearer {api_key}"}
        json_payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(url, headers=headers, json=json_payload)
        except Exception as e:
            # Differentiate DNS vs endpoint mismatch
            try:
                with httpx.Client(timeout=5.0) as client:
                    client.get(base_url, headers=headers)
            except Exception as e2:
                pytest.fail(f"Connection to OpenAI base_url failed: {base_url} -> {e2}")
            else:
                pytest.skip(f"Base URL reachable, but /chat/completions not accepted: {e}")
        # Accept common non-success responses that still prove reachability
        acceptable = {200, 400, 401, 403, 404, 405}
        assert resp.status_code in acceptable, f"Unexpected HTTP status {resp.status_code} from {url}"
        if resp.status_code in (401, 403):
            pytest.skip(f"Connectivity OK, but authorization rejected (status={resp.status_code}).")
        if resp.status_code in (400, 404, 405):
            pytest.skip(f"Connectivity OK, endpoint responded (status={resp.status_code}).")
