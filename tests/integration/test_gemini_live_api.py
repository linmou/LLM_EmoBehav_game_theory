# Integration test for live Gemini API validity using GEMINI_CONFIG from api_configs.py.
import os
import concurrent.futures

import pytest
import requests

from api_configs import GEMINI_CONFIG


@pytest.mark.skipif(
    not os.getenv("RUN_GEMINI_LIVE_TEST"),
    reason="Set RUN_GEMINI_LIVE_TEST=1 to exercise the live Gemini API.",
)
def test_gemini_api_key_is_valid():
    # Use ListModels to validate the API key itself, independent of a specific model name.
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": GEMINI_CONFIG["api_key"]}
    response = requests.get(url, params=params, timeout=20)
    assert response.status_code == 200, (
        f"Gemini ListModels call failed with status {response.status_code}: {response.text}"
    )

    data = response.json()
    assert "models" in data and data["models"], "No models returned from Gemini."


@pytest.mark.skipif(
    not os.getenv("RUN_GEMINI_LIVE_TEST"),
    reason="Set RUN_GEMINI_LIVE_TEST=1 to exercise the live Gemini API.",
)
def test_gemini_listmodels_concurrent():
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": GEMINI_CONFIG["api_key"]}

    def single_call():
        r = requests.get(url, params=params, timeout=20)
        return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        statuses = list(executor.map(lambda _: single_call(), range(5)))

    assert all(code == 200 for code in statuses), f"Unexpected status codes: {statuses}"


@pytest.mark.skipif(
    not os.getenv("RUN_GEMINI_LIVE_TEST"),
    reason="Set RUN_GEMINI_LIVE_TEST=1 to exercise the live Gemini API.",
)
def test_gemini_long_prompt_generate_content():
    # Use a current model that supports generateContent; ignore the possibly stale name in GEMINI_CONFIG.
    model = "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    params = {"key": GEMINI_CONFIG["api_key"]}

    # Build a long input (~tens of thousands of characters) to exercise context handling.
    base = "This is a long test paragraph for Gemini. " * 50
    long_text = base * 40  # Adjust to keep runtime reasonable while still non-trivial.

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": long_text},
                ]
            }
        ]
    }

    response = requests.post(url, params=params, json=payload, timeout=60)

    assert response.status_code == 200, (
        f"Gemini long-prompt call failed with status {response.status_code}: {response.text}"
    )

    data = response.json()
    assert "candidates" in data and data["candidates"], "No candidates returned for long prompt."


@pytest.mark.skipif(
    not os.getenv("RUN_GEMINI_LIVE_TEST"),
    reason="Set RUN_GEMINI_LIVE_TEST=1 to exercise the live Gemini API.",
)
def test_gemini_long_prompt_generate_content_concurrent():
    model = "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    params = {"key": GEMINI_CONFIG["api_key"]}

    base = "This is a long test paragraph for Gemini. " * 50
    long_text = base * 40
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": long_text},
                ]
            }
        ]
    }

    def single_call():
        r = requests.post(url, params=params, json=payload, timeout=60)
        return r.status_code, r.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: single_call(), range(5)))

    statuses = [s for s, _ in results]
    assert all(code == 200 for code in statuses), f"Unexpected status codes: {statuses}"

    for _, data in results:
        assert "candidates" in data and data["candidates"], "No candidates returned for long prompt (concurrent)."
