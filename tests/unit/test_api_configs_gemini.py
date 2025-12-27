# Tests for api_configs.py: ensure Gemini config is exposed with required fields.
import importlib

import pytest


def test_gemini_config_is_defined_with_required_fields():
    api_configs = importlib.import_module("api_configs")
    assert hasattr(api_configs, "GEMINI_CONFIG"), "GEMINI_CONFIG should be defined in api_configs"
    gemini_config = api_configs.GEMINI_CONFIG
    assert isinstance(gemini_config, dict), "GEMINI_CONFIG should be a dictionary"

    for field in ("api_key", "model"):
        assert field in gemini_config, f"{field} missing from GEMINI_CONFIG"
        value = gemini_config[field]
        assert isinstance(value, str) and value.strip(), f"{field} should be a non-empty string"
