"""Unit tests for LLMs/text2text/health_check.py"""

import os
import pytest
from unittest.mock import patch, MagicMock

from LLMs.text2text.health_check import (
    PROVIDER_REGISTRY,
    _check_key_openai,
    _check_key_gemini,
    validate_provider_keys,
)


class TestProviderRegistry:
    def test_expected_providers_present(self):
        for provider in ["mistral", "openai", "groq", "deepseek", "gemini"]:
            assert provider in PROVIDER_REGISTRY

    def test_each_provider_has_required_fields(self):
        required = {"env_var", "model_env_var", "base_url", "api_format"}
        for provider, config in PROVIDER_REGISTRY.items():
            for field in required:
                assert field in config, f"Provider '{provider}' missing field '{field}'"

    def test_api_formats_are_valid(self):
        valid_formats = {"openai", "gemini"}
        for provider, config in PROVIDER_REGISTRY.items():
            assert config["api_format"] in valid_formats

    def test_gemini_uses_gemini_format(self):
        assert PROVIDER_REGISTRY["gemini"]["api_format"] == "gemini"

    def test_mistral_uses_openai_format(self):
        assert PROVIDER_REGISTRY["mistral"]["api_format"] == "openai"


class TestCheckKeyOpenai:
    def test_200_status_returned(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("requests.post", return_value=mock_response):
            result = _check_key_openai("test-key", "https://api.example.com/v1", "test-model", 5.0)
        assert result == 200

    def test_401_status_returned(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch("requests.post", return_value=mock_response):
            result = _check_key_openai("bad-key", "https://api.example.com/v1", "test-model", 5.0)
        assert result == 401

    def test_timeout_returns_408(self):
        import requests
        with patch("requests.post", side_effect=requests.Timeout):
            result = _check_key_openai("key", "https://api.example.com", "model", 1.0)
        assert result == 408

    def test_request_exception_returns_zero(self):
        import requests
        with patch("requests.post", side_effect=requests.RequestException("connection error")):
            result = _check_key_openai("key", "https://api.example.com", "model", 1.0)
        assert result == 0


class TestCheckKeyGemini:
    def test_200_status_returned(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("requests.post", return_value=mock_response):
            result = _check_key_gemini("test-key", "https://generativelanguage.googleapis.com/v1beta", "gemini-pro", 5.0)
        assert result == 200

    def test_403_status_returned(self):
        mock_response = MagicMock()
        mock_response.status_code = 403
        with patch("requests.post", return_value=mock_response):
            result = _check_key_gemini("bad-key", "https://api.google.com", "model", 5.0)
        assert result == 403

    def test_timeout_returns_408(self):
        import requests
        with patch("requests.post", side_effect=requests.Timeout):
            result = _check_key_gemini("key", "https://api.google.com", "model", 1.0)
        assert result == 408

    def test_request_exception_returns_zero(self):
        import requests
        with patch("requests.post", side_effect=requests.RequestException):
            result = _check_key_gemini("key", "https://api.google.com", "model", 1.0)
        assert result == 0


class TestValidateProviderKeys:
    def _mock_check(self, status_map):
        """Return a check function that returns status based on key."""
        def check_fn(key, base_url, model, timeout):
            return status_map.get(key, 0)
        return check_fn

    def test_unknown_provider_returns_empty(self):
        result = validate_provider_keys("unknown_provider_xyz")
        assert result == []

    def test_all_keys_healthy_returns_all(self, monkeypatch):
        monkeypatch.setenv("TEST_MISTRAL_KEY_ENV", "key1,key2")
        with patch("LLMs.text2text.health_check._load_all_keys_from_secrets", return_value=["key1", "key2"]), \
             patch("LLMs.text2text.health_check._CHECK_DISPATCH", {"openai": lambda k, u, m, t: 200}), \
             patch.dict(os.environ, {"MISTRAL_MODEL_NAME": "mistral-small"}):
            result = validate_provider_keys("mistral")
        assert set(result) == {"key1", "key2"}

    def test_no_healthy_keys_raises_runtime_error(self):
        with patch("LLMs.text2text.health_check._load_all_keys_from_secrets", return_value=["bad-key"]), \
             patch("LLMs.text2text.health_check._CHECK_DISPATCH", {"openai": lambda k, u, m, t: 401}), \
             patch.dict(os.environ, {"MISTRAL_MODEL_NAME": "mistral-small"}):
            with pytest.raises(RuntimeError, match="health check"):
                validate_provider_keys("mistral")

    def test_no_keys_found_raises_runtime_error(self):
        with patch("LLMs.text2text.health_check._load_all_keys_from_secrets", return_value=[]):
            with pytest.raises(RuntimeError):
                validate_provider_keys("mistral")

    def test_only_healthy_keys_returned(self):
        with patch("LLMs.text2text.health_check._load_all_keys_from_secrets", return_value=["good", "bad"]), \
             patch("LLMs.text2text.health_check._CHECK_DISPATCH", {
                 "openai": lambda k, u, m, t: 200 if k == "good" else 401
             }), \
             patch.dict(os.environ, {"MISTRAL_MODEL_NAME": "mistral-small"}):
            result = validate_provider_keys("mistral")
        assert result == ["good"]

    def test_healthy_keys_set_in_env(self, monkeypatch):
        with patch("LLMs.text2text.health_check._load_all_keys_from_secrets", return_value=["key-a", "key-b"]), \
             patch("LLMs.text2text.health_check._CHECK_DISPATCH", {"openai": lambda k, u, m, t: 200}), \
             patch.dict(os.environ, {"MISTRAL_MODEL_NAME": "mistral-small", "MISTRAL_API_KEY": ""}):
            validate_provider_keys("mistral")
            # Assert inside the `with` block while patch.dict is still active
            assert "key-a" in os.environ.get("MISTRAL_API_KEY", "")
