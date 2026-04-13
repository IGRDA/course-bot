"""Unit tests for LLMs/text2text/factory.py"""

from unittest.mock import MagicMock, patch

import pytest
from LLMs.text2text.factory import available_text_llms, create_text_llm, resolve_text_model_name


class TestAvailableTextLlms:
    def test_returns_list(self):
        result = available_text_llms()
        assert isinstance(result, list)

    def test_returns_five_providers(self):
        assert len(available_text_llms()) == 5

    def test_known_providers_present(self):
        providers = available_text_llms()
        for expected in ["deepseek", "gemini", "groq", "mistral", "openai"]:
            assert expected in providers

    def test_sorted(self):
        providers = available_text_llms()
        assert providers == sorted(providers)


class TestCreateTextLlm:
    def test_empty_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Provider is required"):
            create_text_llm("")

    def test_none_provider_raises_value_error(self):
        with pytest.raises(ValueError):
            create_text_llm(None)  # type: ignore

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_text_llm("fakeai")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_text_llm("nonexistent")
        except ValueError as e:
            # Error message should hint at available providers
            msg = str(e)
            assert any(p in msg for p in ["mistral", "gemini", "openai"])


class TestResolveTextModelName:
    def test_none_provider_returns_none(self):
        assert resolve_text_model_name("") is None

    def test_unknown_provider_returns_none(self):
        assert resolve_text_model_name("unknown_provider") is None

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_MODEL_NAME", "mistral-test-model")
        result = resolve_text_model_name("mistral")
        assert result == "mistral-test-model"

    def test_returns_none_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
        result = resolve_text_model_name("openai")
        assert result is None

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL_NAME", "groq-test")
        result = resolve_text_model_name("GROQ")
        assert result == "groq-test"


class TestCreateTextLlmBuilders:
    """Test each provider's builder is reached via create_text_llm.

    The factory uses lazy imports inside _get_builder(), so we patch _get_builder
    to return a mock callable instead of trying to patch the not-yet-imported modules.
    """

    def _call_with_mock_builder(self, provider: str):
        mock_llm = MagicMock()
        mock_builder = MagicMock(return_value=mock_llm)
        with patch("LLMs.text2text.factory._get_builder", return_value=mock_builder):
            result = create_text_llm(provider)
        assert result is mock_llm
        return mock_builder

    def test_mistral_builder_reached(self):
        builder = self._call_with_mock_builder("mistral")
        builder.assert_called_once()

    def test_openai_builder_reached(self):
        builder = self._call_with_mock_builder("openai")
        builder.assert_called_once()

    def test_groq_builder_reached(self):
        builder = self._call_with_mock_builder("groq")
        builder.assert_called_once()

    def test_gemini_builder_reached(self):
        builder = self._call_with_mock_builder("gemini")
        builder.assert_called_once()

    def test_deepseek_builder_reached(self):
        builder = self._call_with_mock_builder("deepseek")
        builder.assert_called_once()
