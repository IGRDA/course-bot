"""Unit tests for LLMs/imagetext2text/factory.py"""

import pytest
from LLMs.imagetext2text.factory import available_vision_llms, create_vision_llm, resolve_vision_model_name


class TestAvailableVisionLlms:
    def test_returns_list(self):
        result = available_vision_llms()
        assert isinstance(result, list)

    def test_pixtral_in_list(self):
        assert "pixtral" in available_vision_llms()


class TestCreateVisionLlm:
    def test_empty_provider_raises(self):
        with pytest.raises(ValueError):
            create_vision_llm("")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_vision_llm("dalle")

    def test_unknown_provider_error_lists_available(self):
        try:
            create_vision_llm("unknown")
        except ValueError as e:
            assert "pixtral" in str(e)


class TestResolveVisionModelName:
    def test_empty_returns_none(self):
        assert resolve_vision_model_name("") is None

    def test_unknown_returns_none(self):
        assert resolve_vision_model_name("nonexistent") is None

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("PIXTRAL_MODEL_NAME", "pixtral-test")
        result = resolve_vision_model_name("pixtral")
        assert result == "pixtral-test"

    def test_returns_none_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("PIXTRAL_MODEL_NAME", raising=False)
        result = resolve_vision_model_name("pixtral")
        assert result is None
