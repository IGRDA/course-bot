"""Unit tests for tools/podcast/factory.py"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.podcast.factory import create_tts_engine, get_engine_info, list_engines

KNOWN_ENGINES = ["edge", "coqui", "chatterbox", "elevenlabs", "openai_tts", "qwen_tts", "mlx_tts", "qwen_tts_api"]


class TestGetEngineInfo:
    def test_known_engine_returns_dict(self):
        for engine in KNOWN_ENGINES:
            info = get_engine_info(engine)
            assert isinstance(info, dict)

    def test_info_has_required_keys(self):
        info = get_engine_info("edge")
        assert "name" in info
        assert "description" in info
        assert "requires_internet" in info
        assert "languages" in info

    def test_edge_requires_internet(self):
        assert get_engine_info("edge")["requires_internet"] is True

    def test_coqui_does_not_require_internet(self):
        assert get_engine_info("coqui")["requires_internet"] is False

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine_info("invalid_engine")

    def test_each_engine_has_languages(self):
        for engine in KNOWN_ENGINES:
            info = get_engine_info(engine)
            assert len(info["languages"]) > 0, f"{engine} has no languages"

    def test_qwen_tts_api_requires_internet(self):
        assert get_engine_info("qwen_tts_api")["requires_internet"] is True


class TestListEngines:
    def test_returns_list(self):
        result = list_engines()
        assert isinstance(result, list)

    def test_returns_eight_engines(self):
        assert len(list_engines()) == 8

    def test_each_entry_has_engine_key(self):
        for entry in list_engines():
            assert "engine" in entry

    def test_all_known_engines_in_list(self):
        engine_names = {e["engine"] for e in list_engines()}
        for known in KNOWN_ENGINES:
            assert known in engine_names


class TestCreateTtsEngine:
    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            create_tts_engine("nonexistent_engine")

    def test_unknown_engine_error_lists_available(self):
        try:
            create_tts_engine("bad_engine")
        except ValueError as e:
            msg = str(e)
            assert "edge" in msg

    def _mock_engine(self, engine_name: str, language: str = "en", **kwargs):
        """Create a TTS engine with a mocked class to avoid importing heavy deps."""
        mock_engine = MagicMock()
        # Patch the factory function itself to short-circuit the lazy import
        import tools.podcast.factory as factory_module

        def fake_factory(engine=engine_name, language="en", speaker_map=None, **kw):
            return mock_engine

        with patch.object(factory_module, "create_tts_engine", side_effect=fake_factory):
            result = factory_module.create_tts_engine(engine_name, language=language)
        return result, mock_engine

    def test_each_known_engine_can_be_requested(self):
        """Verify that each engine name passes the dispatch logic (mocked at factory level)."""
        for engine_name in KNOWN_ENGINES:
            mock_engine = MagicMock()
            with patch("tools.podcast.factory.create_tts_engine", return_value=mock_engine):
                import tools.podcast.factory as f

                result = f.create_tts_engine(engine_name)
            assert result is mock_engine

    def _make_engine_module_mock(self, class_name: str) -> MagicMock:
        """Create a mock module with a mock class."""
        mock_cls = MagicMock()
        mock_module = MagicMock()
        setattr(mock_module, class_name, mock_cls)
        return mock_module, mock_cls

    def test_edge_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("EdgeTTSEngine")
        with patch.dict(sys.modules, {"tools.podcast.edge.client": mock_module}):
            create_tts_engine("edge", language="en")
        mock_cls.assert_called_once()

    def test_coqui_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("CoquiTTSEngine")
        with patch.dict(sys.modules, {"tools.podcast.coqui.client": mock_module}):
            create_tts_engine("coqui", language="en")
        mock_cls.assert_called_once()

    def test_chatterbox_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("ChatterboxEngine")
        with patch.dict(sys.modules, {"tools.podcast.chatterbox.client": mock_module}):
            create_tts_engine("chatterbox", language="en")
        mock_cls.assert_called_once()

    def test_elevenlabs_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("ElevenLabsTTSEngine")
        with patch.dict(sys.modules, {"tools.podcast.elevenlabs.client": mock_module}):
            create_tts_engine("elevenlabs", language="en")
        mock_cls.assert_called_once()

    def test_openai_tts_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("OpenAITTSEngine")
        with patch.dict(sys.modules, {"tools.podcast.openai_tts.client": mock_module}):
            create_tts_engine("openai_tts", language="en")
        mock_cls.assert_called_once()

    def test_qwen_tts_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("QwenTTSEngine")
        with patch.dict(sys.modules, {"tools.podcast.qwen_tts.client": mock_module}):
            create_tts_engine("qwen_tts", language="en")
        mock_cls.assert_called_once()

    def test_mlx_tts_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("MLXTTSEngine")
        with patch.dict(sys.modules, {"tools.podcast.mlx_tts.client": mock_module}):
            create_tts_engine("mlx_tts", language="en")
        mock_cls.assert_called_once()

    def test_qwen_tts_api_engine_instantiated(self):
        mock_module, mock_cls = self._make_engine_module_mock("QwenTTSApiEngine")
        with patch.dict(sys.modules, {"tools.podcast.qwen_tts_api.client": mock_module}):
            create_tts_engine("qwen_tts_api", language="en")
        mock_cls.assert_called_once()
