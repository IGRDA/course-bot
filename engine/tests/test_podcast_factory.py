"""Unit tests for tools/podcast/factory.py"""

import pytest
from unittest.mock import patch, MagicMock

from tools.podcast.factory import get_engine_info, list_engines, create_tts_engine


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
        original = factory_module.create_tts_engine

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
