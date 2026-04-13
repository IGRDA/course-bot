"""Unit tests for tools/podcast/models.py"""

import pytest

from tools.podcast.models import (
    Message,
    Conversation,
    LanguageConfig,
    get_language_config,
    LANGUAGE_CONFIGS,
    XTTS_SPEAKERS,
)


class TestMessage:
    def test_valid_message(self):
        msg = Message(role="host", content="Welcome to the show!")
        assert msg.role == "host"
        assert msg.content == "Welcome to the show!"

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content cannot be empty"):
            Message(role="host", content="")

    def test_whitespace_content_raises(self):
        with pytest.raises(ValueError, match="content cannot be empty"):
            Message(role="guest", content="   ")

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="role cannot be empty"):
            Message(role="", content="Hello")

    def test_whitespace_role_raises(self):
        with pytest.raises(ValueError, match="role cannot be empty"):
            Message(role="  ", content="Hello")


class TestConversation:
    def _make_conversation(self):
        return Conversation.from_dicts([
            {"role": "host", "content": "Welcome!"},
            {"role": "guest", "content": "Thanks for having me."},
            {"role": "host", "content": "Let's dive in."},
        ])

    def test_from_dicts(self):
        conv = self._make_conversation()
        assert len(conv) == 3

    def test_from_dicts_with_title(self):
        conv = Conversation.from_dicts(
            [{"role": "host", "content": "Hi"}],
            title="Episode 1",
        )
        assert conv.title == "Episode 1"

    def test_default_title(self):
        conv = Conversation.from_dicts([{"role": "host", "content": "Hi"}])
        assert conv.title == "Podcast"

    def test_get_roles(self):
        conv = self._make_conversation()
        roles = conv.get_roles()
        assert roles == {"host", "guest"}

    def test_len(self):
        conv = self._make_conversation()
        assert len(conv) == 3

    def test_iter(self):
        conv = self._make_conversation()
        messages = list(conv)
        assert len(messages) == 3
        assert all(isinstance(m, Message) for m in messages)

    def test_empty_conversation(self):
        conv = Conversation()
        assert len(conv) == 0
        assert conv.get_roles() == set()

    def test_single_role(self):
        conv = Conversation.from_dicts([
            {"role": "narrator", "content": "Chapter one."},
            {"role": "narrator", "content": "Chapter two."},
        ])
        assert conv.get_roles() == {"narrator"}


class TestGetLanguageConfig:
    def test_english_config(self):
        config = get_language_config("en")
        assert config.language_code == "en"
        assert config.is_multilingual is True

    def test_spanish_config(self):
        config = get_language_config("es")
        assert config.language_code == "es"

    def test_multilingual_config(self):
        config = get_language_config("multilingual")
        assert config.is_multilingual is True

    def test_english_fast_config(self):
        config = get_language_config("en_fast")
        assert config.is_multilingual is False

    def test_invalid_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_language_config("klingon")

    def test_all_configs_have_speakers(self):
        for lang, config in LANGUAGE_CONFIGS.items():
            assert len(config.speakers) > 0, f"{lang} has no speakers"

    def test_all_configs_have_speaker_map(self):
        for lang, config in LANGUAGE_CONFIGS.items():
            assert len(config.default_speaker_map) > 0, f"{lang} has no speaker map"


class TestXttsSpeakers:
    def test_speakers_list_not_empty(self):
        assert len(XTTS_SPEAKERS) > 0

    def test_all_speakers_are_strings(self):
        assert all(isinstance(s, str) for s in XTTS_SPEAKERS)

    def test_no_duplicate_speakers(self):
        assert len(XTTS_SPEAKERS) == len(set(XTTS_SPEAKERS))
