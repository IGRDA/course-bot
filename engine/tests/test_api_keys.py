"""Unit tests for LLMs/api_keys.py"""

import os
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from LLMs.api_keys import parse_api_keys, get_random_key, mask_key, _load_all_keys_from_secrets


class TestParseApiKeys:
    def test_none_returns_empty(self):
        assert parse_api_keys(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_api_keys("") == []

    def test_single_key(self):
        assert parse_api_keys("sk-abc123") == ["sk-abc123"]

    def test_comma_separated(self):
        result = parse_api_keys("key1,key2,key3")
        assert result == ["key1", "key2", "key3"]

    def test_whitespace_stripped(self):
        result = parse_api_keys("  key1 , key2  ,key3")
        assert result == ["key1", "key2", "key3"]

    def test_empty_parts_ignored(self):
        result = parse_api_keys("key1,,key2,")
        assert result == ["key1", "key2"]

    def test_only_commas_returns_empty(self):
        assert parse_api_keys(",,,") == []


class TestGetRandomKey:
    def test_single_element_list(self):
        keys = ["only-key"]
        assert get_random_key(keys) == "only-key"

    def test_returns_element_from_list(self):
        keys = ["key1", "key2", "key3"]
        result = get_random_key(keys)
        assert result in keys

    def test_repeated_calls_can_return_different_values(self):
        keys = ["key1", "key2", "key3", "key4", "key5"]
        results = {get_random_key(keys) for _ in range(50)}
        # With 50 samples from 5 keys, very unlikely to always pick the same one
        assert len(results) > 1


class TestMaskKey:
    def test_short_key_returns_stars(self):
        assert mask_key("abc") == "***"

    def test_exactly_4_chars_returns_stars(self):
        assert mask_key("abcd") == "***"

    def test_normal_key_shows_last_4(self):
        assert mask_key("sk-abcdefgh1234") == "...1234"

    def test_5_chars_shows_last_4(self):
        assert mask_key("ab123") == "...b123"


class TestLoadAllKeysFromSecrets:
    def test_falls_back_to_env_when_no_file(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_API_KEY", "env-key1,env-key2")
        # Patch the secrets file path to a non-existent file
        with patch("LLMs.api_keys._SECRETS_FILE", Path("/nonexistent/env.secrets")):
            result = _load_all_keys_from_secrets("MY_TEST_API_KEY")
        assert result == ["env-key1", "env-key2"]

    def test_reads_from_secrets_file(self, tmp_path):
        secrets = tmp_path / "env.secrets"
        secrets.write_text("export MY_KEY=key-from-file1,key-from-file2\n")
        with patch("LLMs.api_keys._SECRETS_FILE", secrets):
            result = _load_all_keys_from_secrets("MY_KEY")
        assert result == ["key-from-file1", "key-from-file2"]

    def test_missing_var_in_file_falls_back_to_env(self, tmp_path, monkeypatch):
        secrets = tmp_path / "env.secrets"
        secrets.write_text("export OTHER_KEY=other-value\n")
        monkeypatch.setenv("MY_KEY", "from-env")
        with patch("LLMs.api_keys._SECRETS_FILE", secrets):
            result = _load_all_keys_from_secrets("MY_KEY")
        assert result == ["from-env"]

    def test_missing_env_var_returns_empty(self, tmp_path, monkeypatch):
        secrets = tmp_path / "env.secrets"
        secrets.write_text("")
        monkeypatch.delenv("NONEXISTENT_KEY_XYZ", raising=False)
        with patch("LLMs.api_keys._SECRETS_FILE", secrets):
            result = _load_all_keys_from_secrets("NONEXISTENT_KEY_XYZ")
        assert result == []
