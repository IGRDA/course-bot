"""Unit tests for tools/json2book/factory.py"""

from pathlib import Path

import pytest

from tools.json2book.factory import available_templates, get_template_path


class TestAvailableTemplates:
    def test_returns_list(self):
        assert isinstance(available_templates(), list)

    def test_contains_academic(self):
        templates = available_templates()
        assert "academic" in templates

    def test_not_empty(self):
        assert len(available_templates()) > 0


class TestGetTemplatePath:
    def test_academic_template_returns_path(self):
        path = get_template_path("academic")
        assert isinstance(path, Path)

    def test_path_ends_with_tex(self):
        path = get_template_path("academic")
        assert path.suffix == ".tex"

    def test_case_insensitive(self):
        path = get_template_path("ACADEMIC")
        assert isinstance(path, Path)

    def test_unknown_template_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template_path("nonexistent_template")

    def test_error_message_lists_available(self):
        try:
            get_template_path("badtemplate")
        except ValueError as e:
            msg = str(e)
            assert "academic" in msg
