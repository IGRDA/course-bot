"""Unit tests for workflows/output_manager.py"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out the langchain-dependent exporter before importing output_manager.
# workflows/output_manager.py does: from agents.html_formatter.exporter import export_to_html
# That import chain requires langchain (not installed in the test environment).
# We inject a fake module into sys.modules to break the dependency.
# ---------------------------------------------------------------------------
_fake_exporter = types.ModuleType("agents.html_formatter.exporter")
_fake_exporter.export_to_html = MagicMock(return_value=None)
sys.modules.setdefault("agents.html_formatter.exporter", _fake_exporter)

# Also stub the __init__ chains that would be triggered
_fake_html_formatter = types.ModuleType("agents.html_formatter")
_fake_html_formatter.export_to_html = _fake_exporter.export_to_html
sys.modules.setdefault("agents.html_formatter", _fake_html_formatter)

from workflows.output_manager import OutputManager


class TestOutputManagerInit:
    def test_creates_run_folder(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="Test Course", output_dir=str(tmp_path))
        assert os.path.isdir(manager.run_folder)

    def test_creates_steps_subfolder(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="Test Course", output_dir=str(tmp_path))
        assert os.path.isdir(manager.steps_folder)
        assert manager.steps_folder.endswith("steps")

    def test_run_folder_contains_title(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="My Test Course", output_dir=str(tmp_path))
        folder_name = os.path.basename(manager.run_folder)
        assert "My_Test_Course" in folder_name

    def test_run_folder_is_timestamped(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="Course", output_dir=str(tmp_path))
        folder_name = os.path.basename(manager.run_folder)
        # Timestamp format: YYYYMMDD_HHMMSS (14 digits + underscore)
        import re

        assert re.search(r"\d{8}_\d{6}", folder_name)

    def test_special_chars_in_title_sanitized(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="Course: <Advanced>!", output_dir=str(tmp_path))
        folder_name = os.path.basename(manager.run_folder)
        assert "<" not in folder_name
        assert ">" not in folder_name


class TestOutputManagerStepNames:
    def test_step_names_dict_not_empty(self):
        assert len(OutputManager.STEP_NAMES) > 0

    def test_known_steps_present(self):
        step_names = OutputManager.STEP_NAMES
        for step in ["index", "theories", "activities", "html", "images"]:
            assert step in step_names

    def test_step_numbers_are_unique(self):
        numbers = list(OutputManager.STEP_NAMES.values())
        assert len(numbers) == len(set(numbers))

    def test_step_numbers_are_positive(self):
        for name, num in OutputManager.STEP_NAMES.items():
            assert num > 0, f"Step '{name}' has non-positive number {num}"


class TestSaveStep:
    def _make_manager(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            return OutputManager(title="Test", output_dir=str(tmp_path))

    def test_unknown_step_raises(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        with pytest.raises(ValueError, match="Unknown step name"):
            manager.save_step("nonexistent_step", state)

    def test_known_step_creates_file(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        state.model_dump_json.return_value = '{"test": true}'
        with patch.object(manager, "save_final"), patch.object(manager, "save_modules"):
            path = manager.save_step("index", state)
        assert os.path.exists(path)

    def test_step_file_named_correctly(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        state.model_dump_json.return_value = "{}"
        with patch.object(manager, "save_final"), patch.object(manager, "save_modules"):
            path = manager.save_step("index", state)
        filename = os.path.basename(path)
        step_num = OutputManager.STEP_NAMES["index"]
        assert filename == f"{step_num:02d}_index.json"


class TestGetStepPath:
    def _make_manager(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            return OutputManager(title="Test", output_dir=str(tmp_path))

    def test_known_step_returns_path(self, tmp_path):
        manager = self._make_manager(tmp_path)
        path = manager.get_step_path("theories")
        step_num = OutputManager.STEP_NAMES["theories"]
        assert path.endswith(f"{step_num:02d}_theories.json")

    def test_unknown_step_raises(self, tmp_path):
        manager = self._make_manager(tmp_path)
        with pytest.raises(ValueError, match="Unknown step name"):
            manager.get_step_path("fake_step")

    def test_path_is_inside_steps_folder(self, tmp_path):
        manager = self._make_manager(tmp_path)
        path = manager.get_step_path("html")
        assert path.startswith(manager.steps_folder)


class TestGetRunFolder:
    def test_returns_string(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="Test", output_dir=str(tmp_path))
        assert isinstance(manager.get_run_folder(), str)

    def test_matches_run_folder(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            manager = OutputManager(title="Test", output_dir=str(tmp_path))
        assert manager.get_run_folder() == manager.run_folder


class TestSaveModules:
    def _make_manager(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            return OutputManager(title="Test", output_dir=str(tmp_path))

    def test_empty_modules_returns_empty_list(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        state.modules = []
        result = manager.save_modules(state)
        assert result == []

    def test_saves_one_module_file(self, tmp_path):
        manager = self._make_manager(tmp_path)
        mock_module = MagicMock()
        mock_module.model_dump_json.return_value = '{"title": "mod0"}'
        state = MagicMock()
        state.modules = [mock_module]
        paths = manager.save_modules(state)
        assert len(paths) == 1
        assert os.path.exists(paths[0])
        assert "module_0.json" in paths[0]

    def test_saves_multiple_module_files(self, tmp_path):
        manager = self._make_manager(tmp_path)
        modules = []
        for i in range(3):
            m = MagicMock()
            m.model_dump_json.return_value = f'{{"title": "mod{i}"}}'
            modules.append(m)
        state = MagicMock()
        state.modules = modules
        paths = manager.save_modules(state)
        assert len(paths) == 3
        for i, path in enumerate(paths):
            assert f"module_{i}.json" in path


class TestSaveFinal:
    def _make_manager(self, tmp_path):
        with patch("workflows.output_manager.export_to_html"):
            return OutputManager(title="Test", output_dir=str(tmp_path))

    def test_saves_course_json(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        state.model_dump_json.return_value = '{"title": "test course"}'
        with patch("workflows.output_manager.export_to_html"):
            json_path, _html_path = manager.save_final(state)
        assert os.path.exists(json_path)
        assert json_path.endswith("course.json")

    def test_calls_export_to_html(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        state.model_dump_json.return_value = "{}"
        with patch("workflows.output_manager.export_to_html") as mock_export:
            manager.save_final(state)
        mock_export.assert_called_once()

    def test_returns_tuple_of_paths(self, tmp_path):
        manager = self._make_manager(tmp_path)
        state = MagicMock()
        state.model_dump_json.return_value = "{}"
        with patch("workflows.output_manager.export_to_html"):
            result = manager.save_final(state)
        assert len(result) == 2
        json_path, html_path = result
        assert "course.json" in json_path
        assert "course.html" in html_path


class TestLoadStep:
    def test_loads_course_state(self, tmp_path):
        from workflows.config import CourseConfig
        from workflows.output_manager import load_step

        # Create a minimal valid CourseState JSON
        config = CourseConfig(title="Test")
        from workflows.state import CourseState

        state = CourseState(config=config, title="Load Test")
        json_data = state.model_dump_json(indent=2)

        step_file = tmp_path / "01_index.json"
        step_file.write_text(json_data, encoding="utf-8")

        loaded = load_step(str(step_file))
        assert loaded.title == "Load Test"
