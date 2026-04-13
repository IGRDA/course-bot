"""Unit tests for agents/video_html_generator/agent.py"""

import json

import pytest

from agents.video_html_generator.agent import (
    simplify_html_element,
    simplify_module,
    simplify_module_from_path,
    simplify_section,
    simplify_submodule,
)

# ---- Fixtures ----


def make_html_element(type_="p", content="Hello"):
    return {"type": type_, "content": content}


def make_section_dict(**kwargs):
    defaults = {
        "title": "Intro Section",
        "index": 1,
        "description": "First section",
        "theory": "Some theory text",
        "html": [make_html_element("p", "paragraph text")],
        "activities": None,
    }
    defaults.update(kwargs)
    return defaults


def make_submodule_dict(**kwargs):
    defaults = {
        "title": "Submodule A",
        "index": 0,
        "description": "Sub desc",
        "duration": 1.5,
        "sections": [make_section_dict()],
    }
    defaults.update(kwargs)
    return defaults


def make_module_dict(**kwargs):
    defaults = {
        "title": "Module 1",
        "id": "mod1",
        "index": 0,
        "description": "Module description",
        "duration": 3.0,
        "type": "module",
        "submodules": [make_submodule_dict()],
        "video": None,
        "bibliography": None,
    }
    defaults.update(kwargs)
    return defaults


# ---- simplify_html_element ----


class TestSimplifyHtmlElement:
    def test_non_list_content_unchanged(self):
        el = {"type": "p", "content": "plain text"}
        result = simplify_html_element(el)
        assert result["content"] == "plain text"

    def test_list_content_truncated_to_first(self):
        el = {"type": "accordion", "content": [{"title": "A"}, {"title": "B"}, {"title": "C"}]}
        result = simplify_html_element(el)
        assert result["content"] == [{"title": "A"}]

    def test_empty_list_content_unchanged(self):
        el = {"type": "accordion", "content": []}
        result = simplify_html_element(el)
        assert result["content"] == []

    def test_all_fields_copied(self):
        el = {"type": "p", "content": "text", "extra_field": "value"}
        result = simplify_html_element(el)
        assert result["extra_field"] == "value"

    def test_does_not_mutate_original(self):
        el = {"type": "tabs", "content": [1, 2, 3]}
        simplify_html_element(el)
        assert el["content"] == [1, 2, 3]  # original untouched


# ---- simplify_section ----


class TestSimplifySection:
    def test_keeps_title(self):
        result = simplify_section(make_section_dict(title="My Section"))
        assert result["title"] == "My Section"

    def test_keeps_index(self):
        result = simplify_section(make_section_dict(index=3))
        assert result["index"] == 3

    def test_keeps_description(self):
        result = simplify_section(make_section_dict(description="A desc"))
        assert result["description"] == "A desc"

    def test_does_not_include_theory(self):
        result = simplify_section(make_section_dict(theory="Long theory..."))
        assert "theory" not in result

    def test_does_not_include_activities(self):
        result = simplify_section(make_section_dict())
        assert "activities" not in result

    def test_html_elements_simplified(self):
        html = [
            {"type": "accordion", "content": [{"title": "A"}, {"title": "B"}]},
        ]
        result = simplify_section(make_section_dict(html=html))
        assert len(result["html"]) == 1
        assert len(result["html"][0]["content"]) == 1

    def test_missing_html_becomes_empty_list(self):
        section = {"title": "No HTML"}
        result = simplify_section(section)
        assert result["html"] == []

    def test_none_html_becomes_empty_list(self):
        result = simplify_section(make_section_dict(html=None))
        assert result["html"] == []


# ---- simplify_submodule ----


class TestSimplifySubmodule:
    def test_keeps_title(self):
        result = simplify_submodule(make_submodule_dict(title="Sub X"))
        assert result["title"] == "Sub X"

    def test_keeps_index(self):
        result = simplify_submodule(make_submodule_dict(index=2))
        assert result["index"] == 2

    def test_keeps_duration(self):
        result = simplify_submodule(make_submodule_dict(duration=2.5))
        assert result["duration"] == 2.5

    def test_sections_simplified(self):
        submod = make_submodule_dict(sections=[make_section_dict(), make_section_dict(title="S2")])
        result = simplify_submodule(submod)
        assert len(result["sections"]) == 2

    def test_empty_sections(self):
        result = simplify_submodule(make_submodule_dict(sections=[]))
        assert result["sections"] == []


# ---- simplify_module ----


class TestSimplifyModule:
    def test_keeps_title(self):
        result = simplify_module(make_module_dict(title="Module X"))
        assert result["title"] == "Module X"

    def test_keeps_id(self):
        result = simplify_module(make_module_dict(id="m42"))
        assert result["id"] == "m42"

    def test_keeps_index(self):
        result = simplify_module(make_module_dict(index=5))
        assert result["index"] == 5

    def test_keeps_type(self):
        result = simplify_module(make_module_dict(type="module"))
        assert result["type"] == "module"

    def test_does_not_include_video(self):
        result = simplify_module(make_module_dict(video={"url": "http://..."}))
        assert "video" not in result

    def test_does_not_include_bibliography(self):
        result = simplify_module(make_module_dict(bibliography={"books": []}))
        assert "bibliography" not in result

    def test_submodules_simplified(self):
        mod = make_module_dict(submodules=[make_submodule_dict(), make_submodule_dict(title="Sub2")])
        result = simplify_module(mod)
        assert len(result["submodules"]) == 2

    def test_missing_fields_use_defaults(self):
        result = simplify_module({})
        assert result["title"] == ""
        assert result["id"] == ""
        assert result["index"] == 0
        assert result["type"] == "module"


# ---- simplify_module_from_path ----


class TestSimplifyModuleFromPath:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            simplify_module_from_path(tmp_path / "nonexistent.json")

    def test_reads_and_simplifies(self, tmp_path):
        module_data = make_module_dict()
        input_file = tmp_path / "module_0.json"
        input_file.write_text(json.dumps(module_data), encoding="utf-8")

        output_path = simplify_module_from_path(input_file, output_dir=tmp_path / "out")

        assert output_path.exists()
        result = json.loads(output_path.read_text())
        assert result["title"] == module_data["title"]
        assert "video" not in result

    def test_default_output_dir(self, tmp_path):
        module_data = make_module_dict()
        input_file = tmp_path / "module_1.json"
        input_file.write_text(json.dumps(module_data), encoding="utf-8")

        output_path = simplify_module_from_path(input_file)

        expected_dir = tmp_path / "video_html_generator"
        assert output_path.parent == expected_dir
        assert output_path.name == "module_1.json"

    def test_output_is_valid_json(self, tmp_path):
        module_data = make_module_dict()
        input_file = tmp_path / "module_2.json"
        input_file.write_text(json.dumps(module_data), encoding="utf-8")

        output_path = simplify_module_from_path(input_file, output_dir=tmp_path / "out")

        # Should parse without error
        data = json.loads(output_path.read_text())
        assert isinstance(data, dict)
