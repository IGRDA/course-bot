"""Unit tests for workflows/state.py Pydantic models."""

import pytest
from pydantic import ValidationError

from workflows.config import CourseConfig
from workflows.state import (
    Activity,
    BookReference,
    CourseBibliography,
    CourseState,
    GlossaryTerm,
    HtmlElement,
    MetaElements,
    MindmapNode,
    MindmapNodeData,
    MindmapRelation,
    MindmapRelationData,
    Module,
    ModuleMindmap,
    ParagraphBlock,
    PersonReference,
    Section,
    Submodule,
    VideoReference,
)

# ---- Helpers ----


def make_course_config(**kwargs):
    return CourseConfig(title="Test Course", **kwargs)


def make_section(**kwargs):
    defaults = {"title": "Intro", "index": 0}
    defaults.update(kwargs)
    return Section(**defaults)


def make_submodule(**kwargs):
    defaults = {"title": "Sub 1", "index": 0, "sections": [make_section()]}
    defaults.update(kwargs)
    return Submodule(**defaults)


def make_module(**kwargs):
    defaults = {"title": "Module 1", "index": 0, "submodules": [make_submodule()]}
    defaults.update(kwargs)
    return Module(**defaults)


# ---- BookReference ----


class TestBookReference:
    def test_isbn13_dedup_key(self):
        book = BookReference(title="Python Book", isbn_13="9780134444321")
        assert book.get_dedup_key() == "isbn13:9780134444321"

    def test_isbn_dedup_key_when_no_isbn13(self):
        book = BookReference(title="Python Book", isbn="0134444321")
        assert book.get_dedup_key() == "isbn:0134444321"

    def test_title_author_fallback(self):
        book = BookReference(title="Python Book", authors=["Smith, J."])
        key = book.get_dedup_key()
        assert key.startswith("title:")
        assert "smith" in key
        assert "python book" in key

    def test_title_no_author_fallback(self):
        book = BookReference(title="Orphan Book")
        key = book.get_dedup_key()
        assert "unknown" in key

    def test_isbn13_takes_priority_over_isbn(self):
        book = BookReference(title="T", isbn="111", isbn_13="9780000000001")
        assert book.get_dedup_key().startswith("isbn13:")


class TestCourseBibliography:
    def test_get_all_dedup_keys_empty(self):
        bib = CourseBibliography()
        assert bib.get_all_dedup_keys() == set()

    def test_get_all_dedup_keys(self):
        book1 = BookReference(title="A", isbn_13="111")
        book2 = BookReference(title="B", isbn_13="222")
        bib = CourseBibliography(all_books=[book1, book2])
        keys = bib.get_all_dedup_keys()
        assert len(keys) == 2
        assert "isbn13:111" in keys
        assert "isbn13:222" in keys


# ---- HtmlElement validation ----


class TestHtmlElement:
    def test_paragraph_valid(self):
        el = HtmlElement(type="p", content="Hello world")
        assert el.type == "p"

    def test_paragraph_invalid_not_string(self):
        with pytest.raises(ValidationError):
            HtmlElement(type="p", content=["not", "a", "string"])

    def test_ul_valid(self):
        el = HtmlElement(type="ul", content=["item1", "item2"])
        assert el.type == "ul"

    def test_ul_invalid_not_list(self):
        with pytest.raises(ValidationError):
            HtmlElement(type="ul", content="not a list")

    def test_quote_valid(self):
        el = HtmlElement(type="quote", content={"author": "Einstein", "text": "E=mc2"})
        assert el.type == "quote"

    def test_quote_invalid_not_dict(self):
        with pytest.raises(ValidationError):
            HtmlElement(type="quote", content="not a dict")

    def test_table_valid(self):
        el = HtmlElement(type="table", content={"headers": ["col1"], "rows": []})
        assert el.type == "table"

    def test_paragraphs_valid_with_blocks(self):
        block = ParagraphBlock(
            title="Block 1",
            icon="mdi-book",
            image=None,
            elements=[HtmlElement(type="p", content="text")],
        )
        el = HtmlElement(type="paragraphs", content=[block])
        assert el.type == "paragraphs"

    def test_accordion_invalid_not_list(self):
        with pytest.raises(ValidationError):
            HtmlElement(type="accordion", content="not a list")

    def test_all_interactive_types_accept_list(self):
        interactive_types = ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]
        block = ParagraphBlock(
            title="T",
            icon="i",
            image=None,
            elements=[HtmlElement(type="p", content="x")],
        )
        for t in interactive_types:
            el = HtmlElement(type=t, content=[block])
            assert el.type == t


# ---- Section / Submodule / Module ----


class TestSection:
    def test_defaults(self):
        s = Section(title="Intro")
        assert s.index == 0
        assert s.theory == ""
        assert s.html is None
        assert s.activities is None

    def test_custom_fields(self):
        s = Section(title="Advanced", index=2, description="desc", theory="content")
        assert s.title == "Advanced"
        assert s.index == 2
        assert s.theory == "content"


class TestSubmodule:
    def test_requires_sections(self):
        sub = Submodule(title="Sub", sections=[make_section()])
        assert len(sub.sections) == 1

    def test_defaults(self):
        sub = Submodule(title="Sub", sections=[])
        assert sub.index == 0
        assert sub.duration == 0.0


class TestModule:
    def test_requires_submodules(self):
        mod = Module(title="Mod", submodules=[make_submodule()])
        assert len(mod.submodules) == 1

    def test_defaults(self):
        mod = Module(title="Mod", submodules=[])
        assert mod.type == "module"
        assert mod.index == 0
        assert mod.video is None
        assert mod.bibliography is None
        assert mod.relevant_people is None
        assert mod.mindmap is None


class TestCourseState:
    def test_construction(self):
        config = make_course_config()
        state = CourseState(config=config, title="My Course")
        assert state.title == "My Course"
        assert state.modules == []

    def test_language_property(self):
        config = make_course_config(language="Spanish")
        state = CourseState(config=config, title="Course")
        assert state.language == "Spanish"

    def test_description_property(self):
        config = make_course_config(description="A great course")
        state = CourseState(config=config, title="Course")
        assert state.description == "A great course"

    def test_concurrency_property(self):
        config = make_course_config(concurrency=4)
        state = CourseState(config=config, title="Course")
        assert state.concurrency == 4

    def test_max_retries_property(self):
        config = make_course_config(max_retries=5)
        state = CourseState(config=config, title="Course")
        assert state.max_retries == 5


class TestHtmlElementInvalidBlock:
    def test_invalid_block_type_in_interactive_raises(self):
        # An interactive element where content contains a non-dict, non-ParagraphBlock item
        with pytest.raises(ValidationError):
            HtmlElement(type="accordion", content=["not_a_block"])


# ---- MetaElements / Activity ----


class TestMetaElements:
    def test_defaults(self):
        meta = MetaElements()
        assert meta.glossary == []
        assert meta.key_concept == ""
        assert meta.interesting_fact == ""
        assert meta.quote is None

    def test_glossary_term(self):
        term = GlossaryTerm(term="API", explanation="Application Programming Interface")
        assert term.term == "API"


class TestActivity:
    def test_multiple_choice_activity(self):
        from workflows.state import MultipleChoiceContent

        content = MultipleChoiceContent(
            question="What is 2+2?",
            solution="4",
            other_options=["1", "2", "3"],
        )
        activity = Activity(type="multiple_choice", content=content)
        assert activity.type == "multiple_choice"

    def test_fill_gaps_activity(self):
        from workflows.state import FillGapsContent

        content = FillGapsContent(
            question="Python is a *gap* language",
            solution=["programming", "scripting"],
        )
        activity = Activity(type="fill_gaps", content=content)
        assert activity.type == "fill_gaps"


# ---- MindMap Models ----


class TestMindmapModels:
    def test_node_creation(self):
        node = MindmapNode(id="n1", level=1, data=MindmapNodeData(label="Concept"))
        assert node.id == "n1"
        assert node.data.label == "Concept"

    def test_relation_creation(self):
        rel = MindmapRelation(
            id="r1",
            source="n1",
            target="n2",
            data=MindmapRelationData(label="relates to"),
        )
        assert rel.source == "n1"

    def test_module_mindmap(self):
        node = MindmapNode(id="root", level=0, data=MindmapNodeData(label="Root"))
        mm = ModuleMindmap(moduleIdx=1, title="Module 1", nodes=[node], relations=[])
        assert mm.moduleIdx == 1
        assert len(mm.nodes) == 1


# ---- PersonReference ----


class TestPersonReference:
    def test_creation(self):
        p = PersonReference(
            name="Alan Turing",
            description="Pioneer of computer science",
            wikiUrl="https://en.wikipedia.org/wiki/Alan_Turing",
        )
        assert p.name == "Alan Turing"
        assert p.image == ""


# ---- VideoReference ----


class TestVideoReference:
    def test_creation(self):
        v = VideoReference(title="Intro to Python", url="https://youtube.com/xyz")
        assert v.title == "Intro to Python"
        assert v.duration == 0
        assert v.views == 0
