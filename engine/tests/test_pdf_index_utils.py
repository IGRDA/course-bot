"""Unit tests for agents/pdf_index_generator/utils.py"""

import importlib.util
from pathlib import Path

import pytest

# Load utils.py directly, bypassing the package __init__.py which imports langchain.
_UTILS_PATH = Path(__file__).resolve().parent.parent / "agents" / "pdf_index_generator" / "utils.py"
_spec = importlib.util.spec_from_file_location("pdf_index_generator_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

compute_layout = _utils.compute_layout
compute_pages_per_module = _utils.compute_pages_per_module
compute_section_weights = _utils.compute_section_weights
compute_words_per_section = _utils.compute_words_per_section
get_module_count = _utils.get_module_count


class TestGetModuleCount:
    def test_small_course(self):
        result = get_module_count(10)
        assert isinstance(result, int)
        assert result >= 1

    def test_large_course(self):
        result = get_module_count(1000)
        assert isinstance(result, int)
        assert result <= 30

    def test_single_page(self):
        result = get_module_count(1)
        assert result >= 1

    def test_zero_pages_raises(self):
        with pytest.raises(ValueError):
            get_module_count(0)

    def test_negative_pages_raises(self):
        with pytest.raises(ValueError):
            get_module_count(-5)

    def test_result_within_bounds(self):
        for pages in [1, 10, 50, 100, 500, 1000]:
            result = get_module_count(pages)
            assert 1 <= result <= 30


class TestComputeLayout:
    def test_returns_tuple_of_three(self):
        result = compute_layout(100)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_positive(self):
        n_modules, n_submodules, n_sections = compute_layout(50)
        assert n_modules >= 1
        assert n_submodules >= 1
        assert n_sections >= 1

    def test_zero_pages_raises(self):
        with pytest.raises(ValueError):
            compute_layout(0)

    def test_negative_pages_raises(self):
        with pytest.raises(ValueError):
            compute_layout(-1)

    def test_larger_courses_have_more_modules(self):
        _, _, _ = compute_layout(10)
        n_big, _, _ = compute_layout(500)
        n_small, _, _ = compute_layout(10)
        assert n_big >= n_small


class TestComputeSectionWeights:
    def test_weights_sum_to_one(self):
        weights = compute_section_weights([2.0, 3.0, 1.0])
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_proportional_weights(self):
        weights = compute_section_weights([1.0, 2.0])
        assert abs(weights[1] - 2 * weights[0]) < 1e-9

    def test_equal_weights_when_all_zero(self):
        weights = compute_section_weights([0.0, 0.0, 0.0])
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 1e-9

    def test_single_module(self):
        weights = compute_section_weights([5.0])
        assert abs(weights[0] - 1.0) < 1e-9

    def test_returns_dict_with_correct_keys(self):
        weights = compute_section_weights([1.0, 2.0, 3.0])
        assert set(weights.keys()) == {0, 1, 2}


class TestComputePagesPerModule:
    def test_total_pages_matches(self):
        pages = compute_pages_per_module(100, [1.0, 2.0, 3.0])
        assert sum(pages.values()) == 100

    def test_all_modules_get_at_least_one_page(self):
        pages = compute_pages_per_module(10, [1.0, 1.0, 1.0])
        for p in pages.values():
            assert p >= 1

    def test_proportional_distribution(self):
        pages = compute_pages_per_module(60, [1.0, 2.0, 3.0])
        # Module 2 should have ~3x module 0's pages
        assert pages[2] > pages[0]

    def test_returns_dict(self):
        pages = compute_pages_per_module(50, [2.0, 2.0])
        assert isinstance(pages, dict)


class TestComputeWordsPerSection:
    def test_returns_dict(self):
        result = compute_words_per_section(100, 300, [2.0, 3.0], [3, 4])
        assert isinstance(result, dict)

    def test_all_modules_have_positive_words(self):
        result = compute_words_per_section(100, 300, [1.0, 2.0, 3.0], [2, 3, 4])
        for words in result.values():
            assert words >= 100

    def test_minimum_words_enforced(self):
        result = compute_words_per_section(1, 1, [1.0], [100])
        assert result[0] >= 100

    def test_sections_per_module_out_of_range(self):
        # sections_per_module shorter than module_durations — should not crash
        result = compute_words_per_section(100, 300, [1.0, 2.0], [3])
        assert isinstance(result, dict)
