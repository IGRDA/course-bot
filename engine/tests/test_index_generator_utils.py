"""Unit tests for agents/index_generator/utils.py"""

import pytest

from agents.index_generator.utils import get_module_count, compute_layout


class TestGetModuleCount:
    def test_zero_pages_raises(self):
        with pytest.raises(ValueError):
            get_module_count(0)

    def test_negative_pages_raises(self):
        with pytest.raises(ValueError):
            get_module_count(-5)

    def test_minimum_is_2(self):
        # Even 1 page should give at least 2 modules
        assert get_module_count(1) >= 2

    def test_small_course(self):
        # 10 pages → log10(10)=1 → 2 + 2 = 4
        result = get_module_count(10)
        assert 2 <= result <= 12

    def test_medium_course(self):
        # 100 pages → log10(100)=2 → 2 + 4 = 6
        result = get_module_count(100)
        assert 2 <= result <= 12

    def test_large_course(self):
        # 1000 pages → log10(1000)=3 → 2 + 6 = 8
        result = get_module_count(1000)
        assert 2 <= result <= 12

    def test_maximum_is_12(self):
        # Extremely large input should clamp to 12
        result = get_module_count(10 ** 10)
        assert result == 12

    def test_returns_int(self):
        assert isinstance(get_module_count(50), int)


class TestComputeLayout:
    def test_zero_pages_raises(self):
        with pytest.raises(ValueError):
            compute_layout(0)

    def test_negative_pages_raises(self):
        with pytest.raises(ValueError):
            compute_layout(-1)

    def test_returns_three_tuple(self):
        result = compute_layout(50)
        assert len(result) == 3

    def test_all_values_positive(self):
        n_modules, n_submodules, n_sections = compute_layout(50)
        assert n_modules >= 2
        assert n_submodules >= 2
        assert n_sections >= 2

    def test_modules_within_bounds(self):
        n_modules, _, _ = compute_layout(50)
        assert 2 <= n_modules <= 12

    def test_submodules_within_bounds(self):
        _, n_submodules, _ = compute_layout(50)
        assert 2 <= n_submodules <= 8

    def test_larger_course_more_modules(self):
        small_m, _, _ = compute_layout(10)
        large_m, _, _ = compute_layout(1000)
        assert large_m >= small_m

    def test_single_page(self):
        n_modules, n_submodules, n_sections = compute_layout(1)
        assert n_modules >= 2
        assert n_submodules >= 2
        assert n_sections >= 2
