"""Unit tests for workflows/audience_profiles.py"""

from workflows.audience_profiles import AUDIENCE_PROFILES, build_audience_guidelines


class TestBuildAudienceGuidelines:
    def test_none_audience_returns_empty(self):
        assert build_audience_guidelines(None) == ""

    def test_invalid_audience_returns_empty(self):
        assert build_audience_guidelines("expert") == ""  # type: ignore

    def test_kids_returns_non_empty(self):
        result = build_audience_guidelines("kids")
        assert len(result) > 0

    def test_general_returns_non_empty(self):
        result = build_audience_guidelines("general")
        assert len(result) > 0

    def test_advanced_returns_non_empty(self):
        result = build_audience_guidelines("advanced")
        assert len(result) > 0

    def test_kids_contains_audience_name(self):
        result = build_audience_guidelines("kids")
        assert "KIDS" in result

    def test_advanced_contains_audience_name(self):
        result = build_audience_guidelines("advanced")
        assert "ADVANCED" in result

    def test_activities_context_adds_rules(self):
        general_result = build_audience_guidelines("general", context="general")
        activities_result = build_audience_guidelines("general", context="activities")
        # Activities context adds extra rules block
        assert len(activities_result) > len(general_result)
        assert "ACTIVITY RULES" in activities_result

    def test_non_activities_context_no_activity_rules(self):
        result = build_audience_guidelines("kids", context="theory")
        assert "ACTIVITY RULES" not in result

    def test_guidelines_contain_rules(self):
        result = build_audience_guidelines("general")
        # All rules from profile are included as bullet points
        profile = AUDIENCE_PROFILES["general"]
        for rule in profile["rules"]:
            assert rule in result

    def test_all_three_audiences_produce_different_output(self):
        kids = build_audience_guidelines("kids")
        general = build_audience_guidelines("general")
        advanced = build_audience_guidelines("advanced")
        assert kids != general
        assert general != advanced
        assert kids != advanced


class TestAudienceProfiles:
    def test_all_expected_keys_present(self):
        assert "kids" in AUDIENCE_PROFILES
        assert "general" in AUDIENCE_PROFILES
        assert "advanced" in AUDIENCE_PROFILES

    def test_each_profile_has_required_fields(self):
        for audience, profile in AUDIENCE_PROFILES.items():
            assert "name" in profile, f"{audience} missing 'name'"
            assert "summary" in profile, f"{audience} missing 'summary'"
            assert "rules" in profile, f"{audience} missing 'rules'"
            assert isinstance(profile["rules"], list)
            assert len(profile["rules"]) > 0
