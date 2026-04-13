"""Unit tests for tools/peoplesearch/models.py"""

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

# Load the models file directly via its path to avoid tools/peoplesearch/__init__.py
# which chains imports that require optional langchain packages not installed here.
_MODELS_PATH = Path(__file__).resolve().parent.parent / "tools" / "peoplesearch" / "models.py"
_spec = importlib.util.spec_from_file_location("peoplesearch_models", _MODELS_PATH)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)

PersonResult = _models.PersonResult
PersonSuggestion = _models.PersonSuggestion
PeopleSuggestionResponse = _models.PeopleSuggestionResponse


class TestPersonResult:
    def test_valid_creation(self):
        person = PersonResult(
            name="Alan Turing",
            description="Father of computer science",
            wikiUrl="https://en.wikipedia.org/wiki/Alan_Turing",
            image="https://upload.wikimedia.org/turing.jpg",
        )
        assert person.name == "Alan Turing"
        assert person.wikiUrl == "https://en.wikipedia.org/wiki/Alan_Turing"

    def test_empty_image_allowed(self):
        person = PersonResult(
            name="Ada Lovelace",
            description="First programmer",
            wikiUrl="https://en.wikipedia.org/wiki/Ada_Lovelace",
            image="",
        )
        assert person.image == ""

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            PersonResult(
                name="Alan Turing",
                description="Pioneer",
                # wikiUrl missing
                image="",
            )


class TestPersonSuggestion:
    def test_valid_creation(self):
        suggestion = PersonSuggestion(
            name="Grace Hopper",
            reason="Pioneered compiler development",
        )
        assert suggestion.name == "Grace Hopper"
        assert "compiler" in suggestion.reason

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            PersonSuggestion(reason="Some reason")

    def test_missing_reason_raises(self):
        with pytest.raises(ValidationError):
            PersonSuggestion(name="Someone")


class TestPeopleSuggestionResponse:
    def test_valid_creation(self):
        response = PeopleSuggestionResponse(
            people=[
                PersonSuggestion(name="Person A", reason="Reason A"),
                PersonSuggestion(name="Person B", reason="Reason B"),
            ]
        )
        assert len(response.people) == 2

    def test_empty_people_list(self):
        response = PeopleSuggestionResponse(people=[])
        assert response.people == []

    def test_missing_people_raises(self):
        with pytest.raises(ValidationError):
            PeopleSuggestionResponse()
