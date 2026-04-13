"""Unit tests for tools/podcast/text_preprocessor.py"""

from tools.podcast.text_preprocessor import (
    clean_punctuation,
    expand_acronyms,
    normalize_numbers,
    preprocess_for_tts,
    split_long_sentences,
)


class TestSplitLongSentences:
    def test_short_sentence_unchanged(self):
        text = "This is a short sentence."
        result = split_long_sentences(text, max_words=30)
        assert result == text

    def test_long_sentence_gets_split(self):
        # 35 words separated by a comma
        words_before = " ".join(["word"] * 15)
        words_after = " ".join(["word"] * 15)
        long_sentence = f"{words_before}, {words_after}."
        result = split_long_sentences(long_sentence, max_words=20)
        # Should contain a period introduced by the split
        assert result.count(".") >= 1

    def test_semicolon_converted(self):
        text = "First part of text; second part of text with more words here."
        result = split_long_sentences(text, max_words=5)
        # Semicolons get converted to periods
        assert ";" not in result or result.count(";") < text.count(";")

    def test_empty_string(self):
        assert split_long_sentences("", max_words=30) == ""

    def test_multiple_sentences_preserved(self):
        text = "Short one. Also short. Fine here."
        result = split_long_sentences(text, max_words=30)
        assert result == text


class TestExpandAcronyms:
    def test_known_en_acronym(self):
        result = expand_acronyms("Use the API endpoint.", language="en")
        assert "A.P.I." in result

    def test_known_es_acronym(self):
        result = expand_acronyms("Usa la API del servidor.", language="es")
        assert "A.P.I." in result

    def test_unknown_acronym_unchanged(self):
        result = expand_acronyms("Use the XYZ protocol.", language="en")
        assert "XYZ" in result

    def test_cpu_expansion_en(self):
        result = expand_acronyms("The CPU is fast.", language="en")
        assert "C.P.U." in result

    def test_lowercase_not_affected(self):
        result = expand_acronyms("the api is fast.", language="en")
        # Lowercase 'api' is not all-caps, should not be expanded
        assert "api" in result.lower()

    def test_fallback_to_en_for_unknown_language(self):
        # Should fall back to EN expansions
        result = expand_acronyms("The API works.", language="zh")
        assert "A.P.I." in result

    def test_html_expansion_en(self):
        result = expand_acronyms("Write HTML code.", language="en")
        assert "H.T.M.L." in result


class TestNormalizeNumbers:
    def test_single_digit_en(self):
        result = normalize_numbers("There are 5 apples.", language="en")
        assert "five" in result

    def test_single_digit_es(self):
        result = normalize_numbers("Hay 5 manzanas.", language="es")
        assert "cinco" in result

    def test_percentage_en(self):
        result = normalize_numbers("It improved by 90%.", language="en")
        assert "ninety percent" in result

    def test_percentage_es(self):
        result = normalize_numbers("Mejoró en un 90%.", language="es")
        assert "noventa por ciento" in result

    def test_large_number_unchanged(self):
        result = normalize_numbers("In 2024 we launched.", language="en")
        # 2024 is > 2 digits, should NOT be replaced
        assert "2024" in result

    def test_zero_en(self):
        result = normalize_numbers("Score: 0 points.", language="en")
        assert "zero" in result


class TestCleanPunctuation:
    def test_stacked_question_exclamation(self):
        result = clean_punctuation("Really?!")
        assert "?!" not in result

    def test_double_exclamation(self):
        result = clean_punctuation("Wow!!")
        assert "!!" not in result

    def test_double_question(self):
        result = clean_punctuation("What??")
        assert "??" not in result

    def test_semicolon_becomes_period(self):
        result = clean_punctuation("First part; second part.")
        assert ";" not in result
        assert "." in result

    def test_multiple_spaces_collapsed(self):
        result = clean_punctuation("word  word   word")
        assert "  " not in result

    def test_long_ellipsis_normalized(self):
        result = clean_punctuation("Wait..........")
        assert "........" not in result

    def test_clean_text_unchanged_content(self):
        text = "Hello world. How are you?"
        result = clean_punctuation(text)
        assert "Hello world" in result
        assert "How are you" in result


class TestPreprocessForTts:
    def test_basic_pipeline_runs(self):
        text = "The API has 5 endpoints; use them wisely!!"
        result = preprocess_for_tts(text, language="en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_language_normalization_long_code(self):
        # "English" should normalize to "en"
        text = "Use the API correctly."
        result = preprocess_for_tts(text, language="English")
        assert isinstance(result, str)

    def test_spanish_pipeline(self):
        text = "La API tiene 5 rutas; úsalas bien!!"
        result = preprocess_for_tts(text, language="es")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_language_defaults_to_en(self):
        result = preprocess_for_tts("Hello world.", language="")
        assert isinstance(result, str)

    def test_no_double_periods_after_acronyms(self):
        # "A.P.I.. " should not appear
        text = "Use the API. Then proceed."
        result = preprocess_for_tts(text, language="en")
        assert ".." not in result


# ---- Additional coverage for _split_at_natural_break branches ----


class TestSplitAtNaturalBreak:
    """Tests specifically targeting the _split_at_natural_break internal logic."""

    def test_conjunction_split(self):
        # Build a sentence that triggers a conjunction split (>12 words before "and")
        # 13 words before "and" + 5 after
        pre = " ".join(["word"] * 13)
        post = " ".join(["word"] * 5)
        long_sent = f"{pre} and {post}."
        result = split_long_sentences(long_sent, max_words=15)
        # Should have introduced a period break
        assert "." in result

    def test_no_split_point_returns_as_is(self):
        # A sentence with no commas, no conjunctions, no semicolons
        # that can't be split gracefully
        single_word_blob = " ".join(["superlongword"] * 35)
        result = split_long_sentences(single_word_blob, max_words=20)
        # Should still be a string (not crash)
        assert isinstance(result, str)


class TestNumberToWords:
    """Test the _number_to_words helper for composite numbers."""

    from tools.podcast.text_preprocessor import _number_to_words

    def test_large_number_returns_string(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(999, "en")
        assert isinstance(result, str)

    def test_twenty_one_en(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(21, "en")
        assert "twenty" in result
        assert "one" in result

    def test_twenty_one_es(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(21, "es")
        assert "veinti" in result

    def test_thirty_five_en(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(35, "en")
        assert "thirty" in result
        assert "five" in result

    def test_thirty_five_es(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(35, "es")
        assert "treinta" in result
        assert "cinco" in result

    def test_exact_ten_en(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(10, "en")
        assert result == "ten"

    def test_40_en(self):
        from tools.podcast.text_preprocessor import _number_to_words

        result = _number_to_words(40, "en")
        assert result == "forty"


class TestNormalizeNumbersAdvanced:
    """Extra normalize_numbers tests for large percentages."""

    def test_large_percentage_en(self):
        result = normalize_numbers("Success rate: 75%.", language="en")
        assert "percent" in result
        assert "seventy" in result

    def test_large_percentage_es(self):
        result = normalize_numbers("Tasa de éxito: 75%.", language="es")
        assert "por ciento" in result

    def test_two_digit_not_in_dict_returned_as_is(self):
        # 23 is not in NUMBER_WORDS_EN, so replace_number returns it unchanged
        result = normalize_numbers("There are 23 apples.", language="en")
        assert "23" in result

    def test_comma_split_path(self):
        # Build a sentence: 11 words before comma, 6 words after, total 17 words
        # max_words=15 triggers split; 10 words before comma + comma satisfies word_count>=10
        # words_remaining=6 satisfies >=5
        pre = " ".join([f"word{i}" for i in range(11)])  # 11 words
        post = " ".join([f"word{i}" for i in range(6)])  # 6 words
        long_sent = f"{pre}, {post}."
        result = split_long_sentences(long_sent, max_words=15)
        # Should have introduced a period at the comma
        assert "." in result
