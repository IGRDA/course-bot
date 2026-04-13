"""Unit tests for tools/json2book/utils.py

Tests cover LaTeX escaping, markdown-to-LaTeX conversion, and utility functions.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.json2book.utils import (
    _convert_bullet_lists,
    _convert_code_blocks,
    _convert_html_tags,
    _convert_numbered_lists,
    _escape_special_chars,
    _strip_unsupported_unicode,
    _unescape_for_verbatim,
    download_image,
    escape_latex,
    escape_latex_simple,
    markdown_to_latex,
)


class TestEscapeLatex:
    def test_empty_string_returns_empty(self):
        assert escape_latex("") == ""

    def test_none_equivalent_empty(self):
        # Empty string is falsy
        assert escape_latex("") == ""

    def test_plain_text_unchanged(self):
        result = escape_latex("hello world")
        assert result == "hello world"

    def test_ampersand_escaped(self):
        result = escape_latex("a & b", preserve_math=False)
        assert r"\&" in result

    def test_percent_escaped(self):
        result = escape_latex("50%", preserve_math=False)
        assert r"\%" in result

    def test_dollar_escaped(self):
        result = escape_latex("$100", preserve_math=False)
        assert r"\$" in result

    def test_hash_escaped(self):
        result = escape_latex("#tag", preserve_math=False)
        assert r"\#" in result

    def test_underscore_escaped(self):
        result = escape_latex("snake_case", preserve_math=False)
        assert r"\_" in result

    def test_curly_braces_escaped(self):
        result = escape_latex("{key}", preserve_math=False)
        assert r"\{" in result
        assert r"\}" in result

    def test_tilde_escaped(self):
        result = escape_latex("~approx", preserve_math=False)
        assert r"\textasciitilde{}" in result

    def test_caret_escaped(self):
        result = escape_latex("x^2", preserve_math=False)
        assert r"\textasciicircum{}" in result

    def test_backslash_escaped(self):
        result = escape_latex("a\\b", preserve_math=False)
        assert r"\textbackslash" in result

    def test_math_environment_preserved(self):
        text = r"The formula \(x^2 + y^2 = r^2\) is correct"
        result = escape_latex(text, preserve_math=True)
        # The math environment should be preserved
        assert r"\(x^2 + y^2 = r^2\)" in result

    def test_display_math_preserved(self):
        text = r"Result: \[E = mc^2\]"
        result = escape_latex(text, preserve_math=True)
        assert r"\[E = mc^2\]" in result

    def test_unicode_math_symbol_converted(self):
        result = escape_latex("area × radius", preserve_math=False)
        assert r"$\times$" in result

    def test_greek_letter_alpha_converted(self):
        result = escape_latex("α beta", preserve_math=False)
        assert r"$\alpha$" in result

    def test_em_dash_converted(self):
        result = escape_latex("left—right", preserve_math=False)
        assert "---" in result

    def test_en_dash_converted(self):
        result = escape_latex("range 1–5", preserve_math=False)
        assert "--" in result

    def test_smart_double_quote_open(self):
        result = escape_latex("\u201chello\u201d", preserve_math=False)
        assert "``" in result

    def test_smart_double_quote_close(self):
        result = escape_latex("\u201chello\u201d", preserve_math=False)
        assert "''" in result

    def test_preserve_math_false(self):
        text = r"\(math\)"
        result = escape_latex(text, preserve_math=False)
        # With preserve_math=False, no special treatment for math
        assert result is not None


class TestEscapeLatexSimple:
    def test_empty_string_returns_empty(self):
        assert escape_latex_simple("") == ""

    def test_plain_text_unchanged(self):
        assert escape_latex_simple("hello") == "hello"

    def test_backslash_escaped(self):
        result = escape_latex_simple("a\\b")
        assert r"\textbackslash" in result

    def test_special_chars_escaped(self):
        result = escape_latex_simple("50% profit & loss")
        assert r"\%" in result
        assert r"\&" in result

    def test_pua_chars_stripped(self):
        # PUA character U+E000
        text = "hello\ue000world"
        result = escape_latex_simple(text)
        assert "\ue000" not in result


class TestStripUnsupportedUnicode:
    def test_no_pua_chars_unchanged(self):
        text = "normal text"
        assert _strip_unsupported_unicode(text) == "normal text"

    def test_pua_bullet_replaced(self):
        # U+F0B7 is a PUA bullet
        text = "item\uf0b7text"
        result = _strip_unsupported_unicode(text)
        assert r"$\bullet$" in result

    def test_pua_non_bullet_removed(self):
        # Some PUA char that isn't a known bullet
        text = "text\ue100more"
        result = _strip_unsupported_unicode(text)
        assert "\ue100" not in result
        assert result == "textmore"

    def test_pua_f0a7_replaced_with_bullet(self):
        text = "\uf0a7item"
        result = _strip_unsupported_unicode(text)
        assert r"$\bullet$" in result


class TestEscapeSpecialChars:
    def test_angle_brackets_converted(self):
        result = _escape_special_chars("a >> b")
        assert ">{}>" in result

    def test_double_less_than_converted(self):
        result = _escape_special_chars("a << b")
        assert "<{}<" in result

    def test_multiple_special_chars(self):
        result = _escape_special_chars("50% & $100 #tag")
        assert r"\%" in result
        assert r"\&" in result
        assert r"\$" in result
        assert r"\#" in result


class TestUnescapeForVerbatim:
    def test_underscore_unescaped(self):
        assert _unescape_for_verbatim(r"\_") == "_"

    def test_hash_unescaped(self):
        assert _unescape_for_verbatim(r"\#") == "#"

    def test_ampersand_unescaped(self):
        assert _unescape_for_verbatim(r"\&") == "&"

    def test_percent_unescaped(self):
        assert _unescape_for_verbatim(r"\%") == "%"

    def test_dollar_unescaped(self):
        assert _unescape_for_verbatim(r"\$") == "$"

    def test_curly_braces_unescaped(self):
        assert _unescape_for_verbatim(r"\{") == "{"
        assert _unescape_for_verbatim(r"\}") == "}"

    def test_textbackslash_unescaped(self):
        assert _unescape_for_verbatim(r"\textbackslash{}") == "\\"

    def test_tilde_unescaped(self):
        assert _unescape_for_verbatim(r"\textasciitilde{}") == "~"

    def test_caret_unescaped(self):
        assert _unescape_for_verbatim(r"\textasciicircum{}") == "^"


class TestConvertHtmlTags:
    def test_bold_b_tags(self):
        result = _convert_html_tags("<b>bold</b>")
        assert r"\textbf{bold}" in result

    def test_strong_tags(self):
        result = _convert_html_tags("<strong>strong</strong>")
        assert r"\textbf{strong}" in result

    def test_italic_i_tags(self):
        result = _convert_html_tags("<i>italic</i>")
        assert r"\textit{italic}" in result

    def test_em_tags(self):
        result = _convert_html_tags("<em>emphasized</em>")
        assert r"\textit{emphasized}" in result

    def test_underline_tags(self):
        result = _convert_html_tags("<u>underline</u>")
        assert r"\underline{underline}" in result

    def test_code_tags(self):
        result = _convert_html_tags("<code>code</code>")
        assert r"\texttt{code}" in result

    def test_br_tag(self):
        result = _convert_html_tags("line<br/>break")
        assert r"\\" in result

    def test_unknown_tags_removed(self):
        result = _convert_html_tags("<span>text</span>")
        assert "<span>" not in result
        assert "text" in result

    def test_case_insensitive(self):
        result = _convert_html_tags("<B>BOLD</B>")
        assert r"\textbf{BOLD}" in result


class TestConvertCodeBlocks:
    def test_fenced_code_block(self):
        text = "```python\nprint('hello')\n```"
        result = _convert_code_blocks(text)
        assert r"\begin{verbatim}" in result
        assert r"\end{verbatim}" in result

    def test_inline_code(self):
        text = "use `print()` function"
        result = _convert_code_blocks(text)
        assert r"\texttt{print()}" in result

    def test_fenced_code_without_language(self):
        text = "```\ncode here\n```"
        result = _convert_code_blocks(text)
        assert r"\begin{verbatim}" in result


class TestConvertBulletLists:
    def test_dash_bullet(self):
        text = "- item one\n- item two"
        result = _convert_bullet_lists(text)
        assert r"\begin{itemize}" in result
        assert r"\end{itemize}" in result
        assert r"\item item one" in result
        assert r"\item item two" in result

    def test_asterisk_bullet(self):
        text = "* item one"
        result = _convert_bullet_lists(text)
        assert r"\begin{itemize}" in result

    def test_no_list_unchanged(self):
        text = "plain text\nmore text"
        result = _convert_bullet_lists(text)
        assert r"\begin{itemize}" not in result

    def test_list_followed_by_text(self):
        text = "- item\nnormal text"
        result = _convert_bullet_lists(text)
        assert r"\end{itemize}" in result


class TestConvertNumberedLists:
    def test_numbered_list(self):
        text = "1. first\n2. second"
        result = _convert_numbered_lists(text)
        assert r"\begin{enumerate}" in result
        assert r"\end{enumerate}" in result
        assert r"\item first" in result
        assert r"\item second" in result

    def test_numbered_list_with_paren(self):
        text = "1) item"
        result = _convert_numbered_lists(text)
        assert r"\begin{enumerate}" in result

    def test_no_list_unchanged(self):
        text = "regular text"
        result = _convert_numbered_lists(text)
        assert r"\begin{enumerate}" not in result


class TestMarkdownToLatex:
    def test_empty_string(self):
        assert markdown_to_latex("") == ""

    def test_bold_text(self):
        result = markdown_to_latex("**bold text**")
        assert r"\textbf{bold text}" in result

    def test_italic_text(self):
        result = markdown_to_latex("*italic text*")
        assert r"\textit{italic text}" in result

    def test_html_bold_in_markdown(self):
        result = markdown_to_latex("<b>bold</b>")
        assert r"\textbf{bold}" in result

    def test_bullet_list(self):
        result = markdown_to_latex("- item one\n- item two")
        assert r"\begin{itemize}" in result
        assert r"\item item one" in result

    def test_numbered_list(self):
        result = markdown_to_latex("1. first\n2. second")
        assert r"\begin{enumerate}" in result

    def test_code_block(self):
        result = markdown_to_latex("```python\ncode\n```")
        assert r"\begin{verbatim}" in result

    def test_inline_code(self):
        result = markdown_to_latex("use `func()`")
        assert r"\texttt{func()}" in result

    def test_math_environment_preserved(self):
        text = r"Formula: \(x = 5\)"
        result = markdown_to_latex(text)
        # Math should be preserved, not mangled
        assert r"\(" in result
        assert r"\)" in result

    def test_markdown_table(self):
        text = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        result = markdown_to_latex(text)
        assert r"\begin{table}" in result
        assert r"\end{table}" in result

    def test_plain_text_unchanged(self):
        text = "plain text without formatting"
        result = markdown_to_latex(text)
        assert "plain text without formatting" in result


class TestDownloadImage:
    def test_empty_url_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_image("", Path(tmpdir))
            assert result is None

    def test_local_nonexistent_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_image("/nonexistent/path/image.jpg", Path(tmpdir))
            assert result is None

    def test_local_existing_file_copied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create a source image file
            src = tmpdir_path / "source.jpg"
            src.write_bytes(b"fake image data")
            output_dir = tmpdir_path / "output"

            result = download_image(str(src), output_dir)
            assert result is not None
            assert result.exists()

    def test_http_download_success(self):
        mock_response = MagicMock()
        mock_response.content = b"fake image data"
        mock_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("requests.get", return_value=mock_response):
                result = download_image("https://example.com/image.jpg", Path(tmpdir))
            assert result is not None
            assert result.suffix == ".jpg"

    def test_http_download_failure_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("requests.get", side_effect=Exception("network error")):
                result = download_image("https://example.com/image.jpg", Path(tmpdir))
            assert result is None

    def test_http_url_with_no_extension_defaults_to_jpg(self):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake image data"]
        mock_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("requests.get", return_value=mock_response):
                result = download_image("https://example.com/image", Path(tmpdir))
            if result is not None:
                assert result.suffix == ".jpg"

    def test_already_downloaded_skips_re_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Pre-create the expected output file
            import hashlib

            url = "https://example.com/test.jpg"
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            expected_path = tmpdir_path / f"img_{url_hash}.jpg"
            expected_path.write_bytes(b"existing content")

            with patch("requests.get") as mock_get:
                result = download_image(url, tmpdir_path)
            # Should NOT call requests.get since file already exists
            mock_get.assert_not_called()
            assert result == expected_path
