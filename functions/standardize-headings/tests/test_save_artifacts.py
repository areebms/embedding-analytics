"""Tests for the two artifacts collect writes.

`html-standardized/` is what gets chunked and indexed; `text/` is what tokenize reads.
Both are derived from the same standardized blocks, and neither has a schema anywhere
else to catch a change in shape.
"""

from conftest import INDEX, s3_body, s3_content_type
from shared.tables.pipeline_entries import standardized_html_key, text_key

from llm_parse_response.save_artifacts import (
    HTML_CONTENT_TYPE,
    TEXT_CONTENT_TYPE,
    render_html,
    render_text,
    save_html,
    save_text,
)


PAIRS = [
    ("h1", "The Wealth of Nations"),
    ("p", "An inquiry into the nature and causes."),
    ("h2", "BOOK I."),
]


# ── render_html ───────────────────────────────────────────────────────


def test_the_page_is_one_element_per_block_in_order():
    page = render_html(PAIRS, "gutenberg-3300")

    assert "<h1>The Wealth of Nations</h1>" in page
    assert "<p>An inquiry into the nature and causes.</p>" in page
    assert page.index("<h1>") < page.index("<p>") < page.index("<h2>")


def test_the_page_is_a_standalone_utf8_document_titled_by_its_index():
    page = render_html(PAIRS, "gutenberg-3300")

    assert page.startswith("<!DOCTYPE html>\n<html>\n<head>\n")
    assert '<meta charset="utf-8">' in page
    assert "<title>gutenberg-3300</title>" in page
    assert page.endswith("</body>\n</html>\n")


def test_markup_in_the_book_text_is_escaped_not_emitted():
    """Text comes out of somebody else's HTML; re-emitting it raw would let a stray
    tag swallow the rest of the page."""
    page = render_html([("p", "a <b>bold</b> & italic")], "gutenberg-3300")

    assert "<p>a &lt;b&gt;bold&lt;/b&gt; &amp; italic</p>" in page
    assert "<b>" not in page


def test_quotes_are_left_alone():
    """There are no attributes in this document, so escaping quotes would only make
    ordinary prose harder to read."""
    page = render_html([("p", 'he said "yes"')], "gutenberg-3300")

    assert '<p>he said "yes"</p>' in page


def test_a_book_with_no_blocks_still_renders_a_valid_page():
    page = render_html([], "gutenberg-3300")

    assert "<body>\n\n</body>" in page


# ── render_text ───────────────────────────────────────────────────────


def test_blocks_are_separated_by_a_blank_line():
    """Load-bearing: tokenize segments sentences within each block, so a heading that
    ends without a period stays off the front of the paragraph after it."""
    assert render_text(PAIRS) == (
        "The Wealth of Nations\n\n"
        "An inquiry into the nature and causes.\n\n"
        "BOOK I.\n"
    )


def test_the_tags_themselves_do_not_appear_in_the_text():
    assert "h1" not in render_text(PAIRS)


def test_the_file_ends_with_a_newline():
    assert render_text([("p", "one")]) == "one\n"


def test_an_empty_book_renders_to_a_bare_newline():
    assert render_text([]) == "\n"


# ── uploads ───────────────────────────────────────────────────────────


def test_the_html_lands_at_the_key_the_pipeline_table_derives(bucket):
    """The row records that the artifact exists, not where it is, so this key has to
    stay the one shared/ builds."""
    save_html(INDEX, PAIRS)

    assert standardized_html_key(INDEX) == "html-standardized/gutenberg-3300.html"
    assert s3_body(bucket, standardized_html_key(INDEX)) == render_html(PAIRS, INDEX)
    assert s3_content_type(bucket, standardized_html_key(INDEX)) == HTML_CONTENT_TYPE


def test_the_text_lands_at_the_key_tokenize_reads(bucket):
    save_text(INDEX, PAIRS)

    assert text_key(INDEX) == "text/gutenberg-3300.txt"
    assert s3_body(bucket, text_key(INDEX)) == render_text(PAIRS)
    assert s3_content_type(bucket, text_key(INDEX)) == TEXT_CONTENT_TYPE


def test_non_ascii_book_text_survives_the_round_trip(bucket):
    """Half the corpus is not English."""
    pairs = [("h1", "CAPÍTULO I"), ("p", "árboles y niños")]

    save_text(INDEX, pairs)

    assert s3_body(bucket, text_key(INDEX)) == "CAPÍTULO I\n\nárboles y niños\n"
