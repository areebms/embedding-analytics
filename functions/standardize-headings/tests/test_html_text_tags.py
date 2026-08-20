"""Tests for the HTML → (tag, text) reduction that feeds every later stage.

This is the only place in the function that reads a book's raw markup. Whatever it
drops here is gone from the classification prompt, from `html-standardized/` and from
`text/`, so the fixtures below are the shapes Project Gutenberg actually ships:
license wrappers, tables of contents as lists, transcriber's notes as comments.
"""

import pytest
from bs4 import BeautifulSoup

from conftest import BOOK_HTML, BOOK_PAIRS, INDEX, PROSE_ONLY_HTML
from shared.tables.pipeline_entries import html_key

from book_records.html_text_tags import (
    as_paragraph_elements,
    blank_line_separated_texts,
    clean_element_text,
    clean_text,
    definition_list_texts,
    flatten_html_elements,
    list_item_texts,
    load_tag_text_pairs,
    prepare_book_body,
    strip_non_book_elements,
    strip_pg_boilerplate,
    table_row_texts,
)


def flatten(html):
    return list(flatten_html_elements(prepare_book_body(html)))


def soup_of(html):
    return BeautifulSoup(html, "html.parser")


# ── text cleaning ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  spaced  out  ", "spaced out"),
        ("line\nbreak", "line break"),
        ("tabs\tand\r\nnewlines", "tabs and newlines"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_clean_text_collapses_every_run_of_whitespace(raw, expected):
    assert clean_text(raw) == expected


def test_clean_element_text_joins_inline_children_with_a_space():
    """get_text() with no separator would run "<i>a</i><b>b</b>" together as "ab"."""
    element = soup_of("<p><i>first</i><b>second</b></p>").p

    assert clean_element_text(element) == "first second"


def test_as_paragraph_elements_labels_prose_and_drops_what_cleaned_away():
    assert list(as_paragraph_elements(["one", "", "two"])) == [("p", "one"), ("p", "two")]


# ── the per-element text extractors ───────────────────────────────────


def test_list_items_are_one_block_each_and_nested_lists_stay_with_their_parent():
    element = soup_of("<ul><li>one<ul><li>inner</li></ul></li><li>two</li></ul>").ul

    assert list(list_item_texts(element)) == ["one inner", "two"]


def test_definition_lists_yield_terms_and_definitions_in_document_order():
    element = soup_of("<dl><dt>Labour</dt><dd>The real price.</dd><dt>Rent</dt></dl>").dl

    assert list(definition_list_texts(element)) == ["Labour", "The real price.", "Rent"]


def test_table_rows_are_one_block_each_with_cells_joined_by_spaces():
    """Column layout carries no meaning once the book is prose."""
    element = soup_of(
        "<table><tr><th>Year</th><th>Price</th></tr>"
        "<tr><td>1776</td><td>2s.</td></tr></table>"
    ).table

    assert list(table_row_texts(element)) == ["Year Price", "1776 2s."]


def test_empty_table_cells_do_not_leave_double_spaces():
    element = soup_of("<table><tr><td>1776</td><td></td><td>2s.</td></tr></table>").table

    assert list(table_row_texts(element)) == ["1776 2s."]


def test_blockquotes_split_on_blank_lines_because_that_is_all_they_give_us():
    element = soup_of(
        "<blockquote>first para\n\n  \nsecond para</blockquote>"
    ).blockquote

    assert list(blank_line_separated_texts(element)) == ["first para", "second para"]


# ── flatten_html_elements: one test per branch ────────────────────────


def test_headings_keep_their_original_tag():
    assert flatten("<body><h1>Title</h1><h3>Section</h3></body>") == [
        ("h1", "Title"),
        ("h3", "Section"),
    ]


def test_a_heading_that_cleans_away_to_nothing_is_dropped():
    """An empty heading would otherwise take a position in the prompt and force the
    model to classify a blank line."""
    assert flatten("<body><h1>   </h1><p>prose</p></body>") == [("p", "prose")]


def test_loose_text_outside_any_element_becomes_a_paragraph():
    assert flatten("<body>loose text<p>para</p></body>") == [
        ("p", "loose text"),
        ("p", "para"),
    ]


def test_lists_dls_and_tables_all_reduce_to_paragraphs():
    html = (
        "<body><ul><li>bullet</li></ul><ol><li>numbered</li></ol>"
        "<dl><dt>term</dt></dl><table><tr><td>cell</td></tr></table></body>"
    )

    assert flatten(html) == [
        ("p", "bullet"),
        ("p", "numbered"),
        ("p", "term"),
        ("p", "cell"),
    ]


def test_an_unknown_container_holding_structure_is_descended_into():
    """A <div> wrapping a chapter must not collapse into one giant paragraph."""
    assert flatten("<body><div><h2>Chapter</h2><p>prose</p></div></body>") == [
        ("h2", "Chapter"),
        ("p", "prose"),
    ]


def test_an_unknown_container_holding_only_inline_markup_is_one_block():
    assert flatten("<body><div><span>a</span> <span>b</span></div></body>") == [
        ("p", "a b")
    ]


def test_blockquotes_and_preformatted_text_break_on_their_blank_lines():
    """Verse and long quotations arrive as one element with the paragraph breaks
    only in the whitespace."""
    html = "<body><blockquote>quoted\n\nsecond</blockquote><pre>code\n\nmore</pre></body>"

    assert flatten(html) == [
        ("p", "quoted"),
        ("p", "second"),
        ("p", "code"),
        ("p", "more"),
    ]


# ── stripping ─────────────────────────────────────────────────────────


def test_the_pg_licence_wrapper_is_removed():
    soup = strip_pg_boilerplate(
        soup_of('<div id="pg-header">licence</div><p>book</p><div id="pg-footer">tail</div>')
    )

    assert "licence" not in soup.get_text()
    assert "tail" not in soup.get_text()
    assert "book" in soup.get_text()


def test_stripping_boilerplate_is_fine_on_a_page_that_has_none():
    soup = strip_pg_boilerplate(soup_of("<p>book</p>"))

    assert soup.get_text() == "book"


def test_scripts_styles_and_images_are_not_book_text():
    soup = strip_non_book_elements(
        soup_of("<script>var x=1</script><style>p{}</style><img src='x'><p>keep</p>")
    )

    assert clean_text(soup.get_text(" ")) == "keep"


def test_comments_and_doctypes_are_not_paragraphs():
    """They subclass NavigableString, so without NON_TEXT_STRINGS the walk would
    yield "<!-- Transcriber's note -->" as prose."""
    html = "<!DOCTYPE html><body><!-- Transcriber's note --><p>keep</p></body>"

    assert flatten(html) == [("p", "keep")]


def test_prepare_book_body_returns_the_body_when_there_is_one():
    assert prepare_book_body("<html><body><p>x</p></body></html>").name == "body"


def test_prepare_book_body_falls_back_to_the_whole_document():
    """Not every scraped page has a <body>; a fragment still has to flatten."""
    assert flatten("<h1>Title</h1><p>x</p>") == [("h1", "Title"), ("p", "x")]


# ── the whole reduction ───────────────────────────────────────────────


def test_a_gutenberg_page_reduces_to_its_headings_and_prose():
    assert flatten(BOOK_HTML) == BOOK_PAIRS


def test_a_book_of_pure_prose_yields_no_headings():
    assert flatten(PROSE_ONLY_HTML) == [
        ("p", "An inquiry into the nature and causes."),
        ("p", "The greatest improvement in the productive powers of labour."),
    ]


def test_load_tag_text_pairs_reads_the_raw_html_artifact(bucket):
    bucket.put_object(Key=html_key(INDEX), Body=BOOK_HTML.encode("utf-8"))

    assert load_tag_text_pairs(INDEX) == BOOK_PAIRS


def test_load_tag_text_pairs_raises_when_the_book_was_never_scraped(bucket):
    """get_pending_book_tag_text_pairs relies on this raising rather than returning
    an empty list, so that it can skip the book instead of submitting an empty one."""
    with pytest.raises(Exception):
        load_tag_text_pairs(INDEX)
