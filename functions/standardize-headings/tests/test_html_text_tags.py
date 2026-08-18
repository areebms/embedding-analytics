"""Tests for the HTML walker that turns a scraped book into (tag, text) blocks.

Everything here except load_tag_text_pairs is pure, so these are plain strings in
and lists out — no fixtures, no mocking.
"""

import pytest

from conftest import BOOK_HTML, INDEX


def _pairs(html):
    from book_records.html_text_tags import flatten_html_elements, prepare_book_body

    return list(flatten_html_elements(prepare_book_body(html)))


# ── the block kinds ───────────────────────────────────────────────────


def test_headings_keep_their_tag_and_prose_becomes_p():
    assert _pairs("<body><h2>Chapter I</h2><p>Once upon a time.</p></body>") == [
        ("h2", "Chapter I"),
        ("p", "Once upon a time."),
    ]


def test_whitespace_inside_a_block_is_collapsed():
    assert _pairs("<body><p>one\n  two\t\tthree</p></body>") == [
        ("p", "one two three")
    ]


def test_a_heading_that_cleans_away_to_nothing_is_dropped():
    """It is not demoted to prose — it vanishes, and the positions Claude is asked
    about are numbered over what survives."""
    assert _pairs("<body><h2>   </h2><p>Real text.</p></body>") == [("p", "Real text.")]


def test_an_empty_paragraph_is_dropped():
    assert _pairs("<body><p></p><p>Real text.</p></body>") == [("p", "Real text.")]


@pytest.mark.parametrize("tag", ["ul", "ol"], ids=["unordered", "ordered"])
def test_list_items_each_become_their_own_block(tag):
    html = f"<body><{tag}><li>First</li><li>Second</li></{tag}></body>"

    assert _pairs(html) == [("p", "First"), ("p", "Second")]


def test_a_nested_list_folds_into_its_parent_item():
    """Only direct <li> children are enumerated, but each one's text is taken
    recursively, so a nested list arrives inside its parent's block."""
    html = "<body><ul><li>Outer<ul><li>Inner</li></ul></li></ul></body>"

    assert _pairs(html) == [("p", "Outer Inner")]


def test_definition_lists_flatten_terms_and_definitions_in_document_order():
    html = "<body><dl><dt>Term</dt><dd>Meaning</dd><dt>Other</dt></dl></body>"

    assert _pairs(html) == [("p", "Term"), ("p", "Meaning"), ("p", "Other")]


def test_table_rows_collapse_to_one_block_per_row():
    html = (
        "<body><table>"
        "<tr><th>Year</th><th>Price</th></tr>"
        "<tr><td>1776</td><td>Two shillings</td></tr>"
        "</table></body>"
    )

    assert _pairs(html) == [("p", "Year Price"), ("p", "1776 Two shillings")]


def test_a_row_of_empty_cells_produces_no_block():
    html = (
        "<body><table><tr><td></td><td>  </td></tr>"
        "<tr><td>Real</td></tr></table></body>"
    )

    assert _pairs(html) == [("p", "Real")]


def test_a_nested_table_has_its_text_counted_more_than_once():
    """Pins current behaviour, which is almost certainly not what anyone wanted.
    Unlike the list and definition-list helpers, table_row_texts searches
    recursively, so the inner table's cell is picked up by the outer row *and*
    again as a row of its own. Layout tables are common in OCR'd book HTML, so
    this inflates the word counts Claude is given between headings."""
    html = (
        "<body><table><tr><td>Outer"
        "<table><tr><td>Inner</td></tr></table>"
        "</td></tr></table></body>"
    )

    assert _pairs(html) == [("p", "Outer Inner Inner"), ("p", "Inner")]


def test_blockquotes_split_on_blank_lines():
    html = "<body><blockquote>First stanza.\n\nSecond stanza.</blockquote></body>"

    assert _pairs(html) == [("p", "First stanza."), ("p", "Second stanza.")]


def test_preformatted_text_splits_on_blank_lines():
    html = "<body><pre>Line one.\n\nLine two.</pre></body>"

    assert _pairs(html) == [("p", "Line one."), ("p", "Line two.")]


# ── walking wrappers ──────────────────────────────────────────────────


def test_a_wrapper_holding_real_structure_is_walked_through():
    html = "<body><div><div><h2>Deep heading</h2><p>Deep prose.</p></div></div></body>"

    assert _pairs(html) == [("h2", "Deep heading"), ("p", "Deep prose.")]


def test_a_wrapper_holding_no_structure_is_taken_whole():
    html = "<body><span>Just <em>some</em> words</span></body>"

    assert _pairs(html) == [("p", "Just some words")]


# ── what never counts as book text ────────────────────────────────────


def test_project_gutenberg_boilerplate_is_dropped():
    pairs = _pairs(BOOK_HTML)

    assert ("h1", "The Wealth of Nations") in pairs
    assert not any("Project Gutenberg" in text for _, text in pairs)


def test_skipped_tags_contribute_nothing():
    html = "<body><script>var x = 1;</script><style>p {}</style><p>Real.</p></body>"

    assert _pairs(html) == [("p", "Real.")]


def test_comments_and_doctypes_are_not_prose():
    html = "<!DOCTYPE html><body><!-- Transcriber's note --><p>Real.</p></body>"

    assert _pairs(html) == [("p", "Real.")]


def test_a_fragment_with_no_body_is_walked_as_it_stands():
    """html.parser does not invent a <body>, so prepare_book_body falls back to the
    whole soup rather than returning None."""
    assert _pairs("<h1>Title</h1><p>Prose.</p>") == [
        ("h1", "Title"),
        ("p", "Prose."),
    ]


# ── the one impure function ───────────────────────────────────────────


def test_load_tag_text_pairs_reads_the_books_html_object(bucket):
    from shared.tables.pipeline_entries import html_key

    from book_records.html_text_tags import load_tag_text_pairs

    bucket.put_object(Key=html_key(INDEX), Body=BOOK_HTML.encode("utf-8"))

    assert load_tag_text_pairs(INDEX)[0] == ("h1", "The Wealth of Nations")
