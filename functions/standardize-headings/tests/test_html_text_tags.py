"""Reducing a scraped page to the (tag, text) blocks the rest of the stage works on.

This is the widest module in the service and the one whose mistakes are quietest: a
container walked the wrong way does not raise, it just yields a book with its prose
glued together or its headings missing.
"""

from bs4 import BeautifulSoup

from book_records.html_text_tags import (
    flatten_html_elements,
    load_tag_text_pairs,
    strip_non_book_elements,
)

from conftest import BOOK_HTML, BOOK_PAIRS, INDEX, SUBJECT


def pairs(html):
    """The same reduction load_tag_text_pairs performs, minus the S3 read."""
    soup = strip_non_book_elements(BeautifulSoup(html, "html.parser"))
    return list(flatten_html_elements(soup.body or soup))


def test_a_scraped_page_reduces_to_its_headings_and_prose():
    assert pairs(BOOK_HTML) == BOOK_PAIRS


def test_the_project_gutenberg_wrapper_is_never_part_of_the_book():
    texts = [text for _, text in pairs(BOOK_HTML)]

    assert not any("Project Gutenberg" in text for text in texts)


def test_tags_whose_text_is_never_content_are_dropped():
    html = """<body>
      <script>var pageCount = 3;</script>
      <style>h1 { font-size: 2em; }</style>
      <p>Real prose.</p>
    </body>"""

    assert pairs(html) == [("p", "Real prose.")]


def test_comments_and_doctypes_are_not_prose():
    """They all subclass NavigableString, so without the guard a transcriber's note
    would be walked in as a paragraph of the book."""
    html = """<body>
      <!-- Transcriber's note: corrected spelling. -->
      <p>Real prose.</p>
    </body>"""

    assert pairs(html) == [("p", "Real prose.")]


def test_loose_text_between_elements_is_prose():
    assert pairs("<body><h1>Title</h1>An unwrapped line.</body>") == [
        ("h1", "Title"),
        ("p", "An unwrapped line."),
    ]


def test_whitespace_only_blocks_are_dropped():
    assert pairs("<body><p>   </p><h2>  </h2><p>Kept.</p></body>") == [("p", "Kept.")]


def test_a_drop_cap_stays_part_of_its_word():
    """Gutenberg gives a section's opening letter its own span. Separating inline
    elements is what turned "Labour" into "L abour" at the head of every section."""
    html = '<body><p><span class="dropcap">L</span>abour, like all other things</p></body>'

    assert pairs(html) == [("p", "Labour, like all other things")]


def test_small_caps_inside_a_word_stay_part_of_it():
    html = "<body><p>J. M<span class='smcap'>c</span>Creery. Printer</p></body>"

    assert pairs(html) == [("p", "J. McCreery. Printer")]


def test_an_inline_element_does_not_push_its_punctuation_away():
    html = "<body><p><i>Labour</i>, like all other things</p></body>"

    assert pairs(html) == [("p", "Labour, like all other things")]


def test_a_page_number_between_two_words_leaves_no_trace():
    """The printed page number sits mid-sentence. Left in, it would fuse onto the next
    word now that inline elements no longer imply a space."""
    html = (
        '<body><p>the laws which regulate this <span class="pagenum">iv</span>'
        "distribution, is the principal problem</p></body>"
    )

    assert pairs(html) == [
        ("p", "the laws which regulate this distribution, is the principal problem")
    ]


def test_a_page_number_alone_in_a_block_leaves_no_block():
    html = '<body><p><span class="pagenum">iii</span></p><p>Kept.</p></body>'

    assert pairs(html) == [("p", "Kept.")]


def test_a_footnote_reference_leaves_no_trace():
    """The real nesting: the marker is a span inside a pginternal anchor. This is for
    the sake of a quotable `text/` -- tokenize's per-character strip would recover the
    word from a fused "Aristotle2" anyway."""
    html = (
        '<body><p>Aristotle<a href="#note_2" class="pginternal">'
        '<span class="tei tei-noteref">2</span></a> and Xenophon</p></body>'
    )

    assert pairs(html) == [("p", "Aristotle and Xenophon")]


def test_a_table_of_contents_link_is_not_a_reference_marker():
    """`pginternal` wraps both; only the marker span may be stripped."""
    html = '<body><p><a href="#chap01" class="pginternal">CHAPTER I.</a></p></body>'

    assert pairs(html) == [("p", "CHAPTER I.")]


def test_a_block_element_inside_a_block_still_separates():
    """The counterweight: inline elements imply no whitespace, block-level ones do.
    Dropping the separator for both would run these together as one word."""
    html = "<body><table><tr><td>Outer<div>Inner</div></td></tr></table></body>"

    assert pairs(html) == [("p", "Outer Inner")]


def test_an_empty_inline_element_adds_no_whitespace():
    html = "<body><p>Labour<span></span>, and capital</p></body>"

    assert pairs(html) == [("p", "Labour, and capital")]


def test_a_list_yields_one_block_per_item():
    html = "<body><ul><li>First item</li><li>Second item</li></ul></body>"

    assert pairs(html) == [("p", "First item"), ("p", "Second item")]


def test_a_nested_list_comes_along_inside_its_parent_item():
    html = """<body><ul>
      <li>Outer<ul><li>Inner</li></ul></li>
    </ul></body>"""

    assert pairs(html) == [("p", "Outer Inner")]


def test_a_definition_list_yields_one_block_per_term_and_definition():
    html = "<body><dl><dt>Rent</dt><dd>The price of land.</dd></dl></body>"

    assert pairs(html) == [("p", "Rent"), ("p", "The price of land.")]


def test_a_table_yields_one_block_per_row():
    """Column layout carries no meaning here, so cells are joined by spaces."""
    html = """<body><table>
      <tr><th>Year</th><th>Price</th></tr>
      <tr><td>1776</td><td>Two shillings</td></tr>
      <tr><td></td><td></td></tr>
    </table></body>"""

    assert pairs(html) == [("p", "Year Price"), ("p", "1776 Two shillings")]


def test_blank_lines_are_the_paragraph_breaks_inside_pre_and_blockquote():
    html = "<body><pre>First stanza.\n\nSecond stanza.</pre></body>"

    assert pairs(html) == [("p", "First stanza."), ("p", "Second stanza.")]


def test_an_unrecognised_container_holding_structure_is_walked_into():
    html = """<body><section>
      <h2>A chapter</h2><p>Its prose.</p>
    </section></body>"""

    assert pairs(html) == [("h2", "A chapter"), ("p", "Its prose.")]


def test_an_unrecognised_container_holding_no_structure_is_one_block():
    html = "<body><section><span>A</span> <span>run of text.</span></section></body>"

    assert pairs(html) == [("p", "A run of text.")]


def test_the_library_record_supplies_title_and_author(bucket):
    import json
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry, metadata_key
    from book_records.utils import load_book_record

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )
    bucket.put_object(
        Key=metadata_key(INDEX),
        Body=json.dumps(
            {"title": ["The Wealth of Nations"], "author": ["Smith, Adam"]}
        ).encode("utf-8"),
    )

    assert load_book_record(entry) == ("The Wealth of Nations", "Smith, Adam")


def test_a_book_with_no_record_classifies_without_one(bucket, caplog):
    import logging
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry
    from book_records.utils import load_book_record

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )

    with caplog.at_level(logging.WARNING):
        assert load_book_record(entry) == (None, None)

    assert "no metadata record" in caplog.text


def test_load_reads_the_html_the_entry_points_at(bucket, entries):
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry, html_key

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )
    bucket.put_object(Key=html_key(INDEX), Body=BOOK_HTML.encode("utf-8"))

    assert load_tag_text_pairs(entry) == BOOK_PAIRS
