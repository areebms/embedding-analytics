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


def test_load_reads_the_html_the_entry_points_at(bucket, entries):
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry, html_key

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )
    bucket.put_object(Key=html_key(INDEX), Body=BOOK_HTML.encode("utf-8"))

    assert load_tag_text_pairs(entry) == BOOK_PAIRS
