"""Reducing a scraped page to the (tag, text) blocks the rest of the stage works on.

This is the widest module in the service and the one whose mistakes are quietest: a
container walked the wrong way does not raise, it just yields a book with its prose
glued together or its headings missing.
"""

from bs4 import BeautifulSoup

from book_records.reduce_html import (
    flatten_html_elements,
    reduce_to_text_tag_pairs,
    strip_non_book_elements,
)

from conftest import BOOK_HTML, BOOK_PAIRS, INDEX, SUBJECT


def pairs(html):
    """The reduction reduce_to_text_tag_pairs performs, minus the stylesheet.

    No CSS is inlined here, so `style` attributes are the only layout this sees and
    `space_out_children` is not part of the walk. Every test whose subject is the
    stylesheet calls reduce_to_text_tag_pairs directly instead.
    """
    soup = strip_non_book_elements(BeautifulSoup(html, "html.parser"))
    return list(flatten_html_elements(soup.body))


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


def test_a_line_break_separates_the_words_it_sits_between():
    """A <br> carries no text; without a separator of its own it welds the words on
    either side ("AN <br>INQUIRY" -> "ANINQUIRY"), which is how a title page stopped
    matching its own library record."""
    html = (
        "<body><h1>AN <br><span>INQUIRY</span><br>INTO THE<br>"
        "<span>PRINCIPLES OF POLITICAL OECONOMY:</span></h1></body>"
    )

    assert pairs(html) == [("h1", "AN INQUIRY INTO THE PRINCIPLES OF POLITICAL OECONOMY:")]


def test_a_page_marker_goes_whatever_markup_carried_it():
    """The three eras side by side: a TEI page break, a span whose only signal was its
    "#pageN" anchor id, and an EPUB pagebreak role. None of them is read here -- what
    the rule matches is the label and number the marker leaves in the text, which is
    the one thing all three write the same way."""
    html = (
        '<body><p><span class="tei tei-pb" id="pageiii">[pg iii]</span>'
        "The produce of the earth, "
        '<span class="x" id="Page_42">Pg 42</span>'
        "the united application of labour, "
        '<span epub:type="pagebreak">(p. 19)</span>'
        "and capital</p></body>"
    )

    assert pairs(html) == [
        (
            "p",
            "The produce of the earth, the united application of labour, and capital",
        )
    ]


def test_a_page_marker_in_no_markup_at_all_still_goes():
    """The reason for moving off the markup: some transcriptions leave the marker as
    bare text, with no span to carry a class, an id or a role."""
    html = "<body><p>the laws which regulate this [Pg 42] distribution</p></body>"

    assert pairs(html) == [("p", "the laws which regulate this distribution")]


def test_a_word_broken_across_a_page_turn_is_put_back_together():
    """The marker falls inside the word, because the printer's page ended there. The
    two halves close up; a space between them would invent a word the book has not."""
    html = '<body><p>the sci<span class="pagenum">Pg 33</span>ence of value</p></body>'

    assert pairs(html) == [("p", "the science of value")]


def test_a_page_marker_between_two_sentences_keeps_them_apart():
    """The mirror of the case above, and the reason the marker cannot simply vanish:
    an inline span carries no space of its own, so a sentence has already closed over
    the marker by the time the text is cleaned. A letter on one side only is what
    separates this from a broken word -- there the halves must rejoin, here the full
    stop must not swallow the sentence after it."""
    html = (
        '<body><p>special to the general.<span class="pagenum">Pg 10</span>'
        "On the other hand</p></body>"
    )

    assert pairs(html) == [("p", "special to the general. On the other hand")]


def test_a_page_marker_alone_in_a_block_leaves_no_block():
    html = '<body><p><span class="pagenum">[Pg 42]</span></p><p>Kept.</p></body>'

    assert pairs(html) == [("p", "Kept.")]


def test_a_bare_roman_numeral_is_not_a_page_marker():
    """What the move off the markup costs. A front-matter page number with no label is
    a roman numeral and nothing else, and no rule can tell it from the book's own
    prose -- "i" and "v" are words, and so is the pronoun "I". The markup knew; the
    text does not, and the numeral stays."""
    html = '<body><p><span class="pagenum">iii</span>The produce</p></body>'

    assert pairs(html) == [("p", "iiiThe produce")]


def test_an_abbreviation_is_not_read_as_a_page_reference():
    """"Chap. II" ends in a period and is followed by a numeral, which is the shape of
    "p. 42" if the letters in front of the period are ignored. They are not."""
    html = "<body><p>Book III, Chap. II, and Dollars. 1800 in all</p></body>"

    assert pairs(html) == [("p", "Book III, Chap. II, and Dollars. in all")]


def test_a_numbered_table_cell_is_not_mistaken_for_a_page_number():
    """A page number can look like any other short numeral, so the rule matches only a
    number wearing a page label -- never "the text is just a number". This cell is
    emptied all the same, by the digit strip that every number meets."""
    html = "<body><table><tr><td>1951</td><td>39.67</td></tr></table></body>"

    assert pairs(html) == []



def test_an_out_of_flow_element_inside_a_word_leaves_it_whole():
    """The same element, one character further on: the page turn fell inside "science"
    rather than between two words, and a separator here would invent a word."""
    html = (
        "<style>.pagenum { position: absolute; left: 92% }</style>"
        '<body><p>the sci<span class="pagenum">Pg 33</span>ence of value</p></body>'
    )

    assert reduce_to_text_tag_pairs(html) == [("p", "the science of value")]


def test_a_rule_inside_a_media_query_is_not_the_page_s_layout():
    """An at-rule holds for a device or a print run, and this stage is neither. The
    cost of reading them anyway is not spacing: the same corpus has books that hide
    `div.nothandheld` behind a media query, and it is a whole table of their data."""
    html = (
        "<style>@media print { .sidenote { float: right } }</style>"
        '<body><p>the work.<span class="sidenote">note</span>which followed</p></body>'
    )

    assert reduce_to_text_tag_pairs(html) == [("p", "the work.notewhich followed")]



def test_an_out_of_flow_element_with_no_text_around_it_needs_no_separator():
    """Nothing precedes or follows it, so there is nothing for it to weld to."""
    html = (
        "<style>.sidenote { float: right }</style>"
        '<body><span class="sidenote">Critique.</span></body>'
    )

    assert reduce_to_text_tag_pairs(html) == [("p", "Critique.")]



def test_an_out_of_flow_element_inside_a_dropped_tag_is_already_gone():
    """The stylesheet is read after <noscript> and friends are stripped, so a floated
    element inside one is never on the list at all, and cannot leave a separator
    standing where its text used to be."""
    html = (
        "<style>.sidenote { float: right }</style>"
        '<body><noscript><span class="sidenote">Enable JS.</span></noscript>'
        "<p>Kept.</p></body>"
    )

    assert reduce_to_text_tag_pairs(html) == [("p", "Kept.")]


def test_a_page_with_no_stylesheet_reduces_the_same_way():
    html = "<body><p>The produce of the earth.</p></body>"

    assert reduce_to_text_tag_pairs(html) == [("p", "The produce of the earth.")]


def test_a_footnote_reference_leaves_no_trace():
    """The real nesting: the marker is a span inside a pginternal anchor, and the
    inline elements collapse it onto the word before it. tokenize's per-character strip
    would recover the lemma from a fused "Aristotle2", but the concordance quotes the
    surface token, so the marker has to go here."""
    html = (
        '<body><p>Aristotle<a href="#note_2" class="pginternal">'
        '<span class="tei tei-noteref">2</span></a> and Xenophon</p></body>'
    )

    assert pairs(html) == [("p", "Aristotle and Xenophon")]


def test_a_reference_marker_goes_whatever_markup_carried_it():
    """The markup is not consulted at all, which is the point: `tei-noteref` is one PG
    template era, `fnanchor` another, and the third styles the marker with no class to
    match on. All three leave the same shape in the text."""
    html = (
        '<body><p>wages<a href="#Footnote_3" class="fnanchor pginternal">[3]</a> fall'
        '</p><p>rent<sup>12</sup> rises</p></body>'
    )

    assert pairs(html) == [("p", "wages fall"), ("p", "rent rises")]


def test_a_table_of_contents_link_is_not_a_reference_marker():
    """Nothing about a link makes it a marker; only a number fused to a word is one."""
    html = '<body><p><a href="#chap01" class="pginternal">CHAPTER I.</a></p></body>'

    assert pairs(html) == [("p", "CHAPTER I.")]


def test_a_number_the_book_wrote_itself_goes_with_all_the_rest():
    """The marker rules spare the book's own numbers -- an index entry, a year, a
    quantity -- and then the digit strip takes them anyway. What it leaves behind is the
    punctuation that surrounded them."""
    html = (
        '<body><p>Rent, <a href="#Page_142" class="pginternal">142</a></p>'
        '<p>In 1884 the price of 12 bushels fell.</p></body>'
    )

    assert pairs(html) == [
        ("p", "Rent,"),
        ("p", "In the price of bushels fell."),
    ]


def test_a_marker_set_apart_from_the_word_goes_with_its_space():
    """Not every transcription welds the marker on; where the source left a space, the
    marker and the space go together so the sentence closes onto its full stop."""
    html = (
        '<body><p>the produce of 100 days\' labour in Poland <a href="#note_1">[1]</a>.'
        '</p><p>indebtedness of corporations<a href="#f15">{15}</a></p></body>'
    )

    assert pairs(html) == [
        ("p", "the produce of days' labour in Poland."),
        ("p", "indebtedness of corporations"),
    ]


def test_an_enumeration_loses_its_numbering_entirely():
    """Parentheses stay out of the marker rules because they number the book's own
    clauses. The digit strip empties them anyway, and the orphan rule takes the brackets
    with it, so the clauses run on unnumbered."""
    html = (
        "<body><p>(1) the amount of business, and (2) the rapidity of circulation.</p>"
        "</body>"
    )

    assert pairs(html) == [
        ("p", "the amount of business, and the rapidity of circulation.")
    ]


def test_a_footnote_label_is_left_with_nothing_to_number():
    """The marker rules spare a label opening its block -- requiring a word before the
    number is what keeps it -- but the digit strip empties it and the orphan rule clears
    the brackets, so the note is left with no number of its own."""
    html = (
        '<body><p><a href="#FNanchor_47" class="fnanchor">[47]</a> Only from each '
        '52 acres is hay obtained.</p></body>'
    )

    assert pairs(html) == [("p", "Only from each acres is hay obtained.")]


def test_a_marker_on_a_heading_goes_too():
    """Headings run through the same cleaning, and PG marks them up the same way."""
    html = '<body><h2>Of the Rent of Land<a href="#Footnote_9">9</a></h2></body>'

    assert pairs(html) == [("h2", "Of the Rent of Land")]


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

    assert pairs(html) == [("p", "Year Price"), ("p", "Two shillings")]


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
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry
    from book_records.io import load_metadata

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )
    bucket.put_object(
        Key=f"metadata/{INDEX}.json",
        Body=json.dumps(
            {"title": ["The Wealth of Nations"], "author": ["Smith, Adam"]}
        ).encode("utf-8"),
    )

    assert load_metadata(entry) == ("The Wealth of Nations", "Smith, Adam")


def test_a_book_with_no_record_classifies_without_one(bucket, caplog):
    import logging
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry
    from book_records.io import load_metadata

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )

    with caplog.at_level(logging.WARNING):
        assert load_metadata(entry) == (None, None)

    assert "no metadata record" in caplog.text


def test_load_reads_the_html_the_entry_points_at(bucket, entries):
    from shared.tables.pipeline_entries import EntryStatus, PipelineEntry
    from book_records.io import load_html

    entry = PipelineEntry(
        book_id=INDEX, subject_ids={SUBJECT}, status=EntryStatus.SCRAPED_HTML
    )
    bucket.put_object(Key=f"html/{INDEX}.html", Body=BOOK_HTML.encode("utf-8"))

    assert reduce_to_text_tag_pairs(load_html(entry)) == BOOK_PAIRS
