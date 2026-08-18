import re
from collections.abc import Iterable, Iterator

from bs4 import (
    BeautifulSoup,
    CData,
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
    Tag,
)

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import html_key

from book_records.constants import HEADING_ELEMENTS
from book_records.schemas import TagTextPair

# The Project Gutenberg license wrapper: never part of the book itself.
PG_BOILERPLATE_IDS = ("pg-header", "pg-footer")

# Tags whose text is never book content.
SKIP_TAGS = ("img", "hr", "script", "style", "head", "meta", "link", "noscript")

# Comments, doctypes and friends all subclass NavigableString, so without this the
# walk would treat "<!-- Transcriber's note -->" as a paragraph of the book.
NON_TEXT_STRINGS = (Comment, Doctype, CData, ProcessingInstruction, Declaration)

# Finding any of these inside an unrecognised container means it holds real structure
# worth descending into, rather than being one block of text in a wrapper.
CONTAINER_ELEMENTS = HEADING_ELEMENTS + (
    "p",
    "table",
    "ul",
    "ol",
    "dl",
    "div",
    "blockquote",
    "pre",
)

WHITESPACE = re.compile(r"\s+")
BLANK_LINE = re.compile(r"\n\s*\n")


def clean_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def clean_element_text(element: Tag) -> str:
    return clean_text(element.get_text(" "))


def as_paragraph_elements(texts: Iterable[str]) -> Iterator[TagTextPair]:
    """Label each text as body prose, dropping any that cleaned away to nothing."""
    return (("p", text) for text in texts if text)


def list_item_texts(element: Tag) -> Iterator[str]:
    """One text per <li>. Nested lists come along inside their parent item."""
    for item in element.find_all("li", recursive=False):
        yield clean_element_text(item)


def definition_list_texts(element: Tag) -> Iterator[str]:
    """One text per term and per definition, in document order."""
    for item in element.find_all(["dt", "dd"], recursive=False):
        yield clean_element_text(item)


def table_row_texts(element: Tag) -> Iterator[str]:
    """One text per row, cells joined by spaces.

    Column layout carries no meaning here.
    """
    for row in element.find_all("tr"):
        cells = [clean_element_text(cell) for cell in row.find_all(["td", "th"])]
        yield clean_text(" ".join(cell for cell in cells if cell))


def blank_line_separated_texts(element: Tag) -> Iterator[str]:
    """Blank lines are the only paragraph breaks <blockquote> and <pre> give us."""
    for paragraph in BLANK_LINE.split(element.get_text(" ")):
        yield clean_text(paragraph)


def flatten_html_elements(element: Tag) -> Iterator[TagTextPair]:
    """Yield the (tag, text) blocks of `element`, depth-first in document order."""
    for child in element.children:
        if isinstance(child, NavigableString):
            yield from as_paragraph_elements([clean_text(str(child))])
        elif child.name in HEADING_ELEMENTS:
            if text := clean_element_text(child):
                yield child.name, text
        elif child.name == "p":
            yield from as_paragraph_elements([clean_element_text(child)])
        elif child.name in ("ul", "ol"):
            yield from as_paragraph_elements(list_item_texts(child))
        elif child.name == "dl":
            yield from as_paragraph_elements(definition_list_texts(child))
        elif child.name == "table":
            yield from as_paragraph_elements(table_row_texts(child))
        elif child.name in ("blockquote", "pre"):
            yield from as_paragraph_elements(blank_line_separated_texts(child))
        elif child.find(CONTAINER_ELEMENTS):
            yield from flatten_html_elements(child)
        else:
            yield from as_paragraph_elements([clean_element_text(child)])


def strip_pg_boilerplate(soup: Tag) -> Tag:
    """Remove the PG license header/footer in place, returning the same node."""
    for element_id in PG_BOILERPLATE_IDS:
        element = soup.find(attrs={"id": element_id})
        if element is not None:
            element.decompose()
    return soup


def strip_non_book_elements(soup: Tag) -> Tag:
    """Remove everything that is not book text in place, returning the same node."""
    strip_pg_boilerplate(soup)
    for element in soup.find_all(SKIP_TAGS):
        element.decompose()
    for string in soup.find_all(string=lambda s: isinstance(s, NON_TEXT_STRINGS)):
        string.extract()
    return soup


def prepare_book_body(html: str) -> Tag:
    """Parse a book's HTML and return the element flatten_html_elements should walk."""
    soup = strip_non_book_elements(BeautifulSoup(html, "html.parser"))
    return soup.body or soup


def load_tag_text_pairs(index: str) -> list[TagTextPair]:
    html = get_s3_loader().load_text(html_key(index))
    return list(flatten_html_elements(prepare_book_body(html)))
