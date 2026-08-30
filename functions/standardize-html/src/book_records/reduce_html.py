import logging
import re
from collections.abc import Iterable, Iterator

from premailer import Premailer

from bs4 import (
    BeautifulSoup,
    Comment,
    Doctype,
    NavigableString,
    Tag,
)

from book_records.schemas import TagTextPair
from constants import HEADING_ELEMENTS

INLINE_ELEMENTS = (
    "a",
    "abbr",
    "b",
    "big",
    "cite",
    "code",
    "em",
    "font",
    "i",
    "q",
    "s",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "tt",
    "u",
    "var",
)
SKIP_TAGS = ("img", "hr", "script", "style", "head", "meta", "link", "noscript")
NON_TEXT_STRINGS = (Comment, Doctype)
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

PAGE_ROMAN = r"[ivx][ivxlcdm]*"
DIGITS = re.compile(r"\d+")
PAGE_MARKER_TEXT = re.compile(
    rf"(?i)(?:pg|(?<![\w.])(?:p\.|s\.))(?![\w-])\s*(?:{PAGE_ROMAN}(?![\w-]))?"
)
EMPTY_BRACKETS = re.compile(r"\s*[\[\({]\s*[\]\)}]\s*")
ORPHANED_PUNCTUATION = re.compile(r"(?<!\S)[^\w\s\[\](){}]+(?!\S)")
WHITESPACE = re.compile(r"\s+")
OUT_OF_FLOW = re.compile(
    r"(?i)position\s*:\s*(?:absolute|fixed)|float\s*:\s*(?:left|right)"
)


def get_neighbouring_char(match: re.Match) -> tuple[str, str]:
    """The characters either side of the match -- empty at the ends of the string."""
    before = match.string[match.start() - 1] if match.start() else ""
    return before, match.string[match.end() : match.end() + 1]


def drop_brackets(match: re.Match) -> str:
    """What an emptied bracket pair leaves: nothing when the text closes over it, a
    space when it was holding two things apart."""
    group = match.group()
    before, after = get_neighbouring_char(match)
    if before.isalnum() and after and not after.isalnum() and after not in "([{":
        return ""  # let the word close onto its own punctuation: "Poland [1]." -> "Poland."
    if before.isalnum() and after.isalnum() and group == group.strip():
        return ""  # no whitespace either side, so it sat inside a word: "improve[20]ment"
    return " "


def drop_page_marker(match: re.Match) -> str:
    """A marker that fell inside a word closes it up; anywhere else it leaves a space."""
    before, after = get_neighbouring_char(match)
    return "" if before.isalpha() and after.isalpha() else " "


def clean_text(text: str) -> str:
    spaced = DIGITS.sub(" ", text)
    unpaged = PAGE_MARKER_TEXT.sub(drop_page_marker, spaced)
    unbracketed = EMPTY_BRACKETS.sub(drop_brackets, unpaged)
    stripped = ORPHANED_PUNCTUATION.sub("", unbracketed)
    return WHITESPACE.sub(" ", stripped).strip()


def clean_element_text(node: Tag | str) -> str:
    """Descendant text, separated only where the markup implies a break."""
    if isinstance(node, NavigableString):
        return "" if isinstance(node, NON_TEXT_STRINGS) else str(node)

    parts = []
    for child in node.children:
        if isinstance(child, Tag) and child.name == "br":
            parts.append(" ")
            continue
        text = clean_element_text(child)
        if not text:
            continue
        is_block = isinstance(child, Tag) and child.name not in INLINE_ELEMENTS
        parts.append(f" {text} " if is_block else text)
    return "".join(parts)


def as_paragraph_elements(texts: Iterable[str]) -> Iterator[TagTextPair]:
    """Label each text as body prose, dropping any that cleans away to nothing."""
    for text in texts:
        cleaned = clean_text(text)
        if cleaned:
            yield "p", cleaned


def list_item_texts(element: Tag) -> Iterator[str]:
    """One text per <li>. Nested lists come along inside their parent item."""
    for item in element.find_all("li", recursive=False):
        yield clean_element_text(item)


def definition_list_texts(element: Tag) -> Iterator[str]:
    """One text per term and per definition, in document order."""
    for item in element.find_all(["dt", "dd"], recursive=False):
        yield clean_element_text(item)


def table_row_texts(element: Tag) -> Iterator[str]:
    """One text per row, cells joined by spaces."""
    for row in element.find_all("tr"):
        cells = [clean_element_text(cell) for cell in row.find_all(["td", "th"])]
        yield " ".join(cell for cell in cells if cell)


def blank_line_separated_texts(element: Tag) -> Iterator[str]:
    """Blank lines are the only paragraph breaks <blockquote> and <pre> give us."""
    yield from re.split(r"\n\s*\n", clean_element_text(element))


def flatten_html_elements(element: Tag) -> Iterator[TagTextPair]:
    """Yield the (tag, text) blocks of `element`, depth-first in document order."""
    for child in element.children:
        if isinstance(child, NavigableString):
            yield from as_paragraph_elements([str(child)])
        elif child.name in HEADING_ELEMENTS:
            text = clean_text(clean_element_text(child))
            if text:
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


def has_next_element(element: Tag, next: bool = True) -> bool:
    next_elements = element.next_elements if next else element.previous_elements
    for next_element in next_elements:
        if any(parent is element for parent in next_element.parents):
            continue
        if not isinstance(next_element, NavigableString) or isinstance(
            next_element, NON_TEXT_STRINGS
        ):
            continue
        if next_element:
            return next_element[0 if next else -1].isalpha()
    return False


def space_out_children(tag: Tag) -> None:
    for element in tag.find_all(style=True):
        if not OUT_OF_FLOW.search(element["style"]):
            continue
        if has_next_element(element, next=True) and has_next_element(
            element, next=False
        ):
            continue
        element.insert_before(NavigableString(" "))
        element.insert_after(NavigableString(" "))


def strip_non_book_elements(soup: Tag) -> Tag:
    """Remove everything that is not book text in place, returning the same node."""
    for element in soup.find_all(attrs={"id": ("pg-header", "pg-footer")}):
        element.decompose()
    for element in soup.find_all(SKIP_TAGS):
        element.decompose()
    for string in soup.find_all(string=lambda s: isinstance(s, NON_TEXT_STRINGS)):
        string.extract()
    return soup



def reduce_to_text_tag_pairs(html: str) -> list[TagTextPair]:
    classless_html = Premailer(
        html,
        allow_network=False,
        allow_loading_external_files=False,
        disable_validation=True,
        cssutils_logging_level=logging.CRITICAL,
    ).transform()
    soup = strip_non_book_elements(BeautifulSoup(classless_html, "html.parser"))
    space_out_children(soup)
    return list(flatten_html_elements(soup.body))
