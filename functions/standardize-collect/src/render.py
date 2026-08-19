import html

from book_records.constants import HEADING_ELEMENTS
from llm_classify_request.send_request import HeadingSemanticBlockError

SEMANTIC_BLOCK_TO_LEVEL = {
    "title": "h1",
    "part": "h2",
    "chapter": "h2",
    "section": "h3",
    "subsection": "h3",
    "front_matter": "h3",
    "back_matter": "h3",
}


def apply_semantic_blocks(tag_text_pairs, book_blocks):
    """Retag each heading to the level its semantic block renders as.

    Headings are numbered by their order in the book, and that ordinal -- not the
    block's index among all blocks -- is the position the classifier answered
    against. Non-heading blocks pass through untouched.
    """
    semantic_block_map = {row.position: row.semantic_block for row in book_blocks}
    leveled = []
    position = 0
    for tag, text in tag_text_pairs:
        if tag not in HEADING_ELEMENTS:
            leveled.append((tag, text))
            continue
        if position not in semantic_block_map:
            raise HeadingSemanticBlockError(
                f"no semantic block assigned for heading position {position}"
            )
        leveled.append((SEMANTIC_BLOCK_TO_LEVEL[semantic_block_map[position]], text))
        position += 1
    return leveled


def render_html(leveled, title):
    """A standalone page of h1/h2/h3/p.

    No styling and no attributes: this artifact exists to be chunked and indexed,
    and the raw html/ artifact remains the source of truth for anything that
    needs the original presentation.
    """
    body = "\n".join(
        f"<{tag}>{html.escape(text, quote=False)}</{tag}>" for tag, text in leveled
    )
    return (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title, quote=False)}</title>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def render_text(leveled):
    """Blocks separated by a blank line.

    Those blank lines are load-bearing: tokenize segments sentences within each
    block, so a heading that ends without a period stays off the front of the
    paragraph after it.
    """
    return "\n\n".join(text for _, text in leveled) + "\n"
