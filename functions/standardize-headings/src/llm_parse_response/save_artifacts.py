import html

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import standardized_html_key, text_key

HTML_CONTENT_TYPE = "text/html; charset=utf-8"
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"
UNTRAINABLE_BLOCKS = ("contents", "index", "errata", "advertisement")
HEADING_LEVELS = ("h1", "h2", "h3")


def book_title(blocks, record_title, fallback):
    """The library record's title, else the merged `title` heading, else the index.

    The record comes first because a title page is not always transcribed as a heading
    -- gutenberg-30107 has none at all, and would otherwise be titled by its index.
    """
    if record_title:
        return record_title
    for entry in blocks:
        if entry.block == "title" and entry.tag in HEADING_LEVELS:
            return entry.text
    return fallback


def render_html(blocks, index, record_title=None):
    """A standalone page of h1/h2/h3/p, one element per block.
    """
    body = "\n".join(
        f"<{entry.tag}{f' data-block="{entry.block}"' if entry.block else ''}>"
        f"{html.escape(entry.text, quote=False)}"
        f"</{entry.tag}>"
        for entry in blocks
    )
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(book_title(blocks, record_title, str(index)), quote=False)}</title>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def render_text(blocks):
    """Blocks separated by a blank line, back matter left out.
    """
    return (
        "\n\n".join(
            entry.text for entry in blocks if entry.block not in UNTRAINABLE_BLOCKS
        )
        + "\n"
    )


def save_html(index, blocks, record_title=None):
    get_s3_loader().upload_object(
        standardized_html_key(index),
        render_html(blocks, index, record_title),
        HTML_CONTENT_TYPE,
    )


def save_text(index, blocks):
    get_s3_loader().upload_object(
        text_key(index), render_text(blocks), TEXT_CONTENT_TYPE
    )
