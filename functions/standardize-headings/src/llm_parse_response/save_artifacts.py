import html

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import standardized_html_key, text_key

HTML_CONTENT_TYPE = "text/html; charset=utf-8"
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"


def render_html(tag_text_pairs, title):
    """A standalone page of h1/h2/h3/p.

    No styling and no attributes: this artifact exists to be chunked and indexed,
    and the raw html/ artifact remains the source of truth for anything that
    needs the original presentation.
    """
    body = "\n".join(
        f"<{tag}>{html.escape(text, quote=False)}</{tag}>"
        for tag, text in tag_text_pairs
    )
    return (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title, quote=False)}</title>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def render_text(tag_text_pairs):
    """Blocks separated by a blank line.

    Those blank lines are load-bearing: tokenize segments sentences within each
    block, so a heading that ends without a period stays off the front of the
    paragraph after it.
    """
    return "\n\n".join(text for _, text in tag_text_pairs) + "\n"


def save_html(index, tag_text_pairs):
    get_s3_loader().upload_object(
        standardized_html_key(index),
        render_html(tag_text_pairs, index),
        HTML_CONTENT_TYPE,
    )


def save_text(index, tag_text_pairs):
    get_s3_loader().upload_object(
        text_key(index), render_text(tag_text_pairs), TEXT_CONTENT_TYPE
    )
