import html

from shared.s3 import upload_html, upload_txt
from shared.tables.pipeline_entries import standardized_html_key, text_key

from constants import UNTRAINABLE_BLOCKS


def render_html(blocks, index, record_title=None):
    body = "\n".join(
        f"<{entry.tag}{f' data-block="{entry.block}"' if entry.block else ''}>"
        f"{html.escape(entry.text, quote=False)}"
        f"</{entry.tag}>"
        for entry in blocks
    )
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(record_title or str(index), quote=False)}</title>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def render_text(blocks):
    """Blocks separated by a blank line, the paratext left out."""
    return (
        "\n\n".join(
            entry.text for entry in blocks if entry.block not in UNTRAINABLE_BLOCKS
        )
        + "\n"
    )


def save_html(index, blocks, record_title=None):
    upload_html(standardized_html_key(index), render_html(blocks, index, record_title))


def save_text(index, blocks):
    upload_txt(text_key(index), render_text(blocks))
