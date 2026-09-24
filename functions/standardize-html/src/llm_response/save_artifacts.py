import html

from shared.s3 import upload_html, upload_txt

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


def save_html(entry, blocks, record_title=None):
    upload_html(
        entry.s3_standardized_html_key,
        render_html(blocks, entry.book_id, record_title),
    )


def save_text(entry, blocks):
    upload_txt(entry.s3_text_key, render_text(blocks))
