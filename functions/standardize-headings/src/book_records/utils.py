import logging

from shared.tables.pipeline_entries import EntryStatus, get_pipeline_entries
from book_records.io import load_html, load_metadata
from book_records.reduce_html import reduce_to_text_tag_pairs
from book_records.schemas import BookTagTextPairs
from constants import HEADING_ELEMENTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_book_tag_text_pairs(entries) -> list[BookTagTextPairs]:

    book_tag_text_pairs = []

    for entry in entries:
        tag_text_pairs = reduce_to_text_tag_pairs(load_html(entry))

        tags = {tag for tag, _ in tag_text_pairs}
        if tags.isdisjoint(HEADING_ELEMENTS):
            get_pipeline_entries().set_status(
                entry.book_id, EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS
            )
            logger.info("%s has no headings; skipping.", entry.book_id)
            continue

        title, author = load_metadata(entry)

        book_tag_text_pairs.append(
            BookTagTextPairs(
                index=entry.book_id,
                tag_text_pairs=tag_text_pairs,
                title=title,
                author=author,
            )
        )

    return book_tag_text_pairs
