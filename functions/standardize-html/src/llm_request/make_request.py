import logging

from book_records.io import save_batch_index, save_book_tag_text_pairs
from book_records.utils import get_book_tag_text_pairs
from constants import BATCH_ENDED
from llm_request.send_anthropic_request import send_message_batch
from shared.tables.pipeline_entries import EntryStatus, get_pipeline_entries

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BooksInFlightError(Exception):
    """Some of the requested books already belong to an open batch.

    Refusing the whole call rather than dropping those books: a second batch over a
    book already in flight is paid for twice, and the two would settle against the same
    row. The caller re-runs once the open batch has been collected.
    """


def get_entries(book_ids):
    entries = get_pipeline_entries().get_entries(book_ids)

    submitted_entries = [
        entry for entry in entries if entry.status == EntryStatus.STANDARDIZE_SUBMITTED
    ]
    if submitted_entries:
        raise BooksInFlightError(
            f"{len(submitted_entries)} of {len(entries)} book(s) still in flight: "
            f"{[str(entry.book_id) for entry in submitted_entries]}"
        )

    return [entry for entry in entries if entry.status == EntryStatus.SCRAPED_HTML]


def submit(pending_entries):

    book_tag_text_pairs = get_book_tag_text_pairs(pending_entries)

    if not book_tag_text_pairs:
        return {"batch_id": None, "book_count": 0, "batch_status": BATCH_ENDED}

    save_book_tag_text_pairs(book_tag_text_pairs)

    batch_id, batch_status = send_message_batch(book_tag_text_pairs)

    save_batch_index(batch_id, book_tag_text_pairs)

    pipeline_entries = get_pipeline_entries()
    for book_tag_text_pair in book_tag_text_pairs:
        pipeline_entries.set_status(
            book_tag_text_pair.index, EntryStatus.STANDARDIZE_SUBMITTED
        )

    return {
        "batch_id": batch_id,
        "book_count": len(book_tag_text_pairs),
        "batch_status": batch_status,
    }
