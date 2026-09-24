import logging

from book_records.io import save_batch_index, save_book_tag_text_pairs
from book_records.utils import get_book_tag_text_pairs
from constants import BATCH_ENDED, MAX_BOOKS_PER_SUBJECT
from llm_request.send_anthropic_request import send_message_batch
from shared.tables.pipeline_entries import EntryStatus, get_pipeline_entries

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PENDING = (EntryStatus.SCRAPED_HTML, EntryStatus.STANDARDIZE_UNRESOLVED)


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

    return [entry for entry in entries if entry.status in PENDING]


def resolve_subject(subject_id):
    book_ids = get_pipeline_entries().get_indexes(
        status=EntryStatus.SCRAPED_HTML, subject_id=subject_id
    )

    if len(book_ids) > MAX_BOOKS_PER_SUBJECT:
        logger.info(
            "%s resolved %d books; taking the first %d, the rest keep SCRAPED_HTML "
            "for the next run.",
            subject_id,
            len(book_ids),
            MAX_BOOKS_PER_SUBJECT,
        )
        book_ids = book_ids[:MAX_BOOKS_PER_SUBJECT]

    return book_ids


def submit(pending_entries):

    book_tag_text_pairs = get_book_tag_text_pairs(pending_entries)

    if not book_tag_text_pairs:
        return {"batch_id": None, "book_count": 0, "batch_status": BATCH_ENDED}

    save_book_tag_text_pairs(pending_entries, book_tag_text_pairs)

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
