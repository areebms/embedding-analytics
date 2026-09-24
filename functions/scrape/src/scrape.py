import argparse
import json
import logging
from time import sleep

from shared.commons import BookIndex
from shared.s3 import upload_html, upload_json
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)
from retrieve import MAX_BOOKS_PER_SUBJECT, get_book_ids, get_html, get_metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENGLISH = "English"


def update_status(book_id, status):
    if not get_pipeline_entries().set_status(book_id, status):
        logger.warning(
            "%s: the status guard refused the write to %s; the row keeps the status it "
            "already has.",
            book_id,
            status,
        )


def scrape_subject_book_list(subject_id):
    table = get_pipeline_entries()

    listed_ids = table.get_indexes(subject_id=subject_id)
    if len(listed_ids) >= MAX_BOOKS_PER_SUBJECT:
        logger.info(
            "subject %s: %d books already listed, at the %d cap; skipping the listing.",
            subject_id,
            len(listed_ids),
            MAX_BOOKS_PER_SUBJECT,
        )
        return {
            "subject": subject_id,
            "found": len(listed_ids),
            "created": 0,
            "indexes": [str(book_id) for book_id in listed_ids],
        }

    source_ids = get_book_ids(subject_id)
    subject_index = BookIndex.parse(subject_id)

    book_ids = []
    created = 0
    for source_id in source_ids:

        book_id = BookIndex(source_id)
        try:
            entry = PipelineEntry(
                book_id=book_id,
                subject_ids={subject_index},
                status=EntryStatus.LISTED,
            )
            if table.put_entry(entry):
                created += 1
            else:
                table.add_subject(book_id, subject_index)

        except Exception:
            logger.exception("%s: failed to create pipeline entry", book_id)

        book_ids.append(str(book_id))

    logger.info(
        "subject %s: %d books found, %d new pipeline entries created.",
        subject_id,
        len(source_ids),
        created,
    )

    return {
        "subject": subject_id,
        "found": len(source_ids),
        "created": created,
        "indexes": book_ids,
    }


def scrape_book_metadata(book_id):
    """Fetch and store a book's metadata. Returns the status the book ended at."""

    pipeline_entry = get_pipeline_entries().get_entry(book_id)
    if pipeline_entry.status != EntryStatus.LISTED:
        logger.info(
            "%s is at %s, not %s; skipping metadata scrape.",
            book_id,
            pipeline_entry.status,
            EntryStatus.LISTED,
        )
        return pipeline_entry.status

    metadata = get_metadata(book_id.source_id)

    upload_json(pipeline_entry.s3_metadata_key, json.dumps(metadata))

    if ENGLISH in metadata.get("language", []):
        status = EntryStatus.SCRAPED_METADATA
    else:
        status = EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH

    update_status(book_id, status)

    if status == EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH:
        logger.info("%s is not %s; skipping.", book_id, ENGLISH)
    else:
        logger.info("%s metadata scraped.", book_id)
    return status


def scrape_book_content(book_id):
    """Fetch and store a book's raw HTML. Returns the status the book ended at."""
    pipeline_entry = get_pipeline_entries().get_entry(book_id)
    if pipeline_entry.status != EntryStatus.SCRAPED_METADATA:
        logger.info(
            "%s is at %s, not %s; skipping content scrape.",
            book_id,
            pipeline_entry.status,
            EntryStatus.SCRAPED_METADATA,
        )
        return pipeline_entry.status

    html_content = get_html(book_id.source_id)

    upload_html(pipeline_entry.s3_html_key, html_content)

    update_status(book_id, EntryStatus.SCRAPED_HTML)
    logger.info("%s html scraped.", book_id)
    return EntryStatus.SCRAPED_HTML


def sleepy_map(function, book_ids, sleep_seconds=3):
    """Run `function` over every book in `book_ids`; one failure doesn't stop the rest."""
    logger.info("%s: %d book(s): %s", function.__name__, len(book_ids), book_ids)
    for index in book_ids:
        try:
            function(index)
        except Exception:
            logger.exception("%s: %s failed", index, function.__name__)
        sleep(sleep_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run one scrape-pipeline stage in bulk."
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    subject = stages.add_parser("SUBJECT", help="seed pipeline entries from a subject")
    subject.add_argument("--subject", required=True)
    stages.add_parser(
        "METADATA", help=f"scrape metadata for every book at {EntryStatus.LISTED}"
    )
    stages.add_parser(
        "CONTENT",
        help=f"scrape content for every book at {EntryStatus.SCRAPED_METADATA}",
    )

    args = parser.parse_args()

    if args.stage == "SUBJECT":
        scrape_subject_book_list(args.subject)
    elif args.stage == "METADATA":
        book_ids = get_pipeline_entries().get_indexes(EntryStatus.LISTED)
        sleepy_map(scrape_book_metadata, book_ids, sleep_seconds=3)
    else:
        book_ids = get_pipeline_entries().get_indexes(EntryStatus.SCRAPED_METADATA)
        sleepy_map(scrape_book_content, book_ids, sleep_seconds=3)
