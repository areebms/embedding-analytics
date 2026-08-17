import argparse
import json
import logging
from time import sleep

from shared.commons import BookIndex
from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
    html_key,
    metadata_key,
)
from retrieve import get_book_ids, get_html, get_metadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENGLISH = "English"


def get_status(index):
    """The book's current status. Seeding is a prerequisite, so a missing row is an error."""
    entry = get_pipeline_entries().get_entry(index, ["pipeline_status"])

    if entry is None:
        raise LookupError(
            f"{index} has no pipeline entry; run `scrape.py list --subject ...` first"
        )
    if entry.pipeline_status is None:
        raise LookupError(f"{index} has a pipeline entry with no status")

    return entry.pipeline_status


def update_status(index, status):
    get_pipeline_entries().update_entries(
        PipelineEntry(platform_data=index, pipeline_status=status)
    )


def scrape_subject_book_list(subject_id):
    table = get_pipeline_entries()
    book_ids = get_book_ids(subject_id)

    indexes = []
    created = 0
    for book_id in book_ids:
        index = BookIndex(book_id)

        try:
            entry = PipelineEntry(
                platform_data=index, pipeline_status=EntryStatus.CREATED
            )
            if table.put_entry(entry):
                created += 1
        except Exception:
            logger.exception("%s: failed to create pipeline entry", index)
        indexes.append(str(index))

    logger.info(
        "subject %s: %d books found, %d new pipeline entries created.",
        subject_id,
        len(book_ids),
        created,
    )

    return {
        "subject": subject_id,
        "found": len(book_ids),
        "created": created,
        "indexes": indexes,
    }


def scrape_book_metadata(index):
    """Fetch and store a book's metadata. Returns the status the book ended at."""
    current_status = get_status(index)
    if current_status != EntryStatus.CREATED:
        logger.info(
            "%s is at %s, not %s; skipping metadata scrape.",
            index,
            current_status,
            EntryStatus.CREATED,
        )
        return current_status

    s3_loader = get_s3_loader()

    metadata = get_metadata(index.source_id)
    s3_loader.upload_object(
        metadata_key(index),
        json.dumps(metadata),
        content_type="application/json; charset=utf-8",
    )

    if ENGLISH in metadata.get("language", []):
        status = EntryStatus.SCRAPED_METADATA
    else:
        status = EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH

    update_status(index, status)

    if status == EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH:
        logger.info("%s is not %s; skipping.", index, ENGLISH)
    else:
        logger.info("%s metadata scraped.", index)
    return status


def scrape_book_content(index):
    """Fetch and store a book's raw HTML. Returns the status the book ended at."""
    current_status = get_status(index)
    if current_status != EntryStatus.SCRAPED_METADATA:
        logger.info(
            "%s is at %s, not %s; skipping content scrape.",
            index,
            current_status,
            EntryStatus.SCRAPED_METADATA,
        )
        return current_status

    s3_loader = get_s3_loader()

    key = html_key(index)
    s3_loader.upload_object(key, get_html(index.source_id), "text/html; charset=utf-8")

    update_status(index, EntryStatus.SCRAPED_HTML)
    logger.info("%s scraped: %s", index, key)
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
    subject = stages.add_parser("list", help="seed pipeline entries from a subject")
    subject.add_argument("--subject", required=True)
    stages.add_parser(
        "metadata", help=f"scrape metadata for every book at {EntryStatus.CREATED}"
    )
    stages.add_parser(
        "content",
        help=f"scrape content for every book at {EntryStatus.SCRAPED_METADATA}",
    )

    args = parser.parse_args()

    if args.stage == "list":
        scrape_subject_book_list(args.subject)
    elif args.stage == "metadata":
        book_ids = get_pipeline_entries().get_indexes(EntryStatus.CREATED)
        sleepy_map(scrape_book_metadata, book_ids, sleep_seconds=3)
    else:
        book_ids = get_pipeline_entries().get_indexes(EntryStatus.SCRAPED_METADATA)
        sleepy_map(scrape_book_content, book_ids, sleep_seconds=3)
