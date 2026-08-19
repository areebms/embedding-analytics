import logging

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

from book_records.batch_index import load_batch_index
from llm_classify_request.send_request import get_client
from llm_parse_response.read_response import (
    get_batch_status,
    process_result,
    read_book_record,
    yield_anthropic_content,
)

from render import apply_semantic_blocks, render_html, render_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HTML_CONTENT_TYPE = "text/html; charset=utf-8"
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"


def content_text(content):
    """Join just the text a reply carries, block by block.

    yield_anthropic_content hands back one dict per content block, and only the text
    ones hold the classifier's answer -- thinking is prose, and process_result would
    reject it line by line. Anything else is logged rather than dropped in silence.
    """
    texts = []
    for block in content:
        if block.get("type") == "text":
            texts.append(block["text"])
        else:
            logger.warning("skipping %s block", block.get("type"))
    return "\n".join(texts)


def write_standardized(index, leveled):
    """Render and upload both derived artifacts, then advance the book's status.

    A single atomic update after both artifacts are safely in S3, matching the
    other stages: a book is never left marked STANDARDIZED with a missing page.
    """
    s3_loader = get_s3_loader()
    entries = get_pipeline_entries()

    page_key = f"html-standardized/{index}.html"
    s3_loader.upload_object(page_key, render_html(leveled, index), HTML_CONTENT_TYPE)

    text_key = f"text/{index}.txt"
    s3_loader.upload_object(text_key, render_text(leveled), TEXT_CONTENT_TYPE)

    entries.update_entries(
        PipelineEntry(
            platform_data=index,
            s3_standardized_html_key=page_key,
            s3_text_key=text_key,
            pipeline_status=EntryStatus.STANDARDIZED,
        )
    )
    logger.info("%s standardized: %s", index, page_key)
    logger.info("%s text: %s", index, text_key)


def collect(batch_id):
    """Settle one submitted batch, if it has finished.

    Safe to call repeatedly: it returns immediately while the batch is still
    running, and a book already written has left STANDARDIZE_SUBMITTED so a later
    call passes over it.
    """
    entries = get_pipeline_entries()
    client = get_client()

    batch_status = get_batch_status(client, batch_id)
    if batch_status != "ended":
        return {"batch_id": batch_id, "batch_status": batch_status, "standardized": 0}

    standardized = 0
    failures = []

    id_mapping = dict(load_batch_index(batch_id).id_mapping)

    for custom_id, content in yield_anthropic_content(client, batch_id):
        index = id_mapping.pop(custom_id, None)
        if index is None:
            logger.warning("batch %s: unknown custom_id %s", batch_id, custom_id)
            continue

        try:
            book = read_book_record(index)
            book_blocks = process_result(content_text(content), book)
            leveled = apply_semantic_blocks(book.tag_text_pairs, book_blocks)
            write_standardized(index, leveled)
            standardized += 1
        except Exception as error:
            failures.append((index, str(error)))
            logger.exception("%s: standardization failed", index)

    # Submitted, but the stream never carried a result for it.
    for index in id_mapping.values():
        failures.append((index, "no result returned for this custom_id"))

    # A book left at STANDARDIZE_SUBMITTED is unreachable: submit only sweeps
    # SCRAPED_HTML, so nothing would ever retry it. Hand every failure back.
    for index, reason in sorted(failures):
        logger.error("%s: %s", index, reason)
        entries.update_entries(
            PipelineEntry(
                platform_data=index, pipeline_status=EntryStatus.SCRAPED_HTML
            )
        )

    logger.info(
        "batch %s: %d standardized, %d failed", batch_id, standardized, len(failures)
    )
    return {
        "batch_id": batch_id,
        "batch_status": batch_status,
        "standardized": standardized,
    }
