import logging

from shared.commons import BookIndex, get_index
from shared.s3 import load_text, upload_csv
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries
)
from tokenize_text import Token, tokenize_passage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PENDING = (EntryStatus.STANDARDIZED,)
MAX_BOOKS_PER_SUBJECT = 50


def resolve_subject(subject_id: str) -> list[BookIndex]:
    book_ids = get_pipeline_entries().get_indexes(
        status=EntryStatus.STANDARDIZED, subject_id=subject_id
    )

    if len(book_ids) > MAX_BOOKS_PER_SUBJECT:
        logger.info(
            "%s resolved %d books; taking the first %d, the rest keep STANDARDIZED "
            "for the next run.",
            subject_id,
            len(book_ids),
            MAX_BOOKS_PER_SUBJECT,
        )
        book_ids = book_ids[:MAX_BOOKS_PER_SUBJECT]

    return book_ids


def get_entries(book_ids: list[str]) -> list[PipelineEntry]:

    book_ids = list(book_ids)
    entries = get_pipeline_entries().get_entries(book_ids)

    if len(entries) != len(book_ids):
        logger.warning(
            "%d of %d book(s) have no pipeline entry.",
            len(book_ids) - len(entries),
            len(book_ids),
        )

    pending = []
    for entry in entries:
        if entry.status in PENDING:
            pending.append(entry)
        else:
            logger.info(
                "%s is at %s, not %s; skipping tokenize.",
                entry.book_id,
                entry.status,
                ", ".join(PENDING),
            )

    return pending


def set_status(book_id, status):
    if not get_pipeline_entries().set_status(book_id, status):
        logger.warning("%s: the status guard refused the write to %s.", book_id, status)


def yield_passages(entry: PipelineEntry):
    """Each non-empty passage of the entry's text, in document order."""
    for passage in load_text(entry.s3_text_key).split("\n\n"):
        passage = passage.strip()
        if passage:
            yield passage


def upload_passage_data(
    entry: PipelineEntry, tokenized_passages: list[list[Token]]
) -> None:
    """The three artifacts' rows: one row per passage, one row set per Token field."""
    texts, lemmas, tags = [], [], []
    for passage in tokenized_passages:
        texts.append([token.text for token in passage])
        lemmas.append([token.lemma for token in passage])
        tags.append([token.tag for token in passage])

    upload_csv(entry.s3_token_texts_key, texts)
    upload_csv(entry.s3_token_lemmas_key, lemmas)
    upload_csv(entry.s3_token_tags_key, tags)


def tokenize_entry(entry: PipelineEntry) -> None:
    """One book: spaCy over each passage, the three artifacts, then the status."""
    tokenized_passages: list[list[Token]] = []
    for passage in yield_passages(entry):
        tokenized_passages.append(tokenize_passage(passage))

    if not tokenized_passages:
        raise ValueError(f"{entry.s3_text_key} holds no passages")

    upload_passage_data(entry, tokenized_passages)
    set_status(entry.book_id, EntryStatus.TOKENIZED)


def tokenize_entries(entries: list[PipelineEntry]) -> dict:
    """Every book handed in, one process. A book that raises is left at STANDARDIZED
    and named in `failed` rather than ending the run."""
    tokenized = 0
    failed = []
    for entry in entries:
        try:
            tokenize_entry(entry)
        except Exception:
            logger.exception(
                "%s failed to tokenize; left at %s.",
                entry.book_id,
                EntryStatus.STANDARDIZED,
            )
            failed.append(str(entry.book_id))
            continue

        tokenized += 1

    logger.info(
        "%d of %d book(s) tokenized, %d failed.", tokenized, len(entries), len(failed)
    )
    return {"found": len(entries), "tokenized": tokenized, "failed": failed}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    tokenize_entries(get_entries([get_index()]))
