import logging
from collections import defaultdict

import numpy as np

from shared.commons import BookIndex
from shared.s3 import load_csv, load_file, load_json
from shared.tables.book_terms import get_book_term_table
from shared.tables.corpus_terms import get_corpus_term_table
from shared.tables.pipeline_entries import (
    BookMetadata,
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PENDING = (EntryStatus.EMBEDDINGS_CREATED,)
FIELDS = ["book_id", "status", "metadata"]
MAX_BOOKS_PER_SUBJECT = 50


def resolve_subject(subject_id: str) -> list[BookIndex]:
    book_ids = get_pipeline_entries().get_indexes(
        status=EntryStatus.EMBEDDINGS_CREATED, subject_id=subject_id
    )

    if len(book_ids) > MAX_BOOKS_PER_SUBJECT:
        logger.info(
            "%s resolved %d books; taking the first %d, the rest keep %s "
            "for the next run.",
            subject_id,
            len(book_ids),
            MAX_BOOKS_PER_SUBJECT,
            EntryStatus.EMBEDDINGS_CREATED,
        )
        book_ids = book_ids[:MAX_BOOKS_PER_SUBJECT]

    return book_ids


def get_entries(book_ids: list[str]) -> list[PipelineEntry]:
    book_ids = list(book_ids)
    keys = [{"book_id": str(BookIndex.parse(book_id))} for book_id in book_ids]
    entries = [
        PipelineEntry.model_validate(item)
        for item in get_pipeline_entries().batch_get_entries(keys, FIELDS)
    ]

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
                "%s is at %s, not %s; skipping publish.",
                entry.book_id,
                entry.status,
                ", ".join(PENDING),
            )

    return sorted(pending, key=lambda entry: entry.book_id)


class BookEmbeddings:

    def __init__(self, terms, vectors, counts):
        self.term_ilocs = {str(term): iloc for iloc, term in enumerate(terms)}
        self.vectors = vectors
        self.counts = counts

    @classmethod
    def from_s3(cls, entry: PipelineEntry):
        logger.info("%s: loading embeddings", entry.book_id)
        with load_file(entry.s3_embeddings_key) as (_, local_path):
            with np.load(local_path, allow_pickle=False) as archive:
                return cls(
                    archive["terms"], archive["vectors"], archive["attr_count"]
                )

    @property
    def vocab(self):
        return set(self.term_ilocs)

    def get_vector(self, term) -> bytes:
        return self.vectors[self.term_ilocs[term]].astype(np.float16).tobytes()

    def get_count(self, term) -> int:
        return int(self.counts[self.term_ilocs[term]])


class RawPOSData:

    def __init__(self, token_lemmas, token_tags):
        self.token_lemmas = token_lemmas
        self.token_tags = token_tags

        self.lemma_iloc = defaultdict(set)
        self.lemma_tags = defaultdict(set)

    @classmethod
    def from_s3(cls, entry: PipelineEntry):
        logger.info("%s: loading POS data", entry.book_id)
        token_lemmas = list(load_csv(entry.s3_token_lemmas_key))
        token_tags = list(load_csv(entry.s3_token_tags_key))
        return cls(token_lemmas, token_tags)

    def collect_data(self):
        count = 0
        for sent_lemmas, sent_tags in zip(self.token_lemmas, self.token_tags):
            for lemma, tag in zip(sent_lemmas, sent_tags):
                self.lemma_iloc[lemma].add(count)
                count += 1

                if tag[0] in ("N", "V", "J") or "RB" in tag:
                    self.lemma_tags[lemma].add(tag[0])

        logger.info(
            "POS collect_data complete: %d tokens, %d unique lemmas",
            count,
            len(self.lemma_tags),
        )

    def get_terms(self):
        return set(self.lemma_tags)


def update_book_term_table(book_id, terms, book_embeddings, raw_pos_data_obj):
    logger.info("%s: updating BookTermTable for %d terms", book_id, len(terms))
    term_table = get_book_term_table()

    items = [
        {
            "term": term,
            "book_id": book_id,
            "count_": book_embeddings.get_count(term),
            "vector": book_embeddings.get_vector(term),
            "ilocs": raw_pos_data_obj.lemma_iloc[term],
            "tags": raw_pos_data_obj.lemma_tags[term],
        }
        for term in sorted(terms)
    ]

    logger.info("%s: submitting %d items to DynamoDB batch writer", book_id, len(items))
    term_table.batch_put_entries(items)
    logger.info("%s: BookTermTable batch write complete", book_id)


def published_year(entry):
    return entry.metadata.published_year if entry.metadata else None


def save_metadata(entry, pipeline_entries):
    logger.info("%s: saving metadata from %s", entry.book_id, entry.s3_metadata_key)
    metadata = load_json(entry.s3_metadata_key)
    pipeline_entries.update_entries(
        PipelineEntry(
            book_id=entry.book_id,
            metadata=BookMetadata(
                author=";".join(metadata["author"]),
                title=metadata["title"][0],
                published_year=published_year(entry),
            ),
        )
    )


def update_term_table(book_id, terms):
    logger.info("%s: updating CorpusTermTable for %d terms", book_id, len(terms))
    corpus_term_table = get_corpus_term_table()
    for term in terms:
        corpus_term_table.add_book(term, book_id)
    logger.info("%s: CorpusTermTable update complete", book_id)


def remove_deprecated_terms(book_id, terms):
    term_table = get_book_term_table()

    existing_terms = set(
        row["term"] for row in term_table.get_entries(book_id, fields=["term"])
    )

    if not existing_terms:
        return

    deprecated_terms = existing_terms - terms

    if not deprecated_terms:
        logger.info("%s: republish — no deprecated terms", book_id)
        return

    logger.info(
        "%s: republish — removing %d deprecated terms", book_id, len(deprecated_terms)
    )
    term_table.remove_terms(book_id, deprecated_terms)
    get_corpus_term_table().remove_book_terms(book_id, deprecated_terms)


def publish_entry(entry: PipelineEntry) -> None:
    book_id = entry.book_id
    logger.info(
        "%s: starting publish, status=%s published_year=%s",
        book_id,
        entry.status,
        published_year(entry),
    )

    book_embeddings = BookEmbeddings.from_s3(entry)
    raw_pos_data_obj = RawPOSData.from_s3(entry)
    raw_pos_data_obj.collect_data()

    pos_terms = raw_pos_data_obj.get_terms()
    embedding_vocab = book_embeddings.vocab
    terms = pos_terms & embedding_vocab

    logger.info(
        "%s: %d terms after intersection (pos=%d, embedding=%d)",
        book_id,
        len(terms),
        len(pos_terms),
        len(embedding_vocab),
    )

    remove_deprecated_terms(book_id, terms)
    update_book_term_table(book_id, terms, book_embeddings, raw_pos_data_obj)
    update_term_table(book_id, terms)
    save_metadata(entry, get_pipeline_entries())
    logger.info("%s: publish complete (%d terms)", book_id, len(terms))


def publish_entries(entries: list[PipelineEntry]) -> dict:
    published = []
    failed = []
    for entry in entries:
        try:
            publish_entry(entry)
        except Exception:
            logger.exception(
                "%s failed to publish; left at %s.",
                entry.book_id,
                EntryStatus.EMBEDDINGS_CREATED,
            )
            failed.append(str(entry.book_id))
            continue

        published.append(str(entry.book_id))

    logger.info(
        "%d of %d book(s) published, %d failed.",
        len(published),
        len(entries),
        len(failed),
    )
    return {"found": len(entries), "published": len(published), "failed": failed}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    book_ids = get_pipeline_entries().get_indexes(EntryStatus.EMBEDDINGS_CREATED)
    logger.info("Publishing %d books: %s", len(book_ids), book_ids)

    publish_entries(get_entries(book_ids))
