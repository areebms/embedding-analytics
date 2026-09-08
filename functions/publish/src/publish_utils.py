import logging
from collections import defaultdict

import numpy as np
from gensim.models import KeyedVectors

from shared.commons import BookIndex
from shared.s3 import load_csv, load_file, load_json
from shared.tables.book_terms import get_book_term_table
from shared.tables.corpus_terms import get_corpus_term_table
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

logger = logging.getLogger(__name__)


class BookCentroidData:

    def __init__(self, centroid):
        self.centroid_kvector = centroid

    @classmethod
    def from_s3(cls, entry: PipelineEntry):
        logger.info("%s: loading centroid model", entry.book_id)
        with load_file(f"kvectors/{entry.book_id}/aligned/centroid.model") as (
            _,
            local_path,
        ):
            return cls(KeyedVectors.load(local_path))

    @property
    def vocab(self):
        return set(self.centroid_kvector.key_to_index)

    def get_vector(self, term) -> bytes:
        return self.centroid_kvector[term].astype(np.float16).tobytes()

    def get_count(self, term) -> int:
        return int(self.centroid_kvector.get_vecattr(term, "count"))


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


def update_book_term_table(book_id, terms, centroid_data_obj, raw_pos_data_obj):
    logger.info("%s: updating BookTermTable for %d terms", book_id, len(terms))
    term_table = get_book_term_table()

    items = [
        {
            "term": term,
            "book_id": book_id,
            "count_": centroid_data_obj.get_count(term),
            "vector": centroid_data_obj.get_vector(term),
            "ilocs": raw_pos_data_obj.lemma_iloc[term],
            "tags": raw_pos_data_obj.lemma_tags[term],
        }
        for term in sorted(terms)
    ]

    logger.info("%s: submitting %d items to DynamoDB batch writer", book_id, len(items))
    term_table.batch_put_entries(items)
    logger.info("%s: BookTermTable batch write complete", book_id)


def save_metadata(entry, pipeline_entries):
    logger.info("%s: saving metadata from %s", entry.book_id, entry.s3_metadata_key)
    metadata = load_json(entry.s3_metadata_key)
    pipeline_entries.update_entries(
        PipelineEntry(
            book_id=entry.book_id,
            author=";".join(metadata["author"]),
            title=metadata["title"][0],
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
    get_corpus_term_table().remove_book_terms(book_id, deprecated_terms)


def publish(book_id: BookIndex):
    logger.info("%s: starting publish", book_id)
    pipeline_entries = get_pipeline_entries()

    try:
        entry = pipeline_entries.get_entry(book_id, ["status", "published_year"])
    except LookupError:
        logger.warning("%s: has no pipeline entry", book_id)
        return

    if entry.status is None or entry.status < EntryStatus.EMBEDDINGS_CREATED:
        logger.warning("%s: has no embeddings", book_id)
        return

    logger.info(
        "%s: status=%s published_year=%s", book_id, entry.status, entry.published_year
    )

    centroid_data_obj = BookCentroidData.from_s3(entry)
    raw_pos_data_obj = RawPOSData.from_s3(entry)
    raw_pos_data_obj.collect_data()

    pos_terms = raw_pos_data_obj.get_terms()
    centroid_vocab = centroid_data_obj.vocab
    terms = pos_terms & centroid_vocab

    logger.info(
        "%s: %d terms after intersection (pos=%d, centroid=%d)",
        book_id,
        len(terms),
        len(pos_terms),
        len(centroid_vocab),
    )

    remove_deprecated_terms(book_id, terms)
    update_book_term_table(book_id, terms, centroid_data_obj, raw_pos_data_obj)
    update_term_table(book_id, terms)
    save_metadata(entry, pipeline_entries)
    logger.info("%s: publish complete (%d terms)", book_id, len(terms))
