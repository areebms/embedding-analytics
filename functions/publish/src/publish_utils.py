import logging
from collections import defaultdict
from decimal import Decimal

import numpy as np
from gensim.models import KeyedVectors

from shared.commons import BookIndex
from shared.s3 import load_csv, load_file, load_json, yield_s3_files
from shared.tables.book_terms import get_book_term_table
from shared.tables.corpus_terms import get_corpus_term_table
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

logger = logging.getLogger(__name__)


class BookCentroidData:

    alignment_quality_attr = ["variance", "disparity", "r_squared"]

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

    def get_alignment_stats(self, term):
        return {
            attr: Decimal(str(self.centroid_kvector.get_vecattr(term, attr)))
            for attr in self.alignment_quality_attr
        }

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


class RawKVectorStack:

    def __init__(self, s3_keys, kvectors):
        self.s3_keys = s3_keys
        self.kvectors = kvectors

        self.term_vectors = defaultdict(dict)

    @staticmethod
    def get_seed(model_s3_key):
        return int(model_s3_key.split("/")[-1].split("-")[0])

    @classmethod
    def from_s3(cls, book_id):
        s3_keys = []
        kvectors = []
        for key, local_path in yield_s3_files(
            f"kvectors/{book_id}/aligned/", ".model"
        ):
            if key.endswith("/centroid.model"):
                continue
            s3_keys.append(key)
            kvectors.append(KeyedVectors.load(local_path))

        logger.info("%s: loaded %d seed models", book_id, len(kvectors))
        return cls(s3_keys, kvectors)

    def collect_data(self):
        if not self.kvectors:
            logger.warning("no seed kvectors found; skipping term vector collection")
            return self.term_vectors
        for term in self.kvectors[0].key_to_index:
            for s3_key, kvector in zip(self.s3_keys, self.kvectors):
                self.term_vectors[term][self.get_seed(s3_key)] = (
                    kvector[term].astype(np.float16).tobytes()
                )

        logger.info(
            "kvector collect_data complete: %d terms across %d seeds",
            len(self.term_vectors),
            len(self.kvectors),
        )
        return self.term_vectors


def update_book_term_table(book_id, terms, centroid_data_obj, raw_pos_data_obj):
    logger.info("%s: updating BookTermTable for %d terms", book_id, len(terms))
    raw_vector_stack_data = RawKVectorStack.from_s3(book_id).collect_data()
    term_table = get_book_term_table()

    items = []
    for term in sorted(terms):
        seeds = sorted(raw_vector_stack_data[term].keys())
        items.append(
            {
                "term": term,
                "book_id": book_id,
                "alignment_stats": centroid_data_obj.get_alignment_stats(term),
                "count_": centroid_data_obj.get_count(term),
                "ilocs": raw_pos_data_obj.lemma_iloc[term],
                "tags": raw_pos_data_obj.lemma_tags[term],
                "seeds": seeds,
                "vectors": [raw_vector_stack_data[term][seed] for seed in seeds],
            }
        )

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
