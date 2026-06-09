import csv
import io
import json
import logging
from collections import defaultdict
from decimal import Decimal

import numpy as np
from gensim.models import KeyedVectors

from shared.s3 import get_s3_loader
from shared.tables.pipeline import get_pipeline_table
from shared.tables.book_terms import get_book_term_table
from shared.tables.corpus_terms import get_corpus_term_table
from shared.tables.vectors import get_pinecone_table

logger = logging.getLogger(__name__)


class BookCentroidData:

    alignment_quality_attr = ["variance", "disparity", "r_squared"]

    def __init__(self, centroid):
        self.centroid_kvector = centroid

    @classmethod
    def from_s3(cls, loader, index):
        logger.info("%s: loading centroid model", index)
        with loader.load_file(f"kvectors/{index}/aligned/centroid.model") as (
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

    def get_pinecone_vector(self, term) -> np.ndarray:
        return np.array(self.centroid_kvector[term], dtype=np.float32)

    def get_dynamodb_vector(self, term):
        return np.array(self.centroid_kvector[term], dtype=np.float16).tobytes()

    def get_count(self, term) -> int:
        return int(self.centroid_kvector.get_vecattr(term, "count"))


class RawPOSData:

    def __init__(self, token_lemmas, token_tags):
        self.token_lemmas = token_lemmas
        self.token_tags = token_tags

        self.lemma_iloc = defaultdict(set)
        self.lemma_tags = defaultdict(set)

    @classmethod
    def from_s3(cls, loader, index):
        table = get_pipeline_table()
        item = table.get_entry(index, ["s3_token_lemmas_key", "s3_token_tags_key"])

        logger.info("%s: loading POS data", index)
        token_lemmas = list(
            csv.reader(io.StringIO(loader.load_text(item["s3_token_lemmas_key"])))
        )
        token_tags = list(
            csv.reader(io.StringIO(loader.load_text(item["s3_token_tags_key"])))
        )
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
    def from_s3(cls, loader, index):
        s3_keys = []
        kvectors = []
        for key, local_path in loader.yield_s3_files(
            f"kvectors/{index}/aligned/", ".model"
        ):
            if key.endswith("/centroid.model"):
                continue
            s3_keys.append(key)
            kvectors.append(KeyedVectors.load(local_path))

        logger.info("%s: loaded %d seed models", index, len(kvectors))
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


def update_pinecone_table(index, terms, centroid_data_obj, raw_pos_data_obj):
    logger.info("%s: upserting %d vectors to Pinecone", index, len(terms))
    pinecone = get_pinecone_table()

    pinecone_data = []
    for term in terms:
        pinecone_data.append(
            {
                "book_id": index,
                "term": term,
                "vector": centroid_data_obj.get_pinecone_vector(term),
                "pos": sorted(raw_pos_data_obj.lemma_tags[term]),
                "count": centroid_data_obj.get_count(term),
            }
        )

    pinecone.batch_upsert(pinecone_data)
    logger.info("%s: Pinecone upsert complete", index)


def update_book_term_table(
    index, terms, s3_loader, centroid_data_obj, raw_pos_data_obj
):
    logger.info("%s: updating BookTermTable for %d terms", index, len(terms))
    raw_vector_stack_data = RawKVectorStack.from_s3(s3_loader, index).collect_data()
    term_table = get_book_term_table()

    items = []
    for term in sorted(terms):
        seeds = sorted(raw_vector_stack_data[term].keys())
        items.append(
            {
                "term": term,
                "platform_data": index,
                "alignment_stats": centroid_data_obj.get_alignment_stats(term),
                "count_": centroid_data_obj.get_count(term),
                "ilocs": raw_pos_data_obj.lemma_iloc[term],
                "tags": raw_pos_data_obj.lemma_tags[term],
                "seeds": seeds,
                "vectors": [raw_vector_stack_data[term][seed] for seed in seeds],
            }
        )

    logger.info("%s: submitting %d items to DynamoDB batch writer", index, len(items))
    term_table.batch_put_entries(items)
    logger.info("%s: BookTermTable batch write complete", index)


def save_metadata(index, s3_loader, s3_metadata_key, pipeline_table):
    logger.info("%s: saving metadata from %s", index, s3_metadata_key)
    metadata = json.loads(s3_loader.load_text(s3_metadata_key))
    pipeline_table.update_entries(
        index,
        {"author": ";".join(metadata["author"]), "title": metadata["title"][0]},
    )


def update_term_table(index, terms):
    logger.info("%s: updating CorpusTermTable for %d terms", index, len(terms))
    corpus_term_table = get_corpus_term_table()
    for term in terms:
        corpus_term_table.add_book(term, index)
    logger.info("%s: CorpusTermTable update complete", index)


def remove_deprecated_terms(index, terms):
    term_table = get_book_term_table()

    existing_terms = set(
        row["term"] for row in term_table.get_entries(index, fields=["term"])
    )

    if not existing_terms:
        return

    deprecated_terms = existing_terms - terms

    if not deprecated_terms:
        logger.info("%s: republish — no deprecated terms", index)
        return

    logger.info(
        "%s: republish — removing %d deprecated terms", index, len(deprecated_terms)
    )
    corpus_term_table = get_corpus_term_table()
    pinecone_table = get_pinecone_table()
    corpus_term_table.remove_book_terms(index, deprecated_terms)
    pinecone_table.delete_book_terms(index, deprecated_terms)


def publish(index):
    logger.info("%s: starting publish", index)
    s3_loader = get_s3_loader()
    pipeline_table = get_pipeline_table()

    item = pipeline_table.get_entry(index, ["s3_metadata_key", "published_year"])
    s3_metadata_key = item.get("s3_metadata_key")
    if not s3_metadata_key:
        logger.warning("%s: has not been scraped", index)
        return

    logger.info(
        "%s: s3_metadata_key=%s published_year=%s",
        index,
        s3_metadata_key,
        item.get("published_year"),
    )

    centroid_data_obj = BookCentroidData.from_s3(s3_loader, index)
    raw_pos_data_obj = RawPOSData.from_s3(s3_loader, index)
    raw_pos_data_obj.collect_data()

    pos_terms = raw_pos_data_obj.get_terms()
    centroid_vocab = centroid_data_obj.vocab
    terms = pos_terms & centroid_vocab

    logger.info(
        "%s: %d terms after intersection (pos=%d, centroid=%d)",
        index,
        len(terms),
        len(pos_terms),
        len(centroid_vocab),
    )

    remove_deprecated_terms(index, terms)
    update_book_term_table(index, terms, s3_loader, centroid_data_obj, raw_pos_data_obj)
    update_pinecone_table(index, terms, centroid_data_obj, raw_pos_data_obj)
    update_term_table(index, terms)
    save_metadata(index, s3_loader, s3_metadata_key, pipeline_table)
    logger.info("%s: publish complete (%d terms)", index, len(terms))
