import csv
import os
import io
import json
from collections import defaultdict
from decimal import Decimal

import numpy as np
from gensim.models import KeyedVectors


from shared.aws import (
    PipelineTable,
    TermTable,
    get_session,
    load_file_from_s3,
    load_text_from_s3,
    yield_keys_with_prefix,
)


S3_BUCKET = os.getenv("S3_BUCKET")


def load_kvector_from_s3(session, s3_key):
    local_path = load_file_from_s3(session, s3_key)
    try:
        return KeyedVectors.load(local_path)
    finally:
        os.unlink(local_path)


def to_table_format(vector):
    return vector.astype(np.float16).tobytes()


class RawCentroidData:
    def __init__(self, centroid):
        self.centroid = centroid
        self.data = defaultdict(dict)

    @classmethod
    def from_s3(cls, session, index):
        centroid = load_kvector_from_s3(session, f"kvectors/{index}/centroid.model")
        return cls(centroid)

    def collect_data(self):
        for term in list(self.centroid.key_to_index):
            self.data[term]["centroid_vector"] = to_table_format(self.centroid[term])
            self.data[term]["count_"] = int(self.centroid.get_vecattr(term, "count"))
            self.data[term]["variance_"] = Decimal(str(self.centroid.get_vecattr(term, "variance")))
            self.data[term]["disparity"] = Decimal(str(self.centroid.get_vecattr(term, "disparity")))
            self.data[term]["r_squared"] = Decimal(str(self.centroid.get_vecattr(term, "r_squared")))

        return self.data


class RawPOSData:

    def __init__(self, token_lemmas, token_tags):
        self.token_lemmas = token_lemmas
        self.token_tags = token_tags

        self.lemma_iloc = defaultdict(set)
        self.lemma_tags = defaultdict(set)

    @classmethod
    def from_s3(cls, session, index):
        table = PipelineTable(session)
        item = table.get_entry(index, expression="s3_token_lemmas_key,s3_token_tags_key")

        token_lemmas = list(
            csv.reader(
                io.StringIO(load_text_from_s3(session, item.get("s3_token_lemmas_key")))
            )
        )
        token_tags = list(
            csv.reader(
                io.StringIO(load_text_from_s3(session, item.get("s3_token_tags_key")))
            )
        )
        return cls(token_lemmas, token_tags)

    def collect_data(self):
        count = 0
        for i in range(len(self.token_lemmas)):
            for j in range(len(self.token_lemmas[i])):
                self.lemma_iloc[self.token_lemmas[i][j]].add(count)
                count += 1

                if (
                    self.token_tags[i][j][0] in ["N", "V", "J"]
                    or "RB" in self.token_tags[i][j]
                ):
                    self.lemma_tags[self.token_lemmas[i][j]].add(
                        self.token_tags[i][j][0]
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
    def from_s3(cls, session, index):
        s3_keys = []
        kvectors = []
        for key in yield_keys_with_prefix(session, f"kvectors/{index}/aligned/"):
            if not key.endswith(".model"):
                continue
            s3_keys.append(key)
            kvectors.append(load_kvector_from_s3(session, key))

        return cls(s3_keys, kvectors)

    def collect_data(self):
        for term in list(self.kvectors[0].key_to_index):
            for i in range(len(self.s3_keys)):
                self.term_vectors[term][self.get_seed(self.s3_keys[i])] = (
                    to_table_format(self.kvectors[i][term])
                )

        return self.term_vectors


def save_term_data(index):

    session = get_session()

    raw_centroid_data = RawCentroidData.from_s3(session, index).collect_data()
    raw_pos_data_obj = RawPOSData.from_s3(session, index)
    raw_pos_data_obj.collect_data()
    raw_vector_stack_data = RawKVectorStack.from_s3(session, index).collect_data()

    terms = raw_pos_data_obj.get_terms() & set(raw_centroid_data)

    table = TermTable(session)

    for term in terms:
        seeds = sorted(raw_vector_stack_data[term].keys())
        table.update_entries(term, index, {
            **raw_centroid_data[term],
            "ilocs": raw_pos_data_obj.lemma_iloc[term],
            "tags": raw_pos_data_obj.lemma_tags[term],
            "seeds": seeds,
            "vectors": [raw_vector_stack_data[term][seed] for seed in seeds],
        })


def publish(table, session, index):
    item = table.get_entry(index, expression="s3_metadata_key")
    s3_metadata_key = item.get("s3_metadata_key")
    if not s3_metadata_key:
        print(f"{index} has not been scraped.")
        return

    metadata = json.loads(load_text_from_s3(session, s3_metadata_key))

    table.update_entries(
        index,
        {
            "author": ";".join(metadata["author"]),
            "title": metadata["title"][0],
        },
    )
