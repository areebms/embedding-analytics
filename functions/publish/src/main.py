import csv
import io
import json
from collections import defaultdict
from decimal import Decimal

import numpy as np
from gensim.models import KeyedVectors


from shared.aws import (
    get_pipeline_table,
    TermTable,
    get_session,
    load_text_from_s3,
    S3Loader,
)

from shared.commons import get_index


def load_kvector(loader, s3_key):
    with loader.load_file(s3_key) as (_, local_path):
        return KeyedVectors.load(local_path)


def to_table_format(vector):
    return vector.astype(np.float16).tobytes()


class CentroidData:
    def __init__(self, centroid):
        self.centroid = centroid
        self.data = defaultdict(dict)

    @classmethod
    def from_s3(cls, loader, index):
        centroid = load_kvector(loader, f"kvectors/{index}/centroid.model")
        return cls(centroid)

    def collect_data(self):
        for term in self.centroid.key_to_index:
            self.data[term]["centroid_vector"] = to_table_format(self.centroid[term])
            self.data[term]["count_"] = int(self.centroid.get_vecattr(term, "count"))
            self.data[term]["variance_"] = Decimal(
                str(self.centroid.get_vecattr(term, "variance"))
            )
            self.data[term]["disparity"] = Decimal(
                str(self.centroid.get_vecattr(term, "disparity"))
            )
            self.data[term]["r_squared"] = Decimal(
                str(self.centroid.get_vecattr(term, "r_squared"))
            )

        return self.data


class RawPOSData:

    def __init__(self, token_lemmas, token_tags):
        self.token_lemmas = token_lemmas
        self.token_tags = token_tags

        self.lemma_iloc = defaultdict(set)
        self.lemma_tags = defaultdict(set)

    @classmethod
    def from_s3(cls, session, index):
        table = get_pipeline_table()
        item = table.get_entry(index, ["s3_token_lemmas_key", "s3_token_tags_key"])

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
        for sent_lemmas, sent_tags in zip(self.token_lemmas, self.token_tags):
            for lemma, tag in zip(sent_lemmas, sent_tags):
                self.lemma_iloc[lemma].add(count)
                count += 1

                if tag[0] in ("N", "V", "J") or "RB" in tag:
                    self.lemma_tags[lemma].add(tag[0])

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
        for key, local_path in loader.yield_s3_files(f"kvectors/{index}/aligned/", ".model"):
            s3_keys.append(key)
            kvectors.append(KeyedVectors.load(local_path))

        return cls(s3_keys, kvectors)

    def collect_data(self):
        for term in self.kvectors[0].key_to_index:
            for s3_key, kvector in zip(self.s3_keys, self.kvectors):
                self.term_vectors[term][self.get_seed(s3_key)] = to_table_format(kvector[term])

        return self.term_vectors


def publish(index):
    session = get_session()
    loader = S3Loader(session)
    pipeline_table = get_pipeline_table()
    term_table = TermTable(session)

    item = pipeline_table.get_entry(index, ["s3_metadata_key"])
    s3_metadata_key = item.get("s3_metadata_key")
    if not s3_metadata_key:
        print(f"{index} has not been scraped.")
        return

    centroid_data = CentroidData.from_s3(loader, index).collect_data()
    raw_pos_data_obj = RawPOSData.from_s3(session, index)
    raw_pos_data_obj.collect_data()
    raw_vector_stack_data = RawKVectorStack.from_s3(loader, index).collect_data()

    terms = raw_pos_data_obj.get_terms() & set(centroid_data) & set(raw_vector_stack_data)

    for term in terms:
        seeds = sorted(raw_vector_stack_data[term].keys())
        term_table.update_entries(
            term,
            index,
            {
                **centroid_data[term],
                "ilocs": raw_pos_data_obj.lemma_iloc[term],
                "tags": raw_pos_data_obj.lemma_tags[term],
                "seeds": seeds,
                "vectors": [raw_vector_stack_data[term][seed] for seed in seeds],
            },
        )

    metadata = json.loads(load_text_from_s3(session, s3_metadata_key))

    pipeline_table.update_entries(
        index,
        {
            "author": ";".join(metadata["author"]),
            "title": metadata["title"][0],
        },
    )


if __name__ == "__main__":
    publish(get_index())
