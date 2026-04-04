from __future__ import annotations

import os

from gensim.models import KeyedVectors

from shared.aws import (
    get_session,
    load_file_from_s3,
    get_keys_with_prefix,
)


class KeyedVectorGroup:

    def __init__(self, index):
        self.s3_prefix = f"kvectors/{index}"
        self.keyed_vectors_stack = []
        self.centroid = None
        self.label = index.split("-")[1]
        self.term_stability_data = {}

    def fetch_keyed_vectors_stack(self, session):
        model_keys = get_keys_with_prefix(session, f"{self.s3_prefix}/aligned/")
        for key in sorted(model_keys):
            self.keyed_vectors_stack.append(self.load_keyed_vector_from_s3(key))

    @staticmethod
    def load_keyed_vector_from_s3(session, s3_key):
        local_path = load_file_from_s3(session, s3_key)
        try:
            return KeyedVectors.load(local_path)
        finally:
            os.unlink(local_path)

    def fetch_centroid_data(self, session):
        self.centroid = self.load_keyed_vector_from_s3(
            session, f"{self.s3_prefix}/centroid.model"
        )

    def fetch_precalculated_data(self):
        session = get_session()
        self.fetch_centroid_data(session)
