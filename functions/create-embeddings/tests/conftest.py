import csv
import io
import os
import random
import tempfile

import numpy as np
import pytest


os.environ.update(
    AWS_REGION="us-east-1",
    AWS_DEFAULT_REGION="us-east-1",
    AWS_ACCESS_KEY_ID="testing",
    AWS_SECRET_ACCESS_KEY="testing",
    AWS_SESSION_TOKEN="testing",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
)
os.environ.pop("AWS_PROFILE", None)

from shared.commons import BookIndex
from shared.tables.pipeline_entries import EntryStatus, PipelineEntry
from shared.tests_utils import (  # noqa: F401
    aws,
    bucket,
    entries,
)

from constants import MIN_COUNT, VECTOR_SIZE


INDEX = BookIndex(3300)
INDEX_2 = BookIndex(11)
SUBJECT = BookIndex(42)

TOKEN_LEMMAS = [
    ["labour", "the", "rent", "productive", "1776"],
    ["value", "of", "commodity", "wealth"],
]

KEPT_LEMMAS = [
    ["labour", "rent", "productive"],
    ["value", "commodity", "wealth"],
]

VOCABULARY = [
    f"term{chr(97 + position // 26)}{chr(97 + position % 26)}"
    for position in range(VECTOR_SIZE + 30)
]
SOLITARY_TERM = "solitary"

_rng = random.Random(0)
SYNTHETIC_PASSAGES = [
    [_rng.choice(VOCABULARY) for _ in range(24)] for _ in range(600)
]
SOLITARY_PASSAGES = SYNTHETIC_PASSAGES + [[SOLITARY_TERM]] * MIN_COUNT


@pytest.fixture
def seed(entries):
    def _seed(status=EntryStatus.TOKENIZED, index=INDEX, subject_ids={SUBJECT}):
        entries.put_entry(
            PipelineEntry(book_id=index, subject_ids=subject_ids, status=status)
        )
        return index

    return _seed


@pytest.fixture
def token_lemmas(bucket):
    def _token_lemmas(index=INDEX, passages=TOKEN_LEMMAS):
        buffer = io.StringIO()
        csv.writer(buffer).writerows(passages)
        bucket.put_object(
            Key=f"token_lemmas/{index}.csv", Body=buffer.getvalue().encode("utf-8")
        )
        return index

    return _token_lemmas


@pytest.fixture
def tokenized_book(seed, token_lemmas):
    def _tokenized_book(index=INDEX, passages=TOKEN_LEMMAS):
        seed(EntryStatus.TOKENIZED, index)
        token_lemmas(index, passages)
        return index

    return _tokenized_book


@pytest.fixture
def events_client(mocker):
    import create_embeddings

    client = mocker.Mock()
    client.put_events.return_value = {"FailedEntryCount": 0, "Entries": [{}]}
    mocker.patch.object(
        create_embeddings, "get_session"
    ).return_value.client.return_value = client
    return client


@pytest.fixture
def uploaded_embeddings(bucket):
    def read(index=INDEX):
        with tempfile.NamedTemporaryFile(suffix=".npz") as file:
            bucket.Object(f"embeddings/{index}.npz").download_file(file.name)
            with np.load(file.name, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}

    return read


def status_of(entries, index=INDEX):
    return entries.get_entry(index).status
