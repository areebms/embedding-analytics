import json
import logging

import numpy as np
import pytest
from botocore.exceptions import ClientError

import app
import create_embeddings as create_embeddings_module
from create_embeddings import (
    EmbeddingData,
    create_embeddings,
    get_embedding_data,
    get_entries,
    load_passages,
    resolve_subject,
    set_status,
    upload_embedding_data,
)
from shared.commons import BookIndex
from shared.tables.pipeline_entries import EntryStatus, PipelineEntry

from constants import VECTOR_SIZE
from conftest import (
    INDEX,
    INDEX_2,
    KEPT_LEMMAS,
    SOLITARY_PASSAGES,
    SOLITARY_TERM,
    SUBJECT,
    SYNTHETIC_PASSAGES,
    TOKEN_LEMMAS,
    status_of,
)


def test_a_named_book_is_embedded_and_advanced(tokenized_book, entries):
    index = tokenized_book(passages=SYNTHETIC_PASSAGES)

    status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {"found": 1, "embedded": 1}
    assert status_of(entries, index) == EntryStatus.EMBEDDINGS_CREATED


def test_the_uploaded_file_is_row_aligned_across_terms_vectors_and_counts(
    tokenized_book, uploaded_embeddings
):
    index = tokenized_book(passages=SYNTHETIC_PASSAGES)

    app.handler({"book_ids": [str(index)]}, None)
    uploaded = uploaded_embeddings(index)

    assert len(uploaded["terms"]) > VECTOR_SIZE
    assert uploaded["vectors"].shape == (len(uploaded["terms"]), VECTOR_SIZE)
    assert uploaded["attr_count"].shape == (len(uploaded["terms"]),)


def test_advancing_the_status_leaves_the_other_columns_intact(
    tokenized_book, entries
):
    index = tokenized_book(passages=SYNTHETIC_PASSAGES)

    app.handler({"book_ids": [str(index)]}, None)

    assert entries.get_entry(index).subject_ids == {SUBJECT}


def test_a_book_is_read_out_of_a_json_body(tokenized_book, entries):
    index = tokenized_book(passages=SYNTHETIC_PASSAGES)

    status = app.handler({"body": json.dumps({"book_ids": [str(index)]})}, None)

    assert status == {"found": 1, "embedded": 1}
    assert status_of(entries, index) == EntryStatus.EMBEDDINGS_CREATED


def test_a_subject_is_resolved_to_the_books_standing_at_tokenized(
    tokenized_book, seed, entries
):
    tokenized_book(INDEX, SYNTHETIC_PASSAGES)
    tokenized_book(INDEX_2, SYNTHETIC_PASSAGES)
    seed(EntryStatus.STANDARDIZED, BookIndex(999))

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status == {"found": 2, "embedded": 2}
    assert status_of(entries, INDEX) == EntryStatus.EMBEDDINGS_CREATED
    assert status_of(entries, INDEX_2) == EntryStatus.EMBEDDINGS_CREATED


def test_a_subject_is_capped_and_the_overflow_is_left_for_the_next_run(
    seed, monkeypatch
):
    monkeypatch.setattr("create_embeddings.MAX_BOOKS_PER_SUBJECT", 2)
    indexes = [BookIndex(source_id) for source_id in range(1, 6)]
    for index in indexes:
        seed(EntryStatus.TOKENIZED, index)

    assert resolve_subject(str(SUBJECT)) == sorted(indexes)[:2]


def test_running_the_same_book_twice_does_not_redo_the_work(
    tokenized_book, entries
):
    index = tokenized_book(passages=SYNTHETIC_PASSAGES)

    app.handler({"book_ids": [str(index)]}, None)
    status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {"found": 0, "embedded": 0}
    assert status_of(entries, index) == EntryStatus.EMBEDDINGS_CREATED


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"book_ids": [], "subject_id": ""},
        {"book_ids": ["gutenberg-3300"], "subject_id": "gutenberg-42"},
    ],
    ids=["neither", "empty", "both"],
)
def test_naming_no_books_or_two_ways_at_once_is_refused(event, mocker):
    embed = mocker.patch("app.create_embeddings")

    with pytest.raises(ValueError, match="Exactly one"):
        app.handler(event, None)

    embed.assert_not_called()


def test_a_book_past_tokenized_is_dropped_before_the_decomposition(
    seed, token_lemmas, bucket, entries
):
    seed(EntryStatus.EMBEDDINGS_CREATED)
    token_lemmas(passages=SYNTHETIC_PASSAGES)

    status = app.handler({"book_ids": [str(INDEX)]}, None)

    assert status == {"found": 0, "embedded": 0}
    assert list(bucket.objects.filter(Prefix="embeddings/")) == []
    assert status_of(entries, INDEX) == EntryStatus.EMBEDDINGS_CREATED


def test_a_book_with_no_pipeline_entry_is_reported_and_skipped(aws, caplog):
    with caplog.at_level(logging.WARNING, logger="create_embeddings"):
        status = app.handler({"book_ids": ["gutenberg-404"]}, None)

    assert status == {"found": 0, "embedded": 0}
    assert "1 of 1 book(s) have no pipeline entry." in caplog.text


def test_a_book_whose_lemmas_are_missing_is_skipped_rather_than_failed(
    seed, entries, caplog
):
    seed(EntryStatus.TOKENIZED)

    with caplog.at_level(logging.WARNING, logger="create_embeddings"):
        status = app.handler({"book_ids": [str(INDEX)]}, None)

    assert status == {"found": 1, "embedded": 0}
    assert status_of(entries, INDEX) == EntryStatus.TOKENIZED
    assert "has not been tokenized" in caplog.text


def test_a_book_with_too_few_terms_is_parked_at_the_terminal_status(
    tokenized_book, bucket, entries, caplog
):
    index = tokenized_book()

    with caplog.at_level(logging.WARNING, logger="create_embeddings"):
        status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {"found": 1, "embedded": 0}
    assert list(bucket.objects.filter(Prefix="embeddings/")) == []
    assert status_of(entries, index) == EntryStatus.EMBEDDINGS_CREATION_FAILED
    assert "too few terms" in caplog.text


def test_a_programming_error_ends_the_run(tokenized_book, entries, monkeypatch):
    tokenized_book(INDEX, SYNTHETIC_PASSAGES)
    tokenized_book(INDEX_2, SYNTHETIC_PASSAGES)
    real_upload = upload_embedding_data

    def mistyped_upload(entry, embedding_data):
        if entry.book_id == INDEX:
            raise TypeError("upload_embedding_data() takes an entry")
        real_upload(entry, embedding_data)

    monkeypatch.setattr("create_embeddings.upload_embedding_data", mistyped_upload)

    with pytest.raises(TypeError):
        app.handler({"book_ids": [str(INDEX), str(INDEX_2)]}, None)

    assert status_of(entries, INDEX_2) == EntryStatus.EMBEDDINGS_CREATED
    assert status_of(entries, INDEX) == EntryStatus.TOKENIZED


def test_an_empty_run_is_reported_without_touching_the_bucket(aws, bucket):
    assert create_embeddings([]) == {"found": 0, "embedded": 0}
    assert list(bucket.objects.filter(Prefix="embeddings/")) == []


def test_get_entries_keeps_only_the_books_standing_at_tokenized(seed):
    seed(EntryStatus.TOKENIZED, INDEX)
    seed(EntryStatus.STANDARDIZED, INDEX_2)

    assert [entry.book_id for entry in get_entries([INDEX, INDEX_2])] == [INDEX]


def test_a_status_write_the_guard_turns_down_is_reported(seed, caplog):
    seed(EntryStatus.EMBEDDINGS_CREATED)

    with caplog.at_level(logging.WARNING, logger="create_embeddings"):
        set_status(INDEX, EntryStatus.TOKENIZED)

    assert "the status guard refused the write" in caplog.text


def test_load_passages_drops_short_and_non_alphabetic_tokens(seed, token_lemmas):
    seed(EntryStatus.TOKENIZED)
    token_lemmas()

    assert load_passages(get_entries([INDEX])[0]) == KEPT_LEMMAS


def test_load_passages_returns_none_when_the_lemmas_are_missing(seed):
    seed(EntryStatus.TOKENIZED)

    assert load_passages(get_entries([INDEX])[0]) is None


def test_load_passages_reraises_an_error_that_is_not_a_missing_key(seed, mocker):
    seed(EntryStatus.TOKENIZED)
    mocker.patch(
        "create_embeddings.load_csv",
        side_effect=ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject"),
    )

    with pytest.raises(ClientError):
        load_passages(get_entries([INDEX])[0])


def test_get_embedding_data_returns_none_below_vector_size():
    assert get_embedding_data(TOKEN_LEMMAS) is None


def test_get_embedding_data_returns_one_row_per_term():
    data = get_embedding_data(SYNTHETIC_PASSAGES)

    assert data.vectors.shape == (len(data.terms), VECTOR_SIZE)
    assert data.term_counts.shape == (len(data.terms),)


def test_get_embedding_data_ships_float32_vectors_and_integer_counts():
    data = get_embedding_data(SYNTHETIC_PASSAGES)

    assert data.vectors.dtype == np.float32
    assert data.term_counts.dtype == np.int64


def test_get_embedding_data_counts_every_occurrence_in_the_book():
    data = get_embedding_data(SYNTHETIC_PASSAGES)
    occurrences = sum(passage.count(data.terms[0]) for passage in SYNTHETIC_PASSAGES)

    assert data.term_counts[0] == occurrences


def test_get_embedding_data_drops_a_term_that_co_occurs_with_nothing(caplog):
    with caplog.at_level(logging.INFO, logger="create_embeddings"):
        data = get_embedding_data(SOLITARY_PASSAGES)

    assert SOLITARY_TERM not in data.terms
    assert "co-occur with nothing" in caplog.text


def test_get_embedding_data_leaves_no_zero_length_vector():
    data = get_embedding_data(SOLITARY_PASSAGES)

    assert np.linalg.norm(data.vectors, axis=1).min() > 0


def test_upload_embedding_data_writes_the_terms_vectors_and_counts(
    aws, uploaded_embeddings
):
    data = EmbeddingData(
        ["labour", "value"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([11, 22], dtype=np.int64),
    )

    upload_embedding_data(PipelineEntry(book_id=INDEX), data)
    uploaded = uploaded_embeddings()

    assert uploaded["terms"].tolist() == data.terms
    assert np.array_equal(uploaded["vectors"], data.vectors)
    assert np.array_equal(uploaded["attr_count"], data.term_counts)


def test_the_books_that_reached_embeddings_created_are_announced(
    tokenized_book, events_client
):
    tokenized_book(INDEX, SYNTHETIC_PASSAGES)
    tokenized_book(INDEX_2, SYNTHETIC_PASSAGES)

    app.handler({"book_ids": [str(INDEX), str(INDEX_2)]}, None)

    (entry,) = events_client.put_events.call_args.kwargs["Entries"]

    assert entry["Source"] == create_embeddings_module.ANNOUNCE_SOURCE
    assert entry["DetailType"] == create_embeddings_module.ANNOUNCE_DETAIL_TYPE

    detail = json.loads(entry["Detail"])

    assert sorted(detail["book_ids"]) == sorted([str(INDEX), str(INDEX_2)])
    assert detail["embedded"] == 2


def test_a_run_that_embedded_nothing_announces_nothing(seed, events_client):
    seed(EntryStatus.EMBEDDINGS_CREATED)

    status = app.handler({"book_ids": [str(INDEX)]}, None)

    assert status == {"found": 0, "embedded": 0}
    events_client.put_events.assert_not_called()


def test_an_announcement_the_bus_rejects_ends_the_run(
    tokenized_book, events_client, entries
):
    index = tokenized_book(passages=SYNTHETIC_PASSAGES)
    events_client.put_events.return_value = {
        "FailedEntryCount": 1,
        "Entries": [{"ErrorCode": "ThrottlingException"}],
    }

    with pytest.raises(RuntimeError, match="rejected"):
        app.handler({"book_ids": [str(index)]}, None)

    assert status_of(entries, index) == EntryStatus.EMBEDDINGS_CREATED
