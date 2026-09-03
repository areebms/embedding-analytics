import numpy as np
import pytest
from botocore.exceptions import ClientError

from shared.commons import BookIndex
from shared.tables.pipeline_entries import EntryStatus, PipelineEntry

from constants import VECTOR_SIZE
from conftest import (
    INDEX,
    KEPT_LEMMAS,
    SOLITARY_PASSAGES,
    SOLITARY_TERM,
    SUBJECT,
    SYNTHETIC_PASSAGES,
    TOKEN_LEMMAS,
)


@pytest.fixture
def seed_entry(pipeline_entries):
    def seed(status=EntryStatus.TOKENIZED):
        pipeline_entries.put_entry(
            PipelineEntry(book_id=INDEX, subject_ids={SUBJECT}, status=status)
        )
        return pipeline_entries

    return seed


@pytest.fixture
def seeded_entry(seed_entry):
    return seed_entry()


@pytest.fixture
def stub_upload(mocker):
    mocker.patch(
        "create_embeddings.get_embedding_data",
        return_value="embedding_data",
    )
    return mocker.patch("create_embeddings.upload_embedding_data")


def test_create_embeddings_skips_a_book_with_no_pipeline_entry(
    pipeline_entries, token_lemmas, stub_upload
):
    from create_embeddings import create_embeddings

    token_lemmas()

    assert create_embeddings(INDEX) is None
    stub_upload.assert_not_called()


def test_create_embeddings_skips_a_book_that_has_not_been_tokenized(
    seeded_entry, stub_upload
):
    from create_embeddings import create_embeddings

    assert create_embeddings(INDEX) is None
    stub_upload.assert_not_called()


def test_create_embeddings_skips_a_book_that_has_not_reached_tokenized(
    seed_entry, token_lemmas, stub_upload
):
    from create_embeddings import create_embeddings

    seed_entry(EntryStatus.STANDARDIZED)
    token_lemmas()

    assert create_embeddings(INDEX) is None
    stub_upload.assert_not_called()


def test_create_embeddings_skips_a_book_that_is_already_embedded(
    seed_entry, token_lemmas, stub_upload
):
    from create_embeddings import create_embeddings

    seed_entry(EntryStatus.EMBEDDED)
    token_lemmas()

    assert create_embeddings(INDEX) is None
    stub_upload.assert_not_called()


def test_create_embeddings_does_not_redo_the_work_of_a_finished_book(
    seeded_entry, token_lemmas, stub_upload
):
    from create_embeddings import create_embeddings

    token_lemmas()
    create_embeddings(INDEX)
    create_embeddings(INDEX)

    stub_upload.assert_called_once()


def test_create_embeddings_skips_a_book_with_too_few_terms(
    seeded_entry, token_lemmas, mocker
):
    from create_embeddings import create_embeddings

    upload = mocker.patch("create_embeddings.upload_embedding_data")
    token_lemmas()

    assert create_embeddings(INDEX) is None
    upload.assert_not_called()


def test_create_embeddings_reads_the_key_tokenize_writes(
    seeded_entry, token_lemmas, mocker, stub_upload
):
    from create_embeddings import create_embeddings

    build = mocker.patch(
        "create_embeddings.get_embedding_data",
        return_value="embedding_data",
    )
    token_lemmas()

    assert create_embeddings(INDEX) == {"book_id": INDEX}
    build.assert_called_once_with(KEPT_LEMMAS)


def test_create_embeddings_sets_the_status_the_api_reads(
    seeded_entry, token_lemmas, stub_upload, pipeline_item
):
    from create_embeddings import create_embeddings

    token_lemmas()
    create_embeddings(INDEX)

    assert pipeline_item()["status"] == EntryStatus.EMBEDDED


def test_advancing_the_status_leaves_other_columns_intact(
    seeded_entry, token_lemmas, stub_upload, pipeline_item
):
    from create_embeddings import create_embeddings

    token_lemmas()
    create_embeddings(INDEX)

    assert pipeline_item()["subject_ids"] == {str(SUBJECT)}


def test_create_embeddings_reports_a_status_write_the_guard_turned_down(
    seeded_entry, token_lemmas, stub_upload, mocker, caplog
):
    from create_embeddings import create_embeddings

    mocker.patch.object(seeded_entry, "set_status", return_value=False)
    token_lemmas()

    assert create_embeddings(INDEX) == {"book_id": INDEX}
    assert "was not advanced" in caplog.text


def test_a_skipped_book_stays_invisible_to_the_api(
    seeded_entry, stub_upload, pipeline_item
):
    from create_embeddings import create_embeddings

    create_embeddings(INDEX)

    assert pipeline_item()["status"] == EntryStatus.TOKENIZED


def test_load_passages_drops_short_and_non_alphabetic_tokens(
    seeded_entry, token_lemmas
):
    from create_embeddings import load_passages

    token_lemmas()

    assert load_passages(seeded_entry.get_entry(INDEX)) == KEPT_LEMMAS


def test_load_passages_returns_none_when_the_lemmas_are_missing(seeded_entry):
    from create_embeddings import load_passages

    assert load_passages(seeded_entry.get_entry(INDEX)) is None


def test_load_passages_reraises_an_error_that_is_not_a_missing_key(
    seeded_entry, mocker
):
    from create_embeddings import load_passages

    mocker.patch(
        "create_embeddings.load_csv",
        side_effect=ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetObject"
        ),
    )

    with pytest.raises(ClientError):
        load_passages(seeded_entry.get_entry(INDEX))


def test_get_embedding_data_returns_none_below_vector_size():
    from create_embeddings import get_embedding_data

    assert get_embedding_data(TOKEN_LEMMAS) is None


def test_get_embedding_data_returns_one_row_per_term():
    from create_embeddings import get_embedding_data

    data = get_embedding_data(SYNTHETIC_PASSAGES)

    assert data.vectors.shape == (len(data.terms), VECTOR_SIZE)
    assert data.term_counts.shape == (len(data.terms),)


def test_get_embedding_data_ships_float32_vectors_and_integer_counts():
    from create_embeddings import get_embedding_data

    data = get_embedding_data(SYNTHETIC_PASSAGES)

    assert data.vectors.dtype == np.float32
    assert data.term_counts.dtype == np.int64


def test_get_embedding_data_counts_every_occurrence_in_the_book():
    from create_embeddings import get_embedding_data

    data = get_embedding_data(SYNTHETIC_PASSAGES)
    occurrences = sum(passage.count(data.terms[0]) for passage in SYNTHETIC_PASSAGES)

    assert data.term_counts[0] == occurrences


def test_get_embedding_data_drops_a_term_that_co_occurs_with_nothing():
    from create_embeddings import get_embedding_data

    data = get_embedding_data(SOLITARY_PASSAGES)

    assert SOLITARY_TERM not in data.terms


def test_get_embedding_data_leaves_no_zero_length_vector():
    from create_embeddings import get_embedding_data

    data = get_embedding_data(SOLITARY_PASSAGES)

    assert np.linalg.norm(data.vectors, axis=1).min() > 0


def test_upload_embedding_data_writes_what_publish_reads(
    moto_dynamo, uploaded_embeddings
):
    from create_embeddings import EmbeddingData, upload_embedding_data

    data = EmbeddingData(
        ["labour", "value"],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([11, 22], dtype=np.int64),
    )

    upload_embedding_data(INDEX, data)
    uploaded = uploaded_embeddings()

    assert uploaded["terms"].tolist() == data.terms
    assert np.array_equal(uploaded["vectors"], data.vectors)
    assert np.array_equal(uploaded["attr_count"], data.term_counts)


def test_create_embeddings_uploads_a_readable_book(
    seeded_entry, token_lemmas, uploaded_embeddings
):
    from create_embeddings import create_embeddings

    token_lemmas(passages=SYNTHETIC_PASSAGES)

    assert create_embeddings(INDEX) == {"book_id": INDEX}
    assert len(uploaded_embeddings()["terms"]) > VECTOR_SIZE


def test_handler_parses_the_index_before_calling_through(mocker):
    import app

    create = mocker.patch("app.create_embeddings", return_value={"book_id": INDEX})

    app.handler({"index": "gutenberg-3300"}, None)

    called_with = create.call_args.args[0]
    assert isinstance(called_with, BookIndex)
    assert called_with.source_id == 3300


def test_handler_returns_what_create_embeddings_returned(mocker):
    import app

    mocker.patch("app.create_embeddings", return_value={"book_id": INDEX})

    assert app.handler({"index": "gutenberg-3300"}, None) == {"book_id": INDEX}


def test_handler_reports_a_skip(mocker):
    import app

    mocker.patch("app.create_embeddings", return_value=None)

    response = app.handler({"index": "gutenberg-3300"}, None)

    assert response == {"book_id": INDEX, "skipped": True}


def test_handler_rejects_a_request_with_no_index():
    import app

    with pytest.raises(ValueError, match="index is required"):
        app.handler({}, None)
