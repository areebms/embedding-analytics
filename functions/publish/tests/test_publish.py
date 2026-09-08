"""End-to-end tests for the publish() function in main.py.

Uses moto for DynamoDB (BookTermTable, CorpusTermTable, PipelineTable) and
S3 (the embeddings archive, POS CSVs, metadata JSON).

These tests call the actual publish() function, not a reimplementation.
"""

import json

import numpy as np

from shared.s3 import upload_json
from shared.tables.pipeline_entries import EntryStatus, PipelineEntry

from conftest import (
    BOOK_SMITH,
    BOOK_RICARDO,
    SMITH_METADATA,
    SMITH_PUBLISHED_YEAR,
    SMITH_STALE_METADATA,
    SMITH_TERM_COUNTS,
    SMITH_TOKEN_LEMMAS,
    SMITH_TOKEN_TAGS,
    VECTOR_DIM,
    _seed_book,
    _upload_embeddings,
    _upload_pos_data,
)


def _run_publish(book_id=BOOK_SMITH):
    from publish_utils import publish

    publish(book_id)


def _written_terms(term_table, book_id=BOOK_SMITH):
    return {row["term"] for row in term_table.get_entries(book_id, fields=["term"])}


# ── First-time publish ────────────────────────────────────────────────


def test_publish_writes_terms_to_term_table(smith_s3_data, term_table):
    _run_publish()

    assert _written_terms(term_table) == set(SMITH_TERM_COUNTS)


def test_publish_writes_correct_counts_to_term_table(smith_s3_data, term_table):
    _run_publish()

    for term, count in SMITH_TERM_COUNTS.items():
        row = term_table.get_entry(term, BOOK_SMITH, fields=["count_"])
        assert int(row["count_"]) == count


def test_publish_writes_pos_tags_to_term_table(smith_s3_data, term_table):
    _run_publish()

    # "labour" appears as both NN and VB in the test POS data.
    labour_row = term_table.get_entry("labour", BOOK_SMITH, fields=["tags"])
    assert "N" in labour_row["tags"]
    assert "V" in labour_row["tags"]

    # "rent" appears only as NN.
    rent_row = term_table.get_entry("rent", BOOK_SMITH, fields=["tags"])
    assert rent_row["tags"] == {"N"}


def test_publish_writes_the_vector_to_term_table(smith_s3_data, term_table):
    _run_publish()

    row = term_table.get_entry("labour", BOOK_SMITH, fields=["vector"])
    vector = np.frombuffer(bytes(row["vector"]), dtype=np.float16)
    assert vector.shape == (VECTOR_DIM,)


def test_publish_writes_ilocs_to_term_table(smith_s3_data, term_table):
    _run_publish()

    for term in SMITH_TERM_COUNTS:
        row = term_table.get_entry(term, BOOK_SMITH, fields=["ilocs"])
        assert row["ilocs"], f"Expected non-empty ilocs for '{term}'"


def test_publish_populates_corpus_term_table(smith_s3_data, corpus_term_table):
    _run_publish()

    for term in SMITH_TERM_COUNTS:
        row = corpus_term_table.get_term(term)
        assert row is not None, f"Missing corpus row for '{term}'"
        assert BOOK_SMITH in row["book_ids"]


def test_publish_updates_pipeline_metadata(smith_s3_data, pipeline_entries):
    _run_publish()

    entry = pipeline_entries.get_entry(BOOK_SMITH, fields=["metadata"])
    assert entry.metadata.author == "Smith, Adam"
    assert entry.metadata.title == "The Wealth of Nations"


def test_publish_keeps_the_published_year(smith_s3_data, pipeline_entries):
    _run_publish()

    entry = pipeline_entries.get_entry(BOOK_SMITH, fields=["metadata"])
    assert entry.metadata.published_year == SMITH_PUBLISHED_YEAR


def test_publish_leaves_the_status_intact(smith_s3_data, pipeline_entries):
    _run_publish()

    entry = pipeline_entries.get_entry(BOOK_SMITH, fields=["status"])
    assert entry.status is EntryStatus.EMBEDDINGS_CREATED


def test_publish_skips_a_book_with_no_pipeline_entry(moto_dynamo, term_table):
    _run_publish()

    assert _written_terms(term_table) == set()


def test_publish_skips_a_book_with_no_embeddings(pipeline_entries, term_table):
    """A book that has not reached EMBEDDED has no embeddings to read."""
    pipeline_entries.put_entry(
        PipelineEntry(book_id=BOOK_SMITH, status=EntryStatus.TOKENIZED)
    )

    _run_publish()

    assert _written_terms(term_table) == set()


def test_publish_filters_terms_with_non_content_pos_tags(moto_dynamo, term_table):
    """Terms that only appear with non-content POS tags (e.g. DT) should be
    excluded from the term intersection and not written to BookTermTable."""
    rng = np.random.RandomState(99)

    # Include "the" in the embeddings alongside the normal terms.
    counts_with_stopword = dict(SMITH_TERM_COUNTS)
    counts_with_stopword["the"] = 500

    _upload_embeddings(BOOK_SMITH, counts_with_stopword, rng)

    entry = PipelineEntry(
        book_id=BOOK_SMITH,
        status=EntryStatus.EMBEDDINGS_CREATED,
        metadata=SMITH_STALE_METADATA,
    )

    # "the" only gets a DT tag — should be filtered out.
    token_lemmas = [["labour", "value", "rent", "the"], ["labour", "value"]]
    token_tags = [["NN", "NN", "NN", "DT"], ["VB", "NN"]]
    _upload_pos_data(entry, token_lemmas, token_tags)

    upload_json(entry.s3_metadata_key, json.dumps(SMITH_METADATA))

    from shared.tables.pipeline_entries import get_pipeline_entries

    get_pipeline_entries().put_entry(entry)

    _run_publish()

    written_terms = _written_terms(term_table)
    assert "the" not in written_terms
    assert written_terms == set(SMITH_TERM_COUNTS)


# ── Republish (prior data exists) ────────────────────────────────────


def test_republish_does_not_duplicate_book_id_in_corpus(
    smith_s3_data, corpus_term_table
):
    """Republishing the same book should not add a duplicate entry to
    the corpus book_ids set."""
    _run_publish()

    row = corpus_term_table.get_term("labour")
    assert BOOK_SMITH in row["book_ids"]
    assert len(row["book_ids"]) == 1

    # Republish the same book.
    _run_publish()

    row = corpus_term_table.get_term("labour")
    assert BOOK_SMITH in row["book_ids"]
    assert len(row["book_ids"]) == 1


def test_republish_does_not_affect_other_books(
    smith_s3_data, term_table, corpus_term_table
):
    """Seed Ricardo's data manually, then republish Smith. Ricardo's
    corpus entries should be untouched."""
    _run_publish()

    # Manually seed Ricardo (not going through publish since we don't
    # have Ricardo's S3 data).
    _seed_book(
        term_table,
        corpus_term_table,
        BOOK_RICARDO,
        [
            {"term": "labour", "count_": 20, "tags": {"N"}},
        ],
    )

    # Republish Smith.
    _run_publish()

    row = corpus_term_table.get_term("labour")
    assert BOOK_SMITH in row["book_ids"]
    assert BOOK_RICARDO in row["book_ids"]


def test_republish_drops_terms_the_re_embed_lost(
    smith_s3_data, term_table, corpus_term_table
):
    """A term that survives the first publish but is absent from the second
    embedding must leave both BookTermTable and the corpus row."""
    _run_publish()

    assert "rent" in _written_terms(term_table)

    surviving_counts = {
        term: count for term, count in SMITH_TERM_COUNTS.items() if term != "rent"
    }
    _upload_embeddings(BOOK_SMITH, surviving_counts, np.random.RandomState(7))
    _upload_pos_data(
        smith_s3_data,
        [[term for term in row if term != "rent"] for row in SMITH_TOKEN_LEMMAS],
        [SMITH_TOKEN_TAGS[0][:-1], SMITH_TOKEN_TAGS[1]],
    )

    _run_publish()

    assert _written_terms(term_table) == set(surviving_counts)
    assert term_table.get_entry("rent", BOOK_SMITH) is None
    assert corpus_term_table.get_term("rent") is None
    assert corpus_term_table.get_term("labour")["book_ids"] == {BOOK_SMITH}
    assert corpus_term_table.get_term("value")["book_ids"] == {BOOK_SMITH}
