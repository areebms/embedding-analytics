"""End-to-end tests for the publish() function in main.py.

Uses moto for DynamoDB (BookTermTable, CorpusTermTable, PipelineTable) and
S3 (centroid models, seed models, POS CSVs, metadata JSON).

These tests call the actual publish() function, not a reimplementation.
"""

import json
from decimal import Decimal

import numpy as np

from shared.s3 import upload_json
from shared.tables.pipeline_entries import EntryStatus, PipelineEntry

from conftest import (
    BOOK_SMITH,
    BOOK_RICARDO,
    SMITH_METADATA,
    SMITH_TERM_ATTRS,
    VECTOR_DIM,
    NUM_SEEDS,
    _create_keyed_vectors,
    _save_keyed_vectors_to_s3,
    _seed_book,
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

    assert _written_terms(term_table) == set(SMITH_TERM_ATTRS)


def test_publish_writes_correct_counts_to_term_table(smith_s3_data, term_table):
    _run_publish()

    for term, attrs in SMITH_TERM_ATTRS.items():
        row = term_table.get_entry(term, BOOK_SMITH, fields=["count_"])
        assert int(row["count_"]) == attrs["count"]


def test_publish_writes_pos_tags_to_term_table(smith_s3_data, term_table):
    _run_publish()

    # "labour" appears as both NN and VB in the test POS data.
    labour_row = term_table.get_entry("labour", BOOK_SMITH, fields=["tags"])
    assert "N" in labour_row["tags"]
    assert "V" in labour_row["tags"]

    # "rent" appears only as NN.
    rent_row = term_table.get_entry("rent", BOOK_SMITH, fields=["tags"])
    assert rent_row["tags"] == {"N"}


def test_publish_writes_seed_vectors_to_term_table(smith_s3_data, term_table):
    _run_publish()

    row = term_table.get_entry("labour", BOOK_SMITH, fields=["vectors", "seeds"])
    assert len(row["vectors"]) == 2  # NUM_SEEDS
    assert len(row["seeds"]) == 2


def test_publish_writes_alignment_stats_to_term_table(smith_s3_data, term_table):
    _run_publish()

    row = term_table.get_entry("labour", BOOK_SMITH, fields=["alignment_stats"])
    stats = row["alignment_stats"]
    attrs = SMITH_TERM_ATTRS["labour"]
    assert stats["variance"] == Decimal(str(attrs["variance"]))
    assert stats["disparity"] == Decimal(str(attrs["disparity"]))
    assert stats["r_squared"] == Decimal(str(attrs["r_squared"]))


def test_publish_writes_ilocs_to_term_table(smith_s3_data, term_table):
    _run_publish()

    for term in SMITH_TERM_ATTRS:
        row = term_table.get_entry(term, BOOK_SMITH, fields=["ilocs"])
        assert row["ilocs"], f"Expected non-empty ilocs for '{term}'"


def test_publish_populates_corpus_term_table(smith_s3_data, corpus_term_table):
    _run_publish()

    for term in SMITH_TERM_ATTRS:
        row = corpus_term_table.get_term(term)
        assert row is not None, f"Missing corpus row for '{term}'"
        assert BOOK_SMITH in row["book_ids"]


def test_publish_updates_pipeline_metadata(smith_s3_data, pipeline_entries):
    _run_publish()

    entry = pipeline_entries.get_entry(BOOK_SMITH, fields=["author", "title"])
    assert entry.author == "Smith, Adam"
    assert entry.title == "The Wealth of Nations"


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

    # Include "the" in the centroid model alongside the normal terms.
    attrs_with_stopword = dict(SMITH_TERM_ATTRS)
    attrs_with_stopword["the"] = {
        "count": 500,
        "variance": 0.01,
        "disparity": 0.01,
        "r_squared": 0.99,
    }

    centroid_kv = _create_keyed_vectors(attrs_with_stopword, VECTOR_DIM, rng)
    _save_keyed_vectors_to_s3(
        f"kvectors/{BOOK_SMITH}/aligned/centroid.model", centroid_kv
    )

    for seed_idx in range(NUM_SEEDS):
        seed_kv = _create_keyed_vectors(attrs_with_stopword, VECTOR_DIM, rng)
        _save_keyed_vectors_to_s3(
            f"kvectors/{BOOK_SMITH}/aligned/{seed_idx}-seed.model", seed_kv
        )

    entry = PipelineEntry(
        book_id=BOOK_SMITH,
        status=EntryStatus.EMBEDDINGS_CREATED,
        published_year=1776,
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
    assert written_terms == set(SMITH_TERM_ATTRS)


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
