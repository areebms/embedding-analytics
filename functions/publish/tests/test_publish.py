"""End-to-end tests for the publish() function in main.py.

Uses moto for DynamoDB (BookTermTable, CorpusTermTable, PipelineTable) and
S3 (centroid models, seed models, POS CSVs, metadata JSON). Pinecone
is mocked since moto doesn't support it.

These tests call the actual publish() function, not a reimplementation.
"""

import os
from decimal import Decimal
from unittest.mock import patch

import numpy as np

from shared.tables.pipeline_entries import EntryStatus, metadata_key

from conftest import (
    BOOK_SMITH,
    BOOK_RICARDO,
    SMITH_TERM_ATTRS,
    VECTOR_DIM,
    NUM_SEEDS,
    _create_keyed_vectors,
    _save_keyed_vectors_to_s3,
    _upload_csv_to_s3,
    _upload_json_to_s3,
    _seed_book,
)


def _run_publish(pinecone_table, book_id=BOOK_SMITH):
    """Call the real publish() with Pinecone mocked out."""
    with patch("publish_utils.get_pinecone_table", return_value=pinecone_table):
        from publish_utils import publish

        publish(book_id)


# ── First-time publish ────────────────────────────────────────────────


def test_publish_writes_terms_to_term_table(smith_s3_data, term_table, pinecone_table):
    _run_publish(pinecone_table)

    written_terms = {
        row["term"] for row in term_table.get_entries(BOOK_SMITH, fields=["term"])
    }
    assert written_terms == set(SMITH_TERM_ATTRS.keys())


def test_publish_writes_correct_counts_to_term_table(
    smith_s3_data, term_table, pinecone_table
):
    _run_publish(pinecone_table)

    for term, attrs in SMITH_TERM_ATTRS.items():
        row = term_table.get_entry(term, BOOK_SMITH, fields=["count_"])
        assert int(row["count_"]) == attrs["count"]


def test_publish_writes_pos_tags_to_term_table(
    smith_s3_data, term_table, pinecone_table
):
    _run_publish(pinecone_table)

    # "labour" appears as both NN and VB in the test POS data.
    labour_row = term_table.get_entry("labour", BOOK_SMITH, fields=["tags"])
    assert "N" in labour_row["tags"]
    assert "V" in labour_row["tags"]

    # "rent" appears only as NN.
    rent_row = term_table.get_entry("rent", BOOK_SMITH, fields=["tags"])
    assert rent_row["tags"] == {"N"}


def test_publish_writes_seed_vectors_to_term_table(
    smith_s3_data, term_table, pinecone_table
):
    _run_publish(pinecone_table)

    row = term_table.get_entry("labour", BOOK_SMITH, fields=["vectors", "seeds"])
    assert len(row["vectors"]) == 2  # NUM_SEEDS
    assert len(row["seeds"]) == 2


def test_publish_writes_alignment_stats_to_term_table(
    smith_s3_data, term_table, pinecone_table
):
    _run_publish(pinecone_table)

    row = term_table.get_entry("labour", BOOK_SMITH, fields=["alignment_stats"])
    stats = row["alignment_stats"]
    attrs = SMITH_TERM_ATTRS["labour"]
    assert stats["variance"] == Decimal(str(attrs["variance"]))
    assert stats["disparity"] == Decimal(str(attrs["disparity"]))
    assert stats["r_squared"] == Decimal(str(attrs["r_squared"]))


def test_publish_writes_ilocs_to_term_table(smith_s3_data, term_table, pinecone_table):
    _run_publish(pinecone_table)

    for term in SMITH_TERM_ATTRS:
        row = term_table.get_entry(term, BOOK_SMITH, fields=["ilocs"])
        assert row["ilocs"], f"Expected non-empty ilocs for '{term}'"


def test_publish_upserts_to_pinecone(smith_s3_data, pinecone_table):
    _run_publish(pinecone_table)

    pinecone_table.batch_upsert.assert_called_once()
    batch = pinecone_table.batch_upsert.call_args[0][0]

    upserted_terms = {item["term"] for item in batch}
    assert upserted_terms == set(SMITH_TERM_ATTRS.keys())

    expected_pos = {"labour": ["N", "V"], "value": ["N"], "rent": ["N"]}
    expected_count = {"labour": 42, "value": 30, "rent": 10}
    for item in batch:
        assert item["book_id"] == BOOK_SMITH
        assert item["vector"].shape[0] == 10  # VECTOR_DIM
        assert item["vector"].dtype.name == "float32"
        assert item["pos"] == expected_pos[item["term"]]
        assert item["count"] == expected_count[item["term"]]


def test_publish_populates_corpus_term_table(
    smith_s3_data, corpus_term_table, pinecone_table
):
    _run_publish(pinecone_table)

    for term in SMITH_TERM_ATTRS:
        row = corpus_term_table.get_term(term)
        assert row is not None, f"Missing corpus row for '{term}'"
        assert BOOK_SMITH in row["book_ids"]


def test_publish_updates_pipeline_metadata(
    smith_s3_data, pipeline_table, pinecone_table
):
    _run_publish(pinecone_table)

    row = pipeline_table.get_entry(BOOK_SMITH, fields=["author", "title"])
    assert row["author"] == "Smith, Adam"
    assert row["title"] == "The Wealth of Nations"


def test_publish_skips_unscraped_book(pipeline_table, pinecone_table):
    """A book with no pipeline_status should be skipped entirely."""
    pipeline_table.table.put_item(
        Item={
            "platform_data": BOOK_SMITH,
            "published_year": 1776,
            # No pipeline_status.
        }
    )

    _run_publish(pinecone_table)

    pinecone_table.batch_upsert.assert_not_called()


def test_publish_filters_terms_with_non_content_pos_tags(
    moto_dynamo, term_table, pinecone_table
):
    """Terms that only appear with non-content POS tags (e.g. DT) should be
    excluded from the term intersection and not written to BookTermTable."""
    s3 = moto_dynamo.resource("s3")
    bucket = os.environ["S3_BUCKET"]
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
        s3, bucket, f"kvectors/{BOOK_SMITH}/aligned/centroid.model", centroid_kv
    )

    for seed_idx in range(NUM_SEEDS):
        seed_kv = _create_keyed_vectors(attrs_with_stopword, VECTOR_DIM, rng)
        _save_keyed_vectors_to_s3(
            s3, bucket, f"kvectors/{BOOK_SMITH}/aligned/{seed_idx}-seed.model", seed_kv
        )

    # "the" only gets a DT tag — should be filtered out.
    token_lemmas = [["labour", "value", "rent", "the"], ["labour", "value"]]
    token_tags = [["NN", "NN", "NN", "DT"], ["VB", "NN"]]
    lemmas_key = f"tokens/{BOOK_SMITH}/lemmas.csv"
    tags_key = f"tokens/{BOOK_SMITH}/tags.csv"
    _upload_csv_to_s3(s3, bucket, lemmas_key, token_lemmas)
    _upload_csv_to_s3(s3, bucket, tags_key, token_tags)

    _upload_json_to_s3(
        s3,
        bucket,
        metadata_key(BOOK_SMITH),
        {"author": ["Smith, Adam"], "title": ["The Wealth of Nations"]},
    )

    from shared.tables.pipeline import get_pipeline_table

    pt = get_pipeline_table()
    pt.table.put_item(
        Item={
            "platform_data": BOOK_SMITH,
            "pipeline_status": EntryStatus.SCRAPED_HTML,
            "s3_token_lemmas_key": lemmas_key,
            "s3_token_tags_key": tags_key,
            "published_year": 1776,
        }
    )

    _run_publish(pinecone_table)

    written_terms = {
        row["term"] for row in term_table.get_entries(BOOK_SMITH, fields=["term"])
    }
    assert "the" not in written_terms
    assert written_terms == set(SMITH_TERM_ATTRS.keys())


# ── Republish (prior data exists) ────────────────────────────────────


def test_republish_does_not_duplicate_book_id_in_corpus(
    smith_s3_data, corpus_term_table, pinecone_table
):
    """Republishing the same book should not add a duplicate entry to
    the corpus book_ids set."""
    _run_publish(pinecone_table)

    row = corpus_term_table.get_term("labour")
    assert BOOK_SMITH in row["book_ids"]
    assert len(row["book_ids"]) == 1

    # Republish the same book.
    _run_publish(pinecone_table)

    row = corpus_term_table.get_term("labour")
    assert BOOK_SMITH in row["book_ids"]
    assert len(row["book_ids"]) == 1


def test_republish_does_not_affect_other_books(
    smith_s3_data, term_table, corpus_term_table, pinecone_table
):
    """Seed Ricardo's data manually, then republish Smith. Ricardo's
    corpus entries should be untouched."""
    _run_publish(pinecone_table)

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
    _run_publish(pinecone_table)

    row = corpus_term_table.get_term("labour")
    assert BOOK_SMITH in row["book_ids"]
    assert BOOK_RICARDO in row["book_ids"]
