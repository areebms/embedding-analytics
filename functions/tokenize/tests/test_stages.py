"""The stage end to end, through the handler the step function invokes."""

import logging

import pytest

import app
import main
from shared.tables.pipeline_entries import EntryStatus

from conftest import (
    BOOK_PASSAGES,
    BOOK_TEXT,
    INDEX,
    INDEX_2,
    SUBJECT,
    csv_rows,
    status_of,
)


# ── The books it is handed ────────────────────────────────────────────


def test_a_named_book_is_tokenized_and_advanced(standardized_book, entries):
    index = standardized_book()

    status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {"found": 1, "tokenized": 1, "failed": []}
    assert status_of(entries, index) == EntryStatus.TOKENIZED


def test_the_three_artifacts_are_row_aligned_with_the_passages(
    standardized_book, bucket
):
    """Row `n` is passage `n` in all three files: the back-pointer a passage-grained
    reader needs to get from a lemma to the text it occurred in."""
    index = standardized_book()

    app.handler({"book_ids": [str(index)]}, None)

    texts = csv_rows(bucket, f"token_texts/{index}.csv")
    lemmas = csv_rows(bucket, f"token_lemmas/{index}.csv")
    tags = csv_rows(bucket, f"token_tags/{index}.csv")

    assert len(texts) == len(BOOK_PASSAGES)
    assert [len(row) for row in lemmas] == [len(row) for row in texts]
    assert [len(row) for row in tags] == [len(row) for row in texts]
    for row, passage in zip(texts, BOOK_PASSAGES):
        assert "".join(row) == passage.replace(" ", "")
    assert "labour" in lemmas[1]


def test_a_blank_line_run_does_not_become_an_empty_row(standardized_book, bucket):
    index = standardized_book(text=f"\n\n{BOOK_TEXT}\n\n\n\n")

    app.handler({"book_ids": [str(index)]}, None)

    assert len(csv_rows(bucket, f"token_texts/{index}.csv")) == len(BOOK_PASSAGES)


def test_a_subject_is_resolved_to_the_books_standing_at_standardized(
    standardized_book, seed, entries
):
    """The hand re-run's way in: same work, named by subject instead of by book."""
    standardized_book(INDEX)
    standardized_book(INDEX_2)
    seed(EntryStatus.LISTED, index=main.BookIndex(999))

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status == {"found": 2, "tokenized": 2, "failed": []}
    assert status_of(entries, INDEX) == EntryStatus.TOKENIZED
    assert status_of(entries, INDEX_2) == EntryStatus.TOKENIZED


def test_a_subject_is_capped_and_the_overflow_is_left_for_the_next_run(
    standardized_book, entries, monkeypatch
):
    """The one path whose size the caller does not set: the standardize machine hands
    over a book_ids list it sized itself, but a subject is however many books the table
    holds, and the whole list goes through spaCy in one invocation under a 900s ceiling.
    The overflow keeps STANDARDIZED and comes back next run, which makes re-invoking the
    drain. Ids are sorted, so which books make the cut is the same answer twice rather
    than whatever the Scan returned first.
    """
    monkeypatch.setattr("main.MAX_BOOKS_PER_SUBJECT", 3)
    indexes = [main.BookIndex(source_id) for source_id in range(1, 6)]
    for index in indexes:
        standardized_book(index)

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status == {"found": 3, "tokenized": 3, "failed": []}
    tokenized = [
        index for index in indexes if status_of(entries, index) == EntryStatus.TOKENIZED
    ]
    assert tokenized == sorted(indexes)[:3]
    for index in sorted(indexes)[3:]:
        assert status_of(entries, index) == EntryStatus.STANDARDIZED


# ── The books it refuses or drops ─────────────────────────────────────


@pytest.mark.parametrize(
    "event",
    [{}, {"book_ids": ["gutenberg-3300"], "subject_id": "gutenberg-12345"}],
    ids=["neither", "both"],
)
def test_naming_no_books_or_two_ways_at_once_is_refused(event):
    """A payload naming nothing must not be read as "everything at STANDARDIZED"."""
    with pytest.raises(ValueError):
        app.handler(event, None)


def test_a_book_past_standardized_is_dropped_before_spacy(seed, bucket, entries):
    """The duplicate-delivery case: the standardize machine's event may name books an
    earlier execution already took to TOKENIZED."""
    seed(EntryStatus.TOKENIZED)

    status = app.handler({"book_ids": [str(INDEX)]}, None)

    assert status == {"found": 0, "tokenized": 0, "failed": []}
    assert list(bucket.objects.filter(Prefix="token_texts/")) == []
    assert status_of(entries, INDEX) == EntryStatus.TOKENIZED


def test_a_book_with_no_pipeline_entry_is_reported_and_skipped(aws, caplog):
    with caplog.at_level(logging.WARNING, logger="main"):
        status = app.handler({"book_ids": ["gutenberg-404"]}, None)

    assert status == {"found": 0, "tokenized": 0, "failed": []}
    assert "1 of 1 book(s) have no pipeline entry." in caplog.text


# ── The books that fail ───────────────────────────────────────────────


def test_a_book_whose_text_has_no_passages_fails_instead_of_advancing(
    standardized_book, bucket, entries
):
    """Three 0-row CSVs at TOKENIZED would hand train-kvector a book of no passages;
    the standardize output is what needs looking at."""
    index = standardized_book(text="\n\n   \n\n")

    status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {"found": 1, "tokenized": 0, "failed": [str(index)]}
    assert status_of(entries, index) == EntryStatus.STANDARDIZED
    assert list(bucket.objects.filter(Prefix="token_texts/")) == []


def test_one_failing_book_does_not_end_the_run(
    standardized_book, entries, monkeypatch
):
    """The whole list is one invocation, so a book that raises is counted and left
    where it was rather than taking the other books down with it."""
    standardized_book(INDEX)
    standardized_book(INDEX_2, text="This passage explodes.")

    real_tokenize_passage = main.tokenize_passage

    def exploding_tokenize_passage(passage):
        if "explodes" in passage:
            raise RuntimeError("spaCy fell over")
        return real_tokenize_passage(passage)

    monkeypatch.setattr(main, "tokenize_passage", exploding_tokenize_passage)

    status = app.handler({"book_ids": [str(INDEX), str(INDEX_2)]}, None)

    assert status == {"found": 2, "tokenized": 1, "failed": [str(INDEX_2)]}
    assert status_of(entries, INDEX) == EntryStatus.TOKENIZED
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZED


def test_a_status_write_the_guard_turns_down_is_reported(seed, caplog):
    """`set_status` is a conditional write; a refusal comes back as False rather than
    raising, and silently discarding it would report the book as advanced."""
    seed(EntryStatus.TOKENIZED)

    with caplog.at_level(logging.WARNING, logger="main"):
        main.set_status(INDEX, EntryStatus.TOKENIZED)

    assert "the status guard refused the write" in caplog.text
