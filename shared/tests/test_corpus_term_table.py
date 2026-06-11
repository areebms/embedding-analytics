"""Tests for CorpusTermTable DynamoDB operations (add, remove, list)."""


def test_add_single_book(corpus_term_table):
    corpus_term_table.add_book("labour", "gutenberg-3300")

    row = corpus_term_table.get_term("labour")
    assert row["term"] == "labour"
    assert row["book_ids"] == {"gutenberg-3300"}


def test_add_two_books_same_term(corpus_term_table):
    """Two books containing 'labour' should both appear in the set."""
    corpus_term_table.add_book("labour", "gutenberg-3300")
    corpus_term_table.add_book("labour", "gutenberg-33310")

    row = corpus_term_table.get_term("labour")
    assert row["book_ids"] == {"gutenberg-3300", "gutenberg-33310"}


def test_add_same_book_twice_deduplicates_set(corpus_term_table):
    """Adding the same book twice should not create a duplicate in book_ids
    (DynamoDB ADD on a set is idempotent for membership)."""
    corpus_term_table.add_book("labour", "gutenberg-3300")
    corpus_term_table.add_book("labour", "gutenberg-3300")

    row = corpus_term_table.get_term("labour")
    assert row["book_ids"] == {"gutenberg-3300"}


def test_remove_one_of_two_books(corpus_term_table):
    corpus_term_table.add_book("labour", "gutenberg-3300")
    corpus_term_table.add_book("labour", "gutenberg-33310")
    corpus_term_table.remove_book("labour", "gutenberg-3300")

    row = corpus_term_table.get_term("labour")
    assert row["book_ids"] == {"gutenberg-33310"}


def test_remove_last_book_empties_book_ids(corpus_term_table):
    corpus_term_table.add_book("labour", "gutenberg-3300")
    corpus_term_table.remove_book("labour", "gutenberg-3300")

    row = corpus_term_table.get_term("labour")
    assert "gutenberg-3300" not in (row or {}).get("book_ids", set())


def test_list_returns_all_terms(corpus_term_table):
    corpus_term_table.add_book("labour", "gutenberg-3300")
    corpus_term_table.add_book("value", "gutenberg-3300")
    corpus_term_table.add_book("rent", "gutenberg-3300")

    rows = corpus_term_table.list_all_terms()
    terms = {row["term"] for row in rows}
    assert terms == {"labour", "value", "rent"}
