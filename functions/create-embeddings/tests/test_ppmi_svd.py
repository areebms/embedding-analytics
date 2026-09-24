import numpy as np

from constants import GAMMA, MIN_COUNT, VECTOR_SIZE, WINDOW
from ppmi_svd import (
    build_vocab,
    positive_pointwise_mutual_info,
    term_cooccurrence_in_window,
    truncated_svd,
)

from conftest import SYNTHETIC_PASSAGES


TERMS = [f"term{i}" for i in range(WINDOW + 4)]
VOCAB_INDEX = {term: i for i, term in enumerate(TERMS)}


def counts_for(passages):
    return term_cooccurrence_in_window(passages, VOCAB_INDEX).toarray()


def synthetic_counts():
    terms, term_indexes, _ = build_vocab(SYNTHETIC_PASSAGES)
    return terms, term_cooccurrence_in_window(SYNTHETIC_PASSAGES, term_indexes)


def test_passage_counts_is_symmetric():
    counts = counts_for([TERMS[:5], TERMS[2:8]])

    assert np.array_equal(counts, counts.T)


def test_passage_counts_pairs_every_token_within_the_window():
    counts = counts_for([TERMS[:3]])

    assert counts[0, 1] == 1
    assert counts[1, 2] == 1
    assert counts[0, 2] == 1


def test_passage_counts_ignores_tokens_beyond_the_window():
    counts = counts_for([TERMS[: WINDOW + 2]])

    assert counts[0, WINDOW] == 1
    assert counts[0, WINDOW + 1] == 0


def test_passage_counts_does_not_pair_across_passages():
    counts = counts_for([[TERMS[0]], [TERMS[1]]])

    assert counts.sum() == 0


def test_passage_counts_closes_the_gap_left_by_an_out_of_vocabulary_token():
    with_gap = counts_for([[TERMS[0], "unknown", TERMS[1]]])
    adjacent = counts_for([[TERMS[0], TERMS[1]]])

    assert np.array_equal(with_gap, adjacent)


def test_passage_counts_counts_a_repeated_term_on_the_diagonal():
    counts = counts_for([[TERMS[0], TERMS[1], TERMS[0]]])

    assert counts[0, 0] == 2


def test_passage_counts_returns_an_empty_matrix_for_an_empty_corpus():
    matrix = term_cooccurrence_in_window([], VOCAB_INDEX)

    assert matrix.shape == (len(VOCAB_INDEX), len(VOCAB_INDEX))
    assert matrix.nnz == 0


def test_build_vocab_drops_a_term_below_the_count_threshold():
    passages = [["common"] * MIN_COUNT, ["rare"] * (MIN_COUNT - 1)]

    terms, term_indexes, counts = build_vocab(passages)

    assert terms == ["common"]
    assert term_indexes == {"common": 0}
    assert counts["rare"] == MIN_COUNT - 1


def test_build_vocab_orders_by_count_then_alphabetically():
    passages = [
        ["frequent"] * (MIN_COUNT * 2)
        + ["beta"] * MIN_COUNT
        + ["alpha"] * MIN_COUNT
    ]

    terms, _, _ = build_vocab(passages)

    assert terms == ["frequent", "alpha", "beta"]


def test_build_vocab_indexes_follow_the_kept_order():
    terms, term_indexes, _ = build_vocab(SYNTHETIC_PASSAGES)

    assert [terms[index] for index in range(len(terms))] == terms
    assert all(term_indexes[term] == index for index, term in enumerate(terms))


def test_ppmi_is_never_negative():
    _, counts = synthetic_counts()

    scores = positive_pointwise_mutual_info(counts)

    assert scores.data.min() > 0


def test_ppmi_keeps_no_more_cells_than_the_counts_it_came_from():
    _, counts = synthetic_counts()

    assert positive_pointwise_mutual_info(counts).nnz <= counts.nnz


def test_ppmi_of_an_empty_matrix_is_empty():
    counts = term_cooccurrence_in_window([], VOCAB_INDEX)

    assert positive_pointwise_mutual_info(counts).nnz == 0


def test_ppmi_scores_an_exclusive_pair_above_a_shared_one():
    passages = [["alpha", "beta"]] * MIN_COUNT + [["alpha", "gamma"]] * MIN_COUNT
    passages += [["gamma", "delta"]] * MIN_COUNT
    terms, term_indexes, _ = build_vocab(passages)
    counts = term_cooccurrence_in_window(passages, term_indexes)

    scores = positive_pointwise_mutual_info(counts).toarray()
    beta, gamma = term_indexes["beta"], term_indexes["gamma"]
    alpha = term_indexes["alpha"]

    assert scores[alpha, beta] > scores[alpha, gamma]


def test_truncated_svd_returns_a_vector_size_embedding_per_term():
    terms, counts = synthetic_counts()

    vectors = truncated_svd(positive_pointwise_mutual_info(counts))

    assert vectors.shape == (len(terms), VECTOR_SIZE)


def test_truncated_svd_is_deterministic():
    _, counts = synthetic_counts()
    scores = positive_pointwise_mutual_info(counts)

    assert np.array_equal(truncated_svd(scores), truncated_svd(scores))


def test_truncated_svd_orders_dimensions_by_descending_strength():
    _, counts = synthetic_counts()

    vectors = truncated_svd(positive_pointwise_mutual_info(counts))
    strengths = np.linalg.norm(vectors, axis=0) ** (1 / GAMMA)

    assert np.all(np.diff(strengths) <= 0)
