import pytest

from gensim.models import KeyedVectors


def test_perform_alignment_returns_valid_outputs(alignment_result):
    mean_book_disparity, centroid = alignment_result

    assert 0.0 <= mean_book_disparity <= 0.04
    assert isinstance(centroid, KeyedVectors)


def test_perform_alignment_centroid_vecattrs(alignment_result):
    _, centroid = alignment_result

    for term in centroid.key_to_index:
        assert centroid.get_vecattr(term, "disparity") >= 0.0
        assert centroid.get_vecattr(term, "variance") >= 0.0
        assert centroid.get_vecattr(term, "r_squared") <= 1.0


def test_gradient_descent_alignment_raises_exception(kvector_stack):
    from procrustes_utils import gradient_descent_alignment

    terms = list(kvector_stack[0].key_to_index)

    with pytest.raises(Exception, match="Kvectors not aligned"):
        gradient_descent_alignment(terms, kvector_stack, max_iterations=0)


def test_gradient_descent_alignment_converges(kvector_stack):
    from procrustes_utils import gradient_descent_alignment

    terms = list(kvector_stack[0].key_to_index)

    _, _, mean_book_disparity, iteration = gradient_descent_alignment(
        terms, kvector_stack, max_iterations=10
    )

    assert mean_book_disparity == pytest.approx(0.039, 0.1)
    assert iteration == 5
