from __future__ import annotations

import numpy as np

from app.search.constants import (
    MAX_RANK_FOR_STABLE_TERM,
    MAX_RANK_FOR_UNSTABLE_TERM,
    MIN_BOOKS_WITH_TERM,
    MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    MIN_BOOKS_WITH_UNSTABLE_TERM_AS_TOP_50,
    NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING,
    NUM_RELEVANT_TERMS_FOR_INSTABILITY,
    NUM_COMPARATIVE_TERMS,
    T_CRIT_95,
)
from app.search.errors import NoLocalNearestTermsError
from app.search.schemas.semantic_drift import (
    DefinitionalAgreement,
    DefinitionalAgreementToCorpus,
    TermStats,
)
from app.search.services.semantic_drift.book_similarity_vectors import (
    BooksSimilarityCache,
)
from app.search.services.semantic_drift.utils import SearchExpr
from shared.commons import BookIndex


def get_n_highest_similarities(similarities: np.ndarray, n: int) -> np.ndarray:

    if not len(similarities):
        return similarities

    n = min(n, len(similarities))

    return np.partition(similarities, -n)[-n:]


def center_locally(similarities: np.ndarray, n: int) -> np.ndarray:

    highest_similarities = get_n_highest_similarities(similarities, n)

    if not len(highest_similarities):
        return similarities

    return similarities - highest_similarities.mean()


def get_is_local(similarities: np.ndarray, n: int) -> np.ndarray:

    highest_similarities = get_n_highest_similarities(similarities, n)

    if not len(highest_similarities):
        return np.zeros(0, dtype=bool)

    return similarities >= highest_similarities.min()


def get_unique_terms(all_terms):
    # faster equivalent of np.unique(all_terms, return_inverse=True)
    order = np.argsort(all_terms, kind="stable")
    sorted_terms = all_terms[order]
    is_first = np.ones(len(all_terms), dtype=bool)
    np.not_equal(sorted_terms[1:], sorted_terms[:-1], out=is_first[1:])

    terms = sorted_terms[is_first]
    term_iloc = np.empty(len(all_terms), dtype=np.intp)
    term_iloc[order] = np.cumsum(is_first) - 1
    return terms, term_iloc


def get_comparative_terms(
    books_similarity_cache: BooksSimilarityCache,
    book_ids: list[BookIndex],
    query: SearchExpr,
    *,
    selected_book_id: BookIndex | None = None,
) -> list[TermStats]:

    book_similarities_list = []
    book_terms_list = []
    book_similarities_centered_list = []
    book_terms_is_in_top_50_list = []
    book_terms_is_in_top_100_list = []

    for book_id in book_ids:
        book_similarity_vectors = books_similarity_cache.load_book(book_id, query)
        book_similarities = book_similarity_vectors.mean_similarities_to_query
        book_similarities_list.append(book_similarities)
        book_terms_list.append(book_similarity_vectors.terms)
        book_similarities_centered_list.append(
            center_locally(
                book_similarities, NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING
            )
        )
        book_terms_is_in_top_50_list.append(
            get_is_local(book_similarities, MAX_RANK_FOR_STABLE_TERM)
        )
        book_terms_is_in_top_100_list.append(
            get_is_local(book_similarities, MAX_RANK_FOR_UNSTABLE_TERM)
        )

    terms, term_iloc = get_unique_terms(np.concat(book_terms_list))
    n_books_in = np.bincount(term_iloc)  # Terms are unique within a book

    is_relevant_to_corpus = (n_books_in >= MIN_BOOKS_WITH_TERM) & ~np.isin(
        terms, query.terms
    )
    if selected_book_id is not None:
        selected_terms = books_similarity_cache.books_term_cache[selected_book_id].terms
        position = np.searchsorted(selected_terms, terms)
        is_relevant_to_corpus &= np.append(selected_terms, "")[position] == terms

    is_in_top_50 = np.concat(book_terms_is_in_top_50_list)
    is_in_top_100 = np.concat(book_terms_is_in_top_100_list)

    n_books_50 = np.bincount(term_iloc, weights=is_in_top_50).astype(np.intp)
    n_books_100 = np.bincount(term_iloc, weights=is_in_top_100).astype(np.intp)

    all_similarities = np.concat(book_similarities_centered_list)
    centered_similarity_stability = np.bincount(term_iloc, weights=all_similarities) / n_books_in

    variance = np.bincount(
        term_iloc, weights=(all_similarities - centered_similarity_stability[term_iloc]) ** 2
    ) / np.maximum(n_books_in - 1, 1)

    def make_term_stats(iloc: np.intp) -> TermStats:
        return TermStats(
            term=str(terms[iloc]),
            stability=float(centered_similarity_stability[iloc]),
            instability=float(variance[iloc]),
            n_books_in=int(n_books_in[iloc]),
            n_books_as_top50=int(n_books_50[iloc]),
            n_books_as_top100=int(n_books_100[iloc]),
        )

    mean_iloc = np.flatnonzero(
        is_relevant_to_corpus & (n_books_50 >= MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS)
    )
    mean_ranked = mean_iloc[np.argsort(-centered_similarity_stability[mean_iloc], kind="stable")][
        :NUM_COMPARATIVE_TERMS
    ]

    variance_iloc = np.flatnonzero(
        is_relevant_to_corpus
        & (n_books_100 >= MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS)
        & (n_books_50 >= MIN_BOOKS_WITH_UNSTABLE_TERM_AS_TOP_50)
    )
    most_similar_iloc = variance_iloc[
        np.argsort(-centered_similarity_stability[variance_iloc], kind="stable")
    ]
    pool_iloc = np.sort(most_similar_iloc[:NUM_RELEVANT_TERMS_FOR_INSTABILITY])
    variance_ranked = pool_iloc[
        np.argsort(-variance[pool_iloc], kind="stable")[:NUM_COMPARATIVE_TERMS]
    ]

    return [make_term_stats(iloc) for iloc in np.union1d(mean_ranked, variance_ranked)]


def t_crit_95(df: int) -> float:
    """Two-tailed 95% critical value for `df` degrees of freedom."""
    return T_CRIT_95[df] if df < len(T_CRIT_95) else 1.96


def standard_error_half_width(observations: np.ndarray) -> float:
    """Half-width of the 95% interval on the mean of `observations`."""
    n = len(observations)
    if n < 2:
        return 0.0
    sd = float(np.std(observations, ddof=1))
    return t_crit_95(n - 1) * sd / np.sqrt(n)


def get_mean_local_similarity_per_book(
    book_id: BookIndex,
    local_similarities_per_peer: list[np.ndarray],
    occurrences: int,
    *,
    against_corpus: bool,
):

    n_seeds = min(len(similarity) for similarity in local_similarities_per_peer)
    truncated = [similarity[:n_seeds] for similarity in local_similarities_per_peer]

    mean_local_similarities_per_seed = np.mean(truncated, axis=0)
    mean_local_similarity = float(np.mean(mean_local_similarities_per_seed))

    per_peer_means = np.array([similarity.mean() for similarity in truncated])
    if against_corpus and len(per_peer_means) > 1:
        ci_half = standard_error_half_width(per_peer_means)
    else:
        ci_half = standard_error_half_width(mean_local_similarities_per_seed)

    ci = (mean_local_similarity - ci_half, mean_local_similarity + ci_half)

    if against_corpus:
        return DefinitionalAgreementToCorpus(
            book_id=book_id.source_id,
            mean_local_similarity=mean_local_similarity,
            ci=ci,
            occurrences=occurrences,
            n_seeds=n_seeds,
            n_books=len(local_similarities_per_peer),
        )
    return DefinitionalAgreement(
        book_id=book_id.source_id,
        mean_local_similarity=mean_local_similarity,
        ci=ci,
        occurrences=occurrences,
        n_seeds=n_seeds,
    )


def get_mean_local_similarities(
    books_similarity_cache: BooksSimilarityCache,
    expr: SearchExpr,
    book_ids: list[BookIndex],
    selected_book_id: BookIndex | None = None,
) -> list[DefinitionalAgreement] | list[DefinitionalAgreementToCorpus]:

    books_similarity_vectors = books_similarity_cache.load_books(book_ids, expr)

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [books_similarity_cache.load_book(selected_book_id, expr)]

    books_data = []
    for book_similarity_vectors in books_similarity_vectors:
        local_similarities_per_peer = []
        for peer in peers:
            if peer.book_id == book_similarity_vectors.book_id:
                continue
            try:
                local_similarity = book_similarity_vectors.get_local_similarity(peer)
            except NoLocalNearestTermsError:
                continue
            local_similarities_per_peer.append(local_similarity)

        if not local_similarities_per_peer:
            continue

        # For a compound expression, the vocabulary volume behind the line.
        book_terms = books_similarity_cache.books_term_cache[
            book_similarity_vectors.book_id
        ]
        occurrences = sum(book_terms.get_term_count(term) for term in expr.terms)

        books_data.append(
            get_mean_local_similarity_per_book(
                book_similarity_vectors.book_id,
                local_similarities_per_peer,
                occurrences,
                against_corpus=selected_book_id is None,
            )
        )

    return books_data
