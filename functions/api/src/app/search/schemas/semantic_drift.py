from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

MAX_TREE_DEPTH = 5


def check_tree_depth(tree):
    """Reject trees deeper than MAX_TREE_DEPTH."""
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_TREE_DEPTH:
            raise ValueError(f"expression nesting exceeds max depth {MAX_TREE_DEPTH}")
        args = getattr(node, "args", [])
        for arg in args:
            stack.append((arg, depth + 1))
    return tree


class TermNode(BaseModel):
    term: str

    @field_validator("term")
    @classmethod
    def term_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("term must not be blank")
        return v


class OpNode(BaseModel):
    op: Literal["+", "-"]
    args: Annotated[list["TermNode | OpNode"], Field(min_length=2, max_length=2)]


OpNode.model_rebuild()


ExprTree = Annotated[TermNode | OpNode, AfterValidator(check_tree_depth)]


class SemanticDriftRequestBody(BaseModel):

    tree: ExprTree
    book_ids: list[int] = Field(min_length=1, max_length=16)

    @field_validator("book_ids")
    @classmethod
    def book_ids_unique(cls, book_ids: list[int]) -> list[int]:
        """Reject repeated targets, naming them so the 422 is actionable."""
        seen: set[int] = set()
        repeated: list[int] = []
        for book_id in book_ids:
            if book_id in seen and book_id not in repeated:
                repeated.append(book_id)
            seen.add(book_id)
        if repeated:
            raise ValueError(f"book_ids must be unique; repeated: {repeated}")
        return book_ids


class SemanticDriftRequest(SemanticDriftRequestBody):

    source_book_id: int | None = None

    @model_validator(mode="after")
    def selected_book_not_in_book_ids(self):

        if self.source_book_id is not None and self.source_book_id in self.book_ids:
            raise ValueError(
                f"book_ids must not contain the selected book {self.source_book_id}"
            )
        return self


class BookSummary(BaseModel):

    model_config = ConfigDict(extra="forbid")

    id: int
    n_shared_terms: int
    missing_terms: list[str] = Field(default_factory=list)


class DefinitionalAgreement(BaseModel):
    """One book read against the nominated source book.

    `mean_local_similarity` is the mean across seeds of a single pairwise local
    similarity, over the 75 terms nearest the query in the measuring book. With
    one peer there is no between-book variation to estimate, so `ci` covers seed
    noise alone.
    """

    model_config = ConfigDict(extra="forbid")

    book_id: int
    mean_local_similarity: float
    ci: tuple[float, float]
    occurrences: int
    n_seeds: int


class DefinitionalAgreementToCorpus(BaseModel):
    """One book read against every other requested book.

    `mean_local_similarity` is the mean of the pairwise local similarities
    against each peer in turn -- not a comparison against one aggregate corpus
    profile, which would be a different quantity. `ci` treats the peers as the
    unit of replication, so it carries between-book disagreement as well as the
    seed noise inside each pairwise figure, and is not comparable in width to
    the interval on `DefinitionalAgreement`.
    """

    model_config = ConfigDict(extra="forbid")

    book_id: int
    mean_local_similarity: float
    ci: tuple[float, float]
    occurrences: int
    n_seeds: int
    n_books: int


class TermStats(BaseModel):

    model_config = ConfigDict(extra="forbid")

    term: str
    stability: float
    instability: float
    n_books_in: int
    n_books_as_top50: int
    n_books_as_top100: int


class TermData(TermStats):

    books: list[DefinitionalAgreement] | list[DefinitionalAgreementToCorpus]


class ExprData(BaseModel):

    model_config = ConfigDict(extra="forbid")

    expr: str
    terms: list[str]
    books: list[DefinitionalAgreement] | list[DefinitionalAgreementToCorpus]


class SemanticDriftResponse(BaseModel):

    expr: ExprData
    comparative_terms: list[TermData]
    books: list[BookSummary]
