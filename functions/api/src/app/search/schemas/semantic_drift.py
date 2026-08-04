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
    sort: Literal["mean_similarity", "slope"] = "mean_similarity"

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

    model_config = ConfigDict(extra="forbid")  # see BookLocalMeanSimilarity

    id: int
    missing_terms: list[str] = Field(default_factory=list)


class BookLocalMeanSimilarity(BaseModel):

    model_config = ConfigDict(extra="forbid")

    book_id: int
    similarity: float
    similarity_ci: tuple[float, float]
    n_seeds: int
    min_local_terms: int
    n_books: int


class TermData(BaseModel):

    model_config = ConfigDict(extra="forbid")

    term: str
    books: list[BookLocalMeanSimilarity]


class ExprData(BaseModel):

    model_config = ConfigDict(extra="forbid")

    expr: str
    terms: list[str]
    books: list[BookLocalMeanSimilarity]


class SemanticDriftResponse(BaseModel):

    expr: ExprData
    nearest_terms: list[TermData]
    books: list[BookSummary]
