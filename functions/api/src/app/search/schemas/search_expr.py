from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


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


class SimilarityRequest(BaseModel):
    tree: TermNode | OpNode
    top_k: int = Field(100, ge=1, le=500)


class SimilarityResult(BaseModel):
    term: str
    count: int
    similarity: float


class SimilarityResponse(BaseModel):
    results: list[SimilarityResult]
    query_vectors: list[list[float]]  # normalized per-seed query vectors


class ConfidenceRequest(BaseModel):
    terms: list[str]
    query_vectors: list[list[float]]


class ConfidenceResult(BaseModel):
    term: str
    similarity: float
    similarity_ci: tuple[float, float]
