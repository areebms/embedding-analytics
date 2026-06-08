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


class SimilarityResult(BaseModel):
    term: str
    pos: set[str]
    count: int
    similarity: float
    similarity_ci: tuple[float, float]
