from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ParseChatRequest(BaseModel):
    message: str


class SubstitutionResult(BaseModel):
    original: str
    resolved: str


class ParseChatResponse(BaseModel):
    expression: str
    terms: list[str]
    substitutions: list[SubstitutionResult]


class DiachronicRequest(BaseModel):
    tree: TermNode | OpNode
    ref_book_ids: Annotated[list[int], Field(min_length=1)]


class DiachronicResult(BaseModel):
    id: int
    similarity: float
    similarity_ci: tuple[float, float]
    n: int


class BookResponse(BaseModel):
    id: int
    label: str
    author: str
    title: str
    published_year: int
 
    @model_validator(mode="before")
    @classmethod
    def from_entry(cls, data: dict) -> dict:
        if "platform_data" in data:
            data["id"] = int(data.pop("platform_data").split("-")[-1])
            data["label"] = f"{data['author'].split(',')[0]} ({data['published_year']})"
        return data


class TermResponse(BaseModel):
    term: str
    books: list[str]
