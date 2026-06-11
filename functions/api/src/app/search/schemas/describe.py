from pydantic import BaseModel


class ParseDescribeRequest(BaseModel):
    message: str


class SubstitutionResult(BaseModel):
    original: str
    resolved: str


class ParseDescribeResponse(BaseModel):
    expression: str
    terms: list[str]
    substitutions: list[SubstitutionResult]
