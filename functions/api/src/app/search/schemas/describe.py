from pydantic import BaseModel


class ParseDescribeRequest(BaseModel):
    message: str


class SubstitutionResponse(BaseModel):
    original: str
    resolved: str


class ParseDescribeResponse(BaseModel):
    expression: str
    terms: list[str]
    substitutions: list[SubstitutionResponse]
