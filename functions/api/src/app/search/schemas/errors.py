"""
Response models for the search endpoints' error bodies, declared per-route via
`responses=` so they land in the OpenAPI schema and the generated client. The
handlers in app.search.errors construct and dump these models, so each model is
the single source of truth for its body shape and `reason` -- the served body
can't drift from the documented one.

Each is a finding the client renders, sharing one shape: a flat 404 body with a
`reason` Literal, so the frontend treats them as one discriminated union.
Request/parse failures keep their own codes (422, 400).
"""

from typing import ClassVar, Literal

from pydantic import BaseModel


class TermResolutionResponse(BaseModel):
    """candidates is empty when nothing in the vocabulary is close."""

    openapi_description: ClassVar[str] = (
        "A term in the request could not be matched to any word in the vocabulary."
    )

    reason: Literal["term_resolution"] = "term_resolution"
    message: str
    term: str
    candidates: list[str]
