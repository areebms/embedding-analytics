from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import add_to_log
from app.search.schemas.errors import TermResolutionResponse


class TermResolutionError(Exception):
    """Raised when a term cannot be resolved to any vocabulary entry."""

    def __init__(self, term: str, candidates: list[str]):
        self.term = term
        self.candidates = candidates
        if candidates:
            msg = (
                f"No matching term found for '{term}'. "
                f"Did you mean: {', '.join(candidates)}?"
            )
        else:
            msg = (
                f"No matching term found for '{term}'. No similar terms in vocabulary."
            )
        super().__init__(msg)


async def term_resolution_handler(request: Request, exc: TermResolutionError):
    # A finding with "did you mean" candidates. 404 (not 422) keeps it off
    # request-validation's code so the two never collide in the client.
    add_to_log(error=f"{type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=404,
        content=TermResolutionResponse(
            message=str(exc), term=exc.term, candidates=exc.candidates
        ).model_dump(),
    )


async def request_validation_handler(request: Request, exc: RequestValidationError):
    # Not a finding: FastAPI's own 422 is served unchanged. The handler exists
    # only so the canonical log line carries the payload that failed to
    # validate -- otherwise a 422 logs as a bare status with no trace of what
    # the client sent. `exc.body` is None when the error is in a query or path
    # param rather than a body.
    add_to_log(query=exc.body, error="validation")
    return await request_validation_exception_handler(request, exc)


def add_exception_handlers(app):
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(TermResolutionError, term_resolution_handler)
