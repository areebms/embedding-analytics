from fastapi import APIRouter


def post_route(router: APIRouter, path: str, *, responses: dict | None = None, **kwargs):
    """Register a POST route, expanding a bare response model into OpenAPI's
    {"model", "description"} form so each error schema documents itself."""
    return router.post(
        path,
        responses={
            status_code: (
                {"model": value, "description": value.openapi_description}
                if isinstance(value, type)
                else value
            )
            for status_code, value in (responses or {}).items()
        },
        **kwargs,
    )
