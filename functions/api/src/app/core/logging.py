"""Request-scoped context for canonical (one-line-per-request) logging.

The middleware seeds a fresh dict per request; anything running inside the
request adds to it via add_to_log(); the middleware emits one JSON line at
the end with everything attached. This is the only place the app builds a log
line.
"""

import json
import logging
import time
from contextvars import ContextVar
from functools import partial

from starlette.types import Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

request_log: ContextVar[dict] = ContextVar("request_log")


def add_to_log(**fields) -> None:
    """Attach fields to the current request's canonical log line.

    A no-op outside a request (get() falls back to a throwaway dict), and only
    ever *mutates* the dict -- never rebinds it. Endpoints here are sync `def`,
    so FastAPI runs them in a threadpool, and that hop copies the context: a
    rebind inside an endpoint wouldn't be seen back in the middleware, a
    mutation is. Turning this into a `.set()` silently drops every field added
    from inside an endpoint."""
    request_log.get({}).update(fields)


async def _send_with_logging(send: Send, scope: Scope, message: Message) -> None:
    """Record the response fields as the headers go out.

    Headers are only sent once the router has matched, so `route` and
    `path_params` are already on the scope by the time this runs."""
    if message["type"] == "http.response.start":
        add_to_log(
            **(scope.get("path_params") or {}),
            endpoint=getattr(scope.get("route"), "path", None),
            status=message["status"],
        )
    await send(message)


class RequestLoggingMiddleware:
    """Emits exactly one canonical JSON line per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan/websocket pass through untouched
            return await self.app(scope, receive, send)

        start = time.perf_counter()
        token = request_log.set({"method": scope["method"], "path": scope["path"]})
        try:
            await self.app(scope, receive, partial(_send_with_logging, send, scope))
        except Exception as exc:
            # Nothing was sent, so there's no status to conflict with. Without
            # this, an unhandled exception would skip the emit entirely and a
            # 500 would leave no trace on the canonical line.
            add_to_log(status=500, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            add_to_log(dur_ms=round((time.perf_counter() - start) * 1000, 1))
            logger.info("%s", json.dumps(request_log.get(), default=str))
            request_log.reset(token)
