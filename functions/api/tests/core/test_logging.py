import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.logging import RequestLoggingMiddleware, add_to_log

LOGGER_NAME = "app.core.logging"


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/book/{source_book_id}")
    def ok(source_book_id: int):
        add_to_log(query="from inside the endpoint")
        return {"ok": True}

    @app.get("/boom")
    def boom():
        raise ValueError("kaboom")

    return app


@pytest.fixture
def caplog_json(caplog):
    """Parsed canonical lines emitted during the test."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    def _lines():
        return [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == LOGGER_NAME
        ]

    return _lines


def test_emits_exactly_one_line_per_request(app, caplog_json):
    with TestClient(app) as client:
        client.get("/book/42")
        client.get("/book/7")

    assert len(caplog_json()) == 2


def test_line_carries_request_and_response_fields(app, caplog_json):
    with TestClient(app) as client:
        client.get("/book/42")

    (line,) = caplog_json()
    assert line["method"] == "GET"
    assert line["path"] == "/book/42"
    assert line["endpoint"] == "/book/{source_book_id}"
    assert line["status"] == 200
    assert isinstance(line["dur_ms"], float)
    # Path params are merged in at top level, as Starlette's convertor output
    # (a str) rather than the endpoint signature's Pydantic-coerced int.
    assert line["source_book_id"] == "42"


def test_fields_added_inside_the_endpoint_survive_the_threadpool_hop(app, caplog_json):
    """The regression test that matters: if add_to_log is ever changed to
    rebind the contextvar instead of mutating it, this is what breaks -- the
    middleware's own fields still appear, but the endpoint's silently vanish."""
    with TestClient(app) as client:
        client.get("/book/42")

    (line,) = caplog_json()
    assert line["query"] == "from inside the endpoint"


def test_unhandled_exception_still_emits_a_line(app, caplog_json):
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/boom").status_code == 500

    (line,) = caplog_json()
    assert line["status"] == 500
    assert line["error"] == "ValueError: kaboom"
    assert line["path"] == "/boom"


def test_non_http_scopes_pass_through(app, caplog_json):
    """TestClient's context manager runs the lifespan scope through the stack;
    it must not be logged, and must not raise on the missing `method` key."""
    with TestClient(app):
        pass

    assert caplog_json() == []


def test_add_log_fields_outside_a_request_is_a_noop():
    add_to_log(orphan=True)  # no contextvar set; must not raise
