"""Tests for the handler's stage dispatch.

One Lambda runs both stages and the event's `stage` picks which. Both are corpus-wide
and invoked by hand, so a mistyped payload has to be refused rather than interpreted:
defaulting to submit would open a batch over the whole corpus by accident.
"""

import json

import pytest

from conftest import BATCH_ID

import app


def test_the_submit_stage_runs_and_returns_its_status(mocker):
    submit = mocker.patch.object(
        app, "submit", return_value={"batch_id": BATCH_ID, "book_count": 42}
    )

    assert app.handler({"stage": "submit"}, None) == {
        "batch_id": BATCH_ID,
        "book_count": 42,
    }
    submit.assert_called_once_with()


def test_the_collect_stage_runs_and_returns_its_status(mocker):
    standardize = mocker.patch.object(
        app,
        "standardize_from_batch",
        return_value={"batch_id": BATCH_ID, "batch_status": "ended", "standardized": 41},
    )

    result = app.handler({"stage": "collect", "batch_id": BATCH_ID}, None)

    assert result["standardized"] == 41
    standardize.assert_called_once_with(BATCH_ID)


def test_submit_needs_no_batch_id(mocker):
    """It is the stage that creates one."""
    mocker.patch.object(app, "submit", return_value={"batch_id": None, "book_count": 0})

    assert app.handler({"stage": "submit"}, None)["book_count"] == 0


def test_the_batch_id_is_read_from_a_json_body(mocker):
    standardize = mocker.patch.object(
        app, "standardize_from_batch", return_value={"standardized": 0}
    )

    app.handler({"stage": "collect", "body": json.dumps({"batch_id": BATCH_ID})}, None)

    standardize.assert_called_once_with(BATCH_ID)


def test_an_event_with_no_stage_is_rejected():
    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({"batch_id": BATCH_ID}, None)


def test_an_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match=r"stage must be one of \['collect', 'submit'\]"):
        app.handler({"stage": "standardize"}, None)


def test_an_empty_event_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({}, None)


def test_a_missing_event_is_rejected_rather_than_crashing():
    """`aws lambda invoke` with no payload sends null, not {}."""
    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler(None, None)


def test_the_stage_is_validated_before_the_batch_id():
    """submit takes no batch_id, so an event missing both has to fail on the stage."""
    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({}, None)


def test_collect_without_a_batch_id_is_rejected():
    with pytest.raises(ValueError, match="batch_id is required"):
        app.handler({"stage": "collect"}, None)


def test_collect_with_an_empty_batch_id_is_rejected():
    with pytest.raises(ValueError, match="batch_id is required"):
        app.handler({"stage": "collect", "batch_id": ""}, None)


def test_the_returned_status_is_json_serialisable(mocker):
    """The payload goes back out through `aws lambda invoke`, so it has to survive
    serialisation — book indexes included."""
    from conftest import INDEX

    mocker.patch.object(
        app, "submit", return_value={"batch_id": BATCH_ID, "index": INDEX}
    )

    result = app.handler({"stage": "submit"}, None)

    assert json.loads(json.dumps(result)) == {
        "batch_id": BATCH_ID,
        "index": "gutenberg-3300",
    }
