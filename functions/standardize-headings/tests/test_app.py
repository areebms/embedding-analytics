"""Tests for the Lambda handler."""


def test_the_handler_returns_the_submit_summary(mocker):
    import app

    summary = {"batch_id": "msgbatch_abc", "book_count": 2}
    mocker.patch.object(app, "submit", return_value=summary)

    assert app.handler({}, None) == summary


def test_the_handler_takes_no_index_from_the_event(mocker):
    """A corpus-wide sweep, not a per-book stage: nothing in the payload is read,
    so the state machine cannot aim it at one book by accident."""
    import app

    submit = mocker.patch.object(app, "submit", return_value={})

    app.handler({"index": "gutenberg-3300", "platform_data": "gutenberg-11"}, None)

    submit.assert_called_once_with()
