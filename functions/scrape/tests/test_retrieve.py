"""Tests for the Gutenberg HTML scrapers."""

import pytest

import retrieve
from conftest import BIBREC_ENGLISH


class FakeResponse:
    def __init__(self, body, raises=None):
        self.content = body.encode("utf-8")
        self.text = body
        self._raises = raises

    def raise_for_status(self):
        if self._raises:
            raise self._raises


def test_get_metadata_keys_each_row_by_its_header(mocker):
    """The bug this pins: the header variable was renamed and the key line missed it,
    so every call died with UnboundLocalError on the first row."""
    mocker.patch.object(
        retrieve, "get", return_value=FakeResponse(BIBREC_ENGLISH)
    )

    metadata = retrieve.get_metadata(3300)

    assert metadata["language"] == ["English"]
    assert metadata["author"] == ["Smith, Adam"]
    assert metadata["title"] == ["The Wealth of Nations"]


def test_get_metadata_records_hrefs_under_a_link_key(mocker):
    mocker.patch.object(
        retrieve, "get", return_value=FakeResponse(BIBREC_ENGLISH)
    )

    metadata = retrieve.get_metadata(3300)

    assert metadata["language_link"] == ["/browse/languages/en"]
    # The title row has no <a>, so it gets no link key at all.
    assert "title_link" not in metadata


def test_get_metadata_skips_rows_with_no_header(mocker):
    mocker.patch.object(
        retrieve, "get", return_value=FakeResponse(BIBREC_ENGLISH)
    )

    metadata = retrieve.get_metadata(3300)

    assert not any("no header" in value for values in metadata.values() for value in values)


def test_get_metadata_raises_when_the_page_has_no_bibrec_table(mocker):
    mocker.patch.object(
        retrieve, "get", return_value=FakeResponse("<html><body>404</body></html>")
    )

    with pytest.raises(ValueError, match="No metadata table"):
        retrieve.get_metadata(3300)


def test_get_html_returns_the_body_verbatim(mocker):
    raw = "<html><body>  Chapter I.\r\n  </body></html>"
    mocker.patch.object(retrieve, "get", return_value=FakeResponse(raw))

    # Stored raw, license boilerplate and all -- deriving text is standardize's job.
    assert retrieve.get_html(3300) == raw


# ── subject book listing ──────────────────────────────────────────────


def _subject_page(book_ids):
    links = "".join(f'<a href="/ebooks/{i}">Book {i}</a>' for i in book_ids)
    return FakeResponse(f"<html><body>{links}</body></html>")


def test_get_book_ids_stops_after_a_short_page(mocker):
    """Fewer results than a full page means there is no next page to ask for."""
    get = mocker.patch.object(retrieve, "get", return_value=_subject_page([1, 2, 3]))
    mocker.patch.object(retrieve, "sleep")

    assert retrieve.get_book_ids(42) == ["1", "2", "3"]
    assert get.call_count == 1


def test_get_book_ids_asks_for_another_page_while_pages_come_back_full(mocker):
    page_size = retrieve.MAX_BOOK_IDS_PER_PAGE
    first = list(range(1, page_size + 1))
    second = list(range(101, 101 + page_size))
    get = mocker.patch.object(
        retrieve,
        "get",
        side_effect=[
            _subject_page(first),
            _subject_page(second),
            _subject_page(second),  # all duplicates: nothing new, so the loop ends
        ],
    )
    mocker.patch.object(retrieve, "sleep")

    book_ids = retrieve.get_book_ids(42)

    assert book_ids == [str(i) for i in first + second]
    # start_index is 1-based and follows the running total.
    assert [call.args[0].rsplit("=", 1)[1] for call in get.call_args_list] == [
        "1",
        str(page_size + 1),
        str(2 * page_size + 1),
    ]


def test_get_book_ids_stops_on_a_partial_page(mocker):
    """A page that is not exactly full is the last one, even if it added results."""
    first = list(range(1, retrieve.MAX_BOOK_IDS_PER_PAGE + 1))
    get = mocker.patch.object(
        retrieve,
        "get",
        side_effect=[_subject_page(first), _subject_page([100, 101])],
    )
    mocker.patch.object(retrieve, "sleep")

    assert len(retrieve.get_book_ids(42)) == retrieve.MAX_BOOK_IDS_PER_PAGE + 2
    assert get.call_count == 2


def test_get_book_ids_ignores_non_book_links_and_duplicates(mocker):
    page = FakeResponse(
        "<html><body>"
        '<a href="/ebooks/3300">dupe target</a>'
        '<a href="/ebooks/3300">dupe</a>'
        '<a href="/ebooks/subject/42">not a book id</a>'
        '<a href="/about/">unrelated</a>'
        "<a>no href at all</a>"
        "</body></html>"
    )
    mocker.patch.object(retrieve, "get", return_value=page)
    mocker.patch.object(retrieve, "sleep")

    assert retrieve.get_book_ids(42) == ["3300"]
