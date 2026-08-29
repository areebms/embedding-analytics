"""Round-trip coverage for the S3 helpers.

The three stages that still read and write through this module -- publish,
train-kvector, align-kvectors -- verify none of it: publish's suite errors in setup,
train-kvector has no tests, and align-kvectors' suite runs against real S3 and only
exercises the alignment math. These are what actually run the module.
"""

import json

import pytest

from shared.s3 import (
    load_csv,
    load_file,
    load_json,
    load_text,
    upload_csv,
    upload_file,
    upload_html,
    upload_json,
    upload_object,
    upload_txt,
    yield_s3_files,
)
from shared.tests_utils import aws, bucket, s3_content_type  # noqa: F401


def test_text_round_trips(bucket):
    upload_txt("book.txt", "one\n\ntwo\n")

    assert load_text("book.txt") == "one\n\ntwo\n"
    assert s3_content_type(bucket, "book.txt") == "text/plain; charset=utf-8"


def test_json_round_trips(bucket):
    upload_json("record.json", json.dumps({"title": ["Ulysses"], "author": ["Joyce"]}))

    assert load_json("record.json") == {"title": ["Ulysses"], "author": ["Joyce"]}
    assert s3_content_type(bucket, "record.json") == "application/json; charset=utf-8"


def test_csv_round_trips(bucket):
    upload_csv("lemmas.csv", [["the", "sea"], ["snot", "green"]])

    assert list(load_csv("lemmas.csv")) == [["the", "sea"], ["snot", "green"]]
    assert s3_content_type(bucket, "lemmas.csv") == "text/csv; charset=utf-8"


def test_load_csv_is_lazy(bucket):
    """The token CSVs are a whole book of lemmas, so the read must not materialize."""
    upload_csv("lemmas.csv", [["a"], ["b"], ["c"]])

    rows = load_csv("lemmas.csv")

    assert next(rows) == ["a"]
    assert next(rows) == ["b"]


def test_upload_html_sets_its_content_type(bucket):
    upload_html("book.html", "<p>hello</p>")

    assert load_text("book.html") == "<p>hello</p>"
    assert s3_content_type(bucket, "book.html") == "text/html; charset=utf-8"


def test_upload_object_takes_any_content_type(bucket):
    upload_object("page.xml", "<x/>", "application/xml")

    assert s3_content_type(bucket, "page.xml") == "application/xml"


def test_non_ascii_survives_the_round_trip(bucket):
    upload_txt("french.txt", "l'été où l'on naît")

    assert load_text("french.txt") == "l'été où l'on naît"


def test_file_round_trips(bucket, tmp_path):
    source = tmp_path / "centroid.model"
    source.write_bytes(b"\x00binary\xff")

    upload_file("kvectors/1/aligned/centroid.model", str(source))

    with load_file("kvectors/1/aligned/centroid.model") as (key, local_path):
        assert key == "kvectors/1/aligned/centroid.model"
        with open(local_path, "rb") as handle:
            assert handle.read() == b"\x00binary\xff"


def test_load_file_cleans_up_its_temp_file(bucket, tmp_path):
    source = tmp_path / "centroid.model"
    source.write_bytes(b"x")
    upload_file("kvectors/1/aligned/centroid.model", str(source))

    with load_file("kvectors/1/aligned/centroid.model") as (_, local_path):
        pass

    with pytest.raises(FileNotFoundError):
        open(local_path, "rb")


def test_yield_s3_files_filters_by_prefix_and_extension(bucket, tmp_path):
    source = tmp_path / "m.model"
    source.write_bytes(b"m")
    upload_file("kvectors/1/aligned/a.model", str(source))
    upload_file("kvectors/1/aligned/b.model", str(source))
    upload_file("kvectors/1/aligned/notes.txt", str(source))
    upload_file("kvectors/2/aligned/other.model", str(source))

    keys = [key for key, _ in yield_s3_files("kvectors/1/aligned/", ".model")]

    assert sorted(keys) == ["kvectors/1/aligned/a.model", "kvectors/1/aligned/b.model"]
