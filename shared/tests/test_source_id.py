"""Tests for BookIndex, the composite gutenberg-{source_id} key type."""

import copy
import pickle

import pytest

from shared.commons import BookIndex


def test_str_value():
    assert BookIndex(3300) == "gutenberg-3300"


def test_interchangeable_with_str_as_dict_key():
    key = BookIndex(3300)
    table = {key: "smith"}
    assert table["gutenberg-3300"] == "smith"


def test_source_id_round_trips():
    assert BookIndex(3300).source_id == 3300


def test_parse_recovers_source_id():
    parsed = BookIndex.parse("gutenberg-3300")
    assert parsed == "gutenberg-3300"
    assert parsed.source_id == 3300


def test_non_numeric_source_id_fails_at_construction():
    with pytest.raises(ValueError):
        BookIndex("abc")


@pytest.mark.parametrize("raw", ["007", " 7", "+7", 7])
def test_string_form_is_normalized_through_int(raw):
    assert BookIndex(raw) == BookIndex(7) == "gutenberg-7"
    assert BookIndex(raw).source_id == 7


@pytest.mark.parametrize("clone", [
    copy.copy,
    copy.deepcopy,
    lambda s: pickle.loads(pickle.dumps(s)),
])
def test_copy_and_pickle_do_not_double_prefix(clone):
    # botocore deep-copies request params; a BookIndex used as a DynamoDB key
    # must survive copy/pickle without re-prefixing to gutenberg-gutenberg-3300.
    original = BookIndex(3300)
    cloned = clone(original)
    assert cloned == "gutenberg-3300"
    assert cloned.source_id == 3300
