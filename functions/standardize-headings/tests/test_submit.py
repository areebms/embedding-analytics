"""End-to-end tests for submit(), the corpus sweep that opens one Anthropic batch.

moto backs S3 and DynamoDB, so the real get_pending_book_records(), the real
manifest writes and the real status transitions all run. Only the Anthropic
Batches API is mocked — nothing fakes it the way moto fakes AWS — which is the
same split publish uses for Pinecone.
"""

import json
from unittest.mock import patch

from shared.tables.pipeline_entries import EntryStatus

from conftest import BATCH_ID, INDEX, INDEX_2, PROSE_ONLY_HTML


def _run_submit(anthropic_client):
    """Call the real submit() with the Anthropic client mocked out."""
    target = "llm_classify_request.send_request.get_client"
    with patch(target, return_value=anthropic_client):
        from submit import submit

        return submit()


def _body(bucket, key):
    return bucket.Object(key).get()["Body"].read().decode("utf-8")


# ── nothing to do ─────────────────────────────────────────────────────


def test_an_empty_corpus_opens_no_batch(entries, anthropic_client):
    summary = _run_submit(anthropic_client)

    assert summary == {"batch_id": None, "book_count": 0}
    anthropic_client.messages.batches.create.assert_not_called()


def test_a_second_run_cannot_resubmit_a_corpus_already_in_flight(
    seed, scraped_book, anthropic_client
):
    """The status is the only guard against paying twice for the same corpus."""
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX_2)
    scraped_book(INDEX)

    summary = _run_submit(anthropic_client)

    assert summary == {"batch_id": None, "book_count": 0}
    anthropic_client.messages.batches.create.assert_not_called()


# ── the submitting run ────────────────────────────────────────────────


def test_submit_opens_exactly_one_batch_for_the_whole_corpus(
    scraped_book, anthropic_client
):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    summary = _run_submit(anthropic_client)

    assert summary == {"batch_id": BATCH_ID, "book_count": 2}
    anthropic_client.messages.batches.create.assert_called_once()
    requests = anthropic_client.messages.batches.create.call_args.kwargs["requests"]
    assert len(requests) == 2


def test_every_submitted_book_advances_to_standardize_submitted(
    scraped_book, anthropic_client, statuses
):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    _run_submit(anthropic_client)

    assert statuses(INDEX) == EntryStatus.STANDARDIZE_SUBMITTED
    assert statuses(INDEX_2) == EntryStatus.STANDARDIZE_SUBMITTED


def test_a_book_of_pure_prose_is_marked_skipped_and_never_reaches_the_batch(
    scraped_book, anthropic_client, statuses
):
    scraped_book(INDEX)
    scraped_book(INDEX_2, PROSE_ONLY_HTML)

    summary = _run_submit(anthropic_client)

    assert summary["book_count"] == 1
    assert statuses(INDEX) == EntryStatus.STANDARDIZE_SUBMITTED
    assert statuses(INDEX_2) == EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS


# ── the manifest ──────────────────────────────────────────────────────


def test_the_manifest_maps_every_custom_id_back_to_its_book(
    scraped_book, anthropic_client, bucket
):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    _run_submit(anthropic_client)

    manifest = json.loads(_body(bucket, "standardize-batches/index.json"))
    assert manifest["batch_id"] == BATCH_ID
    assert manifest["custom_ids"] == {
        str(INDEX): str(INDEX),
        str(INDEX_2): str(INDEX_2),
    }


def test_each_book_gets_its_own_record_object(scraped_book, anthropic_client, bucket):
    scraped_book(INDEX)

    _run_submit(anthropic_client)

    record = json.loads(_body(bucket, f"standardize-batches/books/{INDEX}.json"))
    assert record["index"] == str(INDEX)
    assert record["custom_id"] == str(INDEX)
    assert ["h1", "The Wealth of Nations"] in record["tag_text_pairs"]


def test_the_manifest_is_written_before_any_book_is_marked_submitted(
    scraped_book, anthropic_client, entries, mocker
):
    """A book left at STANDARDIZE_SUBMITTED with no manifest to render from is out
    of reach of both stages: submit only sweeps SCRAPED_HTML, collect needs the
    manifest. The write has to land first."""
    import submit as submit_module

    scraped_book(INDEX)
    calls = []
    mocker.patch.object(
        submit_module,
        "save_batch_index",
        side_effect=lambda *_: calls.append("manifest"),
    )
    mocker.patch.object(
        entries, "update_entries", side_effect=lambda _: calls.append("status")
    )

    _run_submit(anthropic_client)

    assert calls == ["manifest", "status"]
