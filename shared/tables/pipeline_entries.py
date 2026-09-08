from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator
from boto3.dynamodb.conditions import Key

from shared.commons import BookIndex
from shared.session import get_session
from shared.tables.pipeline import PipelineTable


class EntryStatus(StrEnum):
    """`<rank>[T]_<NAME>`: the fixed-width rank orders the pipeline, and a `T` before
    the underscore marks a state a book never leaves."""

    LISTED = "0000_LISTED"
    SCRAPED_METADATA = "0100_SCRAPED_METADATA"
    SCRAPED_SKIPPED_NON_ENGLISH = "0101T_SCRAPED_SKIPPED_NON_ENGLISH"
    SCRAPED_HTML = "0150_SCRAPED_HTML"
    SCRAPED_SKIPPED_NO_HEADINGS = "0151T_SCRAPED_SKIPPED_NO_HEADINGS"
    STANDARDIZE_SUBMITTED = "0200_STANDARDIZE_SUBMITTED"
    STANDARDIZE_UNRESOLVED = "0201_STANDARDIZE_UNRESOLVED"
    STANDARDIZED = "0250_STANDARDIZED"
    TOKENIZED = "0300_TOKENIZED"
    EMBEDDINGS_CREATED = "0350_EMBEDDINGS_CREATED"
    EMBEDDINGS_CREATION_FAILED = "0351T_EMBEDDINGS_CREATION_FAILED"

    @property
    def is_terminal(self) -> bool:
        return self.split("_", 1)[0].endswith("T")


TERMINAL_STATUSES = tuple(status for status in EntryStatus if status.is_terminal)


BookIndexField = Annotated[
    BookIndex,
    PlainValidator(
        lambda value: value if isinstance(value, BookIndex) else BookIndex.parse(value)
    ),
    PlainSerializer(str, return_type=str),
]


class BookMetadata(BaseModel):
    author: str
    title: str
    published_year: int | None = None


class PipelineEntry(BaseModel):

    model_config = ConfigDict(extra="ignore")

    book_id: BookIndexField
    subject_ids: set[BookIndexField] = Field(default_factory=set)
    status: EntryStatus | None = None
    metadata: BookMetadata | None = None

    @property
    def s3_metadata_key(self) -> str:
        return f"metadata/{self.book_id}.json"

    @property
    def s3_html_key(self) -> str:
        return f"html/{self.book_id}.html"

    @property
    def s3_book_pairs_key(self) -> str:
        return f"standardize-html/books/{self.book_id}.json"

    @property
    def s3_standardized_html_key(self) -> str:
        return f"html-standardized/{self.book_id}.html"

    @property
    def s3_text_key(self) -> str:
        return f"text/{self.book_id}.txt"

    @property
    def s3_token_texts_key(self) -> str:
        return f"token_texts/{self.book_id}.csv"

    @property
    def s3_token_lemmas_key(self) -> str:
        return f"token_lemmas/{self.book_id}.csv"

    @property
    def s3_token_tags_key(self) -> str:
        return f"token_tags/{self.book_id}.csv"

    @property
    def s3_embeddings_key(self) -> str:
        return f"embeddings/{self.book_id}.npz"


_pipeline_entries = None


def get_pipeline_entries():
    global _pipeline_entries
    if _pipeline_entries is None:
        _pipeline_entries = PipelineEntries(get_session())
    return _pipeline_entries


class PipelineEntries(PipelineTable):
    """PipelineTable with PipelineEntry at the boundary instead of raw dicts."""

    STATUS_INDEX = "status-index"

    FIELDS = ["book_id", "subject_ids", "status"]

    def get_entry(self, book_id, fields=None):
        item = super().get_entry(book_id, fields)
        if item is None:
            raise LookupError(f"{book_id} has no pipeline entry")

        return PipelineEntry.model_validate({**item, "book_id": book_id})

    def get_indexes(self, status=None, subject_id=None) -> list[BookIndex]:
        if subject_id is not None:
            items = self.get_all_entries(
                ["book_id"], **self.build_subject_filter(subject_id, status)
            )
        elif status is None:
            items = self.get_all_entries(["book_id"])
        else:
            items = self.list_all(
                IndexName=self.STATUS_INDEX,
                KeyConditionExpression=Key("status").eq(status),
            )

        return sorted(BookIndex.parse(item["book_id"]) for item in items)

    def get_entries(self, book_ids) -> list[PipelineEntry]:
        keys = [{"book_id": str(BookIndex.parse(book_id))} for book_id in book_ids]
        items = self.batch_get_entries(keys, self.FIELDS)

        return sorted(
            (PipelineEntry.model_validate(item) for item in items),
            key=lambda entry: entry.book_id,
        )

    @staticmethod
    def build_subject_filter(subject_id, status=None):
        """Scan kwargs matching one subject, optionally narrowed to one status."""
        params = {
            "FilterExpression": "contains(#subject_ids, :subject_id)",
            "ExpressionAttributeNames": {"#subject_ids": "subject_ids"},
            "ExpressionAttributeValues": {
                ":subject_id": str(BookIndex.parse(subject_id))
            },
        }
        if status is not None:
            params["FilterExpression"] += " AND #status = :status"
            params["ExpressionAttributeNames"]["#status"] = "status"
            params["ExpressionAttributeValues"][":status"] = str(status)
        return params

    @staticmethod
    def get_modified_fields(entry: PipelineEntry):
        """The entry's non-key columns. Only what the caller set, so a write from one
        stage never clobbers another stage's columns."""
        attributes = entry.model_dump(exclude_unset=True, mode="json")
        attributes.pop("book_id", None)
        if "subject_ids" in attributes:
            attributes["subject_ids"] = set(attributes["subject_ids"])
        return attributes

    def put_entry(self, entry: PipelineEntry):
        return super().put_entry(entry.book_id, self.get_modified_fields(entry))

    @staticmethod
    def build_status_guard():
        """Only let a status write move a book forward, and never off a terminal state."""
        terminals = {
            f":terminal{n}": status.value for n, status in enumerate(TERMINAL_STATUSES)
        }
        condition = (
            "attribute_not_exists(#status) OR "
            f"(#status < :status AND NOT (#status IN ({', '.join(terminals)})))"
        )
        return condition, terminals

    def add_subject(self, book_id, subject_id):
        """Record another subject for a book already in the table.

        `put_entry` is a conditional create, so a book first listed under one subject
        would otherwise drop every later subject that lists it.
        """
        self.add_to_set(book_id, "subject_ids", {str(BookIndex.parse(subject_id))})

    def set_status(self, book_id, status: EntryStatus) -> bool:
        """Advance a book's status. False when the guard turned the write down.

        A stage advancing a status knows the book, not the subject it was listed under,
        so this writes the one column instead of going through `PipelineEntry`."""
        condition, condition_values = self.build_status_guard()
        return super().update_entries(
            book_id, {"status": status.value}, condition, condition_values
        )

    def update_entries(self, entry: PipelineEntry):
        """Returns False when the status guard turned the write down; True otherwise,
        including for writes that carry no status at all."""
        modified = self.get_modified_fields(entry)
        if not modified:
            return True

        condition, condition_values = (
            self.build_status_guard() if "status" in modified else (None, None)
        )
        return super().update_entries(
            entry.book_id, modified, condition, condition_values
        )
