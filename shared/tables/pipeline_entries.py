from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

from boto3.dynamodb.conditions import Key


from shared.commons import BookIndex
from shared.session import get_session
from shared.tables.pipeline import PipelineTable

from enum import StrEnum
from typing import Annotated


class EntryStatus(StrEnum):
    CREATED = "CREATED"
    SCRAPED_METADATA = "SCRAPED_METADATA"
    SCRAPED_HTML = "SCRAPED_HTML"
    SCRAPED_SKIPPED_NON_ENGLISH = "SCRAPED_SKIPPED_NON_ENGLISH"
    SCRAPED_SKIPPED_NO_HEADINGS = "SCRAPED_SKIPPED_NO_HEADINGS"
    STANDARDIZE_SUBMITTED = "STANDARDIZE_SUBMITTED"
    STANDARDIZED = "STANDARDIZED"


BookIndexField = Annotated[
    BookIndex,
    PlainValidator(
        lambda value: value if isinstance(value, BookIndex) else BookIndex.parse(value)
    ),
    PlainSerializer(str, return_type=str),
]


def metadata_key(index: BookIndex) -> str:
    return f"metadata/{index}.json"


def html_key(index: BookIndex) -> str:
    return f"html/{index}.html"


class PipelineEntry(BaseModel):

    model_config = ConfigDict(extra="ignore")

    platform_data: BookIndexField
    pipeline_status: EntryStatus | None = None

    # Written by standardize-collect and by nothing else. Absent until a book
    # reaches STANDARDIZED, and left unset rather than defaulted by every other
    # stage, so get_modified_fields' exclude_unset keeps them out of those writes.
    s3_standardized_html_key: str | None = None
    s3_text_key: str | None = None

    @property
    def s3_metadata_key(self) -> str:
        return metadata_key(self.platform_data)

    @property
    def s3_html_key(self) -> str:
        return html_key(self.platform_data)


_pipeline_entries = None


def get_pipeline_entries():
    global _pipeline_entries
    if _pipeline_entries is None:
        _pipeline_entries = PipelineEntries(get_session())
    return _pipeline_entries


class PipelineEntries(PipelineTable):
    """PipelineTable with PipelineEntry at the boundary instead of raw dicts."""

    STATUS_INDEX = "pipeline_status-index"

    def get_entry(self, platform_data, fields=None):
        item = super().get_entry(platform_data, fields)
        if item is None:
            return None

        return PipelineEntry.model_validate({**item, "platform_data": platform_data})

    def get_indexes(self, status=None):

        if status is None:
            items = self.get_all_entries(["platform_data"])
        else:
            items = self.list_all(
                IndexName=self.STATUS_INDEX,
                KeyConditionExpression=Key("pipeline_status").eq(status),
            )

        return sorted(BookIndex.parse(item["platform_data"]) for item in items)

    @staticmethod
    def get_modified_fields(entry: PipelineEntry):
        """The entry's non-key columns. Only what the caller set, so a write from one
        stage never clobbers another stage's columns."""
        attributes = entry.model_dump(exclude_unset=True, mode="json")
        attributes.pop("platform_data", None)
        return attributes

    def put_entry(self, entry: PipelineEntry):
        return super().put_entry(entry.platform_data, self.get_modified_fields(entry))

    def update_entries(self, entry: PipelineEntry):
        modified = self.get_modified_fields(entry)
        if not modified:
            return
        super().update_entries(entry.platform_data, modified)
