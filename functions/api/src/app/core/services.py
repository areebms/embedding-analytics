from typing import NamedTuple

from shared.commons import BookIndex
from shared.tables.pipeline_entries import EntryStatus, PipelineEntries, PipelineEntry


class BookMetadata(NamedTuple):

    book_id: BookIndex
    author: str
    title: str
    published_year: int | None


class BooksMetadataCache:
    """Every aligned book's pipeline metadata, scanned once per warm container."""

    FIELDS = ["book_id", "status", "metadata"]

    def __init__(self, table: PipelineEntries):
        self.table = table
        self._books_metadata: dict[BookIndex, BookMetadata] | None = None

    def __repr__(self) -> str:
        # The /books and /terms responses are keyed on this dependency's repr by
        # fastapi-cache's default key builder, so it must not carry an address.
        return f"{type(self).__name__}()"

    @property
    def books_metadata(self) -> dict[BookIndex, BookMetadata]:
        """Loaded on first access, not on construction -- a request that never
        asks about books never pays for the scan."""

        if self._books_metadata is None:
            entries = self.table.get_all_entries(self.FIELDS)
            self._books_metadata = {
                data.book_id: data
                for data in map(self._parse_entry, entries)
                if data is not None
            }
        return self._books_metadata

    @staticmethod
    def _parse_entry(item: dict) -> BookMetadata | None:

        entry = PipelineEntry.model_validate(item)

        if entry.status != EntryStatus.EMBEDDINGS_CREATED or entry.metadata is None:
            return None

        return BookMetadata(
            book_id=entry.book_id,
            author=entry.metadata.author,
            title=entry.metadata.title,
            published_year=entry.metadata.published_year,
        )

    @property
    def book_ids(self) -> list[BookIndex]:
        return list(self.books_metadata)
