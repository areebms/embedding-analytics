from typing import NamedTuple

from shared.commons import BookIndex
from shared.tables.pipeline import PipelineTable


class BookMetadata(NamedTuple):

    book_id: BookIndex
    author: str
    title: str
    published_year: int | None


class BooksMetadataCache:
    """Every aligned book's pipeline metadata, scanned once per warm container."""

    FIELDS = ["platform_data", "author", "title", "published_year", "s3_prefix_models"]

    def __init__(self, table: PipelineTable):
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
    def _parse_entry(entry: dict) -> BookMetadata | None:

        # A book with no models built is not one the app can say anything about.
        if "s3_prefix_models" not in entry:
            return None

        published_year = entry.get("published_year")
        return BookMetadata(
            book_id=BookIndex.parse(entry["platform_data"]),
            author=entry.get("author", ""),
            title=entry.get("title", ""),
            published_year=None if published_year is None else int(published_year),
        )

    @property
    def book_ids(self) -> list[BookIndex]:
        return list(self.books_metadata)
