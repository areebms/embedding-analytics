from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineEvent:

    source: str
    detail_type: str
    detail_keys: frozenset[str]


SUBJECT_BOOKS_SCRAPED = PipelineEvent(
    source="embedding-analytics.scrape",
    detail_type="Subject Books Scraped",
    detail_keys=frozenset({"subject", "book_ids", "scrape_execution"}),
)

BOOKS_STANDARDIZED = PipelineEvent(
    source="embedding-analytics.standardize",
    detail_type="Books Standardized",
    detail_keys=frozenset(
        {"batch_id", "book_ids", "standardized", "standardize_execution"}
    ),
)
