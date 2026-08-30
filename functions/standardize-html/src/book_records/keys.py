"""Every S3 key this stage writes, built in one place.

The three shapes were built in three different modules, one of them as an inline
f-string, so the prefix had to be exported to reach them. Naming them together keeps
`S3_STANDARDIZE_PREFIX` private to the only code that interpolates it, and mirrors the
key builders in `shared.tables.pipeline_entries`.
"""

S3_STANDARDIZE_PREFIX = "standardize-html"


def book_pairs_key(index) -> str:
    return f"{S3_STANDARDIZE_PREFIX}/books/{index}.json"


def batch_index_key(batch_id: str) -> str:
    return f"{S3_STANDARDIZE_PREFIX}/batch-details/{batch_id}.json"


def batch_result_key(batch_id: str, custom_id: str) -> str:
    return f"{S3_STANDARDIZE_PREFIX}/batch-results/{batch_id}/{custom_id}.json"
