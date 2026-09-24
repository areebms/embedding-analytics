# shared

*[Pipeline overview](../docs/pipeline.md) · [Project README](../README.md)*

The contract all six services share: three DynamoDB tables, one S3 bucket, and the book id
that keys them. Each service's Dockerfile copies this directory into its image, so a change
here is a change to — and a rebuild of — every service.

## Book id

`BookIndex` is a `str` subclass rendering as `gutenberg-<n>`: the pipeline table's partition
key, a column on every term row, and the stem of every S3 key.

- `BookIndex(3300)` builds one; `BookIndex.parse("gutenberg-3300")` recovers one from storage.
- `.source_id` is the bare integer — the form URLs and Gutenberg listings use.
- `__reduce__` makes copies and pickles round-trip through `source_id`. Without it a copy
  re-prefixes the id and every `batch_get_item` silently matches nothing.

## Pipeline table

`PIPELINE_TABLE`: one row per book, and the only place a status is written. Stages use
`PipelineEntries` (rows as `PipelineEntry`); `PipelineTable` is the raw-dict layer beneath it.

| Column | Type | Written by |
|---|---|---|
| `book_id` | str, partition key | `scrape` (`SUBJECT`), as a conditional create |
| `subject_ids` | set[str] | `scrape`, added to rather than replaced |
| `status` | str, `EntryStatus` | every pipeline stage except `publish` |
| `metadata` | map — `author`, `title`, `published_year` | `publish` |

- **`status-index`** (GSI on `status`) is how every `subject_id` path and standalone CLI
  finds work, via `get_indexes(status=...)`. Without it those calls fail with a validation
  error rather than returning nothing; the deployed table has not always had it.
- **Status only moves forward.** `build_status_guard` accepts a write if the row has no
  status, or if the new value sorts above the current one and the current one isn't
  terminal. Values are `<rank>[T]_<NAME>`: `0300_TOKENIZED` sorts above `0250_STANDARDIZED`
  as a plain string, and `T` (`0351T_EMBEDDINGS_CREATION_FAILED`) marks a terminal state.
  A refused write makes `set_status` return `False`; callers log it and carry on, since
  that is the normal re-run path.
- **`update_entries(entry)` writes only the fields the caller set** (`exclude_unset`), so one
  stage can't blank another's column.

S3 keys are not stored: each is a `PipelineEntry` property derived from the book id, so
adding an artifact means adding a property.

| Property | Key | Written by |
|---|---|---|
| `s3_metadata_key` | `metadata/{index}.json` | `scrape` |
| `s3_html_key` | `html/{index}.html` | `scrape` |
| `s3_book_pairs_key` | `standardize-html/books/{index}.json` | `standardize-html` (`SEND`) |
| `s3_standardized_html_key` | `html-standardized/{index}.html` | `standardize-html` (`RETRIEVE`) |
| `s3_text_key` | `text/{index}.txt` | `standardize-html` (`RETRIEVE`) |
| `s3_token_texts_key` | `token_texts/{index}.csv` | `tokenize` |
| `s3_token_lemmas_key` | `token_lemmas/{index}.csv` | `tokenize` |
| `s3_token_tags_key` | `token_tags/{index}.csv` | `tokenize` |
| `s3_embeddings_key` | `embeddings/{index}.npz` | `create-embeddings` |

## Book-term table

`BOOK_TERM_TABLE`: one row per (term, book), written by `publish`. Queries are answered from it.

| Column | Type | Contents |
|---|---|---|
| `term` | str, partition key | the lemma |
| `book_id` | str, sort key | which book's reading of it |
| `vector` | bytes | the term's vector, `float16` |
| `count_` | int | whole-book occurrences |
| `ilocs` | set[int] | every position of the lemma in the book's token stream |
| `tags` | set[str] | first-letter POS tags — `N`, `V`, `J`, `R`, `W` |

- `book_id-index` (GSI on `book_id`) loads a whole book in one query; the base key order
  serves the opposite question, one term across books.
- `EXCLUDED_POS_TAGS` lives here so the writer and the API mean the same letters. `GET /terms`
  drops a term whose tags are a subset of `{"J", "R", "W"}`; a term that is ever a noun or
  verb keeps its row.

## Corpus-term table

`TERM_CORPUS_TABLE`: one row per term across the corpus, written by `publish`.
**Nothing reads it.** `GET /terms` still groups every book's rows in memory, which is the
query this table was meant to replace. Either the API should read it or `publish` should
stop writing it.

| Column | Type | Contents |
|---|---|---|
| `partition` | str, partition key — always `#ALL` | one partition, so the whole vocabulary is a single query |
| `term` | str, sort key | the lemma |
| `book_ids` | set[str] | every book carrying it |
| `updated_at` | str, ISO-8601 UTC | last write |

- `add_book` / `remove_book` are atomic `ADD`/`DELETE` on the set, so concurrent publishes
  can't lose each other's membership.
- `remove_book_terms` is the republish path: it reads first, and deletes a term that loses
  its last book rather than leaving an empty row.
- The single `#ALL` partition makes the vocabulary one query but caps write throughput at
  one partition. That hasn't been the limit at current scale; if it becomes one, shard the
  partition key.

## How the API reads it

Only through `shared.tables.book_terms` (term rows) and `PipelineEntries` (metadata), so a
change here reaches the reader as well as the writers.

A book is queryable when it is at `EMBEDDINGS_CREATED` **and** has a `metadata` map.
`publish` writes no status, so the map is the only thing separating a published book from
a merely embedded one.

## S3 and session

- `shared/s3.py`: one bucket, `S3_BUCKET`, with loaders `load_text`, `load_json`, `load_csv`
  (a generator, so a book's tokens aren't held twice) and `load_file` (a context manager over
  a temporary file, used for the `.npz` archives). Uploads mirror them, each setting its own
  content type.
- `shared/session.py`: one lazily built boto3 `Session`, reused by warm containers. Tables and
  the S3 resource are singletons behind `get_*` functions; nothing is built at import.

## Tests

42 tests over the book id, pipeline entries, the corpus-term table and the S3 helpers. They
need `pytest`, `boto3` and `moto`, and run from the repo root:

```bash
python3.13 -m pytest shared/tests
```

**Nothing else runs them.** Each service's `pytest.ini` covers only its own `tests/`, only
the `api` image contains `shared/tests` (and it lacks `moto`), and the deploy gate skips
them. Otherwise a change here is covered only by the service suites that exercise it indirectly.
