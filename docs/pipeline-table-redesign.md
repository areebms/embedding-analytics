# PipelineTable: usage review and redesign proposal

This is a design critique of `PipelineTable` (`shared/tables/pipeline.py`) as it's actually used across the pipeline, plus a proposed target design. It's a proposal to react to, not an implemented change — no code or schema has been modified.

See also: [`pipeline.md`](pipeline.md) for how the stages are meant to work, [`alignment.md`](alignment.md) for the Procrustes alignment math.

---

## Context

`PipelineTable` is a single DynamoDB table, hash-keyed on `platform_data` (e.g. `gutenberg-1234`), with no sort key and no GSIs (confirmed from the moto table definition in `functions/publish/tests/conftest.py:118-128`). It's read/written by every stage:

```text
scrape --> tokenize --> train-kvector Map(N seeds) --> align-kvectors --> publish --> api
```

This sequence is actually orchestrated by a real AWS Step Functions state machine (`infra/step-function.template.json`), *not* by this table. `PipelineTable` is really a passive per-book record — S3 artifact locations plus a handful of scalar facts (author, title, published_year, disparity metrics) — that each stage reads to find its inputs and writes to publish its outputs. The table's name and its one `pipeline_status` field imply it's the pipeline's source of truth for stage progress; in practice it isn't, and that mismatch is the root of most of the problems below.

`docs/pipeline.md` even documents "Idempotent stages that skip completed work on retry or rerun" as a design feature (line 15) — true for `scrape` and `tokenize`, but not for `train-kvector` (finding 3 below), which is a real gap between the docs and the code.

---

## Findings

1. **`pipeline_status` never advances past scrape.** Scrape's two stages now advance it through `CREATED → SCRAPED_METADATA → SCRAPED_HTML` (with `SCRAPED_SKIPPED_NON_ENGLISH` and `SCRAPED_SKIPPED_NO_HEADINGS` as terminal branches ranked above every progress state), which is the pattern this proposal wants — but it stops there. No later stage sets `TOKENIZED`, `TRAINED`, `ALIGNED`, or `PUBLISHED`. Every later stage invents its own ad hoc "has this already run" check by testing presence of unrelated fields — there is no single field you can read to know where a book is in the pipeline.

2. **`s3_prefix_models` is a boolean wearing a string costume.** `create_book_centroid.py:94` writes it as `f"kvectors/{index}/"` — a constant fully derivable from `index`, carrying no information beyond "alignment ran". It's nonetheless used as the completion/eligibility gate in four separate places: `functions/api/src/app/list/services.py:5-8`, `functions/api/src/app/search/services/describe.py:32-36`, `functions/align-kvectors/src/create_corpus_centroid.py:61-63`, `functions/align-kvectors/src/app.py:47-51`. Each call site has to independently know "presence of this key = book is usable" and strip/ignore the value itself.

3. **`train-kvector` is invisible to the table.** It reads `s3_token_lemmas_key` but writes nothing back (`functions/train-kvector/src/main.py`) — no completion marker, no record of which seeds trained, no S3 key. `align-kvectors` instead discovers trained models via a raw S3 `ListObjects` under `kvectors/{index}/collected/` (`procrustes_utils.py`'s `S3Kvectors.load`). Two consequences:
   - You cannot query "how many seeds has this book trained" without hitting S3.
   - **No idempotency.** The model filename is `{seed}-{timestamp}-{random}.model` (`train-kvector/src/main.py:73-74`). The Step Functions Map state has a `Retry` block per seed (`step-function.template.json:91-103`); a retry after a late/duplicate success produces a second file for the same seed with a different name. `S3Kvectors.load` has no seed-based dedup, so a retried iteration can silently inject a duplicate "extra seed" into the GPA/centroid computation instead of being overwritten or ignored.

4. **`published_year` has no write path at all.** `publish_utils.save_metadata` (`publish_utils.py:192-198`) writes `author` and `title` from scraped metadata but never `published_year`. `BookResponse` (`functions/api/src/app/list/schemas.py:9,16`) requires it and will `KeyError` if it's ever missing — today this is silently papered over by hand-editing DynamoDB rows, with nothing in code documenting that this manual step exists or is required.

5. **Every "list books at stage X" is a full table Scan.** `get_all_entries` (`shared/tables/pipeline.py:47`) always scans; there is no GSI. `list/services.py`, `describe.py`, `create_corpus_centroid.py`, `align-kvectors/app.py`, and `publish_to_api.py` all scan the whole table and filter client-side for the field-presence flags above. Tolerable at current scale; will not scale gracefully as the corpus grows.

6. **No real stage-gating in the batch/backfill scripts.** `publish_to_api.py:20-26` calls `publish()` for *every* `platform_data` row regardless of whether that book was ever tokenized or aligned. `publish()` only guards on `s3_metadata_key` — "was scraped" (`publish_utils.py:239-243`). A book that was scraped but never finished tokenize/train/align will throw inside `RawPOSData.from_s3` (`publish_utils.py:68-76`, unguarded dict access on `s3_token_lemmas_key`) instead of skipping cleanly the way `scrape`/`tokenize` do. (See "Compatibility with Step Functions" below — this only bites the standalone scripts, not the real per-book Step Functions path.)

7. **Naming mismatch.** "Pipeline" + "pipeline_status" imply this table drives/tracks execution. It doesn't — Step Functions execution history is the actual source of truth for in-flight runs. This table is better described as a per-book asset table.

8. **Dead code, found in passing.** `build_corpus_handler` in `functions/align-kvectors/src/app.py:31` is commented `# Not being Used.` — an unwired handler that should either be removed or have its intended trigger documented.

---

## Compatibility with Step Functions

There are two distinct orchestration paths today, and they relate to the table very differently:

- **The real Step Functions flow** invokes each stage's Lambda `handler` in `app.py` (`scrape/src/app.py:11`, `tokenize/src/app.py:11`, `train-kvector/src/app.py:11`, `align-kvectors/src/app.py:11`, `publish/src/app.py:12`) in strict sequence per book. Step Functions *is* the state machine here — it already sequences scrape→tokenize→train→align→publish correctly via its own state transitions, and it never reads/writes DynamoDB directly (each Lambda does its own boto3 calls, invisible to the ASL definition). Adding a `stage` field to the table is purely additive observability on top of a flow Step Functions already gets right; it requires **zero changes to `step-function.template.json`**.

- **Standalone batch/backfill scripts**, wired to nothing in Step Functions: the `if __name__ == "__main__":` blocks in `create_book_centroid.py:103`, `create_corpus_centroid.py:50`, and `publish_to_api.py:12` (plus the dead `build_corpus_handler`). These loop over the *whole table* for bulk reprocessing. This is where the field-presence-as-gate antipatterns (findings 2 and 6) actually matter — the per-book Step Functions path already has correct sequencing for free. The `stage` field's main consumers are these three scripts.

- **The train-kvector idempotency fix is specifically a Step-Functions-retry fix.** The Map state's per-seed `Retry` block is exactly the mechanism that can produce a duplicate `{seed}-{timestamp}-{random}.model` file on a late-success retry today. Deterministic `{seed}.model` naming plus an idempotent `trained_seeds` set update make the design safe under exactly that retry behavior, rather than merely "not conflicting" with it.

Net: this redesign changes Lambda internals and the DynamoDB schema only — the state machine definition doesn't need to change.

---

## Redesign proposal

Keep the overall shape (one row per book, accumulating artifacts across a linear pipeline) — that's a reasonable fit for a few-thousand-row corpus. Fix the actual problem, which is that "what stage is this book at" is inferred from incidental field presence instead of stated explicitly.

- **Rename** `PipelineTable` → something like `BookAssetTable`, matching what it actually stores (per-book assets + facts), to stop implying it's the pipeline's execution state.

- **One explicit, monotonic `stage` field**, advanced by every stage in the same atomic `update_entries` call it already makes for its own outputs (the pattern `scrape` already uses correctly): `SCRAPED → TOKENIZED → TRAINED → ALIGNED → PUBLISHED`. Every consumer that currently tests "does field X exist" instead reads `stage`. This single change removes findings 1, 2, and most of 6.

- **`train-kvector` writes back.** Record trained seeds explicitly (e.g. a `trained_seeds` number-set, added via an idempotent `ADD`/`SET` per seed) and make the model's S3 key deterministic — `kvectors/{index}/collected/{seed}.model` instead of `{seed}-{timestamp}-{random}.model`. This makes retries overwrite instead of duplicate (fixes finding 3's idempotency bug) and lets `align-kvectors` check readiness from DynamoDB instead of an S3 listing.

- **Drop or demote `s3_prefix_models`.** It's derivable from `index` alone (`f"kvectors/{index}/"`); either stop storing it and have callers compute it, or keep it purely as data (never as the completion signal) now that `stage` is the real gate.

- **`publish_to_api.py` filters by `stage == ALIGNED`** before calling `publish()`, rather than relying on `publish()`'s single internal guard — makes "skip if not ready" consistent across every stage instead of only the first dependency.

- **`published_year`:** either derive it automatically at scrape time if Gutenberg metadata exposes a date (worth a quick check of `retrieve.get_metadata`'s fields), or — if it's genuinely not always derivable — make `BookResponse.published_year` `Optional[int]` so a missing value degrades gracefully instead of crashing `/books`, and add an explicit `needs_manual_review` style flag so the gap is visible in the data instead of silently patched by hand outside of code.

- **GSI on `stage`** so "all books at stage ≥ ALIGNED" becomes a Query. Flagged as a follow-up, not urgent at current scale — the Scan-and-filter pattern works fine today.

- **Remove or wire up** the dead `build_corpus_handler`.

### Field types

Not DDL — these are the DynamoDB attribute types the proposal *implies*, called out because three of the changes introduce types the table has never used.

Today every attribute is a string (`S`) or number (`N`) scalar — no sets, maps, or booleans:

| Field | Type | Written by |
|---|---|---|
| `platform_data` (hash key) | `S` | `put_entry` |
| `pipeline_status` | `S` (`CREATED`/`SCRAPED_METADATA`/`SCRAPED_HTML`/`SCRAPED_SKIPPED_NON_ENGLISH`/`SCRAPED_SKIPPED_NO_HEADINGS`) | scrape's two stages |
| `s3_metadata_key` | `S` | scrape (metadata) |
| `s3_html_key` | `S` | scrape (content) |
| `s3_standardized_html_key`, `s3_text_key` | `S` | scrape (standardize) |
| `s3_token_texts_key`, `s3_token_lemmas_key`, `s3_token_tags_key` | `S` | tokenize |
| `s3_prefix_models` | `S` (`"kvectors/{index}/"`) | align (`create_book_centroid.py:94`) |
| `mean_disparity`, `corpus_disparity` | `N` (`Decimal`) | align |
| `author`, `title` | `S` | publish (`publish_utils.py:197`) |
| `published_year` | `N` (int) | **nobody** — hand-edited (finding 4) |

(`variance`/`disparity`/`r_squared` in `publish_utils.py:22` are per-term Pinecone metadata, not columns on this table.)

The type-affecting decisions inside the proposal:

- **`stage` monotonicity is a type choice.** As a string enum (`S`), like today's `pipeline_status`, you *cannot* enforce "only advance forward" in a single atomic `ConditionExpression` — string comparison won't order `TOKENIZED` before `ALIGNED`. To actually guarantee the "monotonic" advance this proposal promises via `update_entries`, either add a numeric rank (`N`, e.g. `stage_rank`, guarded with `ConditionExpression: stage_rank < :new`) or accept that monotonicity is only convention enforced by call ordering. The plain string type can't guarantee it on its own.

- **`trained_seeds` introduces the table's first set type** — a DynamoDB number-set (`NS`), updated with `ADD`. That's the right fit: `ADD` on an `NS` is a genuine no-op for a member already present, which is exactly the retry-idempotency property finding 3 needs. Caveats worth stating: an `NS` can't be empty (the attribute is simply absent until the first seed is added), and it deserializes to a Python `set[Decimal]`.

- **`needs_manual_review` introduces the table's first `BOOL`.** Minor, but a new type for this table.

Two constraints that carry over to any new field:

- **Floats must be written as `Decimal`** (already the pattern at align time via `Decimal(str(...))`); easy to forget for a new `stage_rank` or a computed `published_year`.
- **GSI key type:** the proposed `stage` GSI is fine as `S`, but a GSI key must be a top-level scalar — so `trained_seeds` (`NS`) could never serve as one if that ever comes up.

---

## Not in scope for this pass

No code changes, no migration script, no schema DDL. A follow-up implementation plan (field renames, per-stage code edits, table recreation) would be a separate planning pass once the target design above is agreed on.
