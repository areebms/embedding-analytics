# publish

*Stage 5 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** NumPy

Flattens S3 artifacts into DynamoDB rows, which is what turns an embedded book into a
book the API can answer from. For each term present in both the embeddings archive and
the POS tag set, writes one row carrying the term vector (float16), the token occurrence
positions (`ilocs`), the POS tags and the whole-book count, and writes author and title
onto the pipeline row.

## How it is invoked

Payload and reply are the [standard batch contract](../../infra/README.md#how-a-stage-is-invoked);
`${ENV_PREFIX}-publish-trigger` supplies the `book_ids` form, carrying the ids
`create-embeddings` just announced. Nothing listens for a publish, so the chain stops
there.

## The intersection is the vocabulary

A book's published terms are the intersection of two sources that were built
independently:

- **the embeddings archive** (`embeddings/{index}.npz`) — every term that cleared
  `MIN_COUNT` and came out of the SVD with a direction
- **the POS data** (`token_lemmas` × `token_tags`) — every lemma tagged as a noun, verb,
  adjective or adverb somewhere in the book

A term in neither set is not published, so a lemma that is only ever a preposition never
reaches DynamoDB even though it may have a vector, and a term the embedding dropped is
not published on the strength of its tags. Both files are read per book and the row count
follows the intersection, which is logged beside both input sizes.

`ilocs` is the position of every occurrence of that lemma in the book's token stream,
counted across passages in reading order. It is what a passage-grained reader would need
to get from a term back to the text it occurred in; nothing in the API reads it yet.

## Republishing prunes

`remove_deprecated_terms` reads the book's existing rows before writing the new ones and
deletes the difference from both `BookTermTable` and the corpus vocabulary. Without it a
re-embedding that shrank a book's vocabulary would leave the dropped terms answerable
forever — they would keep their old vectors, and the corpus term table would keep listing
the book against words it no longer uses.

The status is deliberately untouched. There is no status past `EMBEDDINGS_CREATED`, so a
book stays at it whether or not it has been published, and a republish is always
available. What tells you a book is in the API is its terms being there.

## Running it

`publish.py` runs standalone and sweeps every book at `EMBEDDINGS_CREATED`, which needs
the `status-index` GSI on the table it is pointed at:

```bash
docker compose run --rm lambda-publish python publish.py
```

The `local` target clears the Lambda entrypoint, so no override is needed, and `src/` and
`shared/` are bind-mounted, so edits apply without a rebuild.

Its suite runs inside the image, against moto rather than AWS
([infra § Deploying](../../infra/README.md#deploying)):

```bash
docker build -f functions/publish/Dockerfile --target test -t publish-test . && docker run --rm publish-test
```
