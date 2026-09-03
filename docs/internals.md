# Pipeline

How the system works, stage by stage. Each stage documents itself in its own
directory; this page is the map.

Content spanning more than one stage — orchestration, observability,
configuration, deployment — lives in [Operations](./operations.md).

Five containerized Lambda functions, each a single stage with its own resource
profile. Artifacts flow through S3 during training; `publish` flattens everything
into DynamoDB for sub-second API reads.

Every stage is idempotent: failed or repeated runs skip already-completed work
and resume safely.

| Stage | Input | Output |
|---|---|---|
| `scrape` | Gutenberg subject, then book ID | Pipeline rows, then HTML, text, and metadata in S3 |
| `tokenize` | Raw text | Token, lemma, and POS-tag CSVs in S3 |
| `create-embeddings` | Token lemmas | Centroid and replicate models in S3 |
| `publish` | Aligned models + token metadata | Term vectors, counts, POS tags, stability metrics in DynamoDB and Pinecone |
| `api` | HTTP request | Books, terms, semantic-drift, and parse responses as JSON |

```text
scrape → tokenize → create-embeddings → publish
```

The state machine templates and retry policy are cross-cutting orchestration
concerns — see
[Operations § Orchestration](./operations.md#orchestration), which also covers the
separate scrape machine that seeds a subject and scrapes every book in it.

| Stage | Documentation |
|---|---|
| `scrape` | [functions/scrape](../functions/scrape/README.md) |
| `tokenize` | [functions/tokenize](../functions/tokenize/README.md) |
| `create-embeddings` | [functions/create-embeddings](../functions/create-embeddings/) |
| `publish` | [functions/publish](../functions/publish/README.md) |
| `api` | [functions/api](../functions/api/README.md) |

---

## Scope of the claims

What the system is designed to claim, stated precisely.

- **The corpus is five books spanning 1767 to 1850, and the resolution follows
  from that.** Five works sample that period thinly, so any comparison rests on
  five independent readings of a term and no more. Denser sampling would extend
  the range of claims the design can support.
- **The confidence interval quantifies different things in the two request
  modes, and the two are not comparable in width.** Against a nominated book it
  is computed over the seed ensemble, so a tight interval means the relationship
  held across independent retrainings. Against the corpus the peers are the unit
  of replication, so it reports how far the books sat from one another — a claim
  about the corpus rather than about the training. Neither samples beyond the
  books the caller asked for.
