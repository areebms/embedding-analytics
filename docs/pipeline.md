# Pipeline

How a book goes from Project Gutenberg to something the website can query. Each step
has its own README; how the steps are wired together, recovered and deployed is in
[infra](../infra/README.md).

## What happens to a book

A book is found on Project Gutenberg and downloaded. Claude reads its headings and marks
which are chapters and which are sections, and the text is flattened into plain
passages. Each passage is split into words, and each word is reduced to its dictionary
form, so *commodities* counts as *commodity*. From which words appear near which in that
one book, the book gets its own set of word vectors: each word's vector records the
company it keeps. The vectors go into a database, and the website's queries are answered
from it.

| Step | What it does | Details |
|---|---|---|
| 1. Download | Finds the books under a Gutenberg subject and downloads each one's text and catalogue record | [scrape](../functions/scrape/README.md) |
| 2. Tidy | Has Claude label every heading, then keeps a clean plain-text copy | [standardize-html](../functions/standardize-html/README.md) |
| 3. Split into words | Splits the text into words, reduces each to its dictionary form, and tags its part of speech | [tokenize](../functions/tokenize/README.md) |
| 4. Build word vectors | Builds the book's word vectors from which words appear near which | [create-embeddings](../functions/create-embeddings/README.md) |
| 5. Publish | Loads the vectors and word counts into the database | [publish](../functions/publish/README.md) |
| 6. Answer queries | Answers the website's queries across every published book | [api](../functions/api/README.md) |

## How the steps hand over

Each step announces when it has finished a set of books, and the next step starts on
exactly those books. No step needs to know which one comes after it, so a step can be
changed or re-run without touching the others.

A book that fails at a step stays where it is, and the rest carry on. Every step skips
work that is already done, so running one again is safe, and an announcement that
arrives twice does not pay for the same work twice.

Publishing is the last step. A book appears on the website once it has been published.

## Where it runs

Every step runs on AWS as a small on-demand function. Files are kept in S3, and the
published vectors in DynamoDB, which is what lets a query be answered in under a second.
All six steps, the query service included, are deployed together.

## More detail

- [infra](../infra/README.md) — how the steps are wired together, how a stalled run is
  recovered, and how everything is deployed
- [shared](../shared/README.md) — the database tables every step agrees on
- each step's own README, linked in the table above
