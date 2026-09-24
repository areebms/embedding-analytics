# tokenize

*Stage 3 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** spaCy (`en_core_web_sm`), NLTK, WordNet

Turns a standardized book's text into the passage rows `create-embeddings` counts over.
Takes books at `STANDARDIZED` and leaves them at `TOKENIZED`.

Payload and reply are the [standard batch contract](../../infra/README.md#how-a-stage-is-invoked);
the standardize machine's handover supplies `book_ids`. Two rules are this stage's own:

- **A book with no passages raises** and is counted in `failed`. Empty CSVs would hand
  `create-embeddings` an empty book; the standardize output is what needs a look.
- **Tokenized books are announced as `Books Tokenized`**, which invokes `create-embeddings`,
  before the reply returns — only if at least one book made it. A bus that rejects the
  announcement raises: work nothing downstream hears about is a failure.

## What it does

1. Lemmatizes each passage with spaCy (NER disabled)
2. Normalizes American spelling to British ("labor" → "labour") against the vendored
   1796-word `src/data/american_spellings.json`. The corpus is British-side, so it never
   runs the other way
3. Collapses nouns onto a related verb ("production" → "produce"), except those listed in
   `src/data/ignored_nouns.txt`

| S3 artifact | Contents |
|---|---|
| `token_texts/{index}.csv` | Original tokens |
| `token_lemmas/{index}.csv` | Lowercased lemmas (the embedding input) |
| `token_tags/{index}.csv` | POS tags |

The keys are derived from the book id (`shared/tables/pipeline_entries.py`), so status is
the only column the stage writes.

## One row per passage

`text/{index}.txt` holds one passage per block-level element, separated by blank lines.
Row `n` of each artifact is passage `n`, so a lemma can always be traced back to the text
it came from.

The passage is also the co-occurrence unit: a `create-embeddings` context window never
crosses a row. Windows span sentences within a paragraph, but a heading — which rarely
ends in `.`, `!` or `?` — never welds onto the paragraph below it.

Passages are never split. spaCy's million-character cap is far beyond any paragraph, so
hitting it means `standardize-html` collapsed a whole subtree into one passage. The book
raises into `failed` so its markup gets checked, rather than being silently chunked.

## Sizing

Measured on `nltk.corpus.gutenberg` (1.1M characters, eight books), pinned to one core:

| | throughput | a 50-book backlog |
|---|---|---|
| one `Doc` at a time (before) | 71 Kchar/s | ~420s |
| `nlp.pipe(batch_size=32)` | 128 Kchar/s | ~235s |

- **Same output, 1.8× faster** — byte-identical across 244k tokens. 32 is the knee: 48 and
  64 are no faster, and spaCy's default of 1000 is slower.
- **1769 MB, down from 2048.** That is one full vCPU, and the stage is single-threaded, so
  the extra memory bought CPU it couldn't use. Peak RSS is ~520 MB.
- **`n_process` doesn't pay.** Child processes pickle their `Doc`s back one at a time: two
  vCPUs gave 1.12× for 2× the memory, three gave 1.33× for 3×.
- **900s is the ceiling.** A full `MAX_BOOKS_PER_SUBJECT` backlog fits (~440s even at
  *Moby-Dick* length). A timeout leaves finished books at `TOKENIZED` and the rest at
  `STANDARDIZED`; re-run to recover.
- **The parser stays**, though nothing reads dependencies and dropping it would be another
  1.8×. Without the parse, `attribute_ruler` tags auxiliaries `AUX` instead of `VERB`, they
  fall out of `SPACY_TO_WORDNET`, and "had" stops lemmatizing to "have" (0.32% of tokens).
  That changes the corpus, not just the speed. `ner` is the only component disabled.

## Spelling normalization

The lookup is on the whole token, so "laboratory" and "collaborate" are left alone (a
substring replace would give "labouratory"). Two mappings merge senses because the list
can't see part of speech: `practice → practise` and `program → programme`. One merged
vector beats two split ones here. If a book ever shows a problem, add an override list
like `ignored_nouns.txt` rather than patching the vendored file.

## Credits

The spelling list is [hyperreality/American-British-English-Translator][translator]
(MIT © 2016), vendored verbatim. Re-pull it with:

```bash
curl -sSL -o functions/tokenize/src/data/american_spellings.json \
  https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json
```

Use `american_spellings.json`, not `british_spellings.json` — the latter maps British to
American and would normalize the wrong way.

[translator]: https://github.com/hyperreality/American-British-English-Translator

## Running it

To see what one book tokenizes to, without going through an event:

```bash
docker compose run --rm lambda-tokenize python main.py --platform gutenberg --id 3300
```

`app.py`, `main.py`, `src/data/ignored_nouns.txt` and `shared/` are bind-mounted; an edit
to `tokenize_text.py` or `american_spellings.json` needs a rebuild.

Its suite runs inside the image ([infra § Deploying](../../infra/README.md#deploying)):

```bash
docker build -f functions/tokenize/Dockerfile --target test -t tokenize-test . && docker run --rm tokenize-test
```
