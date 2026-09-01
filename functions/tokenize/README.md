# tokenize

*Stage 3 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** spaCy (`en_core_web_sm`), NLTK, WordNet

Turns a standardized book's text into the passage rows `train-kvector` trains on. Takes
books at `STANDARDIZED` and leaves them at `TOKENIZED`.

The payload names the work, as in `standardize-html`: `book_ids` is the standardize
machine's handover, `subject_id` is the same work named by subject for a hand re-run
capped at `MAX_BOOKS_PER_SUBJECT`, and exactly one of them must be present — a payload
naming no books is refused rather than read as "everything at `STANDARDIZED`". One
invocation takes the whole list, so the model is loaded once and a book that raises is
counted in `failed` and left at `STANDARDIZED` rather than ending the run. A book whose
text holds no passages raises for that same reason: three 0-row CSVs at `TOKENIZED`
would hand `train-kvector` a book of nothing, so it is counted in `failed` and the
standardize output gets looked at. Out: `{ found, tokenized, failed }`, where `found`
is how many of the named books were at `STANDARDIZED`, not how many were named.

`book_ids` is sized by whoever sent it; a subject is however many books the table holds
at `STANDARDIZED`, which is why only that path is capped. The overflow keeps
`STANDARDIZED` and comes back on the next run, so re-invoking by subject is the drain.
Ids are sorted, so which books make the cut is the same answer twice rather than
whatever the Scan returned first.

That list is the only way in. This stage no longer queries the status index for its own
work, so a book stranded by a failed run rejoins only when something names it again —
re-run the execution rather than waiting for a later one to sweep it up.

## One row per passage

`text/{index}.txt` is one passage per block-level element, passages separated by a blank
line. `yield_passages` hands each passage to spaCy separately and `upload_passage_data`
writes one CSV row per passage, so **row `n` is passage `n`** — the three artifacts stay
token-aligned with each other and positionally aligned with the source passages. That
back-pointer is what a passage-grained reader needs to get from a lemma back to the
text it occurred in.

The passage is also the training unit. A row of `token_lemmas/{index}.csv` is what gensim
treats as one sentence, and a context window never crosses a row — so windows now span
sentence boundaries *inside* a paragraph, which is the point, while a heading still
cannot be trained against the paragraph below it. A heading rarely ends in `.`, `!` or
`?`, so when a segmenter sees the whole book as one string it welds the two: on a book of
24 passages the implementation before this one — `sent_tokenize` over the whole book,
chunks rejoined with a space — produced 8 rows spanning more than one passage, one per
chapter.

A passage is never split. Passages are paragraphs and spaCy's cap is a million characters,
which no paragraph reaches, so the stage passes each passage to spaCy whole and one
doc comes back per passage. If a book ever does produce an oversized passage it will raise
in spaCy and be counted in `failed` — which is the wanted outcome, because the only
way to reach that size is `standardize-html` collapsing a whole subtree into one passage,
and a book that hits it needs its markup looked at, not its text silently chunked.

## The rest

1. Lemmatizes with spaCy (NER disabled for speed)
2. Normalizes American spelling to British against a vendored 1796-word list,
   `src/data/american_spellings.json` ("labor" → "labour", "organize" → "organise") — the
   corpus is British-side, so the map runs American → British and never the reverse
3. Applies domain-specific aggressive lemmatization: nouns with derivationally
   related verbs collapse to the verb form ("production" → "produce"), controlled
   by a curated `src/data/ignored_nouns.txt` override list

| S3 artifact | Contents |
|---|---|
| `token_texts/{index}.csv` | Original tokens, one passage per row |
| `token_lemmas/{index}.csv` | Lowercased lemmas (training input) |
| `token_tags/{index}.csv` | POS tags |

All three keys are derived from the book id (`shared/tables/pipeline_entries.py`), so the
stage no longer mirrors them into the row as `s3_token_*_key` columns — status is the one
column it writes.

`train-kvector` and `publish` have not been migrated: both still read those columns off
the row (`functions/train-kvector/src/main.py`, `functions/publish/src/publish_utils.py`)
rather than deriving the keys. Rows written before this change keep their columns, but a
book tokenized after it gives those two nothing to read — `train-kvector` reports it as
untokenized and exits, `publish` raises a `KeyError`. They need to derive the keys from
the book id before either runs on a newly tokenized book.

## Sizing, measured

Profiled on the Gutenberg corpus (`nltk.corpus.gutenberg`, 1.1M characters across eight
books), pinned to one core so the numbers mean what Lambda would give:

| | throughput | a 50-book backlog |
|---|---|---|
| a `Doc` at a time (before) | 71 Kchar/s | ~420s |
| `nlp.pipe(batch_size=32)` | 128 Kchar/s | ~235s |

**1.8× on the same output** — the three artifacts are byte-identical across 244k token
tuples, because nothing about the model or the pipeline changed, only how many passages
it is handed per call. 32 is the knee: 48 and 64 measure the same and only hold more
`Doc`s live, and spaCy's own default of 1000 is worse than 32 on a book of paragraphs.

Sizing is now 1769 MB, down from 2048. Lambda scales CPU with memory and hands out a
full vCPU at 1769 MB, and this stage cannot use more than one: it runs in a single OS
thread, on thinc's `NumpyOps`, against a blis built without threading. The extra 279 MB
was buying CPU no part of the process could reach. Capacity is not the constraint either
— peak RSS is ~520 MB on the longest book in the corpus.

Going *higher* only pays with `nlp.pipe(n_process=)`, and that measured badly: the child
processes pickle their `Doc`s back and the parent deserializes them one at a time, so two
vCPUs bought 1.12× the speed for 2× the memory — 1.8× the bill. Three bought 1.33× for
3×. The stage stays single-process.

900s is the Lambda ceiling. A full `MAX_BOOKS_PER_SUBJECT` backlog now fits with room
(~440s even if every book were the length of *Moby-Dick*); a timeout still leaves the
finished books at `TOKENIZED` and the rest at `STANDARDIZED`, so recovery is to re-run.

One thing the profile did *not* buy: disabling the parser. Nothing here reads a
dependency, so it looks free to drop — but `attribute_ruler` refines `pos_` from the
parse, and without it auxiliaries move from `VERB` to `AUX`, fall out of
`SPACY_TO_WORDNET`, and stop lemmatizing: "had" stays "had" instead of collapsing to
"have". That is 0.32% of tokens for another 1.8×, and it is a change to what the corpus
trains on rather than a speedup, so the parser stays. `ner` remains the only component
disabled.


## Spelling normalization

The whole word list replaced a `"labor" → "labour"` special case, so a book printed in the
US no longer trains `organize` and `organise`, or `favor` and `favour`, as two vectors. The
lookup is on the whole cleaned token — the substring replace this stage used to do turned
"laboratory" into "labouratory" and "collaborate" into "collabourate", and each non-word was
then lemmatized and trained as a term of its own.

Two mappings are worth knowing about for this corpus. `practice → practise` collapses both
senses onto the verb form, because British keeps `practice` as the noun and the list cannot
see part of speech; `program → programme` does the same for the computing sense. Both are
acceptable here — one merged vector beats two split ones — but neither is invisible, and the
fix if a book ever shows a problem is an override list like `src/data/ignored_nouns.txt`,
not a patch to the vendored file.

## Credits

The spelling list is [hyperreality/American-British-English-Translator][translator]
(MIT © 2016). Its `data/american_spellings.json` is vendored verbatim at
`src/data/american_spellings.json`; re-pull it with:

```bash
curl -sSL -o functions/tokenize/src/data/american_spellings.json \
  https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json
```

Take `american_spellings.json` and not `british_spellings.json` — the latter is keyed by the
British spelling (`"labour": "labor"`) and would normalize the corpus the wrong way.

[translator]: https://github.com/hyperreality/American-British-English-Translator
