# User Guide

Embedding Analytics helps researchers explore how classical economists used concepts across full texts. Search a term, compare authors, or build a vector expression to test a conceptual relationship like `labour + (productive - unproductive)`.

The app covers texts by Adam Smith, David Ricardo, John Stuart Mill, James Steuart, and Frederic Bastiat from Project Gutenberg. Each query runs against the selected texts and returns related terms, similarity scores, and confidence intervals.

---

## What you can do

- Search a single concept like `market`, `labour`, or `rent`
- Compare how different authors use the same concept
- Build vector expressions with `+`, `-`, and parentheses
- Type plain English in Describe mode to generate an editable expression
- Read confidence intervals to judge whether a result is stable or noisy

---

## Select authors

Use the author filter to choose which texts to compare.

Selecting one author lets you inspect the vocabulary of a single text. Selecting multiple lets you compare how the same query behaves across writers. Books are color-coded chronologically: warmer tones for earlier writers, cooler for later ones.

Every selected author is queried independently, so differences in the chart reflect actual differences in usage patterns within each text.

---

## Search a concept

Type a word into the search box and select from the autocomplete dropdown. The dropdown contains terms drawn directly from the corpus.

| Query | What it explores |
|---|---|
| `market` | Terms used in similar contexts to "market" |
| `labour` | Labour's surrounding vocabulary across authors |
| `rent` | Concepts associated with rent, land, and income |

Results show contextual proximity, not dictionary meaning. `labour` and `wage` may score highly because the authors discuss them in similar contexts, not because they are synonyms.

---

## Use vector expressions

Vector expressions let you move from simple search to targeted conceptual probing.

### Adding terms

Words can carry multiple meanings, so a single-term query may return unrelated concepts. Use `+` to pull results toward a more specific meaning.

```text
capital + profit
```

This pulls "capital" toward its economic sense by combining it with "profit", filtering out the geographical meaning.

### Subtracting terms

Word2Vec places antonyms close together because they appear in overlapping contexts. A single-term query may return opposites alongside related terms. Use `-` to push an unwanted direction away.

```text
productive - unproductive
```

This pushes the query away from "unproductive" to isolate what is distinctive about "productive".

### Combining with parentheses

Without parentheses, `labour + productive - unproductive` averages all three directions together. With parentheses, `labour + (productive - unproductive)` isolates the productive/unproductive contrast first, then adds "labour". The parenthesized version surfaces terms similar to productive labour but not unproductive labour.

| Operator | Meaning | Example |
|---|---|---|
| `+` | Blend two concepts | `labour + wage` |
| `-` | Create a contrast direction | `productive - unproductive` |
| `()` | Group a sub-expression | `labour + (productive - unproductive)` |

Reversing the contrast with `labour + (unproductive - productive)` tilts the other way and should surface a different vocabulary cluster.

Expressions are best understood as contextual directions, not ordinary arithmetic. Each sub-expression is normalized before being combined, so a contrast like `(productive - unproductive)` acts as a direction in the semantic space rather than a measured quantity.

---

## Use Describe mode

Describe mode lets you type a plain-English query and have the backend convert it into a vector expression.

Example:

```text
productive vs unproductive labour
```

The app converts the request into `labour + (productive - unproductive)`, switches back to vector mode, and shows the interpreted expression as editable chips. You can review, adjust terms, and rerun the query.

If a term in your description is not found in the corpus, the closest match is substituted automatically. A warning appears below the input showing what changed.

Describe mode is a translation step, not a chatbot. It does not keep conversational state.

---

## Read the chart

The main chart is a horizontal dot-and-interval view.

| Element | Meaning |
|---|---|
| Row | A related term returned by the query |
| Dot | One author's similarity score for that term |
| Horizontal whisker | 95% confidence interval across model runs |
| Further right | Stronger contextual association |

Look for patterns rather than isolated terms. If several production-related words cluster together for one query, the result may indicate a coherent vocabulary group.

Divergence across books on the same row indicates a measurable difference in how the term is used. That divergence is the starting point for analysis: is the meaning of the word shifting over time? Is the author engaged in a different debate?

---

## Read confidence intervals

Confidence intervals show whether a result was stable across the ensemble of Word2Vec models.

| Pattern | Interpretation |
|---|---|
| Tight interval | The relationship was consistent across training runs |
| Wide interval | The score varied across runs; treat cautiously |
| Non-overlapping intervals across authors | Stronger evidence of genuinely different usage |

Wide intervals are especially important for contrast expressions. Close antonyms like `productive` and `unproductive` often live in the same semantic neighborhood. Their difference can be small and noisy, so the interval helps reveal whether the contrast was reliably learned from the text.

---

## Use the table

The table gives exact values behind the chart: similarity score, confidence interval bounds, word count, and part-of-speech tags.

When a single book is pinned, a relative emphasis column (z-score) appears, showing where that author's usage departs from the group.

Use the chart to see the pattern, then use the table when you need precise values or want to sort by a specific metric.

---

## Suggested workflow

1. Start with a broad concept: `labour`, `market`, `value`, or `rent`
2. Compare authors to see whether the same concept has different neighborhoods
3. Narrow with addition: `capital + profit`, `rent + land`, or `value + exchange`
4. Test a contrast: `labour + (productive - unproductive)`
5. Reverse the contrast and compare the vocabulary shift
6. Use wide confidence intervals as a signal to verify results through close reading

---

## How to interpret results

Embedding Analytics generates hypotheses, not final conclusions.

The app reads entire texts statistically and surfaces aggregate usage patterns. It can reveal relationships that are difficult to spot through manual reading alone, such as one author associating `value` more closely with monetary vocabulary while another associates it with labour vocabulary.

Those patterns should be starting points for close reading. The strongest results are ones that are coherent, stable across confidence intervals, and historically interpretable in the source texts.
