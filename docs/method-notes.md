# Method notes

What the results can and cannot claim, why no score comes with a margin of error, and
what the differences between books turn out to be. The figures behind each point are in
[Technical detail](#technical-detail).

## Scope of the claims

- **24 books, 1752 to 1927.** That samples the period thinly. A comparison rests on at
  most 24 readings of a word, and usually fewer, since not every book uses every word.
- **Each book is read on its own.** Its word vectors come only from its own text; nothing
  is averaged or aligned across books. A comparison sets two separate readings of a word
  side by side.
- **Each score is a single value**, with no margin of error. The next section says why.

---

## No confidence intervals

A confidence interval says how far an estimate might be from the true value in some
larger population. There are two reasons not to give one here, and either is enough.

**A book is not a sample.** Smith wrote one *Wealth of Nations*. It is not one draw from
the books he might have written, and there is no larger population these texts stand for
(Koplenig 2019; Berk, Western & Weiss 1995).

**The biggest uncertainty is in the method, not the text.** Removing a random tenth of a
book's passages moves a score by 27% on average. Changing how many surrounding words
count as a word's context, from 5 to 20, moves it by 32%, and reorders which books look
closest in every query tested. An interval built by resampling passages would cover the
first and miss the second, and so look more certain than the result is.

Instead, a result is checked for whether it holds up when those settings change. One that
holds is reported; one that does not is flagged or withheld.

## How far to trust a result

Every book was rebuilt ten times with a random tenth of its passages removed, and ten
queries were rerun: *value, supply, demand, price, labour, land, capital, wage, rent,
profit*. The lists were five terms long at the time.

- **Consistent terms are sturdy.** The first four survive at least 95% of the time, the
  fifth 78%.
- **Only the top of the contested list is sturdy.** The first two survive over 90% of the
  time, the third about 70%, and the fourth and fifth only about half the time.
- **A book that rarely uses the query is the weakest link.** Below about 40 uses, a book's
  picture of the word is unsettled. Books that use a word 10–20 times agree with the
  others less than half as well as books that use it 320 times or more, and cutting a
  word's uses down to that level reproduces most of the gap. The low count is the cause,
  not the book or its author. Each book's count is returned with every result.
- **No per-word fix was found.** Fewer dimensions, one vote per author and subtracting
  estimated noise were all tried, and none made the contested list steadier. A word's
  vector is built from the whole book, so removing passages that never mention it still
  moves it.

---

## What the variation is: author, not date

The differences between books do not follow time. Books far apart in date agree as well
as books close together, in all ten queries tested. The differences follow the author.

| Possible source | What was found |
|---|---|
| Date of publication | No overall effect. The one apparent exception, *capital*, is Hume and Steuart using it for a capital city. A few single word pairs do trend with date, such as *supply* and *utility*. |
| Author | The strongest signal. Two books by the same author agree far more than their size predicts. |
| How often the book uses the word | Large. A book that uses a word rarely gives an unreliable picture of it. |
| Type of book (treatise or not) | No effect once book size is allowed for. |
| Senses found in one book only | *capital* as a city; *demand* as bank deposits in Fetter. |

**What other research says.** Linguists call this *synchronic variation*: differences
between groups of writers, as opposed to change over time, and a normal object of study
(Weinreich, Labov & Herzog 1968; Azarbonyad et al. 2017; Gonen et al. 2020). Individual
writers' word use is distinctive and consistent (Zhu & Jurgens 2021; Welch et al. 2020),
which matches the strong same-author agreement here. On small texts, a rarely used
word's nearest terms are unstable (Wendlandt et al. 2018; Antoniak & Mimno 2018). A
measure like this one captures what a word is associated with, not what it means
(Hamilton, Leskovec & Jurafsky 2016b): Ricardo's *land* sits near *quality* and
*inferior* because he discusses rent through the quality of land, not because *land*
means something else to him. Words stand in for concepts only through a reader's
interpretation (Wevers & Koolen 2020). One check the literature recommends, shuffling
passages between books to see how much of the author difference survives, has not been
run (Dubossarsky et al. 2017).

**For the product:** the right name is *variation*, not *drift*, which implies a change
over time these books do not show.

**Not found in the literature:** word-vector comparisons of individual authors within
one field; a study separating author from date within one period; word-vector work on
the history of economic thought. The closest precedent in the history of economics
counts word frequencies instead (Ballandonne & Cersosimo 2023).

## What would change this

- A much larger corpus, where the swings above are small next to the differences being
  claimed.
- A clear larger population the books could stand for, such as each book as one of many
  by an author or from a period, with enough books in each.
- A margin of error that comes from how the vectors are built rather than from
  resampling the text.

---

## Technical detail

### No confidence intervals, in full

`/semantic-drift` reports point estimates with no `ci` field. The replicate machinery
that produced it — `N_BOOTSTRAP = 15` in `create-embeddings`, the per-replicate
`vectors` column, `bootstrap_half_width` in the API — was removed with it. There are two
independent reasons, either sufficient on its own.

**A book is not a sample.** A confidence interval claims coverage of a parameter in a
population. Smith wrote one *Wealth of Nations*; it is not a draw from a distribution of
books he might have written, and no wider population can be named that these texts
generalise to. Koplenig (2019) argues the randomness assumption behind inferential tests
is not met by naturally occurring language data; Berk, Western & Weiss (1995) give the
general form — inference on a complete population is meaningful only if a wider target
population can be named and argued for.

**What is reported instead:** a robustness audit deciding which comparisons are stable
enough to publish: one that holds across `WINDOW` 5–20 and across passage deletion is
reported; one that does not is flagged or withheld. It tells a reader not how wide the
answer is, but whether it is an answer.

### Perturbations

**The interval measured one source of variation out of several.** It resampled passages
and held every modelling choice fixed. On gutenberg-3300, -33310 and -30107, over
`value`/`labour`/`price` and all three book pairs (top-75 local window):

| perturbation | mean swing in the reported value | pair rankings that flip |
|---|---|---|
| drop 10% of passages | 27% | 2 of 3 queries |
| context window 5 → 20 | 32% | 3 of 3 queries |

`WINDOW = 10` is an undocumented constant, and moving it across a defensible range
changes the answer more than resampling does. A band that omits a source of equal size
implies the uncertainty has been characterised when it has not. Deletion-based bands are
no better: their width is set by a hand-chosen fraction (±0.095 at 20%, ±0.054 at 5%),
and Yu's delete-d rescale (2013) does not remove that dependence here — the scatter
falls off as `f^0.41`, not the `f^0.50` it assumes.

### What PPMI/SVD removed

The interval dates from Word2Vec seed ensembles. Of the three perturbations Antoniak &
Mimno (2018) test, PPMI/SVD eliminates two:

| perturbation | Word2Vec | this pipeline |
|---|---|---|
| random initialisation | large | none — SVD is deterministic |
| passage order | substantial | **exactly zero**, verified |
| corpus composition | large | unchanged — 27% swing |

Order-invariance is exact: `term_cooccurrence_in_window` counts only pairs inside a row,
so the matrix is a sum over rows, and the counts are whole numbers in float64. On
gutenberg-33310, two shuffles and a full reversal give `max |vector difference| =
0.000e+00`. The pipeline has no token subsampling and `CDS = 0.75` is deterministic, so
Antoniak & Mimno's one residual PPMI source does not apply. What survives is which
passages are in the file — real, but not a sampling problem.

Reporting an interval is the exception in comparable work. Hamilton, Leskovec &
Jurafsky (2016b) report none on their local-neighbourhood measure itself; Antoniak & Mimno report variability and overlap, not intervals;
Kozlowski, Taddy & Evans (2019) give a 90% bootstrap over 20 corpus resamples;
Vallebueno et al. (2024) derive variances from reconstruction error, explicitly not
from collections of documents. Rettenmeier (2020) sums up the field: "the current
practice in research is to provide a single score, without information on its variance."

### Passage-deletion audit

Run 2026-09-18 on the full `/semantic-drift` response. Each of the 24 economics books was
rebuilt from its S3 token lemmas with `create-embeddings`' own `get_embedding_data`
(matching the stored vectors to within 2e-4 in cosine, float16 rounding), then rebuilt 10
times with a random 10% of passages deleted. Queries: *value, supply, demand, price,
labour, land, capital, wage, rent, profit*. The replicates measure **sensitivity** —
whether a returned term rests on the whole text or a few passages — not sampling error.

**Which slots survive** a 10% deletion:

| slot | 1 | 2 | 3 | 4 | 5 | slots changed per deletion |
|---|--:|--:|--:|--:|--:|--:|
| consistent | 100% | 95% | 100% | 96% | 78% | 0.31 |
| contested | 93% | 91% | 72% | 46% | 44% | 1.54 |

That is about twice what leaving out a whole book does (0.78). Contested slots 4 and 5
are coin flips; the top two hold.

**Occurrences drive it.** In 28 book–query pairs with 300–1,400 uses, only the query's
own tokens were deleted down to a fixed count and the book rebuilt:

| uses kept | agreement with the book's full-text profile | nearest 50 unchanged | agreement with other books, as a share of full |
|--:|--:|--:|--:|
| 10 | 0.47 | 38% | 54% |
| 20 | 0.54 | 40% | 59% |
| 40 | 0.65 | 52% | 77% |
| 80 | 0.75 | 64% | 87% |
| 160 | 0.85 | 76% | 98% |

Real books using a query 10–20 times agree with the rest at 0.095, against 0.237 at 320
or more; the reduction reproduces 75–80% of that gap, so a low count causes most of it,
not the book or the author.

**Sensitivity cannot be corrected term by term.** Nothing tried inside the API made the
contested list less sensitive:

| change | contested slots changed per deletion |
|---|--:|
| none | 1.54 |
| SVD cut to 50 / 25 / vocabulary ÷ 15 dimensions | 1.78 / 2.39 / 1.82 |
| one vote per author | 1.61 |
| variance minus noise predicted from counts | 1.56 |
| variance minus noise measured from the replicates (jackknife-scaled ×9) | 1.85 |
| jackknife (leave-one-book-out minimum) variance | 1.51 |

Counts, pair co-occurrence and passage dispersion predict a term's sensitivity at R²
0.25, against a ceiling of 0.49; counts alone reach 0.07. A term's vector comes from one
SVD of the whole book, so deleting passages that never mention it still moves it.

**An occurrence gate helps a little.** Dropping a book from a query below 15 uses
removes about one book per query and lowers contested slots changed from 1.54 to 1.39,
leaving leave-one-book-out sensitivity unchanged (0.78). From 20 uses up, that
sensitivity rises.

**What this supports reporting:** contested slots 1–2 as stable, slot 3 as likely,
slots 4–5 withheld or marked, and a book's point as unsettled below about 40 uses of the
query, which the response carries as `occurrences`.

### What would reverse this

- A corpus large enough that the swings above fall well below the differences being
  claimed. At 24 books they do not.
- A defensible superpopulation — each book as one draw from an author's or a period's
  output, with enough books per stratum. The peer-axis interval
  (`standard_error_half_width(per_peer_means)`) is the piece worth revisiting first.
- A reconstruction-error variance in the manner of GloVe-V: truncated SVD discards
  singular mass that is a per-term residual, computable in the pass already performed.
  The GloVe-V derivation is specific to weighted least squares and would need reworking
  for SVD.

### Author and date: evidence and literature

| Source of between-book differences | Evidence |
|---|---|
| Publication date | None overall. Mantel ρ between −0.11 and +0.04 in nine queries; *capital*'s −0.32 is Hume and Steuart using it for a capital city. Four single term pairs trend with year, e.g. *supply*–*utility*. |
| Author | Same-author pairs sit at a median 90th percentile of count-controlled agreement; permutation p < 0.001. |
| How often the book uses the query | Cutting a query's uses to 40 keeps half its 50 nearest terms; a low count causes 75–80% of low-count books' disagreement ([audit](#passage-deletion-audit)). |
| Type of book (treatise or not) | No effect beyond book size (p = 0.33). |
| Single-book senses | *capital* as a city; *demand* as bank deposits in Fetter. |

What the literature says about each:

- **Synchronic variation is established.** Azarbonyad et al. (2017) measure shifts
  between *viewpoints* — any set of texts sharing a metadata feature, such as a party; an
  author is one. Del Tredici & Fernández (2017) and Lucy & Bamman (2021) measure variation
  between communities; Gonen et al. (2020) compare nearest-neighbour sets across any two
  corpora, the operation performed here per book; Schlechtweg et al. (2019) evaluate
  change across times and domains together; Chen et al. (2026) list cross-group variation
  as an established use. In sociolinguistics this is the older position (Weinreich,
  Labov & Herzog 1968).
- **Individual usage is distinctive and consistent.** Zhu & Jurgens (2021) find writers'
  styles "idiosyncratic but not arbitrary"; Welch et al. (2020) train per-user
  embeddings that support authorship attribution; Perifanos et al. (2018) use them as
  stylistic fingerprints. This matches the strongest signal here: same-author books agree
  far more than their counts predict.
- **Author against topic.** Sundararajan & Woodard (2018) show content words carry topic
  as well as style. Here, type of book added nothing once size was controlled, though
  type and size are correlated at 24 books.
- **Evidence behind a word governs its neighbourhood.** Sahlgren & Lenci (2016) find
  factorised count models, the family PPMI/SVD belongs to, the most reliable on small
  data; Wendlandt et al. (2018), Hellrich & Hahn (2016) and Antoniak & Mimno (2018) find
  neighbours unstable on small corpora; Herbelot & Baroni (2017) need a background space
  to learn from a handful of uses, which a per-book model cannot borrow.
- **A neighbourhood measure registers association, not sense.** Hamilton, Leskovec &
  Jurafsky (2016b) find local measures more sensitive to cultural shifts in what a word
  is associated with. Ricardo's *land* sits with *quality* and *inferior* because he
  discusses rent through land quality, not because *land* means something else to him.
- **Conceptual history needs interpretive grounding.** Wevers & Koolen (2020) stress that
  words are proxies for concepts; Ballandonne & Cersosimo (2023), the closest precedent
  in the history of economics, use frequencies, not embeddings.
- **Controls still apply.** Dubossarsky et al. (2017) show published laws of change
  weaken against a sentence-shuffling control. The author analogue — shuffle passages
  between books, re-embed, measure what between-author spread survives — has not been
  run.

**For the product:** the literature's name is *synchronic variation* ("drift" implies
change over time, which these books do not show); the author, not the date, is the unit
the variation follows; `occurrences` is the main reliability factor; Gonen et al.'s
neighbour overlap is the direct cross-corpus analogue, and Rodriguez, Spirling &
Stewart's embedding regression is the route to inference on small texts.

**Searched for, not found:** per-author embeddings compared across one scientific or
economic field; a study separating author from date within one period's texts; embedding
work on the history of economic thought.

---

## References

- Antoniak, M. & Mimno, D. (2018). Evaluating the Stability of Embedding-based Word
  Similarities. *TACL* 6. <https://aclanthology.org/Q18-1008/>
- Azarbonyad, H., Dehghani, M., Beelen, K., Arkut, A., Marx, M. & Kamps, J. (2017). Words
  are Malleable: Computing Semantic Shifts in Political and Media Discourse. *CIKM*.
  <https://arxiv.org/abs/1711.05603>
- Ballandonne, M. & Cersosimo, I. (2023). Towards a "Text as Data" Approach in the History
  of Economics: An Application to Adam Smith's Classics. *Journal of the History of
  Economic Thought* 45(1).
  <https://www.cambridge.org/core/journals/journal-of-the-history-of-economic-thought/article/abs/towards-a-text-as-data-approach-in-the-history-of-economics-an-application-to-adam-smiths-classics/30DC11E004DE0F7A33F5B30669D195DD>
- Berk, R. A., Western, B. & Weiss, R. E. (1995). Statistical inference for apparent
  populations. *Sociological Methodology*.
- Chen, J., Chersoni, E., Schlechtweg, D. & Huang, C.-R. (2026). Lexical semantic change
  detection: A survey of tasks, benchmarks, models, and potential impacts in digital
  humanities and social sciences. *Natural Language Processing*.
  <https://www.cambridge.org/core/journals/natural-language-processing/article/lexical-semantic-change-detection-a-survey-of-tasks-benchmarks-models-and-potential-impacts-in-digital-humanities-and-social-sciences/ECBF8E39442831EDB77BBA8C8B3D8EDF>
- Del Tredici, M. & Fernández, R. (2017). Semantic Variation in Online Communities of
  Practice. *IWCS*. <https://arxiv.org/abs/1806.05847>
- Dubossarsky, H., Weinshall, D. & Grossman, E. (2017). Outta Control: Laws of Semantic
  Change and Inherent Biases in Word Representation Models. *EMNLP*.
  <https://aclanthology.org/D17-1118/>
- Gonen, H., Jawahar, G., Seddah, D. & Goldberg, Y. (2020). Simple, Interpretable and
  Stable Method for Detecting Words with Usage Change across Corpora. *ACL*.
  <https://aclanthology.org/2020.acl-main.51/>
- Hamilton, W. L., Leskovec, J. & Jurafsky, D. (2016b). Cultural Shift or Linguistic
  Drift? Comparing Two Computational Measures of Semantic Change. *EMNLP*.
  <https://aclanthology.org/D16-1229/>
- Hellrich, J. & Hahn, U. (2016). Bad Company — Neighborhoods in Neural Embedding Spaces
  Considered Harmful. *COLING*. <https://aclanthology.org/C16-1262/>
- Herbelot, A. & Baroni, M. (2017). High-risk learning: acquiring new word vectors from
  tiny data. *EMNLP*, 304–309. <https://aclanthology.org/D17-1030/>
- Koplenig, A. (2019). Against statistical significance testing in corpus linguistics.
  *Corpus Linguistics and Linguistic Theory*.
- Kozlowski, A. C., Taddy, M. & Evans, J. A. (2019). *American Sociological Review*.
- Lucy, L. & Bamman, D. (2021). Characterizing English Variation across Social Media
  Communities with BERT. *TACL* 9, 538–556. <https://aclanthology.org/2021.tacl-1.33/>
- Perifanos, K., Florou, E. & Goutsos, D. (2018). Word embeddings for idiolect
  identification. *IISA*. <https://arxiv.org/abs/1902.03658>
- Rodriguez, P. L., Spirling, A. & Stewart, B. M. (2023). Embedding Regression: Models for
  Context-Specific Description and Inference. *American Political Science Review* 117(4),
  1255–1274.
- Sahlgren, M. & Lenci, A. (2016). The Effects of Data Size and Frequency Range on
  Distributional Semantic Models. *EMNLP*. <https://aclanthology.org/D16-1099/>
- Schlechtweg, D., Hätty, A., Del Tredici, M. & Schulte im Walde, S. (2019). A Wind of
  Change: Detecting and Evaluating Lexical Semantic Change across Times and Domains.
  *ACL*. <https://aclanthology.org/P19-1072/>
- Sundararajan, K. & Woodard, D. (2018). What represents "style" in authorship
  attribution? *COLING*. <https://aclanthology.org/C18-1238/>
- Vallebueno, A. et al. (2024). GloVe-V. *EMNLP*.
- Weinreich, U., Labov, W. & Herzog, M. I. (1968). Empirical Foundations for a Theory of
  Language Change. In W. P. Lehmann & Y. Malkiel (eds.), *Directions for Historical
  Linguistics*. University of Texas Press.
- Welch, C., Kummerfeld, J. K., Pérez-Rosas, V. & Mihalcea, R. (2020). Exploring the Value
  of Personalized Word Embeddings. *COLING*. <https://aclanthology.org/2020.coling-main.604/>
- Wendlandt, L., Kummerfeld, J. K. & Mihalcea, R. (2018). Factors Influencing the
  Surprising Instability of Word Embeddings. *NAACL*. <https://aclanthology.org/N18-1190/>
- Wevers, M. & Koolen, M. (2020). Digital begriffsgeschichte: Tracing semantic change using
  word embeddings. *Historical Methods*. <https://doi.org/10.1080/01615440.2020.1760157>
- Yu, B. (2013). Stability. *Bernoulli* 19(4).
- Zhu, J. & Jurgens, D. (2021). Idiosyncratic but not Arbitrary: Learning Idiolects in
  Online Registers Reveals Distinctive yet Consistent Individual Styles. *EMNLP*.
  <https://aclanthology.org/2021.emnlp-main.25/>

## Provenance

The perturbation and deletion-band figures come from three real books and nine
query × pair cells; the 24-book audit and the author/date measurements come from later
analyses. None of the scripts are committed. Citations were checked 2026-09-18 against
the ACL Anthology, arXiv and publisher pages; summaries come from abstracts, not full
texts. Weinreich, Labov & Herzog is cited for its general position on heterogeneity; no
quotation from it was verified, so none is given.
