# tokenize

*Stage 2 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** spaCy (`en_core_web_sm`), NLTK, WordNet

1. Segments into sentences with NLTK
2. Lemmatizes with spaCy (NER disabled for speed)
3. Smart-chunks large documents without splitting mid-sentence
4. Normalizes British/American spelling ("labor" → "labour")
5. Applies domain-specific aggressive lemmatization: nouns with derivationally
   related verbs collapse to the verb form ("production" → "produce"), controlled
   by a curated `ignored_nouns.txt` override list

| S3 artifact | Contents |
|---|---|
| `token_texts/{index}.csv` | Original tokens, one sentence per row |
| `token_lemmas/{index}.csv` | Lowercased lemmas (training input) |
| `token_tags/{index}.csv` | POS tags |
