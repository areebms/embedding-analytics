# train-kvector

*Stage 3 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Library:** Gensim

One Word2Vec model per invocation with an explicit seed. Filters lemmas to
alphabetic tokens longer than 3 characters. Output:
`kvectors/{index}/collected/{seed}-{timestamp}-{randint}.model`

| Parameter | Value |
|---|---|
| Vector size | 200 |
| Window | 10 tokens |
| Min count | 10 |
| Algorithm | Skip-gram (`sg=1`) |
| Training | Hierarchical softmax (`hs=1`) |
| Subsampling | `5e-4` |
| Negative sampling | Off |
| Epochs | 30 |
