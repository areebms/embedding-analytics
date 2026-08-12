# publish

*Stage 5 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** Gensim, NumPy

Flattens S3 artifacts into DynamoDB rows. For each term present in the centroid,
POS tag set, and aligned model stack, writes one row containing the centroid
vector (float16), per-seed aligned vectors (float16), token occurrence positions
(`ilocs`), POS tags, word count, disparity/variance/R-squared, and author/title
metadata.

Republishing prunes: terms that no longer exist after retraining are removed from
both Pinecone and the corpus vocabulary table.
