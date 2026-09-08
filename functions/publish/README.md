# publish

*Stage 4 of 5. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** NumPy

Flattens S3 artifacts into DynamoDB rows. For each term present in both the
embeddings archive and the POS tag set, writes one row containing the term vector
(float16), token occurrence positions (`ilocs`), POS tags and word count, and
writes author/title onto the pipeline row.

Republishing prunes: terms that no longer exist after re-embedding are removed from
the corpus vocabulary table.
