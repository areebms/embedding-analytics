# scrape

*Stage 1 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Requests

Fetches a Project Gutenberg book by ID, strips the standard header and footer,
writes clean artifacts to S3. Skips re-scraping if the pipeline table already has
an `s3_text_key`.

| S3 artifact | Contents |
|---|---|
| `html/{index}.html` | Raw HTML |
| `text/{index}.txt` | Extracted body text |
| `metadata/{index}.json` | Title, author, publication metadata |
