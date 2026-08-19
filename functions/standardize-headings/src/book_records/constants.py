import re

# The tags that count as a heading. Both sides of the handoff read this same list:
# standardize-headings numbers the headings it sends to Claude by it, and collect
# maps the positions that come back onto the same pairs.
HEADING_ELEMENTS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Where the handoff objects live, and what they are written as. standardize-headings
# writes both keys under this prefix and collect reads them back, so the two stages
# share the one definition rather than each spelling the prefix out.
S3_STANDARDIZE_PREFIX = "standardize-headings"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"

# Everything Anthropic disallows in a custom_id. book_records.utils substitutes on
# it to build one and llm_classify_request.schemas validates against it, so it is
# defined here -- the one module light enough for both to import -- rather than in
# either of them.
CUSTOM_ID_ILLEGAL = re.compile(r"[^a-zA-Z0-9_-]")
