import re

HEADING_ELEMENTS = ("h1", "h2", "h3", "h4", "h5", "h6")

INLINE_ELEMENTS = (
    "a",
    "abbr",
    "b",
    "big",
    "cite",
    "code",
    "em",
    "font",
    "i",
    "q",
    "s",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "tt",
    "u",
    "var",
)
REFERENCE_MARKER_CLASSES = ("pagenum", "pageno", "pagenumber", "tei-noteref")
S3_STANDARDIZE_PREFIX = "standardize-headings"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
LLM_INDEX_ILLEGAL = re.compile(r"[^a-zA-Z0-9_-]")
