import re

HEADING_ELEMENTS = ("h1", "h2", "h3", "h4", "h5", "h6")
S3_STANDARDIZE_PREFIX = "standardize-headings"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
LLM_INDEX_ILLEGAL = re.compile(r"[^a-zA-Z0-9_-]")
