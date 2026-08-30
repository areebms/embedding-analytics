SEMANTIC_BLOCK_TO_LEVEL = {"part": "h2", "chapter": "h2", "section": "h3", "drop": "h3"}

UNTRAINABLE_BLOCKS = ("drop",)

DEFAULT_BLOCK = "section"

HEADING_ELEMENTS = ("h1", "h2", "h3", "h4", "h5", "h6")

MODEL = "claude-sonnet-5"
HEADING_TEXT_TRUNCATE = 100
MAX_OUTPUT_TOKENS = 16000

BATCH_ENDED = "ended"

VALID_BLOCKS = ", ".join(SEMANTIC_BLOCK_TO_LEVEL)

SYSTEM_PROMPT = f"""Classify headings extracted from OCR'd book HTML into structural semantic blocks. Heading tag levels come from the printed font size. They are reliable within a single book because the printer set higher levels in larger type: headings sharing a tag almost always share a semantic block, and a smaller tag appearing under a larger one is subordinate to it. However, their absolute value means nothing across books. A chapter heading is h4 in one book and h2 in another. 

Classify the whole list: the paratext a book was printed with as well as the structure its author wrote. 

INPUT FORMAT: one heading per line:
  <position>|<original_tag>|<heading text, may be truncated>|<words before next heading>

The "words before next heading" number is the word count of plain paragraph text between this heading and the very next heading in the document (0 means another heading follows immediately, with no paragraph text between them).

OUTPUT FORMAT: exactly one line per input line, same order, nothing else:
  <position>|<semantic_block>

Valid semantic blocks: {VALID_BLOCKS}

THE PARATEXT. Printed with the book, not written as part of it. All of it is "drop".
- The title page: the title, the byline, the author's degrees and affiliation, the translator, the publisher, the place, the date, the printer's line -- however many headings they are set across. The record's title and author are given above the heading list; use them to recognise where it is. It ends at the first heading that is followed by real paragraph text, or that names paratext or front matter; that terminating heading is not itself part of the title page. A transcription may print no title page at all -- if the opening headings do not reproduce the record's title, there is none.
- A table of contents, a list of illustrations, tables or plates. The listing's own heading AND its entries: a run of headings with 0 words between them whose text turns up again further down as real headings with content is that listing, even if the entries read "CHAPTER I." A TOC entry is not the real chapter.
- The reverse is the commonest way to lose a book: "CHAPTER I." followed immediately by "ON VALUE." followed by 8,000 words of prose is the BODY, not a listing. Before marking a 0-gap run as a listing, check that its entries reappear later; if the prose comes right after the run instead, it is the body.
- An alphabetical index, including the bare single-letter headings ("A.", "B.") that divide one, a table of errata or corrigenda, and a publisher's catalogue of other titles for sale, however headed ("NEW PUBLICATIONS", "WORKS BY THE SAME AUTHOR", "Advertisements").
- Everything the author actually wrote stays out of "drop" - appendices, conclusions, epilogues and collected footnotes are "section". When in doubt between the two, choose "section": getting it wrong one way feeds an alphabetical index to a language model, and the other way deletes an author's appendix.

THE BODY.
- "part" is the largest division a book uses (PART I, BOOK III, FIRST MEMOIR, DIVISION A). "chapter" is the division below it, and the one that carries the argument. "section" is anything below that, however deeply nested -- a book's fourth level and its second both render the same, so do not agonise over the difference.
- Identify the chapter level by its tag, not by its wording. The tag that repeats a long, evenly spread sequence through the body is the chapter level, whether its headings read "CHAPTER I", "CHAP. VII.", a bare roman numeral, or a title with no number at all.
- A heading followed immediately (0 words before next) by exactly one more heading, which is then followed by real paragraph content, is a two-part title (e.g. "CHAPTER II." then "OF VALUE." then paragraphs). Give the FIRST heading the real structural block (part/chapter) and the SECOND heading "section".
- "PREFACE", "INTRODUCTION", "FOREWORD", "PROLOGUE", "CONCLUSION", "APPENDIX", "EPILOGUE" and collected footnotes are "section" when real paragraph content follows them - UNLESS the book's structure treats them as full chapters, in which case use "chapter".
- Headings with essentially no content following (0-2 words) that are not part of an identifiable listing are usually "section" rather than a real structural break.

Output ONLY the mapping lines. No explanation, no markdown fences, no header row."""
