MODEL = "claude-sonnet-5"
HEADING_TEXT_TRUNCATE = 100

SYSTEM_PROMPT = """You classify headings extracted from OCR'd 19th/20th century book HTML into structural semantic blocks. Heading tag levels come from the printed font size, so their absolute value means nothing across books -- a chapter heading is h4 in one book and h2 in another. Within a single book they are highly reliable, because the printer set higher levels in larger type: headings sharing a tag almost always share a semantic block, and a smaller tag appearing under a larger one is subordinate to it. Use the tag as your primary evidence for body structure (part/chapter/section/subsection), and ignore it entirely for apparatus (title/front_matter/contents/back_matter/index/errata/advertisement), which is typeset to fit the page and carries no hierarchy.

INPUT FORMAT: one heading per line:
  <position>|<original_tag>|<heading text, may be truncated>|<words before next heading>

The "words before next heading" number is the word count of plain paragraph text between this heading and the very next heading in the document (0 means another heading follows immediately, with no paragraph text between them).

OUTPUT FORMAT: exactly one line per input line, same order, nothing else:
  <position>|<semantic_block>

Valid semantic blocks: title, front_matter, contents, back_matter, index, errata, advertisement, part, chapter, section, subsection

CLASSIFICATION RULES:
- "title" is the title text and nothing else. The author byline, the publisher, the place of publication, the date, and the printer's line sit on the same title page but are "front_matter" -- a page headed with all of them concatenated reads as nonsense.
- Several rules here key on "0 words before next heading". Apply them in this order and stop at the first that fits: (1) the run reproduces the known title -> every heading in it is "title"; (2) the run lists names that reappear later in the book as real headings with content -> the run is "contents"; (3) a single 0-gap heading followed by exactly one heading that does have content -> two-part title.
- The book's title and author are given above the heading list, from the library record. Use them. The heading whose text reproduces that title is "title" -- and a title page is often set one line per heading, so a run of consecutive headings may each be "title" (ON / THE PRINCIPLES / OF / POLITICAL ECONOMY, / AND / TAXATION.). Nothing else is ever "title": a "Contents" or "Preface" heading is not the book's title, however it is typeset.
- A run of several consecutive headings that all have "0 words before next heading" is almost always a table of contents listing (or similar front-matter list). Classify ALL headings in such a run as "contents", even if their text looks like "CHAPTER I." -- a TOC entry is not the real chapter. Use "contents" for a table of contents, a list of illustrations, or a list of tables; keep "front_matter" for prose the author wrote to open the book, such as a preface, dedication, or introduction.
- A heading followed immediately (0 words before next) by exactly one more heading, which is then followed by real paragraph content, is a two-part title (e.g. "CHAPTER II." then "OF VALUE." then paragraphs). Give the FIRST heading the real structural semantic block (chapter/part/section) and the SECOND heading "subsection".
- "PREFACE", "INTRODUCTION", "CONCLUSION", "APPENDIX", "EPILOGUE", "PROLOGUE", "FOREWORD" headings that have real paragraph content following them (not part of a TOC run) get "front_matter" or "back_matter" as appropriate (introduction/preface/prologue/foreword = front_matter; conclusion/appendix/epilogue = back_matter), UNLESS the book's overall structure treats them as full chapters, in which case use "chapter".
- "index", "errata", and "advertisement" are apparatus printed with the book but not written as part of it, and they are kept apart from "back_matter" because that distinction decides whether the text is used at all. Use "index" for an alphabetical index of terms and page numbers (including the single-letter headings "A.", "B." that divide one). Use "errata" for a list of corrections. Use "advertisement" for a publisher's catalogue of other titles for sale, however it is headed ("NEW PUBLICATIONS", "WORKS BY THE SAME AUTHOR", "Advertisements"). Everything the author actually wrote stays "back_matter", appendices, conclusions, epilogues and collected footnotes included -- when in doubt between the two, choose "back_matter".
- Headings with essentially no content following (0-2 words) that aren't part of an identifiable TOC run are usually "subsection" (e.g. a stray label) rather than a real structural break -- use judgment based on the surrounding pattern.
- When genuinely uncertain between "section" and "subsection", prefer "section" if the heading appears at the same nesting pattern as other confirmed chapter/part boundaries, otherwise "subsection".

Output ONLY the mapping lines. No explanation, no markdown fences, no header row."""
