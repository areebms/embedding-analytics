MODEL = "claude-sonnet-5"
HEADING_TEXT_TRUNCATE = 100

SYSTEM_PROMPT = """You classify headings extracted from OCR'd 19th/20th century book HTML into structural semantic blocks. Heading tag levels in the source are unreliable -- they were assigned by font size during OCR, not by document logic -- so classify by content and structural pattern, not by the tag shown.

INPUT FORMAT: one heading per line:
  <position>|<original_tag>|<heading text, may be truncated>|<words before next heading>

The "words before next heading" number is the word count of plain paragraph text between this heading and the very next heading in the document (0 means another heading follows immediately, with no paragraph text between them).

OUTPUT FORMAT: exactly one line per input line, same order, nothing else:
  <position>|<semantic_block>

Valid semantic blocks: title, front_matter, back_matter, part, chapter, section, subsection

CLASSIFICATION RULES:
- Exactly one heading is "title" -- normally position 0, the book's main title.
- A run of several consecutive headings that all have "0 words before next heading" is almost always a table of contents listing (or similar front-matter list). Classify ALL headings in such a run as "front_matter", even if their text looks like "CHAPTER I." -- a TOC entry is not the real chapter.
- A heading followed immediately (0 words before next) by exactly one more heading, which is then followed by real paragraph content, is a two-part title (e.g. "CHAPTER II." then "OF VALUE." then paragraphs). Give the FIRST heading the real structural semantic block (chapter/part/section) and the SECOND heading "subsection".
- Headings need not be in English. Treat a non-English structural word (CAPITOLO, CHAPITRE, CAPITULO, KABANATA, HOOFDSTUK, FEJEZET, and so on) exactly as you would its English equivalent.
- "PREFACE", "INTRODUCTION", "CONCLUSION", "APPENDIX", "EPILOGUE", "PROLOGUE", "FOREWORD" headings that have real paragraph content following them (not part of a TOC run) get "front_matter" or "back_matter" as appropriate (introduction/preface/prologue/foreword = front_matter; conclusion/appendix/epilogue = back_matter), UNLESS the book's overall structure treats them as full chapters, in which case use "chapter".
- Headings with essentially no content following (0-2 words) that aren't part of an identifiable TOC run are usually "subsection" (e.g. a stray label) rather than a real structural break -- use judgment based on the surrounding pattern.
- When genuinely uncertain between "section" and "subsection", prefer "section" if the heading appears at the same nesting pattern as other confirmed chapter/part boundaries, otherwise "subsection".

Output ONLY the mapping lines. No explanation, no markdown fences, no header row."""
