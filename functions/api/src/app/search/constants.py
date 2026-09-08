import re

PARSE_SYSTEM_PROMPT = (
    "Convert natural language into a vector arithmetic expression for querying "
    "word embeddings from classical economics texts (Adam Smith, David Ricardo, "
    "John Stuart Mill, etc.).\n\n"
    "The expression format:\n"
    "- A single concept is a bare lowercase word: labour\n"
    '- "+" combines concepts (find words related to both)\n'
    '- "-" creates contrast (find words related to left but not right)\n'
    "- Every binary operation beyond the outermost must be wrapped in parentheses\n\n"
    "Examples:\n"
    '- "words about labour" -> labour\n'
    '- "trade and commerce" -> trade + commerce\n'
    '- "labour but not wages" -> labour - wage\n'
    '- "productive labour vs unproductive" -> labour + (productive - unproductive)\n'
    '- "domestic vs foreign trade" -> trade + (domestic - foreign)\n'
    '- "use value" -> useful + value\n'
    '- "exchange value" -> exchange + value\n'
    '- "wealth from land minus industrial profit" -> (wealth + land) - (industrial + profit)\n\n'
    "Rules:\n"
    '- Use lemmatized forms: "wages" -> "wage", "markets" -> "market", '
    '"produced" -> "produce"\n'
    "- Every operator takes exactly two arguments\n"
    "- Use vocabulary appropriate for 18th-19th century economics texts\n"
    "- Parenthesize all nested operations, but not the outermost one\n"
    "- When contrasting two modifiers of a shared concept (e.g. "
    '"productive vs unproductive labour"), keep the shared concept and subtract '
    "the modifiers: concept + (modifierA - modifierB)\n"
    "- Multi-word economic terms are combinations: join their component words "
    "with +. Use the adjective form where appropriate "
    '(e.g. "use value" -> useful + value)\n'
    "- Respond with only the expression string, no other text\n"
    "- Do not use the same word twice"
)

FALLBACK_PROMPT = (
    "The user searched for the term '{term}' but it is not in the vocabulary. "
    "Pick the single most semantically similar term from this list, "
    "or respond with NONE if nothing fits.\n\n"
    "Candidates: {candidates}"
)

BARE_TERM_PATTERN = re.compile(r"[^\s()+-]+")
MIN_BOOKS_WITH_TERM = 4
MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS = 2
MIN_BOOKS_WITH_UNSTABLE_TERM_AS_TOP_50 = 1
NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY = 75
NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING = 100
MAX_RANK_FOR_STABLE_TERM = 50
MAX_RANK_FOR_UNSTABLE_TERM = 100
NUM_COMPARATIVE_TERMS = 5
NUM_RELEVANT_TERMS_FOR_INSTABILITY = 100