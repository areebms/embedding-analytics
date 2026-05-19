import re
import os
import logging
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache

from shared.aws import TermTable, get_pipeline_table, get_session
from constants import PARSE_SYSTEM_PROMPT, FALLBACK_PROMPT
from schemas import OpNode, TermNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vocabulary() -> tuple[set[str], list[str]]:
    """Load the term vocabulary from DynamoDB (same source as /terms).

    Cached for the lifetime of the Lambda instance.
    """
    book_ids = [
        item["platform_data"]
        for item in get_pipeline_table().get_all_entries(
            ["platform_data", "s3_prefix_models"]
        )
        if "s3_prefix_models" in item
    ]

    terms: set[str] = set()
    term_table = TermTable(get_session())
    for book_id in book_ids:
        for item in term_table.get_entries(book_id, fields=["term", "tags"]):
            if item.get("tags") == {"R"}:
                continue
            terms.add(item["term"])

    sorted_terms = sorted(terms)
    logger.info("Loaded vocabulary: %d terms", len(sorted_terms))
    return terms, sorted_terms


# ---------------------------------------------------------------------------
# Expression serialization
# ---------------------------------------------------------------------------

def serialize_expression(
    tree: TermNode | OpNode, *, strip_outer: bool = False
) -> str:
    """Convert a TermNode/OpNode tree back into an expression string.

    Every binary operation is wrapped in parentheses.
    If strip_outer is True, the outermost parentheses are removed.
    """
    if isinstance(tree, TermNode):
        return tree.term

    if isinstance(tree, OpNode):
        left = serialize_expression(tree.args[0])
        right = serialize_expression(tree.args[1])
        result = f"({left} {tree.op} {right})"
        return result[1:-1] if strip_outer else result

    raise TypeError(f"Expected TermNode or OpNode, got {type(tree).__name__}")


# ---------------------------------------------------------------------------
# Expression parsing
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[()+-]|[^\s()+-]+")


def _tokenize(expression: str) -> list[str]:
    return _TOKEN_PATTERN.findall(expression)


def parse_expression(raw: str) -> TermNode | OpNode | None:
    """Parse an expression string into a TermNode/OpNode tree.

    Returns None on parse failure.
    """
    if not isinstance(raw, str):
        return None

    expression = raw.strip()
    if not expression:
        return None

    tokens = _tokenize(expression)
    cursor = 0

    def peek() -> str | None:
        return tokens[cursor] if cursor < len(tokens) else None

    def take() -> str | None:
        nonlocal cursor
        if cursor >= len(tokens):
            return None
        token = tokens[cursor]
        cursor += 1
        return token

    def primary() -> TermNode | OpNode | None:
        token = take()
        if token is None or token in ("+", "-", ")"):
            return None

        if token == "(":
            node = binary_expression()
            if node is None or take() != ")":
                return None
            return node

        return TermNode(term=token)

    def binary_expression() -> TermNode | OpNode | None:
        left = primary()

        while left and peek() in ("+", "-"):
            operator = take()
            right = primary()
            if right is None:
                return None
            left = OpNode(op=operator, args=[left, right])

        return left

    tree = binary_expression()
    return tree if tree and cursor == len(tokens) else None


def extract_terms(tree: TermNode | OpNode) -> list[str]:
    """Extract all term strings from a parsed tree."""
    terms: list[str] = []
    stack: list[TermNode | OpNode] = [tree]

    while stack:
        node = stack.pop()
        if isinstance(node, TermNode):
            terms.append(node.term)
        elif isinstance(node, OpNode):
            for i in range(len(node.args) - 1, -1, -1):
                stack.append(node.args[i])

    return terms


# ---------------------------------------------------------------------------
# Term resolution (vocabulary-level)
# ---------------------------------------------------------------------------

class TermResolutionError(Exception):
    """Raised when a term cannot be resolved to any vocabulary entry."""

    def __init__(self, term: str, candidates: list[str]):
        self.term = term
        self.candidates = candidates
        if candidates:
            msg = (
                f"No matching term found for '{term}'. "
                f"Did you mean: {', '.join(candidates)}?"
            )
        else:
            msg = (
                f"No matching term found for '{term}'. "
                f"No similar terms in vocabulary."
            )
        super().__init__(msg)


def _get_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Natural language parsing is not configured")
    return OpenAI(api_key=api_key)


def resolve_vocabulary_term(term: str) -> str:
    """Resolve a single term against the vocabulary.

    Returns the term unchanged if it exists, a fuzzy match if one is close
    enough, or an LLM-selected match from a narrowed candidate list.

    Raises TermResolutionError if no resolution is possible.
    """
    vocab, vocab_list = get_vocabulary()

    if term in vocab:
        return term

    # Step 1: high-confidence fuzzy match (free)
    close = get_close_matches(term, vocab_list, n=1, cutoff=0.6)
    if close:
        return close[0]

    # Step 2: LLM fallback with narrowed candidates
    candidates = get_close_matches(term, vocab_list, n=20, cutoff=0.3)
    if not candidates:
        fallback = [w for w in vocab_list if term and w[0] == term[0]]
        raise TermResolutionError(term, fallback[:5])

    client = _get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=20,
        temperature=0,
        messages=[{
            "role": "user",
            "content": FALLBACK_PROMPT.format(
                term=term, candidates=", ".join(candidates)
            ),
        }],
    )

    result = response.choices[0].message.content.strip().lower()
    if result in vocab:
        return result

    raise TermResolutionError(term, candidates[:5])


# ---------------------------------------------------------------------------
# Tree resolution
# ---------------------------------------------------------------------------

@dataclass
class Substitution:
    original: str
    resolved: str


def autocorrect_term_tree(
    tree: TermNode | OpNode,
) -> tuple[TermNode | OpNode, list[Substitution]]:
    """Walk the tree and resolve every term against the vocabulary.

    Returns the corrected tree and a list of substitutions made.
    Raises TermResolutionError if any term cannot be resolved.
    """
    substitutions: list[Substitution] = []

    def walk(node: TermNode | OpNode) -> TermNode | OpNode:
        if isinstance(node, TermNode):
            resolved = resolve_vocabulary_term(node.term)
            if resolved != node.term:
                substitutions.append(
                    Substitution(original=node.term, resolved=resolved)
                )
            return TermNode(term=resolved)

        return OpNode(op=node.op, args=[walk(node.args[0]), walk(node.args[1])])

    resolved_tree = walk(tree)
    return resolved_tree, substitutions


# ---------------------------------------------------------------------------
# Chat pipeline (steps 2-4)
# ---------------------------------------------------------------------------

def llm_generate_expression(message: str) -> str:
    """Call gpt-4o-mini to convert natural language into an expression string."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def process_chat_query(
    message: str,
) -> tuple[str, list[str], list[Substitution]]:
    """End-to-end: natural language in, resolved expression out.

    Returns (expression, terms, substitutions).

    Raises:
        ValueError: if the LLM output cannot be parsed.
        TermResolutionError: if any term cannot be resolved.
    """
    # Step 2: LLM generates expression string
    raw_expression = llm_generate_expression(message)

    # Step 3: parse into tree
    tree = parse_expression(raw_expression)
    if tree is None:
        raise ValueError(
            f"Failed to parse LLM output into a valid expression: '{raw_expression}'"
        )

    # Step 4: resolve terms and rebuild
    resolved_tree, substitutions = autocorrect_term_tree(tree)

    expression = serialize_expression(resolved_tree, strip_outer=True)
    terms = extract_terms(resolved_tree)

    return expression, terms, substitutions
