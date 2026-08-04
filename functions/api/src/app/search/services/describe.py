import re
import os
from collections import deque
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from typing import Literal

from openai import OpenAI

from app.core.logging import add_to_log
from app.search.services.utils import extract_terms, serialize_expression
from app.search.constants import PARSE_SYSTEM_PROMPT, FALLBACK_PROMPT
from app.search.schemas.semantic_drift import OpNode, TermNode
from app.search.errors import TermResolutionError
from shared.commons import BookIndex
from shared.tables.book_terms import get_book_term_table


@lru_cache(maxsize=1)
def get_vocabulary(book_ids: tuple[BookIndex, ...]) -> tuple[set[str], list[str]]:
    """Load the term vocabulary from DynamoDB (same source as /terms).

    Cached for the lifetime of the Lambda instance.
    """
    terms: set[str] = set()
    term_table = get_book_term_table()
    for book_id in book_ids:
        for item in term_table.get_entries(book_id, fields=["term", "tags"]):
            if item.get("tags") == {"R"}:
                continue
            terms.add(item["term"])

    sorted_terms = sorted(terms)
    add_to_log(vocab_terms=len(sorted_terms))
    return terms, sorted_terms


# ---------------------------------------------------------------------------
# Expression parsing
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[()+-]|[^\s()+-]+")


def _tokenize(expression: str) -> list[str]:
    return TOKEN_PATTERN.findall(expression)


def parse_expression(raw: str) -> TermNode | OpNode | None:
    """Parse an expression string into a TermNode/OpNode tree.

    Returns None on parse failure.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    q: deque[str] = deque(_tokenize(raw.strip()))

    def get_operand() -> TermNode | OpNode | None:
        token = q.popleft() if q else None
        if token is None or token in ("+", "-", ")"):
            return None
        if token == "(":
            node = binary_expression()
            if node is None or (q.popleft() if q else None) != ")":
                return None
            return node
        return TermNode(term=token)

    def binary_expression() -> TermNode | OpNode | None:
        left_operand = get_operand()
        while left_operand and q and q[0] in ("+", "-"):
            op: Literal["+", "-"] = "+" if q.popleft() == "+" else "-"
            right_operand = get_operand()
            if right_operand is None:
                return None
            left_operand = OpNode(op=op, args=[left_operand, right_operand])
        return left_operand

    tree = binary_expression()
    return tree if tree and not q else None


def resolve_vocabulary_term(term: str, book_ids: tuple[BookIndex, ...]) -> str:
    """Resolve a single term against the vocabulary.

    Returns the term unchanged if it exists, a fuzzy match if one is close
    enough, or an LLM-selected match from a narrowed candidate list.

    Raises TermResolutionError if no resolution is possible.
    """
    vocab, vocab_list = get_vocabulary(book_ids)

    if term in vocab:
        return term

    close = get_close_matches(term, vocab_list, n=1, cutoff=0.6)
    if close:
        return close[0]

    candidates = get_close_matches(term, vocab_list, n=20, cutoff=0.3)
    if not candidates:
        fallback = [w for w in vocab_list if term and w[0] == term[0]]
        raise TermResolutionError(term, fallback[:5])

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=20,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": FALLBACK_PROMPT.format(
                    term=term, candidates=", ".join(candidates)
                ),
            }
        ],
    )

    content = response.choices[0].message.content
    if content and content.strip().lower() in vocab:
        return content.strip().lower()

    raise TermResolutionError(term, candidates[:5])


@dataclass
class Substitution:
    original: str
    resolved: str


def autocorrect_term_tree(
    tree: TermNode | OpNode, book_ids: tuple[BookIndex, ...]
) -> tuple[TermNode | OpNode, list[Substitution]]:
    """Walk the tree and resolve every term against the vocabulary.

    Returns the corrected tree and a list of substitutions made.
    Raises TermResolutionError if any term cannot be resolved.
    """
    substitutions: list[Substitution] = []

    def walk(node: TermNode | OpNode) -> TermNode | OpNode:
        if isinstance(node, TermNode):
            resolved = resolve_vocabulary_term(node.term, book_ids)
            if resolved != node.term:
                substitutions.append(
                    Substitution(original=node.term, resolved=resolved)
                )
            return TermNode(term=resolved)

        return OpNode(op=node.op, args=[walk(node.args[0]), walk(node.args[1])])

    resolved_tree = walk(tree)
    return resolved_tree, substitutions


def llm_generate_expression(message: str) -> str:
    """Call gpt-4o-mini to convert natural language into an expression string."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=200,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def process_describe_query(
    message: str, book_ids: tuple[BookIndex, ...]
) -> tuple[str, list[str], list[Substitution]]:
    """End-to-end: natural language in, resolved expression out.

    Returns (expression, terms, substitutions).

    Raises:
        ValueError: if the LLM output cannot be parsed.
        TermResolutionError: if any term cannot be resolved.
    """
    raw_expression = llm_generate_expression(message)

    tree = parse_expression(raw_expression)
    if tree is None:
        raise ValueError(
            f"Failed to parse LLM output into a valid expression: '{raw_expression}'"
        )

    resolved_tree, substitutions = autocorrect_term_tree(tree, book_ids)

    expression = serialize_expression(resolved_tree, strip_outer=True)
    terms = extract_terms(resolved_tree)

    return expression, terms, substitutions
