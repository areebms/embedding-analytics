from app.search.schemas.semantic_drift import OpNode, TermNode


def serialize_expression(tree: TermNode | OpNode, *, strip_outer: bool = False) -> str:
    """Convert a TermNode/OpNode tree back into an expression string."""
    if isinstance(tree, TermNode):
        return tree.term

    if isinstance(tree, OpNode):
        left = serialize_expression(tree.args[0])
        right = serialize_expression(tree.args[1])
        result = f"({left} {tree.op} {right})"
        return result[1:-1] if strip_outer else result

    raise TypeError(f"Expected TermNode or OpNode, got {type(tree).__name__}")


def extract_terms(tree: TermNode | OpNode) -> list[str]:
    """Every term in the tree, left to right, first occurrence only."""
    seen: set[str] = set()
    terms: list[str] = []
    stack: list[TermNode | OpNode] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, TermNode):
            if node.term not in seen:
                seen.add(node.term)
                terms.append(node.term)
        else:
            for i in range(len(node.args) - 1, -1, -1):
                stack.append(node.args[i])
    return terms
