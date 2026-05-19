# Describe Mode and Expression Evaluation

Describe mode is a one-shot translation layer: the user types plain English, the backend converts it into a validated vector expression, and the UI switches back to vector mode showing the interpreted expression as editable chips. There is no persistent conversational state.

This document covers the LLM integration pattern, term resolution pipeline, expression evaluation, and how contrast queries behave.

---

## Product goal

The main UI supports structured vector expressions:

```text
labour + (productive - unproductive)
```

That is powerful but assumes the user knows how to express a research question as vector arithmetic.

Describe mode solves that usability problem:

```text
terms related to labour but more productive than unproductive
```

The backend converts the message into a validated expression. The frontend shows the result as editable chips so the user can inspect or adjust before relying on it. The LLM proposes; deterministic code validates; the user decides.

---

## Frontend flow

1. The user selects **Describe** from the `InputModeControl` dropdown inside `VectorExpressionInput`. The current expression is cleared and replaced with a plain text field.
2. The user types a message and submits.
3. `handleDescribeSubmit` in `VectorExpressionInput` calls `onDescribeSubmit(trimmed)`, wired to `describeMutation.mutateAsync(message)` in `App.jsx`.
4. The mutation calls `parseDescribeQuery`, which POSTs `{ message }` to `/parse-describe`.
5. The backend returns `{ expression, terms, substitutions }`.
6. `App.jsx` calls `setExpression(result.expression)`, triggering `parseExpression` and the `useSimilarityQueries` hook. Similarity requests fire for each visible book.
7. `VectorExpressionInput` switches back to vector mode and displays the returned expression as editable chips.
8. If the backend made substitutions, a warning Alert shows what changed.

The user never stays in describe mode after submitting. It is a translation step, not a conversation.

---

## Backend pipeline

The `/parse-describe` endpoint converts natural language into a validated expression tree in four steps:

```text
message --> LLM expression --> parser --> term resolution --> validated expression
```

---

## Step 1: LLM expression generation

The user message is sent to `gpt-4o-mini` with a system prompt tuned for classical economics vocabulary.

The prompt instructs the model to produce a vector arithmetic expression using terms, `+`, `-`, and parentheses. It includes rules for handling multi-word concepts (join components with `+`, prefer adjective forms), contrastive phrasing (shared concept with subtracted modifiers), and lemmatization conventions.

Example:

```text
"productive labour vs unproductive"  -->  labour + (productive - unproductive)
```

---

## Step 2: Recursive descent parsing

The generated expression is tokenized and parsed into a tree of `TermNode` and `OpNode` objects. Every binary operator takes exactly two arguments. Nested operations must be parenthesized; the outermost operation does not require parentheses.

```json
{
  "op": "+",
  "args": [
    { "term": "labour" },
    {
      "op": "-",
      "args": [
        { "term": "productive" },
        { "term": "unproductive" }
      ]
    }
  ]
}
```

If the LLM returns malformed syntax, parsing fails and the endpoint returns a 400.

This is the key boundary: the LLM proposes an expression, but deterministic code decides whether it is structurally valid. No LLM output reaches the evaluation layer without passing the parser.

---

## Step 3: Term resolution

Every parsed term is validated against the DynamoDB vocabulary, cached per Lambda instance via `lru_cache`. Resolution uses three tiers, escalating cost only when needed:

**Exact match:** The term exists in the vocabulary as written. No external call.

**Fuzzy match:** `difflib.get_close_matches` at a 0.6 cutoff. If a single close match is found, it is used automatically. No external call.

**LLM fallback:** If fuzzy matching at 0.6 fails, the backend widens to a 0.3 cutoff and collects up to 20 candidates, then asks `gpt-4o-mini` to pick the single most semantically appropriate term. The selected term must exist in the vocabulary. If the model returns an invalid term or no candidates exist, the endpoint raises a `TermResolutionError`.

A `TermResolutionError` returns a 422 with the unresolved term and up to 5 candidate suggestions, giving the frontend enough context for manual recovery.

---

## Step 4: Rebuild and return

Substitutions are applied to the tree. The corrected tree is serialized back into an expression string and returned:

```json
{
  "expression": "labour + (productive - unproductive)",
  "terms": ["labour", "productive", "unproductive"],
  "substitutions": [
    { "original": "wages", "resolved": "wage" }
  ]
}
```

The frontend uses this expression to populate the chip input and fire similarity queries.

---

## Expression evaluation and normalization

The `/similarity/{book_id}` endpoint evaluates the returned expression tree. For each term, the backend fetches per-seed aligned vectors from DynamoDB. Operations are applied across the ensemble so every seed produces its own query vector.

The evaluator normalizes after each sub-expression, not once at the end.

For `labour + (productive - unproductive)`:

1. Fetch per-seed vectors for `productive` and `unproductive`
2. Compute `productive - unproductive` element-wise across seeds
3. Normalize the contrast direction to unit length
4. Add the normalized contrast to the per-seed `labour` vectors
5. Normalize the final query vectors to unit length
6. Compute cosine similarity against target terms in each aligned model
7. Return mean similarity and 95% confidence interval

### Why per-operation normalization

Normalizing at each step prevents high-frequency or high-norm terms from dominating combined expressions. It also makes contrast expressions more interpretable: `productive - unproductive` becomes a direction that tilts the base term, not a raw magnitude that might be too small to matter.

The tradeoff: if two terms are semantically close, their raw difference vector is small, and normalizing amplifies whatever residual signal (or noise) remains. If that residual varies across seeds, the confidence interval widens accordingly. This makes the CI a direct quality signal for contrast queries.

---

## Interpreting contrast results

Two example queries illustrate how the contrast direction reshapes results.

### `labour + (productive - unproductive)`

Surfaces vocabulary around production systems: capital, machine, industry, commodity, profit, tool, land, soil, skill, improve. This is labour situated inside a production context: operating machinery, cultivating land, advancing capital, generating vendible commodities.

### `labour + (unproductive - productive)`

Surfaces vocabulary around service, dependence, and expenditure: servant, hire, spend, lawyer, subsistence, earn, daily, class, workman, capitalist. This maps to how classical economists defined unproductive labour: paid from revenue rather than capital, not resulting in a vendible commodity.

### Overlap between directions

Terms like `wage`, `employ`, `value`, and `capital` appear in both. That is expected. Productive and unproductive labour are not separate topics; they are opposing categories within the same debate. The contrast changes which surrounding terms move to the foreground; it does not create disjoint clusters.

---

## Engineering value

Describe mode demonstrates an LLM integration pattern that applies beyond this project:

- Use the LLM for translation, not for unchecked execution
- Parse the LLM output with deterministic code before it reaches any execution layer
- Validate every extracted entity against an application-controlled vocabulary
- Resolve ambiguity through tiered fallbacks that escalate cost only when cheaper options fail
- Return structured errors with enough context for the frontend to offer manual recovery
- Keep the final state editable by the user

The result is an AI-assisted product workflow with explicit guardrails, not an open-ended chat interface.
