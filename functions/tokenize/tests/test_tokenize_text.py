"""One passage through spaCy, WordNet and the spelling map.

Against the real model and the real WordNet, not mocks: what is asserted here is the
lemma a term ends up trained under, and a mocked pipeline would prove nothing about it.
"""

from tokenize_text import (
    AMERICAN_TO_BRITISH,
    get_nlp,
    get_related_verbs,
    ignored_nouns,
    tokenize_passage,
)


def lemmas(passage):
    return {token.text: token.lemma for token in tokenize_passage(passage)}


def test_a_token_keeps_its_text_and_tag_beside_the_lemma():
    tokens = tokenize_passage("The greatest improvement in the productive powers.")

    assert [token.text for token in tokens][:3] == ["The", "greatest", "improvement"]
    assert [token.tag for token in tokens][:3] == ["DT", "JJS", "NN"]


def test_a_token_with_no_letters_gets_an_empty_lemma():
    """The row still carries it, so the three artifacts stay token-aligned."""
    tokens = tokenize_passage("Wealth, indeed!")

    assert [token.text for token in tokens] == ["Wealth", ",", "indeed", "!"]
    assert [token.lemma for token in tokens] == ["wealth", "", "indeed", ""]


def test_american_spellings_are_normalized_to_british():
    """One vector for organize/organise, one for labor/labour."""
    assert lemmas("They organize the labor.") == {
        "They": "they",
        "organize": "organise",
        "the": "the",
        "labor": "labour",
        ".": "",
    }


def test_a_word_that_merely_contains_an_american_spelling_is_untouched():
    """The lookup is on the whole cleaned token. The substring replace this stage used
    to do turned "laboratory" into "labouratory" and trained it as a term of its own."""
    assert "laboratory" not in AMERICAN_TO_BRITISH
    assert lemmas("Their laboratories collaborate.") == {
        "Their": "their",
        "laboratories": "laboratory",
        "collaborate": "collaborate",
        ".": "",
    }


def test_a_noun_collapses_to_its_derivationally_related_verb():
    assert lemmas("The production of improvement.") == {
        "The": "the",
        "production": "produce",
        "of": "of",
        "improvement": "improve",
        ".": "",
    }


def test_a_noun_on_the_override_list_keeps_its_own_lemma():
    """`building` has `build` in WordNet; `ignored_nouns.txt` is what stops the
    collapse, so the same lemma is asserted both with and without the entry."""
    assert "building" in ignored_nouns
    assert get_related_verbs("building") is None

    ignored_nouns.discard("building")
    try:
        assert get_related_verbs("building") == "build"
    finally:
        ignored_nouns.add("building")


def test_a_noun_with_no_related_verb_keeps_its_own_lemma():
    assert get_related_verbs("wealth") is None
    assert lemmas("Their wealth.")["wealth"] == "wealth"


def test_the_model_is_loaded_once_per_process():
    """Loading it is the expensive part, and one invocation is a whole list of books."""
    assert get_nlp() is get_nlp()
