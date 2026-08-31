import json
from pathlib import Path
from typing import NamedTuple

import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from spacy.language import Language

SPACY_TO_WORDNET = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
}

lemmatizer = WordNetLemmatizer()

DATA_DIR = Path(__file__).parent / "data"

with (DATA_DIR / "ignored_nouns.txt").open(encoding="utf-8") as file:
    ignored_nouns = set(file.read().splitlines())


# `american_spellings.json` is vendored verbatim from hyperreality/American-British-
# English-Translator (MIT, (c) 2016), whose `data/american_spellings.json` is keyed by the
# American spelling: https://github.com/hyperreality/American-British-English-Translator
#
# The corpus is British-side -- "labour" is the term the README queries and the term
# `corpus_terms` rows carry -- so the mapping runs American to British, and the same repo's
# `british_spellings.json` is the wrong file here: it is keyed the other way round.
#
# Lookups are on an already-lowercased, alpha-only token, so the keys are lowercased and the
# four entries that are not words at all (the `estr-`, `feto-`, `leuk-`, `paleo-` prefix
# stubs) are dropped -- no cleaned token can ever equal them.
AMERICAN_TO_BRITISH: dict[str, str] = {}
with (DATA_DIR / "american_spellings.json").open(encoding="utf-8") as file:
    for american, british in json.load(file).items():
        if american.isalpha():
            AMERICAN_TO_BRITISH[american.lower()] = british.lower()


class Token(NamedTuple):
    text: str
    lemma: str
    tag: str

_nlp = None

def get_nlp() -> Language:
    """The loaded model, once per process; loading it is the expensive part."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp

def get_related_verbs(lemma: str) -> str | None:
    if lemma in ignored_nouns:
        return

    derivations = set()
    for synset in wordnet.synsets(lemma):
        for synset_lemma in synset.lemmas():
            if synset_lemma.name().lower() != lemma:
                continue

            for related_form in synset_lemma.derivationally_related_forms():
                if related_form.synset().pos() != wordnet.VERB:
                    continue

                related_lemma = related_form.name().lower()
                if len(related_lemma) < len(lemma) and lemma[0] == related_lemma[0]:
                    derivations.add(related_lemma)
    return min(derivations, key=len) if derivations else None


def aggressively_lemmatize(token: str, pos: str | None) -> str:
    if pos is None:
        return token

    lemma = lemmatizer.lemmatize(token, pos=pos)
    if pos == wordnet.NOUN:
        related_verb = get_related_verbs(lemma)
        if related_verb:
            return related_verb

    return lemma


def tokenize_passage(passage: str) -> list[Token]:
    """One `Token` per token of `passage`, in order."""

    nlp = get_nlp()

    tokens: list[Token] = []
    for spacy_token in nlp(passage):
        cleaned_token = "".join(
            char for char in spacy_token.text if char.isalpha()
        ).lower()

        if len(cleaned_token) == 0:
            lemma = ""
        else:
            cleaned_token = AMERICAN_TO_BRITISH.get(cleaned_token, cleaned_token)
            token_pos = SPACY_TO_WORDNET.get(spacy_token.pos_)
            lemma = aggressively_lemmatize(cleaned_token, token_pos)

        tokens.append(Token(text=spacy_token.text, lemma=lemma, tag=spacy_token.tag_))

    return tokens
