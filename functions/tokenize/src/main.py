import csv
import io
import logging
from pathlib import Path

import spacy
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from shared.aws import PipelineTable, upload_object, load_text_from_s3, get_session
from shared.commons import get_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SENTENCE_ENDINGS = {".", "!", "?"}
SPACY_TO_WORDNET = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
}

lemmatizer = WordNetLemmatizer()

with Path(__file__).with_name("ignored_nouns.txt").open(encoding="utf-8") as file:
    ignored_nouns = set(file.read().splitlines())


def chunk_text(nlp, text):
    all_sentences = sent_tokenize(text)
    num_chunks = len(all_sentences) % nlp.max_length
    if num_chunks <= 1:
        return [text]

    sentences_per_chunk = int(len(all_sentences) / num_chunks)
    sentence_chunks = []
    current_chunk = []

    for sentence in all_sentences:
        current_chunk.append(sentence)
        if sentence[-1] not in SENTENCE_ENDINGS:
            continue

        if len(current_chunk) >= sentences_per_chunk:
            sentence_chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        sentence_chunks.append(current_chunk)

    return [" ".join(chunk) for chunk in sentence_chunks]


def csv_bytes(rows):
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue().encode("utf-8")


def get_related_verbs(lemma):
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


def aggressively_lemmatize(token, pos):
    if pos is None:
        return token

    lemma = lemmatizer.lemmatize(token, pos=pos)
    if pos == wordnet.NOUN:
        related_verb = get_related_verbs(lemma)
        if related_verb:
            return related_verb

    return lemma


def tokenize(index):
    session = get_session()

    table = PipelineTable(session)

    item = table.get(
        index,
        expression="s3_text_key,s3_token_texts_key,s3_token_lemmas_key,s3_token_tags_key",
    )

    s3_text_key = item.get("s3_text_key")
    if not s3_text_key:
        logger.info("Index has not been scraped", extra={"index": index})
        return

    text = load_text_from_s3(session, s3_text_key)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    doc_texts = chunk_text(nlp, text)

    token_texts = []
    token_lemmas = []
    token_tags = []
    for doc in nlp.pipe(doc_texts, batch_size=32, n_process=4):
        for sentence in doc.sents:
            token_lemmas.append([])
            token_texts.append([])
            token_tags.append([])
            for token in sentence:
                cleaned_token = "".join(
                    char for char in token.text if char.isalpha()
                ).lower()

                if len(cleaned_token) == 0:
                    token_lemmas[-1].append("")
                else:
                    cleaned_token = cleaned_token.replace("labor", "labour")
                    token_pos = SPACY_TO_WORDNET.get(token.pos_)
                    token_lemmas[-1].append(
                        aggressively_lemmatize(cleaned_token, token_pos)
                    )

                token_texts[-1].append(token.text)
                token_tags[-1].append(token.tag_)
    s3_writes = [
        ("s3_token_texts_key", f"token_texts/{index}.csv", token_texts),
        ("s3_token_lemmas_key", f"token_lemmas/{index}.csv", token_lemmas),
        ("s3_token_tags_key", f"token_tags/{index}.csv", token_tags),
    ]
    for field, s3_key, rows in s3_writes:
        upload_object(
            session,
            s3_key,
            csv_bytes(rows),
            "text/csv; charset=utf-8",
        )
        table.update_entry(index, field, s3_key)


if __name__ == "__main__":
    tokenize(get_index())
