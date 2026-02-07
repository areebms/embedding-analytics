import os
import logging
from datetime import datetime
from pathlib import Path
from random import randint

from gensim.models import Word2Vec

from shared.aws import (
    PipelineTable,
    get_session,
    yield_sentences_from_s3,
    upload_object,
)
from shared.commons import get_index

VECTOR_SIZE = 200
EPOCHS = 30
MIN_TOKEN_SIZE = 3


_session = None
_table = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_session_and_table():
    global _session, _table
    if _session is None:
        _session = get_session()
    if _table is None:
        _table = PipelineTable(_session)
    return _session, _table


def generate_kvector(index, sentences):

    models_dir = Path("/tmp/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    aws_allocated_memory = int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", 0))
    vcpu = aws_allocated_memory / 1769.0   # AWS proportional allocation
    workers = max(1, min(6, int(vcpu)))
    print("Calculated Workers:", workers)
    print("os.cpu_count()", os.cpu_count())

    # never exceed what python thinks exists
    workers = min(workers, os.cpu_count() or 1) 

    wv = Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=10,
        min_count=2,
        workers=workers,
        sg=1,
        hs=1,
        sample=0,
        negative=0,
        epochs=EPOCHS,
    ).wv

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = models_dir / f"{timestamp}-{randint(0, 3600)}.model"

    wv.save(str(save_path))
    logger.info("Model generation completed", extra={"index": index})

    s3_key = f"word_vectors/{index}/{save_path.name}"
    session, table = _get_session_and_table()
    upload_object(session, s3_key, save_path.read_bytes(), "application/octet-stream")

    table.update_entry(index, "s3_word_vectors_prefix", f"word_vectors/{index}/")
    print(f"{save_path.name} uploaded.")


if __name__ == "__main__":

    index = get_index()
    session = get_session()
    table = PipelineTable(session)

    item = table.get(index, expression="platform_data,s3_token_lemmas_key")

    s3_token_lemmas_key = item.get("s3_token_lemmas_key")
    if not s3_token_lemmas_key:
        print(f"{index} has not been tokenized.")
        exit()

    sentences = []
    for sentence in yield_sentences_from_s3(session, s3_token_lemmas_key):
        sentences.append(
            [word for word in sentence if word.isalpha() and len(word) > MIN_TOKEN_SIZE]
        )

    generate_kvector(index, sentences)
