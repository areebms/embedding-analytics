import argparse
import logging
from datetime import datetime
from pathlib import Path
from random import randint

from gensim.models import Word2Vec

from shared.aws import get_pipeline_table, get_session, upload_file, yield_sentences_from_s3
from shared.commons import get_index

VECTOR_SIZE = 200
S3_SUBDIR = "collected"
EPOCHS = 30
MIN_TOKEN_SIZE = 3
MIN_COUNT = 10


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


sentences = {}
def _get_sentences(session, table, index):
    global sentences

    if index in sentences:
        return sentences[index]

    item = table.get_entry(index, ["s3_token_lemmas_key"])
    s3_token_lemmas_key = item.get("s3_token_lemmas_key")

    if not s3_token_lemmas_key:
        print(f"{index} has not been tokenized.")
        exit()

    sentences[index] = []
    for sentence in yield_sentences_from_s3(session, s3_token_lemmas_key):
        sentences[index].append(
            [word for word in sentence if word.isalpha() and len(word) > MIN_TOKEN_SIZE]
        )

    return sentences[index]


def train_kvector(session, table, index, seed):

    models_dir = Path("/tmp/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    return Word2Vec(
        _get_sentences(session, table, index),
        vector_size=VECTOR_SIZE,
        window=10,
        min_count=MIN_COUNT,
        workers=1,
        sg=1,
        hs=1,
        sample=5e-4,
        negative=0,
        epochs=EPOCHS,
        seed=seed,
    ).wv


def upload_kvector(session, index, kvector, seed):

    models_dir = Path("/tmp/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = models_dir / f"{seed}-{timestamp}-{randint(0, 3600)}.model"

    kvector.save(str(save_path))
    logger.info("Model generation completed", extra={"index": index})

    upload_file(session, f"kvectors/{index}/{S3_SUBDIR}/{save_path.name}", save_path)


def train_and_upload_kvector(index, seed):
    session, table = get_session(), get_pipeline_table()
    kvector = train_kvector(session, table, index, seed)
    upload_kvector(session, index, kvector, seed)


if __name__ == "__main__":
    index = get_index()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    args, _ = parser.parse_known_args()

    train_and_upload_kvector(index, int(args.seed))
