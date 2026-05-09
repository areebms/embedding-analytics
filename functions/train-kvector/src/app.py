import logging

from main import train_and_upload_kvector
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Generate model request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Generate model request missing index")
        raise ValueError("index is required")
    
    seed = event.get("seed")

    if seed is None:
        logger.warning("Generate model request missing seed")
        raise ValueError("seed is required")

    train_and_upload_kvector(index, seed)
    return {"index": index, "seed": seed}
