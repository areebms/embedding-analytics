import logging

from collect import collect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Standardize collect request received", extra={"event": event})
    batch_id = (event or {}).get("batch_id")
    if not batch_id:
        logger.warning("Standardize collect request missing batch_id")
        raise ValueError("batch_id is required")

    summary = collect(batch_id)
    logger.info("Standardize collect completed", extra=summary)
    return summary
