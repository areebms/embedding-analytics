import logging

from llm_parse_response.standardize import standardize_from_batch
from shared.lambda_event import extract_field
from submit import submit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STAGES = ("collect", "submit")


def handler(event, context):
    """Run one standardize stage. Both are corpus-wide, invoked by hand."""
    logger.info("Standardize request received", extra={"event": event})

    stage = (event or {}).get("stage")
    if stage not in STAGES:
        logger.warning(
            "Standardize request has no runnable stage", extra={"stage": stage}
        )
        raise ValueError(f"stage must be one of {list(STAGES)}")

    if stage == "submit":
        status = submit()
    else:
        batch_id = extract_field(event, "batch_id")
        if not batch_id:
            logger.warning("Standardize collect request missing batch_id")
            raise ValueError("batch_id is required")

        status = standardize_from_batch(batch_id)

    logger.info("Standardize completed", extra={"stage": stage, **status})
    return status
