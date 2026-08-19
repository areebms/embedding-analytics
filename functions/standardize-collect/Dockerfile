FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/standardize-collect/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY ./shared shared

# The handoff models, the Anthropic client, and the response contract are all
# standardize-headings', shared with this stage. Two modules come along inside
# book_records/ that this image cannot run: html_text_tags.py needs BeautifulSoup and
# utils.py needs it transitively, neither of which is installed here. Both stay inert
# -- nothing in this stage imports them, the package __init__ is empty, and the
# modules the stage does import reach neither. book_records/constants.py is
# stdlib-only for that reason: llm_classify_request/schemas.py imports it, so a
# dependency there would land in this image.
COPY functions/standardize-headings/src/book_records book_records
COPY functions/standardize-headings/src/llm_classify_request llm_classify_request
COPY functions/standardize-headings/src/llm_parse_response llm_parse_response

COPY functions/standardize-collect/src/ .

FROM base AS local
ENTRYPOINT []

FROM base AS lambda
CMD ["app.handler"]
