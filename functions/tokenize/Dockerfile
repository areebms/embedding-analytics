FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/tokenize/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader wordnet omw-1.4

FROM base AS lambda
COPY shared/*.py shared/
COPY shared/tables shared/tables
COPY functions/tokenize/src/ .
CMD ["app.handler"]

FROM lambda AS local
ENTRYPOINT []

FROM lambda AS test
COPY functions/tokenize/requirements-test.txt /requirements-test.txt
RUN pip install --no-cache-dir -r /requirements-test.txt
COPY functions/tokenize/tests tests
COPY functions/tokenize/pytest.ini pytest.ini
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-v"]
