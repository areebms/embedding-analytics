FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/create-embeddings/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY shared/*.py shared/
COPY shared/tables shared/tables
COPY functions/create-embeddings/src/ .


FROM base AS test
COPY functions/create-embeddings/requirements-test.txt /requirements-test.txt
RUN pip install --no-cache-dir -r /requirements-test.txt
COPY functions/create-embeddings/tests tests
COPY functions/create-embeddings/pytest.ini pytest.ini
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-v"]


FROM base AS local
ENTRYPOINT []


FROM base AS lambda
CMD ["app.handler"]
