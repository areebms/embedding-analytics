FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/standardize-headings/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt

FROM base AS lambda
COPY shared/*.py shared/
COPY shared/tables shared/tables
COPY functions/standardize-headings/src/ .
CMD ["app.handler"]

FROM lambda AS local
ENTRYPOINT []

FROM lambda AS test
COPY functions/standardize-headings/requirements-test.txt /requirements-test.txt
RUN pip install --no-cache-dir -r /requirements-test.txt
COPY functions/standardize-headings/tests tests
COPY functions/standardize-headings/pytest.ini pytest.ini
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-v"]
