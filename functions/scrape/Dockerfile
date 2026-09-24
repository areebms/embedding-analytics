FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/scrape/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt

FROM base AS lambda
COPY shared/*.py shared/
COPY shared/tables shared/tables
COPY functions/scrape/src/ .
CMD ["app.handler"]

FROM lambda AS test
COPY functions/scrape/requirements-test.txt /requirements-test.txt
RUN pip install --no-cache-dir -r /requirements-test.txt
COPY functions/scrape/tests tests
COPY functions/scrape/pytest.ini pytest.ini
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-v"]
