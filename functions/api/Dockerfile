FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/api/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./shared shared
COPY functions/api/src/ .


# Test stage: production deps + pytest + tests. Built and executed
# by deploy_lambdas.sh as the gate before docker push. Not deployed
# to Lambda; the 'lambda' stage below is what ships.
FROM base AS test
COPY functions/api/requirements-test.txt /requirements-test.txt
RUN pip install --no-cache-dir -r /requirements-test.txt
COPY functions/api/tests tests
COPY functions/api/pytest.ini pytest.ini
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-v"]


FROM base AS local
RUN dnf install -y tar gzip && dnf clean all
ENTRYPOINT []
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS lambda
CMD ["main.handler"]
