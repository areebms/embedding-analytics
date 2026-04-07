FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/api/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY ./shared shared
COPY functions/api/src/ .

FROM base AS local
ENTRYPOINT []
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS lambda
CMD ["app.handler"]
