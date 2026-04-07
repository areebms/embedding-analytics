FROM public.ecr.aws/lambda/python:3.13 AS base

COPY functions/align-kvectors/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY ./shared shared
COPY functions/align-kvectors/src/ .


FROM base AS local
ENTRYPOINT []

FROM base AS lambda
CMD ["app.handler"]
