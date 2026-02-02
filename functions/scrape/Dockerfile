FROM comparative-nuance-engine/lambda-base:0.1

COPY functions/scrape/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY functions/scrape/src/ .
CMD ["app.handler"]
