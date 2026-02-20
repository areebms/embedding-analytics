FROM public.ecr.aws/lambda/python:3.13 as base

COPY functions/tokenize/requirements.txt /requirements.txt
WORKDIR ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r /requirements.txt -t ${LAMBDA_TASK_ROOT}
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader wordnet omw-1.4
RUN python -c "import nltk; nltk.download('punkt_tab')"

COPY ./shared shared
COPY functions/tokenize/src/ .

FROM base AS local
ENTRYPOINT []

FROM base AS lambda
CMD ["app.handler"]
