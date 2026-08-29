import csv
import io
import json
import os
import tempfile
from contextlib import contextmanager

from shared.session import get_session

S3_BUCKET = os.getenv("S3_BUCKET")

JSON_CONTENT_TYPE = "application/json; charset=utf-8"
HTML_CONTENT_TYPE = "text/html; charset=utf-8"
TEXT_CONTENT_TYPE = "text/plain; charset=utf-8"
CSV_CONTENT_TYPE = "text/csv; charset=utf-8"
BINARY_CONTENT_TYPE = "application/octet-stream"

_s3_resource = None


def get_s3_resource():
    global _s3_resource
    if _s3_resource is None:
        _s3_resource = get_session().resource("s3")
    return _s3_resource


def load_text(s3_key):
    return (
        get_s3_resource().Object(S3_BUCKET, s3_key).get()["Body"].read().decode("utf-8")
    )


def load_json(s3_key):
    return json.loads(load_text(s3_key))


def load_csv(s3_key):
    body = get_s3_resource().Object(S3_BUCKET, s3_key).get()["Body"]
    yield from csv.reader(io.TextIOWrapper(body, encoding="utf-8"))


@contextmanager
def load_file(s3_key):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        get_s3_resource().Object(S3_BUCKET, s3_key).download_fileobj(tmp_file)
    try:
        yield s3_key, tmp_file.name
    finally:
        os.unlink(tmp_file.name)


def yield_s3_files(s3_prefix, file_extension):
    bucket = get_s3_resource().Bucket(S3_BUCKET)
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        if file_extension not in obj.key:
            continue
        with load_file(obj.key) as result:
            yield result


def upload_object(s3_key, text, content_type):
    get_s3_resource().Object(S3_BUCKET, s3_key).put(
        Body=text.encode("utf-8"), ContentType=content_type
    )


def upload_json(s3_key, text):
    upload_object(s3_key, text, JSON_CONTENT_TYPE)


def upload_html(s3_key, text):
    upload_object(s3_key, text, HTML_CONTENT_TYPE)


def upload_txt(s3_key, text):
    upload_object(s3_key, text, TEXT_CONTENT_TYPE)


def upload_csv(s3_key, rows):
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    upload_object(s3_key, buffer.getvalue(), CSV_CONTENT_TYPE)


def upload_file(s3_key, path):
    get_s3_resource().meta.client.upload_file(
        path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": BINARY_CONTENT_TYPE},
    )
