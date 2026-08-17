import csv
import io
import os
import tempfile
from contextlib import contextmanager

from shared.session import get_session

S3_BUCKET = os.getenv("S3_BUCKET")



def upload_file(session, s3_key, path):
    session.client("s3").upload_file(
        path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )

_s3_loader = None

def get_s3_loader():
    global _s3_loader
    if _s3_loader is None:
        _s3_loader = S3Loader(get_session())
    return _s3_loader


class S3Loader:

    def __init__(self, session):
        self.s3_resource = session.resource("s3")

    @contextmanager
    def load_file(self, s3_object_key):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self.s3_resource.Object(S3_BUCKET, s3_object_key).download_fileobj(tmp_file)
        try:
            yield s3_object_key, tmp_file.name
        finally:
            os.unlink(tmp_file.name)

    def yield_s3_files(self, s3_prefix, file_extension):
        bucket = self.s3_resource.Bucket(S3_BUCKET)
        for obj in bucket.objects.filter(Prefix=s3_prefix):
            if file_extension not in obj.key:
                continue
            with self.load_file(obj.key) as result:
                yield result

    def load_text(self, s3_key):
        return (
            self.s3_resource.Object(S3_BUCKET, s3_key)
            .get()["Body"]
            .read()
            .decode("utf-8")
        )

    def upload_object(self, s3_key, file_bytes, content_type):
        self.s3_resource.meta.client.upload_fileobj(
            io.BytesIO(file_bytes.encode("utf-8")),
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )


def yield_sentences_from_s3(session, s3_key):
    body = session.resource("s3").Object(S3_BUCKET, s3_key).get()["Body"]
    yield from csv.reader(io.TextIOWrapper(body, encoding="utf-8"))
