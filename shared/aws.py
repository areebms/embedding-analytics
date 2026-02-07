import csv
import io
import os
import json

from boto3 import Session

from botocore.exceptions import ClientError


AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_BUCKET = os.getenv("S3_BUCKET")
PIPELINE_TABLE = os.getenv("PIPELINE_TABLE")


def get_session():
    return Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


class PipelineTable:

    def __init__(self, session):
        self.table = session.resource("dynamodb").Table(PIPELINE_TABLE)

    def update_entry(self, index, field, value):
        self.table.update_item(
            Key={"platform_data": index},
            UpdateExpression=f"SET {field} = :{field}",
            ExpressionAttributeValues={f":{field}": value},
        )

    def get(self, index, expression="platform_data"):
        params = {"Key": {"platform_data": index}}
        if expression is not None:
            params["ProjectionExpression"] = expression
        return self.table.get_item(**params).get("Item")

    def put(self, index):
        item = {"platform_data": index}
        try:
            self.table.put_item(
                Item=item, ConditionExpression="attribute_not_exists(platform_data)"
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise


def upload_object(
    session, s3_key, file_bytes, content_type="text/plain; charset=utf-8"
):
    session.resource("s3").Object(S3_BUCKET, s3_key).put(
        Body=file_bytes, ContentType=content_type
    )


def load_bytes_from_s3(session, s3_key):
    response = session.resource("s3").Object(S3_BUCKET, s3_key).get()
    return response["Body"].read()


def load_text_from_s3(session, s3_key):
    return load_bytes_from_s3(session, s3_key).decode("utf-8")


def get_keys_with_prefix(session, s3_prefix):
    bucket = session.resource("s3").Bucket(S3_BUCKET)
    return [obj.key for obj in bucket.objects.filter(Prefix=s3_prefix)]


def extract_index(event):
    if not isinstance(event, dict):
        return None

    if "index" in event:
        return event["index"]

    body = event.get("body")
    if not body:
        return None

    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(payload, dict):
        return payload.get("index")

    return None


def yield_sentences_from_s3(session, s3_key):
    yield from csv.reader(io.StringIO(load_text_from_s3(session, s3_key)))
