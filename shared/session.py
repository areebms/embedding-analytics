import os

from boto3 import Session


AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")

_session = None


def get_session():
    global _session
    if _session is None:
        _session = Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return _session
