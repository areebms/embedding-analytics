"""The synthesized stack, built once, against a pinned environment.

The environment is set before `import config`, and with `update` rather than
`setdefault`, because config reads ENV_PREFIX at import time and its
`load_dotenv(override=False)` lets anything already set win. That is the point: the
deploy gate runs this suite in a shell where the real .env is present, and a setdefault
would hand these tests the production bucket and table names.
"""

import os

import pytest

os.environ.update(
    ENV_PREFIX="test-prefix",
    LAMBDA_ROLE_ARN="arn:aws:iam::000000000000:role/test-lambda",
    STEP_FUNCTION_ROLE_ARN="arn:aws:iam::000000000000:role/test-sfn",
    PUT_EVENT_ROLE_ARN="arn:aws:iam::000000000000:role/test-events",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
    ANTHROPIC_API_KEY="test-key",
    CDK_DEFAULT_ACCOUNT="000000000000",
    CDK_DEFAULT_REGION="us-east-1",
)

import app  # noqa: E402
import config  # noqa: E402
from aws_cdk import assertions  # noqa: E402


@pytest.fixture(scope="session")
def stack():
    """The one stack app.build() puts in the app, found by the name it is given."""
    built = app.build()
    return built.node.find_child(config.PREFIX)


@pytest.fixture(scope="session")
def template(stack):
    return assertions.Template.from_stack(stack)


@pytest.fixture(scope="session")
def resources(template) -> dict:
    return template.to_json()["Resources"]
