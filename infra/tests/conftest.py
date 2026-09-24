import os

import pytest

os.environ.update(
    ENV_PREFIX="test-prefix",
    LAMBDA_ROLE_ARN="arn:aws:iam::000000000000:role/test-lambda",
    STEP_FUNCTION_ROLE_ARN="arn:aws:iam::000000000000:role/test-sfn",
    PUT_EVENT_ROLE_ARN="arn:aws:iam::000000000000:role/test-events",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
    BOOK_TERM_TABLE="book-terms-test",
    TERM_CORPUS_TABLE="corpus-terms-test",
    ANTHROPIC_API_KEY="test-key",
    PRODUCTION_DOMAIN="https://test.example",
    OPENAI_API_KEY="test-key",
    REDIS_URL="redis://test:6379",
    REDIS_PREFIX="test-prefix",
    CDK_DEFAULT_ACCOUNT="000000000000",
    CDK_DEFAULT_REGION="us-east-1",
)

import app  # noqa: E402
import config  # noqa: E402
from aws_cdk import assertions  # noqa: E402

FUNCTION = "AWS::Lambda::Function"
MACHINE = "AWS::StepFunctions::StateMachine"
RULE = "AWS::Events::Rule"
PERMISSION = "AWS::Lambda::Permission"


def of_type(resources: dict, type_name: str) -> dict:
    """{logical id: resource} for one CloudFormation type."""
    return {lid: r for lid, r in resources.items() if r["Type"] == type_name}


def by_name(resources: dict, type_name: str, key: str) -> dict:
    """One type indexed by the physical name it carries in `key`."""
    return {r["Properties"][key]: r for r in of_type(resources, type_name).values()}


def get_att(value: dict) -> str:
    """The logical id an `Fn::GetAtt` points at, or fail saying what it was instead."""
    assert isinstance(value, dict) and "Fn::GetAtt" in value, (
        f"expected a reference to a construct in this stack, got {value!r}"
    )
    return value["Fn::GetAtt"][0]


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
