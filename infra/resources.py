import os

from aws_cdk import (
    Duration,
    aws_ecr_assets,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam,
    aws_lambda,
    aws_stepfunctions as sfn,
)
from constructs import Construct

import config


def get_test_files(name: str) -> list[str]:

    return sorted(
        [
            f"functions/{d.name}"
            for d in (config.REPO_ROOT / "functions").iterdir()
            if d.is_dir() and d.name != name
        ]
        + [
            f"functions/{name}/tests",
            f"functions/{name}/pytest.ini",
            f"functions/{name}/requirements-test.txt",
            "shared/tests",
        ]
    )


def get_role(scope: Construct, construct_id: str, env_var: str) -> aws_iam.IRole:
    return aws_iam.Role.from_role_arn(
        scope, construct_id, os.environ[env_var], mutable=False
    )


def build_function_from_container(
    scope: Construct,
    *,
    service: str,
    role: aws_iam.IRole,
) -> aws_lambda.DockerImageFunction:
    """One service's Lambda, from the `lambda` stage of its Dockerfile.
    """
    svc = config.service(service)

    return aws_lambda.DockerImageFunction(
        scope,
        f"{service}-function",
        function_name=f"{config.PREFIX}-{service}",
        code=aws_lambda.DockerImageCode.from_image_asset(
            directory=str(config.REPO_ROOT),
            file=f"functions/{service}/Dockerfile",
            target="lambda",
            platform=aws_ecr_assets.Platform.LINUX_AMD64,
            exclude=get_test_files(service),
        ),
        role=role,
        memory_size=svc["memory"],
        timeout=Duration.seconds(svc["timeout"]),
        environment=config.env_for(service),
    )


def build_state_machine(
    scope: Construct,
    *,
    machine: str,
    role: aws_iam.IRole,
    calls: aws_lambda.IFunction,
) -> sfn.StateMachine:

    return sfn.StateMachine(
        scope,
        f"{machine}-pipeline",
        state_machine_name=f"{config.PREFIX}-{machine}",
        definition_body=sfn.DefinitionBody.from_file(
            str(config.ASL_DIR / f"{machine}.asl.json")
        ),
        definition_substitutions={
            "FUNCTION_ARN": calls.function_arn,
            "ENV_PREFIX": config.PREFIX,
        },
        role=role,
    )


def build_event_rule(
    scope: Construct,
    construct_id: str,
    *,
    rule_name: str,
    description: str,
    source: str,
    detail_type: str,
    target: events.IRuleTarget,
) -> events.Rule:
    """One event, one target. `$.detail` is the payload in both directions."""
    return events.Rule(
        scope,
        construct_id,
        rule_name=rule_name,
        description=description,
        event_pattern=events.EventPattern(source=[source], detail_type=[detail_type]),
        targets=[target],
    )


def build_standardize_trigger(
    scope: Construct, *, role: aws_iam.IRole, standardize: sfn.IStateMachine
) -> events.Rule:
    return build_event_rule(
        scope,
        "StandardizeTrigger",
        rule_name=f"{config.PREFIX}-standardize-trigger",
        description="Turns a 'Subject Books Scraped' event into a standardize execution.",
        source="embedding-analytics.scrape",
        detail_type="Subject Books Scraped",
        target=targets.SfnStateMachine(
            standardize,
            role=role,
            input=events.RuleTargetInput.from_event_path("$.detail"),
        ),
    )


def build_tokenize_trigger(
    scope: Construct, *, tokenize: aws_lambda.IFunction
) -> events.Rule:
    return build_event_rule(
        scope,
        "TokenizeTrigger",
        rule_name=f"{config.PREFIX}-tokenize-trigger",
        description="Turns a 'Books Standardized' event into a tokenize invocation.",
        source="embedding-analytics.standardize",
        detail_type="Books Standardized",
        target=targets.LambdaFunction(
            tokenize,
            event=events.RuleTargetInput.from_event_path("$.detail"),
        ),
    )
