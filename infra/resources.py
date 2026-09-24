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
from pipeline_events import (
    BOOKS_EMBEDDINGS_CREATED,
    BOOKS_STANDARDIZED,
    BOOKS_TOKENIZED,
    SUBJECT_BOOKS_SCRAPED,
    PipelineEvent,
)


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

    TODO: ANTHROPIC_API_KEY, OPENAI_API_KEY and REDIS_URL are read here and baked into the synthesized template, which cdk deploy uploads to the CDK
    staging bucket. Move the secrets to SSM and reference them with
    ssm.StringParameter.value_for_string_parameter, so only the parameter name lands in
    the template.
    """
    service_envs = config.get_service_config(service, "env")
    missing_envs = [env for env in service_envs if not os.getenv(env)]
    if missing_envs:
        raise SystemExit(f"missing in .env: {', '.join(missing_envs)}")

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
        memory_size=config.get_service_config(service, "memory"),
        timeout=Duration.seconds(config.get_service_config(service, "timeout")),
        environment={env: os.environ[env] for env in service_envs},
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
    *,
    event: PipelineEvent,
    consumer: str,
    description: str,
    target: events.IRuleTarget,
) -> events.Rule:
    return events.Rule(
        scope,
        f"{consumer.title()}Trigger",
        rule_name=f"{config.PREFIX}-{consumer}-trigger",
        description=description,
        event_pattern=events.EventPattern(
            source=[event.source], detail_type=[event.detail_type]
        ),
        targets=[target],
    )


def build_standardize_trigger(
    scope: Construct, *, role: aws_iam.IRole, standardize: sfn.IStateMachine
) -> events.Rule:
    return build_event_rule(
        scope,
        event=SUBJECT_BOOKS_SCRAPED,
        consumer="standardize",
        description="Turns a 'Subject Books Scraped' event into a standardize execution.",
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
        event=BOOKS_STANDARDIZED,
        consumer="tokenize",
        description="Turns a 'Books Standardized' event into a tokenize invocation.",
        target=targets.LambdaFunction(
            tokenize,
            event=events.RuleTargetInput.from_event_path("$.detail"),
        ),
    )


def build_create_embeddings_trigger(
    scope: Construct, *, create_embeddings: aws_lambda.IFunction
) -> events.Rule:
    return build_event_rule(
        scope,
        event=BOOKS_TOKENIZED,
        consumer="create-embeddings",
        description="Turns a 'Books Tokenized' event into a create-embeddings invocation.",
        target=targets.LambdaFunction(
            create_embeddings,
            event=events.RuleTargetInput.from_event_path("$.detail"),
        ),
    )


def build_publish_trigger(
    scope: Construct, *, publish: aws_lambda.IFunction
) -> events.Rule:
    return build_event_rule(
        scope,
        event=BOOKS_EMBEDDINGS_CREATED,
        consumer="publish",
        description="Turns a 'Books Embedded' event into a publish invocation.",
        target=targets.LambdaFunction(
            publish,
            event=events.RuleTargetInput.from_event_path("$.detail"),
        ),
    )
