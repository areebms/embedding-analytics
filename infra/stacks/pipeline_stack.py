from aws_cdk import (
    Duration,
    Stack,
    aws_ecr_assets,
    aws_lambda,
    aws_stepfunctions,
)
from constructs import Construct

import config


def _asset_exclude(name: str) -> list[str]:

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


class PipelineStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *,
        prefix: str, service: str, machine: str, **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        lambda_role = config.get_role(self, "LambdaRole", "LAMBDA_ROLE_ARN")
        sfn_role = config.get_role(self, "SfnRole", "STEP_FUNCTION_ROLE_ARN")

        svc = config.service(service)

        function = aws_lambda.DockerImageFunction(
            self, "Function",
            function_name=f"{prefix}-{service}",
            code=aws_lambda.DockerImageCode.from_image_asset(
                directory=str(config.REPO_ROOT),
                file=f"functions/{service}/Dockerfile",
                target="lambda",
                platform=aws_ecr_assets.Platform.LINUX_AMD64,
                exclude=_asset_exclude(service),
            ),
            role=lambda_role,
            memory_size=svc["memory"],
            timeout=Duration.seconds(svc["timeout"]),
            environment=config.env_for(service),
        )

        state_machine = aws_stepfunctions.StateMachine(
            self, "Machine",
            state_machine_name=f"{prefix}-{machine}",
            definition_body=aws_stepfunctions.DefinitionBody.from_file(
                str(config.ASL_DIR / f"{machine}.asl.json")
            ),
            definition_substitutions={
                "AWS_REGION": self.region,
                "AWS_ACCOUNT_ID": self.account,
                "LAMBDA_PREFIX": prefix,
            },
            role=sfn_role,
        )
        # The ARNs above are plain strings, so nothing tells CDK the machine needs its
        # Lambda first.
        state_machine.node.add_dependency(function)
