#!/usr/bin/env python3.13
import aws_cdk as cdk

import config
from resources import (
    build_create_embeddings_trigger,
    build_function_from_container,
    get_role,
    build_publish_trigger,
    build_standardize_trigger,
    build_state_machine,
    build_tokenize_trigger,
)

def build(outdir: str | None = None) -> cdk.App:

    app = cdk.App(outdir=outdir)
    stack = cdk.Stack(app, config.PREFIX)

    lambda_role = get_role(stack, "LambdaRole", "LAMBDA_ROLE_ARN")
    sfn_role = get_role(stack, "SfnRole", "STEP_FUNCTION_ROLE_ARN")
    rule_role = get_role(stack, "RuleRole", "PUT_EVENT_ROLE_ARN")

    functions = {}
    for name in config.get_services():
        functions[name] = build_function_from_container(
            stack, service=name, role=lambda_role
        )

    build_state_machine(
        stack, machine="scrape", role=sfn_role, calls=functions["scrape"]
    )
    standardize_machine = build_state_machine(
        stack,
        machine="standardize-html",
        role=sfn_role,
        calls=functions["standardize-html"],
    )

    build_standardize_trigger(stack, role=rule_role, standardize=standardize_machine)
    build_tokenize_trigger(stack, tokenize=functions["tokenize"])
    build_create_embeddings_trigger(
        stack, create_embeddings=functions["create-embeddings"]
    )
    build_publish_trigger(stack, publish=functions["publish"])

    return app


if __name__ == "__main__":
    build().synth()
