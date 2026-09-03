#!/usr/bin/env python3.13
import aws_cdk as cdk

import config
from resources import (
    build_function_from_container,
    get_role,
    build_standardize_trigger,
    build_state_machine,
    build_tokenize_trigger,
)

# The services CloudFormation owns. Adding one here is what deploys it; deploy.py then
# picks it up off the synthesized assembly and gates its suite.
DEPLOYED = ["scrape", "standardize-html", "tokenize"]


def build(outdir: str | None = None) -> cdk.App:

    app = cdk.App(outdir=outdir)
    stack = cdk.Stack(app, config.PREFIX)

    lambda_role = get_role(stack, "LambdaRole", "LAMBDA_ROLE_ARN")
    sfn_role = get_role(stack, "SfnRole", "STEP_FUNCTION_ROLE_ARN")
    rule_role = get_role(stack, "RuleRole", "PUT_EVENT_ROLE_ARN")

    functions = {}
    for name in DEPLOYED:
        functions[name] = build_function_from_container(stack, service=name, role=lambda_role)

    build_state_machine(stack, machine="scrape", role=sfn_role, calls=functions["scrape"])
    standardize_machine = build_state_machine(
        stack,
        machine="standardize-html",
        role=sfn_role,
        calls=functions["standardize-html"],
    )

    build_standardize_trigger(stack, role=rule_role, standardize=standardize_machine)
    build_tokenize_trigger(stack, tokenize=functions["tokenize"])

    return app


if __name__ == "__main__":
    build().synth()
