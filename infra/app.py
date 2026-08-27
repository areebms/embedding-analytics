#!/usr/bin/env python3.13
import os

import aws_cdk as cdk

from stacks.pipeline_stack import PipelineStack
from stacks.relay_stack import RelayStack

app = cdk.App()

# -c prefix=<name> to allow parallel deployment.
prefix = app.node.try_get_context("prefix") or os.environ["LAMBDA_PREFIX"]

env = cdk.Environment(
    account=os.environ["AWS_ACCOUNT_ID"], region=os.environ["AWS_REGION"]
)

PipelineStack(
    app,
    f"{prefix}-scrape",
    prefix=prefix,
    service="scrape",
    machine="scrape-pipeline",
    env=env,
)

standardize = PipelineStack(
    app,
    f"{prefix}-standardize",
    prefix=prefix,
    service="standardize-headings",
    machine="standardize",
    env=env,
)

relay = RelayStack(app, f"{prefix}-relay", prefix=prefix, env=env)

# For deploy ordering.
relay.add_stack_dependency(
    standardize, "the rule targets the standardize machine by ARN"
)

app.synth()
