"""EventBridge-based relay between the pipelines
"""
from aws_cdk import (
    ArnFormat,
    Stack,
    aws_events as events,
    aws_events_targets as targets,
    aws_stepfunctions as sfn,
)
from constructs import Construct

import config


class RelayStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, prefix: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        rule_role = config.get_role(self, "RuleRole", "PUT_EVENT_ROLE_ARN")

        standardize = sfn.StateMachine.from_state_machine_arn(
            self, "StandardizeMachine",
            self.format_arn(
                service="states",
                resource="stateMachine",
                resource_name=f"{prefix}-standardize",
                arn_format=ArnFormat.COLON_RESOURCE_NAME,
            ),
        )

        events.Rule(
            self, "StandardizeTrigger",
            rule_name=f"{prefix}-standardize-trigger",
            description="Turns a 'Subject Books Scraped' event into a standardize execution.",
            event_pattern=events.EventPattern(
                source=["embedding-analytics.scrape"],
                detail_type=["Subject Books Scraped"],
            ),
            targets=[
                targets.SfnStateMachine(
                    standardize,
                    role=rule_role,
                    # from_object emits both placeholders unquoted, where the
                    # hand-written rule quoted subject and left book_ids bare. Same
                    # result: EventBridge adds quotes to string variables to keep the
                    # output valid JSON, and leaves objects/arrays alone. Quoting a
                    # string is permitted, not required -- do not "fix" this back.
                    input=events.RuleTargetInput.from_object({
                        "book_ids": events.EventField.from_path("$.detail.book_ids"),
                        "subject": events.EventField.from_path("$.detail.subject"),
                    }),
                )
            ],
        )
