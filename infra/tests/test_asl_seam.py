import json

import pytest

import config
from conftest import MACHINE, RULE, by_name, of_type
from pipeline_events import BOOKS_STANDARDIZED, SUBJECT_BOOKS_SCRAPED

INTEGRATION_ACTIONS = {
    "arn:aws:states:::lambda:invoke": "lambda:InvokeFunction",
    "arn:aws:states:::events:putEvents": "events:PutEvents",
}

EXPECTED_PERMISSIONS = {
    "scrape": {
        ("lambda:InvokeFunction", "${FUNCTION_ARN}:$LATEST"),
        ("events:PutEvents", "default"),
    },
    "standardize-html": {
        ("lambda:InvokeFunction", "${FUNCTION_ARN}:$LATEST"),
        ("events:PutEvents", "default"),
    },
}

EXPECTED_EVENTS = {
    (SUBJECT_BOOKS_SCRAPED.source, SUBJECT_BOOKS_SCRAPED.detail_type): (
        SUBJECT_BOOKS_SCRAPED
    ),
    (BOOKS_STANDARDIZED.source, BOOKS_STANDARDIZED.detail_type): BOOKS_STANDARDIZED,
}


def definition(stage: str) -> dict:
    return json.loads((config.ASL_DIR / f"{stage}.asl.json").read_text())


def tasks(stage: str):
    """Every Task state in a stage's definition, Map iterations included."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("Type") == "Task":
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(definition(stage))
    return found


def required_permissions(stage: str) -> set[tuple[str, str]]:
    """Every (action, target) a stage's definition asks its role to perform."""
    needed = set()

    for task in tasks(stage):
        resource = task["Resource"]
        action = INTEGRATION_ACTIONS.get(resource)
        assert action, (
            f"{stage}: unmapped Step Functions integration {resource!r}. It needs an "
            f"IAM action in INTEGRATION_ACTIONS, and the role that runs this machine "
            f"needs that permission adding outside this repo."
        )

        if action == "lambda:InvokeFunction":
            needed.add((action, task["Arguments"]["FunctionName"]))
        else:
            needed.add((action, "default"))

    return needed


def announcements(stage: str) -> list[dict]:
    """Every PutEvents entry a stage's definition puts on the bus."""
    entries = []
    for task in tasks(stage):
        if task["Resource"] == "arn:aws:states:::events:putEvents":
            entries.extend(task["Arguments"]["Entries"])
    return entries


@pytest.fixture(scope="session")
def stages(resources) -> list[str]:
    return [
        name.removeprefix(f"{config.PREFIX}-")
        for name in by_name(resources, MACHINE, "StateMachineName")
    ]


def test_definitions_ask_for_exactly_the_permissions_we_think(stages):
    assert {stage: required_permissions(stage) for stage in stages} == {
        stage: EXPECTED_PERMISSIONS[stage] for stage in stages
    }


def test_every_invoked_function_keeps_its_latest_qualifier(stages):
    for stage in stages:
        for action, target in required_permissions(stage):
            if action == "lambda:InvokeFunction":
                assert target.endswith(":$LATEST"), f"{stage}: {target}"


def test_rules_forward_only_the_event_detail(resources):
    for rule in of_type(resources, RULE).values():
        (target,) = rule["Properties"]["Targets"]
        assert target["InputPath"] == "$.detail", rule["Properties"]["Name"]


def test_announced_detail_matches_the_declared_payload(stages):
    for stage in stages:
        for entry in announcements(stage):
            event = EXPECTED_EVENTS.get((entry["Source"], entry["DetailType"]))
            assert event, (
                f"{stage} announces an event infra/pipeline_events.py has no record "
                f"of: {entry['Source']} / {entry['DetailType']}"
            )

            assert set(entry["Detail"]) == set(event.detail_keys), (
                f"{stage}: {event.detail_type} announces {sorted(entry['Detail'])}, "
                f"but infra/pipeline_events.py declares {sorted(event.detail_keys)}"
            )
