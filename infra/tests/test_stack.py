"""What the synthesized stack must look like.

Each of these is an invariant docs/operations.md states in prose and nothing checked.
"""

import json
import os
from collections import Counter

import pytest

import app
import config
from conftest import FUNCTION, MACHINE, PERMISSION, RULE, by_name, get_att, of_type


def announced_events(stage: str) -> set[tuple[str, str]]:
    """Every (source, detail-type) a stage's ASL puts on the bus."""
    definition = json.loads((config.ASL_DIR / f"{stage}.asl.json").read_text())
    found = set()

    def walk(node):
        if isinstance(node, dict):
            if "Source" in node and "DetailType" in node:
                found.add((node["Source"], node["DetailType"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(definition)
    return found


def test_stack_holds_exactly_these_resources(resources):
    """The eight of docs/operations.md: three Lambdas, two machines, two rules and one
    permission -- and nothing else. A stray construct shows up here."""
    assert Counter(r["Type"] for r in resources.values()) == {
        FUNCTION: 3,
        MACHINE: 2,
        RULE: 2,
        PERMISSION: 1,
    }


def test_one_function_per_deployed_service(resources):
    assert set(by_name(resources, FUNCTION, "FunctionName")) == {
        f"{config.PREFIX}-{service}" for service in app.DEPLOYED
    }


@pytest.mark.parametrize("service", app.DEPLOYED)
def test_function_is_sized_from_services_yaml(resources, service):
    """A `default:` quietly winning over the service's own entry is caught here."""
    function = by_name(resources, FUNCTION, "FunctionName")[f"{config.PREFIX}-{service}"]
    declared = config.service(service)

    assert function["Properties"]["MemorySize"] == declared["memory"]
    assert function["Properties"]["Timeout"] == declared["timeout"]


@pytest.mark.parametrize("service", app.DEPLOYED)
def test_function_environment_is_exactly_the_declared_names(resources, service):
    """services.yaml lists the names; nothing else may reach the running function.

    AWS_REGION in particular is reserved -- Lambda injects it and rejects it as a key.
    """
    function = by_name(resources, FUNCTION, "FunctionName")[f"{config.PREFIX}-{service}"]
    variables = function["Properties"].get("Environment", {}).get("Variables", {})

    assert set(variables) == set(config.service(service).get("env", []))


def test_machines_are_named_for_their_stage(resources):
    """build_state_machine derives the name and the ASL filename from one argument, so
    a machine cannot end up named for one stage and defined by another."""
    machines = by_name(resources, MACHINE, "StateMachineName")

    assert set(machines) == {
        f"{config.PREFIX}-scrape",
        f"{config.PREFIX}-standardize-html",
    }
    for name in machines:
        stage = name.removeprefix(f"{config.PREFIX}-")
        assert (config.ASL_DIR / f"{stage}.asl.json").is_file()


def test_machines_resolve_their_function_by_reference(resources):
    """FUNCTION_ARN is a GetAtt on the real construct, not an ARN rebuilt from region,
    account and prefix -- the six-stack hazard docs/operations.md describes, where
    nothing verified that the ARN resolved and delivery failed silently."""
    functions = of_type(resources, FUNCTION)

    for name, machine in by_name(resources, MACHINE, "StateMachineName").items():
        stage = name.removeprefix(f"{config.PREFIX}-")
        substitutions = machine["Properties"]["DefinitionSubstitutions"]
        target = functions[get_att(substitutions["FUNCTION_ARN"])]

        assert target["Properties"]["FunctionName"] == f"{config.PREFIX}-{stage}"


def test_triggers_match_the_events_the_stages_announce(resources):
    """Every event a machine puts on the bus has a rule listening for it, and no rule
    listens for an event nothing sends. The ASL and the rule are edited separately, and
    a mismatch is delivered to nothing."""
    announced = set()
    for name in by_name(resources, MACHINE, "StateMachineName"):
        announced |= announced_events(name.removeprefix(f"{config.PREFIX}-"))

    listened = set()
    for rule in of_type(resources, RULE).values():
        pattern = rule["Properties"]["EventPattern"]
        assert len(pattern["source"]) == 1 and len(pattern["detail-type"]) == 1
        listened.add((pattern["source"][0], pattern["detail-type"][0]))

    assert listened == announced


def test_the_two_triggers_authorise_differently(resources):
    """A Step Functions target is authorised by the rule's role; a Lambda target is
    authorised by a resource policy on the function and takes no role."""
    rules = by_name(resources, RULE, "Name")
    standardize = rules[f"{config.PREFIX}-standardize-trigger"]
    tokenize = rules[f"{config.PREFIX}-tokenize-trigger"]

    (to_machine,) = standardize["Properties"]["Targets"]
    (to_function,) = tokenize["Properties"]["Targets"]

    assert to_machine["RoleArn"] == os.environ["PUT_EVENT_ROLE_ARN"]
    assert "RoleArn" not in to_function


def test_the_lambda_target_carries_its_invoke_permission(resources):
    """The permission CDK emits only because the rule targets a construct in this stack.
    Under the six-stack layout the function was imported by ARN, CDK skipped the
    permission with a warning rather than an error, and every delivery was refused."""
    (permission,) = of_type(resources, PERMISSION).values()
    properties = permission["Properties"]
    tokenize = of_type(resources, FUNCTION)[get_att(properties["FunctionName"])]
    rule = of_type(resources, RULE)[get_att(properties["SourceArn"])]

    assert properties["Action"] == "lambda:InvokeFunction"
    assert properties["Principal"] == "events.amazonaws.com"
    assert tokenize["Properties"]["FunctionName"] == f"{config.PREFIX}-tokenize"
    assert rule["Properties"]["Name"] == f"{config.PREFIX}-tokenize-trigger"
