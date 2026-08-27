"""resolves services.yaml + .env, and imports the IAM roles named there"""

import os
from pathlib import Path

import yaml
from aws_cdk import aws_iam
from constructs import Construct
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
INFRA_DIR = REPO_ROOT / "infra"
SERVICES_FILE = INFRA_DIR / "services.yaml"
ASL_DIR = INFRA_DIR / "step-functions"

load_dotenv(REPO_ROOT / ".env", override=False)

_services_doc = yaml.safe_load(SERVICES_FILE.read_text())
_defaults = _services_doc.get("default", {})


def service(name: str) -> dict:
    """One services.yaml entry with `default:` filled in."""
    try:
        entry = _services_doc["services"][name] or {}
    except KeyError:
        raise SystemExit(f"no such service in {SERVICES_FILE}: {name}") from None
    return {**_defaults, **entry}


def env_for(name: str) -> dict[str, str]:
    """The service's Lambda environment.

    services.yaml lists variable NAMES; the values come from .env, so no secret lives
    in the repo. A name with no value aborts the synth rather than producing a function
    that is half-configured and fails at runtime -- same contract as env_json() in the
    deploy_lambdas.sh this replaces.

    AWS_REGION is never listed: Lambda injects it into every runtime and rejects it as
    a reserved key.

    TODO: ANTHROPIC_API_KEY (and OPENAI/PINECONE, when those services convert) is read
    here and baked into the synthesized template, which cdk deploy uploads to the CDK
    staging bucket. Move the secrets to SSM and reference them with
    ssm.StringParameter.value_for_string_parameter, so only the parameter name lands in
    the template. See docs/operations.md, "Configuration and deployment".
    """
    names = service(name).get("env", [])
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def get_role(scope: Construct, construct_id: str, env_var: str) -> aws_iam.IRole:
    """An existing role, named by ARN in .env and imported read-only.

    Lives next to the load_dotenv above, which is what puts the ARN in the environment.
    """
    return aws_iam.Role.from_role_arn(scope, construct_id, os.environ[env_var], mutable=False)
