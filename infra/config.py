"""resolves services.yaml + .env"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

INFRA_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFRA_DIR.parent
SERVICES_FILE = INFRA_DIR / "services.yaml"
ASL_DIR = INFRA_DIR / "step-functions"

load_dotenv(REPO_ROOT / ".env", override=False)

# Every physical name in the stack starts with this, and so does the stack itself.
PREFIX = os.environ["ENV_PREFIX"]

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
