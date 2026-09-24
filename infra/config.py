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

service_configs = yaml.safe_load(SERVICES_FILE.read_text())["services"]


def get_service_config(service, config):
    service_data = service_configs[service] or {}
    if config in service_data:
        if config == "env":
            return service_configs["default"]["env"] + service_data["env"]
        return service_data[config]
    return service_configs["default"].get(config)

def get_services():
    services = list(service_configs)
    services.remove("default")
    return services