#!/usr/bin/env python3.13

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

SHELL_ENV = os.environ.copy()

# single, flat image from buildx instead of manifest list.
SHELL_ENV["BUILDX_NO_DEFAULT_ATTESTATIONS"] = "1"

import app  # noqa: E402
import config  # noqa: E402

USAGE = "usage: deploy.py [-- CDK ARGS]"

TEST_STAGE = re.compile(r"^\s*FROM\s.*\sAS\s+test\s*$", re.IGNORECASE | re.MULTILINE)


class GateError(Exception):
    """A service the gate cannot run, so the deploy must not proceed."""


def run_cmd(cmd: list[str], *, cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, env=SHELL_ENV).returncode


def validate_dockerfile(service: str) -> None:

    service_dir = config.REPO_ROOT / "functions" / service
    dockerfile = service_dir / "Dockerfile"

    if not dockerfile.is_file():
        raise GateError(f"{service}: no {dockerfile.relative_to(config.REPO_ROOT)}")
    if not (service_dir / "tests").is_dir():
        raise GateError(
            f"{service}: no tests/ -- nothing gates it. Add the suite (and the `test` "
            f"stage, if the Dockerfile has none)"
        )
    if not TEST_STAGE.search(dockerfile.read_text()):
        raise GateError(
            f"{service}: has tests/ but its Dockerfile declares no `test` stage -- "
            f"copy the one in functions/scrape/Dockerfile with the paths swapped"
        )


def run_test_container(service: str) -> tuple[str, int]:
    tag = f"{service}:test"
    build = run_cmd(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--target",
            "test",
            "--load",
            "-t",
            tag,
            "-f",
            f"functions/{service}/Dockerfile",
            ".",
        ],
        cwd=config.REPO_ROOT,
    )
    if build != 0:
        return "build", build
    return "test", run_cmd(["docker", "run", "--rm", tag], cwd=config.REPO_ROOT)


def run_test_gate(services: list[str]) -> int:
    code = run_cmd([sys.executable, "-m", "pytest", "-q"], cwd=config.INFRA_DIR)
    if code != 0:
        print(
            "\ndeploy.py: pytest infra failed. nothing built or pushed.",
            file=sys.stderr,
        )
        return os.EX_SOFTWARE

    for service in services:
        try:
            validate_dockerfile(service)
        except GateError as exc:
            print(f"deploy.py: cannot gate this deploy: {exc}", file=sys.stderr)
            return os.EX_CONFIG

    for service in services:
        print(f"\n=== Testing {service} ===")

        phase, code = run_test_container(service)
        if phase == "build" and code != 0:
            print(
                f"\ndeploy.py: {service}: the test image did not build (exit {code}), "
                f"so the suite never ran -- not a test failure. "
                f"nothing built or pushed.",
                file=sys.stderr,
            )
            return os.EX_UNAVAILABLE
        if code != 0:
            print(
                f"\ndeploy.py: {service} failed. nothing built or pushed.",
                file=sys.stderr,
            )
            return os.EX_SOFTWARE

    return os.EX_OK


def main() -> int:

    argv = sys.argv[1:]
    if argv and argv[0] != "--":
        print(f"deploy.py: unexpected argument {argv[0]!r}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return os.EX_USAGE
    cdk_args = argv[1:]

    print("=== Building ===")
    app.build()
    print(f"stack ships: {', '.join(app.DEPLOYED)}")

    code = run_test_gate(app.DEPLOYED)
    if code != 0:
        return code

    command = ["cdk", "deploy", *cdk_args]
    print("\n=== Deploying ===")
    print(f"$ {' '.join(command)}")
    os.chdir(config.INFRA_DIR)
    try:
        os.execvpe("cdk", command, SHELL_ENV)
    except OSError as exc:
        # execvpe only returns by raising. Nothing has been built or pushed either way.
        print(f"deploy.py: cannot run cdk: {exc}", file=sys.stderr)
        return os.EX_UNAVAILABLE
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
