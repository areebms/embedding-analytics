#!/usr/bin/env python3.13

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

# Snapshotted before `import config`, which calls load_dotenv: .env holds a restricted
# deploy principal that cannot call DescribeStacks, and cdk makes its CloudFormation
# calls as whoever invoked this script. The app subprocess cdk spawns loads .env for
# itself. Moving this line below the imports breaks every deploy -- see
# docs/operations.md, "Configuration and deployment".
SHELL_ENV = os.environ.copy()

# buildx otherwise attaches provenance/SBOM attestations, which turn `--load` into a
# manifest list rather than the single image `docker run` expects. Carried over from the
# deploy.sh this replaces.
SHELL_ENV["BUILDX_NO_DEFAULT_ATTESTATIONS"] = "1"

import app  # noqa: E402
import config  # noqa: E402

USAGE = "usage: deploy.py [-- CDK ARGS]"

TEST_STAGE = re.compile(r"^\s*FROM\s.*\sAS\s+test\s*$", re.IGNORECASE | re.MULTILINE)


class GateError(Exception):
    """A service the gate cannot run, so the deploy must not proceed.

    Its own type so the caller catches a verdict and nothing else: an OSError reading the
    Dockerfile is a fault, not a misconfigured service, and should not be reported as one.
    """


def run_cmd(cmd: list[str], *, cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, env=SHELL_ENV).returncode


def validate_dockerfile(service: str) -> None:
    """Both halves of one service's test gate: the suite, and the stage that runs it.

    They live in different files and can be added or removed independently, and one of
    those directions never fails -- it just deploys having run nothing.
    """
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


def run_test_container(service: str) -> int:
    """One service's suite, inside the `test` stage of its own Dockerfile.

    Built from the repo root, not from the staged asset/ directory cdk.out holds: that
    context has tests/ excluded (see get_test_files in resources.py) so a test-only edit
    does not republish an identical image. The gate has to read the real tree.

    No --env-file .env, deliberately. The suites pin their own AWS region, credentials,
    bucket and table in conftest; handing them the deploy environment instead points them
    at production names.
    """
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
        return build
    return run_cmd(["docker", "run", "--rm", tag], cwd=config.REPO_ROOT)


def run_test_gate(services: list[str]) -> int:
    """The CDK suite, then every suite the stack ships. 0 means the deploy may go ahead.

    Stops at the first failure: the point is to not build anything, so there is nothing
    to gain from running the rest. Every service is validated before the first container
    is built, so a service that cannot be gated is reported in seconds rather than after
    the ones ahead of it in the list have each built and run.
    """
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

        code = run_test_container(service)
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
