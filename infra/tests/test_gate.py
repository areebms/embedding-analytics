"""The deploy gate's preflight, checked against the real functions/ tree.

`functions/<service>/tests/` is what arms the gate and the Dockerfile's `test` stage is
what runs it. They live in different files and can be added or removed independently,
and one of those directions never fails on its own -- it just deploys having run nothing.
"""

import pytest

import app
import config
from deploy import GateError, validate_dockerfile


@pytest.mark.parametrize("service", app.DEPLOYED)
def test_every_deployed_service_can_be_gated(service):
    """Adding a service to DEPLOYED without a suite or a `test` stage fails here."""
    validate_dockerfile(service)


def make_service(root, name: str, *, dockerfile: str | None, tests: bool) -> None:
    """One functions/<name>/ as validate_dockerfile expects to find it."""
    service_dir = root / "functions" / name
    service_dir.mkdir(parents=True)
    if dockerfile is not None:
        (service_dir / "Dockerfile").write_text(dockerfile)
    if tests:
        (service_dir / "tests").mkdir()


LAMBDA_ONLY = "FROM public.ecr.aws/lambda/python:3.13 AS base\nFROM base AS lambda\n"
WITH_TEST = LAMBDA_ONLY + "FROM lambda AS test\n"


def test_a_service_with_no_dockerfile_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    make_service(tmp_path, "nope", dockerfile=None, tests=True)

    with pytest.raises(GateError, match=r"^nope: no functions/nope/Dockerfile$"):
        validate_dockerfile("nope")


def test_an_unknown_service_is_reported_rather_than_skipped(tmp_path, monkeypatch):
    """A name that is not a directory at all reports the path, it does not pass."""
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)

    with pytest.raises(GateError, match=r"^ghost: no functions/ghost/Dockerfile$"):
        validate_dockerfile("ghost")


def test_a_service_with_no_tests_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    make_service(tmp_path, "bare", dockerfile=WITH_TEST, tests=False)

    with pytest.raises(GateError, match=r"bare: no tests/ -- nothing gates it"):
        validate_dockerfile("bare")


def test_a_service_missing_its_test_stage_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    make_service(tmp_path, "untested", dockerfile=LAMBDA_ONLY, tests=True)

    with pytest.raises(GateError, match=r"untested: has tests/ .* no `test` stage"):
        validate_dockerfile("untested")


@pytest.mark.parametrize(
    "last_line",
    [
        "FROM base AS test\n",
        "FROM lambda as test\n",
        "FROM lambda AS test",  # no trailing newline
        "FROM lambda AS test  \n",
    ],
)
def test_a_test_stage_is_found_however_it_is_spelled(tmp_path, monkeypatch, last_line):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    make_service(tmp_path, "svc", dockerfile=LAMBDA_ONLY + last_line, tests=True)

    validate_dockerfile("svc")


@pytest.mark.parametrize(
    "last_line",
    [
        "FROM lambda AS testing\n",  # a longer name is not the test stage
        "# FROM lambda AS test\n",  # commented out
        "FROM lambda AS pytest\n",
    ],
)
def test_a_stage_that_only_looks_like_test_does_not_arm_the_gate(
    tmp_path, monkeypatch, last_line
):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    make_service(tmp_path, "svc", dockerfile=LAMBDA_ONLY + last_line, tests=True)

    with pytest.raises(GateError, match=r"no `test` stage"):
        validate_dockerfile("svc")
