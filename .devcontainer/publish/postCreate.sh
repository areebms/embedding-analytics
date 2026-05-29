#!/usr/bin/env bash
# postCreate.sh - prepare /var/task to match the Lambda production layout.
#
# In production, the Lambda Dockerfile flattens functions/publish/src/*.py
# into ${LAMBDA_TASK_ROOT} (/var/task) and copies ./shared as a
# sibling package. The dev container preserves the repo layout under
# /workspace (the bind mount), so we mirror the Lambda layout via
# symlinks pointing into /workspace.
#
# Rerun this script after adding, renaming, or deleting source files.
# It is idempotent: it wipes the previous /var/task layout and
# reconstructs it.

set -e

SRC_DIR=/workspace/functions/publish/src
SHARED_DIR=/workspace/shared
TESTS_DIR=/workspace/functions/publish/tests
PYTEST_CFG=/workspace/functions/publish/pytest.ini
TASK_DIR=/var/task
REQS=/workspace/functions/publish/requirements-dev.txt

echo "Rebuilding $TASK_DIR..."

sudo mkdir -p "$TASK_DIR"

# Wipe previous Python file links and known package links so
# this script is safe to re-run when files are removed or renamed.
sudo rm -f "$TASK_DIR"/*.py
sudo rm -f "$TASK_DIR/shared" "$TASK_DIR/tests" "$TASK_DIR/pytest.ini"

# Link each .py file in functions/publish/src/ as a sibling of /var/task.
# `ln -sfn` is force + no-dereference, so it overwrites cleanly if a
# symlink with the same name already exists.
shopt -s nullglob
for f in "$SRC_DIR"/*.py; do
    name="$(basename "$f")"
    sudo ln -sfn "$f" "$TASK_DIR/$name"
done
shopt -u nullglob

# Link shared as a directory, so `from shared.aws import ...` works.
sudo ln -sfn "$SHARED_DIR" "$TASK_DIR/shared"

# Link tests and pytest.ini so `cd /var/task && pytest` works
# identically to how the deploy script's test image runs.
sudo ln -sfn "$TESTS_DIR" "$TASK_DIR/tests"
sudo ln -sfn "$PYTEST_CFG" "$TASK_DIR/pytest.ini"

echo "Installing test dependencies..."
pip install --no-cache-dir -r "$REQS"

echo "Done. Source layout:"
ls -la "$TASK_DIR"
