# Dev container image for the api Lambda.
#
# Production fidelity: same Python and same OS-level libs as the
# Lambda image, so anything that runs here will run in production.
# Adds dev-only tools: pytest deps, git, sudo, bash improvements,
# and a non-root user so files created in the container don't end
# up owned by root on the host.
FROM public.ecr.aws/lambda/python:3.13

ENV PATH="/usr/sbin:/sbin:${PATH}"

# AWS Lambda base is Amazon Linux with microdnf. Install dev tools
# we'd never want in production.
RUN microdnf install -y git sudo shadow-utils tar gzip which \
    && microdnf clean all

# Create a non-root user. UID 1000 typically matches the first user
# on a Linux host. Macs don't care about UID, but consistency helps.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Production dependencies (matches what ships to Lambda).
COPY functions/api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt

# Test-only deps install in postCreateCommand so the layer doesn't
# rebuild every time we tweak them. Keeps the image incremental.

# Lambda's base image expects an entrypoint that handles invocations.
# In a dev container we just want the container to stay running so
# VS Code can attach.
ENTRYPOINT []
CMD ["sleep", "infinity"]

# Switch to non-root for VS Code's user-facing shell.
USER $USERNAME
WORKDIR /workspace