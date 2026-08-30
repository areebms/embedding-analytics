#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "${1:-}" in
    scrape)      SERVICES=(scrape);           SELECT=(--exclusively "*-scrape");      shift ;;
    standardize) SERVICES=(standardize-html); SELECT=(--exclusively "*-standardize"); shift ;;
    relay)       SERVICES=();                 SELECT=(--exclusively "*-relay");       shift ;;
    *)           SERVICES=(scrape standardize-html); SELECT=(--all) ;;
esac

export BUILDX_NO_DEFAULT_ATTESTATIONS=1

for SERVICE in "${SERVICES[@]}"; do
    echo
    echo "=== Testing $SERVICE ==="
    docker buildx build \
        --platform linux/amd64 \
        --target test \
        --load \
        -t "$SERVICE:test" \
        -f "functions/$SERVICE/Dockerfile" .

    docker run --rm "$SERVICE:test"
done

echo
echo "=== Deploying ==="
cd "$REPO_ROOT/infra"

if [ "${1:-}" = "--" ]; then
    shift
else
    set -- deploy "$@"
fi

exec cdk "$@" "${SELECT[@]}"
