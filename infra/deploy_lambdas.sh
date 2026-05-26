#!/usr/bin/env bash
set -e

source .env

export AWS_PAGER=""

: "${AWS_ACCOUNT_ID:?}"
: "${AWS_ECR_REPO:?}"
: "${LAMBDA_PREFIX:?}"
: "${LAMBDA_ROLE_ARN:?}"

TAG="${TAG:-0.1}"
SERVICES_FILE="${SERVICES_FILE:-infra/services.yaml}"
AWS_URI_PREFIX="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"


DEFAULT_MEMORY="$(yq e '.default.memory' "$SERVICES_FILE")"
DEFAULT_TIMEOUT="$(yq e '.default.timeout' "$SERVICES_FILE")"

deploy_service() {
    local SERVICE="$1"

    local FUNCTION="$(yq e ".services.$SERVICE.function_name" "$SERVICES_FILE")"
    local FUNCTION_NAME="$LAMBDA_PREFIX-$FUNCTION"
    local IMAGE="$(yq e ".services.$SERVICE.image" "$SERVICES_FILE")"
    local MEMORY="$(yq e ".services.$SERVICE.memory // $DEFAULT_MEMORY" "$SERVICES_FILE")"
    local TIMEOUT="$(yq e ".services.$SERVICE.timeout // $DEFAULT_TIMEOUT" "$SERVICES_FILE")"
    local SMOKE_CMD="$(yq e ".services.$SERVICE.smoke_cmd // \"\"" "$SERVICES_FILE")"

    local FULL_TAG="$AWS_URI_PREFIX/$AWS_ECR_REPO/$IMAGE:$TAG"
    local TEST_TAG="$IMAGE:test-$TAG"  # local-only, never pushed
    local ECR_REPO="$AWS_ECR_REPO/$IMAGE"
    local DOCKERFILE="functions/$FUNCTION/Dockerfile"

    echo
    echo "=== $SERVICE (fn=$FUNCTION_NAME img=$IMAGE mem=$MEMORY timeout=$TIMEOUT) ==="

    # Tests run inside the Docker image, using the 'test' stage of the
    # multi-stage Dockerfile. This gates the deploy: a non-zero exit
    # here aborts before any production image is built or pushed.
    if [ -d "functions/$FUNCTION/tests" ]; then
        echo "Building test image for $FUNCTION"
        docker buildx build \
            --platform linux/amd64 \
            --provenance=false \
            --sbom=false \
            --target test \
            --load \
            -t "$TEST_TAG" \
            -f "$DOCKERFILE" .

        echo "Running tests inside $TEST_TAG"
        # --env-file mounts the project .env so tests that need any
        # values (REDIS_PREFIX, table names) see them.
        docker run --rm --env-file .env "$TEST_TAG"
    fi

    # Production image. --no-cache because the test image just exercised
    # the layers and we want the production build to be clean.
    docker buildx build \
        --platform linux/amd64 \
        --provenance=false \
        --sbom=false \
        --no-cache \
        --target lambda \
        --load \
        -t "$FULL_TAG" \
        -f "$DOCKERFILE" .

    if [ -n "$SMOKE_CMD" ]; then
        echo "Running smoke test: $SMOKE_CMD"
        docker run --rm --env-file .env --entrypoint "" "$FULL_TAG" $SMOKE_CMD
    fi

    docker push "$FULL_TAG"

    local DIGEST
    DIGEST="$(aws ecr describe-images --region "$AWS_REGION" \
        --repository-name "$ECR_REPO" --image-ids "imageTag=$TAG" \
        --query 'imageDetails[0].imageDigest' --output text)"

    local IMAGE_URI="$AWS_URI_PREFIX/$ECR_REPO@$DIGEST"

    if aws lambda get-function --region "$AWS_REGION" --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
        local CURRENT_URI="$(aws lambda get-function --region "$AWS_REGION" \
            --function-name "$FUNCTION_NAME" \
            --query 'Code.ImageUri' --output text)"
        local CURRENT_MEMORY="$(aws lambda get-function-configuration --region "$AWS_REGION" \
            --function-name "$FUNCTION_NAME" \
            --query 'MemorySize' --output text)"
        local CURRENT_TIMEOUT="$(aws lambda get-function-configuration --region "$AWS_REGION" \
            --function-name "$FUNCTION_NAME" \
            --query 'Timeout' --output text)"

        if [[ "$CURRENT_URI" == "$IMAGE_URI" \
           && "$CURRENT_MEMORY" == "$MEMORY" \
           && "$CURRENT_TIMEOUT" == "$TIMEOUT" ]]; then
            echo "No changes, skipping update."
            return
        fi

        if [[ "$CURRENT_URI" != "$IMAGE_URI" ]]; then
            echo "Updating function code..."
            aws lambda update-function-code --region "$AWS_REGION" \
                --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI"
            aws lambda wait function-updated-v2 --region "$AWS_REGION" \
                --function-name "$FUNCTION_NAME"
        fi

        if [[ "$CURRENT_MEMORY" != "$MEMORY" || "$CURRENT_TIMEOUT" != "$TIMEOUT" ]]; then
            echo "Updating function configuration..."
            aws lambda update-function-configuration --region "$AWS_REGION" \
                --function-name "$FUNCTION_NAME" \
                --memory-size "$MEMORY" --timeout "$TIMEOUT"
        fi
    else
        echo "Creating function..."
        aws lambda create-function --region "$AWS_REGION" \
            --function-name "$FUNCTION_NAME" --package-type Image \
            --code "ImageUri=$IMAGE_URI" --role "$LAMBDA_ROLE_ARN" \
            --memory-size "$MEMORY" --timeout "$TIMEOUT"
    fi

    aws lambda wait function-active-v2 --region "$AWS_REGION" --function-name "$FUNCTION_NAME"
    echo "Done: $SERVICE"
}

if [[ $# -eq 0 ]]; then
    echo "No services provided. Usage: $0 service1 [service2 ...]"
    exit 1
fi

echo "Logging in to ECR"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_URI_PREFIX"

for SERVICE in "$@"; do
    deploy_service "$SERVICE"
done