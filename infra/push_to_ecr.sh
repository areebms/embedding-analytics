#!/usr/bin/env bash
set -e

source .env

FILE="${FILE:-infra/services.yaml}"
TAG="${TAG:-0.1}"
REGION="${AWS_REGION}"
export AWS_PAGER="" # Disable pager.

# check required envs
: "${AWS_URI_PREFIX:?}"
: "${AWS_ECR_REPO:?}"
: "${LAMBDA_ROLE_ARN:?}"

# set defaults
DEFAULT_MEMORY="$(yq e '.default.memory' "$FILE")"
DEFAULT_TIMEOUT="$(yq e '.default.timeout' "$FILE")"

echo "Logging in"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$AWS_URI_PREFIX"

for SERVICE in "$*"; do
  FUNCTION="$(yq e ".services.$SERVICE.function_name" "$FILE")"
  FUNCTION_NAME="$LAMBDA_PREFIX-$FUNCTION"
  IMAGE="$(yq e ".services.$SERVICE.image" "$FILE")"
  MEMORY="$(yq e ".services.$SERVICE.memory // $DEFAULT_MEMORY" "$FILE")"
  TIMEOUT="$(yq e ".services.$SERVICE.timeout // $DEFAULT_TIMEOUT" "$FILE")"

  FULL_TAG="$AWS_URI_PREFIX/$AWS_ECR_REPO/$IMAGE:$TAG"
  ECR_REPO="$AWS_ECR_REPO/$IMAGE"
  DOCKERFILE="functions/$FUNCTION/Dockerfile"

  echo
  echo "=== $SERVICE (fn=$FUNCTION img=$IMAGE mem=$MEMORY timeout=$TIMEOUT) ==="

  # run tests before docker work, fail fast
  if [ -d "functions/$FUNCTION/tests" ]; then
    echo "Running tests for $FUNCTION"
    pytest "functions/$FUNCTION" -v
  fi

  docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --no-cache \
  --load \
  -t "$FULL_TAG" \
  -f "$DOCKERFILE" .

  # run tests
  docker run --rm -it --env-file .env --entrypoint python $FULL_TAG main.py --platform gutenberg --id 60411 --seed 10

  docker push "$FULL_TAG"

  DIGEST="$(aws ecr describe-images --region "$REGION" \
    --repository-name "$ECR_REPO" --image-ids "imageTag=$TAG" \
    --query 'imageDetails[0].imageDigest' --output text)"

  IMAGE_URI="$AWS_URI_PREFIX/$ECR_REPO@$DIGEST"
  
  # TODO: Only Update if changed.
  if aws lambda get-function --region "$REGION" --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
    aws lambda update-function-code --region "$REGION" --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI"
    aws lambda update-function-configuration --region "$REGION" --function-name "$FUNCTION_NAME" --memory-size "$MEMORY" --timeout "$TIMEOUT"
  else
    aws lambda create-function --region "$REGION" \
      --function-name "$FUNCTION_NAME" --package-type Image \
      --code "ImageUri=$IMAGE_URI" --role "$LAMBDA_ROLE_ARN" \
      --memory-size "$MEMORY" --timeout "$TIMEOUT"
  fi

  aws lambda wait function-active-v2 --region "$REGION" --function-name "$FUNCTION_NAME"
  echo "Done: $SERVICE"
done
