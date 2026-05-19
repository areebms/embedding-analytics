#!/usr/bin/env bash
set -e

source .env

export AWS_PAGER=""

: "${AWS_ACCOUNT_ID:?}"
: "${LAMBDA_PREFIX:?}"
: "${STEP_FUNCTION_ROLE_ARN:?}"

STEP_FUNCTION_TEMPLATE="${STEP_FUNCTION_TEMPLATE:-infra/step-function.template.json}"
STATE_MACHINE_NAME="${LAMBDA_PREFIX}-pipeline"
STATE_MACHINE_ARN="arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:stateMachine:${STATE_MACHINE_NAME}"

DEFINITION="$(envsubst '${AWS_REGION} ${AWS_ACCOUNT_ID} ${LAMBDA_PREFIX}' < "$STEP_FUNCTION_TEMPLATE")"

echo "=== Step Function: $STATE_MACHINE_NAME ==="

if aws stepfunctions describe-state-machine \
    --state-machine-arn "$STATE_MACHINE_ARN" \
    --region "$AWS_REGION" &>/dev/null; then
    echo "Updating..."
    aws stepfunctions update-state-machine \
        --state-machine-arn "$STATE_MACHINE_ARN" \
        --definition "$DEFINITION" \
        --region "$AWS_REGION"
else
    echo "Creating..."
    aws stepfunctions create-state-machine \
        --name "$STATE_MACHINE_NAME" \
        --definition "$DEFINITION" \
        --role-arn "$STEP_FUNCTION_ROLE_ARN" \
        --region "$AWS_REGION"
fi

echo "Done: $STATE_MACHINE_NAME"
