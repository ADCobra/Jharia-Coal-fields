#!/usr/bin/env bash
# Build the arm64 image, push it to ECR, then create the stack or roll the
# running instance onto the new image.
#   ALERT_EMAIL=you@example.com DATA_SOURCE=synthetic infra/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-us-west-2}"
STACK="${STACK_NAME:-jharia-fire-watch}"
REPO="${ECR_REPO:-jharia-fire-watch}"
TAG="${IMAGE_TAG:-$(date -u +%Y%m%d%H%M%S)}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPO:$TAG"

aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO" --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker buildx build --platform linux/arm64 -t "$IMAGE" --push .

deploy_stack() {
  aws cloudformation deploy --region "$REGION" --stack-name "$STACK" \
    --template-file infra/cloudformation.yaml --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
    --parameter-overrides ImageUri="$1" AlertEmail="${ALERT_EMAIL:-}" DataSource="${DATA_SOURCE:-synthetic}" AllowedCidr="${ALLOWED_CIDR:-0.0.0.0/0}"
}

output() {
  aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

if aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" >/dev/null 2>&1; then
  CURRENT="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Parameters[?ParameterKey=='ImageUri'].ParameterValue" --output text)"
  deploy_stack "$CURRENT"
  INSTANCE="$(output InstanceId)"
  aws ssm send-command --region "$REGION" --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
    --comment "roll out $TAG" --parameters "commands=[\"/opt/firewatch/update.sh $IMAGE\"]" --query Command.CommandId --output text
  echo "Rolling $INSTANCE onto $IMAGE"
else
  deploy_stack "$IMAGE"
fi

aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
