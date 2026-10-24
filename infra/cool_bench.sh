#!/usr/bin/env bash
# Launch a Graviton4 instance from the COOL (Cloud-Optimized OpenCV) Marketplace AMI, benchmark COOL against the
# opencv-contrib wheel on it (infra/cool_remote.sh, run over SSM), bring the results back as reports/cool.md and
# reports/cool.json, and terminate the instance.
#
#   COOL_AMI_ID=ami-0123456789abcdef0 infra/cool_bench.sh
#
# Subscribe to "Cloud Optimized OpenCV For AWS Graviton4" in AWS Marketplace first; the AMI ID is shown under
# "Continue to Configuration" for your region. The COOL build is OpenCV 5, which Fire Watch needs.
#   AWS_REGION      region of that AMI (default us-west-2)
#   INSTANCE_TYPE   default c8g.large (2 vCPUs, like the t4g.large that serves the dashboard)
#   ROUNDS          alternating benchmark rounds per build (default 3)
#   REPO_URL        repository cloned on the instance (default the public GitHub repo)
#   KEEP_INSTANCE=1 leave the instance running afterwards
set -euo pipefail
cd "$(dirname "$0")/.."

: "${COOL_AMI_ID:?set COOL_AMI_ID to the COOL Graviton4 AMI ID from AWS Marketplace}"
REGION="${AWS_REGION:-us-west-2}"
TYPE="${INSTANCE_TYPE:-c8g.large}"
ROUNDS="${ROUNDS:-3}"
REPO_URL="${REPO_URL:-https://github.com/ADCobra/Jharia-Coal-fields.git}"
NAME=firewatch-cool-bench
aws=(aws --region "$REGION")

log() { echo "[cool-bench] $*" >&2; }

# An instance role that lets SSM run the benchmark, so no SSH key or open port is needed.
if ! aws iam get-instance-profile --instance-profile-name "$NAME" >/dev/null 2>&1; then
  log "creating IAM role and instance profile $NAME"
  aws iam create-role --role-name "$NAME" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam create-instance-profile --instance-profile-name "$NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$NAME" --role-name "$NAME"
  sleep 15
fi

VPC="$("${aws[@]}" ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
[ "$VPC" != None ] || { log "no default VPC in $REGION; create one with: aws ec2 create-default-vpc --region $REGION"; exit 2; }
SG="$("${aws[@]}" ec2 describe-security-groups --filters Name=vpc-id,Values="$VPC" Name=group-name,Values="$NAME" --query 'SecurityGroups[0].GroupId' --output text)"
if [ "$SG" = None ]; then
  SG="$("${aws[@]}" ec2 create-security-group --vpc-id "$VPC" --group-name "$NAME" --description "Fire Watch COOL benchmark: outbound only" --query GroupId --output text)"
fi
ROOT_DEV="$("${aws[@]}" ec2 describe-images --image-ids "$COOL_AMI_ID" --query 'Images[0].RootDeviceName' --output text)"

log "launching $TYPE from $COOL_AMI_ID in $REGION"
INSTANCE="$("${aws[@]}" ec2 run-instances --image-id "$COOL_AMI_ID" --instance-type "$TYPE" \
  --iam-instance-profile Name="$NAME" --security-group-ids "$SG" \
  --block-device-mappings "DeviceName=$ROOT_DEV,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}" \
  --metadata-options HttpTokens=required \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query 'Instances[0].InstanceId' --output text)"
cleanup() {
  if [ "${KEEP_INSTANCE:-0}" = 1 ]; then
    log "leaving $INSTANCE running (KEEP_INSTANCE=1); terminate it with: aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE"
  else
    log "terminating $INSTANCE"
    "${aws[@]}" ec2 terminate-instances --instance-ids "$INSTANCE" >/dev/null || true
  fi
}
trap cleanup EXIT
"${aws[@]}" ec2 wait instance-running --instance-ids "$INSTANCE"

log "waiting for the SSM agent on $INSTANCE"
for _ in $(seq 1 60); do
  STATUS="$("${aws[@]}" ssm describe-instance-information --filters Key=InstanceIds,Values="$INSTANCE" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"
  [ "$STATUS" = Online ] && break
  sleep 10
done
if [ "${STATUS:-}" != Online ]; then
  log "SSM never came online on $INSTANCE. Run infra/cool_remote.sh on it over SSH instead (see the README), or rerun with KEEP_INSTANCE=1 to look."
  exit 3
fi

SCRIPT_B64="$(base64 < infra/cool_remote.sh | tr -d '\n')"
COMMAND="echo $SCRIPT_B64 | base64 -d > /tmp/cool_remote.sh && PRINT_RESULTS=1 ROUNDS=$ROUNDS REPO_URL=$REPO_URL bash /tmp/cool_remote.sh"
log "running the benchmark (about 20 to 40 minutes on a c8g.large)"
CMD_ID="$("${aws[@]}" ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
  --comment "Fire Watch COOL benchmark" --timeout-seconds 600 \
  --parameters "$(printf '{"commands":[%s],"executionTimeout":["7200"]}' "$(printf '%s' "$COMMAND" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
  --query Command.CommandId --output text)"
while :; do
  sleep 30
  STATE="$("${aws[@]}" ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" --query Status --output text 2>/dev/null || echo Pending)"
  case "$STATE" in Pending|InProgress|Delayed) log "still running ($STATE)";; *) break;; esac
done
OUT="$("${aws[@]}" ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" --query StandardOutputContent --output text)"
if [ "$STATE" != Success ]; then
  log "the benchmark ended with status $STATE. Last error output:"
  "${aws[@]}" ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" --query StandardErrorContent --output text | tail -40 >&2
  exit 4
fi

DEST=reports/cool-runs
rm -rf "$DEST" && mkdir -p "$DEST"
printf '%s\n' "$OUT" | sed -n '/===RESULTS-BEGIN===/,/===RESULTS-END===/p' | sed '1d;$d' | tr -d '\n' | base64 -d | tar -xzf - -C "$DEST"
ARGS=(--cool "$DEST"/cool-*.json --pip "$DEST"/pip-*.json --out reports)
if [ -f "$DEST/pipeline-cool/bench.json" ] && [ -f "$DEST/pipeline-pip/bench.json" ]; then
  ARGS+=(--pipeline-cool "$DEST/pipeline-cool/bench.json" --pipeline-pip "$DEST/pipeline-pip/bench.json")
fi
python3 scripts/cool_compare.py "${ARGS[@]}" >/dev/null
log "wrote reports/cool.md and reports/cool.json (raw runs in $DEST)"
