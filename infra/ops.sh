#!/usr/bin/env bash
# Day-to-day commands for the stack that infra/deploy.sh created. All of them go through SSM, so no SSH is needed.
#   infra/ops.sh info          dashboard URL, password, and links to the CloudWatch trace and SNS topic
#   infra/ops.sh bench         run `firewatch eval bench` on the Graviton instance; writes reports/graviton/bench.{md,json}
#   infra/ops.sh alert [N]     process N more synthetic passes (default 3) so the agent acts and SNS sends alert emails
#   infra/ops.sh logs          print the last 30 minutes of agent traces from CloudWatch Logs
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-us-west-2}"
STACK="${STACK_NAME:-jharia-fire-watch}"
aws=(aws --region "$REGION")

output() { "${aws[@]}" cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

# Run a shell command on the instance and print its standard output.
remote() {
  local instance cmd_id state
  instance="$(output InstanceId)"
  cmd_id="$("${aws[@]}" ssm send-command --instance-ids "$instance" --document-name AWS-RunShellScript \
    --comment "$2" --parameters "$(python3 -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": ["3600"]}))' "$1")" \
    --query Command.CommandId --output text)"
  while :; do
    sleep 10
    state="$("${aws[@]}" ssm get-command-invocation --command-id "$cmd_id" --instance-id "$instance" --query Status --output text 2>/dev/null || echo Pending)"
    case "$state" in Pending|InProgress|Delayed) echo "  ... $state" >&2;; *) break;; esac
  done
  "${aws[@]}" ssm get-command-invocation --command-id "$cmd_id" --instance-id "$instance" --query StandardOutputContent --output text
  if [ "$state" != Success ]; then
    "${aws[@]}" ssm get-command-invocation --command-id "$cmd_id" --instance-id "$instance" --query StandardErrorContent --output text >&2
    return 1
  fi
}

case "${1:-info}" in
  info)
    LOG_GROUP="$(output TraceLogGroup)"
    echo "Dashboard:  $(output DashboardUrl)dashboard   (user demo)"
    echo "Password:   $(eval "$(output DashboardPasswordCommand)")"
    echo "Trace logs: https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#logsV2:log-groups/log-group/$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(urllib.parse.quote(sys.argv[1], safe=""), safe="").replace("%", "$"))' "$LOG_GROUP")"
    echo "Alerts:     https://$REGION.console.aws.amazon.com/sns/v3/home?region=$REGION#/topic/$(output AlertTopicArn)"
    ;;
  bench)
    mkdir -p reports/graviton
    # On instances with under 3 GB of memory, leave out DISK (set BENCH_SKIP= to force it in).
    SKIP="${BENCH_SKIP-auto}"
    if [ "$SKIP" = auto ]; then
      SKIP="$(remote "awk '/MemTotal/ {print (\$2 < 3000000) ? \"disk\" : \"\"}' /proc/meminfo" "Fire Watch memory check" | tr -d '[:space:]')"
    fi
    remote "docker exec -e FIREWATCH_BENCH_SKIP=$SKIP firewatch-api sh -c 'firewatch eval bench --out /tmp/bench >&2 && tar -C /tmp/bench -czf - bench.md bench.json | base64 -w0'" "Fire Watch eval bench" \
      | tr -d '\n' | base64 -d | tar -xzf - -C reports/graviton
    echo "Wrote reports/graviton/bench.md:"
    cat reports/graviton/bench.md
    ;;
  alert)
    N="${2:-3}"
    remote "for i in \$(seq 1 $N); do docker exec firewatch-api firewatch -c config/aws.yaml run --next; done" "Fire Watch: $N more passes"
    echo "Processed $N passes. Alert emails go to the confirmed subscription on $(output AlertTopicArn)."
    ;;
  logs)
    "${aws[@]}" logs tail "$(output TraceLogGroup)" --since 30m --format short
    ;;
  *)
    echo "usage: infra/ops.sh info|bench|alert [N]|logs" >&2
    exit 2
    ;;
esac
