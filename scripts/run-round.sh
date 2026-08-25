#!/usr/bin/env bash
# Launch one four-arm LHTB experiment round and summarize its Harbor jobs.
#
#   export OPENROUTER_API_KEY=...  # in the calling shell
#   scripts/run-round.sh r5
#
# Environment:
#   LHTB_DIR  pinned LHTB checkout (default: /srv/LHTB)

set -euo pipefail

LHTB_DIR="${LHTB_DIR:-/srv/LHTB}"
BIN="$LHTB_DIR/harbor/.venv/bin/driftlock-lhtb"
JOBS_DIR="$LHTB_DIR/jobs"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ARMS=(stock retry driftlock-heuristic driftlock)
TASKS=(
  commit0-multilib-tdd
  riscv-core-debug
  spice-ephemeris-regression
  epidemic-inverse-control-audit
  alp-paper-reproduction
  unknown-config-semantics
  sudoku-recovery
  2048
)

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$#" -eq 1 ] || die "usage: $0 ROUND_PREFIX (for example: r5)"
PREFIX="$1"
[[ "$PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
  || die "round prefix contains unsafe characters: $PREFIX"

# The credential must be inherited from the caller, never loaded from disk.
[ -n "${OPENROUTER_API_KEY:-}" ] || die "OPENROUTER_API_KEY is not set in the calling shell.
Attach to the credentialed smoke shell with: tmux attach -t smoke"
key_length="${#OPENROUTER_API_KEY}"
if [ "$key_length" -gt 3 ]; then
  key_suffix="${OPENROUTER_API_KEY: -3}"
else
  # Do not reveal a short value in full, even if it is almost certainly invalid.
  key_suffix="***"
fi
log "OPENROUTER_API_KEY present (length=$key_length, last3=$key_suffix)"

# This must succeed before provider probes or workers are allowed to spend tokens.
log "Running paid-run preflight"
"$BIN" preflight --lhtb-dir "$LHTB_DIR" \
  || die "preflight failed; no round jobs were launched"

# Harbor silently skips tasks when a job name is reused, so this is deliberately
# checked only after preflight and before creating any round output.
for arm in "${ARMS[@]}"; do
  job_dir="$JOBS_DIR/$PREFIX-$arm"
  [ ! -e "$job_dir" ] \
    || die "refusing to reuse existing job directory: $job_dir"
done

LOG_DIR="$JOBS_DIR/$PREFIX-logs"
mkdir -p "$LOG_DIR"

PIDS=()
on_signal() {
  trap - INT TERM
  printf '\n' >&2
  log "Signal received; stopping the round"
  "$SCRIPT_DIR/stop-round.sh" || true
  exit 130
}
trap on_signal INT TERM

launch_arm() {
  local arm="$1"
  local job_name="$PREFIX-$arm"
  local extra_args=()

  if [ "$arm" = stock ]; then
    extra_args+=(--ack-unbounded-stock-tokens)
  else
    extra_args+=(--max-total-tokens 20000000)
  fi

  "$BIN" run \
    --lhtb-dir "$LHTB_DIR" \
    --jobs-dir "$JOBS_DIR" \
    --job-name "$job_name" \
    --arm "$arm" \
    --tasks "${TASKS[@]}" \
    --concurrency 1 \
    "${extra_args[@]}" \
    >"$LOG_DIR/$arm.log" 2>&1 &
  PIDS+=("$!")
  log "Launched $arm as PID $! (log: $LOG_DIR/$arm.log)"
}

log "Launching round $PREFIX"
for arm in "${ARMS[@]}"; do
  launch_arm "$arm"
done

round_status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    log "${ARMS[$index]} finished"
  else
    arm_status="$?"
    round_status=1
    log "warning: ${ARMS[$index]} exited with status $arm_status"
  fi
done
trap - INT TERM

summarize_arm() {
  local arm="$1"
  local job_dir="$JOBS_DIR/$PREFIX-$arm"
  local summary="$job_dir/result.json"
  local completed="unavailable"
  local errored="unavailable"
  local cost="unavailable"
  local exception_counts
  local result

  if [ -f "$summary" ] && jq -e '.stats | type == "object"' "$summary" >/dev/null 2>&1; then
    completed="$(jq -r '.stats.n_completed_trials // "unavailable"' "$summary")"
    errored="$(jq -r '.stats.n_errored_trials // "unavailable"' "$summary")"
    cost="$(jq -r '.stats.cost_usd // "unavailable"' "$summary")"
  fi

  exception_counts="$({
    for result in "$job_dir"/*/result.json; do
      [ -e "$result" ] || continue
      jq -r \
        'select(.exception_info != null) | (.exception_info.exception_type // "unknown")' \
        "$result" 2>/dev/null || printf '%s\n' unreadable-result
    done
  } | sort | uniq -c | awk '
    BEGIN { separator = "" }
    { printf "%s%s=%s", separator, $2, $1; separator = ", " }
    END { if (separator == "") printf "none" }
  ')"

  printf '  %-20s completed=%s errored=%s cost_usd=%s exceptions=%s\n' \
    "$arm" "$completed" "$errored" "$cost" "$exception_counts"
}

log "Round $PREFIX summary"
for arm in "${ARMS[@]}"; do
  summarize_arm "$arm"
done
printf '\nAgentTimeoutError is the kept 90-minute cap.\n'

exit "$round_status"
