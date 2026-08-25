#!/usr/bin/env bash
# Analyze a completed four-arm LHTB round without making provider calls.
#
#   scripts/analyze-round.sh r5
#
# Environment:
#   LHTB_DIR  pinned LHTB checkout (default: /srv/LHTB)

set -euo pipefail

LHTB_DIR="${LHTB_DIR:-/srv/LHTB}"
BIN="$LHTB_DIR/harbor/.venv/bin/driftlock-lhtb"
JOBS_DIR="$LHTB_DIR/jobs"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$#" -eq 1 ] || die "usage: $0 ROUND_PREFIX (for example: r5)"
PREFIX="$1"
[[ "$PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
  || die "round prefix contains unsafe characters: $PREFIX"

OUTPUT="$JOBS_DIR/$PREFIX-report.json"
log "Analyzing round $PREFIX (this spends no provider tokens)"
"$BIN" analyze \
  --lhtb-dir "$LHTB_DIR" \
  --arm-dir "stock=$JOBS_DIR/$PREFIX-stock" \
  --arm-dir "retry=$JOBS_DIR/$PREFIX-retry" \
  --arm-dir "driftlock-heuristic=$JOBS_DIR/$PREFIX-driftlock-heuristic" \
  --arm-dir "driftlock=$JOBS_DIR/$PREFIX-driftlock" \
  --exclude-dead-tasks \
  --output "$OUTPUT"
log "Wrote $OUTPUT"
