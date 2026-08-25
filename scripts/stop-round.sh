#!/usr/bin/env bash
# Stop every active LHTB round worker, wrapper, and running Docker container.

set -euo pipefail

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }

# Harbor is the real worker. Killing only driftlock-lhtb leaves paid work alive.
log "Stopping Harbor workers"
pkill -f 'harbor .*run -c' || true

log "Stopping driftlock-lhtb wrappers"
pkill -f 'driftlock-lhtb run' || true

log "Stopping running Docker containers"
if command -v docker >/dev/null 2>&1; then
  if running_containers="$(docker ps -q)"; then
    if [ -n "$running_containers" ]; then
      containers=()
      while IFS= read -r container; do
        [ -n "$container" ] && containers+=("$container")
      done <<<"$running_containers"
      docker stop "${containers[@]}" || true
    fi
  else
    warn "could not query the Docker daemon"
  fi
else
  warn "docker executable not found"
fi

survivors=0
if harbor_survivors="$(pgrep -af 'harbor .*run -c')"; then
  warn "Harbor workers survived shutdown:
$harbor_survivors"
  survivors=1
fi
if wrapper_survivors="$(pgrep -af 'driftlock-lhtb run')"; then
  warn "driftlock-lhtb wrappers survived shutdown:
$wrapper_survivors"
  survivors=1
fi
if command -v docker >/dev/null 2>&1; then
  if container_survivors="$(docker ps -q)"; then
    if [ -n "$container_survivors" ]; then
      warn "Docker containers survived shutdown:
$container_survivors"
      survivors=1
    fi
  else
    warn "could not re-check the Docker daemon"
    survivors=1
  fi
fi

if [ "$survivors" -eq 0 ]; then
  log "No round workers or running containers survived"
fi
exit "$survivors"
