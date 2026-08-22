#!/usr/bin/env bash
# Provision a fresh Ubuntu 24.04 amd64 host for driftlock LHTB experiments.
#
# This is the source of truth for the experiment host. A snapshot is a cache of
# whatever this script produced; if the two ever disagree, this file wins.
#
#   curl -fsSL <raw url>/scripts/setup-server.sh | bash
#   # or, after cloning driftlock:
#   sudo bash scripts/setup-server.sh
#
# Environment:
#   DRIFTLOCK_REF   git ref of driftlock to install      (default: main)
#   DRIFTLOCK_REPO  driftlock clone URL                  (default: public repo)
#   ROOT_DIR        where LHTB and driftlock are placed   (default: /srv)
#
# It never accepts, reads, or writes a provider credential. Inject
# OPENROUTER_API_KEY into the shell that runs experiments, not into this script
# and not into any file on disk.

set -euo pipefail

DRIFTLOCK_REPO="${DRIFTLOCK_REPO:-https://github.com/hyj28/driftlock.git}"
DRIFTLOCK_REF="${DRIFTLOCK_REF:-main}"
ROOT_DIR="${ROOT_DIR:-/srv}"

# Pinned by integrations/lhtb/README.md. Changing any of these invalidates every
# measurement taken against the previous values.
LHTB_REPO="https://github.com/zli12321/LHTB.git"
LHTB_COMMIT="0d9918f6b66eda0752f8c7d17c9a73a18ee32f98"
EXPECTED_LITELLM="1.83.14"
EXPECTED_PATCH_VERSION=10

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preconditions

log "Checking host architecture"
arch="$(uname -m)"
[ "$arch" = "x86_64" ] || die "LHTB images are amd64-only; this host is $arch.
Running them under emulation is slow enough to change the experiment's wall clock,
and some images do not run at all. Provision a native amd64 host."

[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash $0)"

. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || log "warning: tested on Ubuntu 24.04, found ${PRETTY_NAME:-unknown}"

# ------------------------------------------------------------------- base tools

log "Installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  ca-certificates curl git gnupg tmux jq rsync unzip build-essential python3

# ----------------------------------------------------------------------- docker

if command -v docker >/dev/null 2>&1; then
  log "Docker already present: $(docker --version)"
else
  log "Installing Docker from Docker's own apt repository"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  cat >/etc/apt/sources.list.d/docker.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable
EOF
  apt-get update -qq
  apt-get install -y -qq \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

systemctl enable --now docker
docker run --rm hello-world >/dev/null 2>&1 \
  || die "Docker is installed but cannot run a container"
log "Docker OK: $(docker --version)"

# -------------------------------------------------------------------------- uv

if command -v uv >/dev/null 2>&1; then
  log "uv already present: $(uv --version)"
else
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
command -v uv >/dev/null 2>&1 || die "uv did not land on PATH"

# ------------------------------------------------------------------- driftlock

mkdir -p "$ROOT_DIR"

DRIFTLOCK_DIR="$ROOT_DIR/driftlock"
if [ -d "$DRIFTLOCK_DIR/.git" ]; then
  # The checkout may have been placed here by other means (rsync from a laptop,
  # a restored snapshot) on a host with no credentials for the remote. A fetch
  # that cannot reach the remote is not a reason to abandon provisioning.
  log "Updating driftlock checkout"
  git -C "$DRIFTLOCK_DIR" fetch --all --tags --quiet \
    || log "warning: could not reach the remote; using the checkout as it stands"
else
  log "Cloning driftlock"
  git clone --quiet "$DRIFTLOCK_REPO" "$DRIFTLOCK_DIR"
fi
git -C "$DRIFTLOCK_DIR" checkout --quiet "$DRIFTLOCK_REF"
log "driftlock at $(git -C "$DRIFTLOCK_DIR" rev-parse --short HEAD) ($DRIFTLOCK_REF)"

# ------------------------------------------------------------------------ LHTB

LHTB_DIR="$ROOT_DIR/LHTB"
if [ -d "$LHTB_DIR/.git" ]; then
  log "LHTB checkout already exists"
else
  log "Cloning LHTB"
  git clone --quiet "$LHTB_REPO" "$LHTB_DIR"
fi

cd "$LHTB_DIR"
# A dirty tree would silently change the benchmark. Preflight rejects it later,
# but failing here costs nothing instead of failing after the images are pulled.
if ! git diff --quiet || ! git diff --cached --quiet; then
  die "LHTB checkout has local modifications; preflight requires a byte-exact tree"
fi
git checkout --quiet "$LHTB_COMMIT"
log "LHTB pinned at $LHTB_COMMIT"

log "Syncing Harbor's frozen environment"
uv sync --project harbor --frozen --no-dev

log "Installing driftlock into Harbor's environment"
uv pip install --quiet --python harbor/.venv/bin/python "$DRIFTLOCK_DIR"

# ----------------------------------------------------------------- companion patch

if git -C "$LHTB_DIR" diff --quiet -- harbor; then
  log "Applying the driftlock companion patch"
  PATCH="$(harbor/.venv/bin/python -c \
    'from driftlock import lhtb_harbor_patch_path; print(lhtb_harbor_patch_path())')"
  git apply "$PATCH"
else
  log "Harbor tree already modified; assuming the companion patch is applied"
fi

# ---------------------------------------------------------------------- verify

log "Verifying the pinned integration"
harbor/.venv/bin/python - <<PY
from importlib.metadata import version

from harbor._driftlock_pin import (
    DRIFTLOCK_HARBOR_PATCH_VERSION,
    LHTB_REPOSITORY_REVISION,
)

assert LHTB_REPOSITORY_REVISION == "${LHTB_COMMIT}", LHTB_REPOSITORY_REVISION
assert DRIFTLOCK_HARBOR_PATCH_VERSION == ${EXPECTED_PATCH_VERSION}, DRIFTLOCK_HARBOR_PATCH_VERSION
assert version("litellm") == "${EXPECTED_LITELLM}", version("litellm")
print("pinned LHTB Harbor integration ready")
PY

log "Running the no-network integration smoke tests"
uv pip install --quiet --python harbor/.venv/bin/python pytest pytest-asyncio
harbor/.venv/bin/python -m pytest -q \
  "$DRIFTLOCK_DIR/integrations/lhtb/test_runtime_smoke.py" \
  "$DRIFTLOCK_DIR/integrations/lhtb/test_harbor_agent.py"

# ------------------------------------------------------------------------- done

cat <<EOF

$(log "Host is ready")

  LHTB          $LHTB_DIR   (pinned $LHTB_COMMIT)
  driftlock     $DRIFTLOCK_DIR
  Harbor python $LHTB_DIR/harbor/.venv/bin/python

Nothing above spent a provider token. Before the first paid call:

  export OPENROUTER_API_KEY=...        # from your password manager, not a file
  $LHTB_DIR/harbor/.venv/bin/driftlock-lhtb preflight --lhtb-dir $LHTB_DIR

Then take a snapshot, so a rebuild does not repeat this install.
EOF
