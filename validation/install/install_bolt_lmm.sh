#!/usr/bin/env bash
# Install BOLT-LMM into validation/tools/bolt_lmm/.
# Pre-built static tarball from the Alkes group.
set -euo pipefail

BOLT_VERSION="2.5"
BOLT_TARBALL="BOLT-LMM_v${BOLT_VERSION}.tar.gz"
BOLT_URL="https://alkesgroup.broadinstitute.org/BOLT-LMM/downloads/${BOLT_TARBALL}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HERE}/tools/bolt_lmm"
MARKER="${PREFIX}/.installed"

if [[ -f "${MARKER}" ]]; then
    echo "bolt_lmm already installed at ${PREFIX}"
    exit 0
fi

mkdir -p "${PREFIX}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Downloading BOLT-LMM v${BOLT_VERSION}..."
curl -fL --retry 3 -o "${TMP}/bolt.tgz" "${BOLT_URL}"
# Drop the top-level versioned directory so the binary path is stable.
tar --strip-components=1 -xzf "${TMP}/bolt.tgz" -C "${PREFIX}"
if [[ ! -x "${PREFIX}/bolt" ]]; then
    echo "bolt binary not found at ${PREFIX}/bolt" >&2
    exit 1
fi
mkdir -p "${PREFIX}/bin"
ln -sf ../bolt "${PREFIX}/bin/bolt"
"${PREFIX}/bin/bolt" --help 2>&1 | head -1 || true
echo "${BOLT_VERSION}" > "${MARKER}"
echo "bolt_lmm installed at ${PREFIX}"
